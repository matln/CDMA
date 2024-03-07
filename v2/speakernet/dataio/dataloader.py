"""PyTorch compatible DataLoaders

Essentially we extend PyTorch DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Example
-------
>>> import torch
>>> from .checkpoints import Checkpointer
>>> # An example "dataset" and its loader
>>> dataset = torch.randn(10, 1)
>>> dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> # Setup the checkpointer:
>>> tmpdir = getfixture('tmpdir')
>>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
>>> # Iterate:
>>> for i, data_point in enumerate(dataloader):
...     # Here you would process the data:
...     rainfall_amount_prediction = data_point * 4.
...     # Now, imagine the experiment gets killed on the fifth batch:
...     if i == 4:
...         break
...     # Luckily, you had just saved a checkpoint:
...     if i == 3:
...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
>>> # So when you restart the experiment:
>>> new_dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
>>> _ = new_checkpointer.recover_if_possible()
>>> # The dataloader fast-forwards to the position where we left off:
>>> assert next(iter(new_dataloader)) == dataset[4]

Copyright 2020 Aku Rouhe
          2022 Jianchen Li
"""
import torch
import random
import logging
import warnings
import functools
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter

from speakernet.training.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

logger = logging.getLogger(__name__)

# We essentially want to make the DataLoader iterators able to skip ahead
# after checkpoint recovery
# This should be handled by the DataLoader iterators' base class.
# To make the implementation here a little more maintainable
# we decide to patch some PyTorch functionality


def __new_init(self, loader, *args, **kwargs):
    self.__old_init__(loader, *args, **kwargs)
    if (
        hasattr(loader, "_speechbrain_recovery_skip_to")
        and loader._speechbrain_recovery_skip_to is not None
    ):
        # Fast forward the sampler iterator since we have recovered:
        for i in range(loader._speechbrain_recovery_skip_to):
            try:
                next(self._sampler_iter)
            except StopIteration:
                MSG = "Tried to fast-forward Sampler after checkpoint "
                f"recovery by {loader._speechbrain_recovery_skip_to} "
                "indices, but now Sampler raised StopIteration after "
                f"{i} indices. Ignoring this mismatch."
                warnings.warn(MSG)
                break
            self._num_yielded = i + 1
        # Mark recovery as done:
        loader._speechbrain_recovery_skip_to = None


def __new_reset(self, loader, first_iter=False, *args, **kwargs):
    # On the first iteration, these have already normally been set by the init anyway.
    # And we don't want to overwrite them if we've recovered
    if not first_iter:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        loader._speechbrain_num_batches = len(loader)
    else:
        # For MultiprocessDataLoader, start from the worker that we have recovered
        if (
            hasattr(self, "_worker_queue_idx_cycle")     # SingleProcessDataLoader no this attribute
            and hasattr(loader, "_speechbrain_next_worker")     # For valid_loader that not used SaveableDataLoader
            and loader._speechbrain_next_worker is not None
        ):
            for _ in range(loader._speechbrain_next_worker):
                next(self._worker_queue_idx_cycle)

            # Mark recovery as done:
            loader._speechbrain_next_worker = None
        else:
            loader._speechbrain_num_batches = len(loader)



# functools.update_wrapper is meant for decorators, but it should basically
# preserve what we want:
functools.update_wrapper(__new_init, _BaseDataLoaderIter.__init__)
_BaseDataLoaderIter.__old_init__ = _BaseDataLoaderIter.__init__
_BaseDataLoaderIter.__init__ = __new_init
if hasattr(_BaseDataLoaderIter, "_reset"):
    _BaseDataLoaderIter._reset = __new_reset


@register_checkpoint_hooks
class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    SpeechBrain's Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None
        # For _MultiProcessingDataLoaderIter._try_put_index().worker_queue_idx
        self._speechbrain_next_worker = None
        self._speechbrain_num_batches = len(self)

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speechbrain_iterator = iterator
        return iterator

    @mark_as_saver
    def _save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = [self._speechbrain_iterator._num_yielded]

            # MultiprocessDataLoader
            if self.num_workers > 0:
                # idx of the next task to be returned in __next__
                rcvd_idx = self._speechbrain_iterator._rcvd_idx
                if rcvd_idx < self._speechbrain_num_batches:
                    # task_info of the current task has been deleted with torch.utils.data.dataloader:1202
                    current_worker_id = self._speechbrain_iterator._task_info[rcvd_idx][0] - 1
                    if current_worker_id == -1: current_worker_id += self.num_workers
                else:
                    current_worker_id = len(self) % self.num_workers - 1
                to_save.append(current_worker_id + 1)

                to_save.append(self.dataset.worker_state)
            else:
                if 'aug_state' in self.dataset.worker_state:
                    to_save.append(self.dataset.worker_state['aug_state'])
            torch.save(to_save, path)

    @mark_as_loader
    def _recover(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return

        saved = torch.load(path)
        if saved is None:
            # Saved at a point where e.g. an iterator did not yet exist.
            return
        else:
            if self.num_workers == 0:
                if not end_of_epoch:
                    # Don't load at end of epoch, as we actually want to start a fresh
                    # epoch iteration next.
                    self._speechbrain_recovery_skip_to = saved[0]
                if len(saved) == 2:
                    self.dataset.aug.recover_aug_state(saved[1])
            else:
                if not end_of_epoch:
                    # Don't load at end of epoch, as we actually want to start a fresh
                    # epoch iteration next.
                    self._speechbrain_recovery_skip_to = saved[0]
                    self._speechbrain_next_worker = saved[1]
                    self.dataset.worker_state = saved[2]
                    self._speechbrain_num_batches -= self._speechbrain_recovery_skip_to


def worker_init_fn(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    if dataset.worker_state != {} and worker_id in dataset.worker_state:
        np.random.set_state(dataset.worker_state[worker_id]['np_state'])
        # random_state (tuple) will be converted to a list when pin_memory=True. See pytorch issue #48419
        random_state = dataset.worker_state[worker_id]['random_state']
        random_state = (random_state[0], tuple(random_state[1]), random_state[2])
        random.setstate(random_state)
        torch.set_rng_state(dataset.worker_state[worker_id]['torch_state'])
        if 'aug_state' in dataset.worker_state[worker_id]:
            dataset.aug.recover_aug_state(dataset.worker_state[worker_id]['aug_state'])
