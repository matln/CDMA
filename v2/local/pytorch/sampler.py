import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

# import libs.support.utils as utils
# utils.set_all_seed(666, deterministic=True)


class RandomSpeakerSampler(Sampler):
    """Randomly samples N identities each with K instances.
    Args:
        data_source (list): contains lists of label.
        batch_size (int): batch size.
        num_chunks (int): number of chunks per speaker in a batch.
    """

    def __init__(self, data_source, batch_size, num_chunks):
        if batch_size < num_chunks:
            raise ValueError(
                "batch_size={} must be no less " "than num_chunks={}".format(batch_size, num_chunks)
            )

        self.data_source = np.array(data_source)
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.num_spks_per_batch = self.batch_size // self.num_chunks
        self.spk2idxs = defaultdict(list)
        for idx, label in enumerate(self.data_source):
            self.spk2idxs[label].append(idx)
        self.spks = list(self.spk2idxs.keys())
        print(len(self.spks))

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for spk in self.spks:
            idxs = self.spk2idxs[spk]
            num = len(idxs)
            if num < self.num_chunks:
                num = self.num_chunks
            self.length += num - num % self.num_chunks
        # self.__iter__()

    def __iter__(self):
        # print(np.random.get_state()[1][0], np.random.get_state()[1][1])
        batch_spk2idxs = defaultdict(list)

        for spk in self.spks:
            idxs = copy.deepcopy(self.spk2idxs[spk])
            if len(idxs) < self.num_chunks:
                idxs = np.random.choice(idxs, size=self.num_chunks, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_chunks:
                    batch_spk2idxs[spk].append(batch_idxs)
                    batch_idxs = []

        avai_spks = copy.deepcopy(self.spks)
        final_idxs = []

        while len(avai_spks) >= self.num_spks_per_batch:
            selected_spks = random.sample(avai_spks, self.num_spks_per_batch)
            for spk in selected_spks:
                batch_idxs = batch_spk2idxs[spk].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_spk2idxs[spk]) == 0:
                    avai_spks.remove(spk)
        print("------------------")
        print(len(final_idxs))
        print("------------------")

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomSpeakerBatchSampler(Sampler):
    """
    """

    def __init__(self, data_source, num_spks, num_chunks):
        self.data_source = np.array(data_source)
        self.spks = list(set(self.data_source))
        print(len(self.spks))
        self.spk2idxs = {spk: np.where(self.data_source == spk)[0] for spk in self.spks}
        print(self.spk2idxs)
        self.used_count = {spk: 0 for spk in self.spks}
        self.count = 0
        self.num_spks = num_spks
        self.num_chunks = num_chunks

        # estimate number of examples in an epoch
        self.length = 0
        for spk in self.spks:
            idxs = self.spk2idxs[spk]
            num = len(idxs) / self.num_chunks
            if num < self.num_chunks:
                num = self.num_chunks
            self.length += num - num % self.num_chunks

    def __iter__(self):
        # generator
        for spk in self.spks:
            random.shuffle(self.spk2idxs[spk])
        avai_spks = copy.deepcopy(self.spks)
        # TODO
        while len(avai_spks) > 0:
            if len(avai_spks) >= self.num_spks:
                current_spks = random.sample(avai_spks, self.num_spks)
            else:
                current_spks = random.sample(
                    list(set(self.spks) - set(avai_spks)), self.num_spks - len(avai_spks)
                )
                current_spks.extend(avai_spks)
            idxs = []
            for spk in current_spks:
                _idxs = self.spk2idxs[spk]
                if len(_idxs) < self.num_chunks:
                    print(spk)
                    _idxs = np.random.choice(_idxs, size=self.num_chunks, replace=True)

                idxs.extend(_idxs[self.used_count[spk] : self.used_count[spk] + self.num_chunks])
                self.used_count[spk] += self.num_chunks
                if self.used_count[spk] + self.num_chunks > len(self.spk2idxs[spk]):
                    if spk in avai_spks:
                        avai_spks.remove(spk)
                    random.shuffle(self.spk2idxs[spk])
                    self.used_count[spk] = 0
            yield idxs

    def __len__(self):
        return self.length
