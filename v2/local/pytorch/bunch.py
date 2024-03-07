# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""

import os
import yaml
import torch
import logging
from rich import print
from copy import deepcopy
import torch.distributed as dist
from torch.utils.data import DataLoader

import speakernet.utils.utils as utils
from speakernet.dataio.collate import default_collate
from speakernet.dataio.sampler import SpeakerAwareSampler
from speakernet.dataio.dataloader import SaveableDataLoader, worker_init_fn

from local.pytorch.dataset import WaveDataset, SpeakerAwareWaveDataset
# from local.pytorch.dataset1 import WaveDataset, SpeakerAwareWaveDataset
from local.pytorch.sampler import RandomSpeakerBatchSampler, RandomSpeakerSampler

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


class DataBunch:
    """DataBunch:(trainset,[valid]).
    """

    def __init__(
        self,
        s_trainset,
        t_trainset,
        valid=None,
        batch_size=512,
        valid_batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        speaker_aware_sample=False,
        num_samples_cls=8,
        prefetch_factor=2,
        seed=1024,
    ):

        self.s_generator = torch.Generator()
        self.t_generator = torch.Generator()

        if utils.use_ddp():
            # The num_replicas/world_size and rank will be set automatically with DDP.
            s_train_sampler = torch.utils.data.distributed.DistributedSampler(
                s_trainset, shuffle=shuffle
            )
            t_train_sampler = torch.utils.data.distributed.DistributedSampler(
                t_trainset, shuffle=shuffle
            )
            shuffle = False
        elif speaker_aware_sample:
            if s_trainset.speed_up is not None:
                s_spk_labels = [int(item["spk_label"]) for item in s_trainset.data.values()]
                s_spk_labels.extend(
                    [
                        int(item["spk_label"]) + s_trainset.num_targets
                        for item in s_trainset.data.values()
                    ]
                )
                s_spk_labels.extend(
                    [
                        int(item["spk_label"]) + 2 * s_trainset.num_targets
                        for item in s_trainset.data.values()
                    ]
                )
            else:
                s_spk_labels = [int(item["spk_label"]) for item in s_trainset.data.values()]
            s_train_sampler = SpeakerAwareSampler(
                s_spk_labels,
                num_samples_cls=num_samples_cls,
                generator=self.s_generator,
            )
            # s_train_sampler = RandomSpeakerSampler(s_spk_labels, batch_size, num_samples_cls)

            if t_trainset.speed_up is not None:
                t_spk_labels = [int(item["spk_label"]) for item in t_trainset.data.values()]
                t_spk_labels.extend(
                    [
                        int(item["spk_label"]) + t_trainset.num_targets
                        for item in t_trainset.data.values()
                    ]
                )
                t_spk_labels.extend(
                    [
                        int(item["spk_label"]) + 2 * t_trainset.num_targets
                        for item in t_trainset.data.values()
                    ]
                )
            else:
                t_spk_labels = [int(item["spk_label"]) for item in t_trainset.data.values()]
            # t_spk_labels = [int(item["spk_label"]) for item in t_trainset.data.values()]
            t_train_sampler = SpeakerAwareSampler(
                t_spk_labels,
                num_samples_cls=num_samples_cls,
                generator=self.t_generator,
            )
            # t_train_sampler = RandomSpeakerSampler(t_spk_labels, batch_size, num_samples_cls)
            shuffle = False
        else:
            s_train_sampler = None
            t_train_sampler = None

        self.s_train_loader = SaveableDataLoader(
            s_trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=s_train_sampler,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            collate_fn=default_collate,
            generator=self.s_generator,
        )

        self.t_train_loader = SaveableDataLoader(
            t_trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=t_train_sampler,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            collate_fn=default_collate,
            generator=self.t_generator,
        )

        self.num_batch_train = len(self.t_train_loader)
        # self.batch_size = batch_size
        self.s_train_sampler = s_train_sampler
        self.t_train_sampler = t_train_sampler
        self.seed = seed

        if len(self.s_train_loader) <= 0 or len(self.t_train_loader) <= 0:
            raise ValueError("Expected num_batch of trainset > 0")

        if valid is not None:
            valid_batch_size = min(valid_batch_size, len(valid))  # To save GPU memory

            if len(valid) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            self.valid_loader = DataLoader(
                valid,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=pin_memory,
                drop_last=False,
            )

            self.num_batch_valid = len(self.valid_loader)
        else:
            self.valid_loader = None
            self.num_batch_valid = 0

    @classmethod
    def get_bunch_from_csv(
        cls,
        s_trainset_csv: str,
        t_trainset_csv: str,
        valid_csv: str = None,
        s_egs_params: dict = {},
        t_egs_params: dict = {},
        data_loader_params_dict: dict = {},
    ):
        if data_loader_params_dict["speaker_aware_sample"]:
            s_trainset = SpeakerAwareWaveDataset(s_trainset_csv, **s_egs_params)
            t_trainset = SpeakerAwareWaveDataset(t_trainset_csv, **t_egs_params)
        else:
            s_trainset = WaveDataset(s_trainset_csv, **s_egs_params)
            t_trainset = WaveDataset(t_trainset_csv, **t_egs_params)

        # For multi-GPU training.
        if not utils.is_main_training():
            valid = None
        else:
            if valid_csv != "" and valid_csv is not None:
                s_egs_params["aug"] = False
                s_egs_params["aug_conf"] = ""
                valid = WaveDataset(valid_csv, valid=True, **s_egs_params)
            else:
                valid = None
        return cls(s_trainset, t_trainset, valid, **data_loader_params_dict)

    @classmethod
    def get_bunch_from_egsdir(
        self,
        s_egsdir: str,
        t_egsdir: str,
        egs_params: dict = {},
        data_loader_params_dict: dict = {},
    ):
        train_csv_name = None
        valid_csv_name = None

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        num_s_targets, s_train_csv, s_valid_csv = self.get_info_from_egsdir(
            s_egsdir, train_csv_name=train_csv_name, valid_csv_name=valid_csv_name
        )
        num_t_targets, t_train_csv, t_valid_csv = self.get_info_from_egsdir(
            t_egsdir, train_csv_name=train_csv_name, valid_csv_name=valid_csv_name
        )

        def get_targets_multiplier(aug_classes):
            num_targets_multiplier = 1
            for aug_class in aug_classes:
                if "aug_classes" in aug_class:
                    num_targets_multiplier = get_targets_multiplier(
                        aug_class["aug_classes"]
                    )
                else:
                    if aug_class["aug_type"] == "Speed":
                        return 3
            return num_targets_multiplier

        num_targets_multiplier = egs_params.pop("s_target_num_multiplier", 1)
        if egs_params["source_aug"] and num_targets_multiplier == 1:
            with open(egs_params["source_aug_conf"], "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                num_targets_multiplier = get_targets_multiplier(
                    speech_aug_conf["aug_classes"]
                )
        info = {"num_s_targets": num_s_targets * num_targets_multiplier}
        s_egs_params = deepcopy(egs_params)
        s_egs_params["num_targets"] = num_s_targets
        s_egs_params["aug"] = s_egs_params["source_aug"]
        s_egs_params["aug_conf"] = s_egs_params["source_aug_conf"]
        del (
            s_egs_params["source_aug"],
            s_egs_params["source_aug_conf"],
            s_egs_params["target_aug"],
            s_egs_params["target_aug_conf"],
        )

        num_targets_multiplier = 1
        if egs_params["target_aug"]:
            with open(egs_params["target_aug_conf"], "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                num_targets_multiplier = get_targets_multiplier(
                    speech_aug_conf["aug_classes"]
                )
        info["num_t_targets"] = num_t_targets * num_targets_multiplier
        t_egs_params = deepcopy(egs_params)
        t_egs_params["num_targets"] = num_t_targets
        t_egs_params["aug"] = t_egs_params["target_aug"]
        t_egs_params["aug_conf"] = t_egs_params["target_aug_conf"]
        del (
            t_egs_params["source_aug"],
            t_egs_params["source_aug_conf"],
            t_egs_params["target_aug"],
            t_egs_params["target_aug_conf"],
        )

        bunch = self.get_bunch_from_csv(
            s_train_csv,
            t_train_csv,
            s_valid_csv,
            s_egs_params,
            t_egs_params,
            data_loader_params_dict,
        )
        return bunch, info

    @classmethod
    def get_info_from_egsdir(self, egsdir, train_csv_name=None, valid_csv_name=None):
        if os.path.exists(egsdir + "/info"):
            num_targets = int(utils.read_file_to_list(egsdir + "/info/num_targets")[0])

            train_csv_name = (
                train_csv_name if train_csv_name is not None else "train.egs.csv"
            )
            valid_csv_name = (
                valid_csv_name if valid_csv_name is not None else "validation.egs.csv"
            )

            train_csv = egsdir + "/" + train_csv_name
            valid_csv = egsdir + "/" + valid_csv_name

            if not os.path.exists(valid_csv):
                valid_csv = None

            return num_targets, train_csv, valid_csv
        else:
            raise ValueError("Expected dir {0} to exist.".format(egsdir + "/info"))


if __name__ == "__main__":
    pass
