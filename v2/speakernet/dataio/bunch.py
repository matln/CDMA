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
import torch.distributed as dist
from torch.utils.data import DataLoader

import speakernet.utils.utils as utils
from speakernet.dataio.collate import default_collate
from speakernet.dataio.sampler import SpeakerAwareSampler
from speakernet.dataio.dataset import WaveDataset, SpeakerAwareWaveDataset
from speakernet.dataio.dataloader import SaveableDataLoader, worker_init_fn

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


class DataBunch:
    """DataBunch:(trainset,[valid]).
    """

    def __init__(
        self,
        trainset,
        valid=None,
        batch_size=512,
        valid_batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        speaker_aware_sample=False,
        num_samples_cls=8,
        seed=1024,
    ):

        num_gpu = 1
        self.generator = torch.Generator()

        if utils.use_ddp():
            # The num_replicas/world_size and rank will be set automatically with DDP.
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=shuffle
            )
            shuffle = False
            num_gpu = dist.get_world_size()
        elif speaker_aware_sample:
            if trainset.speed_up is not None:
                spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]
                spk_labels.extend(
                    [
                        int(item["spk_label"]) + trainset.num_targets
                        for item in trainset.data.values()
                    ]
                )
                spk_labels.extend(
                    [
                        int(item["spk_label"]) + 2 * trainset.num_targets
                        for item in trainset.data.values()
                    ]
                )
            else:
                spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]

            train_sampler = SpeakerAwareSampler(
                spk_labels, num_samples_cls=num_samples_cls, generator=self.generator,
            )
            shuffle = False
        else:
            train_sampler = None

        self.train_loader = SaveableDataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            collate_fn=default_collate,
            generator=self.generator,
        )

        self.num_batch_train = len(self.train_loader)
        self.batch_size = batch_size
        self.train_sampler = train_sampler
        self.seed = seed

        if self.num_batch_train <= 0:
            raise ValueError(
                "Expected num_batch of trainset > 0. There are your egs info: num_gpu={}, num_samples/gpu={}, "
                "batch-size={}, drop_last={}.\nNote: If batch-size > num_samples/gpu and drop_last is true, then it "
                "will get 0 batch.".format(num_gpu, len(trainset) / num_gpu, batch_size, drop_last)
            )

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
        trainset_csv: str,
        valid_csv: str = None,
        egs_params: dict = {},
        data_loader_params_dict: dict = {},
    ):
        if data_loader_params_dict["speaker_aware_sample"]:
            trainset = SpeakerAwareWaveDataset(trainset_csv, **egs_params)
        else:
            trainset = WaveDataset(trainset_csv, **egs_params)

        # For multi-GPU training.
        if not utils.is_main_training():
            valid = None
        else:
            if valid_csv != "" and valid_csv is not None:
                egs_params["aug"] = False
                egs_params["aug_conf"] = ""
                valid = WaveDataset(valid_csv, valid=True, **egs_params)
            else:
                valid = None
        return cls(trainset, valid, **data_loader_params_dict)

    @classmethod
    def get_bunch_from_egsdir(
        self, egsdir: str, egs_params: dict = {}, data_loader_params_dict: dict = {}
    ):
        train_csv_name = None
        valid_csv_name = None

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        num_targets, train_csv, valid_csv = self.get_info_from_egsdir(
            egsdir, train_csv_name=train_csv_name, valid_csv_name=valid_csv_name
        )

        # target_num_multiplier = 1

        # If speed perturbation was used before, set target_num_multiplier to 3
        # in egs_params when performing large-margin fine-tuning.
        target_num_multiplier = egs_params.pop("target_num_multiplier", 1)
        if egs_params["aug"]:

            def get_targets_multiplier(aug_classes):
                # target_num_multiplier = 1
                _target_num_multiplier = target_num_multiplier
                for aug_class in aug_classes:
                    if "aug_classes" in aug_class:
                        _target_num_multiplier = get_targets_multiplier(aug_class["aug_classes"])
                    else:
                        if aug_class["aug_type"] == "Speed":
                            return 3
                return _target_num_multiplier

            with open(egs_params["aug_conf"], "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                target_num_multiplier = get_targets_multiplier(speech_aug_conf["aug_classes"])
        info = {"num_targets": num_targets * target_num_multiplier}

        egs_params["num_targets"] = num_targets
        bunch = self.get_bunch_from_csv(train_csv, valid_csv, egs_params, data_loader_params_dict)
        return bunch, info

    @classmethod
    def get_info_from_egsdir(self, egsdir, train_csv_name=None, valid_csv_name=None):
        if os.path.exists(egsdir + "/info"):
            num_targets = int(utils.read_file_to_list(egsdir + "/info/num_targets")[0])

            train_csv_name = train_csv_name if train_csv_name is not None else "train.egs.csv"
            valid_csv_name = valid_csv_name if valid_csv_name is not None else "validation.egs.csv"

            train_csv = egsdir + "/" + train_csv_name
            valid_csv = egsdir + "/" + valid_csv_name

            if not os.path.exists(valid_csv):
                valid_csv = None

            return num_targets, train_csv, valid_csv
        else:
            raise ValueError("Expected dir {0} to exist.".format(egsdir + "/info"))


if __name__ == "__main__":
    data_loader_params_dict = {"batch_size": 5, "shuffle": False, "num_workers": 0}
    bunch, info = DataBunch.get_bunch_from_egsdir(
        "../../../../voxceleb/exp/egs/mfcc_23_pitch-voxceleb1_train_aug-speaker_balance",
        data_loader_params_dict=data_loader_params_dict,
    )
    for i, (data, label) in enumerate(bunch.train_loader):
        print("-----------")
        if i > 2:
            break
        # print(data.size())
        # print(label.size())
