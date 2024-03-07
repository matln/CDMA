# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import yaml
import math
import torch
import random
import logging
import torchaudio
import numpy as np
from rich import print
from torch.utils.data import Dataset

import sys

sys.path.insert(0, "./")

import speakernet.utils.utils as utils
from speakernet.features.augment import SpeechAug

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer


class WaveDataset(Dataset):
    """
    """

    def __init__(
        self,
        egs_csv,
        duration,
        samplerate=16000,
        frame_overlap=0.015,
        random_segment=False,
        replacements={},
        delimiter=",",
        repl_field="wav_path",
        valid=False,
        num_targets=0,
        aug=False,
        aug_conf="",
    ):
        """
        @egs_csv:
            utt_id:str  wav_path:str duration:float  start_position:int  end_position:int  spk_label:int

        Other option
        """
        self.duration = duration
        self.samplerate = samplerate
        self.frame_overlap_size = int(frame_overlap * samplerate)
        self.chunk_size = int(self.duration * samplerate) + self.frame_overlap_size
        self.random_segment = random_segment
        self.num_targets = num_targets

        assert egs_csv != "" and egs_csv is not None
        self.data = utils.load_data_csv(
            egs_csv, replacements, repl_field=repl_field, delimiter=delimiter
        )
        self.data_ids = list(self.data.keys())

        if (
            "start_position" in self.data[self.data_ids[0]].keys()
            and "end_position" in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = True
        elif (
            "start_position" not in self.data[self.data_ids[0]].keys()
            and "end_position" not in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = False
        else:
            raise TypeError(
                "Expected both start-position and end-position are exist in {}.".format(egs_csv)
            )

        self.worker_state = {}
        self.valid = valid

        # Augmentation.
        if aug:
            # self.aug = AugmentOnline(aug, aug_params)
            with open(aug_conf, "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
            self.aug = SpeechAug(**speech_aug_conf)
        else:
            self.aug = None

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        wav_size = int(data_point["duration"] * self.samplerate)
        if wav_size < self.chunk_size:
            # logger.warning(f"wav_size {wav_size} < self.chunk_size {self.chunk_size}")
            pass

        if self.chunk_position:
            wav_path = data_point["wav_path"].split(" ")
            if self.random_segment:
                if len(wav_path) == 1:
                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size
                    sig, fs = utils.load_wavs(data_point["wav_path"], str(start), str(stop))
                else:
                    every_size = [int(x) for x in data_point["wav_size"].split(" ")]
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)

                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size

                    for i, item in enumerate(cumsum_size):
                        if start - item < 0:
                            start_position = "{}_{}".format(i - 1, start - cumsum_size[i - 1])
                            break
                    for i, item in enumerate(cumsum_size):
                        if stop - item <= 0:
                            end_position = "{}_{}".format(i - 1, stop - cumsum_size[i - 1])
                            break

                    sig, fs = utils.load_wavs(data_point["wav_path"], start_position, end_position)
            else:
                sig, fs = utils.load_wavs(
                    data_point["wav_path"], data_point["start_position"], data_point["end_position"]
                )
        else:
            # Custom validset
            assert self.aug is None
            sig, fs = utils.load_wavs(data_point["wav_path"])

        label = int(data_point["spk_label"])

        if self.aug is not None:
            sig, label_multiplier = self.aug(sig)
            label = label + label_multiplier * self.num_targets

        # 1. self.chunk_position is True (self.aug may be None), but wav_size < self.chunk_size
        # 2. sig.size(1) == self.chunk_size, but SpeedPerturb or TempoPerturb in self.aug change the sig length
        if self.chunk_position:
            if sig.size(1) > self.chunk_size:
                start = random.randint(0, sig.size(1) - self.chunk_size)
                sig = sig[:, start : start + self.chunk_size]
            else:
                pad_warp_num = self.chunk_size // sig.size(1)
                pad_size = self.chunk_size % sig.size(1)
                cat_list = [sig for _ in range(pad_warp_num)]
                if pad_size != 0:
                    pad_start = random.randint(0, sig.size(1) - pad_size)
                    pad_chunk = sig[:, pad_start : pad_start + pad_size]
                    cat_list.append(pad_chunk)
                sig = torch.cat(cat_list, dim=1)

        if not self.valid:
            # state will be omitted when num_workers=0, i.e., SingleProcessDataLoader
            state = self.get_random_state()
            return sig.squeeze(0), label, state
        else:
            return sig.squeeze(0), label

    def get_random_state(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_id = torch.utils.data.get_worker_info().id
            np_state = np.random.get_state()
            random_state = random.getstate()
            torch_state = torch.get_rng_state()
            worker_state = {
                "worker_id": worker_id,
                "np_state": np_state,
                "random_state": random_state,
                "torch_state": torch_state,
            }
            if self.aug is not None:
                worker_state["aug_state"] = self.aug.get_aug_state()
            return worker_state
        else:
            # SingleProcessing
            if self.aug is not None:
                self.worker_state["aug_state"] = self.aug.get_aug_state()
            return {}

    def __len__(self):
        return len(self.data_ids)


class SpeakerAwareWaveDataset(WaveDataset):
    """
    When performing speed perturbation, it is inconvenient to obtain all spk labels
    for the SpeakerAwareSampler in advance. Thus we split the speed perturbation and
    other augmentation methods, and perform speed perturbation according to the
    label multiplier derived by index.
    """

    def __init__(
        self,
        egs_csv,
        duration,
        samplerate=16000,
        frame_overlap=0.015,
        random_segment=False,
        replacements={},
        delimiter=",",
        repl_field="wav_path",
        valid=False,
        num_targets=0,
        aug=False,
        aug_conf="",
    ):
        """
        @egs_csv:
            utt_id: str  wav_path: str duration: float  start_position:int  end_position:int  spk_label:int

        Other option
        """
        self.duration = duration
        self.samplerate = samplerate
        self.frame_overlap_size = int(frame_overlap * samplerate)
        self.chunk_size = int(self.duration * samplerate) + self.frame_overlap_size
        self.random_segment = random_segment
        self.num_targets = num_targets

        assert egs_csv != "" and egs_csv is not None
        self.data = utils.load_data_csv(
            egs_csv, replacements, repl_field=repl_field, delimiter=delimiter
        )
        self.data_ids = list(self.data.keys())

        if (
            "start_position" in self.data[self.data_ids[0]].keys()
            and "end_position" in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = True
        elif (
            "start_position" not in self.data[self.data_ids[0]].keys()
            and "end_position" not in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = False
        else:
            raise TypeError(
                "Expected both start-position and end-position are exist in {}.".format(egs_csv)
            )

        self.worker_state = {}
        self.valid = valid
        self.data_lens = len(self.data_ids)

        # Augmentation.
        # It is assumed that speed perturbation is performed prior to other augmentation methods
        # and the mode is chain.
        if aug:
            with open(aug_conf, "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                if (
                    speech_aug_conf["mod"] == "chain"
                    and speech_aug_conf["aug_classes"][0]["aug_type"] == "Speed"
                ):
                    _ = speech_aug_conf["aug_classes"].pop(0)
                    speed_up_conf = [
                        {
                            "aug_name": "speed_up",
                            "aug_type": "Speed",
                            "perturb_prob": 1.0,
                            "sample_rate": 16000,
                            "speeds": [1.1],
                        }
                    ]
                    speed_down_conf = [
                        {
                            "aug_name": "speed_down",
                            "aug_type": "Speed",
                            "perturb_prob": 1.0,
                            "sample_rate": 16000,
                            "speeds": [0.9],
                        }
                    ]
                    self.speed_up = SpeechAug(aug_classes=speed_up_conf, mod="chain")
                    self.speed_down = SpeechAug(aug_classes=speed_down_conf, mod="chain")
                else:
                    self.speed_up = None
                    self.speed_down = None
            self.aug = SpeechAug(**speech_aug_conf)
        else:
            self.aug = None
            self.speed_up = None
            self.speed_down = None

    def __getitem__(self, index):
        label_multiplier = index // self.data_lens
        index = index % self.data_lens

        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        wav_size = int(data_point["duration"] * self.samplerate)
        if wav_size < self.chunk_size:
            # logger.warning(f"wav_size {wav_size} < self.chunk_size {self.chunk_size}")
            pass

        if self.chunk_position:
            wav_path = data_point["wav_path"].split(" ")
            if self.random_segment:
                if len(wav_path) == 1:
                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size
                    sig, fs = utils.load_wavs(data_point["wav_path"], str(start), str(stop))
                else:
                    every_size = [int(x) for x in data_point["wav_size"].split(" ")]
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)

                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size

                    for i, item in enumerate(cumsum_size):
                        if start - item < 0:
                            start_position = "{}_{}".format(i - 1, start - cumsum_size[i - 1])
                            break
                    for i, item in enumerate(cumsum_size):
                        if stop - item <= 0:
                            end_position = "{}_{}".format(i - 1, stop - cumsum_size[i - 1])
                            break

                    sig, fs = utils.load_wavs(data_point["wav_path"], start_position, end_position)
            else:
                sig, fs = utils.load_wavs(
                    data_point["wav_path"], data_point["start_position"], data_point["end_position"]
                )
        else:
            # Custom validset
            assert self.aug is None
            sig, fs = utils.load_wavs(data_point["wav_path"])

        label = int(data_point["spk_label"])

        if label_multiplier == 1:
            sig, multiplier = self.speed_down(sig)
        elif label_multiplier == 2:
            sig, multiplier = self.speed_up(sig)

        if self.aug is not None:
            sig, multiplier = self.aug(sig)
            assert multiplier == 0

        label = label + label_multiplier * self.num_targets

        # 1. self.chunk_position is True (self.aug may be None), but wav_size < self.chunk_size
        # 2. sig.size(1) == self.chunk_size, but SpeedPerturb or TempoPerturb in self.aug change the sig length
        if self.chunk_position:
            if sig.size(1) > self.chunk_size:
                start = random.randint(0, sig.size(1) - self.chunk_size)
                sig = sig[:, start : start + self.chunk_size]
            else:
                pad_warp_num = self.chunk_size // sig.size(1)
                pad_size = self.chunk_size % sig.size(1)
                cat_list = [sig for _ in range(pad_warp_num)]
                if pad_size != 0:
                    pad_start = random.randint(0, sig.size(1) - pad_size)
                    pad_chunk = sig[:, pad_start : pad_start + pad_size]
                    cat_list.append(pad_chunk)
                sig = torch.cat(cat_list, dim=1)

        if not self.valid:
            # state will be omitted when num_workers=0, i.e., SingleProcessDataLoader
            state = self.get_random_state()
            return sig.squeeze(0), label, state
        else:
            return sig.squeeze(0), label

    def __len__(self):
        if self.speed_up is not None:
            return self.data_lens * 3
        else:
            return self.data_lens


if __name__ == "__main__":
    from speakernet.dataio.collate import default_collate
    from speakernet.dataio.sampler import SpeakerAwareSampler
    from speakernet.dataio.dataloader import SaveableDataLoader, worker_init_fn

    trainset = SpeakerAwareWaveDataset(
        "/data/lijianchen/workspace/sre/CDMA+/exp/egs/unlabeled_waveform_2s/train.egs.csv",
        2,
        samplerate=16000,
        num_targets=1806,
        aug=True,
        aug_conf="/data/lijianchen/workspace/sre/CDMA+/hparams/speech_aug_chain.yaml",
    )

    spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]
    spk_labels.extend(
        [int(item["spk_label"]) + trainset.num_targets for item in trainset.data.values()]
    )
    spk_labels.extend(
        [int(item["spk_label"]) + 2 * trainset.num_targets for item in trainset.data.values()]
    )

    generator = torch.Generator()
    train_sampler = SpeakerAwareSampler(spk_labels, num_samples_cls=4, generator=generator,)

    train_loader = SaveableDataLoader(
        trainset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        collate_fn=default_collate,
        generator=generator,
    )

    for i, batch in enumerate(train_loader):
        data, targets, state = batch
        print(targets)
        if i == 100:
            break
