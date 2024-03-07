# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import copy
import torch
import logging
import torchaudio
import numpy as np
import pandas as pd

from speakernet.utils.utils import load_wavs
from speakernet.utils.rich_utils import track
from speakernet.pipelines.modules.kaldi_dataset import KaldiDataset

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


class WaveformSamples:
    def __init__(
        self,
        dataset: KaldiDataset,
        duration: float,
        chunk_type="speaker_balance",
        chunk_num_selection=0,
        scale=1.5,
        overlap=0.1,
        drop_last_duration=0.2,
        frame_overlap=0.015,
        samplerate=16000,
        min_duration=2.0,
        amp_th=0.0005,
    ):
        """
        Parameters:
            self.dataset: the object which contain the dicts such as utt2spk, utt2spk_int and so on.
            self.chunk_size: the number of frames in a chunk.
            self.chunk_type: which decides how to chunk the feats for training.
            chunk_num_selection: -1->suggestion scale, 0->max, >0->specify.
            self.overlap: the proportion of overlapping for every chunk.
        """
        self.dataset = dataset
        self.duration = duration
        self.chunk_type = chunk_type
        self.chunk_num_selection = chunk_num_selection
        self.scale = scale
        self.overlap = overlap
        self.amp_th = amp_th
        self.samplerate = samplerate
        self.min_duration = min_duration
        self.chunk_size = int((self.duration + frame_overlap) * samplerate)
        self.min_chunk_size = int((self.min_duration + frame_overlap) * samplerate)
        self.overlap_size = int(
            self.overlap * self.duration * samplerate + frame_overlap * samplerate
        )
        self.drop_last_size = int((drop_last_duration + frame_overlap) * samplerate)
        if self.overlap == 0:
            self.overlap_size = 0

        assert 0 <= self.overlap < 1

        if self.chunk_type == "full_length":
            self.head = ["utt_id", "wav_path", "spk_label"]
        elif self.chunk_type in ["sequential_with_domain"]:
            self.head = [
                "utt_id",
                "wav_path",
                "duration",
                "wav_size",
                "start_position",
                "end_position",
                "spk_label",
                "dom_label",
            ]
        else:
            self.head = [
                "utt_id",
                "wav_path",
                "duration",
                "wav_size",
                "start_position",
                "end_position",
                "spk_label",
            ]

        self.skipped_utts = []
        self.chunk_samples = self.__sample()

    def __sample(self):
        # JFZhou: speaker_balance and sequential.
        chunk_samples = []

        if self.chunk_type == "speaker_balance":
            self.spk2chunks = {}
            utt2dur = {}
            total_chunks = 0
            # max number of chunks for all speakers
            max_chunk_num = 0
            # chunk num of every utterance
            chunk_counter = {}
            for spk in self.dataset.spk2utt.keys():
                utt_selected = self.dataset.spk2utt[spk]
                # chunk num of every speaker
                spk_chunk_num = 0
                for utt in utt_selected:
                    wav_path = self.dataset.wav_scp[utt]
                    if len(wav_path) == 1:
                        single_wav = True
                        wav_path = wav_path[0]
                        signal, fs, wav_size = load_wavs(wav_path, return_size=True)
                        every_size = [wav_size]
                    else:
                        # Format:
                        # utt1 sox audio1.wav audio2.wav -t wav - |
                        single_wav = False
                        assert ".wav" not in wav_path[0]
                        wav_path.pop(0)
                        for _ in range(1, 5, 1):
                            assert ".wav" not in wav_path[-1]
                            wav_path.pop(-1)
                        wav_path = " ".join(wav_path)
                        signal, fs, every_size = load_wavs(wav_path, return_size=True)
                        wav_size = sum(every_size)
                        cumsum_size = list(np.cumsum(every_size))
                        cumsum_size.insert(0, 0)
                    signal = signal.squeeze(0)
                    wav_duration = wav_size / self.samplerate
                    utt2dur[utt] = wav_duration

                    if wav_size < self.chunk_size:
                        if wav_size < self.min_chunk_size:
                            logger.warn(
                                "The sample num {0} of {1} is less than the minimum {2}, "
                                "so skip it.".format(wav_size, utt, self.min_chunk_size)
                            )
                        else:
                            # Avoid chunks with very small energy
                            mean_sig = torch.mean(np.abs(signal))
                            if mean_sig < self.amp_th:
                                logger.warn(f"Skip {utt}")
                            else:
                                if single_wav:
                                    start = 0
                                    end = wav_size
                                else:
                                    start = "0_0"
                                    end = "{}_{}".format(len(every_size) - 1, every_size[-1])
                                chunk = "{0} {1} {2} {3} {4}".format(
                                    utt + "-0",
                                    wav_path,
                                    wav_duration,
                                    " ".join([str(x) for x in every_size]),
                                    start,
                                    end,
                                    self.dataset.utt2spk_int[utt],
                                )

                                self.spk2chunks.setdefault(spk, []).append(chunk)

                                total_chunks += 1
                                spk_chunk_num += 1
                    else:
                        chunk_counter[utt] = 0
                        offset = 0
                        while offset + self.chunk_size <= wav_size:
                            start_position = offset
                            end_position = start_position + self.chunk_size
                            offset += self.chunk_size - self.overlap_size

                            #  Avoid chunks with very small energy
                            mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                            if mean_sig < self.amp_th:
                                logger.info(f"Skip {start_position}-{end_position} in {utt}")
                            else:
                                if single_wav:
                                    start = start_position
                                    end = end_position
                                else:
                                    for i, item in enumerate(cumsum_size):
                                        if start_position - item < 0:
                                            start = "{}_{}".format(
                                                i - 1, start_position - cumsum_size[i - 1]
                                            )
                                            break
                                    for i, item in enumerate(cumsum_size):
                                        if end_position - item <= 0:
                                            end = "{}_{}".format(
                                                i - 1, end_position - cumsum_size[i - 1]
                                            )
                                            break

                                chunk = "{0} {1} {2} {3} {4} {5}".format(
                                    utt + "-" + str(chunk_counter[utt]),
                                    wav_path,
                                    wav_duration,
                                    " ".join([str(x) for x in every_size]),
                                    start,
                                    end,
                                    self.dataset.utt2spk_int[utt],
                                )

                                self.spk2chunks.setdefault(spk, []).append(chunk)

                                chunk_counter[utt] += 1
                                total_chunks += 1
                                spk_chunk_num += 1

                        if offset + self.drop_last_size < wav_size:
                            start_position = wav_size - self.chunk_size
                            end_position = wav_size

                            #  Avoid chunks with very small energy
                            mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                            if mean_sig < self.amp_th:
                                logger.info(f"Skip {start_position}-{end_position} in {utt}")
                            else:
                                if single_wav:
                                    start = start_position
                                    end = end_position
                                else:
                                    for i, item in enumerate(cumsum_size):
                                        if start_position - item < 0:
                                            start = "{}_{}".format(
                                                i - 1, start_position - cumsum_size[i - 1]
                                            )
                                            break
                                    for i, item in enumerate(cumsum_size):
                                        if end_position - item <= 0:
                                            end = "{}_{}".format(
                                                i - 1, end_position - cumsum_size[i - 1]
                                            )
                                            break

                                chunk = "{0} {1} {2} {3} {4} {5}".format(
                                    utt + "-" + str(chunk_counter[utt]),
                                    wav_path,
                                    wav_duration,
                                    " ".join([str(x) for x in every_size]),
                                    start,
                                    end,
                                    self.dataset.utt2spk_int[utt],
                                )
                                self.spk2chunks[spk].append(chunk)
                                total_chunks += 1
                                spk_chunk_num += 1
                                chunk_counter[utt] += 1

                if spk_chunk_num > max_chunk_num:
                    max_chunk_num = spk_chunk_num

            for spk in self.spk2chunks.keys():
                chunk_selected = self.spk2chunks[spk]
                if self.chunk_num_selection == 0:
                    num_chunks_selected = max_chunk_num
                elif self.chunk_num_selection == -1:
                    num_chunks_selected = int(
                        total_chunks // len(self.dataset.spk2utt) * self.scale
                    )
                else:
                    num_chunks_selected = self.chunk_num_selection

                num_chunks = len(chunk_selected)
                # TODO
                if num_chunks < num_chunks_selected:
                    # valid rather than validation
                    # Make up the insufficient chunks
                    valid_utts = [
                        utt for utt in self.dataset.spk2utt[spk] if utt2dur[utt] >= self.duration
                    ]
                    utts = np.random.choice(
                        valid_utts, num_chunks_selected - num_chunks, replace=True
                    )
                    signal = torch.load(self.dataset.wav_scp[utt])

                    for utt in utts:
                        start_position = np.random.randint(
                            0, int(utt2dur[utt] * self.samplerate) - self.chunk_size + 1
                        )
                        end_position = start_position + self.chunk_size

                        # TODO:
                        # TODO: The case of multiple wavs
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            skipped_chunks += 1
                        else:
                            chunk_selected.append(
                                "{0} {1} {2} {3} {4}".format(
                                    utt + "-" + str(chunk_counter[utt]),
                                    self.dataset.wav_scp[utt],
                                    utt2dur[utt],
                                    start_position,
                                    end_position,
                                    self.dataset.utt2spk_int[utt],
                                )
                            )
                            chunk_counter[utt] += 1
                else:
                    chunk_selected = np.random.choice(
                        self.spk2chunks[spk], num_chunks_selected, replace=False
                    )

                for chunk in chunk_selected:
                    chunk_samples.append(chunk.split())

        elif self.chunk_type == "sequential":
            self.spk2chunks = {}
            for utt, spk in track(self.dataset.utt2spk.items()):
                wav_path = self.dataset.wav_scp[utt]
                if len(wav_path) == 1:
                    single_wav = True
                    wav_path = wav_path[0]
                    signal, fs, wav_size = load_wavs(wav_path, return_size=True)
                    every_size = [wav_size]
                else:
                    # Format:
                    # utt1 sox audio1.wav audio2.wav -t wav - |
                    single_wav = False
                    assert ".wav" not in wav_path[0]
                    wav_path.pop(0)
                    for _ in range(1, 5, 1):
                        assert ".wav" not in wav_path[-1]
                        wav_path.pop(-1)
                    wav_path = " ".join(wav_path)
                    signal, fs, every_size = load_wavs(wav_path, return_size=True)
                    wav_size = sum(every_size)
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)
                signal = signal.squeeze(0)
                wav_duration = wav_size / self.samplerate

                if wav_size < self.chunk_size:
                    if wav_size < self.min_chunk_size:
                        logger.warn(
                            "The sample num {0} of {1} is less than the minimum {2}, "
                            "so skip it.".format(wav_size, utt, self.min_chunk_size)
                        )
                        self.skipped_utts.append(utt)
                    else:
                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {utt}")
                            self.skipped_utts.append(utt)
                        else:
                            if single_wav:
                                start = 0
                                end = wav_size
                            else:
                                start = "0_0"
                                end = "{}_{}".format(len(every_size) - 1, every_size[-1])
                            chunk = [
                                utt + "-0",
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                else:
                    chunk_counter = 0
                    offset = 0
                    while offset + self.chunk_size <= wav_size:
                        start_position = offset
                        end_position = start_position + self.chunk_size
                        offset += self.chunk_size - self.overlap_size
                        # assert self.chunk_size - self.overlap_size == 43200

                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {start_position}-{end_position} in {utt}")
                        else:
                            if single_wav:
                                start = start_position
                                end = end_position
                            else:
                                for i, item in enumerate(cumsum_size):
                                    if start_position - item < 0:
                                        start = "{}_{}".format(
                                            i - 1, start_position - cumsum_size[i - 1]
                                        )
                                        break
                                for i, item in enumerate(cumsum_size):
                                    if end_position - item <= 0:
                                        end = "{}_{}".format(
                                            i - 1, end_position - cumsum_size[i - 1]
                                        )
                                        break

                            chunk = [
                                utt + "-" + str(chunk_counter),
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                            chunk_counter += 1

                    if offset + self.drop_last_size < wav_size:
                        start_position = wav_size - self.chunk_size
                        end_position = wav_size

                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {start_position}-{end_position} in {utt}")
                            if chunk_counter == 0:
                                self.skipped_utts.append(utt)
                        else:
                            if single_wav:
                                start = start_position
                                end = end_position
                            else:
                                for i, item in enumerate(cumsum_size):
                                    if start_position - item < 0:
                                        start = "{}_{}".format(
                                            i - 1, start_position - cumsum_size[i - 1]
                                        )
                                        break
                                for i, item in enumerate(cumsum_size):
                                    if end_position - item <= 0:
                                        end = "{}_{}".format(
                                            i - 1, end_position - cumsum_size[i - 1]
                                        )
                                        break
                            chunk = [
                                utt + "-" + str(chunk_counter),
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                    else:
                        if chunk_counter == 0:
                            self.skipped_utts.append(utt)

        elif self.chunk_type == "sequential_with_domain":
            self.spk2chunks = {}
            for utt, spk in track(self.dataset.utt2spk.items()):
                wav_path = self.dataset.wav_scp[utt]
                if len(wav_path) == 1:
                    single_wav = True
                    wav_path = wav_path[0]
                    signal, fs, wav_size = load_wavs(wav_path, return_size=True)
                    every_size = [wav_size]
                else:
                    # Format:
                    # utt1 sox audio1.wav audio2.wav -t wav - |
                    single_wav = False
                    assert ".wav" not in wav_path[0]
                    wav_path.pop(0)
                    for _ in range(1, 5, 1):
                        assert ".wav" not in wav_path[-1]
                        wav_path.pop(-1)
                    wav_path = " ".join(wav_path)
                    signal, fs, every_size = load_wavs(wav_path, return_size=True)
                    wav_size = sum(every_size)
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)
                signal = signal.squeeze(0)
                wav_duration = wav_size / self.samplerate

                if wav_size < self.chunk_size:
                    if wav_size < self.min_chunk_size:
                        logger.warn(
                            "The sample num {0} of {1} is less than the minimum {2}, "
                            "so skip it.".format(wav_size, utt, self.min_chunk_size)
                        )
                        self.skipped_utts.append(utt)
                    else:
                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {utt}")
                            self.skipped_utts.append(utt)
                        else:
                            if single_wav:
                                start = 0
                                end = wav_size
                            else:
                                start = "0_0"
                                end = "{}_{}".format(len(every_size) - 1, every_size[-1])
                            chunk = [
                                utt + "-0",
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                                self.dataset.utt2domain_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                else:
                    chunk_counter = 0
                    offset = 0
                    while offset + self.chunk_size <= wav_size:
                        start_position = offset
                        end_position = start_position + self.chunk_size
                        offset += self.chunk_size - self.overlap_size
                        # assert self.chunk_size - self.overlap_size == 43200

                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {start_position}-{end_position} in {utt}")
                        else:
                            if single_wav:
                                start = start_position
                                end = end_position
                            else:
                                for i, item in enumerate(cumsum_size):
                                    if start_position - item < 0:
                                        start = "{}_{}".format(
                                            i - 1, start_position - cumsum_size[i - 1]
                                        )
                                        break
                                for i, item in enumerate(cumsum_size):
                                    if end_position - item <= 0:
                                        end = "{}_{}".format(
                                            i - 1, end_position - cumsum_size[i - 1]
                                        )
                                        break
                            chunk = [
                                utt + "-" + str(chunk_counter),
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                                self.dataset.utt2domain_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                            chunk_counter += 1

                    if offset + self.drop_last_size < wav_size:
                        start_position = wav_size - self.chunk_size
                        end_position = wav_size

                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {start_position}-{end_position} in {utt}")
                            if chunk_counter == 0:
                                self.skipped_utts.append(utt)
                        else:
                            if single_wav:
                                start = start_position
                                end = end_position
                            else:
                                for i, item in enumerate(cumsum_size):
                                    if start_position - item < 0:
                                        start = "{}_{}".format(
                                            i - 1, start_position - cumsum_size[i - 1]
                                        )
                                        break
                                for i, item in enumerate(cumsum_size):
                                    if end_position - item <= 0:
                                        end = "{}_{}".format(
                                            i - 1, end_position - cumsum_size[i - 1]
                                        )
                                        break
                            chunk = [
                                utt + "-" + str(chunk_counter),
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                                self.dataset.utt2domain_int[utt],
                            ]
                            chunk_samples.append(chunk)
                            self.spk2chunks.setdefault(spk, []).append(chunk)
                    else:
                        if chunk_counter == 0:
                            self.skipped_utts.append(utt)

        elif self.chunk_type == "random_segment":
            for utt in track(self.dataset.utt2spk.keys()):
                wav_path = self.dataset.wav_scp[utt]
                if len(wav_path) == 1:
                    single_wav = True
                    wav_path = wav_path[0]
                    signal, fs, wav_size = load_wavs(wav_path, return_size=True)
                    every_size = [wav_size]
                else:
                    # Format:
                    # utt1 sox audio1.wav audio2.wav -t wav - |
                    single_wav = False
                    assert ".wav" not in wav_path[0]
                    wav_path.pop(0)
                    for _ in range(1, 5, 1):
                        assert ".wav" not in wav_path[-1]
                        wav_path.pop(-1)
                    wav_path = " ".join(wav_path)
                    signal, fs, every_size = load_wavs(wav_path, return_size=True)
                    wav_size = sum(every_size)
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)
                signal = signal.squeeze(0)
                wav_duration = wav_size / self.samplerate

                if wav_size < self.chunk_size:
                    if wav_size < self.min_chunk_size:
                        logger.warn(
                            "The sample num {0} of {1} is less than the minimum {2}, "
                            "so skip it.".format(wav_size, utt, self.min_chunk_size)
                        )
                        self.skipped_utts.append(utt)
                    else:
                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {utt}")
                            self.skipped_utts.append(utt)
                        else:
                            if single_wav:
                                start = 0
                                end = wav_size
                            else:
                                start = "0_0"
                                end = "{}_{}".format(len(every_size) - 1, every_size[-1])

                            chunk_samples.append(
                                [
                                    utt + "-0",
                                    wav_path,
                                    wav_duration,
                                    " ".join([str(x) for x in every_size]),
                                    start,
                                    end,
                                    self.dataset.utt2spk_int[utt],
                                ]
                            )
                else:
                    start_position = 0
                    end_position = wav_size

                    #  Avoid chunks with very small energy
                    mean_sig = torch.mean(np.abs(signal))
                    if mean_sig < self.amp_th:
                        logger.info(f"Skip in {utt}")
                        self.skipped_utts.append(utt)
                    else:
                        if single_wav:
                            start = 0
                            end = wav_size
                        else:
                            start = "0_0"
                            end = "{}_{}".format(len(every_size) - 1, every_size[-1])
                        chunk_samples.append(
                            [
                                utt,
                                wav_path,
                                wav_duration,
                                " ".join([str(x) for x in every_size]),
                                start,
                                end,
                                self.dataset.utt2spk_int[utt],
                            ]
                        )

        # every_utt for validation set
        elif self.chunk_type == "every_utt":
            chunk_selected = []
            for utt in track(self.dataset.utt2spk.keys()):
                wav_path = self.dataset.wav_scp[utt]
                if len(wav_path) == 1:
                    single_wav = True
                    wav_path = wav_path[0]
                    signal, fs, wav_size = load_wavs(wav_path, return_size=True)
                    every_size = [wav_size]
                else:
                    # Format:
                    # utt1 sox audio1.wav audio2.wav -t wav - |
                    single_wav = False
                    assert ".wav" not in wav_path[0]
                    wav_path.pop(0)
                    for _ in range(1, 5, 1):
                        assert ".wav" not in wav_path[-1]
                        wav_path.pop(-1)
                    wav_path = " ".join(wav_path)
                    signal, fs, every_size = load_wavs(wav_path, return_size=True)
                    wav_size = sum(every_size)
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)
                signal = signal.squeeze(0)
                wav_duration = wav_size / self.samplerate

                if wav_size < self.chunk_size:
                    logger.warning(
                        "The sample num {0} of {1} is less than the minimum {2}, "
                        "so skip it.".format(wav_size, utt, self.min_chunk_size)
                    )
                else:
                    selected_chunks = 0
                    skipped_chunks = 0
                    while selected_chunks < self.chunk_num_selection:
                        start_position = np.random.randint(
                            0, int(wav_duration * self.samplerate) - self.chunk_size + 1
                        )
                        end_position = start_position + self.chunk_size

                        #  Avoid chunks with very small energy
                        mean_sig = torch.mean(np.abs(signal[start_position:end_position]))
                        if mean_sig < self.amp_th:
                            logger.info(f"Skip {start_position}-{end_position} in {utt}")
                            skipped_chunks += 1
                        else:
                            if single_wav:
                                start = start_position
                                end = end_position
                            else:
                                for i, item in enumerate(cumsum_size):
                                    if start_position - item < 0:
                                        start = "{}_{}".format(
                                            i - 1, start_position - cumsum_size[i - 1]
                                        )
                                        break
                                for i, item in enumerate(cumsum_size):
                                    if end_position - item <= 0:
                                        end = "{}_{}".format(
                                            i - 1, end_position - cumsum_size[i - 1]
                                        )
                                        break
                            chunk_selected.append(
                                "{0} {1} {2} {3} {4} {5} {6}".format(
                                    utt + "-" + str(selected_chunks),
                                    wav_path,
                                    wav_duration,
                                    " ".join([str(x) for x in every_size]),
                                    start,
                                    end,
                                    self.dataset.utt2spk_int[utt],
                                )
                            )
                            selected_chunks += 1

                        if skipped_chunks == 20:
                            logger.warning(
                                f"{utt} contains too many silent frames to select chunks"
                            )
                            break

            for chunk in chunk_selected:
                chunk_samples.append(chunk.split())

        # 自己通过valid_data指定的验证集（测试集），不分割chunk
        elif self.chunk_type == "full_length":
            for i, utt in enumerate(self.dataset.utt2spk.keys()):
                wav_path = self.dataset.wav_scp[utt]

                if len(wav_path) == 1:
                    wav_path = wav_path[0]
                else:
                    # Format:
                    # utt1 sox audio1.wav audio2.wav -t wav - |
                    assert ".wav" not in wav_path[0]
                    wav_path.pop(0)
                    for _ in range(1, 5, 1):
                        assert ".wav" not in wav_path[-1]
                        wav_path.pop(-1)
                    wav_path = " ".join(wav_path)

                chunk_samples.append([utt, wav_path, self.dataset.utt2spk_int[utt]])

        else:
            raise TypeError("Do not support chunk type {0}.".format(self.chunk_type))

        return chunk_samples

    def save(self, save_path: str, force=True):
        if os.path.exists(save_path) and not force:
            raise ValueError("The path {0} is exist. Please rm it by yourself.".format(save_path))

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Some spks include too few utts, resulting in no utts left after removing short utts.
        ori_labels = [x[-1] for x in self.chunk_samples]
        tgt_chunk_samples = []
        tgt_labels = {}
        label = 0
        for chunk in self.chunk_samples:
            if chunk[-1] not in tgt_labels:
                tgt_labels[chunk[-1]] = label
                label += 1
            tgt_chunk_samples.append(chunk[:-1] + [tgt_labels[chunk[-1]]])
        self.chunk_samples = tgt_chunk_samples

        data_frame = pd.DataFrame(self.chunk_samples, columns=self.head)
        data_frame.to_csv(save_path, sep=",", header=True, index=False)
