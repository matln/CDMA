"""
Copyright 2022 Jianchen Li
"""

import sys
import copy
import torch
import random
import logging
import torchaudio
import pandas as pd
from rich import print
import torch.nn.functional as F

from speakernet.features.signal_processing import reverberate
from speakernet.utils.utils import load_data_csv, is_main_training

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NoiseData:
    def __init__(self, csv_file, padding=0):
        head = pd.read_csv(csv_file, sep=",", nrows=0).columns
        assert "ID" in head
        assert "wav" in head
        assert "duration" in head
        assert padding in [0, "wrap"]
        self.padding = padding

        self.data = load_data_csv(csv_file, id_field="ID", delimiter=",")
        self.data_ids = list(self.data.keys())

    def load(self, id, max_len=None):
        data_point = self.data[id]
        audio_len = int(data_point["tot_frame"])
        if max_len is not None:
            if audio_len >= max_len:
                start = random.randint(0, audio_len - max_len)
                waveforms, fs = torchaudio.load(
                    data_point["wav"], num_frames=max_len, frame_offset=start
                )
                valid_len = max_len
            else:
                waveforms, fs = torchaudio.load(data_point["wav"])
                if self.padding == 0:
                    padding = (0, max_len - audio_len)
                    waveforms = torch.nn.functional.pad(waveforms, padding)
                    valid_len = audio_len
                else:
                    pad_warp_num = max_len // audio_len
                    pad_size = max_len % audio_len
                    cat_list = [waveforms for _ in range(pad_warp_num)]
                    if pad_size != 0:
                        cat_list.append(waveforms[:, 0:pad_size])
                    waveforms = torch.cat(cat_list, dim=1)
                    valid_len = max_len
        else:
            waveforms, fs = torchaudio.load(data_point["wav"])
            valid_len = audio_len

        return waveforms, valid_len


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : str
        int 0 or str "warp"
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    scale: bool
        Scale clean signal to keep the amplitude.

    """

    def __init__(self, csv_file=None, snr_low=0, snr_high=0, pad_noise=0, mix_prob=1.0, scale=True):
        super().__init__()

        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob
        self.scale = scale
        assert csv_file is not None
        self.noises_data = NoiseData(csv_file, padding=pad_noise)
        self.noises = copy.deepcopy(self.noises_data.data_ids)

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)

        # Copy clean waveform
        clean_waveform = waveforms.clone()

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) >= self.mix_prob:
            return clean_waveform

        # Root mean square amplitude
        clean_amp = torch.sqrt(torch.mean(clean_waveform ** 2))

        # Pick an SNR and use it to compute the mixture amplitude factors
        snr_db = random.uniform(self.snr_low, self.snr_high)
        snr_amp = 10 ** (snr_db / 20)

        # Sample a noise audio and load it
        choiced_noise = random.choice(self.noises)
        # No replacement sampling
        self.noises.remove(choiced_noise)
        if len(self.noises) == 0:
            self.noises = copy.deepcopy(self.noises_data.data_ids)
        noise_waveform, _ = self.noises_data.load(choiced_noise, max_len=waveforms.size(1))
        # Root mean square amplitude
        noise_amp = torch.sqrt(torch.mean(noise_waveform ** 2))

        if self.scale:
            # 为什么分母要加1呢?目的是保证加噪声后waveform的平均幅度不变，所以下面
            # 也对clean signal 进行缩放. 这样也相当于对说话人语音进行volume的增强.
            # snr_amp 是干净信号与噪声信号的幅度比，所以
            # noise_scale_factor = noise / (clean + noise)
            noise_scale_factor = 1 / (snr_amp + 1)
            new_noise_amp = noise_scale_factor * clean_amp

            # Scale clean signal appropriately
            clean_waveform *= 1 - noise_scale_factor

            # Scale noise signal
            noise_waveform *= new_noise_amp / (noise_amp + 1e-14)

            noisy_waveform = clean_waveform + noise_waveform
        else:
            # clean_amp / noise_amp / snr_amp
            # = new_noise_amp / noise_amp
            noise_scale_factor = clean_amp / noise_amp / snr_amp

            noisy_waveform = clean_waveform + noise_scale_factor * noise_waveform

        return noisy_waveform


class AddReverb(torch.nn.Module):
    """This class convolves an audio signal with an impulse response.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        impulse response files.
    reverb_prob : float
        The chance that the audio signal will be reverbed.
        By default, every batch is reverbed.
    rir_scale_factor: float
        It compresses or dilates the given impulse response.
        If 0 < scale_factor < 1, the impulse response is compressed
        (less reverb), while if scale_factor > 1 it is dilated
        (more reverb).

    """

    def __init__(
        self, csv_file, reverb_prob=1.0, rir_scale_factor=1.0,
    ):
        super().__init__()
        self.reverb_prob = reverb_prob
        self.rir_scale_factor = rir_scale_factor

        # RIR waveforms
        assert csv_file is not None
        self.rir_data = NoiseData(csv_file)
        self.noises = copy.deepcopy(self.rir_data.data_ids)

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) >= self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Load and prepare RIR
        choiced_rir = random.choice(self.noises)
        rir_waveform, _ = self.rir_data.load(choiced_rir)

        # No replacement sampling
        self.noises.remove(choiced_rir)
        if len(self.noises) == 0:
            self.noises = copy.deepcopy(self.rir_data.data_ids)

        # Make sure RIR has correct channels
        assert len(rir_waveform.shape) == 2
        rir_waveform = rir_waveform.unsqueeze(-1)

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose(1, -1),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_waveform = rir_waveform.transpose(1, -1)

        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform


class AddBabble(torch.nn.Module):
    """Add babble noise.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise :
        int 0 or str "warp"
    mix_prob : float
        The probability that the batch of signals will be
        mixed with babble noise. By default, every signal is mixed.
    speaker_count_low : int
        The low end of signals number to mix with the original signal.
    speaker_count_high : int
        The high end of signals number to mix with the original signal.
    scale: bool
        Scale clean signal to keep the amplitude.
    """

    def __init__(
        self,
        csv_file=None,
        snr_low=0,
        snr_high=0,
        pad_noise=0,
        mix_prob=1.0,
        speaker_count_low=4,
        speaker_count_high=4,
        scale=True,
    ):
        super().__init__()

        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob
        self.scale = scale
        self.speaker_count_low = speaker_count_low
        self.speaker_count_high = speaker_count_high

        assert csv_file is not None
        self.noises_data = NoiseData(csv_file, padding=pad_noise)
        self.noises = copy.deepcopy(self.noises_data.data_ids)

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)

        # Copy clean waveform
        clean_waveform = waveforms.clone()

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) >= self.mix_prob:
            return clean_waveform

        # Root mean square amplitude, sum instead of mean
        clean_amp = torch.sqrt(torch.mean(clean_waveform ** 2))

        # Pick an SNR and use it to compute the mixture amplitude factors
        snr_db = random.uniform(self.snr_low, self.snr_high)
        snr_amp = 10 ** (snr_db / 20)

        noise_selected = self.sample_noise()
        noise_waveforms = []
        for noise in noise_selected:
            _noise_waveform, _ = self.noises_data.load(noise, max_len=waveforms.size(1))
            noise_waveforms.append(_noise_waveform)
        babble_waveform = torch.sum(torch.cat(noise_waveforms, dim=0), dim=0, keepdims=True)
        # Root mean square amplitude, sum instead of mean
        babble_amp = torch.sqrt(torch.mean(babble_waveform ** 2))

        if self.scale:
            # 为什么分母要加1呢?目的是保证加噪声后waveform的平均幅度不变，所以下面
            # 也对clean signal 进行缩放. 这样也相当于对说话人语音进行volume的增强.
            # snr_amp 是干净信号与噪声信号的幅度比，所以
            # babble_scale_factor = noise / (clean + noise)
            babble_scale_factor = 1 / (snr_amp + 1)
            new_babble_amp = babble_scale_factor * clean_amp

            # Scale clean signal appropriately
            clean_waveform *= 1 - babble_scale_factor

            # Scale babble signal
            babble_waveform *= new_babble_amp / (babble_amp + 1e-14)

            babbled_waveform = clean_waveform + babble_waveform
        else:
            babble_scale_factor = clean_amp / babble_amp / snr_amp
            babbled_waveform = clean_waveform + babble_scale_factor * babble_waveform

        return babbled_waveform

    def sample_noise(self):
        speaker_count = random.randint(self.speaker_count_low, self.speaker_count_high)
        # Simulate babble noise by mixing the signals in a batch.
        if len(self.noises) > speaker_count:
            noise_selected = random.sample(self.noises, speaker_count)
            # No replacement sampling
            for noise in noise_selected:
                self.noises.remove(noise)
        else:
            noise_selected = self.noises
            self.noises = copy.deepcopy(self.noises_data.data_ids)
            pad_num = speaker_count - len(noise_selected)
            if pad_num > 0:
                for noise in noise_selected:
                    self.noises.remove(noise)
                _noise_selected = random.sample(self.noises, pad_num)
                self.noises = self.noises + noise_selected
                noise_selected.extend(_noise_selected)
                for noise in _noise_selected:
                    self.noises.remove(noise)
        return noise_selected


class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal with modifying pitch.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    sample_rate : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to.

    """

    def __init__(self, sample_rate=16000, speeds=[0.9, 1.0, 1.1], perturb_prob=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds
        self.perturb_prob = perturb_prob
        self.speed2label_multipler = {1.0: 0, 0.9: 1, 1.1: 2}

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[channels, time]`.

        Returns
        -------
        Tensor of shape `[channels, time]`.
        """

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone(), 0
        speed_factor = random.choice(self.speeds)
        if speed_factor == 1.0:  # no change
            return waveform.clone(), 0

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]

        perturbed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, sox_effects
        )

        return perturbed_waveform, self.speed2label_multipler[speed_factor]


class TempoPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal without modifying pitch.

    This technique is outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    sample_rate : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to.

    """

    def __init__(self, sample_rate=16000, speeds=[0.9, 1.0, 1.1], perturb_prob=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds
        self.perturb_prob = perturb_prob

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[channels, time]`.

        Returns
        -------
        Tensor of shape `[channels, time]`.
        """

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone()

        speed_factor = random.choice(self.speeds)
        if speed_factor == 1.0:  # no change
            return waveform.clone()

        # change speed and resample to original rate:
        sox_effects = [
            ["tempo", str(speed_factor)],
        ]

        perturbed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, sox_effects
        )

        return perturbed_waveform


class EnvCorrupt(torch.nn.Module):
    """Speech augment for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble noise added.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data, if None, means white noise.
    babble_csv : str
        A prepared csv file for loading babble data, if None, means simulated babble noise.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count_low : int
        Lowest number of speakers to use for babble. Must be less than batch size if not use babble_csv but simulate babble noise by the batch data itself.
    babble_speaker_count_high : int
        Highest number of speakers to use for babble. Must be less than batch size if not use babble_csv but simulate babble noise by the batch data itself.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).
    scale: bool
        Scale clean signal to keep the amplitude.

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = EnvCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10))
    """

    def __init__(
        self,
        reverb_prob=0.0,
        noise_prob=0.0,
        babble_prob=0.0,
        reverb_csv=None,
        noise_csv=None,
        babble_csv=None,
        babble_speaker_count_low=0,
        babble_speaker_count_high=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
        rir_scale_factor=1.0,
        pad_noise=0,
        scale=True,
        **ops,
    ):
        super().__init__()

        # Initialize corrupters
        if reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob, csv_file=reverb_csv, rir_scale_factor=rir_scale_factor,
            )

        if babble_speaker_count_low > 0 and babble_prob > 0.0:
            assert babble_speaker_count_high >= babble_speaker_count_low
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                csv_file=babble_csv,
                speaker_count_low=babble_speaker_count_low,
                speaker_count_high=babble_speaker_count_high,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
                pad_noise=pad_noise,
                scale=scale,
            )

        if noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
                pad_noise=pad_noise,
                scale=scale,
            )

    def forward(self, waveforms):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms)

                except Exception:
                    print("add_reverb failed")
                    exit(1)
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms)

        return waveforms


class SpeechAug(torch.nn.Module):
    """This class implement three types of speech augment (chain,random,concat).

    Arguments
    ---------
    aug_classes: list
        A list of aug_classes which contains its config information and sequence.
    mod: str
        Speech augment pipline type, random means random select one augment from the list (NOTE: contains a clean wav automaticaly),
                                     concat means concat all the augment classes,
                                     chain means sequencely apply augment subject to the aug_classes list.


    Example
    -------
    aug_classes=[{'aug_name': 'augment_speed', 'aug_type': 'Time',
                  'random_mod_weight': 1, 'perturb_prob': 1.0,
                  'drop_freq_prob': 0.0, 'drop_chunk_prob': 0.0,
                  'sample_rate': 16000, 'speeds': [95, 100, 105],
                  'keep_shape': True},
                 {'aug_name': 'augment_wavedrop', 'aug_type': 'Time',
                  'random_mod_weight': 1, 'perturb_prob': 0.0,
                  'drop_freq_prob': 1.0, 'drop_chunk_prob': 1.0,
                  'sample_rate': 16000, 'speeds': [100]}]
    speech_aug = SpeechAug(aug_classes)
    signal = torch.randn(52173).unsqueeze(0)
    signal, lens = speech_aug(signal,torch.ones(1))
    """

    def __init__(self, aug_classes=[], mod="random"):
        super().__init__()
        assert mod in ["random", "concat", "chain"]
        self.mod = mod
        self.augments, self.augment_names, self.random_weights = self.load_aug_class(
            aug_classes, self.mod
        )

        if is_main_training():
            self.get_augment()

    def load_aug_class(self, aug_classes, mod):
        augments = []
        augment_names = []
        # define a weight of clean wav type
        random_weights = [1] if mod == "random" else []
        for aug_class in aug_classes:
            # Nested augment
            if "aug_classes" in aug_class:
                assert "mod" in aug_class
                assert aug_class["mod"] in ["random", "chain"]
                _augments, _augment_names, _random_weights = self.load_aug_class(
                    aug_class["aug_classes"], aug_class["mod"]
                )
                augments.append(_augments)
                augment_names.append(_augment_names)
                if mod == "random":
                    # Append the current weight to the returned weight list, then pop it
                    # in self._random_forward()
                    random_weight = 1.0
                    if "random_mod_weight" in aug_class:
                        random_weight = float(aug_class["random_mod_weight"])
                        del aug_class["random_mod_weight"]
                    _random_weights.append(random_weight)
                random_weights.append(_random_weights)
            else:
                assert "aug_name" in aug_class
                augment_names.append(aug_class["aug_name"])
                assert len([item for item in augment_names if type(item) != list]) == len(
                    list(set([item for item in augment_names if type(item) != list]))
                ), "There are redundant aug_names"
                del aug_class["aug_name"]

                if mod == "random":
                    random_weight = 1.0
                    if "random_mod_weight" in aug_class:
                        random_weight = float(aug_class["random_mod_weight"])
                        del aug_class["random_mod_weight"]
                    random_weights.append(random_weight)
                else:
                    aug_class.pop("random_mod_weight", None)
                    random_weights.append(None)

                assert "aug_type" in aug_class
                assert aug_class["aug_type"] in ["Env", "Speed", "Tempo"]
                aug_type = aug_class["aug_type"]
                del aug_class["aug_type"]
                if aug_type == "Env":
                    augments.append(EnvCorrupt(**aug_class))
                if aug_type == "Speed":
                    augments.append(SpeedPerturb(**aug_class))
                if aug_type == "Tempo":
                    augments.append(TempoPerturb(**aug_class))

        return augments, augment_names, random_weights

    def forward(self, waveforms):
        if not self.augments:
            return waveforms

        if self.mod == "random":
            return self._random_forward(
                waveforms, self.augments, self.augment_names, self.random_weights
            )
        elif self.mod == "chain":
            return self._chain_forward(
                waveforms, self.augments, self.augment_names, self.random_weights
            )
        else:
            return self._concat_forward(
                waveforms, self.augments, self.augment_names, self.random_weights
            )

    def _random_forward(
        self, waveforms, augments, augment_names, random_weights, label_multiplier=0
    ):
        assert len(augments) == len(random_weights) - 1
        assert len(augments) == len(augment_names)
        # Transform the weight list to a tensor
        _random_weights = []
        for item in random_weights:
            if type(item) == list:
                _random_weights.append(item.pop(-1))
            else:
                _random_weights.append(item)
        _random_weights = torch.tensor(_random_weights, dtype=torch.float)
        aug_idx = torch.multinomial(_random_weights, 1)[0]
        augment = augments[aug_idx - 1]
        augment_name = augment_names[aug_idx - 1]
        random_weight = random_weights[aug_idx]

        if aug_idx == 0:
            return waveforms, label_multiplier
        else:
            if type(augment) == list:
                assert type(random_weight) == list
                assert type(augment_name) == list
                # Chain mod
                if random_weight[0] is None:
                    waveforms, label_multiplier = self._chain_forward(
                        waveforms, augment, augment_name, random_weight, label_multiplier,
                    )
                # Random mod
                else:
                    waveforms, label_multiplier = self._random_forward(
                        waveforms, augment, augment_name, random_weight, label_multiplier,
                    )
            else:
                if f"{augment}" == "SpeedPerturb()":
                    assert label_multiplier == 0, "SpeedPerturb can only be used once."
                    waveforms, label_multiplier = augment(waveforms)
                else:
                    waveforms = augment(waveforms)

            if torch.any((torch.isnan(waveforms))):
                raise ValueError(
                    "random: {}, type: {}, typename: {}".format(waveforms, augment, augment_name,)
                )
            return waveforms, label_multiplier

    def _concat_forward(self, waveforms, augments, augment_names, random_weights):
        # TODO
        assert len(augments) == len(random_weights)
        assert len(augments) == len(augment_names)
        wavs_aug_tot = []
        wavs_aug_tot.append(waveforms.clone())
        label_multipliers = []
        label_multipliers.append(0)
        for count, augment in enumerate(augments):
            augment_name = augment_names[count]
            random_weight = random_weights[count]
            if type(augment) == list:
                assert type(random_weight) == list
                assert type(augment_name) == list
                # Chain mod
                if random_weights[count][0] is None:
                    wavs_aug, label_multiplier = self._chain_forward(
                        waveforms, augment, augment_name, random_weight
                    )
                # Random mod
                else:
                    wavs_aug, label_multiplier = self._random_forward(
                        waveforms, augment, augment_name, random_weight
                    )
            else:
                if f"{augment}" == "SpeedPerturb()":
                    wavs_aug, label_multiplier = augment(waveforms)
                else:
                    wavs_aug = augment(waveforms)
                    label_multiplier = 0

            if torch.any((torch.isnan(wavs_aug))):
                raise ValueError(
                    "concat: {}, type: {}, typename: {}".format(
                        waveforms, augment, augment_names[count]
                    )
                )
            wavs_aug_tot.append(wavs_aug)
            label_multipliers.append(label_multiplier)
        return wavs_aug_tot, label_multipliers

    def _chain_forward(
        self, waveforms, augments, augment_names, random_weights, label_multiplier=0
    ):
        assert len(augments) == len(random_weights)
        assert len(augments) == len(augment_names)
        for count, augment in enumerate(augments):
            augment_name = augment_names[count]
            random_weight = random_weights[count]
            if type(augment) == list:
                assert type(random_weight) == list
                assert type(augment_name) == list
                # Chain mod
                if random_weight[0] is None:
                    waveforms, label_multiplier = self._chain_forward(
                        waveforms, augment, augment_name, random_weight, label_multiplier,
                    )
                # Random mod
                else:
                    waveforms, label_multiplier = self._random_forward(
                        waveforms, augment, augment_name, random_weight, label_multiplier,
                    )
            else:
                if f"{augment}" == "SpeedPerturb()":
                    assert label_multiplier == 0, "SpeedPerturb can only be used once."
                    waveforms, label_multiplier = augment(waveforms)
                else:
                    waveforms = augment(waveforms)

            if torch.any((torch.isnan(waveforms))):
                raise ValueError(
                    "chain: {}, type: {}, typename: {}".format(waveforms, augment, augment_name)
                )

        return waveforms, label_multiplier

    def get_num_concat(self):
        if self.mod == "concat":
            return len(self.augment_names) + 1
        else:
            return 1

    def get_augment(self):
        def print_aug_dict(augments, augment_names, p_count=None):
            aug_dict = {}
            for count in range(len(augment_names)):
                if type(augments[count]) == list:
                    assert type(augment_names[count]) == list
                    print_aug_dict(
                        augments[count],
                        augment_names[count],
                        f"{p_count}-{count}" if p_count is not None else count,
                    )
                else:
                    aug_dict[augment_names[count]] = augments[count]
                    print(
                        "({}) {}:  {}".format(
                            f"{p_count}-{count}" if p_count is not None else count,
                            augment_names[count],
                            augments[count],
                        )
                    )

        if self.augments:
            print("speech augment type is {}.".format(self.mod))
            print_aug_dict(self.augments, self.augment_names)
            # for i, k in enumerate(aug_dict.items()):
            #     print("({}) {}:  {}".format(i, k[0], k[1]))
        else:
            print("no speech augment applied")

    def get_aug_state(self, augment_names=None, augments=None):
        aug_state = {}
        for name, augment in zip(
            self.augment_names if augment_names is None else augment_names,
            self.augments if augments is None else augments,
        ):
            if type(augment) == list:
                _aug_state = self.get_aug_state(name, augment)
                key = ":".join(name)
                aug_state[key] = _aug_state
            else:
                if "EnvCorrupt" in f"{augment}":
                    if hasattr(augment, "add_reverb"):
                        aug_state.setdefault(name, {})["add_reverb"] = augment.add_reverb.noises
                    if hasattr(augment, "add_babble"):
                        aug_state.setdefault(name, {})["add_babble"] = augment.add_babble.noises
                    if hasattr(augment, "add_noise"):
                        aug_state.setdefault(name, {})["add_noise"] = augment.add_noise.noises
        return aug_state

    def recover_aug_state(self, aug_state, augment_names=None, augments=None):
        for name, augment in zip(
            self.augment_names if augment_names is None else augment_names,
            self.augments if augments is None else augments,
        ):
            if type(augment) == list:
                self.recover_aug_state(aug_state[":".join(name)], name, augment)
            else:
                if "EnvCorrupt" in f"{augment}":
                    for aug_type in aug_state[name]:
                        _augment = eval(f"augment.{aug_type}")
                        _augment.noises = aug_state[name][aug_type]


if __name__ == "__main__":
    import yaml
    from rich import print

    # with open(
    #     "/data/lijianchen/workspace/sre/subtools/recipe/voxceleb2/hparams/speech_aug_chain.yaml",
    #     "r",
    # ) as fin:
    #     speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
    # # print(speech_aug_conf)
    # aug = SpeechAug(**speech_aug_conf)

    # sig = torch.rand([1, 32400])
    # sig = aug(sig)
    # print(aug.get_aug_state().keys())

    add_noise = AddNoise(
        csv_file="/data/lijianchen/workspace/sre/subtools/recipe/voxceleb2/exp/aug_csv/musan_music.csv",
        snr_low=10,
        snr_high=10,
        scale=True,
    )
    add_babble = AddBabble(
        csv_file="/data/lijianchen/workspace/sre/subtools/recipe/voxceleb2/exp/aug_csv/musan_speech.csv",
        snr_low=0,
        snr_high=0,
        scale=True,
    )
    clean_sig, _ = torchaudio.load("/home/lijianchen/00001.wav")
    noisy_sig = add_babble(clean_sig)
    torchaudio.save("/home/lijianchen/babble_scale_0.wav", noisy_sig, sample_rate=16000)
