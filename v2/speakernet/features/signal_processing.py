"""
Low level signal processing utilities

https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/signal_processing.py

"""
import math
import torch
import numpy as np
from packaging import version


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            # out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
            out = torch.sqrt(torch.mean(waveforms ** 2))
        else:
            # wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            # out = wav_sum / lengths
            raise ValueError
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    """This function normalizes a signal to unitary average or peak amplitude.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether one wants to normalize with respect to "avg" or "peak"
        amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
        is not prevented and can occur.
    eps : float
        A small number to add to the denominator to prevent NaN.

    Returns
    -------
    waveforms : tensor
        Normalized level waveform.
    """

    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
    """This functions performs signal rescaling to a target level.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    target_lvl : float
        Target lvl in dB or linear scale.
    amp_type : str
        Whether one wants to rescale with respect to "avg" or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        whether target_lvl belongs to linear or dB scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    waveforms : tensor
        Rescaled waveforms.
    """

    assert amp_type in ["peak", "avg"]
    assert scale in ["linear", "dB"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    waveforms = normalize(waveforms, lengths, amp_type)

    if scale == "linear":
        out = target_lvl * waveforms
    elif scale == "dB":
        out = dB_to_amplitude(target_lvl) * waveforms

    else:
        raise NotImplementedError("Invalid scale, choose between dB and linear")

    if batch_added:
        out = out.squeeze(0)

    return out


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution.
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by the number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0).unsqueeze(2)
    >>> kernel = torch.rand(1, 10, 1)
    >>> signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose(2, 1)
    kernel = kernel.transpose(2, 1)

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(
            input=waveform, pad=padding, mode=pad_type,
        )

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(
            kernel.size(0), kernel.size(1), zero_length, device=kernel.device
        )
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        # Multiply in frequency domain to convolve in time domain
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            import torch.fft as fft

            result = fft.rfft(waveform) * fft.rfft(kernel)
            convolved = fft.irfft(result, n=waveform.size(-1))
        else:
            f_signal = torch.rfft(waveform, 1)
            f_kernel = torch.rfft(kernel, 1)
            sig_real, sig_imag = f_signal.unbind(-1)
            ker_real, ker_imag = f_kernel.unbind(-1)
            f_result = torch.stack(
                [
                    sig_real * ker_real - sig_imag * ker_imag,
                    sig_real * ker_imag + sig_imag * ker_real,
                ],
                dim=-1,
            )
            convolved = torch.irfft(
                f_result, 1, signal_sizes=[waveform.size(-1)]
            )

    # Use the implementation given by torch, which should be efficient on GPU
    else:
        convolved = torch.nn.functional.conv1d(
            input=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(2, 1)


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    General function to contaminate a given signal with reverberation given a
    Room Impulse Response (RIR).
    It performs convolution between RIR and signal, but without changing
    the original amplitude of the signal.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    rir_waveform : tensor
        RIR tensor, shape should be [time, channels].
    rescale_amp : str
        Whether reverberated signal is rescaled (None) and with respect either
        to original signal "peak" amplitude or "avg" average amplitude.
        Choose between [None, "avg", "peak"].

    Returns
    -------
    waveforms: tensor
        Reverberated signal.

    """

    orig_shape = waveforms.shape

    if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0).unsqueeze(-1)
    elif len(waveforms.shape) == 2:
        waveforms = waveforms.unsqueeze(-1)

    if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
        rir_waveform = rir_waveform.unsqueeze(0).unsqueeze(-1)
    elif len(rir_waveform.shape) == 2:
        rir_waveform = rir_waveform.unsqueeze(-1)

    # Compute the average amplitude of the clean
    orig_amplitude = compute_amplitude(
        waveforms, None, rescale_amp
    )

    # Compute index of the direct signal, so we can preserve alignment
    value_max, direct_index = rir_waveform.abs().max(axis=1, keepdim=True)

    # Making sure the max is always positive (if not, flip)
    # mask = torch.logical_and(rir_waveform == value_max,  rir_waveform < 0)
    # rir_waveform[mask] = -rir_waveform[mask]

    # Use FFT to compute convolution, because of long reverberation filter
    waveforms = convolve1d(
        waveform=waveforms,
        kernel=rir_waveform,
        use_fft=True,
        rotation_index=direct_index,
    )

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(
        waveforms, None, orig_amplitude, rescale_amp
    )

    if len(orig_shape) == 1:
        waveforms = waveforms.squeeze(0).squeeze(-1)
    if len(orig_shape) == 2:
        waveforms = waveforms.squeeze(-1)

    return waveforms


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)
