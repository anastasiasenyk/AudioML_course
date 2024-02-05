import numpy as np
from stft_module import WindowType


def melspectrogram(y, sampling_rate=22050, num_fft=2048, hop_length=None, power=2.0, num_mels=128, fmin=0.0, fmax=None, win_length=None, window: WindowType = WindowType.HANN):
    """
    Creating mel spectrogram of an audio signal

    :param y: Input signal
    :param sampling_rate: Sampling rate
    :param num_fft: Number of FFT points
    :param hop_length: Number of samples between adjacent frames
    :param power: Exponent for the magnitude spectrogram
    :param num_mels:  The number of mel bands needed to be generated
    :param fmin: The lowest frequency in Hz
    :param fmax: The highest frequency in Hz
    :param win_length: Size of the window function
    :param window: Window function
    :return:
    """
    hop_length = hop_length or num_fft // 4  # default is 25% of the window size
    win_length = win_length or num_fft
    fmax = fmax or sampling_rate / 2

    if isinstance(window, WindowType):
        window_type = window.value
    else:
        raise ValueError("Unsupported window type. Choose only from WindowType enum")

    if window_type == WindowType.HANN.value:
        window = np.hanning(win_length)  # Hann window
    elif window_type == WindowType.HAMMING.value:
        window = np.hamming(win_length)  # Hamming window
    elif window_type == WindowType.RECTANGULAR.value:  # Rectangular window
        window = np.ones(win_length)

    # magnitude spectrogram
    S = np.abs(
        np.array([np.fft.rfft(window * y[i : i + num_fft], n=num_fft)
                  for i in range(0, len(y) - num_fft + 1, hop_length)])
    )

    mel_scale = np.linspace(0, 2595 * np.log10(1 + fmax / 700), num_mels + 2)
    # converting to Hz
    mel_scale = 700 * (10 ** (mel_scale / 2595) - 1)

    # find corresponding FFT indices
    fft_freqs = np.linspace(0, sampling_rate / 2, num_fft // 2 + 1)
    mel_fft_indices = np.searchsorted(fft_freqs, mel_scale)

    # filter
    mel_filters = np.zeros((num_mels, num_fft // 2 + 1))
    for i in range(1, num_mels + 1):
        left_idx = mel_fft_indices[i - 1]
        center_idx = mel_fft_indices[i]
        right_idx= mel_fft_indices[i + 1]

        mel_filters[i - 1, left_idx:center_idx] = ((fft_freqs[left_idx:center_idx] - fft_freqs[left_idx]) / (fft_freqs[center_idx] - fft_freqs[left_idx]))
        mel_filters[i - 1, center_idx:right_idx] = ((fft_freqs[right_idx:center_idx:-1] - fft_freqs[right_idx]) / (fft_freqs[center_idx] - fft_freqs[right_idx]))
    return np.power(np.dot(S, mel_filters.T), power).T
