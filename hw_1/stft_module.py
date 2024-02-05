import numpy as np
from enum import Enum


class WindowType(Enum):
    HANN = 'hann'
    HAMMING = 'hamming'
    RECTANGULAR = 'rectangular'


def stft(y: np.ndarray, num_fft: int = 2048, hop_length=None, win_length=None, window: WindowType = WindowType.HANN, center: bool = True) -> np.ndarray:
    """
    :param y: Input signal
    :param num_fft: Number of FFT points
    :param hop_length: Number of samples between adjacent frames
    :param win_length: Size of the window function
    :param window: Window function
    :param center: If True, pads the input signal so that the frame is centered at y[t]

    :return: 2D array representing the STFT of the input signal.
    """
    # default values
    hop_length = hop_length or num_fft // 4 # default is 25% of the window size
    win_length = win_length or num_fft

    if center:
        # pad signal
        y = np.pad(y, int(num_fft // 2), mode='reflect')

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

    # number of frames with overlap
    n_frames = 1 + (len(y) - num_fft) // hop_length

    # initialise array
    stft_matrix = np.empty((num_fft // 2 + 1, n_frames), dtype=np.complex128)

    # window func + FFT to each frame
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + num_fft
        frame = y[start_idx:end_idx] * window  # window function
        stft_matrix[:, i] = np.fft.rfft(frame, n=num_fft)  # FFT
    return stft_matrix


