import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from stft_module import WindowType, stft


def test_1_stft():
    y, sr = librosa.load(librosa.ex('trumpet'))

    n_fft = 2048
    hop_length = 512
    stft_res = stft(y, num_fft=n_fft, hop_length=hop_length, window=WindowType.HAMMING)
    magnitude_spectrogram = np.abs(stft_res)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude Spectrogram')
    plt.show()


if __name__=='__main__':
    test_1_stft()