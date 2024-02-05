import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from stft_module import WindowType, stft
from melspectogram_module import melspectrogram


def test_1_stft():
    y, sr = librosa.load(librosa.ex('trumpet'))

    num_fft = 2048
    hop_length = 512
    stft_res = stft(y, num_fft=num_fft, hop_length=hop_length, window=WindowType.HAMMING)
    magnitude_spectrogram = np.abs(stft_res)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude Spectrogram')
    plt.show()


def test_mel_spectogram():
    y, sr = librosa.load(librosa.ex('trumpet'))


    mel_spec = melspectrogram(y, sampling_rate=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()


if __name__=='__main__':
    test_1_stft()
    test_mel_spectogram()