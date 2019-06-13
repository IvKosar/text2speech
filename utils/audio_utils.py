import librosa
import numpy as np


def save_wav(wav, path, sample_rate):
    # wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # sf.write(path, wav, sample_rate)
    librosa.output.write_wav(path, wav, sample_rate)

def load_wav(filename, sample_rate):
    # Load an audio file as a floating point time series
    audio = librosa.core.load(path=filename, sr=sample_rate)
    return np.asarray(a=audio[0], dtype=np.float32)


def amplitude_to_dbs(amplitude):
    return 20 * np.log10(np.maximum(1e-5, amplitude))


def dbs_to_amplitude(dbs):
    return np.power(10.0, dbs * 0.05)


def normalize_spectrogram(dbs, min_dbs):
    return np.clip(a=((dbs - min_dbs) / -min_dbs), a_min=0, a_max=1)


def denormalize_spectrogram(dbs, min_dbs):
    return (np.clip(a=dbs, a_min=0, a_max=1) * -min_dbs) + min_dbs


def griffin_lim(spectrogram, griffin_lim_iters, n_fft, hop_length, win_length):
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    spectrogram_complex = np.abs(spectrogram).astype(np.complex)
    # Inverse short-time Fourier transform (ISTFT)
    y = librosa.core.istft(stft_matrix=(spectrogram_complex * angles), hop_length=hop_length, win_length=win_length,
                           window='hann', center=True)
    for i in range(griffin_lim_iters):
        angles = np.exp(
            1j * np.angle(librosa.core.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.core.istft(stft_matrix=(spectrogram_complex * angles), hop_length=hop_length, win_length=win_length,
                               window='hann', center=True)
    return y


def find_endpoint(wav, sample_rate, threshold_db=-40, min_silence_sec=1.0):
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < dbs_to_amplitude(dbs=threshold_db):
            return x + hop_length
    return len(wav)
