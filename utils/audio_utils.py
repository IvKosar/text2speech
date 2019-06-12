import numpy as np
import librosa


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.float), sample_rate)


def load_wav(filename, sample_rate):
    audio = librosa.core.load(filename, sr=sample_rate)
    return np.asarray(audio[0], dtype=np.float32)


def amplitude_to_dbs(amplitude):
    return 20 * np.log10(np.maximum(1e-5, amplitude))


def dbs_to_amplitude(dbs):
    return np.power(10.0, dbs * 0.05)


def normalize_spectrogram(dbs, min_dbs):
    return np.clip((dbs - min_dbs) / -min_dbs, 0, 1)


def denormalize_spectrogram(dbs, min_dbs):
    return (np.clip(dbs, 0, 1) * -min_dbs) + min_dbs


def griffin_lim(spectrogram, griffin_lim_iters, n_fft, hop_length, win_length):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    spectrogram_complex = np.abs(spectrogram).astype(np.complex)
    y = librosa.istft(spectrogram_complex * angles, hop_length=hop_length, win_length=win_length)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.istft(spectrogram_complex * angles, hop_length=hop_length, win_length=win_length)
    return y


def find_endpoint(wav, sample_rate, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = dbs_to_amplitude(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)
