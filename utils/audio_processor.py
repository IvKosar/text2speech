from scipy import signal
from librosa import stft
from utils.audio_utils import *


class AudioProcessor:
    def __init__(self, sample_rate, preemphasis, frequency, frame_length, frame_shift, min_dbs):
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.frequency = frequency
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.min_dbs = min_dbs

        self.n_fft = (self.frequency - 1) * 2
        self.hop_length = int(frame_shift / 1000 * sample_rate)
        self.win_length = int(frame_length / 1000 * sample_rate)

    def generate_spectrogram(self, waveform):
        filtered_wav = signal.lfilter([1, -self.preemphasis], [1], waveform)
        stft_wav = stft(y=filtered_wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        spectrogram = amplitude_to_dbs(stft_wav)
        return normalize_spectrogram(spectrogram, self.min_dbs)

    def generate_mel_spectrogram(self):
        pass

