import numpy as np
from scipy import signal
from librosa import stft, filters
from utils.audio_utils import amplitude_to_dbs, dbs_to_amplitude, normalize_spectrogram, denormalize_spectrogram,\
    griffin_lim


class AudioProcessor:
    def __init__(self, sample_rate, preemphasis, frequency, frame_length, frame_shift, min_dbs, ref_dbs, mels_size,
                 griff_lim_iters, power):
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.frequency = frequency
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.min_dbs = min_dbs
        self.ref_dbs = ref_dbs
        self.mels_size = mels_size
        self.griff_lim_iters = griff_lim_iters
        self.power = power

        self.n_fft = (self.frequency - 1) * 2
        self.hop_length = int(frame_shift / 1000 * sample_rate)
        self.win_length = int(frame_length / 1000 * sample_rate)

        self.mel_basis = filters.mel(self.sample_rate, self.n_fft, n_mels=self.mels_size)

    def wav_to_linear_spectrogram(self, waveform):
        filtered_wav = signal.lfilter([1, -self.preemphasis], [1], waveform)
        linear_spectrogram = stft(y=filtered_wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        spectrogram_dbs = amplitude_to_dbs(np.abs(linear_spectrogram)) - self.ref_dbs
        return normalize_spectrogram(spectrogram_dbs, self.min_dbs)

    def wav_to_mel_spectrogram(self, waveform):
        filtered_wav = signal.lfilter([1, -self.preemphasis], [1], waveform)
        linear_spectrogram = stft(y=filtered_wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        mel_spectorgram = np.dot(self.mel_basis, np.abs(linear_spectrogram))
        mel_dbs = amplitude_to_dbs(mel_spectorgram) - self.ref_dbs
        return normalize_spectrogram(mel_dbs, self.min_dbs)

    def spectrogram_to_wav(self, spectrogram):
        denormalized = denormalize_spectrogram(spectrogram, self.min_dbs)
        linear = dbs_to_amplitude(denormalized + self.ref_dbs)
        waveform = griffin_lim(linear ** self.power, self.griff_lim_iters, self.n_fft, self.hop_length, self.win_length)
        return signal.lfilter([1], [1 - self.preemphasis], waveform)


