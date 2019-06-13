import numpy as np
from librosa import filters
from librosa.core import stft
from scipy import signal

from utils.audio_utils import amplitude_to_dbs, dbs_to_amplitude, normalize_spectrogram, denormalize_spectrogram, \
    griffin_lim


class AudioProcessor(object):
    def __init__(self, sample_rate, preemphasis, frequency, frame_length, frame_shift, min_dbs, ref_dbs, mels_size,
                 griff_lim_iters, power):
        self.preemphasis = preemphasis
        self.n_fft = (frequency - 1) * 2
        self.win_length = int(frame_length / 1e3 * sample_rate)
        self.hop_length = int(frame_shift / 1e3 * sample_rate)
        self.min_dbs = min_dbs
        self.ref_dbs = ref_dbs
        self.griff_lim_iters = griff_lim_iters
        self.power = power

        # Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
        self.mel_basis = filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=mels_size)

    def wav_to_linear_spectrogram(self, waveform):
        # Filter data along one-dimension with an IIR or FIR filter
        filtered_wav = signal.lfilter(b=[1, -self.preemphasis], a=[1], x=waveform, axis=-1, zi=None)
        # Short-time Fourier transform (STFT)
        linear_spectrogram = stft(y=filtered_wav, n_fft=self.n_fft, hop_length=self.hop_length,
                                  win_length=self.win_length)
        spectrogram_dbs = amplitude_to_dbs(amplitude=np.abs(linear_spectrogram)) - self.ref_dbs
        return normalize_spectrogram(dbs=spectrogram_dbs, min_dbs=self.min_dbs)

    def wav_to_mel_spectrogram(self, waveform):
        filtered_wav = signal.lfilter(b=[1, -self.preemphasis], a=[1], x=waveform, axis=-1, zi=None)
        linear_spectrogram = stft(y=filtered_wav, n_fft=self.n_fft, hop_length=self.hop_length,
                                  win_length=self.win_length)
        mel_spectorgram = np.dot(a=self.mel_basis, b=np.abs(linear_spectrogram))
        mel_dbs = amplitude_to_dbs(amplitude=mel_spectorgram) - self.ref_dbs
        return normalize_spectrogram(dbs=mel_dbs, min_dbs=self.min_dbs)

    def spectrogram_to_wav(self, spectrogram):
        denormalized = denormalize_spectrogram(dbs=spectrogram, min_dbs=self.min_dbs)
        linear = dbs_to_amplitude(dbs=(denormalized + self.ref_dbs))
        waveform = griffin_lim(spectrogram=(linear ** self.power), griffin_lim_iters=self.griff_lim_iters,
                               n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        return signal.lfilter(b=[1], a=[1 - self.preemphasis], x=waveform, axis=-1, zi=None)
