import numpy as np


def amplitude_to_dbs(amplitude):
    return 20 * np.log10(np.maximum(1e-5, amplitude))


def dbs_to_amplitude(dbs):
    return np.power(10.0, dbs * 0.05)


def normalize_spectrogram(dbs, min_level_db):
    return np.clip((dbs - min_level_db) / -min_level_db, 0, 1)


def denormalize_spectrogram(dbs, min_level_db):
    return (np.clip(dbs, 0, 1) * -min_level_db) + min_level_db
