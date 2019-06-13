import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.audio_processor import AudioProcessor
from utils.audio_utils import load_wav
from utils.data_utils import pad_data, prepare_tensor
from utils.text_utils import text_to_sequence


class TextSpeechDataset(Dataset):
    def __init__(self, root_dir, annotations_file, parameters):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(root_dir, annotations_file), sep="|", header=None)
        self.cleaners = parameters["text_cleaner"]
        # self.outputs_per_step = parameters["outputs_per_step"]
        self.sample_rate = parameters["sample_rate"]

        self.ap = AudioProcessor(sample_rate=self.sample_rate, preemphasis=parameters["preemphasis"],
                                 frequency=parameters["frequency"], frame_length=parameters["frame_length"],
                                 frame_shift=parameters["frame_shift"], min_dbs=parameters["min_dbs"],
                                 ref_dbs=parameters["ref_dbs"],
                                 mels_size=parameters["mels_size"],
                                 griff_lim_iters=parameters["griff_lim_iters"],
                                 power=parameters["spectro_power"]
                                 )

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        text = self.annotations.iloc[index][1]
        text = np.asarray(text_to_sequence(text=text, cleaner_names=[self.cleaners]), dtype=np.int32)

        wav_path = os.path.join(self.root_dir, "wavs", f"{self.annotations.iloc[index][0]}.wav")
        wav = load_wav(filename=wav_path, sample_rate=self.sample_rate)
        return {"text": text, "wav": wav}

    def collate_fn(self, batch):
        wavs = list(map(lambda sample: sample["wav"], batch))
        texts = list(map(lambda sample: sample["text"], batch))

        max_text_len = np.max(list(map(lambda txt: len(txt), texts)))
        texts = np.stack(list(map(lambda txt: pad_data(txt, max_text_len), texts))).astype(np.int32)
        max_wav_len = np.max(list(map(lambda waveform: len(waveform), wavs)))
        wavs = np.stack(list(map(lambda waveform: pad_data(x=waveform, length=max_wav_len), wavs))).astype(np.int32)

        linears = list(
            map(lambda waveform: self.ap.wav_to_linear_spectrogram(waveform=waveform).astype(np.float32), wavs))
        mels = list(map(lambda waveform: self.ap.wav_to_mel_spectrogram(waveform=waveform).astype(np.float32), wavs))
        mel_lengths = list(map(lambda mel_: mel_.shape[1] + 1, mels))

        linears = prepare_tensor(inputs=linears, out_steps=self.outputs_per_step)
        mels = prepare_tensor(inputs=mels, out_steps=self.outputs_per_step)
        linears = linears.transpose(0, 2, 1)
        mels = mels.transpose(0, 2, 1)

        texts = torch.LongTensor(texts)
        linears = torch.FloatTensor(linears)
        mels = torch.FloatTensor(mels)
        mel_lengths = torch.LongTensor(mel_lengths)

        return texts, linears, mels, mel_lengths
