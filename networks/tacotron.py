from torch.nn import Module, Embedding, Linear

from modules.cbhg import CBHG
from networks.decoder import Decoder
from networks.encoder import Encoder
from utils.text_utils import symbols


class Tacotron(Module):
    def __init__(self, embedding_dim=256, linear_dim=1025, mel_dim=80, r=5, padding_idx=None):
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.embedding = Embedding(num_embeddings=len(symbols), embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0, std=0.3)
        self.encoder = Encoder(in_features=embedding_dim)
        self.decoder = Decoder(in_features=256, memory_dim=mel_dim, r=r)
        self.postnet = CBHG(sample_size=mel_dim, conv_bank_max_filter_size=8,
                            conv_projections_channel_size=[256, mel_dim], num_highways=4)
        self.last_linear = Linear(in_features=(mel_dim * 2), out_features=linear_dim)

    def forward(self, characters, mel_specs=None):
        inputs = self.embedding(inputs=characters)
        encoder_outputs = self.encoder(inputs=inputs)
        mel_outputs, alignments = self.decoder(inputs=encoder_outputs, memory=mel_specs)
        linear_outputs = self.last_linear(self.postnet(mel_outputs.view(characters.size(0), -1, self.mel_dim)))
        return mel_outputs, linear_outputs, alignments
