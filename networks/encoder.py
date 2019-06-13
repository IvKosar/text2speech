from torch.nn import Module

from modules.cbhg import CBHG
from modules.prenet import Prenet


class Encoder(Module):
    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features=in_features, out_features=[256, 128])
        self.cbhg = CBHG(sample_size=128, conv_bank_max_filter_size=16, conv_projections_channel_size=[128, 128])

    def forward(self, inputs):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)
