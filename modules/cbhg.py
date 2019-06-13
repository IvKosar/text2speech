from batch_norm_conv1d import BatchNormConv1d
from highway import Highway
from torch import cat
from torch.nn import Module, ModuleList, Linear, GRU, ReLU, MaxPool1d


class CBHG(Module):
    """Encoder CBHD
        It consists of:
            - 1-D convolution bank
            - highway network
            - bidirectional GRU
    """

    def __init__(self, sample_size, conv_bank_max_filter_size=16, conv_projections_channel_size=(128, 128),
                 num_highways=4):
        super(CBHG, self).__init__()
        self.sample_size = sample_size
        self.relu = ReLU()

        # conv_bank_max_filter_size sets of 1-D convolutional filters
        self.conv1d_banks = []
        for k in range(1, conv_bank_max_filter_size + 1):
            self.conv1d_banks.append(
                BatchNormConv1d(in_channels=sample_size, out_channels=sample_size, kernel_size=k, stride=1,
                                padding=k // 2, activation=self.relu))
        self.conv1d_banks = ModuleList(modules=self.conv1d_banks)

        # max pooling of conv bank (to increase local invariances)
        self.max_pool1d = MaxPool1d(kernel_size=2, stride=1, padding=1)

        out_features = [conv_bank_max_filter_size * sample_size] + conv_projections_channel_size[:-1]
        activations = [self.relu] * (len(conv_projections_channel_size) - 1) + [None]

        # conv1d projection layers
        self.conv1d_projections = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections_channel_size, activations):
            self.conv1d_projections.append(
                BatchNormConv1d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1,
                                padding=1, activation=ac))
        self.conv1d_projections = ModuleList(modules=self.conv1d_projections)

        # Highway layers
        self.pre_highway = Linear(in_features=conv_projections_channel_size[-1], out_features=sample_size, bias=False)
        self.highways = ModuleList(
            modules=[Highway(in_size=sample_size, out_size=sample_size) for _ in range(num_highways)])

        # bi-directional GPU layer
        self.gru = GRU(input_size=sample_size, hidden_size=sample_size, num_layers=1, batch_first=True,
                       bidirectional=True)

    def forward(self, inputs):
        x = inputs

        if x.size(-1) == self.sample_size:
            x = x.transpose(1, 2)

        T = x.size(-1)

        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            out = out[:, :, :T]
            outs.append(out)

        x = cat(tensors=outs, dim=1)
        assert x.size(1) == self.sample_size * len(self.conv1d_banks)

        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        x = x.transpose(1, 2)

        if x.size(-1) != self.sample_size:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        self.gru.flatten_parameters()
        return self.gru(x)[0]
