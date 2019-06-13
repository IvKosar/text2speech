from torch.nn import ReLU, Sigmoid, Linear, Module


class Highway(Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = Linear(in_features=in_size, out_features=out_size)
        self.H.bias.data.zero_()
        self.T = Linear(in_features=in_size, out_features=out_size)
        self.T.bias.data.fill_(-1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)
