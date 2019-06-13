from torch.nn import Module, ModuleList, Linear, ReLU, Dropout


class Prenet(Module):
    def __init__(self, in_features, out_features=(256, 128)):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = ModuleList(modules=[Linear(in_features=in_size, out_features=out_size) for (in_size, out_size) in
                                          zip(in_features, out_features)])
        self.relu = ReLU()
        self.dropout = Dropout(p=0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs
