import torch.nn as nn

class BatchNormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm1d(out_channels=out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)