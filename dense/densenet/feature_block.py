from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d

from ..utils import RichRepr


class FeatureBlock(RichRepr, Sequential):
    def __init__(self, in_channels, out_channels):
        super(FeatureBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)),
        self.add_module('norm', BatchNorm2d(out_channels)),
        self.add_module('relu', ReLU(inplace=True)),
        self.add_module('pool', MaxPool2d(kernel_size=3, stride=2, padding=1)),

    def __repr__(self):
        return super(FeatureBlock, self).__repr__(self.in_channels, self.out_channels)
