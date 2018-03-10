from math import ceil

from torch.nn import Sequential, BatchNorm2d, ReLU, Conv2d, AvgPool2d

from ..utils import RichRepr


class Transition(RichRepr, Sequential):
    r"""
    Transition Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet

    Consists of:
    - Batch Normalization
    - ReLU
    - 1x1 Convolution (with optional compression of the number of channels)
    - 2x2 Average Pooling
    """

    def __init__(self, in_channels, compression: float = 1.0):
        super(Transition, self).__init__()
        if not 0.0 < compression <= 1.0:
            raise ValueError(f'Compression must be in (0, 1] range, got {compression}')

        self.in_channels = in_channels
        self.out_channels = int(ceil(compression * in_channels))

        self.add_module('norm', BatchNorm2d(num_features=self.in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False))
        self.add_module('pool', AvgPool2d(kernel_size=2, stride=2))

    def __repr__(self):
        return super(Transition, self).__repr__(self.in_channels, self.out_channels)
