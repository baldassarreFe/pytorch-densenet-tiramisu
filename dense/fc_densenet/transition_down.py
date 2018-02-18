from torch.nn import Sequential, BatchNorm2d, ReLU, Conv2d, Dropout2d, MaxPool2d


class TransitionDown(Sequential):
    r"""
    Transition Down Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326)
    """

    def __init__(self, in_channels: int, dropout: float = 0.0):
        super(TransitionDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False))
        if dropout > 0:
            self.add_module('drop', Dropout2d(dropout))
        self.add_module('pool', MaxPool2d(kernel_size=2, stride=2))
