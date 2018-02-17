from itertools import zip_longest
from typing import Sequence, Union

from torch.nn import Sequential, Conv2d, BatchNorm2d, Linear, init
from torch.nn import functional as F

from .feature_block import FeatureBlock
from .classification_block import ClassificationBlock
from ..shared import DenseBlock, Transition


class DenseNet(Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 output_classes: int = 1000,
                 initial_num_features: int = 64,
                 dropout: float = 0.0,
                 growth_rates: Union[int, Sequence[int]] = 32,
                 bottleneck_ratios: Union[int, Sequence[int]] = 4,
                 dense_blocks_num_layers: Union[int, Sequence[int]] = (6, 12, 24, 16),
                 compression_factors: Union[float, Sequence[float]] = 0.5):
        super(DenseNet, self).__init__()

        # Parameters handling
        self.in_channels = in_channels
        self.output_classes = output_classes

        if type(growth_rates) == int:
            growth_rates = (growth_rates,) * 4
        if type(bottleneck_ratios) == int:
            bottleneck_ratios = (bottleneck_ratios,) * 4
        if type(dense_blocks_num_layers) == int:
            dense_blocks_num_layers = (dense_blocks_num_layers,) * 4
        if type(compression_factors) == float:
            compression_factors = (compression_factors,) * 3

        # First convolution
        features = FeatureBlock(in_channels, initial_num_features)
        current_channels = features.out_channels
        self.add_module('features', features)

        # Dense Blocks and Transition layers
        block_parameters = zip_longest(growth_rates, bottleneck_ratios, dense_blocks_num_layers, compression_factors)
        for i, (growth_rate, bottleneck_ratio, num_layers, compression) in enumerate(block_parameters):
            block = DenseBlock(current_channels, growth_rate, num_layers, concat_input=True,
                               dense_layer_params={'bottleneck_ratio': bottleneck_ratio, 'dropout': dropout})
            current_channels = block.out_channels
            self.add_module(f'block_{i}', block)

            if compression is not None:
                transition = Transition(current_channels, compression)
                current_channels = transition.out_channels
                self.add_module(f'trans_{i}', transition)

        # Classification block
        self.add_module('classification', ClassificationBlock(current_channels, output_classes))

        # Weight initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform(module.weight)
                init.constant(module.bias, 0)

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits)
