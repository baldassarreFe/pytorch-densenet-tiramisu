from itertools import zip_longest
from typing import Sequence, Union, Optional

from torch.nn import Sequential, Conv2d, BatchNorm2d, Linear, init
from torch.nn import functional as F

from .classification_block import ClassificationBlock
from .feature_block import FeatureBlock
from .transition import Transition
from ..shared import DenseBlock


class DenseNet(Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 output_classes: int = 1000,
                 initial_num_features: int = 64,
                 dropout: float = 0.0,

                 dense_blocks_growth_rates: Union[int, Sequence[int]] = 32,
                 dense_blocks_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = 4,
                 dense_blocks_num_layers: Union[int, Sequence[int]] = (6, 12, 24, 16),
                 transition_blocks_compression_factors: Union[float, Sequence[float]] = 0.5):
        super(DenseNet, self).__init__()

        # region Parameters handling
        self.in_channels = in_channels
        self.output_classes = output_classes

        if type(dense_blocks_growth_rates) == int:
            dense_blocks_growth_rates = (dense_blocks_growth_rates,) * 4
        if dense_blocks_bottleneck_ratios is None or type(dense_blocks_bottleneck_ratios) == int:
            dense_blocks_bottleneck_ratios = (dense_blocks_bottleneck_ratios,) * 4
        if type(dense_blocks_num_layers) == int:
            dense_blocks_num_layers = (dense_blocks_num_layers,) * 4
        if type(transition_blocks_compression_factors) == float:
            transition_blocks_compression_factors = (transition_blocks_compression_factors,) * 3
        # endregion

        # region First convolution
        features = FeatureBlock(in_channels, initial_num_features)
        current_channels = features.out_channels
        self.add_module('features', features)
        # endregion

        # region Dense Blocks and Transition layers
        dense_blocks_params = [
            {
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in zip(dense_blocks_growth_rates, dense_blocks_num_layers, dense_blocks_bottleneck_ratios)
        ]
        transition_blocks_params = [
            {
                'compression': c
            }
            for c in transition_blocks_compression_factors
        ]

        block_pairs_params = zip_longest(dense_blocks_params, transition_blocks_params)
        for block_pair_idx, (dense_block_params, transition_block_params) in enumerate(block_pairs_params):
            block = DenseBlock(current_channels, **dense_block_params)
            current_channels = block.out_channels
            self.add_module(f'block_{block_pair_idx}', block)

            if transition_block_params is not None:
                transition = Transition(current_channels, **transition_block_params)
                current_channels = transition.out_channels
                self.add_module(f'trans_{block_pair_idx}', transition)
        # endregion

        # region Classification block
        self.add_module('classification', ClassificationBlock(current_channels, output_classes))
        # endregion

        # region Weight initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                init.constant_(module.bias, 0)
        # endregion

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits)
