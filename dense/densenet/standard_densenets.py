from .densenet import DenseNet


class DenseNet121(DenseNet):
    def __init__(self, dropout: float = 0.0):
        super(DenseNet121, self).__init__(
            in_channels=3,
            output_classes=1000,
            initial_num_features=64,
            dropout=dropout,
            dense_blocks_growth_rates=32,
            dense_blocks_bottleneck_ratios=4,
            dense_blocks_num_layers=(6, 12, 24, 16),
            transition_blocks_compression_factors=0.5
        )


class DenseNet169(DenseNet):
    def __init__(self, dropout: float = 0.0):
        super(DenseNet169, self).__init__(
            in_channels=3,
            output_classes=1000,
            initial_num_features=64,
            dropout=dropout,
            dense_blocks_growth_rates=32,
            dense_blocks_bottleneck_ratios=4,
            dense_blocks_num_layers=(6, 12, 32, 32),
            transition_blocks_compression_factors=0.5
        )


class DenseNet201(DenseNet):
    def __init__(self, dropout: float = 0.0):
        super(DenseNet201, self).__init__(
            in_channels=3,
            output_classes=1000,
            initial_num_features=64,
            dropout=dropout,
            dense_blocks_growth_rates=32,
            dense_blocks_bottleneck_ratios=4,
            dense_blocks_num_layers=(6, 12, 48, 32),
            transition_blocks_compression_factors=0.5
        )


class DenseNet161(DenseNet):
    def __init__(self, dropout: float = 0.0):
        super(DenseNet161, self).__init__(
            in_channels=3,
            output_classes=1000,
            initial_num_features=64,
            dropout=dropout,
            dense_blocks_growth_rates=48,
            dense_blocks_bottleneck_ratios=4,
            dense_blocks_num_layers=(6, 12, 36, 24),
            transition_blocks_compression_factors=0.5
        )

