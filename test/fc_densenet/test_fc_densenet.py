import unittest

from dense import FCDenseNet, FCDenseNet103
from dense.utils import count_parameters, count_conv2d

import torch


class TestDenseNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.rgb_channels = 3
        self.H = 180
        self.W = 240
        self.num_classes = 11

        self.images = torch.empty(self.batch_size, self.rgb_channels, self.H, self.W)
        print('Images:', self.images.shape)

    def test_fc_densenet(self):
        densenet = FCDenseNet(
            in_channels=self.rgb_channels,
            out_channels=self.num_classes,
            initial_num_features=24,
            dropout=0.2,

            down_dense_growth_rates=8,
            down_dense_bottleneck_ratios=None,
            down_dense_num_layers=(4, 5, 7),
            down_transition_compression_factors=1.0,

            middle_dense_growth_rate=8,
            middle_dense_bottleneck=None,
            middle_dense_num_layers=10,

            up_dense_growth_rates=8,
            up_dense_bottleneck_ratios=None,
            up_dense_num_layers=(7, 5, 4)
        )

        print(densenet)
        print('Layers:', count_conv2d(densenet))
        print('Parameters:', count_parameters(densenet))

        logits = densenet(self.images)
        print('Logits:', logits.shape)
        self.assertEqual(logits.shape, torch.Size((self.batch_size, self.num_classes, self.H, self.W)))

    def test_fc_densenet_103(self):
        densenet = FCDenseNet103()
        layers = count_conv2d(densenet)

        print('Layers:', layers)
        print('Parameters:', count_parameters(densenet))
        self.assertEqual(103, layers)


if __name__ == '__main__':
    unittest.main()
