import unittest

import torch

from dense.shared import DenseBlock
from dense.utils import count_parameters


class TestDenseBlock(unittest.TestCase):

    def test_dense_block(self):
        in_channels = 64
        x = torch.empty(1, in_channels, 32, 64)
        print('x:', x.shape)

        block_params = [
            {'growth_rate': 4, 'num_layers': 2, 'concat_input': False},
            {'growth_rate': 4, 'num_layers': 2, 'concat_input': True},
            {'growth_rate': 4, 'num_layers': 3, 'concat_input': False},
            {'growth_rate': 4, 'num_layers': 3, 'concat_input': True},
        ]
        dense_layer_params = {'bottleneck_ratio': 4, 'dropout': 0.2}

        for params in block_params:
            with self.subTest(**params):
                dense_block = DenseBlock(in_channels, **params, dense_layer_params=dense_layer_params)
                print(dense_block)
                print('Parameters:', count_parameters(dense_block))

                y = dense_block(x)
                print('y:', y.shape)

                expected_out_channels = params['growth_rate'] * params['num_layers']
                if params['concat_input']:
                    expected_out_channels += in_channels
                self.assertEqual(y.shape[1], expected_out_channels)


if __name__ == '__main__':
    unittest.main()
