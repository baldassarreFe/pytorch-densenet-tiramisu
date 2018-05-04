import unittest

import torch

from dense.shared import Bottleneck
from dense.utils import count_parameters


class TestBettleneck(unittest.TestCase):
    def test_bottleneck(self):
        in_channels = 512
        out_channels = 8
        x = torch.empty(1, in_channels, 32, 64)
        print('x:', x.shape)

        bottleneck = Bottleneck(in_channels, out_channels)
        print(bottleneck)
        print('Parameters:', count_parameters(bottleneck))

        y = bottleneck(x)
        print('y:', y.shape)
        self.assertEqual(y.shape[1], out_channels)


if __name__ == '__main__':
    unittest.main()
