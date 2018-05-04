import unittest
from math import ceil

import torch

from dense.densenet.transition import Transition
from dense.utils import count_parameters


class TestTransition(unittest.TestCase):

    def test_transition(self):
        in_channels = 512
        H = 32
        W = 64
        x = torch.empty(1, in_channels, H, W)
        print('x:', x.shape)

        for c in [1.0, 0.9, 0.001]:
            with self.subTest(compression=c):
                transition = Transition(in_channels, compression=c)
                print(transition)
                print('Parameters:', count_parameters(transition))

                y = transition(x)
                print('y:', y.shape)

                self.assertEqual(y.shape[1], int(ceil(c * in_channels)))
                self.assertEqual(y.shape[2], H // 2)
                self.assertEqual(y.shape[3], W // 2)

    def test_compression_range(self):
        with self.assertRaises(ValueError):
            Transition(32, compression=-1.0)
        with self.assertRaises(ValueError):
            Transition(32, compression=0.0)
        with self.assertRaises(ValueError):
            Transition(32, compression=1.5)


if __name__ == '__main__':
    unittest.main()
