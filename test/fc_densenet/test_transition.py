import unittest

import torch

from dense.fc_densenet.transition_down import TransitionDown
from dense.fc_densenet.transition_up import TransitionUp, CenterCropConcat
from dense.utils import count_parameters


class TestTransition(unittest.TestCase):

    def test_transition_down(self):
        in_channels = 512
        H = 32
        W = 64
        x = torch.zeros(1, in_channels, H, W)
        print('x:', x.shape)

        for d in [0.0, 0.2]:
            with self.subTest(dropout=d):
                transition = TransitionDown(in_channels, dropout=d)
                print(transition)
                print('Parameters:', count_parameters(transition))

                y = transition(x)
                print('y:', y.shape)

                self.assertEqual(y.shape[1], in_channels)
                self.assertEqual(y.shape[2], H // 2)
                self.assertEqual(y.shape[3], W // 2)

    def test_center_crop_concat(self):
        x = torch.zeros(1, 3, 2, 6)
        y = torch.zeros(1, 7, 5, 4)
        x[:, :, :, 1:5] = 1
        y[:, :, 1:3, :] = 1
        print('x:', x.shape)
        print('y:', y.shape)

        ccc = CenterCropConcat()
        res = ccc(x, y)
        print('res:', res.shape)

        self.assertEqual(res.size(1), x.size(1) + y.size(1))
        self.assertEqual(res.size(2), min(x.size(2), y.size(2)))
        self.assertEqual(res.size(3), min(x.size(3), y.size(3)))
        self.assertTrue((res == 1).all())

    def test_transition_up(self):
        # Transposed convolution upsamples this to (:, :, 13, 17)
        upsample = torch.zeros(1, 3, 6, 8)
        skip = torch.zeros(1, 7, 15, 14)
        print('upsample:', upsample.shape)
        print('skip:', skip.shape)

        transition = TransitionUp(upsample.size(1))
        print(transition)
        print('Parameters:', count_parameters(transition))

        res = transition(upsample, skip)
        print('res:', res.shape)

        self.assertEqual(res.size(1), upsample.size(1) + skip.size(1))
        self.assertEqual(res.size(2), min(13, skip.size(2)))
        self.assertEqual(res.size(3), min(17, skip.size(3)))


if __name__ == '__main__':
    unittest.main()
