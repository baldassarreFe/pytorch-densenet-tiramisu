import unittest

from torch import FloatTensor, Size
from torch.autograd import Variable

from densenet import DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201
from densenet.utils import count_parameters


class TestDenseNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.rgb_channels = 3
        imagenet_h = 224
        imagenet_w = 224
        self.imagenet_classes = 1000

        self.images = Variable(FloatTensor(self.batch_size, self.rgb_channels, imagenet_h, imagenet_w))
        print('Images:', self.images.shape)

    def test_densenet(self):

        densenets = [
            DenseNet(self.rgb_channels, self.imagenet_classes),
            DenseNet121(),
            DenseNet161(),
            DenseNet169(),
            DenseNet201(),
        ]

        for densenet in densenets:
            with self.subTest(klass=type(densenet).__name__):
                print(densenet)
                print('Parameters:', count_parameters(densenet))

                logits = densenet(self.images)
                print('Logits:', logits.shape)
                self.assertEqual(logits.shape, Size((self.batch_size, self.imagenet_classes)))

if __name__ == '__main__':
    unittest.main()
