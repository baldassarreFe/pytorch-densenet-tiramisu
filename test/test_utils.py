import unittest

from torch.nn import Sequential, ReLU, Linear, Softmax

from dense.utils import RichRepr


class DummyMLP(RichRepr, Sequential):

    def __init__(self, in_features=8, layers=(16, 8, 4), final_softmax=True):
        super(DummyMLP, self).__init__()
        self.in_features = in_features
        self.layers = layers
        self.final_softmax = final_softmax
        for i, of in enumerate(layers):
            self.add_module(f'lin_{i}', Linear(in_features, of))
            in_features = of
            self.add_module(f'relu_{i}', ReLU(inplace=True))
        if final_softmax:
            self.__delattr__(f'relu_{i}')
            self.add_module('soft', Softmax())

    def __repr__(self):
        return super(DummyMLP, self).__repr__(
            self.in_features, *self.layers, 'softmax' if self.final_softmax else 'logits')


class TestUtils(unittest.TestCase):
    def test_rich_repr(self):
        dl = DummyMLP()
        string = str(dl)
        self.assertIn(dl.__class__.__name__, string)
        self.assertIn('softmax', string)

        dl = DummyMLP(in_features=2, layers=(4, 6, 4, 2), final_softmax=False)
        string = str(dl)
        self.assertIn(dl.__class__.__name__, string)
        self.assertIn('2, 4, 6, 4, 2', string)
        self.assertIn('logits', string)


if __name__ == '__main__':
    unittest.main()
