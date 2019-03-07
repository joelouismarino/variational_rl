import torch
import torch.nn as nn
from ..network import Network
from ...layers import FullyConnectedLayer, TransposedConvLayer, ConvolutionalLayer


class SkipConvDecoder(Network):
    """
    Convolutional decoder model from Ha & Schmidhuber, 2018.
    """
    def __init__(self, n_input=64, non_linearity='relu'):
        super(SkipConvDecoder, self).__init__(n_layers=6)
        self.layers[0] = FullyConnectedLayer(n_input=n_input, n_output=1024)

        self.layers[1] = TransposedConvLayer(n_input=1024, n_output=128, filter_size=5,
                                             padding=0, stride=2, non_linearity=non_linearity)
        self.layers[2] = TransposedConvLayer(n_input=128, n_output=64, filter_size=5,
                                             padding=0, stride=2, non_linearity=non_linearity)
        self.layers[3] = TransposedConvLayer(n_input=64, n_output=32, filter_size=6,
                                             padding=0, stride=2, non_linearity=non_linearity)

        self.layers[4] = ConvolutionalLayer(n_input=1024, n_output=16, filter_size=1, padding=0, stride=1)
        self.layers[5] = ConvolutionalLayer(n_input=128, n_output=16, filter_size=3, padding=1, stride=1)

        self.n_out = 32 * 2

    def forward(self, input):
        output = input
        output = self.layers[0](output)
        a = output.view(-1, 1024, 1, 1)
        b = self.layers[1](a)
        c = self.layers[2](b)
        d = self.layers[3](c)
        e = self.layers[4](a)
        e = nn.functional.interpolate(e, scale_factor=30)
        f = self.layers[5](b)
        f = nn.functional.interpolate(f, scale_factor=6)
        return torch.cat((d, e, f), dim=1)
