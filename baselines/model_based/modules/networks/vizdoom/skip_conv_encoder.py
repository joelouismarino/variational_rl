import torch
from ..network import Network
from ...layers import ConvolutionalLayer


class SkipConvEncoder(Network):
    """
    Convolutional encoder model from Ha & Schmidhuber, 2018.
    """
    def __init__(self, non_linearity='relu'):
        super(SkipConvEncoder, self).__init__(n_layers=6)
        self.layers[0] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[1] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[2] = ConvolutionalLayer(n_input=96, n_output=128, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[3] = ConvolutionalLayer(n_input=192, n_output=256, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[4] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=9, padding=0, stride=4)
        self.layers[5] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=9, padding=0, stride=4)

        self.n_out = 1024

    def forward(self, input):
        a = self.layers[0](input)
        b = self.layers[1](a)
        skip_a = self.layers[4](input)
        c = self.layers[2](torch.cat((b, skip_a), dim=1))
        skip_b = self.layers[5](a)
        d = self.layers[3](torch.cat((c, skip_b), dim=1))
        output = d.view(-1, 1024)
        return output
