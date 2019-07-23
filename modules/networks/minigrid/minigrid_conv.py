from ..network import Network
from ...layers import ConvolutionalLayer
import torch.nn as nn


class MiniGridConv(Network):
    """
    Convolutional encoder model for minigrid envs.
    """
    def __init__(self, non_linearity='relu'):
        super(MiniGridConv, self).__init__(n_layers=4)
        self.layers[0] = ConvolutionalLayer(n_input=3, n_output=16, filter_size=2,
                                            padding=0, stride=1, non_linearity=non_linearity)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.layers[1] = ConvolutionalLayer(n_input=16, n_output=32, filter_size=2,
                                            padding=0, stride=1, non_linearity=non_linearity)
        self.layers[2] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=2,
                                            padding=0, stride=2, non_linearity=non_linearity)
        self.n_out = 64

    def forward(self, input):
        output = input
        output = self.layers[0](output)
        output = self.max_pool(output)
        output= self.layers[1](output)
        output = self.layers[2](output)
        output = output.view(-1, self.n_out)
        return output
