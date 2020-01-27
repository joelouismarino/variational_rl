from ..network import Network
from ...layers import FullyConnectedLayer, TransposedConvLayer
import torch.nn as nn


class MiniGridDeconv(Network):
    """
    Transposed convolutional decoder model for minigrid envs.
    """
    def __init__(self, n_input, non_linearity='relu'):
        super(MiniGridDeconv, self).__init__(n_layers=3)
        self.layers[0] = FullyConnectedLayer(n_input=n_input, n_output=64)
        self.layers[1] = TransposedConvLayer(n_input=64, n_output=32, filter_size=2,
                                            padding=0, stride=2, non_linearity=non_linearity)
        # self.max_pool = nn.MaxPool2d((2, 2))
        self.layers[2] = TransposedConvLayer(n_input=32, n_output=16, filter_size=3,
                                            padding=0, stride=2, non_linearity=non_linearity)
        self.n_out = 16

    def forward(self, input):
        output = input
        output = self.layers[0](output)
        output = output.view(-1, 64, 1, 1)
        output = self.layers[1](output)
        output = self.layers[2](output)
        return output
