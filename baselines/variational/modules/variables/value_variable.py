import torch.nn as nn
from ..layers import FullyConnectedLayer


class ValueVariable(nn.Module):

    def __init__(self, n_input):
        super(ValueVariable, self).__init__()
        self.linear = FullyConnectedLayer(n_input, 1)

    def forward(self, x):
        return self.linear(x)
