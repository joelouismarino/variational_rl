import torch.nn as nn
from ..layers import FullyConnectedLayer

INIT_W = 3e-3

class ValueVariable(nn.Module):

    def __init__(self, n_input):
        super(ValueVariable, self).__init__()
        self.linear = FullyConnectedLayer(n_input, 1)
        # nn.init.uniform_(self.linear.linear.weight, -INIT_W, INIT_W)
        # nn.init.uniform_(self.linear.linear.bias, -INIT_W, INIT_W)

        nn.init.constant_(self.linear.linear.weight, 0.)
        nn.init.constant_(self.linear.linear.bias, 0.)

    def forward(self, x):
        return self.linear(x)
