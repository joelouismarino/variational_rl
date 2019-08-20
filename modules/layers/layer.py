import torch.nn as nn
from misc.initialization import fanin_init


class Layer(nn.Module):
    """
    Parent class for a neural network layer.
    """
    def __init__(self):
        super(Layer, self).__init__()
        self.linear = lambda x: x
        self.batch_norm = lambda x: x
        self.non_linearity = lambda x: x
        self.dropout = lambda x: x
        self.init_gain = 1.
        self.bias_init = 0.

    def initialize(self):
        if type(self.linear) == nn.Module:
            # nn.init.xavier_normal_(self.linear.weight, gain=self.init_gain)
            # nn.init.kaiming_uniform_(self.linear.weight, gain=self.init_gain)
            fanin_init(self.linear.weight)
            nn.init.constant_(self.linear.bias, self.bias_init)
        if type(self.batch_norm) == nn.Module:
            nn.init.normal_(self.batch_norm.weight, 1, 0.02)
            nn.init.constant_(self.batch_norm.bias, 0.)

    def forward(self, input):
        x = self.linear(input)
        x = self.batch_norm(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        return x
