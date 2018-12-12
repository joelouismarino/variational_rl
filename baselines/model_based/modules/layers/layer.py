import torch.nn as nn
from torch.nn import init


class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()
        self.linear = lambda x: x
        self.batch_norm = lambda x: x
        self.non_linearity = lambda x: x
        self.dropout = lambda x: x
        self.init_gain = 1.

    def initialize(self):
        if type(self.linear) == nn.Module:
            init.xavier_normal_(self.linear.weight, gain=self.init_gain)
            init.constant_(self.linear.bias, 0.)
        if type(self.batch_norm) == nn.Module:
            init.normal_(self.batch_norm.weight, 1, 0.02)
            init.constant_(self.batch_norm.bias, 0.)

    def forward(self, input):
        x = self.linear(input)
        x = self.batch_norm(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        return x
