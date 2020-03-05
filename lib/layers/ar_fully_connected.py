import torch
import torch.nn as nn
from .fully_connected import FullyConnectedLayer


class ARFullyConnectedLayer(FullyConnectedLayer):

    def __init__(self, n_input, n_output, batch_norm=False, layer_norm=False,
                 non_linearity=None, dropout=None):
        assert n_output == n_input
        super(ARFullyConnectedLayer, self).__init__(n_input, n_output,
                                                    batch_norm=False,
                                                    layer_norm=False,
                                                    non_linearity=None,
                                                    dropout=None)

    def forward(self, input):
        x = torch.addmm(self.linear.bias, input, self.linear.weight.tril(-1))
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        return x
