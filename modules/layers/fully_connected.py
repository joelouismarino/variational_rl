import torch.nn as nn
from .layer import Layer


class FullyConnectedLayer(Layer):
    """
    A fully-connected layer.
    """
    def __init__(self, n_input, n_output, batch_norm=False, layer_norm=False,
                 non_linearity=None, dropout=None):
        super(FullyConnectedLayer, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_output)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(n_output)

        if non_linearity is None or non_linearity == 'linear':
            pass
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
            self.init_gain = nn.init.calculate_gain('relu')
            self.bias_init = 0.1
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
            self.init_gain = nn.init.calculate_gain('tanh')
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

        if dropout:
            self.dropout = nn.Dropout1d(dropout)

        self.initialize()
