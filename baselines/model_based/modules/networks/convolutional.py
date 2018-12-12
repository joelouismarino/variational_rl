from .network import Network
from ..layers import ConvolutionalLayer


class ConvolutionalNetwork(Network):

    def __init__(self, n_layers, n_input, n_units, filter_sizes, paddings,
                 strides, connectivity='sequential', batch_norm=False,
                 non_linearity='linear', dropout=None):):
        super(ConvolutionalNetwork, self).__init__(n_layers)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = ConvolutionalLayer(n_in, n_units[l], filter_sizes[l],
                                                paddings[l], strides[l],
                                                batch_norm, non_linearity, dropout)

            if connectivity == 'sequential':
                n_in = n_units[l]
            elif connectivity == 'residual':
                if l > 0:
                    n_in = n_units[l] + n_units[l-1]
            elif connectivity == 'concat':
                n_in += n_units[l]
            elif connectivity == 'concat_input':
                n_in = n_units[l] + n_input

    def forward(self, input):
        pass
