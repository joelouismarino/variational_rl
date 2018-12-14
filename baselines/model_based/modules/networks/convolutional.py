from .network import Network
from ..layers import ConvolutionalLayer


class ConvolutionalNetwork(Network):
    """
    A convolutional neural network.
    """
    def __init__(self, n_layers, n_input, n_units, filter_sizes, paddings,
                 strides, connectivity='sequential', batch_norm=False,
                 non_linearity='linear', dropout=None):
        super(ConvolutionalNetwork, self).__init__(n_layers, connectivity)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]

        if type(filter_sizes) == int:
            filter_sizes = [filter_sizes for _ in range(n_layers)]

        if type(paddings) == int:
            paddings = [paddings for _ in range(n_layers)]

        if type(strides) == int:
            strides = [strides for _ in range(n_layers)]

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = ConvolutionalLayer(n_in, n_units[l], filter_sizes[l],
                                                paddings[l], strides[l],
                                                batch_norm, non_linearity, dropout)

            if connectivity in ['sequential', 'residual']:
                n_in = n_units[l]
            elif connectivity == 'highway':
                n_in = n_units[l]
                if l > 0:
                    self.gates[l] = ConvolutionalLayer(n_in, n_units[l],
                                                       filter_sizes[l],
                                                       paddings[l], strides[l],
                                                       non_linearity='sigmoid')
            elif connectivity == 'concat':
                n_in += n_units[l]
            elif connectivity == 'concat_input':
                n_in = n_units[l] + n_input
            else:
                raise NotImplementedError

        self.n_out = n_in
