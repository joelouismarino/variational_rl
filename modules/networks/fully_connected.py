from .network import Network
from ..layers import FullyConnectedLayer


class FullyConnectedNetwork(Network):
    """
    A fully-connected neural network.
    """
    def __init__(self, n_layers, n_input, n_units, connectivity='sequential',
                 batch_norm=False, layer_norm=False, non_linearity='linear',
                 dropout=None):
        super(FullyConnectedNetwork, self).__init__(n_layers, connectivity)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]
        else:
            assert len(n_units) == n_layers

        if type(non_linearity) == str:
            non_linearity = [non_linearity for _ in range(n_layers)]
        else:
            assert len(non_linearity) == n_layers

        if type(batch_norm) == bool:
            batch_norm = [batch_norm for _ in range(n_layers)]
        else:
            assert len(batch_norm) == n_layers

        if type(layer_norm) == bool:
            layer_norm = [layer_norm for _ in range(n_layers)]
        else:
            assert len(layer_norm) == n_layers

        if type(dropout) == float or dropout is None:
            dropout = [dropout for _ in range(n_layers)]
        else:
            assert len(dropout) == n_layers

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = FullyConnectedLayer(n_in, n_units[l],
                                                 batch_norm=batch_norm[l],
                                                 layer_norm=layer_norm[l],
                                                 non_linearity=non_linearity[l],
                                                 dropout=dropout[l])

            if connectivity in ['sequential', 'residual']:
                n_in = n_units[l]
            elif connectivity == 'highway':
                n_in = n_units[l]
                if l > 0:
                    self.gates[l] = FullyConnectedLayer(n_in, n_units[l],
                                                        non_linearity='sigmoid')
            elif connectivity == 'concat':
                n_in += n_units[l]
            elif connectivity == 'concat_input':
                n_in = n_units[l] + n_input
            else:
                raise NotImplementedError

        self.n_out = n_in
