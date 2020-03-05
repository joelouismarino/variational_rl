from .network import Network
from ..layers import ARFullyConnectedLayer


class ARFullyConnectedNetwork(Network):
    """
    An auto-regressive fully-connected neural network.
    """
    def __init__(self, n_layers, n_input, n_units, connectivity='sequential',
                 batch_norm=False, layer_norm=False, non_linearity='linear',
                 dropout=None):
        super(ARFullyConnectedNetwork, self).__init__(n_layers, connectivity)

        assert connectivity in ['sequential', 'residual', 'highway']

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]
        else:
            assert len(n_units) == n_layers

        if type(non_linearity) == str:
            non_linearity = [non_linearity for _ in range(n_layers)]
        else:
            assert len(non_linearity) == n_layers

        if type(batch_norm) == bool:
            assert batch_norm == False
            batch_norm = [False for _ in range(n_layers)]
        else:
            assert batch_norm[0] == False
            assert len(batch_norm) == n_layers

        if type(layer_norm) == bool:
            assert layer_norm == False
            layer_norm = [False for _ in range(n_layers)]
        else:
            assert layer_norm[0] == False
            assert len(layer_norm) == n_layers

        if type(dropout) == float or dropout is None:
            assert dropout is None
            dropout = [None for _ in range(n_layers)]
        else:
            assert dropout[0] is None
            assert len(dropout) == n_layers

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = ARFullyConnectedLayer(n_in, n_units[l],
                                                   batch_norm=batch_norm[l],
                                                   layer_norm=layer_norm[l],
                                                   non_linearity=non_linearity[l],
                                                   dropout=dropout[l])

            if connectivity in ['sequential', 'residual']:
                n_in = n_units[l]
            elif connectivity == 'highway':
                n_in = n_units[l]
                if l > 0:
                    self.gates[l] = ARFullyConnectedLayer(n_in, n_units[l],
                                                          non_linearity='sigmoid')
            elif connectivity == 'concat':
                n_in += n_units[l]
            elif connectivity == 'concat_input':
                n_in = n_units[l] + n_input
            else:
                raise NotImplementedError

        self.n_out = n_in
