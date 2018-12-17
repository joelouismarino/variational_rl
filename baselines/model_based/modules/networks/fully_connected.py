from .network import Network
from ..layers import FullyConnectedLayer


class FullyConnectedNetwork(Network):
    """
    A fully-connected neural network.
    """
    def __init__(self, n_layers, n_input, n_units, connectivity='sequential',
                 batch_norm=False, non_linearity='linear', dropout=None):
        super(FullyConnectedNetwork, self).__init__(n_layers, connectivity)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = FullyConnectedLayer(n_in, n_units[l], batch_norm,
                                                 non_linearity, dropout)

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
