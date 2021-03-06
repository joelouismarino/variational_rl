import torch
from .network import Network
from ..layers import RecurrentLayer, FullyConnectedLayer


class RecurrentNetwork(Network):
    """
    An LSTM neural network.
    """
    def __init__(self, n_layers, n_input, n_units, connectivity='sequential',
                 dropout=None, *args, **kwargs):
        super(RecurrentNetwork, self).__init__(n_layers, connectivity)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]

        n_in = n_input
        for l in range(n_layers):

            self.layers[l] = RecurrentLayer(n_in, n_units[l], dropout)

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

    def forward(self, input, step=True):
        output = super(RecurrentNetwork, self).forward(input)
        if step:
            self.step()
        return output

    def step(self):
        for layer in self.layers:
            layer.step()

    @property
    def state(self):
        return torch.cat([layer.state for layer in self.layers], dim=1)

    def reset(self, batch_size):
        for layer in self.layers:
            layer.reset(batch_size)

    def planning_mode(self, batch_size):
        for layer in self.layers:
            layer.planning_mode(batch_size)

    def acting_mode(self):
        for layer in self.layers:
            layer.acting_mode()

    def detach_hidden_state(self):
        for layer in self.layers:
            layer.detach_hidden_state()

    def attach_hidden_state(self):
        for layer in self.layers:
            layer.attach_hidden_state()
