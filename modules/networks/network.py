import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Base class for a neural network.
    """
    def __init__(self, n_layers, connectivity='sequential'):
        super(Network, self).__init__()
        self.n_out = None
        self.layers = nn.ModuleList([None for _ in range(n_layers)])
        self.connectivity = connectivity
        if self.connectivity == 'highway':
            self.gates = nn.ModuleList([None for _ in range(n_layers)])

    def forward(self, input):
        out = input
        for ind, layer in enumerate(self.layers):
            if self.connectivity == 'sequential':
                out = layer(out)
            elif self.connectivity == 'residual':
                new_out = layer(out)
                if ind == 0:
                    out = new_out
                else:
                    out = new_out + out
            elif self.connectivity == 'highway':
                new_out = layer(out)
                if ind > 0:
                    gate_out = self.gates[ind](out)
                    out = gate_out * out + (1. - gate_out) * new_out
                else:
                    out = new_out
            elif self.connectivity == 'concat':
                out = torch.cat([layer(out), out], dim=1)
            elif self.connectivity == 'concat_input':
                out = torch.cat([layer(out), input], dim=1)
        return out

    def reset(self, *args, **kwargs):
        pass

    def planning_mode(self, batch_size):
        pass

    def acting_mode(self):
        pass

    def detach_hidden_state(self):
        pass

    def attach_hidden_state(self):
        pass
