import copy
import torch.nn as nn


class NetworkEnsemble(nn.Module):
    """
    Wrapper to ensemble multiple networks.
    """
    def __init__(self, network, n_networks=1):
        super(NetworkEnsemble, self).__init__()
        self.net_ensemble = nn.ModuleList([copy.deepcopy(network) for _ in range(n_networks)])

    @property
    def n_out(self):
        return self.net_ensemble[0].n_out

    def forward(self, input):
        output = []
        for network in self.net_ensemble:
            output.append(network.forward(input))
        if len(output) == 1:
            return output[0]
        return output

    def reset(self, batch_size):
        for network in self.net_ensemble:
            network.reset(batch_size)

    def planning_mode(self, batch_size):
        for network in self.net_ensemble:
            network.planning_mode(batch_size)

    def acting_mode(self):
        for network in self.net_ensemble:
            network.acting_mode(batch_size)

    def detach_hidden_state(self):
        for network in self.net_ensemble:
            network.detach_hidden_state(batch_size)

    def attach_hidden_state(self):
        for network in self.net_ensemble:
            network.attach_hidden_state(batch_size)
