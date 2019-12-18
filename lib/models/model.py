import torch
import torch.nn as nn
from ..networks import get_network


class Model(nn.Module):
    """
    Wrapper class around network to parameterize each of the conditional mappings.
    """
    def __init__(self, network_args):
        super(Model, self).__init__()
        self.inputs = network_args.pop('inputs')
        self.network = get_network(network_args)

    @property
    def n_out(self):
        if self.network is not None:
            return self.network.n_out
        else:
            raise NotImplementedError

    def reset(self, batch_size=0):
        if self.network is not None:
            self.network.reset(batch_size)

    def planning_mode(self, batch_size=0):
        if self.network is not None:
            self.network.planning_mode(batch_size)

    def acting_mode(self):
        if self.network is not None:
            self.network.acting_mode()

    def detach_hidden_state(self):
        if self.network is not None:
            self.network.detach_hidden_state()

    def attach_hidden_state(self):
        if self.network is not None:
            self.network.attach_hidden_state()

    def forward(self, **kwargs):
        inputs = []
        for k in self.inputs:
            if k in kwargs:
                inputs += [kwargs[k]]
            else:
                raise InputError
        if len(inputs) > 1:
            inputs = torch.cat(inputs, dim=1)
        else:
            inputs = inputs[0]
        return self.network(inputs)
