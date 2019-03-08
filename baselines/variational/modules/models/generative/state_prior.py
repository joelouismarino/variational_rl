import torch
from ..model import Model
from ...networks.network import FullyConnectedNetwork


class StatePrior(Model):

    def __init__(self, network_args):
        super(StatePrior, self).__init__()
        self.network = FullyConnectedNetwork(**network_args)

    def forward(self, state, action):
        # combine inputs
        inputs = torch.cat([state, action], dim=1)
        return self.network(inputs)
