import torch
from ..model import Model
from ...networks import get_network


class StatePrior(Model):

    def __init__(self, network_args):
        super(StatePrior, self).__init__()
        self.network = get_network(network_args)

    def forward(self, observation, reward, state, action):
        # combine inputs
        bs = observation.shape[0]
        inputs = torch.cat([observation.contiguous().view(bs, -1), reward, state, action], dim=1)
        return self.network(inputs)
