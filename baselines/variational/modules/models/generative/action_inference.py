import torch
from ..model import Model
from ...networks import get_network


class ActionInference(Model):

    def __init__(self, network_args):
        super(ActionInference, self).__init__()
        self.network = get_network(network_args)

    def forward(self, state, action):
        # combine inputs
        inputs = torch.cat([state, action], dim=1)
        return self.network(inputs)