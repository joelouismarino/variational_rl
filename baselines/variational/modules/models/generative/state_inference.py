import torch
from ..model import Model
from ...networks import get_network


class StateInference(Model):

    def __init__(self, network_args):
        super(StateInference, self).__init__()
        self.network = get_network(network_args)

    def forward(self, params, grads):
        # combine inputs
        inputs = torch.cat([params, grads], dim=1)
        return self.network(inputs)
