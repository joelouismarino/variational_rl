import torch
from ..model import Model
from ...networks import get_network


class ObservationLikelihood(Model):

    def __init__(self, network_args):
        super(ObservationLikelihood, self).__init__()
        self.network = get_network(network_args)

    def forward(self, state):
        return self.network(state)
