import torch
from ..model import Model
from ...networks import get_network


class RewardLikelihood(Model):

    def __init__(self, network_args):
        super(RewardLikelihood, self).__init__()
        self.network = get_network(network_args)

    def forward(self, state):
        return self.network(state)

    # def forward(self, state, hidden_state):
    #     inputs = torch.cat([state, hidden_state], dim=1)
    #     return self.network(inputs)
