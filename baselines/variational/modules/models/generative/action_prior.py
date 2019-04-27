import torch
from ..model import Model
from ...networks import get_network


class ActionPrior(Model):

    def __init__(self, network_args):
        super(ActionPrior, self).__init__()
        self.network = get_network(network_args)

    def forward(self, state, action):
        # combine inputs
        inputs = torch.cat([state, action], dim=1)
        return self.network(inputs)
        # return self.network(state)

    # def forward(self, state, action, hidden_state):
    #     # combine inputs
    #     inputs = torch.cat([state, action, hidden_state], dim=1)
    #     return self.network(inputs)
    #     # return self.network(state)
