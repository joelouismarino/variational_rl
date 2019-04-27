import torch
from ..model import Model
from ...networks import get_network


# class ActionInference(Model):
#
#     def __init__(self, network_args):
#         super(ActionInference, self).__init__()
#         self.network = get_network(network_args)
#
#     def forward(self, params, grads):
#         # combine inputs
#         inputs = torch.cat([params, grads], dim=1)
#         return self.network(inputs)

class ActionInference(Model):

    def __init__(self, network_args):
        super(ActionInference, self).__init__()
        self.network = get_network(network_args)

    # def forward(self, state, action, hidden_state):
    def forward(self, state, action):
        # combine inputs
        # bs = observation.shape[0]
        # inputs = torch.cat([observation.contiguous().view(bs, -1), reward, state, action], dim=1)
        # inputs = torch.cat([state, action, hidden_state], dim=1)
        inputs = torch.cat([state, action], dim=1)
        return self.network(inputs)
