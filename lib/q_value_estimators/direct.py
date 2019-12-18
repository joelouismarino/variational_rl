import copy
import torch.nn as nn
from lib.models import get_model


class DirectEstimator(nn.Module):

    def __init__(self, agent, network_args):
        self.agent = agent
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])

    def forward(self, state, action):
        # estimate q value
        return q_value

    def reset(self):
        pass
