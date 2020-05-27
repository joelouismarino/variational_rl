import torch
import torch.nn as nn
from torch.distributions import Normal
from ..layers import FullyConnectedLayer

INIT_W = 3e-3

class ValueVariable(nn.Module):
    """
    A simple module to map a full-connected input to a state or action-value
    estimate.

    Args:
        n_input (int): number of input dimensions
        stochastic (bool): whether to estimate a Gaussian std. dev.
    """
    def __init__(self, n_input, stochastic=False):
        super(ValueVariable, self).__init__()
        self.linear = FullyConnectedLayer(n_input, 1)
        nn.init.uniform_(self.linear.linear.weight, -INIT_W, INIT_W)
        nn.init.uniform_(self.linear.linear.bias, -INIT_W, INIT_W)
        # self.stochastic = stochastic
        # if self.stochastic:
        #     self.log_std_linear = FullyConnectedLayer(n_input, 1)
        #     nn.init.uniform_(self.log_std_linear.linear.weight, -INIT_W, INIT_W)
        #     nn.init.uniform_(self.log_std_linear.linear.bias, -INIT_W, INIT_W)
        #
        # self.loc = self.std = None

    def forward(self, x):
        """
        Sample the value estimate.
        """
        # self.loc = self.linear(x)
        # if self.stochastic:
        #     # estimate std. dev., reparameterization sampling
        #     self.std = self.log_std_linear(x).exp()
        #     value_estimate = self.loc + self.std * torch.zeros_like(self.std).normal_()
        # else:
        #     # identity std. dev., return loc as value estimate
        #     self.std = torch.ones_like(self.loc)
        #     value_estimate = self.loc
        # return value_estimate
        return self.linear(x)

    def reset(self):
        # self.loc = self.std =
        pass

    # def log_prob(self, x):
    #     """
    #     Evaluates the Gaussian log-probability density for an observation.
    #
    #     Args:
    #         x (torch.Tensor): the observed (target) value
    #     """
    #     return Normal(self.loc, self.std).log_prob(x)
