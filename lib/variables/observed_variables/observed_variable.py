import torch
import torch.nn as nn
import numpy as np
from lib.distributions import Distribution
import torch.distributions.constraints as constraints


class ObservedVariable(nn.Module):
    """
    An observed latent variable.

    Args:
        likelihood_dist (str): the name of the conditional likelihood distribution
        n_variables (int): number of observed variables
        n_input (int): input size to the output models
        constant_scale (bool): whether to use a constant scale
        residual_loc (bool): whether to use a residual mapping for loc parameter
    """
    def __init__(self, likelihood_dist, n_variables, n_input, stochastic,
                 constant_scale, residual_loc, manual_loc, manual_loc_alpha):
        super(ObservedVariable, self).__init__()
        self.cond_likelihood = Distribution(likelihood_dist, n_variables, n_input,
                                            stochastic=stochastic,
                                            constant_scale=constant_scale,
                                            residual_loc=residual_loc,
                                            manual_loc=manual_loc,
                                            manual_loc_alpha=manual_loc_alpha)
        self.n_variables = n_variables
        self.planning = False
        self._n_planning_samples = 1
        self._batch_size = 1

    def generate(self, input, **kwargs):
        """
        Generate the conditional likelihood distribution from the input.

        Args:
            input (torch.Tensor): input to the final linear layers to the
                                  distribution parameters
        """
        self.cond_likelihood.step(input, **kwargs)

    def sample(self):
        """
        Sample the conditional likelihood disribution.
        """
        return self.cond_likelihood.sample()

    def cond_log_likelihood(self, x):
        """
        Evaluate the log conditional likelihood of the value x.

        Args:
            x (torch.Tensor) [batch_size, n_variables]: the point of evaluation
        """
        return self.cond_likelihood.log_prob(x).sum(dim=1)

    def planning_mode(self, n_planning_samples=1):
        self.planning = True
        self.cond_likelihood.planning_mode()
        self._n_planning_samples = n_planning_samples

    def acting_mode(self):
        self.planning = False
        self.cond_likelihood.acting_mode()
        self._n_planning_samples = 1

    def reset(self, batch_size=1, prev_x=None):
        self.cond_likelihood.reset(batch_size, prev_x=prev_x)
        self.planning = False
        self._batch_size = batch_size

    def set_prev_x(self, prev_x):
        self.cond_likelihood.set_prev_x(prev_x)

    def parameters(self):
        return self.cond_likelihood.parameters()
