import torch
import torch.nn as nn
import numpy as np
from modules.distributions import Distribution
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
    def __init__(self, likelihood_dist, n_variables, n_input, constant_scale,
                 residual_loc, manual_loc, manual_loc_alpha):
        super(ObservedVariable, self).__init__()
        self.cond_likelihood = Distribution(likelihood_dist, n_variables, n_input,
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

    def marg_log_likelihood(self, x, log_iw):
        """
        Estimate the log marginal likelihood of the value x.

        Args:
            x (torch.Tensor): [batch_size, n_variables], the point of evaluation
            log_iw (torch.Tensor): [n_samples, batch_size, 1] the log importance weights
        """
        n_samples = log_iw.shape[0]
        cll = self.cond_log_likelihood(x).view(n_samples, -1, 1)
        log_iw_cl = log_iw + cll
        mll = log_iw_cl.logsumexp(dim=0) - torch.tensor(float(n_samples)).log()
        return mll

    def info_gain(self, x, log_iw, marg_factor=1.):
        """
        Estimate the information gain of the value x.

        Args:
            x (torch.Tensor): [btach_size, n_variables], the point of evaluation
            log_iw (torch.Tensor): [n_samples, batch_size, 1] the log importance weights
            marg_factor (float): the (annealed) factor for the marginal likelihood
        """
        n_samples = log_iw.shape[0]
        cll = self.cond_log_likelihood(x).view(n_samples, -1, 1)
        mll = self.marg_log_likelihood(x, log_iw)
        ig = cll - marg_factor * mll.repeat(n_samples, 1).view(n_samples, -1, 1)
        return ig.mean(dim=0)

    def mutual_info(self, n_obs_samples=10):
        """
        Estimate the mutual information.

        Args:
            n_obs_samples (int): number of observations to sample
        """
        # estimate the mutual information between the internal state and the observed variable
        assert self.planning, 'Must be in planning mode to estimate MI.'
        # estimate the conditional term
        #   evaluate entropy (average negative log prob)
        #   reshape into [n_state_samples, batch_size x n_planning_samples, 1]
        #   average over the state samples
        cond = self.cond_likelihood.entropy().mul(-1)
        cond = cond.view(-1, self._batch_size * self._n_planning_samples, 1)
        cond = cond.mean(dim=0)

        # estimate the marginal term
        #   sample observations from the conditional likelihoods
        #   evaluate log probabilities from the same prior
        #   estimate marginal using logsumexp
        #   average over the observations samples
        #   average over the state samples
        obs_samples = self.cond_likelihood.sample((n_obs_samples,))
        n_variables = obs_samples.shape[2]
        obs_samples = obs_samples.view(n_obs_samples, -1, self._batch_size * self._n_planning_samples, n_variables)
        n_state_samples = obs_samples.shape[1]
        obs_samples = obs_samples.repeat(1, n_state_samples, 1, 1)
        obs_samples = obs_samples.view(n_obs_samples * n_state_samples, -1, n_variables)
        cond_log_prob = self.cond_likelihood.log_prob(obs_samples).sum(dim=2, keepdim=True)
        cond_log_prob = cond_log_prob.view(n_obs_samples, n_state_samples, n_state_samples, -1, 1)
        marg = cond_log_prob.logsumexp(dim=1) - torch.tensor(float(n_state_samples)).log()
        marg = marg.mean(dim=0).mean(dim=0)

        mi = cond - marg
        return mi

    def planning_mode(self, n_planning_samples=1):
        self.planning = True
        self.cond_likelihood.planning_mode()
        self._n_planning_samples = n_planning_samples

    def acting_mode(self):
        self.planning = False
        self.cond_likelihood.acting_mode()
        self._n_planning_samples = 1

    def reset(self, batch_size=1, prev_obs=None):
        self.cond_likelihood.reset(batch_size, prev_obs=prev_obs)
        self.planning = False
        self._batch_size = batch_size

    def parameters(self):
        return self.cond_likelihood.parameters()
