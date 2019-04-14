import torch
import torch.nn as nn
import numpy as np
import torch.distributions.constraints as constraints


class ObservedVariable(nn.Module):
    """
    A observed latent variable.

    Args:
        likelihood_dist (str): the name of the conditional likelihood distribution
        integration_window (optional, float): window over which to integrate the likelihood
    """
    def __init__(self, likelihood_dist, integration_window=1.):
        super(ObservedVariable, self).__init__()
        self.distribution_type = getattr(torch.distributions, likelihood_dist)
        self.integration_window = integration_window
        self.likelihood_dist = None
        self.likelihood_dist_pred = None
        self.planning_likelihood_dist = None
        self.likelihood_log_scale = None
        if likelihood_dist in ['Bernoulli', 'Categorical']:
            # output the logits
            parameter_names = ['logits']
        else:
            parameter_names = list(self.distribution_type.arg_constraints.keys())
        if 'scale' in parameter_names:
            # global log scale
            self.likelihood_log_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
            parameter_names.remove('scale')
        self.likelihood_models = nn.ModuleDict({name: None for name in parameter_names})

    def generate(self, input, planning=False):
        parameters = {}
        for parameter_name in self.likelihood_models:
            # calculate the value
            parameter_value = self.likelihood_models[parameter_name](input)
            # satisfy any constraints on the parameter value
            constraint = self.distribution_type.arg_constraints[parameter_name]
            if type(constraint) == constraints.greater_than:
                # positive value
                if constraint.lower_bound == 0:
                    parameter_value = torch.exp(parameter_value)
            elif constraint == constraints.simplex:
                # between 0 and 1
                parameter_value = nn.Softmax()(parameter_value)
            # set the parameter
            parameters[parameter_name] = parameter_value
        if self.likelihood_log_scale is not None:
            output_shape = parameters['loc'].shape
            log_scale = self.likelihood_log_scale.repeat(output_shape)
            log_scale = torch.clamp(log_scale, -15, 5)
            parameters['scale'] = torch.exp(log_scale)
        # create a new distribution with the parameters
        if planning:
            self.planning_likelihood_dist = self.distribution_type(**parameters)
        else:
            self.likelihood_dist = self.distribution_type(**parameters)

    def sample(self, planning=False):
        if planning:
            assert self.planning_likelihood_dist is not None
            sampling_dist = self.planning_likelihood_dist
        else:
            assert self.likelihood_dist is not None
            sampling_dist = self.likelihood_dist
        if self.distribution_type.has_rsample:
            return sampling_dist.rsample()
        else:
            return sampling_dist.sample()

    def _cond_log_likelihood(self, observation, dist='likelihood'):
        if dist == 'likelihood':
            d = self.likelihood_dist
        elif dist == 'likelihood_pred':
            d = self.likelihood_dist_pred
        else:
            raise NotImplementedError
        observation = self._change_device(observation)
        if d.has_enumerate_support:
            # probability mass function
            # expand for the number of samples
            batch_size = observation.shape[0]
            n_samples = int(d.logits.shape[0] / batch_size)
            observation = torch.cat(n_samples * [observation], dim=0)
            return d.log_prob(observation)
        else:
            # TODO: remove requirement for integration window
            # probability density function
            # expand for the number of samples
            batch_size = observation.shape[0]
            n_samples = int(d.loc.shape[0] / batch_size)
            observation = torch.cat(n_samples * [observation], dim=0)
            if len(observation.shape) > len(d.loc.shape):
                # convert image to fully-connected
                batch_size = observation.shape[0]
                observation = observation.contiguous().view(batch_size, -1)
            if type(observation) != tuple:
                # convert to tuple for integration
                observation = (observation - self.integration_window/2, observation + self.integration_window/2)

            cll = torch.log(d.cdf(observation[1]) - d.cdf(observation[0]) + 1e-6)
            if len(cll.shape) > 2:
                # if image, sum over height and width
                cll = cll.sum(dim=(2,3))
            return cll.sum(dim=1, keepdim=True)

    def cond_log_likelihood(self, observation):
        # observation = self._change_device(observation)
        # if self.likelihood_dist.has_enumerate_support:
        #     # probability mass function
        #     return self.likelihood_dist.log_prob(observation)
        # else:
        #     # probability density function
        #     if len(observation.shape) > len(self.likelihood_dist.loc.shape):
        #         # convert image to fully-connected
        #         batch_size = observation.shape[0]
        #         observation = observation.contiguous().view(batch_size, -1)
        #     if type(observation) != tuple:
        #         # convert to tuple for integration
        #         observation = (observation - self.integration_window/2, observation + self.integration_window/2)
        #
        #     cll = torch.log(self.likelihood_dist.cdf(observation[1]) - self.likelihood_dist.cdf(observation[0]) + 1e-6)
        #     if len(cll.shape) > 2:
        #         # if image, sum over height and width
        #         cll = cll.sum(dim=(2,3))
        #     return cll
        cll = self._cond_log_likelihood(observation)
        # TODO: average over samples
        return cll

    def marginal_log_likelihood(self, observation, log_importance_weights):
        # estimate the marginal log likelihood
        n_samples = log_importance_weights.shape[0]
        cll = self._cond_log_likelihood(observation).view(n_samples, -1, 1)
        log_imp_weighted_cl = log_importance_weights + cll
        mll = log_imp_weighted_cl.logsumexp(dim=0) - torch.tensor(float(n_samples)).log()
        return mll
        # return self._cond_log_likelihood(observation, dist='likelihood_pred').view(n_samples, -1, 1).mean(dim=0)

    def info_gain(self, observation, log_importance_weights, alpha):
        # calculate the information gain from the conditional and marginal likelihoods
        observation = self._change_device(observation)
        n_samples = log_importance_weights.shape[0]
        cll = self.cond_log_likelihood(observation).view(n_samples, -1, 1)
        mll = self.marginal_log_likelihood(observation, log_importance_weights)
        ig = cll - alpha * mll.repeat(n_samples, 1).view(n_samples, -1, 1)
        return ig.mean(dim=0)
        # ig = cll.mean(dim=0) - mll
        # return ig

    def save_prediction(self):
        self.likelihood_dist_pred = self.likelihood_dist

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _change_device(self, observation):
        if type(observation) in [float, bool]:
            observation = torch.tensor(observation).to(torch.float32)
        if observation.device != self.device:
            observation = observation.to(self.device)
        return observation

    def reset(self):
        self.likelihood_dist = None
        self.likelihood_dist_pred = None
        self.planning_likelihood_dist = None

    def entropy(self, planning=False):
        if planning:
            return self.planning_likelihood_dist.entropy()
        else:
            return self.likelihood_dist.entropy()
