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
    def __init__(self, likelihood_dist, n_variables, integration_window=None, constant_scale=True):
        super(ObservedVariable, self).__init__()
        self.distribution_type = getattr(torch.distributions, likelihood_dist)
        self.n_variables = n_variables
        self.integration_window = integration_window
        self.likelihood_dist = None
        self.likelihood_dist_pred = None
        self.planning_likelihood_dist = None
        self.likelihood_params = None
        if likelihood_dist in ['Bernoulli', 'Categorical']:
            # output the logits
            parameter_names = ['logits']
        else:
            parameter_names = list(self.distribution_type.arg_constraints.keys())
        if 'scale' in parameter_names and constant_scale:
            # constant log scale
            log_scale = nn.Parameter(torch.zeros(1, n_variables), requires_grad=True)
            # log_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.likelihood_params = nn.ParameterDict({'log_scale': log_scale})
            parameter_names.remove('scale')
        self.likelihood_models = nn.ModuleDict({name: None for name in parameter_names})
        self._log_scale_limits = [-15, 0]
        self._planning = False
        self._n_planning_samples = None

    def generate(self, input):
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
        if self.likelihood_params is not None:
            # TODO: currently only works for fully-connected outputs
            # output_shape = parameters['loc'].shape
            batch_size = parameters['loc'].shape[0]
            # log_scale = self.likelihood_params['log_scale'].repeat(output_shape)
            log_scale = self.likelihood_params['log_scale'].repeat(batch_size, 1)
            log_scale = torch.clamp(log_scale, self._log_scale_limits[0], self._log_scale_limits[1])
            parameters['scale'] = torch.exp(log_scale)
        # create a new distribution with the parameters
        if self._planning:
            self.planning_likelihood_dist = self.distribution_type(**parameters)
        else:
            self.likelihood_dist = self.distribution_type(**parameters)

    def sample(self):
        if self._planning:
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
        elif dist == 'planning_likelihood':
            d = self.planning_likelihood_dist
        else:
            raise NotImplementedError
        observation = self._change_device(observation)
        if d.has_enumerate_support:
            # probability mass function
            # expand for the number of samples
            batch_size = observation.shape[0]
            n_samples = int(d.logits.shape[0] / batch_size)
            observation = torch.cat(n_samples * [observation], dim=0)
            # factor = observation.new_ones(observation.shape) + observation * 10
            return d.log_prob(observation) # * factor
        else:
            # probability density function
            # expand for the number of samples
            batch_size = observation.shape[0]
            n_samples = int(d.loc.shape[0] / batch_size)
            observation = torch.cat(n_samples * [observation], dim=0)
            if len(observation.shape) > len(d.loc.shape):
                # convert image to fully-connected
                batch_size = observation.shape[0]
                observation = observation.contiguous().view(batch_size, -1)
            if self.integration_window is not None:
                if type(observation) != tuple:
                    # convert to tuple for integration
                    observation = (observation - self.integration_window/2, observation + self.integration_window/2)
                cll = torch.log(d.cdf(observation[1]) - d.cdf(observation[0]) + 1e-6)
            else:
                cll = d.log_prob(observation)
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
        # cll = self._cond_log_likelihood(observation).view(n_samples, -1, 1)
        # log_imp_weighted_cl = log_importance_weights + cll
        # mll = log_imp_weighted_cl.logsumexp(dim=0) - torch.tensor(float(n_samples)).log()
        # return mll
        return self._cond_log_likelihood(observation, dist='likelihood_pred').view(n_samples, -1, 1).mean(dim=0)

    def info_gain(self, observation=None, log_importance_weights=None, alpha=1.):
        # calculate the information gain from the conditional and marginal likelihoods
        observation = self._change_device(observation)
        n_samples = log_importance_weights.shape[0]
        cll = self.cond_log_likelihood(observation).view(n_samples, -1, 1)
        mll = self.marginal_log_likelihood(observation, log_importance_weights)
        ig = cll - alpha * mll.repeat(n_samples, 1).view(n_samples, -1, 1).detach()
        # ig = cll - alpha * mll.repeat(n_samples, 1).view(n_samples, -1, 1)
        return ig.mean(dim=0)
        # ig = cll.mean(dim=0) - mll
        # return ig

    def mutual_info(self, n_obs_samples=10):
        # estimate the mutual information between the internal state and the observed variable
        assert self._planning, 'Must be in planning mode to estimate MI.'
        # import ipdb; ipdb.set_trace()
        # estimate the conditional term
        #   evaluate entropy (average negative log prob)
        #   reshape into [n_state_samples, batch_size x n_planning_samples, 1]
        #   average over the state samples
        cond = self.planning_likelihood_dist.entropy().mul(-1)
        cond = cond.view(-1, self._batch_size * self._n_planning_samples, 1)
        cond = cond.mean(dim=0)

        # estimate the marginal term
        #   sample observations from the conditional likelihoods
        #   evaluate log probabilities from the same prior
        #   estimate marginal using logsumexp
        #   average over the observations samples
        #   average over the state samples
        obs_samples = self.planning_likelihood_dist.sample((n_obs_samples,))
        n_variables = obs_samples.shape[2]
        obs_samples = obs_samples.view(n_obs_samples, -1, self._batch_size * self._n_planning_samples, n_variables)
        n_state_samples = obs_samples.shape[1]
        obs_samples = obs_samples.repeat(1, n_state_samples, 1, 1)
        obs_samples = obs_samples.view(n_obs_samples * n_state_samples, -1, n_variables)
        cond_log_prob = self.planning_likelihood_dist.log_prob(obs_samples).sum(dim=2, keepdim=True)
        cond_log_prob = cond_log_prob.view(n_obs_samples, n_state_samples, n_state_samples, -1, 1)
        marg = cond_log_prob.logsumexp(dim=1) - torch.tensor(float(n_state_samples)).log()
        marg = marg.mean(dim=0).mean(dim=0)

        mi = cond - marg
        return mi

    def save_prediction(self):
        self.likelihood_dist_pred = self.likelihood_dist

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _change_device(self, observation):
        if type(observation) in [float, bool]:
            observation = torch.tensor(observation).to(torch.float32)
        if self.distribution_type == torch.distributions.Categorical:
            if len(observation.shape) == 2:
                observation = observation.view(-1)
            elif len(observation.shape) == 4:
                raise NotImplementedError
        if observation.device != self.device:
            observation = observation.to(self.device)
        return observation

    def reset(self, batch_size=1):
        self.likelihood_dist = None
        self.likelihood_dist_pred = None
        self.planning_likelihood_dist = None
        self._planning = False
        self._batch_size = batch_size

    def planning_mode(self, n_planning_samples=1):
        self._planning = True
        self.planning_likelihood_dist = None
        self._n_planning_samples = n_planning_samples

    def acting_mode(self):
        self._planning = False
        self.planning_likelihood_dist = None
        self._n_planning_samples = None

    def entropy(self):
        if self._planning:
            return self.planning_likelihood_dist.entropy()
        else:
            return self.likelihood_dist.entropy()
