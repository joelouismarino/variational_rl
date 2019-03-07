import torch
import torch.nn as nn
import numpy as np
import torch.distributions.constraints as constraints


class ObservedVariable(nn.Module):

    def __init__(self, likelihood_dist, integration_window=1.):
        super(ObservedVariable, self).__init__()
        self.distribution_type = getattr(torch.distributions, likelihood_dist)
        self.integration_window = integration_window
        self.likelihood_dist = None
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
        if self.likelihood_log_scale is not None:
            output_shape = parameters['loc'].shape
            log_scale = self.likelihood_log_scale.repeat(output_shape)
            log_scale = torch.clamp(log_scale, -15, 5)
            parameters['scale'] = torch.exp(log_scale)
        # create a new distribution with the parameters
        self.likelihood_dist = self.distribution_type(**parameters)

    def sample(self):
        assert self.likelihood_dist is not None
        if self.distribution_type.has_rsample:
            return self.likelihood_dist.rsample()
        else:
            return self.likelihood_dist.sample()

    def cond_log_likelihood(self, observation):
        observation = self._change_device(observation)
        if self.likelihood_dist.has_enumerate_support:
            # probability mass function
            return self.likelihood_dist.log_prob(observation)
        else:
            # probability density function
            if len(observation.shape) > len(self.likelihood_dist.loc.shape):
                # convert image to fully-connected
                batch_size = observation.shape[0]
                observation = observation.contiguous().view(batch_size, -1)
            if type(observation) != tuple:
                # convert to tuple for integration
                observation = (observation - self.integration_window/2, observation + self.integration_window/2)

            cll = torch.log(self.likelihood_dist.cdf(observation[1]) - self.likelihood_dist.cdf(observation[0]) + 1e-6)
            if len(cll.shape) > 2:
                # if image, sum over height and width
                cll = cll.sum(dim=(2,3))
            return cll

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

    def entropy(self):
        return self.likelihood_dist.entropy()
