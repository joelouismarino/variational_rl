import torch
import torch.nn as nn
import torch.distributions.constraints as constraints


class ObservedVariable(nn.Module):

    def __init__(self, likelihood_dist):
        super(ObservedVariable, self).__init__()
        self.distribution_type = getattr(torch.distributions, likelihood_dist)
        parameter_names = self.distribution_type.arg_constraints.keys()
        self.likelihood_models = nn.ModuleDict({name: None for name in parameter_names})
        self.likelihood_dist = None

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
        # create a new distribution with the parameters
        self.likelihood_dist = self.distribution_type(**parameters)

    def sample(self):
        assert self.likelihood_dist is not None
        if self.distribution_type.has_rsample:
            return self.likelihood_dist.rsample()
        else:
            return self.likelihood_dist.sample()

    def cond_log_likelihood(self, observation):
        if self.likelihood_dist.has_enumerate_support:
            # probability mass function
            return self.likelihood_dist.log_prob(observation)
        else:
            # probability density function
            return torch.log(self.likelihood_dist.cdf(observation[1]) - self.likelihood_dist.cdf(observation[0]) + 1e-6)

    def reset(self):
        self.likelihood_dist = None

    def entropy(self):
        return self.likelihood_dist.entropy()
