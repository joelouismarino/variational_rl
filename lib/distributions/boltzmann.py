import torch
from torch.distributions import Distribution, constraints


class Boltzmann(Distribution):
    """
    A (non-parametric) Boltzmann distribution. Weights the prior probabilities
    by the normalized exponentiated action-value (Q) function.

    Args:
        prior_log_probs (torch.tensor): probabilities of actions drawn from the prior
                                        distribution [action_samples, batch_size, 1]
        q_values (torch.tensor):        q values for the corresponding actions drawn
                                        from the prior [action_samples, batch_size, 1]
        temperature (torch.tensor):     the 'temperature' parameter [1]
        actions (torch.tensor):         the corresponding actions that were used
                                        to calculate the distribution
                                        [action_samples, batch_size, action_size]
    """
    arg_constraints = {'prior_log_probs': constraints.real,
                       'q_values': constraints.real,
                       'temperature': constraints.positive,
                       'actions': constraints.real}
    support = constraints.real
    has_rsample = False
    def __init__(self, prior_log_probs, q_values, temperature, actions,
                 validate_args=None):
        batch_shape = prior_log_probs.size()
        self.prior_log_probs = prior_log_probs
        self.q_values = q_values
        self.temperature = temperature
        self.actions = actions
        # calculate the (Boltzmann) state values
        n_action_samples = self.q_values.shape[0]
        values = torch.logsumexp(self.q_values / self.temperature, dim=0, keepdim=True)
        values = values - torch.tensor(n_action_samples, dtype=torch.float32, device=temperature.device).log()
        self.values = self.temperature * values
        # calculate the advantages
        self.advantages = self.q_values - self.values
        super(Boltzmann, self).__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        action_dim = x.shape[-1]
        log_probs = self.prior_log_probs + self.advantages / self.temperature
        # if x.shape[0] != self.actions.shape[0]:
        #     if x == self.actions[x.shape[0]]:
        #         # evaluate subset of actions
        #         log_probs = log_probs[:x.shape[0]]
        #     else:
        #         # cannot evaluate actions we didn't originally sample
        #         raise NotImplementedError
        return log_probs.repeat(1, 1, action_dim) / action_dim

    def get_weights(self):
        return torch.exp(self.advantages / self.temperature)

    def sample(self, sample_shape=torch.Size()):
        # Note: these are the prior actions, not from the approx. posterior
        if sample_shape[0] > self.actions.shape[0]:
            raise NotImplementedError
        return self.actions[:sample_shape[0]]

    def expand(self, batch_shape, _instance=None):
        return self

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
