import torch
from numbers import Number
from torch.distributions.utils import broadcast_all
from torch.distributions import Normal, Uniform, Distribution, constraints


class NormalUniform(Distribution):
    """
    A mixture of a Normal distribution and a Uniform distribution, defined over
    the interval -1 to 1. Whatever probability mass left over from the Normal
    distribution (outside -1 to 1) is converted into a Uniform.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.interval(-1., 1.)
    has_rsample = False
    def __init__(self, loc, scale, validate_args=None):
        loc = torch.tanh(loc)
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(NormalUniform, self).__init__(batch_shape, validate_args=validate_args)
        self.normal = Normal(self.loc, self.scale)
        dev = self.normal.loc.device
        self.low = -torch.ones(batch_shape, device=dev)
        self.high = torch.ones(batch_shape, device=dev)
        self.uniform = Uniform(self.low, self.high)
        normal_prob = self.normal.cdf(torch.ones(batch_shape, device=dev)) - self.normal.cdf(-torch.ones(batch_shape, device=dev))
        self.uniform_factor = 1 - normal_prob

    def log_prob(self, value):
        normal_prob = self.normal.log_prob(value).exp()
        uniform_prob = self.uniform_factor * self.uniform.log_prob(value).exp()
        return (normal_prob + uniform_prob + torch.finfo(value.dtype).eps).log()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        normal_sample = self.normal.sample(sample_shape)
        uniform_sample = self.uniform.sample(sample_shape)
        # check for places where the normal sample is outside [-1, 1]
        # and replace with a uniform sample
        dist_flag = ((normal_sample > 1) + (normal_sample < -1)).float()
        sample = (1. - dist_flag) * normal_sample + dist_flag * uniform_sample
        return sample

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NormalUniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.normal = Normal(new.loc, new.scale)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        new.uniform = Uniform(new.low, new.high)
        dev = new.normal.loc.device
        normal_prob = new.normal.cdf(torch.ones(batch_shape, device=dev)) - new.normal.cdf(-torch.ones(batch_shape, device=dev))
        new.uniform_factor = 1 - normal_prob
        super(NormalUniform, new).__init__(new.low, new.scale, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
