import torch
from numbers import Number
from torch.distributions.utils import broadcast_all
from torch.distributions import Normal, Categorical, Independent, Distribution, constraints
from lib.distributions.mixture_same_family import MixtureSameFamily


class MixtureOfNormals(Distribution):
    """
    A mixture of a Normal distributions. Wrapper around MixtureSameFamily.

    Args:
        locs (torch.Tensor): the means of the Normals, shape [batch_size, n_components, dim]
        scales (torch.Tensor): the std. devs of the Normals, shape [batch_size, n_components, dim]
        weights (torch.Tensor): the weights of the components, shape [batch_size, n_components]
    """
    arg_constraints = {'locs': constraints.real,
                       'scales': constraints.positive,
                       'weights': constraints.interval(0, 1)}
    support = constraints.real
    has_rsample = False
    def __init__(self, locs, scales, weights, validate_args=None):
        self.locs, self.scales, self.weights = locs, scales, weights
        if isinstance(locs, Number) and isinstance(scales, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.locs.size()
        super(MixtureOfNormals, self).__init__(batch_shape, validate_args=validate_args)
        mixture_distribution = Categorical(self.weights)
        components_distribution = Independent(Normal(self.locs, self.scales), 1)
        self.dist = MixtureSameFamily(mixture_distribution, components_distribution)

    def log_prob(self, value):
        n_dims = value.shape[-1]
        lp = self.dist.log_prob(value)
        return torch.cat(n_dims * [lp.unsqueeze(-1)], dim=-1) / n_dims

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)

    def expand(self, batch_shape, _instance=None):
        # TODO: check for bugs in shape expansion
        new = self._get_checked_instance(MixtureOfNormals, _instance)
        batch_shape = torch.Size(batch_shape)
        new.locs = self.locs.expand(batch_shape)
        new.scales = self.scales.expand(batch_shape)
        new.weights = self.weights.expand(batch_shape)
        mixture_distribution = Categorical(new.weights)
        components_distribution = Independent(Normal(new.locs, new.scales), 1)
        new.dist = MixtureSameFamily(mixture_distribution, components_distribution)
        super(MixtureOfNormals, new).__init__(new.locs, new.scales, new.weights, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
