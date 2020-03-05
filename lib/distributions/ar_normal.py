import torch
from torch.distributions import constraints
from .transforms import AutoregressiveTransform
from torch.distributions import Normal, TransformedDistribution


class ARNormal(TransformedDistribution):
    """
    Transformed distribution that transforms a Normal distribution using
    inverse autoregressive flow.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    def __init__(self, loc, scale, transforms, validate_args=None):
        self.trans = transforms
        super(ARNormal, self).__init__(Normal(loc, scale), transforms,
                                       validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ARNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.base_dist = Normal(new.loc, new.scale)
        new.trans = self.trans
        super(ARNormal, new).__init__(new.base_dist, new.trans,
                                      validate_args=False)
        new._validate_args = self._validate_args
        return new
