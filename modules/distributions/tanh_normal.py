import torch
from torch.distributions import constraints
from .transforms import TanhTransform
from torch.distributions import Normal, TransformedDistribution


class TanhNormal(TransformedDistribution):
    """
    Transform a Gaussian using tanh to ensure samples are in (-1, 1).
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.interval(-1., 1.)
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.base_dist = Normal(loc, scale)
        self.loc = self.base_dist.loc
        self.scale = self.base_dist.scale
        self.trans = [TanhTransform()]
        super(TanhNormal, self).__init__(self.base_dist, self.trans,
                                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.base_dist = Normal(new.loc, new.scale)
        new.trans = self.trans
        super(TanhNormal, new).__init__(new.base_dist, new.trans,
                                        validate_args=False)
        new._validate_args = self._validate_args
        return new
