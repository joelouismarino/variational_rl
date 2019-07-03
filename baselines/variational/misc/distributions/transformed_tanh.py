from torch.distributions import constraints
from .transforms import TanhTransform
from torch.distributions import Normal, TransformedDistribution


class TransformedTanh(TransformedDistribution):
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
        super(TransformedTanh, self).__init__(self.base_dist, self.trans,
                                              validate_args=validate_args)
