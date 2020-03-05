from torch.distributions import constraints
from .ar_normal import ARNormal
from .transforms import TanhTransform


class TanhARNormal(ARNormal):
    """
    Auto-regressive transformed Normal distribution with final tanh transform.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.interval(-1., 1.)
    has_rsample = True

    def __init__(self, loc, scale, transforms, validate_args=None):
        transforms.append(TanhTransform())
        super(TanhARNormal, self).__init__(loc, scale, transforms, validate_args)
