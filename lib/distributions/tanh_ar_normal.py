import torch
from torch.distributions import constraints, Normal
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

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhARNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.base_dist = Normal(new.loc, new.scale)
        new.trans = self.trans
        super(TanhARNormal, new).__init__(new.loc, new.scale, new.trans,
                                          validate_args=False)
        new._validate_args = self._validate_args
        return new
