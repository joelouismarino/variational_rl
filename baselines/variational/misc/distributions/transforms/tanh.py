import torch
from torch.distributions import constraints
from torch.distributions import Transform


class TanhTransform(Transform):
    """
    Transform via tanh().
    """
    domain = constraints.real
    codomain = constraints.interval(-1., 1.)
    bijective = True

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        finfo = torch.finfo(x.dtype)
        return torch.clamp(torch.tanh(x), min=-1 + finfo.eps, max=1. - finfo.eps)

    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=-1 + finfo.eps, max=1. - finfo.eps)
        return 0.5 * torch.log((1 + y) / (1 - y) + finfo.eps)

    def log_abs_det_jacobian(self, x, y):
        finfo = torch.finfo(x.dtype)
        return torch.log(1. - torch.tanh(x) ** 2 + finfo.eps)
