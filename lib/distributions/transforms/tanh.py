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

    def __init__(self):
        super(TanhTransform, self).__init__()
        self._pretanh_value = None

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        self._pretanh_value = x
        finfo = torch.finfo(x.dtype)
        return torch.clamp(torch.tanh(x), min=-1 + finfo.eps, max=1. - finfo.eps)
        # return torch.tanh(x)

    def _inverse(self, y):
        # finfo = torch.finfo(y.dtype)
        # y = y.clamp(min=-1 + finfo.eps, max=1. - finfo.eps)
        # return 0.5 * torch.log((1 + y) / (1 - y) + finfo.eps)
        if self._pretanh_value is not None:
            try:
                return self._pretanh_value.view(y.shape)
            except:
                pass
        return 0.5 * torch.log((1 + y) / (1 - y) + 1e-6)

    def log_abs_det_jacobian(self, x, y):
        # finfo = torch.finfo(x.dtype)
        # return torch.log(1. - torch.tanh(x) ** 2 + finfo.eps)
        return torch.log(1. - torch.tanh(x) ** 2 + 1e-6)
