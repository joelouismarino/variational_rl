import torch
import torch.nn as nn
from torch.distributions import constraints
from .transform_module import TransformModule
from lib.networks import get_network


class AutoregressiveTransform(TransformModule):
    """
    An affine (inverse) autoregressive transform. See Kingma et al.
    [arXiv: 1606.04934].

    Args:
        network_config (dict): configuration for networks
        constant_scale (bool): whether to use a constant scale
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, network_config):
        super(AutoregressiveTransform, self).__init__()
        self.m = get_network(network_config)
        self.s = get_network(network_config)
        

    def _call(self, x):
        """
        y = shift + scale * x
        """
        shift, scale = self.m(x), self.s(x).exp()
        y = shift + scale * x
        self._cached_scale = scale
        return y

    def _inverse(self, y):
        """
        x = (y - shift) / scale
        """
        x = y.new_zeros(y.shape)
        for _ in range(x.shape[-1]):
            shift = self.m(x)
            scale = self.s(x).exp()
            x = (y - shift) / scale
        self._cached_scale = scale
        return x

    @property
    def sign(self):
        return self._cached_scale.sign()

    def log_abs_det_jacobian(self, x, y):
        return self._cached_scale.sum(dim=1).log()
