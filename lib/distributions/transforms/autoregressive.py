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
    event_dim = 0
    def __init__(self, network_config):
        super(AutoregressiveTransform, self).__init__()
        self.shift = get_network(network_config)
        self.log_scale = get_network(network_config)
        # TODO: initialize to the identity transform

    def _call(self, x):
        """
        y = shift + scale * x
        """
        reshaped = False
        if len(x.shape) == 3:
            b, s, n = x.shape
            x = x.view(-1, x.shape[-1])
            reshaped = True
        shift, scale = self.shift(x), self.log_scale(x).exp()
        y = shift + scale * x
        self._cached_scale = scale
        if reshaped:
            y = y.view(b, s, n)
            self._cached_scale = self._cached_scale.view(b, s, n)
        return y

    def _inverse(self, y):
        """
        x = (y - shift) / scale
        """
        x = y.new_zeros(y.shape)
        for _ in range(x.shape[-1]):
            x = (y - self.shift(x)) / self.log_scale(x).exp()
        self._cached_scale = scale
        return x

    @property
    def sign(self):
        return self._cached_scale.sign()

    def log_abs_det_jacobian(self, x, y):
        return self._cached_scale.sum(dim=1).log()
