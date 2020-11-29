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
            s, b, n = x.shape
            x = x.view(-1, x.shape[-1])
            reshaped = True
        shift, scale = self.shift(x), self.log_scale(x).exp().clamp(min=1e-6)
        y = shift + scale * x
        self._cached_scale = scale
        if reshaped:
            y = y.view(s, b, n)
            self._cached_scale = self._cached_scale.view(s, b, n)
        return y

    def _inverse(self, y):
        """
        x = (y - shift) / scale
        """
        reshaped = False
        if len(y.shape) == 3:
            s, b, n = y.shape
            y = y.view(-1, y.shape[-1])
            reshaped = True
        x = y.new_zeros(y.shape)
        for _ in range(x.shape[-1]):
            shift = self.shift(x)
            scale = self.log_scale(x).exp().clamp(min=1e-6)
            x = (y - shift) / scale
        self._cached_scale = scale
        if reshaped:
            x = x.view(s, b, n)
            self._cached_scale = self._cached_scale.view(s, b, n)
        return x

    @property
    def sign(self):
        return self._cached_scale.sign()

    def log_abs_det_jacobian(self, x, y):
        return self._cached_scale.log()
        # shape = x.shape
        # result = self._cached_scale.log()
        # result_size = result.size()[:-self.event_dim] + (-1,)
        # result = result.view(result_size).sum(-1)
        # shape = shape[:-self.event_dim]
        # return result.expand(shape)
        # return self._cached_scale.sum(dim=1).log()
