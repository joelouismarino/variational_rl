import torch
import torch.nn as nn
from torch.distributions import constraints
from .transform_module import TransformModule
from lib.networks import get_network


class ReverseTransform(TransformModule):
    """
    A reverse transform.
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real
    event_dim = 0
    def __init__(self):
        super(ReverseTransform, self).__init__()

    def _call(self, x):
        """
        y = reverse(x)
        """
        return x.flip(-1)

    def _inverse(self, y):
        """
        x = reverse(y)
        """
        return y.flip(-1)

    @property
    def sign(self):
        return 1

    def log_abs_det_jacobian(self, x, y):
        return 0
