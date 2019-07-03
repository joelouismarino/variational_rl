import torch
import numpy as np
from numbers import Number
from torch.distributions import constraints
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all


class Delta(Distribution):

    arg_constraints = {'loc': constraints.real}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    def __init__(self, loc, validate_args=None):
        self.loc = loc # broadcast_all(loc)
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Delta, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return self.loc.expand(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if (value == self.loc).all().item():
            return 0.
        else:
            return torch.tensor(np.inf)
