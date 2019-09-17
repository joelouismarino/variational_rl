import torch
from .distribution import Distribution


def kl_divergence(dist1, dist2, analytical=True):
    """
    Evaluate / estimate the KL divergence.

    Args:
        dist1 (Distribution): the first distribution
        dist2 (Distribution): the second distribution
        analytical (bool): whether to analytically evaluate the KL.
    """
    def _numerical_approx():
        # numerical approximation
        sample = dist1.sample()
        return dist1.log_prob(sample, non_planning=True) - dist2.log_prob(sample, non_planning=True)

    if dist1 is not None:
        if analytical:
            try:
                return torch.distributions.kl_divergence(dist1.dist, dist2.dist)
            except NotImplementedError:
                return _numerical_approx()
        else:
            return _numerical_approx()
    else:
        sample = dist2.sample()
        return sample.new_zeros(sample.shape)
