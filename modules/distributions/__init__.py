import torch
from .distribution import Distribution


def kl_divergence(dist1, dist2, analytical=True, n_samples=1):
    """
    Evaluate / estimate the KL divergence.

    Args:
        dist1 (Distribution): the first distribution
        dist2 (Distribution): the second distribution
        analytical (bool): whether to analytically evaluate the KL
        n_samples (int): number of samples for non-analytical KL
    """
    def numerical_approx(d1, d2, n_s):
        # numerical approximation
        sample = d1.sample(n_s)
        kl = d1.log_prob(sample, non_planning=True) - d2.log_prob(sample, non_planning=True)
        kl = kl.view(n_samples, -1, kl.shape[-1])
        return kl.mean(dim=0)

    if dist1 is not None:
        if analytical:
            try:
                return torch.distributions.kl_divergence(dist1.dist, dist2.dist)
            except NotImplementedError:
                return numerical_approx(dist1, dist2, n_samples)
        else:
            return numerical_approx(dist1, dist2, n_samples)
    else:
        sample = dist2.sample()
        return sample.new_zeros(sample.shape)
