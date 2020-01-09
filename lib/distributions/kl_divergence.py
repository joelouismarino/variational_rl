import torch
from .boltzmann import Boltzmann


def kl_divergence(dist1, dist2, n_samples=1, sample=None):
    """
    Evaluate / estimate the KL divergence.

    Args:
        dist1 (Distribution): the first distribution
        dist2 (Distribution): the second distribution
        n_samples (int): number of samples for non-analytical KL
        sample (torch.tensor): sample at which to evaluate; used for Boltzmann d1
    """
    def numerical_approx(d1, d2, n_s):
        # numerical approximation
        if d1.dist_type == Boltzmann:
            # kl using Boltzmann approximate posterior
            s = sample
            w = d1.dist.get_weights()
            w = w.view(n_s, -1, 1).repeat(1, 1, s.shape[-1])
        else:
            w = 1.

            s = sample if sample is not None else d1.sample(n_s)
        kl = d1.log_prob(s, non_planning=True) - d2.log_prob(s, non_planning=True)
        kl = w * kl
        # NOTE: this n_s might not be the same as the number of samples, in which
        #       case the view will be wrong...
        kl = kl.view(n_s, -1, kl.shape[-1])
        return kl.mean(dim=0)

    try:
        kl = torch.distributions.kl_divergence(dist1.dist, dist2.dist)
        # if sample is not None:
        #     n_samples = int(sample.shape[0] / kl.shape[0])
        # return kl.repeat(n_samples, 1)
        return kl
    except NotImplementedError:
        return numerical_approx(dist1, dist2, n_samples)
