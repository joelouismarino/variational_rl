import torch


def kl_divergence(dist1, dist2, analytical=True, n_samples=1, sample=None):
    """
    Evaluate / estimate the KL divergence.

    Args:
        dist1 (Distribution): the first distribution
        dist2 (Distribution): the second distribution
        analytical (bool): whether to analytically evaluate the KL
        n_samples (int): number of samples for non-analytical KL
        sample (torch.tensor): sample at which to evaluate; used for Boltzmann d1
    """
    def numerical_approx(d1, d2, n_s):
        # numerical approximation
        try:
            s = d1.sample(n_s)
            w = 1.
        except NotImplementedError:
            # kl using Boltzmann approximate posterior
            if sample is not None:
                s = sample
            else:
                s = d2.sample(n_s)
            w = d1.dist.get_weights()
            w = w.view(n_s, -1, 1).repeat(1, 1, s.shape[-1])
            # if s.shape[0] > n_s:
            #     import ipdb; ipdb.set_trace()
        kl = d1.log_prob(s, non_planning=True) - d2.log_prob(s, non_planning=True)
        kl = w * kl
        kl = kl.view(n_s, -1, kl.shape[-1])
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
