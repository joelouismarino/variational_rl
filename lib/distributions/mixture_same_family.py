import torch
from torch import distributions as D
from torch.nn import functional as F


class MixtureSameFamily(D.Distribution):
    """ Mixture (same-family) distribution.
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all components are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` components) and a components
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.
    """

    def __init__(self,
                 mixture_distribution,
                 components_distribution,
                 validate_args=None):
        """ Construct a 'MixtureSameFamily' distribution
        Args::
            mixture_distribution: `torch.distributions.Categorical`-like
                instance. Manages the probability of selecting components.
                The number of categories must match the rightmost batch
                dimension of the `components_distribution`. Must have either
                scalar `batch_shape` or `batch_shape` matching
                `components_distribution.batch_shape[:-1]`
            components_distribution: `torch.distributions.Distribution`-like
                instance. Right-most batch dimension indexes components.
        Examples::
            # Construct Gaussian Mixture Model in 1D consisting of 5 equally
            # weighted normal distributions
            >>> mix = D.Categorical(torch.ones(5,))
            >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
            >>> gmm = MixtureSameFamily(mix, comp)
            # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
            # weighted bivariate normal distributions
            >>> mix = D.Categorical(torch.ones(5,))
            >>> comp = D.Independent(D.Normal(
                    torch.randn(5,2), torch.rand(5,2)), 1)
            >>> gmm = MixtureSameFamily(mix, comp)
            # Construct a batch of 3 Gaussian Mixture Models in 2D each
            # consisting of 5 random weighted bivariate normal distributions
            >>> mix = D.Categorical(torch.rand(3,5))
            >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
            >>> gmm = MixtureSameFamily(mix, comp)
        """
        self._mixture_distribution = mixture_distribution
        self._components_distribution = components_distribution

        if not isinstance(self._mixture_distribution, D.Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._components_distribution, D.Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._components_distribution.batch_shape[:-1]
        if len(mdbs) != 0 and mdbs != cdbs:
            raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                             "compatible with `components_distribution."
                             "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture components matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._components_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution components` ({0}) does not"
                             " equal `components_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_components = km

        event_shape = self._components_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def components_distribution(self):
        return self._components_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.components_distribution.mean,
                         dim=-1-self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs*self.components_distribution.variance,
                                  dim=-1-self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.components_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1-self._event_ndims)
        return mean_cond_var + var_cond_mean

    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            # [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            # [n, B, k, E]
            comp_sample = self.components_distribution.sample(sample_shape)
            # [n, B, k]
            mask = F.one_hot(mix_sample, self._num_components)
            # [n, B, k, [1]*E]
            mask = self._pad_mixture_dimensions(mask)
            return torch.sum(comp_sample * mask.float(),
                             dim=-1-self._event_ndims)

    def _pad(self, x):
        d = len(x.shape) - self._event_ndims
        s = x.shape
        x = x.reshape(*s[:d], 1, *s[d:])
        return x

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        s = x.shape
        x = torch.reshape(x, shape=(*s[:-1], *(pad_ndims*[1]),
                                    *s[-1:], *(self._event_ndims*[1])))
        return x
