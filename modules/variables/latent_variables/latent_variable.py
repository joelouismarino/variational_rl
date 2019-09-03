import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
from modules.distributions import Distribution


class LatentVariable(nn.Module):
    """
    A latent variable, with associated prior and approximate posterior distributions.

    Args:
        prior_dist (str): name of the prior distribution type (e.g. Normal)
        approx_post_dist (str or None): name of the approx. posterior distribution type (e.g. Normal)
        n_variables (int): number of variables
        n_input (list): the size of the inputs to the prior and approximate posterior respectively
        constant_prior (bool): whether to set the prior as a constant
        inference_type (str): direct or iterative
        norm_samples (bool): whether to layer normalize samples
    """
    def __init__(self, prior_dist, approx_post_dist, n_variables, n_input,
                 constant_prior=False, inference_type='direct', norm_samples=False):
        super(LatentVariable, self).__init__()
        self.prior = Distribution(prior_dist, n_variables, n_input[0], constant=constant_prior)

        self.approx_post = None
        if approx_post_dist is not None:
            self.approx_post = Distribution(approx_post_dist, n_variables, n_input[1], update=inference_type)

        self.layer_norm = None
        if norm_samples:
            self.layer_norm = nn.LayerNorm(n_variables)

        self.n_variables = n_variables
        self.planning = False
        self.reinitialized = True
        self.reset()

    def infer(self, input):
        """
        Infer the approximate posterior.

        Args:
            input (torch.Tensor):
        """
        self.approx_post.step(input)
        self.approx_post.retain_grads()
        self.reinitialized = False

    def step(self, input):
        """
        Step the prior.

        Args:
            input (torch.Tensor):
        """
        self.prior.step(input)

    def sample(self, n_samples=1):
        """
        Sample the latent variable.

        Args:
            n_samples (int): number of samples to draw
        """
        if self.planning:
            sample = self.prior.sample(n_samples)
        elif self.approx_post is not None:
            sample = self.approx_post.sample(n_samples)
        else:
            sample = self.prior.sample(n_samples)

        if self.layer_norm is not None:
            sample = self.layer_norm(sample)

        return sample

    def init_approx_post(self):
        """
        Initialize the approximate posterior from the prior.
        """
        if self.approx_post is not None:
            # if self.approx_post.update != 'direct':
            if True:
                parameters = {}
                prior_params = self.prior.get_dist_params()
                for param_name, param in prior_params.items():
                    parameters[param_name] = param.detach().clone().requires_grad_()
                self.approx_post.reset(batch_size=self.prior._batch_size,
                                       dist_params=parameters)
            else:
                self.approx_post.reset(batch_size=self.prior._batch_size)


    def kl_divergence(self, analytical=True):
        """
        Evaluate / estimate the KL divergence.

        Args:
            analytical (bool): whether to analytically evaluate the KL.
        """
        def _numerical_approx():
            # numerical approximation
            sample = self.approx_post.sample()
            return self.approx_post.log_prob(sample) - self.prior.log_prob(sample)

        if self.approx_post is not None:
            if analytical:
                try:
                    return torch.distributions.kl_divergence(self.approx_post.dist, self.prior.dist)
                except NotImplementedError:
                    return _numerical_approx()
            else:
                return _numerical_approx()
        else:
            sample = self.prior.sample()
            return sample.new_zeros(sample.shape)

    def log_importance_weights(self):
        """
        Get the log importance weights (prior / approx post).
        """
        # calculate importance weights for multiple samples
        sample = self.approx_post.sample(self.approx_post._n_samples)
        prior_log_prob = self.prior.log_prob(sample).sum(dim=2, keepdim=True)
        approx_post_log_prob = self.approx_post.log_prob(sample).sum(dim=2, keepdim=True)
        return prior_log_prob - approx_post_log_prob

    def params_and_grads(self, concat=False, normalize=True, norm_type='layer'):
        """
        Get the current distribution parameters and gradients for the approx post.

        Args:
            concat (bool): whether to concatenate the parameters and gradients
            normalize (bool): whether to normalize the parameters and gradients
            norm_type (str): the type of normalization (either batch or layer)
        """
        param_dict = self.approx_post.get_dist_params()
        grad_dict = self.approx_post.get_dist_param_grads()
        params = [param.detach() for _, param in param_dict.items()]
        grads = [grad.detach() for _, grad in grad_dict.items()]
        if normalize:
            norm_dim = -1
            if norm_type == 'batch':
                norm_dim = 0
            elif norm_type == 'layer':
                norm_dim = 1
            else:
                raise NotImplementedError
            for ind, grad in enumerate(grads):
                mean = grad.mean(dim=norm_dim, keepdim=True)
                std = grad.std(dim=norm_dim, keepdim=True)
                grads[ind] = (grad - mean) / (std + 1e-7)
            for ind, param in enumerate(params):
                mean = param.mean(dim=norm_dim, keepdim=True)
                std = param.std(dim=norm_dim, keepdim=True)
                params[ind] = (param - mean) / (std + 1e-7)
        if concat:
            return torch.cat(params + grads, dim=1)
        else:
            return torch.cat(params, dim=1), torch.cat(grads, dim=1)

    def inference_parameters(self):
        return nn.ParameterList() if self.approx_post is None else self.approx_post.parameters()

    def generative_parameters(self):
        return self.prior.parameters()

    def inference_mode(self):
        if self.approx_post is not None:
            self.approx_post.attach()

    def generative_mode(self):
        if self.approx_post is not None:
            self.approx_post.detach()

    def planning_mode(self, n_planning_samples):
        self.planning = True
        parameters = self.approx_post.get_dist_params()
        for param_name, param in parameters.items():
            param = param.requires_grad_()
            parameters[param_name] = param.repeat(n_planning_samples, 1)
        self.prior.planning_mode(parameters)

    def acting_mode(self):
        self.planning = False
        self.prior.acting_mode()

    def reset(self, batch_size=1):
        self.prior.reset(batch_size)
        self.init_approx_post()
        self.planning = False
        self.reinitialized = True
