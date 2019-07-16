import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
from misc.distributions import Delta, TransformedTanh


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
        const_scale (bool): whether to use a constant (learnable) approx. post. variance
        norm_samples (bool): whether to normalize samples
    """
    def __init__(self, prior_dist, approx_post_dist, n_variables, n_input,
                 constant_prior=False, inference_type='direct',
                 const_scale=False, norm_samples=False):
        super(LatentVariable, self).__init__()
        self.n_variables = n_variables
        self.constant_prior = constant_prior
        self.inference_type = inference_type
        self.const_scale = const_scale
        self.norm_samples = norm_samples

        if prior_dist == 'Delta':
            self.prior_dist_type = Delta
        elif approx_post_dist == 'TransformedTanh':
            self.prior_dist_type = TransformedTanh
        else:
            self.prior_dist_type = getattr(torch.distributions, prior_dist)
        self.approx_post_dist_type = None
        if approx_post_dist is not None:
            if approx_post_dist == 'Delta':
                self.approx_post_dist_type = Delta
            elif approx_post_dist == 'TransformedTanh':
                self.approx_post_dist_type = TransformedTanh
            else:
                self.approx_post_dist_type = getattr(torch.distributions, approx_post_dist)

        # prior
        if prior_dist == 'Categorical':
            prior_param_names = ['logits']
        else:
            prior_param_names = self.prior_dist_type.arg_constraints.keys()
        self.prior_models = None
        if not self.constant_prior:
            # learned, conditional prior
            self.prior_models = nn.ModuleDict({name: None for name in prior_param_names})

        # approximate posterior
        if approx_post_dist is not None:
            # amortized approximate posterior
            if approx_post_dist == 'Categorical':
                approx_post_param_names = ['logits']
            else:
                approx_post_param_names = self.approx_post_dist_type.arg_constraints.keys()
            self.approx_post_log_scale = None
            if self.const_scale:
                # use a constant variance
                if 'scale' in approx_post_param_names:
                    self.approx_post_log_scale = nn.Parameter(torch.ones(1, n_variables))
                    approx_post_param_names = ['loc']
            self.approx_post_models = nn.ModuleDict({name: None for name in approx_post_param_names})
            if self.inference_type != 'direct':
                self.approx_post_gates = nn.ModuleDict({name: None for name in approx_post_param_names})

        # initialize the prior
        self.initial_prior_params = nn.ParameterDict({name: None for name in prior_param_names})

        for param in self.initial_prior_params:
            constraint = self.prior_dist_type.arg_constraints[param]
            if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                self.initial_prior_params[param] = nn.Parameter(torch.ones(1, n_variables))
            else:
                self.initial_prior_params[param] = nn.Parameter(torch.zeros(1, n_variables))

        # TODO: we will need to reshape the initial parameters
        self.prior_dist = None
        self.approx_post_dist = None
        self.planning_prior_dist = None
        self.reinitialized = True
        self.reset()

        self._sample = None
        self._n_samples = None
        self._planning_sample = None
        self._planning = False
        self._detach_latent = True
        self.layer_norm = None
        if self.norm_samples:
            self.layer_norm = nn.LayerNorm(self.n_variables)
        self._log_var_limits = [-15, 0]

    def infer(self, input):
        if self.approx_post_dist_type is not None:
            # infer the approximate posterior
            parameters = {}
            for param_name in self.approx_post_models:
                # calculate the parameter update and the gate
                param_update = self.approx_post_models[param_name](input)
                if self.inference_type != 'direct':
                    param_gate = self.approx_post_gates[param_name](input)

                # update the parameter value
                param = getattr(self.approx_post_dist, param_name).detach()
                constraint = self.approx_post_dist.arg_constraints[param_name]
                if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                    # convert to log-space (for scale parameters)
                    param = torch.log(param)

                if self.inference_type == 'direct':
                    param = param_update
                else:
                    param = param_gate * param + (1. - param_gate) * param_update

                # satisfy any constraints on the parameter value
                if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                    # positive value
                    param = torch.clamp(param, self._log_var_limits[0], self._log_var_limits[1])
                    param = torch.exp(param)
                elif constraint == constraints.simplex:
                    # between 0 and 1 for probabilities
                    param = nn.Softmax()(param)

                # set the parameter
                parameters[param_name] = param

            if self.const_scale:
                log_scale = self.approx_post_log_scale.repeat(input.shape[0], 1)
                log_scale = torch.clamp(log_scale, self._log_var_limits[0], self._log_var_limits[1])
                scale = torch.exp(log_scale)
                parameters['scale'] = scale

            # create a new distribution with the parameters
            self.approx_post_dist = self.approx_post_dist_type(**parameters)
            # retain the gradient for further inference
            for param_name in self.approx_post_models:
                getattr(self.approx_post_dist, param_name).retain_grad()
            self._sample = None
            self.reinitialized = False

    def sample(self, n_samples=1):
        # sample the latent variable
        # TODO: handle re-sampling if we don't have enough samples already
        if (self._sample is None and not self._planning) or (self._planning_sample is None and self._planning):
            if self._planning:
                sampling_dist = self.planning_prior_dist
                sampling_dist_type = self.prior_dist_type
            else:
                # sample from the approximate posterior if available
                sampling_dist = self.approx_post_dist
                sampling_dist_type = self.approx_post_dist_type
                if sampling_dist is None:
                    sampling_dist = self.prior_dist
                    sampling_dist_type = self.prior_dist_type
            if sampling_dist.has_rsample:
                batch_size = sampling_dist.loc.shape[0]
                sample = sampling_dist.rsample([n_samples])
                sample = sample.view(batch_size * n_samples, -1)
                if self.norm_samples:
                    sample = self.layer_norm(sample)

            else:
                batch_size = sampling_dist.logits.shape[0]
                sample = sampling_dist.sample([n_samples])
                sample = sample.view(batch_size * n_samples, 1)
            self._n_samples = n_samples
            # TODO: this will only work for fully-connected variables

            if sampling_dist_type == getattr(torch.distributions, 'Categorical'):
                # convert to one-hot encoding
                device = self.initial_prior_params['logits'].device
                one_hot_sample = torch.zeros(sample.shape[0], self.n_variables).to(device)
                one_hot_sample.scatter_(1, sample, 1.)
                sample = one_hot_sample
            if self._planning:
                self._planning_sample = sample
            else:
                self._sample = sample
        if self._planning:
            sample = self._planning_sample
        else:
            sample = self._sample
        if self._detach_latent:
            sample = sample.detach()
        if n_samples < self._n_samples:
            sample = sample.view(self._n_samples, -1, self.n_variables)
            sample = sample[:n_samples].view(-1, self.n_variables)
        return sample

    def step(self, input):
        if not self.constant_prior:
            # set the prior
            parameters = {}
            for param_name in self.prior_models:
                # calculate the value
                param = self.prior_models[param_name](input)
                # satisfy any constraints on the parameter value
                constraint = self.prior_dist.arg_constraints[param_name]
                if type(constraint) == constraints.greater_than:
                    # positive value
                    if constraint.lower_bound == 0:
                        param = torch.clamp(param, self._log_var_limits[0], self._log_var_limits[1])
                        param = torch.exp(param)
                elif constraint == constraints.simplex:
                    # between 0 and 1
                    param = nn.Softmax()(param)
                # set the parameter
                parameters[param_name] = param
            # create a new distribution with the parameters
            new_dist = self.prior_dist_type(**parameters)
            if self._planning:
                self.planning_prior_dist = new_dist
                self._planning_sample = None
            else:
                self.prior_dist = new_dist
                self.reinitialized = False
                self._sample = None

    def init_approx_post(self):
        if self.approx_post_dist_type is not None:
            # initialize the approximate posterior from the prior
            # copies over a detached version of each parameter
            if self.approx_post_dist_type == Delta:
                params = ['loc']
            else:
                assert self.prior_dist_type == self.approx_post_dist_type, 'Only currently support same type.'
                params = self.initial_prior_params
            parameters = {}
            for parameter_name in params:
                parameters[parameter_name] = getattr(self.prior_dist, parameter_name).detach().requires_grad_()
            self.approx_post_dist = self.approx_post_dist_type(**parameters)
            if self.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
                self.approx_post_dist.logits = self.prior_dist.logits.detach().requires_grad_()
            self._sample = None

    def planning_mode(self, n_planning_samples=1):
        # initialize the planning prior from the approximate posterior
        # copies over a detached version of each parameter
        parameters = {}
        for parameter_name in self.initial_prior_params:
            parameter = getattr(self.approx_post_dist, parameter_name).requires_grad_()
            parameters[parameter_name] = parameter.repeat(n_planning_samples, 1)
        self.planning_prior_dist = self.prior_dist_type(**parameters)
        self._planning_sample = None
        self._planning = True

    def acting_mode(self):
        self._planning = False
        self.planning_prior_dist = None
        self._planning_sample = None

    def reset(self, batch_size=1):
        # reset the prior and approximate posterior
        prior_params = {k: v.repeat(batch_size, 1) for k, v in self.initial_prior_params.items()}
        self.prior_dist = self.prior_dist_type(**prior_params)
        self.init_approx_post()
        self.reinitialized = True
        # self.approx_post_dist = None
        self.planning_prior_dist = None
        self._planning_sample = None
        self._sample = None
        self._planning = False

    def kl_divergence(self, analytical=True):
        if self.approx_post_dist_type is not None:
            if self.approx_post_dist_type == Delta:
                analytical = False
            if analytical:
                return torch.distributions.kl_divergence(self.approx_post_dist, self.prior_dist)
            else:
                # numerical approximation
                if self.approx_post_dist.has_rsample:
                    sample = self.approx_post_dist.rsample()
                else:
                    sample = self.approx_post_dist.sample()
                return self.approx_post_dist.log_prob(sample) - self.prior_dist.log_prob(sample)
        else:
            return self._sample.new_zeros(self._sample.shape)

    def log_importance_weights(self):
        # calculate importance weights for multiple samples
        sample = self._sample.view(self._n_samples, -1, self.n_variables)
        # expand the distributions to handle evaluating multiple samples
        expanded_prior = self.prior_dist.expand(torch.Size([self._n_samples]) + self.prior_dist.batch_shape)
        expanded_approx_post = self.approx_post_dist.expand(torch.Size([self._n_samples]) + self.approx_post_dist.batch_shape)
        # calculate the importance weight
        return (expanded_prior.log_prob(sample) - expanded_approx_post.log_prob(sample)).sum(dim=2, keepdim=True)

    def params_and_grads(self, concat=False, normalize=True, norm_type='layer'):
        # get current gradients and parameters
        params = [getattr(self.approx_post_dist, param).detach() for param in self.approx_post_models]
        grads = [getattr(self.approx_post_dist, param).grad.detach() for param in self.approx_post_models]
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
        params = nn.ParameterList()
        for model_name in self.approx_post_models:
            params.extend(list(self.approx_post_models[model_name].parameters()))
        if self.const_scale:
            params.append(self.approx_post_log_scale)
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        for model_name in self.prior_models:
            params.extend(list(self.prior_models[model_name].parameters()))
        for param_name in self.initial_prior_params:
            params.append(self.initial_prior_params[param_name])
        return params

    def inference_mode(self):
        self._detach_latent = False

    def generative_mode(self):
        self._detach_latent = True
