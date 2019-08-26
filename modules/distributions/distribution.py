import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
from .delta import Delta
from .transformed_tanh import TransformedTanh


class Distribution(nn.Module):
    """
    Wrapper around PyTorch distributions.
    """
    def __init__(self, dist_type, n_variables, n_input, constant=False,
                 constant_scale=False, residual_loc=False, update='direct'):
        super(Distribution, self).__init__()
        self.n_variables = n_variables
        self.const_scale = constant_scale
        self.residual_loc = residual_loc
        self.update = update
        self.dist = None
        self.planning_dist = None
        self.planning = False

        self._n_samples = 1
        self._sample = None
        self._planning_sample = None
        self._detach = True
        self._batch_size = 1
        self._prev_obs = None
        self._planning_prev_obs = None

        # distribution type
        if dist_type == 'Delta':
            self.dist_type = Delta
        elif dist_type == 'TransformedTanh':
            self.dist_type = TransformedTanh
        else:
            self.dist_type = getattr(torch.distributions, dist_type)

        # models, initial_params, and update gates
        param_names = ['logits'] if dist_type in ['Categorical', 'Bernoulli'] else self.dist_type.arg_constraints.keys()
        if 'scale' in param_names:
            self._log_scale_lim = [-20, 2]
            if self.const_scale:
                self.log_scale = nn.Parameter(torch.ones(1, self.n_variables))
                param_names = ['loc']

        self.models = nn.ModuleDict({name: None for name in param_names}) if not constant else None

        self.initial_params = nn.ParameterDict({name: None for name in param_names})
        for param in self.initial_params:
            constraint = self.dist_type.arg_constraints[param]
            if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                self.initial_params[param] = nn.Parameter(torch.ones(1, n_variables))
            elif constraint == constraints.dependent and param == 'low':
                self.initial_params[param] = nn.Parameter(-torch.ones(1, n_variables), requires_grad=False)
            elif constraint == constraints.dependent and param == 'high':
                self.initial_params[param] = nn.Parameter(torch.ones(1, n_variables), requires_grad=False)
            else:
                self.initial_params[param] = nn.Parameter(torch.zeros(1, n_variables))

        if self.update != 'direct':
            self.gates = nn.ModuleDict({name: None for name in param_names})

    def step(self, input=None):
        """
        Update the distribution parameters by applying the models to the input.

        Args:
            input (torch.Tensor, optional): the input to the linear layers to
                                            the distribution parameters
        """
        if input is not None:
            parameters = {}
            for param_name in self.models:
                constraint = self.dist.arg_constraints[param_name]

                param_update = self.models[param_name](input)
                if self.update == 'direct':
                    param = param_update
                else:
                    param = getattr(self.dist, param_name).detach()
                    if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                        # convert to log-space (for scale parameters)
                        param = torch.log(param)
                    gate = self.gates[param_name](input)
                    param = gate * param + (1. - gate) * param_update

                if param_name == 'loc' and self.residual_loc:
                    # residual estimation of location parameter
                    prev_obs = self._planning_prev_obs if self.planning else self._prev_obs
                    param = param + prev_obs

                # satisfy any constraints on the parameter value
                if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                    # positive value
                    param = torch.exp(torch.clamp(param, self._log_scale_lim[0], self._log_scale_lim[1]))
                elif constraint == constraints.simplex:
                    # between 0 and 1 for probabilities
                    param = nn.Softmax()(param)

                # set the parameter
                parameters[param_name] = param

            if self.const_scale:
                log_scale = self.log_scale.repeat(input.shape[0], 1)
                scale = torch.exp(torch.clamp(log_scale, self._log_scale_lim[0], self._log_scale_lim[1]))
                parameters['scale'] = scale

            # create a new distribution with the parameters
            if not self.planning:
                self.dist = self.dist_type(**parameters)
                self._sample = None
            else:
                self.planning_dist = self.dist_type(**parameters)
                self._planning_sample = None

    def retain_grads(self):
        """
        Retain the gradient (for further inference)
        """
        for param_name in self.models:
            getattr(self.dist, param_name).retain_grad()

    def sample(self, n_samples=1):
        """
        Sample the distribution.

        Args:
            n_samples (int): number of samples to draw from the distribution
        """
        if (self._sample is None and not self.planning) or (self._planning_sample is None and self.planning):
            # get the appropriate distribution
            d = self.planning_dist if self.planning else self.dist
            # perform the sampling
            if d.has_rsample:
                # sample is of size [batch_size x n_samples x n_variables]
                sample = d.rsample([n_samples])
                # sample = sample.view(self._batch_size * n_samples, -1)
                sample = sample.view(-1, self.n_variables)
            else:
                sample = d.sample([n_samples])
                sample = sample.view(self._batch_size * n_samples, 1)
                # convert to one-hot encoding
                device = self.initial_params['logits'].device
                one_hot_sample = torch.zeros(sample.shape[0], self.n_variables).to(device)
                one_hot_sample.scatter_(1, sample, 1.)
                sample = one_hot_sample
            # update the internal sample
            if self.planning:
                self._planning_sample = sample
                if self.residual_loc:
                    self._planning_prev_obs = sample
            else:
                self._sample = sample
            self._n_samples = n_samples

        sample = self._planning_sample if self.planning else self._sample

        if self._detach and not self.planning:
            sample = sample.detach()

        if n_samples < self._n_samples:
            sample = sample.view(self._n_samples, -1, self.n_variables)
            sample = sample[:n_samples].view(-1, self.n_variables)

        return sample

    def log_prob(self, x):
        """
        Evaluate the log probability at x.

        Args:
            x (torch.Tensor): the point of evaluation [batch_size * n_x_samples, n_variables]
        """
        # get the appropriate distribution
        d = self.planning_dist if self.planning else self.dist
        n_x_samples = int(x.shape[0] / self._batch_size)
        n_d_samples = int(d.batch_shape[0] / self._batch_size)
        if n_x_samples == 1:
            # expand x
            x = torch.cat(n_d_samples * [x], dim=0)
        else:
            # expand the distribution
            x = x.view(-1, self._batch_size, self.n_variables)
            d = d.expand(torch.Size([n_x_samples]) + d.batch_shape)
        # evaluate
        return d.log_prob(x)

    def entropy(self):
        """
        Evaluate the entropy of the distribution.
        """
        # get the appropriate distribution
        d = self.planning_dist if self.planning else self.dist
        return d.entropy()

    def reset(self, batch_size=1, dist_params=None, prev_obs=None):
        """
        Reset the distribution parameters.

        Args:
            batch_size (int): the size of the batch
            dist_params (dict): dictionary of distribution parameters for initialization
        """
        if dist_params is None:
            # initialize distribution parameters from initial parameters
            dist_params = {k: v.repeat(batch_size, 1) for k, v in self.initial_params.items()}
            if self.const_scale:
                dist_params['scale'] = self.log_scale.repeat(batch_size, 1).exp()
        # initialize the distribution
        self.dist = self.dist_type(**dist_params)
        if self.dist_type == getattr(torch.distributions, 'Categorical'):
            self.dist.logits = dist_params['logits']
        # reset other quantities
        self._sample = None
        self._batch_size = batch_size
        if self.residual_loc:
            if prev_obs is not None:
                device = self.initial_params[list(self.initial_params.keys())[0]].device
                self._prev_obs = prev_obs.to(device)
            else:
                obs = self.sample()
                self._prev_obs = obs.new(obs.shape).zero_()
            self._planning_prev_obs = None

    def set_prev_obs(self, prev_obs):
        """
        Sets the previous observation (for residual loc).
        """
        if self.residual_loc:
            if not self.planning:
                self._prev_obs = prev_obs
            else:
                self._planning_prev_obs = prev_obs

    def planning_mode(self, dist_params=None):
        """
        Put the distribution in planning mode. Initialize the planning dist.

        Args:
            dist_params (dict): dictionary of distribution parameters for initialization
        """
        self.planning = True
        self._planning_sample = None
        self.planning_dist = None
        if dist_params is not None:
            self.planning_dist = self.dist_type(**dist_params)

    def acting_mode(self):
        """
        Put the distribution in acting mode.
        """
        self.planning = False
        self._planning_sample = None
        self.planning_dist = None
        self._planning_prev_obs = None

    def parameters(self):
        """
        Get the list of parameters.
        """
        params = nn.ParameterList()
        for model_name in self.models:
            params.extend(list(self.models[model_name].parameters()))
            if self.update != 'direct':
                params.extend(list(self.gates[model_name].parameters()))
        for param_name in self.initial_params:
            params.append(self.initial_params[param_name])
        if self.const_scale:
            params.append(self.log_scale)
        return params

    def get_dist_params(self):
        """
        Get the dictionary of distribution parameters.
        """
        parameters = {}
        for parameter_name in self.initial_params:
            parameters[parameter_name] = getattr(self.dist, parameter_name)
        return parameters

    def get_dist_param_grads(self):
        """
        Get the dictionary of distribution parameter gradients.
        """
        gradients = {}
        for parameter_name in self.initial_params:
            gradients[parameter_name] = getattr(self.dist, parameter_name).grad
        return gradients

    def attach(self):
        """
        Do not detach samples.
        """
        self._detach = False

    def detach(self):
        """
        Detach samples.
        """
        self._detach = True
