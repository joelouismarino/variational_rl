import copy
import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
from lib.layers import FullyConnectedLayer
from .tanh_normal import TanhNormal
from .normal_uniform import NormalUniform
from .boltzmann import Boltzmann

# range for weight initialization
INIT_W = 1e-3

class Distribution(nn.Module):
    """
    Wrapper around PyTorch distributions.

    Args:
        dist_type (str): the type of distribution
        n_variables (int): the number of variables
        n_input (int): number of input dimensions
        stochastic (bool): whether to sample stochastically
        constant (bool): whether to have a constant distribution
        constant_scale (bool): whether to set the scale parameter as constant
        residual_loc (bool): whether the location output should be the residual
        manual_loc (bool): manually include action norm in reward mean estimate (MuJoCo)
        manual_loc_alpha (float): the alpha parameter for manual specification
        update (str): the type of updating (direct or iterative)
        euler_loc (bool): whether to use euler integration for the location
        euler_args (dict): dictionary of euler integration arguments
    """
    def __init__(self, dist_type, n_variables, n_input, stochastic=True,
                 constant=False, constant_scale=False, residual_loc=False,
                 manual_loc=False, manual_loc_alpha=0., update='direct',
                 euler_loc=False, euler_args=None):
        super(Distribution, self).__init__()
        self.n_variables = n_variables
        self.stochastic = stochastic
        self.const_scale = constant_scale
        self.residual_loc = residual_loc
        self.manual_loc = manual_loc
        self.manual_loc_alpha = manual_loc_alpha
        self.update = update
        self.euler_loc = euler_loc
        self.euler_args = euler_args

        self.dist = None
        self.planning_dist = None
        self.planning = False

        self._n_samples = 1
        self._sample = None
        self._planning_sample = None
        # self._detach = True
        self._batch_size = 1
        self._prev_x = None
        self._planning_prev_x = None

        # distribution type
        if dist_type == 'TanhNormal':
            self.dist_type = TanhNormal
        elif dist_type == 'NormalUniform':
            self.dist_type = NormalUniform
        elif dist_type == 'Boltzmann':
            self.dist_type = Boltzmann
        else:
            self.dist_type = getattr(torch.distributions, dist_type)

        # models, initial_params, and update gates
        param_names = self.dist_type.arg_constraints.keys()
        if dist_type == 'Boltzmann':
            # non-parametric distribution
            param_names = []
        self.param_names = param_names
        if 'scale' in param_names:
            # self._log_scale_lim = [-15, 15]
            self.min_log_scale = nn.Parameter(-10 * torch.ones(1, self.n_variables))
            self.max_log_scale = nn.Parameter(0.5 * torch.ones(1, self.n_variables))
            if self.const_scale:
                self.log_scale = nn.Parameter(torch.zeros(1, self.n_variables))
                param_names = ['loc']

        self.models = nn.ModuleDict({name: None for name in param_names}) if not constant else None
        if self.update != 'direct':
            self.gates = nn.ModuleDict({name: None for name in param_names})

        if not constant and n_input is not None:
            for model_name in self.models:
                self.models[model_name] = FullyConnectedLayer(n_input, n_variables)
                nn.init.uniform_(self.models[model_name].linear.weight, -INIT_W, INIT_W)
                nn.init.uniform_(self.models[model_name].linear.bias, -INIT_W, INIT_W)

                if self.update != 'direct':
                    self.gates[model_name] = FullyConnectedLayer(n_input, n_variables,
                                                                 non_linearity='sigmoid')
                    nn.init.uniform_(self.gates[model_name].linear.weight, -INIT_W, INIT_W)
                    nn.init.uniform_(self.gates[model_name].linear.bias, -INIT_W, INIT_W)

        self.initial_params = nn.ParameterDict({name: None for name in param_names})
        for param in self.initial_params:
            req_grad = False if constant else True
            constraint = self.dist_type.arg_constraints[param]
            if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                self.initial_params[param] = nn.Parameter(0.5 * torch.ones(1, n_variables))
            elif constraint == constraints.dependent and param == 'low':
                self.initial_params[param] = nn.Parameter(-torch.ones(1, n_variables), requires_grad=False)
            elif constraint == constraints.dependent and param == 'high':
                self.initial_params[param] = nn.Parameter(torch.ones(1, n_variables), requires_grad=False)
            else:
                self.initial_params[param] = nn.Parameter(torch.zeros(1, n_variables))
            self.initial_params[param].requires_grad_(req_grad)

    def step(self, input=None, detach_params=False, **kwargs):
        """
        Update the distribution parameters by applying the models to the input.

        Args:
            input (torch.Tensor, optional): the input to the linear layers to
                                            the distribution parameters
            detach_params (bool): whether to use detached (copied) parameters
        """
        if detach_params:
            models = {model_name: copy.deepcopy(model) for model_name, model in self.models.items()}
            initial_params = {param_name: copy.deepcopy(param) for param_name, param in self.initial_params.items()}
            if self.update != 'direct':
                gates = {gate_name: copy.deepcopy(gate) for gate_name, gate in self.gates.items()}
            if self.const_scale:
                const_log_scale = copy.deepcopy(self.log_scale)
        else:
            models = self.models
            initial_params = self.initial_params
            if self.update != 'direct':
                gates = self.gates
            if self.const_scale:
                const_log_scale = self.log_scale

        parameters = {}
        if input is not None:
            for ind, param_name in enumerate(models):
                constraint = self.dist.arg_constraints[param_name]
                param_input = input[ind] if type(input) == list else input
                param_update = models[param_name](param_input)
                if self.update == 'direct':
                    param = param_update
                else:
                    param = getattr(self.dist, param_name).detach()
                    if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                        # convert to log-space (for scale parameters)
                        param = torch.log(param)
                    gate = gates[param_name](param_input)
                    param = gate * param + (1. - gate) * param_update

                if param_name == 'loc':

                    if self.manual_loc:
                        # manually include action norm in reward mean estimate (MuJoCo)
                        action_norm = kwargs['action'].norm(dim=1, keepdim=True)
                        param = param - self.manual_loc_alpha * action_norm

                    if self.residual_loc:
                        # residual estimation of location parameter
                        prev_x = self._planning_prev_x if self.planning else self._prev_x
                        param = param + prev_x

                    if self.euler_loc:
                        # euler integration of location parameter
                        param[:, :self.euler_args['n_pos']] += self.euler_args['dt'] * param[:, -self.euler_args['n_pos']:].detach()
                        angle_pred = param[:, self.euler_args['angle_inds']].clone()
                        param[:, self.euler_args['angle_inds']] = torch.atan2(torch.sin(angle_pred),
                                                                              torch.cos(angle_pred))

                # satisfy any constraints on the parameter value (scale parameter)
                if type(constraint) == constraints.greater_than and constraint.lower_bound == 0:
                    # positive value
                    # param = torch.exp(torch.clamp(param, self._log_scale_lim[0], self._log_scale_lim[1]))
                    param = self.max_log_scale - nn.Softplus()(self.max_log_scale - param)
                    param = self.min_log_scale + nn.Softplus()(param - self.min_log_scale)
                    param = torch.exp(param)

                # set the parameter
                parameters[param_name] = param

            if self.const_scale:
                log_scale = const_log_scale.repeat(input.shape[0], 1)
                log_scale = self.max_log_scale - nn.softplus(self.max_log_scale - log_scale)
                log_scale = self.min_log_scale + nn.softplus(log_scale - self.min_log_scale)
                log_scale = torch.exp(log_scale)
                # scale = torch.exp(torch.clamp(log_scale, self._log_scale_lim[0], self._log_scale_lim[1]))
                parameters['scale'] = scale
        elif kwargs is not None:
            # Boltzmann approximate posterior
            parameters = kwargs
        else:
            # use the initial parameters
            parameters = initial_params

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

    def sample(self, n_samples=1, argmax=False):
        """
        Sample the distribution.

        Args:
            n_samples (int): number of samples to draw from the distribution
        """
        if (self._sample is None and not self.planning) or (n_samples != self._n_samples and not self.planning) or (self._planning_sample is None and self.planning):
            # get the appropriate distribution
            d = self.planning_dist if self.planning else self.dist
            # perform the sampling
            # sample is of size [n_samples, batch_size, n_variables]
            if self.stochastic:
                if argmax and 'loc' in self.param_names:
                    sample = torch.cat(n_samples * [d.loc], dim=0)
                    if type(d) == TanhNormal:
                        sample = sample.tanh()
                else:
                    sample = d.rsample([n_samples]) if d.has_rsample else d.sample([n_samples])
            else:
                sample = torch.cat(n_samples * [d.loc], dim=0)
            sample = sample.view(-1, self.n_variables)
            # update the internal sample
            if self.planning:
                self._planning_sample = sample
                if self.residual_loc:
                    self._planning_prev_x = sample
            else:
                self._sample = sample
            self._n_samples = n_samples

        sample = self._planning_sample if self.planning else self._sample

        return sample

    def log_prob(self, x):
        """
        Evaluate the log probability at x.

        Args:
            x (torch.Tensor): the point of evaluation [batch_size * n_x_samples, n_variables]
        """
        # get the appropriate distribution
        d = self.planning_dist if self.planning else self.dist
        batch_size = self.planning_dist.batch_shape[0] if self.planning else self._batch_size
        n_x_samples = int(x.shape[0] / batch_size)
        n_d_samples = int(d.batch_shape[0] / batch_size)
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

    def init(self, dist):
        """
        Reset the distribution from another distribution.

        Args:
            dist (Distribution): the distribution to initialize from
        """
        batch_size = dist._batch_size
        dist_params = dist.get_dist_params() if dist.dist_type == self.dist_type else None
        if dist_params is not None:
            for param_name, param in dist_params.items():
                dist_params[param_name] = param.detach().clone().requires_grad_()
        prev_x = dist._prev_x
        self.reset(batch_size, dist_params, prev_x)

    def reset(self, batch_size=1, dist_params=None, prev_x=None):
        """
        Reset the distribution parameters.

        Args:
            batch_size (int): the size of the batch
            dist_params (dict): dictionary of distribution parameters for initialization
        """
        if dist_params is None:
            # initialize distribution parameters from initial parameters
            dist_params = {k: v.repeat(batch_size, 1).data.requires_grad_() for k, v in self.initial_params.items()}
            if self.const_scale:
                dist_params['scale'] = self.log_scale.repeat(batch_size, 1).exp().data.requires_grad_()
            # for _, v in dist_params.items():
            #     v.retain_grad()
        # initialize the distribution
        d = self.dist_type(**dist_params) if len(dist_params.keys()) > 0 else None
        if self.planning:
            self.planning_dist = d
            self.planning_sample = None
        else:
            self.dist = d
            self._sample = None

        # reset other quantities
        self._batch_size = batch_size
        if self.residual_loc:
            if prev_x is not None:
                device = self.initial_params[list(self.initial_params.keys())[0]].device
                self._prev_x = prev_x.to(device)
            else:
                x = self.sample()
                self._prev_x = x.new(x.shape).zero_()
            self._planning_prev_x = None

    def set_prev_x(self, prev_x):
        """
        Sets the previous observation (for residual loc).
        """
        if self.residual_loc:
            if not self.planning:
                self._prev_x = prev_x
            else:
                self._planning_prev_x = prev_x

    def planning_mode(self, dist_type=None, dist_params=None, n_samples=None):
        """
        Put the distribution in planning mode. Initialize the planning dist.

        Args:
            dist_type (Distribution): the type of distribution for initialization
            dist_params (dict): dictionary of distribution parameters for initialization
            n_samples (int): number of action samples during planning
        """
        self.planning = True
        self._planning_sample = None
        self.planning_dist = None
        if dist_type is not None:
            self.planning_dist = dist_type(**dist_params)
        else:
            if n_samples is not None:
                dist_params = {k: v.repeat(self._batch_size * n_samples, 1) for k, v in self.initial_params.items()}
                if self.const_scale:
                    dist_params['scale'] = self.log_scale.repeat(self._batch_size * n_samples, 1).exp()
                self.planning_dist = self.dist_type(**dist_params)

    def acting_mode(self):
        """
        Put the distribution in acting mode.
        """
        self.planning = False
        self._planning_sample = None
        self.planning_dist = None
        self._planning_prev_x = None

    def parameters(self):
        """
        Get the list of parameters.
        """
        params = nn.ParameterList()
        for model_name in self.models:
            params.extend(list(self.models[model_name].parameters()))
            if self.update != 'direct':
                params.extend(list(self.gates[model_name].parameters()))
        # for param_name in self.initial_params:
        #     params.append(self.initial_params[param_name])
        if self.const_scale:
            params.append(self.log_scale)
        if 'scale' in self.param_names:
            params.append(self.min_log_scale)
            params.append(self.max_log_scale)
        return params

    def get_dist_params(self):
        """
        Get the dictionary of distribution parameters.
        """
        parameters = {}
        for parameter_name in self.param_names:
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

    def params_and_grads(self, concat=False, normalize=True, norm_type='layer'):
        """
        Get the current distribution parameters and gradients for the approx post.

        Args:
            concat (bool): whether to concatenate the parameters and gradients
            normalize (bool): whether to normalize the parameters and gradients
            norm_type (str): the type of normalization (either batch or layer)
        """
        param_dict = self.get_dist_params()
        grad_dict = self.get_dist_param_grads()
        # use log-scale as input instead of scale itself
        grad_dict['scale'] = grad_dict['scale'] * param_dict['scale']
        param_dict['scale'] = param_dict['scale'].log()
        # convert to lists
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

    # def attach(self):
    #     """
    #     Do not detach samples.
    #     """
    #     self._detach = False
    #
    # def detach(self):
    #     """
    #     Detach samples.
    #     """
    #     self._detach = True
