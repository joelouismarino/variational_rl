import torch.nn as nn
from .latent_variable import LatentVariable
from ...layers import FullyConnectedLayer

INIT_W = 1e-3

class FullyConnectedLatentVariable(LatentVariable):

    def __init__(self, prior_dist, approx_post_dist, n_variables, n_input,
                 constant_prior=False, inference_type='direct', norm_samples=False,
                 constant_prior_scale=False):
        super(FullyConnectedLatentVariable, self).__init__(prior_dist,
                                                           approx_post_dist,
                                                           n_variables,
                                                           n_input,
                                                           constant_prior,
                                                           inference_type,
                                                           norm_samples,
                                                           constant_prior_scale)
        # initialize the models
        if not constant_prior and n_input[0] is not None:
            for model_name in self.prior.models:
                self.prior.models[model_name] = FullyConnectedLayer(n_input[0],
                                                                    n_variables)

                nn.init.uniform_(self.prior.models[model_name].linear.weight, -INIT_W, INIT_W)
                nn.init.uniform_(self.prior.models[model_name].linear.bias, -INIT_W, INIT_W)

        if approx_post_dist is not None and n_input[1] is not None:
            for model_name in self.approx_post.models:
                self.approx_post.models[model_name] = FullyConnectedLayer(n_input[1],
                                                                          n_variables)

                nn.init.uniform_(self.approx_post.models[model_name].linear.weight, -INIT_W, INIT_W)
                nn.init.uniform_(self.approx_post.models[model_name].linear.bias, -INIT_W, INIT_W)

                if self.approx_post.update != 'direct':
                    self.approx_post.gates[model_name] = FullyConnectedLayer(n_input[1],
                                                                             n_variables,
                                                                             non_linearity='sigmoid')

        # reshape the initial prior params
        for param_name, param in self.prior.initial_params.items():
            self.prior.initial_params[param_name] = nn.Parameter(param.view(1, -1))

        # reset the variable to re-initialize the prior
        super(FullyConnectedLatentVariable, self).reset()
