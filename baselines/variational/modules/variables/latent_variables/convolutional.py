import torch.nn as nn
from .latent_variable import LatentVariable
from ...layers import ConvolutionalLayer


class ConvolutionalLatentVariable(LatentVariable):

    def __init__(self, prior_dist, approx_post_dist, n_variables, n_input,
                 filter_sizes, paddings, strides):
        super(ConvolutionalLatentVariable, self).__init__(prior_dist,
                                                          approx_post_dist,
                                                          n_variables, n_input)
        # initialize the models
        for model_name in self.prior_models:
            self.prior_models[model_name] = ConvolutionalLayer(n_input[0],
                                                               n_variables,
                                                               filter_sizes[0],
                                                               paddings[0],
                                                               strides[0])
        for model_name in self.approx_post_models:
            self.approx_post_models[model_name] = ConvolutionalLayer(n_input[1],
                                                                     n_variables,
                                                                     filter_sizes[1],
                                                                     paddings[1],
                                                                     strides[1])
            self.approx_post_gates[model_name] = ConvolutionalLayer(n_input[1],
                                                                    n_variables,
                                                                    filter_sizes[1],
                                                                    paddings[1],
                                                                    strides[1],
                                                                    non_linearity='sigmoid')

        # reshape the initial prior params
        for param_name, param in self.initial_prior_params.items():
            self.initial_prior_params[param_name] = nn.Parameter(param.view(1, -1, 1, 1))

        # reset the variable to re-initialize the prior
        super(ConvolutionalLatentVariable, self).reset()
