import torch.nn as nn
from .observed_variable import ObservedVariable
from ...layers import FullyConnectedLayer


class FullyConnectedObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, stochastic=True,
                 constant_scale=False, sigmoid_loc=False, residual_loc=False,
                 manual_loc=False, manual_loc_alpha=0.):
        super(FullyConnectedObservedVariable, self).__init__(likelihood_dist,
                                                             n_variables,
                                                             n_input,
                                                             stochastic,
                                                             constant_scale,
                                                             residual_loc,
                                                             manual_loc,
                                                             manual_loc_alpha)
        for model_name in self.cond_likelihood.models:
            non_linearity = None
            if model_name == 'loc' and sigmoid_loc:
                non_linearity = 'sigmoid'
            self.cond_likelihood.models[model_name] = FullyConnectedLayer(n_input, n_variables,
                                                                     non_linearity=non_linearity)
            nn.init.constant_(self.cond_likelihood.models[model_name].linear.weight, 0.)
            nn.init.constant_(self.cond_likelihood.models[model_name].linear.bias, 0)
