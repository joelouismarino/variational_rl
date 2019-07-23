from .observed_variable import ObservedVariable
from ...layers import FullyConnectedLayer


class FullyConnectedObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, constant_scale=False, sigmoid_loc=False):
        super(FullyConnectedObservedVariable, self).__init__(likelihood_dist,
                                                             n_variables,
                                                             n_input,
                                                             constant_scale)
        for model_name in self.cond_likelihood.models:
            non_linearity = None
            if model_name == 'loc' and sigmoid_loc:
                non_linearity = 'sigmoid'
            self.cond_likelihood.models[model_name] = FullyConnectedLayer(n_input, n_variables,
                                                                     non_linearity=non_linearity)
