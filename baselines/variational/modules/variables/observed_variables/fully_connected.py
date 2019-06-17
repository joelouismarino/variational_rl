from .observed_variable import ObservedVariable
from ...layers import FullyConnectedLayer


class FullyConnectedObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, integration_window=None, sigmoid_loc=False):
        super(FullyConnectedObservedVariable, self).__init__(likelihood_dist=likelihood_dist,
                                                             n_variables=n_variables,
                                                             integration_window=integration_window)
        for model_name in self.likelihood_models:
            non_linearity = None
            if model_name == 'loc' and sigmoid_loc:
                non_linearity = 'sigmoid'
            self.likelihood_models[model_name] = FullyConnectedLayer(n_input, n_variables,
                                                                     non_linearity=non_linearity)
