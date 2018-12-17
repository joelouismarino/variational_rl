from .observed_variable import ObservedVariable
from ...layers import FullyConnectedLayer


class FullyConnectedObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input):
        super(FullyConnectedObservedVariable, self).__init__(likelihood_dist)
        for model_name in self.likelihood_models:
            self.likelihood_models[model_name] = FullyConnectedLayer(n_input, n_variables)