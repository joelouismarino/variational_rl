from .observed_variable import ObservedVariable
from ...layers import TransposedConvLayer


class TransposedConvObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, filter_size, padding, stride):
        super(TransposedConvObservedVariable, self).__init__(likelihood_dist)
        for model_name in self.likelihood_models:
            self.likelihood_models[model_name] = TransposedConvLayer(n_input,
                                                                     n_variables,
                                                                     filter_size,
                                                                     padding,
                                                                     stride)
