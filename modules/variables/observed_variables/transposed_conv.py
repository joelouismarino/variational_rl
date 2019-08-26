from .observed_variable import ObservedVariable
from ...layers import TransposedConvLayer


class TransposedConvObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, filter_size,
                 padding, stride, constant_scale=False, sigmoid_loc=False,
                 residual_loc=False):
        super(TransposedConvObservedVariable, self).__init__(likelihood_dist,
                                                             n_variables,
                                                             n_input,
                                                             constant_scale,
                                                             residual_loc)
        for model_name in self.likelihood_models:
            non_linearity = None
            if model_name == 'loc' and sigmoid_loc:
                # constrain mean to be between 0 and 1
                non_linearity = 'sigmoid'
            self.likelihood_models[model_name] = TransposedConvLayer(n_input,
                                                                     n_variables,
                                                                     filter_size,
                                                                     padding,
                                                                     stride,
                                                                     non_linearity=non_linearity)
