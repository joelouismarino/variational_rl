from .observed_variable import ObservedVariable
from ...layers import ConvolutionalLayer


class ConvolutionalObservedVariable(ObservedVariable):

    def __init__(self, likelihood_dist, n_variables, n_input, filter_size, padding, stride, sigmoid_loc=False):
        super(ConvolutionalObservedVariable, self).__init__(likelihood_dist)
        for model_name in self.likelihood_models:
            non_linearity = None
            if model_name == 'loc' and sigmoid_loc:
                non_linearity = 'sigmoid'
            self.likelihood_models[model_name] = ConvolutionalLayer(n_input,
                                                                    n_variables,
                                                                    filter_size,
                                                                    padding,
                                                                    stride,
                                                                    non_linearity=non_linearity)
