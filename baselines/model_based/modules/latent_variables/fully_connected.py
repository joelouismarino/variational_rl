from .latent_variable import LatentVariable
from ..layers import FullyConnectedLayer


class FullyConnectedLatentVariable(LatentVariable):

    def __init__(self, prior_dist, approx_post_dist, n_variables, n_input):
        super(FullyConnectedLatentVariable, self).__init__(prior_dist,
                                                           approx_post_dist,
                                                           n_variables,
                                                           n_input)
        # initialize the models
        for model_name in self.prior_models:
            self.prior_models[model_name] = FullyConnectedLayer(n_input[0],
                                                                n_variables)
        for model_name in self.approx_post_models:
            self.approx_post_models[model_name] = FullyConnectedLayer(n_input[1],
                                                                      n_variables)

        # reshape the initial prior params
        # TODO

        # reset the variable to re-initialize the prior
        super(FullyConnectedLatentVariable, self).reset()
