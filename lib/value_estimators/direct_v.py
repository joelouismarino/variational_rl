import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable


class DirectVEstimator(nn.Module):
    """
    Estimate the state-value using a network.

    Args:
        network_args (dict): arguments for the network
    """
    def __init__(self, network_args):
        super(DirectVEstimator, self).__init__()
        self.value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        model_output = self.value_models[0].n_out
        self.value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': model_output}) for _ in range(2)])
        self.target_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': model_output}) for _ in range(2)])

    def forward(self, agent, state, target=False, both=False,
                detach_params=False, *args, **kwargs):
        """
        Estimates the state-value.

        Args:
            state (torch.Tensor): the state [batch_size, state_dim]
            target (bool): whether to use the target networks
            both (bool): whether to return both values (or the min value)
            detach_params (bool): whether to use detached (copied) parameters
        """
        # estimate value
        if target:
            value_models = self.target_value_models
            value_variables = self.target_value_variables
        else:
            value_models = self.value_models
            value_variables = self.value_variables
        if detach_params:
            value_models = copy.deepcopy(value_models)
            value_variables = copy.deepcopy(value_variables)
        value_input = [model(state=state) for model in value_models]
        value = [variable(inp) for variable, inp in zip(value_variables, value_input)]
        if not both:
            value = torch.min(value[0], value[1])
        return value

    def reset(self, *args, **kwargs):
        pass

    def parameters(self):
        param_dict = {}
        param_dict['state_value_models'] = nn.ParameterList()
        param_dict['state_value_models'].extend(list(self.value_models.parameters()))
        param_dict['state_value_models'].extend(list(self.value_variables.parameters()))
        param_dict['target_state_value_models'] = nn.ParameterList()
        param_dict['target_state_value_models'].extend(list(self.target_value_models.parameters()))
        param_dict['target_state_value_models'].extend(list(self.target_value_variables.parameters()))
        return param_dict
