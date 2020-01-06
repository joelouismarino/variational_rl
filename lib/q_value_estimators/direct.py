import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable


class DirectEstimator(nn.Module):
    """
    Estimate the Q-value using a learned model and Q network.

    Args:
        network_args (dict): arguments for the Q network
    """
    def __init__(self, network_args):
        super(DirectEstimator, self).__init__()
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        q_model_output = self.q_value_models[0].n_out
        self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])
        self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])

    def forward(self, agent, state, action, target=False, both=False, detach_params=False):
        """
        Estimates the Q-value using the state and action.

        Args:
            state (torch.Tensor): the state
            action (torch.Tensor): the action
            target (bool): whether to use the target networks
            both (bool): whether to return both values (or the min value)
            detach_params (bool): whether to use detached (copied) parameters
        """
        # estimate q value
        if target:
            q_value_models = self.target_q_value_models
            q_value_variables = self.target_q_value_variables
        else:
            q_value_models = self.q_value_models
            q_value_variables = self.q_value_variables
        if detach_params:
            q_value_models = copy.deepcopy(q_value_models)
            q_value_variables = copy.deepcopy(q_value_variables)
        action = action.tanh() if agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in q_value_models]
        q_value = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
        if not both:
            q_value = torch.min(q_value[0], q_value[1])
        return q_value

    def reset(self, *args):
        pass

    def parameters(self):
        param_dict = {}
        param_dict['q_value_models'] = nn.ParameterList()
        param_dict['q_value_models'].extend(list(self.q_value_models.parameters()))
        param_dict['q_value_models'].extend(list(self.q_value_variables.parameters()))
        param_dict['target_q_value_models'] = nn.ParameterList()
        param_dict['target_q_value_models'].extend(list(self.target_q_value_models.parameters()))
        param_dict['target_q_value_models'].extend(list(self.target_q_value_variables.parameters()))
        return param_dict
