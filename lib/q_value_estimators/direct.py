import copy
import torch.nn as nn
from lib.models import get_model


class DirectEstimator(nn.Module):
    """
    Estimate the Q-value using a learned model and Q network.

    Args:
        agent (Agent): the parent agent
        network_args (dict): arguments for the Q network
    """
    def __init__(self, agent, network_args):
        self.agent = agent
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        q_model_output = self.q_value_models[0].n_out
        self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])
        self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])

    def forward(self, state, action):
        # estimate q value
        action = action.tanh() if self.agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in q_value_models]
        q_values = [variable(inp) for variable, inp in zip(self.q_value_variables, q_value_input)]
        q_value = torch.min(q_values[0], q_values[1])
        return q_value

    def reset(self):
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
