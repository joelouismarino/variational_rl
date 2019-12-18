import torch.nn as nn
from lib.models import get_model


class DirectInferenceModel(nn.Module):
    """
    Direct amortized inference.

    Args:
        agent (Agent): the parent agent
        network_args (dict): network arguments for inference model
    """
    def __init__(self, agent, network_args):
        self.agent = agent
        self.action_inference_model = get_model(network_args)

    def forward(self, state):
        inf_input = self.action_inference_model(state=state)
        self.agent.action_variable.infer(inf_input)
        

    def reset(self):
        pass
