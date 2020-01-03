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
        super(DirectInferenceModel, self).__init__()
        # self.agent = agent
        self.inference_model = get_model(network_args)
        # # remove agent to prevent infinite recursion
        # del self.__dict__['_modules']['agent']

    def forward(self, agent, state):
        agent.approx_post.step(self.inference_model(state=state))

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
