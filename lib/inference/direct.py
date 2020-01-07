import torch.nn as nn
from lib.models import get_model


class DirectInferenceModel(nn.Module):
    """
    Direct amortized inference.

    Args:
        network_args (dict): network arguments for inference model
    """
    def __init__(self, network_args):
        super(DirectInferenceModel, self).__init__()
        self.inference_model = get_model(network_args)
        self.n_inf_iters = 1

    def forward(self, agent, state):
        agent.approx_post.step(self.inference_model(state=state))

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
