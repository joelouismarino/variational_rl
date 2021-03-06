import copy
import torch.nn as nn
from lib.models import get_model


class DirectGoalInferenceModel(nn.Module):
    """
    Goal-conditioned direct amortized inference.

    Args:
        network_args (dict): network arguments for inference model
    """
    def __init__(self, network_args):
        super(DirectGoalInferenceModel, self).__init__()
        self.inference_model = get_model(network_args)
        self.n_inf_iters = 1
        self.estimated_objectives = []

    def forward(self, agent, state, detach_params=False, direct=False, target=False, **kwargs):
        if detach_params:
            inference_model = copy.deepcopy(self.inference_model)
        else:
            inference_model = self.inference_model
        goal = agent.q_value_estimator.goal_state
        inf_input = inference_model(state=state, goal=goal)
        if direct and agent.direct_approx_post is not None:
            agent.direct_approx_post.step(inf_input, detach_params)
        elif target:
            agent.target_approx_post.step(inf_input, detach_params)
        else:
            agent.approx_post.step(inf_input, detach_params)

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
        self.estimated_objectives = []
