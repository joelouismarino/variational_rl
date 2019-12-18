import torch.nn as nn
from lib.models import get_model


class IterativeInferenceModel(nn.Module):
    """
    Iterative amortized inference.

    Args:
        agent (Agent): the parent agent
        network_args (dict): network arguments for inference model
        inf_iters (int): number of inference iterations
    """
    def __init__(self, agent, network_args, inf_iters):
        self.agent = agent
        self.action_inference_model = get_model(network_args)
        self.inf_iters = inf_iters

    def forward(self, state):

        for _ in range(self.inf_iters):
            actions = self.agent.action_variable.approx_post.sample()
            q_value = self.agent.q_value_estimator(state, actions)
            q_value.backward()

            params, grads = self.agent.action_variable.params_and_grads()
            inf_input = self.action_inference_model(params=params, grads=grads, state=state)
            self.agent.action_variable.infer(inf_input)

    def reset(self):
        pass
