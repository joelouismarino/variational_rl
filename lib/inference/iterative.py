import torch.nn as nn
from lib.models import get_model
from misc import clear_gradients


class IterativeInferenceModel(nn.Module):
    """
    Iterative amortized inference.

    Args:
        agent (Agent): the parent agent
        network_args (dict): network arguments for inference model
        n_inf_iters (int): number of inference iterations
        n_inf_samples (int): number of action samples drawn during inference
    """
    def __init__(self, agent, network_args, n_inf_iters, n_inf_samples):
        self.agent = agent
        self.inference_model = get_model(network_args)
        self.n_inf_iters = n_inf_iters
        self.n_inf_samples = n_inf_samples

    def forward(self, state):

        for _ in range(self.n_inf_iters):
            actions = self.agent.approx_post.sample(self.n_inf_samples)
            obj = self.agent.estimate_objective(state, actions)
            obj.backward()

            params, grads = self.agent.approx_post.params_and_grads()
            inf_input = self.inference_model(params=params, grads=grads, state=state)
            self.agent.approx_post.step(inf_input)

        clear_gradients(self.agent.generative_parameters())

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
