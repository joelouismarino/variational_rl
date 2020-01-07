import torch.nn as nn
from lib.models import get_model
from misc import clear_gradients


class IterativeInferenceModel(nn.Module):
    """
    Iterative amortized inference.

    Args:
        network_args (dict): network arguments for inference model
        n_inf_iters (int): number of inference iterations
        n_inf_samples (int): number of action samples drawn during inference
    """
    def __init__(self, network_args, n_inf_iters):
        super(IterativeInferenceModel, self).__init__()
        self.inference_model = get_model(network_args)
        self.n_inf_iters = n_inf_iters

    def forward(self, agent, state):

        for _ in range(self.n_inf_iters):
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
            obj.sum().backward(retain_graph=True)

            params, grads = agent.approx_post.params_and_grads()
            inf_input = self.inference_model(params=params, grads=grads, state=state)
            agent.approx_post.step(inf_input)
            agent.approx_post.retain_grads()

        clear_gradients(agent.generative_parameters())

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
