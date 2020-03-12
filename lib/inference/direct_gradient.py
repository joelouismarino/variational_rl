import copy
import torch.nn as nn
from torch import optim
from misc import clear_gradients
from lib.models import get_model

# TODO: work in progress, does not currently work for training
class DirectGradientInference(nn.Module):
    """
    Direct amortized inference with additional gradient steps.

    Args:
        network_args (dict): network arguments for inference model
        lr (float): the learning rate
        n_inf_iters (int): number of inference iterations after direct inference
    """
    def __init__(self, network_args, lr, n_inf_iters):
        super(DirectGradientInference, self).__init__()
        self.inference_model = get_model(network_args)
        self.optimizer = optim.SGD
        self.lr = lr
        self.n_inf_iters = n_inf_iters
        # keep track of estimated objectives for reporting
        self.estimated_objectives = []

    def forward(self, agent, state, detach_params=False, direct=False, **kwargs):
        if detach_params:
            inference_model = copy.deepcopy(self.inference_model)
        else:
            inference_model = self.inference_model
        inf_input = inference_model(state=state)
        if direct and agent.direct_approx_post is not None:
            agent.direct_approx_post.step(inf_input, detach_params)
        else:
            agent.approx_post.step(inf_input, detach_params)

            dist_params = agent.approx_post.get_dist_params()
            params = [param for _, param in dist_params.items()]
            act_opt = self.optimizer(params, lr=self.lr)
            act_opt.zero_grad()

            for _ in range(self.n_inf_iters):
                actions = agent.approx_post.sample(agent.n_action_samples)
                obj = agent.estimate_objective(state, actions)
                obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
                self.estimated_objectives.append(obj.detach())
                obj.sum().backward(retain_graph=True)
                act_opt.step()
                act_opt.zero_grad()

            clear_gradients(agent.generative_parameters())

    def reset(self, batch_size):
        self.inference_model.reset(batch_size)
        self.estimated_objectives = []
