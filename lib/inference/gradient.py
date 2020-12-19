import torch.nn as nn
from torch import optim
from misc import clear_gradients


class GradientBasedInference(nn.Module):
    """
    Gradient-based inference optimizer.

    Args:
        lr (float): the learning rate
        n_inf_iters (int): number of inference iterations
    """
    def __init__(self, lr, n_inf_iters):
        super(GradientBasedInference, self).__init__()
        self.optimizer = optim.SGD
        self.lr = lr
        self.n_inf_iters = n_inf_iters
        # keep track of estimated objectives for reporting
        self.estimated_objectives = []

    def forward(self, agent, state,**kwargs):
        dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
        agent.approx_post.reset(batch_size=state.shape[0], dist_params=dist_params)
        # dist_params = agent.approx_post.get_dist_params()
        params = [param for _, param in dist_params.items()]
        act_opt = self.optimizer(params, lr=self.lr)
        act_opt.zero_grad()

        for it in range(self.n_inf_iters):
            # print(' ITERATION: ' + str(it))
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
            self.estimated_objectives.append(obj.detach())
            # print(' OBJ: ' + str(-obj.mean().item()))
            obj.sum().backward(retain_graph=True)
            act_opt.step()
            act_opt.zero_grad()
            # clear the sample to force resampling
            agent.approx_post._sample = None

        clear_gradients(agent.generative_parameters())

    def reset(self, *args, **kwargs):
        self.estimated_objectives = []
