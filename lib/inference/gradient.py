from torch import optim
from misc import clear_gradients


class GradientBasedInference(object):
    """
    Gradient-based inference optimizer.

    Args:
        lr (float): the learning rate
        n_inf_iters (int): number of inference iterations
    """
    def __init__(self, lr, n_inf_iters):
        self.optimizer = optim.SGD
        self.lr = lr
        self.n_inf_iters = n_inf_iters

    def forward(self, agent, state):
        dist_params = agent.approx_post.get_dist_params()
        params = [param for _, param in dist_params.items()]
        act_opt = self.optimizer(params, lr=self.lr)
        act_opt.zero_grad()

        for _ in range(self.n_inf_iters):
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj.backward(retain_graph=True)
            act_opt.step()
            act_opt.zero_grad()

        clear_gradients(agent.generative_parameters())

    def reset(self):
        pass
