from torch import optim


class GradientBasedInference(Object):
    """
    Gradient-based inference optimizer.

    Args:
        agent (Agent): the parent agent
        lr (float): the learning rate
        n_inf_iters (int): number of inference iterations
        n_inf_samples (int): number of action samples drawn during inference
    """
    def __init__(self, agent, lr, n_inf_iters, n_inf_samples):
        self.agent = agent
        self.optimizer = optim.SGD
        self.lr = lr
        self.n_inf_iters = n_inf_iters
        self.n_inf_samples = n_inf_samples

    def forward(self, state):
        dist_params = self.agent.approx_post.get_dist_params()
        params = [param for _, param in dist_params.items()]
        act_opt = self.optimizer(params, lr=self.lr)
        act_opt.zero_grad()

        for _ in range(self.n_inf_iters):
            actions = self.agent.approx_post.sample(self.n_inf_samples)
            obj = self.agent.estimate_objective(state, actions)
            obj.backward()
            act_opt.step()
            act_opt.zero_grad()

    def reset(self):
        pass
