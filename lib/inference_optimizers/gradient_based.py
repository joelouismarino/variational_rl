from torch import optim


class GradientBasedInference(Object):
    """
    Gradient-based inference optimizer.

    Args:
        agent (Agent): the parent agent
        lr (float): the learning rate
        inf_iters (int): number of inference iterations
    """
    def __init__(self, agent, lr, inf_iters):
        self.agent = agent
        self.optimizer = optim.SGD
        self.lr = lr
        self.inf_iters = inf_iters

    def forward(self, state):
        dist_params = self.action_variable.approx_post.get_dist_params()
        params = [param for _, param in dist_params.items()]
        act_opt = self.optimizer(params, lr=self.lr)
        act_opt.zero_grad()

        for _ in range(self.inf_iters):
            actions = self.agent.action_variable.approx_post.sample()
            q_value = self.agent.q_value_estimator(state, actions)
            q_value.backward()
            act_opt.step()
            act_opt.zero_grad()

    def reset(self):
        pass
