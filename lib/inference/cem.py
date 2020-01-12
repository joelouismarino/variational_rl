import torch


class CEMInference(object):
    """
    Cross-entropy method inference optimizer.

    Args:
        n_top_samples (int): number of top samples to fit at each iteration
        n_inf_iters (int): number of inference iterations
    """
    def __init__(self, n_top_samples, n_inf_iters):
        self.n_top = n_top_samples
        self.n_inf_iters = n_inf_iters
        # keep track of estimated objectives for reporting
        self.estimated_objectives = []

    def __call__(self, agent, state):

        for _ in range(self.n_inf_iters):
            # sample actions, evaluate objective
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = obj.view(agent.n_action_samples, -1, 1)
            self.estimated_objectives.append(obj.mean(dim=0).detach())
            # keep top samples, fit mean and std. dev.
            _, top_inds = obj.topk(self.n_top, dim=0)
            actions = actions.view(agent.n_action_samples, -1, agent.approx_post.n_variables)
            top_actions = actions.gather(0, top_inds.repeat(1, 1, agent.approx_post.n_variables))
            loc = top_actions.mean(dim=0)
            scale = top_actions.std(dim=0)
            # set the approximate posterior
            agent.approx_post.reset(agent.approx_post._batch_size,
                                    dist_params={'loc': loc.detach(), 'scale': scale.detach()})

    def reset(self, *args, **kwargs):
        self.estimated_objectives = []
