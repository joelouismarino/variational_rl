import torch
import torch.nn as nn


class CEMInference(nn.Module):
    """
    Cross-entropy method inference optimizer.

    Args:
        lr (float): learning rate for smoothed updates
        n_top_samples (int): number of top samples to fit at each iteration
        n_inf_iters (int): number of inference iterations
    """
    def __init__(self, lr, n_top_samples, n_inf_iters):
        super(CEMInference, self).__init__()
        self.n_top = n_top_samples
        self.lr = lr
        self.n_inf_iters = n_inf_iters
        # keep track of estimated objectives for reporting
        self.estimated_objectives = []

    def forward(self, agent, state, **kwargs):

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
            scale = torch.sqrt(top_actions.var(dim=0) + 1e-6)
            # smoothed update
            old_loc = agent.approx_post.dist.loc.detach()
            old_scale = agent.approx_post.dist.scale.detach()
            new_loc = (1. - self.lr) * old_loc + self.lr * loc
            new_scale = (1. - self.lr) * old_scale + self.lr * scale
            # set the approximate posterior
            agent.approx_post.reset(agent.approx_post._batch_size,
                                    dist_params={'loc': new_loc.detach(),
                                                 'scale': new_scale.detach()})

    def reset(self, *args, **kwargs):
        self.estimated_objectives = []
