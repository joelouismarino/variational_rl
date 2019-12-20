


class NonParametricInference(Object):
    """
    Estimate the Boltzmann distribution policy (non-parametric).

    Args:
        agent (Agent): the parent agent
    """
    def __init__(self, agent):
        self.agent = agent

    def forward(self, state):
        actions = self.agent.target_prior.sample(self.n_action_samples)
        prior_log_probs = self.agent.target_prior.log_prob(actions).detach()
        q_values = self.agent.q_value_estimator(state, actions)
        temperature = self.agent.alphas['pi'].detach()
        self.agent.approx_post.step(prior_log_probs, q_values, temperature)

    def reset(self, *args, **kwargs):
        pass
