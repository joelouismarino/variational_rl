

class NonParametricInference(object):
    """
    Estimate the Boltzmann distribution policy (non-parametric).
    """
    def __init__(self):
        pass

    def __call__(self, agent, state, **kwargs):
        # sample the actions
        actions = agent.target_prior.sample(agent.n_action_samples)
        # calculate the prior log probabilities
        prior_log_probs = agent.target_prior.log_prob(actions).detach()
        prior_log_probs = prior_log_probs.sum(dim=2, keepdim=True)
        # estimate the Q-values
        state = state.repeat(agent.n_action_samples, 1)
        q_values = agent.q_value_estimator(agent, state, actions).detach()
        q_values = q_values.view(agent.n_action_samples, -1, 1)
        # get the current temperature
        temperature = agent.alphas['pi']
        # calculate the non-parametric distribution
        actions = actions.view(agent.n_action_samples, -1, actions.shape[-1])
        agent.approx_post.step(prior_log_probs=prior_log_probs, q_values=q_values,
                               temperature=temperature, actions=actions)

    def reset(self, *args, **kwargs):
        pass
