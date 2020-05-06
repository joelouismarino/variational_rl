from config import get_agent_args
from lib.distributions import Distribution, kl_divergence


def estimate_agent_kl(env, agent, prev_episode):
    """
    Estimate the change in the agent's policy from the last collected episode.
    Estimated using D_KL (pi_old || pi_new), sampling from previous episode.

    Args:
        env (gym.Env):
        agent (Agent): the most recent version of the agent
        prev_episode (dict): the previously collected episode
    """
    if prev_episode is None:
        return 0.

    # create a distribution to hold old approx post params
    agent_args = get_agent_args(env)
    agent_args['approx_post_args']['n_input'] = None
    old_approx_post = Distribution(**agent_args['approx_post_args']).to(agent.device)

    agent.reset(); agent.eval()

    states = prev_episode['state']
    dist_params = prev_episode['distributions']['action']['approx_post']

    agent_kl = 0

    for timestep in range(prev_episode['state'].shape[0]):

        state = states[timestep:timestep+1]
        params = {k: v[timestep:timestep+1].to(agent.device) for k, v in dist_params.items()}

        old_approx_post.reset(dist_params=params)
        agent.act(state)
        kl = kl_divergence(old_approx_post, agent.approx_post).sum().detach().item()
        agent_kl += kl

    agent_kl /= prev_episode['state'].shape[0]

    return agent_kl
