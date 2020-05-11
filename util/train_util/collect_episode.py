import copy
import numpy as np


def collect_episode(env, agent, random=False, eval=False):
    """
    Collects an episode of experience using the model and environment.

    Args:
        env (gym.env): the environment
        agent (Agent): the agent
        random (bool): whether to use random actions
        eval (bool): whether to evaluate the agent

    Returns episode (dict), n_steps (int), and env_states (dict).
    """
    agent.reset(); agent.eval()
    state = env.reset()
    reward = 0.
    done = False
    n_steps = 0
    env_states = {'qpos': [], 'qvel': []}

    while not done:
        if n_steps > 1000:
            break
        import ipdb; ipdb.set_trace()
        if 'sim' in dir(env.unwrapped):
            env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
            env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
        action = env.action_space.sample() if random else None
        action = agent.act(state, reward, done, action, eval=eval)
        state, reward, done, _ = env.step(action)
        n_steps += 1
    if 'sim' in dir(env.unwrapped):
        env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
        env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
        env_states['qpos'] = np.stack(env_states['qpos'])
        env_states['qvel'] = np.stack(env_states['qvel'])
    agent.act(state, reward, done)
    return agent.collector.get_episode(), n_steps, env_states
