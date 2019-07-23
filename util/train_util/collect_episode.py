
def collect_episode(env, agent, random=False):
    """
    Collects an episode of experience using the model and environment.
    """
    agent.reset(); agent.eval()
    observation = env.reset()
    reward = 0.
    done = False
    n_steps = 0

    while not done:
        action = agent.act(observation, reward, done, random=random)
        if random:
            action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        n_steps += 1
    agent.act(None, reward, done, random=random)

    return agent.get_episode(), n_steps
