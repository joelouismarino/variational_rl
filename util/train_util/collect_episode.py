
def collect_episode(env, agent, random=False):
    """
    Collects an episode of experience using the model and environment.
    """
    agent.reset(); agent.eval()
    state = env.reset()
    reward = 0.
    done = False
    n_steps = 0

    while not done:
        if n_steps > 1000:
            break
        action = env.action_space.sample() if random else None
        action = agent.act(state, reward, done, action)
        state, reward, done, _ = env.step(action)
        n_steps += 1
    agent.act(state, reward, done)
    return agent.collector.get_episode(), n_steps
