
def train_batch(agent, data, optimizer, model_only=False):
    """
    Trains the agent on the data using the optimizer.
    """
    optimizer.zero_grad()
    n_steps, batch_size = data['observation'].shape[:2]
    agent.reset(batch_size, prev_action=data['prev_action'], prev_obs=data['prev_obs'])
    agent.train()

    observation = data['observation']
    reward = data['reward']
    done = data['done']
    valid = data['valid']
    action = data['action']
    log_prob = data['log_prob']

    for step in range(n_steps):
        agent.act(observation[step], reward[step], done[step], action[step], valid[step], log_prob[step])
        optimizer.step(model_only=model_only)
    results = agent.evaluate()
    optimizer.apply(model_only=model_only)
    return results
