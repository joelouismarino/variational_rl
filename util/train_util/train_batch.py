
def train_batch(agent, data, optimizer, model_only=False):
    """
    Trains the agent on the data using the optimizer.
    """
    optimizer.zero_grad()
    n_steps, batch_size = data['state'].shape[:2]
    agent.reset(batch_size, data['prev_action'], data['prev_state'])
    agent.train()

    state = data['state']
    reward = data['reward']
    done = data['done']
    valid = data['valid']
    action = data['action']
    log_prob = data['log_prob']

    # E-step
    for step in range(n_steps):
        agent.act(state[step], reward[step], done[step], action[step], valid[step], log_prob[step])
    # M-step
    results = agent.evaluate()
    optimizer.apply(model_only=model_only)
    return results
