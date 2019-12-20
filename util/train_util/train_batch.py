
def train_batch(agent, data, optimizer, model_only=False):
    """
    Trains the agent on the data using the optimizer.
    """
    optimizer.zero_grad()
    n_steps, batch_size = data['state'].shape[:2]
    agent.reset(batch_size, prev_action=data['prev_action'], prev_state=data['prev_state'])
    agent.train()

    state = data['state']
    reward = data['reward']
    done = data['done']
    valid = data['valid']
    action = data['action']
    log_prob = data['log_prob']

    for step in range(n_steps):
        agent.act(state[step], reward[step], done[step], action[step], valid[step], log_prob[step])
        optimizer.step(model_only=model_only)
    results = agent.evaluate()
    optimizer.apply(model_only=model_only)
    return results
