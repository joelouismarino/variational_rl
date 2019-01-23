import torch


def collect_episode(env, model):
    """
    Collects an episode of experience using the model and environment.
    """
    model.reset(); model.eval()
    observation = env.reset()
    reward = 0.
    done = False
    n_steps = 0

    while not done:
        action = model.act(observation, reward, done)
        observation, reward, done, _ = env.step(action)
        n_steps += 1
    model.act(None, reward, done)

    return model.get_episode(), n_steps


def train(model, data, optimizer):
    """
    Trains the model on the data using the optimizer.
    """
    optimizer.zero_grad()
    n_steps, batch_size = data['observation'].shape[:2]
    model.reset(batch_size); model.train()

    observation = data['observation']
    reward = data['reward']
    done = data['done']
    valid = data['valid']
    action = data['action']

    for step in range(n_steps):
        model.act(observation[step], reward[step], done[step], action[step], valid[step])
        optimizer.step()
    results = model.evaluate()
    optimizer.apply()
    return results
