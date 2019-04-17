import torch
import numpy as np


def collect_episode(env, agent):
    """
    Collects an episode of experience using the model and environment.
    """
    agent.reset(); agent.eval()
    observation = env.reset()
    reward = 0.
    done = False
    n_steps = 0

    while not done:
        action = agent.act(observation, reward, done)
        observation, reward, done, _ = env.step(action)
        n_steps += 1
    agent.act(None, reward, done)

    return agent.get_episode(), n_steps


def train(agent, data, optimizer):
    """
    Trains the agent on the data using the optimizer.
    """
    optimizer.zero_grad()
    n_steps, batch_size = data['observation'].shape[:2]
    agent.reset(batch_size); agent.train()

    observation = data['observation']
    reward = data['reward']
    done = data['done']
    valid = data['valid']
    action = data['action']
    log_prob = data['log_prob']

    for step in range(n_steps):
        agent.act(observation[step], reward[step], done[step], action[step], valid[step], log_prob[step])
        optimizer.step()
    results = agent.evaluate()
    optimizer.apply()
    return results
