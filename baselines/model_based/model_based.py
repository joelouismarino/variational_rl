import torch
from .model import Model
from .replay_buffer import ReplayBuffer

torch.manual_seed(0)


def learn(env):

    model = Model()

    replay_buffer = ReplayBuffer()

    observation = env.reset()
    reward = None

    for step_num in range(n_steps):

        action = model.act(observation, reward)

        observation, reward, done, _ = env.step(action)

        replay_buffer.add()

        if done:
            observation = env.reset()
            reward = None
