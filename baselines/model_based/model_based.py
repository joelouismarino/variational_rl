import torch
from config import get_model_args
from models import Model
from replay_buffer import ReplayBuffer

torch.manual_seed(0)


def learn(env):

    model_args = get_model_args(env)
    model = Model(**model_args)

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
