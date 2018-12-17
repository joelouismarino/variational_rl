import torch
from config import get_model_args
from models import Model
from gradient_buffer import GradientBuffer

torch.manual_seed(0)


def learn(env):

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=32)

    observation = env.reset()
    reward = None

    for step_num in range(n_steps):

        action, free_energy = model.act(observation, reward)
        observation, reward, done, _ = env.step(action)

        grad_buffer.collect()

        if done:
            grad_buffer.update()
            observation = env.reset()
            reward = None
            model.reset()
