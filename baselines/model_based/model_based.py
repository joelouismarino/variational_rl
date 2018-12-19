import torch
from .config import get_model_args
from .models import Model
from .gradient_buffer import GradientBuffer


def learn(env, seed, total_timesteps, **kwargs):

    # torch.manual_seed(seed)

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=5)

    observation = env.reset()
    reward = None
    n_episodes = 0

    for step_num in range(total_timesteps):

        action, free_energy = model.act(observation, reward)
        observation, reward, done, _ = env.step(action)
        print('Step Num: ' + str(step_num) + ', Episode: ' + str(n_episodes+1) + ', Free Energy: ' + str(free_energy.item()))
        env.render()
        grad_buffer.accumulate()

        if done:
            grad_buffer.collect()
            grad_buffer.update()
            n_episodes += 1
            observation = env.reset()
            reward = None
            model.reset()

    return model
