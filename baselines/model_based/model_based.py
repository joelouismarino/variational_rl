import torch
from .config import get_model_args
from .models import Model
from .gradient_buffer import GradientBuffer
from .util import Viewer


def learn(env, seed, total_timesteps, **kwargs):

    # torch.manual_seed(seed)

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=5)

    obs_viewer = Viewer()
    recon_viewer = Viewer()

    observation = env.reset()
    reward = None
    n_episodes = 0

    for step_num in range(total_timesteps):

        obs_viewer.view(observation)

        action = model.act(observation, reward)

        recon_viewer.view(model.observation_variable.likelihood_dist.loc)

        print('Step Num: ' + str(step_num))
        print('     Episode: ' + str(n_episodes+1))
        print('     Free Energy: ' + str(model.free_energy(observation, reward).item()))
        print('         KL Divergence: ' + str(model.kl_divergence().sum().item()))
        print('             State KL: ' + str(model.state_variable.kl_divergence().sum().item()))
        print('             Action KL: ' + str(model.action_variable.kl_divergence().sum().item()))
        print('         Cond. Log Likelihood: ' + str(model.cond_log_likelihood(observation, reward).item()))
        print('             Observation CLL: ' + str(model.observation_variable.cond_log_likelihood(observation).sum().item()))
        if reward is not None:
            print('             Reward CLL: ' + str(model.reward_variable.cond_log_likelihood(reward).sum().item()))
            print('             Optimality CLL: ' + str(reward))

        observation, reward, done, _ = env.step(action)

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
