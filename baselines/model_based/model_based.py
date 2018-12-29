import torch
from .config import get_model_args
from .models import Model
from .gradient_buffer import GradientBuffer
from .util import Viewer
from .util.plot_util import Plot_visdom


def learn(env, seed, total_timesteps, **kwargs):

    # torch.manual_seed(seed)

    plots = Plot_visdom()

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=5)

    obs_viewer = Viewer()
    recon_viewer = Viewer()

    observation = env.reset()
    reward = None
    n_episodes = 0

    log = init_log(log_items = ['Step', 'Free Energy', 'KL', 'State KL', 'Action KL', 'CLL',
                     'Obs CLL', 'Reward CLL', 'Optimality CLL'])

    for step_num in range(total_timesteps):

        obs_viewer.view(observation)

        action = model.act(observation, reward)

        recon_viewer.view(model.observation_variable.likelihood_dist.loc)

        log = add_log(log, step_num, model, observation, reward)
        if step_num % 10 == 0:
            plots.plot_visdom(log)

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

def init_log(log_items):
    log = {}
    for var in log_items:
        log[var] = []
    return log

def add_log(log, step_num, model, observation, reward):
    log['Step'].append(step_num)
    log['Free Energy'].append(model.free_energy(observation, reward).item())
    log['KL'].append(model.kl_divergence().sum().item())
    log['State KL'].append(model.state_variable.kl_divergence().sum().item())
    log['Action KL'].append(model.action_variable.kl_divergence().sum().item())
    log['CLL'].append(model.cond_log_likelihood(observation, reward).item())
    log['Obs CLL'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
    if reward is not None:
        log['Reward CLL'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
        log['Optimality CLL'].append(reward)
    else:
        log['Reward CLL'].append(0)
        log['Optimality CLL'].append(0)
    return log

