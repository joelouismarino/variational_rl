import numpy as np


def print_step_metrics(step_num, n_episodes, model, observation, reward):

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


def print_episode_metrics(episode, episode_log):
    print('Episode: ' + str(episode+1))
    print('     N Steps: ' + str(len(episode_log['free_energy'])))
    print('     Free Energy: ' + str(np.mean(episode_log['free_energy'])))
    print('         State KL: ' + str(np.mean(episode_log['state_kl'])))
    print('         Action KL: ' + str(np.mean(episode_log['action_kl'])))
    print('         Observation CLL: ' + str(np.mean(episode_log['obs_cll'])))
    print('         Reward CLL: ' + str(np.mean(episode_log['reward_cll'])))
    print('         Optimality CLL: ' + str(np.mean(episode_log['action_kl'])))
