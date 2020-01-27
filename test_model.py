import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.train_util import collect_episode


def test_model(agent, env, horizon=None):
    """
    Test the agent's model by predicting future states and rewards and comparing
    with those from the environment.

    Args:
        agent (Agent): the (model-based) agent to be evaluated
        env (gym.env): the environment
        horizon (int): the horizon over which to test the model's predictions
    """
    # collect an episode
    print('Collecting Episode...')
    t_start = time.time()
    episode, n_steps = collect_episode(env, agent)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')

    if horizon is None:
        horizon = agent.q_value_estimator.horizon
    predictions = {'state': [None for _ in range(n_steps)],
                   'reward': [None for _ in range(n_steps)]}
    log_likelihoods = {'state': [None for _ in range(n_steps)],
                       'reward': [None for _ in range(n_steps)]}

    # evaluate the agent's predictions
    print('Evaluating agent predictions...')
    t_start = time.time()
    agent.eval()
    agent.n_action_samples = 1
    # go through each of the episode steps
    for step in range(n_steps-1):
        if step % 100 == 0:
            print('Step: ' + str(step) + ' of ' + str(n_steps))
        agent.reset(1, episode['action'][step], episode['state'][step])
        ep_actions = episode['action'][step:min(step + horizon, n_steps - 1)].to(agent.device)
        ep_states = episode['state'][step+1:min(step + horizon + 1, n_steps)].to(agent.device)
        ep_rewards = episode['reward'][step+1:min(step + horizon + 1, n_steps)].to(agent.device)
        # perform a model rollout
        state = episode['state'][step:step+1].to(agent.device)
        agent.q_value_estimator.planning_mode(agent)
        agent.q_value_estimator.state_variable.cond_likelihood.set_prev_x(state)
        state_predictions = {'loc': [], 'scale': []}
        reward_predictions = {'loc': [], 'scale': []}
        state_lls = []
        reward_lls = []
        rollout_horizon = min(horizon, ep_actions.shape[0])
        for rollout_step in range(rollout_horizon):
            action = ep_actions[rollout_step:rollout_step+1]
            # generate predictions
            agent.q_value_estimator.generate_state(state, action)
            agent.q_value_estimator.generate_reward(state, action)
            # collect the distribution parameters
            state_loc = agent.q_value_estimator.state_variable.cond_likelihood.planning_dist.loc
            state_scale = agent.q_value_estimator.state_variable.cond_likelihood.planning_dist.scale
            state_predictions['loc'].append(state_loc.detach().cpu())
            state_predictions['scale'].append(state_scale.detach().cpu())
            reward_loc = agent.q_value_estimator.reward_variable.cond_likelihood.planning_dist.loc
            reward_scale = agent.q_value_estimator.reward_variable.cond_likelihood.planning_dist.scale
            reward_predictions['loc'].append(reward_loc.detach().cpu())
            reward_predictions['scale'].append(reward_scale.detach().cpu())
            # evaluate log-probability of true state and reward
            state_ll = agent.q_value_estimator.state_variable.cond_log_likelihood(ep_states[rollout_step:rollout_step+1])
            reward_ll = agent.q_value_estimator.reward_variable.cond_log_likelihood(ep_rewards[rollout_step:rollout_step+1])
            state_lls.append(state_ll.detach().cpu())
            reward_lls.append(reward_ll.detach().cpu())
            # sample the predicted state
            state = agent.q_value_estimator.state_variable.sample()

        predictions['state'][step] = {'loc': torch.stack(state_predictions['loc']).view(rollout_horizon, -1),
                                       'scale': torch.stack(state_predictions['scale']).view(rollout_horizon, -1)}
        predictions['reward'][step] = {'loc': torch.stack(reward_predictions['loc']).view(rollout_horizon, -1),
                                       'scale': torch.stack(reward_predictions['scale']).view(rollout_horizon, -1)}
        log_likelihoods['state'][step] = torch.stack(state_lls)
        log_likelihoods['reward'][step] = torch.stack(reward_lls)

    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')

    return episode, predictions, log_likelihoods

def plot_predictions(pred, x):
    """
    Plot the predictions and actual quantities.

    Args:
        pred (dict): the predictions, containing loc and scale parameters [horizon, n_dims]
        x (torch.tensor): the actual quantities [horizon, n_dims]
    """
    plt.figure()
    horizon, n_dims = pred['loc'].shape
    for plot_num in range(n_dims):
        plt.subplot(n_dims, 1, plot_num + 1)
        plt.plot(pred['loc'][:, plot_num].numpy())
        lower = pred['loc'][:, plot_num] - pred['scale'][:, plot_num]
        upper = pred['loc'][:, plot_num] + pred['scale'][:, plot_num]
        plt.fill_between(np.arange(horizon), lower.numpy(), upper.numpy(), alpha=0.5)
        plt.plot(x[:, plot_num].numpy(), '.')

def plot_log_likelihoods(log_likelihoods):
    """
    Plot the log-likelihoods across time steps.

    Args:
        log_likelihoods (torch.tensor): [n_steps, horizon]
    """
    plt.figure()
    mean = log_likelihoods.mean(dim=0)
    std = log_likelihoods.std(dim=0)
    plt.plot(mean.numpy())
    lower = mean - std
    upper = mean + std
    plt.fill_between(np.arange(log_likelihoods.shape[1]), lower.numpy(), upper.numpy(), alpha=0.5)

if __name__ == '__main__':
    import argparse
    from util.env_util import create_env
    from lib import create_agent
    from util.plot_util import load_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('--device_id', default=None, type=int, help='GPU ID number')
    parser.add_argument('--checkpoint_exp_key', default=None, type=str, help='experiment key for the checkpoint to load')
    args = parser.parse_args()

    # create the environment and agent, load checkpoint
    env = create_env(args.env, None)
    agent, agent_args = create_agent(env, args.device_id)
    if args.checkpoint_exp_key is not None:
        load_checkpoint(agent, args.checkpoint_exp_key)

    pred_horizon = 15
    episode, predictions, log_likelihoods = test_model(agent, env, horizon=pred_horizon)
    # plot_log_likelihoods(torch.stack(log_likelihoods['state'][:975]).view(975,-1))
    # plot_log_likelihoods(torch.stack(log_likelihoods['reward'][:975]).view(975,-1))

    time_step = 15
    # plot_predictions(predictions['state'][time_step], episode['state'][time_step+1:time_step+1+pred_horizon])
    import ipdb; ipdb.set_trace()
