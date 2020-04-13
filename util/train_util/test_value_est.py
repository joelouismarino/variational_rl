import comet_ml
import numpy as np
import json
import torch
import copy
from lib import create_agent
from lib.distributions import kl_divergence
from util.env_util import create_env
from util.train_util import collect_episode
from util.plot_util import load_checkpoint
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY


def estimate_monte_carlo_return(env, agent, env_state, state, action, n_samples):
    """
    Estimates the discounted Monte Carlo return (including KL) for a policy from
    a state-action pair.

    Args:
        env (gym.Env): the environment
        agent (Agent): the agent
        env_state (tuple): the environment state from MuJoCo (qpos, qvel)
        state (np.array): the environment state from gym
        action (torch.Tensor): the action
        n_samples (int): the number of Monte Carlo roll-outs

    Returns numpy array of returns of size [n_samples].
    """
    returns = np.zeros(n_samples)
    initial_action = action.numpy()
    for return_sample in range(n_samples):
        if return_sample % 100 == 0:
            print('     Sample ' + str(return_sample+1) + ' of  ' + str(n_samples) + '.')
        agent.reset(); agent.eval()
        # set the environment
        env.reset()
        qpos, qvel = env_state
        env.set_state(qpos=qpos, qvel=qvel)
        state, reward, done, _ = env.step(initial_action)
        # rollout the environment, get return
        rewards = [reward.view(-1).numpy()]
        kls = []
        while not done:
            action = agent.act(state, reward, done, action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward.view(-1).numpy())
            kl = kl_divergence(agent.approx_post, agent.prior, n_samples=agent.n_action_samples).sum(dim=1, keepdim=True)
            kls.append(kl.view(-1).detach().numpy())
        rewards = np.stack(rewards)
        kls = np.stack(kls)
        discounts = np.cumprod(agent.reward_discount * np.ones(kls.shape)).reshape(-1, 1)
        rewards[1:] = discounts * (rewards[1:] - agent.alphas['pi'].numpy() * kls)
        sample_return = np.sum(rewards)
        returns[return_sample] = sample_return
    return returns

def get_agent_value_estimate(agent, state, action):
    """
    Obtains the agent's value estimate for a particular state and action.

    Args:
        state (np.array):
        action ():

    Returns a dictionary of action-value estimates:
        direct: the estimate using the Q-network
        estimate: the full estimate (using the model)
    """
    agent.reset(); agent.eval()
    direct_estimate = agent.q_value_estimator(agent, state, action, direct=True).detach().view(-1).numpy()
    estimate = agent.q_value_estimator(agent, state, action).detach().view(-1).numpy()
    return {'direct': direct_estimate, 'estimate': estimate}

def evaluate_estimator(exp_key, n_state_action, n_mc_samples):
    """
    Evaluates the value estimator of a cached experiment throughout learning.

    Args:
        exp_key (str): the string of the comet experiment key
        n_state_action (int): number of state action pairs to evaluate
        n_mc_samples (int): number of Monte Carlo samples to estimate
                            environment returns

    Returns dictionary containing:
                value_estimates [n_ckpts, n_state_action, 1],
                direct_value_estimates [n_ckpts, n_state_action, 1]
                mc_estimates [n_ckpts, n_state_action, n_mc_samples]
    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the corresponding environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)

    # collect state-action samples using random policy
    print('Collecting ' + str(n_mc_samples) + ' state-action pairs...')
    sa_pairs = {'states': [], 'env_states': [], 'actions': []}
    state = env.reset()
    env_state = (copy.deepcopy(env.sim.data.qpos), copy.deepcopy(env.sim.data.qvel))
    for _ in range(n_state_action):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        sa_pairs['states'].append(state)
        sa_pairs['env_states'].append(env_state)
        sa_pairs['actions'].append(torch.from_numpy(action).view(1, -1))
        state = env.reset() if done else next_state
        env_state = (copy.deepcopy(env.sim.data.qpos), copy.deepcopy(env.sim.data.qvel))
    print('Done.')

    # enumerate state-action pairs, estimating returns at each stage of learning
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
    agent = create_agent(env, agent_args=agent_args)[0]
    # get the list of checkpoint timesteps
    ckpt_asset_list = [a for a in asset_list if 'ckpt' in a['fileName']]
    ckpt_asset_names = [a['fileName'] for a in ckpt_asset_list]
    ckpt_timesteps = [int(s.split('ckpt_step_')[1].split('.ckpt')[0]) for s in ckpt_asset_names]
    value_estimates = np.zeros((len(ckpt_timesteps), n_state_action, 1))
    direct_value_estimates = np.zeros((len(ckpt_timesteps), n_state_action, 1))
    mc_estimates = np.zeros((len(ckpt_timesteps), n_state_action, n_mc_samples))
    # iterate over checkpoint timesteps, evaluating
    for ckpt_ind, ckpt_timestep in enumerate(ckpt_timesteps):
        # load the checkpoint
        print('Evaluating checkpoint ' + str(ckpt_ind + 1) + ' of ' + str(len(ckpt_timesteps)))
        load_checkpoint(agent, exp_key, ckpt_timestep)
        # get value estimate and estimate returns for the state-action pairs
        for sa_ind, (env_state, state, act) in enumerate(zip(sa_pairs['env_states'], sa_pairs['states'], sa_pairs['actions'])):
            if sa_ind % 1 == 0:
                print('  Evaluating state-action pair ' + str(sa_ind + 1) + ' of ' + str(len(sa_pairs['states'])))
            action_value_estimate = get_agent_value_estimate(agent, state, act)
            value_estimates[ckpt_ind, sa_ind, :] = action_value_estimate['estimate']
            direct_value_estimates[ckpt_ind, sa_ind, :] = action_value_estimate['direct']
            returns = estimate_monte_carlo_return(env, agent, env_state, state, act, n_mc_samples)
            mc_estimates[ckpt_ind, sa_ind, :] = returns

    # TODO: log the value estimates to comet (need to json-ify the numpy arrays)
    # prev_exp = comet_ml.ExistingExperiment(api_key=LOGGING_API_KEY,
    #                                        previous_experiment=exp_key)
    # prev_exp.log_asset_data(value_estimates, name='value_estimates')
    # prev_exp.log_asset_data(direct_value_estimates, name='direct_value_estimates')
    # prev_exp.log_asset_data(mc_estimates, name='mc_estimates')

    return {'value_estimates': value_estimates,
            'direct_value_estimates': direct_value_estimates,
            'mc_estimates': mc_estimates}
