import comet_ml
from comet_ml import Experiment
from local_vars import PROJECT_NAME, WORKSPACE, LOGGING_API_KEY, LOADING_API_KEY
import os, io
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

COLORS = ['b', 'g', 'r', 'c']

def flatten(dictionary):
    return pd.io.json.json_normalize(dictionary, sep='_').to_dict()

def get_arg_dict(args):
    arg_list = args._get_kwargs()
    arg_dict = {k: v for (k, v) in arg_list}
    return arg_dict

def flatten_arg_dict(arg_dict):
    flat_dict = {}
    for k, v in arg_dict.items():
        if type(v) == dict:
            flat_v = flatten_arg_dict(v)
            for kk, vv in flat_v.items():
                flat_dict[k + '_' + kk] = vv
        else:
            flat_dict[k] = v
    return flat_dict

def load_checkpoint(agent, checkpoint_exp_key, timestep=None):
    """
    Loads a checkpoint from Comet.

    Args:
        agent (Agent): the agent to be loaded
    """
    assert checkpoint_exp_key is not None, 'Checkpoint experiment key must be set.'
    print('Loading checkpoint from ' + checkpoint_exp_key + '...')
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=checkpoint_exp_key)
    asset_list = experiment.get_asset_list()
    if timestep is not None:
        # get the specified checkpoint
        file_name = 'ckpt_step_' + str(timestep) + '.ckpt'
        asset = [a for a in asset_list if a['fileName'] == file_name]
        if len(asset) == 0:
            raise KeyError('Checkpoint timestep not found.')
        asset = asset[0]
    else:
        # get most recent checkpoint
        asset_times = [asset['createdAt'] for asset in asset_list if 'ckpt' in asset['fileName']]
        asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    print('Checkpoint Name:', asset['fileName'])
    ckpt = experiment.get_asset(asset['assetId'])
    state_dict = torch.load(io.BytesIO(ckpt))
    agent.load_state_dict(state_dict)
    print('Done.')

class Plotter:
    """
    Handles plotting and logging to comet.

    Args:
        exp_args (args.parse_args): arguments for the experiment
        agent_args (dict): arguments for the agent
        agent (Agent): the agent
    """
    def __init__(self, exp_args, agent_args, agent):
        
        self.experiment = Experiment(api_key=LOGGING_API_KEY,
                                     project_name=PROJECT_NAME,
                                     workspace=WORKSPACE)
        self.exp_args = exp_args
        self.agent_args = agent_args
        self.agent = agent
        self.experiment.disable_mp()
        self.experiment.log_parameters(get_arg_dict(exp_args))
        self.experiment.log_parameters(flatten_arg_dict(agent_args))
        self.experiment.log_asset_data(json.dumps(get_arg_dict(exp_args)), name='exp_args')
        self.experiment.log_asset_data(json.dumps(agent_args), name='agent_args')
        if self.exp_args.checkpoint_exp_key is not None:
            self.load_checkpoint()
        self.result_dict = None
        # keep a hard-coded list of returns in case Comet fails
        self.returns = []

    def _plot_ts(self, key, observations, statistics, label, color):
        dim_obs = min(observations.shape[1], 9)
        k = 1
        for i in range(dim_obs):
            plt.subplot(int(str(dim_obs) + '1' + str(k)))
            observations_i = observations[:-1, i].cpu().numpy()
            if key == 'action' and self.agent.postprocess_action:
                observations_i = np.tanh(observations_i)
            plt.plot(observations_i.squeeze(), 'o', label='observation', color='k', markersize=2)
            if len(statistics) == 1:  # Bernoulli distribution
                probs = statistics['probs']
                probs = probs.cpu().numpy()
                plt.plot(probs, label=label, color=color)
            elif len(statistics) == 2:
                if 'loc' in statistics:
                    # Normal distribution
                    mean = statistics['loc']
                    std = statistics['scale']
                    mean = mean[:, i].cpu().numpy()
                    std = std[:, i].cpu().numpy()
                    mean = mean.squeeze()
                    std = std.squeeze()
                    x, plus, minus = mean, mean + std, mean - std
                    if key == 'action' and label == 'approx_post' and self.agent_args['approx_post_args']['dist_type'] in ['TanhNormal', 'TanhARNormal']:
                        # Tanh Normal distribution
                        x, plus, minus = np.tanh(x), np.tanh(plus), np.tanh(minus)
                    if key == 'action' and label == 'direct_approx_post' and self.agent_args['approx_post_args']['dist_type'] in ['TanhNormal', 'TanhARNormal']:
                        # Tanh Normal distribution
                        x, plus, minus = np.tanh(x), np.tanh(plus), np.tanh(minus)
                    if key == 'action' and label == 'prior' and self.agent_args['prior_args']['dist_type'] in ['TanhNormal', 'TanhARNormal']:
                        # Tanh Normal distribution
                        x, plus, minus = np.tanh(x), np.tanh(plus), np.tanh(minus)
                    if key == 'action' and self.agent.postprocess_action:
                        x, plus, minus = np.tanh(x), np.tanh(plus), np.tanh(minus)
                    if key == 'action' and label == 'prior' and self.agent_args['prior_args']['dist_type'] == 'NormalUniform':
                        # Normal + Uniform distribution
                        x, plus, minus = x, np.minimum(plus, 1.), np.maximum(minus, -1)
                elif 'low' in statistics:
                    # Uniform distribution
                    low = statistics['low'][:, i].cpu().numpy()
                    high = statistics['high'][:, i].cpu().numpy()
                    x = low + (high - low) / 2
                    plus, minus = x + high, x + low
                else:
                    raise NotImplementedError
                plt.plot(x, label=label, color=color)
                plt.fill_between(np.arange(len(x)), plus, minus, color=color, alpha=0.2, label=label)
            else:
                NotImplementedError
            k += 1

    def plot_states_and_rewards(self, states, rewards, step):
        """
        Plots the states and rewards for a collected episode.
        """
        # states
        plt.figure()
        dim_obs = states.shape[1]
        for i in range(dim_obs):
            plt.subplot(dim_obs, 1, i+1)
            states_i = states[:-1, i].cpu().numpy()
            plt.plot(states_i.squeeze(), 'o', label='state', color='k', markersize=2)
        self.experiment.log_figure(figure=plt, figure_name='states_ts_'+str(step))
        plt.close()

        # rewards
        plt.figure()
        rewards = rewards[:-1, 0].cpu().numpy()
        plt.plot(rewards.squeeze(), 'o', label='reward', color='k', markersize=2)
        self.experiment.log_figure(figure=plt, figure_name='rewards_ts_'+str(step))
        plt.close()

    def plot_episode(self, episode, step):
        """
        Plots a newly collected episode.
        """
        self.experiment.log_metric('cumulative_reward', episode['reward'].sum(), step)

        def merge_legends():
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)

            plt.legend(newHandles, newLabels)

        for k in episode['distributions'].keys():
            for i, l in enumerate(episode['distributions'][k].keys()):
                color = COLORS[i]
                self._plot_ts(k, episode[k], episode['distributions'][k][l], l, color)
            plt.suptitle(k)
            merge_legends()
            self.experiment.log_figure(figure=plt, figure_name=k + '_ts_'+str(step))
            plt.close()

        self.plot_states_and_rewards(episode['state'], episode['reward'], step)

    def log_eval(self, episode, eval_states, step):
        """
        Plots an evaluation episode performance. Logs the episode.

        Args:
            episode (dict): dictionary containing agent's collected episode
            eval_states (dict): dictionary of MuJoCo simulator states
            step (int): the current step number in training
        """
        # plot and log eval returns
        eval_return = episode['reward'].sum()
        self.experiment.log_metric('eval_cumulative_reward', eval_return, step)
        self.returns.append(eval_return.item())
        json_str = json.dumps(self.returns)
        self.experiment.log_asset_data(json_str, name='eval_returns', overwrite=True)

        # log the episode itself
        for ep_item_str in ['state', 'action', 'reward']:
            ep_item = episode[ep_item_str].tolist()
            json_str = json.dumps(ep_item)
            item_name = 'episode_step_' + str(step) + '_' + ep_item_str
            self.experiment.log_asset_data(json_str, name=item_name)

        # log the MuJoCo simulator states
        for sim_item_str in ['qpos', 'qvel']:
            if len(eval_states[sim_item_str]) > 0:
                sim_item = eval_states[sim_item_str].tolist()
                json_str = json.dumps(sim_item)
                item_name = 'episode_step_' + str(step) + '_' + sim_item_str
                self.experiment.log_asset_data(json_str, name=item_name)

    def plot_agent_kl(self, agent_kl, step):
        self.experiment.log_metric('agent_kl', agent_kl, step)

    def log_results(self, results):
        """
        Log the results dictionary.
        """
        if self.result_dict is None:
            self.result_dict = {}
        for k, v in flatten_arg_dict(results).items():
            if k not in self.result_dict:
                self.result_dict[k] = [v]
            else:
                self.result_dict[k].append(v)

    def plot_results(self, timestep):
        """
        Plot/log the results to Comet.
        """
        for k, v in self.result_dict.items():
            avg_value = np.mean(v)
            self.experiment.log_metric(k, avg_value, timestep)
        self.result_dict = None

    def plot_model_eval(self, episode, predictions, log_likelihoods, step):
        """
        Plot/log the results from model evaluation.

        Args:
            episode (dict): a collected episode
            predictions (dict): predictions from each state, containing [n_steps, horizon, n_dims]
            log_likelihoods (dict): log-likelihood evaluations of predictions, containing [n_steps, horizon, 1]
        """
        for variable, lls in log_likelihoods.items():
            # average the log-likelihood estimates and plot the result at the horizon length
            mean_ll = lls[:, -1].mean().item()
            self.experiment.log_metric(variable + '_pred_log_likelihood', mean_ll, step)
            # plot log-likelihoods as a function of rollout step
            plt.figure()
            mean = lls.mean(dim=0).view(-1)
            std = lls.std(dim=0).view(-1)
            plt.plot(mean.numpy())
            lower = mean - std
            upper = mean + std
            plt.fill_between(np.arange(lls.shape[1]), lower.numpy(), upper.numpy(), alpha=0.2)
            plt.xlabel('Rollout Step')
            plt.ylabel('Prediction Log-Likelihood')
            plt.xticks(np.arange(lls.shape[1]))
            self.experiment.log_figure(figure=plt, figure_name=variable + '_pred_ll_' + str(step))
            plt.close()

        # plot predictions vs. actual values for an arbitrary time step
        time_step = np.random.randint(predictions['state']['loc'].shape[0])
        for variable, preds in predictions.items():
            pred_loc = preds['loc'][time_step]
            pred_scale = preds['scale'][time_step]
            x = episode[variable][time_step+1:time_step+1+pred_loc.shape[0]]
            plt.figure()
            horizon, n_dims = pred_loc.shape
            for plot_num in range(n_dims):
                plt.subplot(n_dims, 1, plot_num + 1)
                plt.plot(pred_loc[:, plot_num].numpy())
                lower = pred_loc[:, plot_num] - pred_scale[:, plot_num]
                upper = pred_loc[:, plot_num] + pred_scale[:, plot_num]
                plt.fill_between(np.arange(horizon), lower.numpy(), upper.numpy(), alpha=0.2)
                plt.plot(x[:, plot_num].numpy(), '.')
            plt.xlabel('Rollout Step')
            plt.xticks(np.arange(horizon))
            self.experiment.log_figure(figure=plt, figure_name=variable + '_pred_' + str(step))
            plt.close()

    def save_checkpoint(self, step):
        """
        Checkpoint the model by getting the state dictionary for each component.
        """
        print('Checkpointing the agent...')
        state_dict = self.agent.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        ckpt_path = os.path.join('./ckpt_step_'+ str(step) + '.ckpt')
        torch.save(cpu_state_dict, ckpt_path)
        self.experiment.log_asset(ckpt_path)
        os.remove(ckpt_path)
        print('Done.')

    def load_checkpoint(self, timestep=None):
        """
        Loads a checkpoint from Comet.

        Args:
            timestep (int, optional): the checkpoint timestep, default is latest
        """
        load_checkpoint(self.agent, self.exp_args.checkpoint_exp_key, timestep)
