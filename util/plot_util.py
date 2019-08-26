from comet_ml import Experiment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


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

class Plotter:
    """
    Handles plotting and logging to comet.

    Args:
        exp_args (args.parse_args): arguments for the experiment
        agent_args (dict): arguments for the agent
    """
    def __init__(self, exp_args, agent_args):
        self.experiment = Experiment(api_key='prsuXaz6RVyjfIWmbZwVjWMug',
                                     project_name='variational-rl',
                                     workspace="joelouismarino")
        self.exp_args = exp_args
        self.agent_args = agent_args
        self.experiment.disable_mp()
        self.experiment.log_parameters(get_arg_dict(exp_args))
        self.experiment.log_parameters(flatten_arg_dict(agent_args))

    def _plot_ts(self, observations, statistics, label, color):
        dim_obs = min(observations.shape[1], 9)
        k = 1
        for i in range(dim_obs):
            plt.subplot(int(str(dim_obs) + '1' + str(k)))
            observations_i = observations[:, i].cpu().numpy()
            plt.plot(observations_i.squeeze(), 'o', label='observation', color='k', markersize=2)
            if len(statistics) == 1:  # Bernoulli distribution
                probs = statistics['probs']
                probs = probs.cpu().numpy()
                plt.plot(probs, label=label, color=color)
            elif len(statistics) == 2:  # Normal distribution
                mean = statistics['loc']
                std = statistics['scale']
                mean = mean[:, i].cpu().numpy()
                std = std[:, i].cpu().numpy()
                mean = mean.squeeze()
                std = std.squeeze()
                x, plus, minus = mean, mean + std, mean - std
                if self.agent_args['action_variable_args']['approx_post_dist'] == 'TransformedTanh':
                    x, plus, minus = np.tanh(x), np.tanh(plus), np.tanh(minus)
                plt.plot(x, label=label, color=color)
                plt.fill_between(np.arange(len(mean)), plus, minus, color=color, alpha=0.2, label=label)
            else:
                NotImplementedError
            k += 1

    def plot_episode(self, episode, step):
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
            i = 0  # TODO: get rid of this hack
            for l in episode['distributions'][k].keys():
                color = 'b' if i == 0 else 'g'
                self._plot_ts(episode[k], episode['distributions'][k][l], l, color)
                i += 1
            plt.suptitle(k)
            merge_legends()
            self.experiment.log_figure(figure=plt, figure_name=k + '_ts_'+str(step))
            plt.close()

    def log_results(self, results, timestep):
        for n, m in flatten(results).items():
            self.experiment.log_metric(n, m, timestep)

    def checkpoint_model(self):
        # checkpoint the model by getting the state dictionary for each component
        state_dict = {}
        variable_names = ['state_variable', 'action_variable',
                          'observation_variable', 'reward_variable',
                          'done_variable', 'value_variable']
        model_names = ['state_prior_model', 'action_prior_model',
                       'obs_likelihood_model', 'reward_likelihood_model',
                       'done_likelihood_model', 'value_model',
                       'state_inference_model', 'action_inference_model']

        for attr in variable_names + model_names:
            if hasattr(self.agent, attr):
                if hasattr(getattr(self.agent, attr), 'state_dict'):
                     sd = getattr(self.agent, attr).state_dict()
                     state_dict[attr] = {k: v.cpu() for k, v in sd.items()}

        # save the state dictionaries
        ckpt_path = os.path.join('./ckpt_episode_'+str(self._episode) + '.ckpt')
        torch.save(state_dict, ckpt_path)
        self.experiment.log_asset(ckpt_path)
        os.remove(ckpt_path)
