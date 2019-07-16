from comet_ml import Experiment
import matplotlib.pyplot as plt
import pandas as pd


def flatten(dictionary):
    return pd.io.json.json_normalize(dictionary, sep='_').to_dict()

def get_arg_dict(args):
    arg_list = args._get_kwargs()
    arg_dict = {k: v for (k, v) in arg_list}
    return arg_dict

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
        self.experiment.disable_mp()
        self.experiment.log_parameters(get_arg_dict(exp_args))
        self.experiment.log_parameters(agent_args)

    def _plot_ts(self, actions, step, name):
        nb_actions = min(9, actions.shape[1])
        k = 1
        j = str(nb_actions)
        for i in range(nb_actions):
            plt.subplot(int(j + '1' + str(k)))
            plt.plot(actions[i])
            k += 1

        self.experiment.log_figure(figure=plt, figure_name=name + '_ts_'+str(step))
        plt.close()

    def plot_episode(self, episode, step):
        episode = flatten(episode)
        for n, m in episode.items():
            if len(m[0]) > 0:
                m = pd.DataFrame(m[0].cpu().numpy())
                self._plot_ts(m, step, n)

    def log_results(self, results):
        for n, m in results.items():
            self.experiment.log_metric(n, m, timestep)
