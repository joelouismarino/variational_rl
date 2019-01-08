from time import strftime
import pickle
import os
import torch

log_items = ['free_energy', 'state_kl', 'action_kl', 'obs_cll', 'reward_cll',
             'optimality_cll', 'state_inf_imp']

# TODO: implement saving of episode frames (both observations and predictions)
#       and maybe also distributions of all variables

# TODO: implement loading model and log from saved files

class Logger:
    """
    A logger class to handle logging steps and episodes. Maintains a log of each
    episode. Writes log to file at the end of each episode.

    Args:
        log_dir (str): path to the directory of all logs
        log_str (str): name of the log (date and time)
        chkpt_interval (int): episode interval for model checkpointing
    """
    def __init__(self, log_dir, log_str=None, ckpt_interval=1000):
        self.log_dir = log_dir
        if log_str is not None:
            self.log_str = log_str
        else:
            self.log_str = strftime("%b_%d_%Y_%H_%M_%S")
        self.log_path = os.path.join(self.log_dir, self.log_str)
        os.makedirs(self.log_path)
        self.episode_log = {}
        self._init_episode_log()
        self._episode = 0
        self._ckpt_interval = ckpt_interval

    def _init_episode_log(self):
        self.episode_log = {}
        for metric in log_items:
            self.episode_log[metric] = []

    def log_step(self, model, observation, reward):
        # metrics
        self.episode_log['free_energy'].append(model.free_energy(observation, reward).item())
        self.episode_log['state_kl'].append(model.state_variable.kl_divergence().sum().item())
        self.episode_log['action_kl'].append(model.action_variable.kl_divergence().sum().item())
        self.episode_log['obs_cll'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
        if reward is not None:
            self.episode_log['reward_cll'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
            self.episode_log['optimality_cll'].append(reward)
        else:
            self.episode_log['reward_cll'].append(0)
            self.episode_log['optimality_cll'].append(0)

        # inference improvement
        state_inf_improvement = model.state_inf_free_energies[0] - model.state_inf_free_energies[-1]
        state_inf_improvement /= model.state_inf_free_energies[0]
        state_inf_improvement *= 100.
        self.episode_log['state_inf_imp'].append(state_inf_improvement.item())

        if self._episode % self._ckpt_interval == 0:
            pass
            # TODO: save observation, prediction, and reconstruction

    def log_episode(self, model, grads):

        def update_metric(file_name, value):
            if os.path.exists(file_name):
                metric = pickle.load(open(file_name, 'rb'))
                metric.extend(value)
                pickle.dump(metric, open(file_name, 'wb'))
            else:
                pickle.dump(value, open(file_name, 'wb'))

        for metric_name, metric_values in self.episode_log.items():
            update_metric(os.path.join(self.log_path, metric_name + '.p'), metric_values)

        if self._episode % self._ckpt_interval == 0:
            self.checkpoint(model)

        # copy the episode log and add the gradient norm
        episode_log = {k: v for k, v in self.episode_log.items()}
        for model_name, grad in grads.items():
            grad_norm = torch.cat([g.view(-1) for g in grad], dim=0).norm()
            episode_log[model_name + '_grad'] = [grad_norm.item()]

        self._init_episode_log()

        return episode_log

    def checkpoint(self, model):
        # checkpoint the model, save episode metrics/observations?
        pass

    # def add_log(self, step_num, model, observation, reward):
    #     self.log['Step'].append(step_num)
    #     self.log['Free Energy'].append(model.free_energy(observation, reward).item())
    #     self.log['KL'].append(model.kl_divergence().sum().item())
    #     self.log['State KL'].append(model.state_variable.kl_divergence().sum().item())
    #     self.log['Action KL'].append(model.action_variable.kl_divergence().sum().item())
    #     self.log['CLL'].append(model.cond_log_likelihood(observation, reward).item())
    #     self.log['Obs CLL'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
    #     if reward is not None:
    #         self.log['Reward CLL'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
    #         self.log['Optimality CLL'].append(reward)
    #     else:
    #         self.log['Reward CLL'].append(0)
    #         self.log['Optimality CLL'].append(0)
    #     state_inf_improvement = model.state_inf_free_energies[0] - model.state_inf_free_energies[-1]
    #     state_inf_improvement /= model.state_inf_free_energies[0]
    #     state_inf_improvement *= 100.
    #     self.log['State Inf. Improvement'].append(state_inf_improvement.item())
