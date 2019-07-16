from time import strftime
import pickle
import numpy as np
import os
import torch
import random

log_items = ['optimality', 'state', 'action', 'state_improvement']

class Logger:
    """
    A logger class to handle logging training steps and episodes.

    Args:
        log_dir (str): path to the directory of all logs
        log_str (str): name of the log (date and time)
        chkpt_interval (int): interval for model checkpointing (in episodes)
    """
    def __init__(self, log_dir, exp_args, agent, log_str=None, ckpt_interval=250):
        self.log_dir = log_dir
        if log_str is not None:
            self.log_str = log_str
        else:
            self.log_str = strftime("%b_%d_%Y_%H_%M_%S")
        self.log_path = os.path.join(self.log_dir, self.log_str)
        os.makedirs(self.log_path)
        os.makedirs(os.path.join(self.log_path, 'metrics'))
        os.makedirs(os.path.join(self.log_path, 'vis'))
        os.makedirs(os.path.join(self.log_path, 'checkpoints'))
        self.save_exp_config(exp_args)
        self.agent = agent
        self._train_step = 0
        self._episode = 0
        self._ckpt_interval = ckpt_interval
        self._init_eval_stats()

    def _init_eval_stats(self):
        stats = ['rewards', 'observations', 'predictions', 'reconstructions', 'episode length']
        self.eval_statistics = {}
        for stat in stats:
            self.eval_statistics[stat] = []

    def save_exp_config(self, args):
        agent_args = args['agent_args']
        exp_config_str = 'EXPERIMENT CONFIG\n'
        for arg_name, arg in args.items():
            arg_str = arg_name + ':'
            arg_str += str(arg) + '\n'
            exp_config_str += arg_str
        agent_config_str = '\nAGENT CONFIG\n'
        for arg_name, arg in agent_args.items():
            arg_str = arg_name + ':'
            arg_str += str(arg) + '\n'
            agent_config_str += arg_str
        # save to text file
        with open(f"{self.log_path}/config.txt", "w") as text_file:
            text_file.write(exp_config_str + agent_config_str)

    def _update_metric(self, file_name, value):
        # appends a new value to the metric list
        if os.path.exists(file_name):
            metric = pickle.load(open(file_name, 'rb'))
            metric.append(value)
            pickle.dump(metric, open(file_name, 'wb'))
        else:
            pickle.dump([value], open(file_name, 'wb'))

    def log_train_step(self, results):
        # log a training step
        for metric_name in log_items:
            if metric_name in results.keys():
                metric = results[metric_name]
                self._update_metric(os.path.join(self.log_path, 'metrics', metric_name + '.p'), metric)
        self._train_step += 1

    def log_episode(self, episode):
        # log rewards
        self.eval_statistics['rewards'].extend(episode['reward'])
        # pick observations to log
        frame_samples = random.sample(range(0, len(episode['observation'])), 2) # select 2 random frames
        self.eval_statistics['observations'].extend(episode['observation'][frame_samples])
        if 'prediction' in episode:
            self.eval_statistics['predictions'].extend(episode['prediction'][frame_samples])
        if 'reconstruction' in episode:
            self.eval_statistics['reconstructions'].append(episode['reconstruction'][frame_samples])
        self.eval_statistics['episode length'].append(len(episode['reward']))
        # save
        file_name = os.path.join(self.log_path, 'metrics', 'eval_statistics.p')
        pickle.dump(self.eval_statistics, open(file_name, 'wb'))

        if self._episode % self._ckpt_interval == 0:
            self.checkpoint()

        self._episode += 1

    def checkpoint(self):
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

        ckpt_path = os.path.join(self.log_path, 'checkpoints', 'ckpt_episode_'+str(self._episode) + '.ckpt')
        torch.save(state_dict, ckpt_path)
