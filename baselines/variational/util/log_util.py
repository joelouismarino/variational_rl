from time import strftime
import pickle
import numpy as np
import os
import torch
import random

# log_items = ['free_energy', 'state_kl', 'action_kl', 'obs_cll', 'done_cll',
#              'reward_cll', 'optimality_cll', 'state_inf_imp', 'state_approx_post_mean',
#              'state_approx_post_log_std', 'state_prior_mean', 'state_prior_log_std',
#              'obs_cond_likelihood_mean', 'obs_cond_likelihood_log_std',
#              'reward_cond_likelihood_mean', 'reward_cond_likelihood_log_std',
#              'done_cond_likelihood_mean']

# log_items = ['free_energy', 'state_kl', 'action_kl', 'obs_cll', 'done_cll',
#              'reward_cll', 'optimality_cll', 'state_inf_imp']

log_items = ['optimality', 'state', 'action', 'state_inf_imp']

class Logger:
    """
    A logger class to handle logging training steps and episodes.

    Args:
        log_dir (str): path to the directory of all logs
        log_str (str): name of the log (date and time)
        chkpt_interval (int): interval for model checkpointing (in train steps)
    """
    def __init__(self, log_dir, exp_args, log_str=None, ckpt_interval=5000):
        self.log_dir = log_dir
        if log_str is not None:
            self.log_str = log_str
        else:
            self.log_str = strftime("%b_%d_%Y_%H_%M_%S")
        self.log_path = os.path.join(self.log_dir, self.log_str)
        os.makedirs(self.log_path)
        os.makedirs(os.path.join(self.log_path, 'metrics'))
        # os.makedirs(os.path.join(self.log_path, 'episodes'))
        # self.episode_log = {}
        # self._init_episode_log()
        self._train_step = 0
        # self._episode = 0
        self._ckpt_interval = ckpt_interval
        self._init_eval_stats()
        # self._saved_episode = {'reconstruction': [], 'prediction': [],
        #                        'observation': []}

    def _init_eval_stats(self):
        stats = ['rewards', 'observations', 'predictions', 'reconstruction']
        self.eval_statistics = {}
        for stat in stats:
            self.eval_statistics[stat] = []

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
        # if self._train_step % self._ckpt_interval == 0:
        #     self.checkpoint(model)

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
        # save
        file_name = os.path.join(self.log_path, 'metrics', 'eval_statistics.p')
        pickle.dump(self.eval_statistics, open(file_name, 'wb'))

    def checkpoint(self, model):
        # checkpoint the model
        pass

    # def _init_episode_log(self):
    #     self.episode_log = {}
    #     for metric in log_items:
    #         self.episode_log[metric] = []
    #
    # def log_step(self, model, observation, reward, done):
    #     # metrics
    #     self.episode_log['free_energy'].append(model.free_energy(observation, reward, done).item())
    #     self.episode_log['state_kl'].append(model.state_variable.kl_divergence().sum().item())
    #     self.episode_log['action_kl'].append(model.action_variable.kl_divergence().sum().item())
    #     self.episode_log['obs_cll'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
    #     self.episode_log['done_cll'].append(model.done_variable.cond_log_likelihood(done).sum().item())
    #     if reward is not None:
    #         self.episode_log['reward_cll'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
    #         self.episode_log['optimality_cll'].append(model.optimality_scale * (reward - 1.))
    #     else:
    #         self.episode_log['reward_cll'].append(0)
    #         self.episode_log['optimality_cll'].append(0)
    #
    #     # inference improvement
    #     state_inf_improvement = model.state_inf_free_energies[0] - model.state_inf_free_energies[-1]
    #     state_inf_improvement /= model.state_inf_free_energies[0]
    #     state_inf_improvement *= 100.
    #     self.episode_log['state_inf_imp'].append(state_inf_improvement.item())
    #
    #     # distributions
    #     self.episode_log['state_approx_post_mean'].append(model.state_variable.approx_post_dist.loc.mean().item())
    #     self.episode_log['state_approx_post_log_std'].append(model.state_variable.approx_post_dist.scale.log().mean().item())
    #     self.episode_log['state_prior_mean'].append(model.state_variable.prior_dist.loc.mean().item())
    #     self.episode_log['state_prior_log_std'].append(model.state_variable.prior_dist.scale.log().mean().item())
    #
    #     self.episode_log['obs_cond_likelihood_mean'].append(model.observation_variable.likelihood_dist.loc.mean().item())
    #     self.episode_log['obs_cond_likelihood_log_std'].append(model.observation_variable.likelihood_dist.scale.log().mean().item())
    #
    #     self.episode_log['reward_cond_likelihood_mean'].append(model.reward_variable.likelihood_dist.loc.mean().item())
    #     self.episode_log['reward_cond_likelihood_log_std'].append(model.reward_variable.likelihood_dist.scale.log().mean().item())
    #
    #     self.episode_log['done_cond_likelihood_mean'].append(model.done_variable.likelihood_dist.logits.sigmoid().mean().item())
    #
    #     # TODO: log action and optimality (discrete distributions)
    #
    #     if self._episode % self._ckpt_interval == 0:
    #         self._saved_episode['reconstruction'].append(model.obs_reconstruction.detach().cpu().numpy()[0])
    #         self._saved_episode['prediction'].append(model.obs_prediction.detach().cpu().numpy()[0])
    #         self._saved_episode['observation'].append(observation.detach().cpu().numpy()[0])
    #
    # def log_episode(self, model, grads):
    #
    #     def update_metric(file_name, value):
    #         if os.path.exists(file_name):
    #             metric = pickle.load(open(file_name, 'rb'))
    #             metric.extend(value)
    #             pickle.dump(metric, open(file_name, 'wb'))
    #         else:
    #             pickle.dump(value, open(file_name, 'wb'))
    #
    #     for metric_name, metric_values in self.episode_log.items():
    #         update_metric(os.path.join(self.log_path, metric_name + '.p'), metric_values)
    #
    #     if self._episode % self._ckpt_interval == 0:
    #         self.checkpoint(model)
    #         self._save_episode()
    #
    #     # copy the episode log and add the gradient mean
    #     episode_log = {k: v for k, v in self.episode_log.items()}
    #     for model_name, grad in grads.items():
    #         grad_mean = torch.cat([g.view(-1) for g in grad], dim=0).abs().mean()
    #         episode_log[model_name + '_grad'] = [grad_mean.item()]
    #
    #     self._init_episode_log()
    #     self._episode += 1
    #     return episode_log
    #
    # def _save_episode(self):
    #     # make a directory for this episode
    #     episode_path = os.path.join(self.log_path, 'episodes', str(self._episode))
    #     os.makedirs(episode_path)
    #
    #     reconstructions = np.stack(self._saved_episode['reconstruction'])
    #     predictions = np.stack(self._saved_episode['prediction'])
    #     observations = np.stack(self._saved_episode['observation'])
    #
    #     pickle.dump(reconstructions, open(os.path.join(episode_path, 'reconstructions.p'), 'wb'))
    #     pickle.dump(predictions, open(os.path.join(episode_path, 'predictions.p'), 'wb'))
    #     pickle.dump(observations, open(os.path.join(episode_path, 'observations.p'), 'wb'))
    #
    #     self._saved_episode = {'reconstruction': [], 'prediction': [],
    #                            'observation': []}
    #
    # def checkpoint(self, model):
    #     # checkpoint the model, save episode metrics/observations?
    #     pass
