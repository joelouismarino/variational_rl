from visdom import Visdom
import numpy as np


class Plotter:
    """
    A plotter class to handle plotting logs to visdom.

    Args:
        log_str (str): name of visdom environment
    """
    def __init__(self, log_str):
        self.env_id = log_str
        self.vis = Visdom(env=self.env_id)
        self.metric_plot_names = ['free_energy', 'reward_cll', 'obs_cll',
                                  'optimality_cll', 'state_kl', 'action_kl',
                                  'state_inf_imp']
        self.dist_plot_names = ['state_approx_post_mean', 'state_approx_post_log_std',
                                'state_prior_mean', 'state_prior_log_std',
                                'obs_cond_likelihood_mean', 'obs_cond_likelihood_log_std',
                                'reward_cond_likelihood_mean', 'reward_cond_likelihood_log_std']
        self.dist_window_names = ['state_mean', 'state_log_std', 'obs_mean',
                                  'obs_log_std', 'reward_mean', 'reward_log_std']
        self.model_grad_plot_names = ['state_inference_model_grad', 'action_inference_model_grad',
                                      'state_prior_model_grad', 'action_prior_model_grad',
                                      'obs_likelihood_model_grad', 'reward_likelihood_model_grad']
        self.img_names = ['recon', 'obs', 'pred']
        windows = self.metric_plot_names + self.dist_window_names + self.img_names + ['grad_means', 'episode_length']
        self._init_windows(windows)
        self.smooth_reward_len = 1
        self._step = 1
        self._episode = 1

    def _init_windows(self, window_names):
        self.window_id = {}
        for w in window_names:
            self.window_id[w] = None

    def plot(self, episode_log):
        # plot metrics
        for metric_name in self.metric_plot_names:
            self._plot_metric(episode_log[metric_name], metric_name,
                              opts=self._get_opts(metric_name))

        # plot the distribution statistics
        for dist_param_name in self.dist_plot_names:
            self._plot_dist(episode_log[dist_param_name], dist_param_name)

        # plot the episode length
        self._plot_episode_length(len(episode_log['free_energy']))

        # plot gradient means
        self._plot_grad_means(episode_log)

        # increment the step and episode counters
        self._step += len(episode_log['free_energy'])
        self._episode += 1
        self.vis.save([self.env_id])

    def _plot_metric(self, metric, win_name, opts=None):
        steps = list(range(self._step, self._step + len(metric)))
        if self.window_id[win_name] is not None:
            self.vis.line(X=steps, Y=metric, update='append', name='Step', win=self.window_id[win_name])
            self.vis.line(X=[steps[-1]], Y=[np.mean(metric)], update='append', name='Episode', win=self.window_id[win_name])
        else:
            self.window_id[win_name] = self.vis.line(X=steps, Y=metric, name='Step', opts=opts)
            self.vis.line(X=[steps[-1]], Y=[np.mean(metric)], update='replace', name='Episode', win=self.window_id[win_name])

    def _plot_dist(self, dist_param, param_name):
        steps = list(range(self._step, self._step + len(dist_param)))
        win_name = self._get_window_name(param_name)
        if self.window_id[win_name] is not None:
            update = 'append' if self._step != 1 else 'replace'
            self.vis.line(X=steps, Y=dist_param, update=update, name=param_name + ' Step', win=self.window_id[win_name])
            self.vis.line(X=[steps[-1]], Y=[np.mean(dist_param)], update=update, name=param_name + ' Episode', win=self.window_id[win_name])
        else:
            opts = self._get_opts(win_name)
            self.window_id[win_name] = self.vis.line(X=steps, Y=dist_param, name=param_name + ' Step', opts=opts)
            self.vis.line(X=[steps[-1]], Y=[np.mean(dist_param)], update='replace', name=param_name + ' Episode', win=self.window_id[win_name])

    def _plot_grad_means(self, episode_log):
        if self.window_id['grad_means'] is not None:
            for model_grad_name in self.model_grad_plot_names:
                self.vis.line(X=[self._step], Y=episode_log[model_grad_name], update='append', name=model_grad_name, win=self.window_id['grad_means'])
        else:
            for model_grad_name in self.model_grad_plot_names:
                if self.window_id['grad_means'] is not None:
                    self.vis.line(X=[self._step], Y=episode_log[model_grad_name], update='replace', name=model_grad_name, win=self.window_id['grad_means'])
                else:
                    opts = self._get_opts('grad_means')
                    self.window_id['grad_means'] = self.vis.line(X=[self._step], Y=episode_log[model_grad_name], name=model_grad_name, opts=opts)

    def _plot_episode_length(self, length):
        if self.window_id['episode_length'] is not None:
            self.vis.line(X=[self._episode], Y=[length], update='append', name='Episode Length', win=self.window_id['episode_length'])
        else:
            self.window_id['episode_length'] = self.vis.line(X=[self._episode], Y=[length], name='Episode Length', opts=self._get_opts('episode_length'))

    def plot_image(self, img, title, size=(200,200)):

        def preprocess_image(image):
            if type(img) != np.ndarray:
                # convert from torch
                image = image.detach().cpu().numpy()
            if len(img.shape) == 4:
                # remove batch dimension
                image = image[0]
            return image

        if title == 'Reconstruction':
            id = 'recon'
        elif title == 'Observation':
            id = 'obs'
        elif title == 'Prediction':
            id = 'pred'
        else:
            id = title

        img = preprocess_image(img)
        opts=dict(width=size[1], height=size[0], title=title)
        if self.window_id[id] is not None:
            self.vis.image(img, win=self.window_id[id], opts=opts)
        else:
            self.window_id[id] = self.vis.image(img, opts=opts)

    def _get_window_name(self, plot_name):
        if plot_name == 'state_approx_post_mean' or plot_name == 'state_prior_mean':
            return 'state_mean'
        elif plot_name == 'state_approx_post_log_std' or plot_name == 'state_prior_log_std':
            return 'state_log_std'
        elif plot_name == 'obs_cond_likelihood_mean':
            return 'obs_mean'
        elif plot_name == 'obs_cond_likelihood_log_std':
            return 'obs_log_std'
        elif plot_name == 'reward_cond_likelihood_mean':
            return 'reward_mean'
        elif plot_name == 'reward_cond_likelihood_log_std':
            return 'reward_log_std'

    def _get_opts(self, win_name):
        xlabel = 'Time Step'
        ylabel = ''
        width = 450
        height = 320
        title = ''
        xtype = 'log'
        showlegend = True
        if win_name == 'free_energy':
            ylabel = 'Free Energy (nats)'
            title = 'Free Energy'
        elif win_name == 'state_kl':
            ylabel = 'State KL (nats)'
            title = 'State KL'
        elif win_name == 'action_kl':
            ylabel = 'Action KL (nats)'
            title = 'Action KL'
        elif win_name == 'obs_cll':
            ylabel = 'Obs. Cond. Log Likelihood (nats)'
            title = 'Obs. Cond. Log Likelihood'
        elif win_name == 'reward_cll':
            ylabel = 'Reward Cond. Log Likelihood (nats)'
            title = 'Reward Cond. Log Likelihood'
            xtype = 'line'
        elif win_name == 'optimality_cll':
            ylabel = 'Optimality Cond. Log Likelihood (nats)'
            title = 'Optimality Cond. Log Likelihood'
            xtype = 'line'
        elif win_name == 'state_inf_imp':
            ylabel = 'Improvement (percent)'
            title = 'State Inference Improvement'
        elif win_name == 'episode_length':
            ylabel = 'Episode Length (steps)'
            title = 'Episode Length'
            showlegend = False
            xlabel = 'Episode'
            xtype = 'line'
        elif 'grad' in win_name:
            ylabel = 'Gradient Means'
            title = 'Gradient Means'
        elif win_name == 'state_mean':
            ylabel = 'Average State Mean'
            title = 'Average State Mean'
        elif win_name == 'state_log_std':
            ylabel = 'Average State Log Std. Dev.'
            title = 'Average State Log Std. Dev.'
        elif win_name == 'obs_mean':
            ylabel = 'Obs. Conditional Likelihood Mean'
            title = 'Obs. Conditional Likelihood Mean'
        elif win_name == 'obs_log_std':
            ylabel = 'Obs. Conditional Likelihood Log Std. Dev.'
            title = 'Obs. Conditional Likelihood Log Std. Dev.'
        elif win_name == 'reward_mean':
            ylabel = 'Reward Conditional Likelihood Mean'
            title = 'Reward Conditional Likelihood Mean'
        elif win_name == 'reward_log_std':
            ylabel = 'Reward Conditional Likelihood Log Std. Dev.'
            title = 'Reward Conditional Likelihood Log Std. Dev.'

        opts = dict(xlabel=xlabel, ylabel=ylabel, title=title, width=width,
                    height=height, xtype=xtype, showlegend=showlegend)
        return opts
