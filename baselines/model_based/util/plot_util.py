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
        self.model_grad_plot_names = ['state_inference_model_grad', 'action_inference_model_grad',
                                      'state_prior_model_grad', 'action_prior_model_grad',
                                      'obs_likelihood_model_grad', 'reward_likelihood_model_grad']
        self.img_names = ['recon', 'obs', 'pred']
        self._init_windows(self.metric_plot_names + ['grad_norms'] + self.img_names)
        self.smooth_reward_len = 1
        self._step = 1

    def _init_windows(self, window_names):
        self.window_id = {}
        for w in window_names:
            self.window_id[w] = None

    def plot(self, episode_log):
        # plot metrics
        for metric_name in self.metric_plot_names:
            self._plot_metric(episode_log[metric_name], metric_name,
                              opts=self._get_opts(metric_name))

        # plot gradient norms
        self._plot_grad_norms(episode_log)

        # increment the step counter
        self._step += len(episode_log['free_energy'])
        self.vis.save([self.env_id])

    def _plot_metric(self, metric, win_name, opts=None):
        steps = list(range(self._step, self._step + len(metric)))
        if self.window_id[win_name] is not None:
            self.vis.line(X=steps, Y=metric, update='append', name='Step', win=self.window_id[win_name])
            self.vis.line(X=[steps[-1]], Y=[np.mean(metric)], update='append', name='Episode', win=self.window_id[win_name])
        else:
            self.window_id[win_name] = self.vis.line(X=steps, Y=metric, name='Step', opts=opts)
            self.vis.line(X=[steps[-1]], Y=[np.mean(metric)], update='replace', name='Episode', win=self.window_id[win_name])

    def _plot_grad_norms(self, episode_log):

        if self.window_id['grad_norms'] is not None:
            for model_grad_name in self.model_grad_plot_names:
                self.vis.line(X=[self._step], Y=episode_log[model_grad_name], update='append', name=model_grad_name, win=self.window_id['grad_norms'])
        else:
            for model_grad_name in self.model_grad_plot_names:
                if self.window_id['grad_norms'] is not None:
                    self.vis.line(X=[self._step], Y=episode_log[model_grad_name], update='replace', name=model_grad_name, win=self.window_id['grad_norms'])
                else:
                    opts = self._get_opts('grad_norms')
                    self.window_id['grad_norms'] = self.vis.line(X=[self._step], Y=episode_log[model_grad_name], name=model_grad_name, opts=opts)

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
        else:
            id = title

        img = preprocess_image(img)
        opts=dict(width=size[1], height=size[0], title=title)
        if self.window_id[id] is not None:
            self.vis.image(img, win=self.window_id[id], opts=opts)
        else:
            self.window_id[id] = self.vis.image(img, opts=opts)

    def _get_opts(self, win_name):
        xlabel = 'Time Step'
        ylabel = ''
        width = 450
        height = 320
        title = ''
        xformat = 'log'
        showlegend=True
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
        elif win_name == 'optimality_cll':
            ylabel = 'Optimality Cond. Log Likelihood (nats)'
            title = 'Optimality Cond. Log Likelihood'
        elif win_name == 'state_inf_imp':
            ylabel = 'Improvement (percent)'
            title = 'State Inference Improvement'
        elif win_name == 'episode_length':
            ylabel = 'Episode Length (steps)'
            title = 'Episode Length'
        elif 'grad' in win_name:
            ylabel = 'Gradient Norms'
            title = 'Gradient Norms'

        opts = dict(xlabel=xlabel, ylabel=ylabel, title=title, width=width,
                    height=height, xformat=xformat, showlegend=showlegend)
        return opts

    # def plot_kl(self, log):
    #     if self.window_id['kl'] is not None:
    #         vis.line(X=log['Step'], Y=log['KL'], update='replace', name='KL', win=self.window_id['kl'])
    #         vis.line(X=log['Step'], Y=log['State KL'], update='replace', name='State KL', win=self.window_id['kl'])
    #         vis.line(X=log['Step'], Y=log['Action KL'], update='replace', name='Action KL', win=self.window_id['kl'])
    #     else:
    #         self.window_id['kl'] = vis.line(X=log['Step'], Y=log['KL'], name='KL',
    #                                         opts=dict(
    #                                        xlabel='Timestep',
    #                                        ylabel='KL',
    #                                        width=450,
    #                                        height=320,
    #                                             title = 'KL (State and Action)'
    #                                         )
    #                                         )
    #         vis.line(X=log['Step'], Y=log['State KL'], update = 'append', name='State KL', win=self.window_id['kl'])
    #         vis.line(X=log['Step'], Y=log['Action KL'], update = 'append', name='Action KL', win=self.window_id['kl'])

    # def plot_cll(self, log):
    #     if self.window_id['cll'] is not None:
    #         vis.line(X=log['Step'], Y=log['CLL'], update='replace', name='CLL', win=self.window_id['cll'])
    #         # vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'replace', name='Reward', win=self.window_id['cll'])
    #         # vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'replace', name='Optimality', win=self.window_id['cll'])
    #         vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'replace', name='Observation', win=self.window_id['cll'])
    #
    #     else:
    #         self.window_id['cll'] = vis.line(X=log['Step'], Y=log['CLL'], name='CLL',
    #                                          opts=dict(
    #                                        xlabel='Timestep',
    #                                        ylabel='CLL',
    #                                        width=450,
    #                                        height=320,
    #                                         title = 'Observation'
    #                                          )
    #                                          )
    #         # vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'append', name='Reward', win=self.window_id['cll'])
    #         # vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'append', name='Optimality', win=self.window_id['cll'])
    #         vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'append', name='Observation', win=self.window_id['cll'])

    # def plot_reward_cll(self, log):
    #     if self.window_id['reward_cll'] is not None:
    #         vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'replace', name='Reward', win=self.window_id['reward_cll'])
    #         vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'replace', name='Optimality', win=self.window_id['reward_cll'])
    #         wind = 100
    #         if len(log['Step']) // wind > self.smooth_reward_len:
    #             self.smooth_reward_len = len(log['Step']) // wind
    #             # compute and plot smoothed line
    #             y = np.convolve(log['Reward CLL'], np.ones(wind)/wind, mode = 'valid')
    #             x = np.linspace(log['Step'][0]+wind//2, log['Step'][-1]-wind//2, len(y))
    #             vis.line(X = x, Y = y, update = 'replace', name = 'Smoothed Reward', win = self.window_id['reward_cll'])
    #
    #     else:
    #         self.window_id['reward_cll'] = vis.line(X=log['Step'], Y=log['Reward CLL'], name='CLL',
    #                                          opts=dict(
    #                                        xlabel='Timestep',
    #                                        ylabel='CLL',
    #                                        width=450,
    #                                        height=320,
    #                                         title = 'Reward'
    #                                          )
    #                                          )
    #         vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'append', name='Optimality', win=self.window_id['reward_cll'])
    #
    # def plot_state_inf_improvement(self, log):
    #     if self.window_id['state_inf_improvement'] is not None:
    #         vis.line(X=log['Step'], Y=log['State Inf. Improvement'], update='replace', name='State Inf. Improvement', win=self.window_id['state_inf_improvement'])
    #     else:
    #         self.window_id['state_inf_improvement'] = vis.line(X=log['Step'], Y=log['State Inf. Improvement'], name='State Inf. Improvement',
    #                                                  opts=dict(
    #                                                            xlabel='Timestep',
    #                                                            ylabel='Percent Improvement',
    #                                                            width=450,
    #                                                            height=320,
    #                                                  )
    #                                                  )
