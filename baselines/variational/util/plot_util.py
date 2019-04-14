from visdom import Visdom
import random
import numpy as np


class Plotter:
    """
    A plotter class to handle plotting logs to visdom.

    Args:
        exp_args (str): arguments for the experiment
    """
    def __init__(self, exp_args):
        self.env_id = exp_args['log_str']
        self.vis = Visdom(env=self.env_id)
        self.metric_plot_names = ['optimality', 'state', 'action', 'value']
        self.metric_plot_names += ['importance_weights', 'policy_gradients', 'advantages']
        self.episode_plot_names = ['length', 'env_return', 'total_steps']
        self.misc_plot_names = ['kl_min']
        self.img_names = ['obs']
        self.grad_names = ['grads', 'grad_norms']
        self.hist_names = ['actions']
        if exp_args['agent_args']['agent_type'] == 'generative':
            # additional plots for likelihoods and inference improvement
            self.metric_plot_names += ['observation', 'reward', 'done', 'state_improvement']
            self.metric_plot_names += ['observation_cll', 'observation_mll']
            self.metric_plot_names += ['reward_cll', 'reward_mll']
            self.metric_plot_names += ['done_cll', 'done_mll']
            # additional images for reconstruction and prediction
            self.img_names += ['recon', 'pred']
            if exp_args['agent_args']['misc_args']['n_inf_iter']['action'] > 0:
                # additional plots for planning inference
                self.metric_plot_names += ['rollout_length', 'action_improvement']
        windows = self.metric_plot_names + self.episode_plot_names + self.misc_plot_names + self.img_names + self.grad_names + self.hist_names
        self._init_windows(windows)
        self.plot_config(exp_args)
        # self.smooth_reward_len = 1
        self._step = 1
        self._episode = 1
        self._total_episode_steps = 0
        # maintain training metrics between plotting
        self._train_results = {}
        self._n_train_steps = 0

    def _init_windows(self, window_names):
        # initializes the windows with empty IDs and data
        self.window_id = {}
        self.data = {}
        for w in window_names:
            self.window_id[w] = None
            self.data[w] = []

    def plot_config(self, args):
        # plot the configuration hyperparameters
        agent_args = args.pop('agent_args')
        exp_config_str = '<b> EXPERIMENT CONFIG </b> <br/>'
        for arg_name, arg in args.items():
            arg_str = '<b> ' + arg_name + ':</b> '
            arg_str += str(arg) + '<br/>'
            exp_config_str += arg_str
        self.vis.text(exp_config_str)

        agent_config_str = '<b> AGENT CONFIG </b> <br/>'
        for arg_name, arg in agent_args.items():
            arg_str = '<b> ' + arg_name + ':</b> '
            arg_str += str(arg) + '<br/>'
            agent_config_str += arg_str
        self.vis.text(agent_config_str)

    def plot_train_step(self, results, plot=True):
        # plots metrics from a training step
        # add the current results to the internal training results
        self._n_train_steps += 1
        for result_name, result in results.items():
            if result_name not in self._train_results:
                if result_name in self.metric_plot_names:
                    self._train_results[result_name] = result
                elif result_name in ['grads', 'grad_norms', 'kl_min']:
                    self._train_results[result_name] = {k:v for k, v in results[result_name].items()}
                    # for key in results[result_name]:
                    #     self._train_results[result_name][key] = result[key]
            else:
                if result_name in self.metric_plot_names:
                    self._train_results[result_name] += result
                elif result_name in ['grads', 'grad_norms', 'kl_min']:
                    for key in results[result_name]:
                        self._train_results[result_name][key] += result[key]
        if plot:
            # average the internal training results
            for result_name in self._train_results:
                if result_name in self.metric_plot_names:
                    self._train_results[result_name] /= self._n_train_steps
                elif result_name in ['grads', 'grad_norms', 'kl_min']:
                    for key in self._train_results[result_name]:
                        self._train_results[result_name][key] /= self._n_train_steps
            # plot the averaged results
            for metric_name in self.metric_plot_names:
                if metric_name in results.keys():
                    self._plot_metric(self._step, results[metric_name], metric_name,
                                      opts=self._get_opts(metric_name))
            self._plot_grads(results['grads'], 'grads')
            self._plot_grads(results['grad_norms'], 'grad_norms')
            self._plot_kl_min(results['kl_min'], 'kl_min')
            self.vis.save([self.env_id])
            # reset the internal training results
            self._train_results = {}
            self._n_train_steps = 0
        self._step += 1

    def plot_episode(self, episode):
        # plots metrics from a newly collected episode
        n_steps = episode['observation'].shape[0] - 1
        self._total_episode_steps += n_steps
        time_step = random.randint(0, n_steps-1)
        if len(episode['observation'][time_step].size()) == 3:
            self.plot_image(episode['observation'][time_step], 'Observation')
            if 'prediction' in episode:
                self.plot_image(episode['prediction'][time_step], 'Prediction')
            if 'reconstruction' in episode:
                self.plot_image(episode['reconstruction'][time_step], 'Reconstruction')

        self._plot_metric(self._episode, n_steps, 'length',
                          opts=self._get_opts('length'), name='Episode')
        self._plot_metric(self._episode, episode['reward'].sum().item(), 'env_return',
                          opts=self._get_opts('env_return'), name='Episode')
        self._plot_metric(self._episode, self._total_episode_steps, 'total_steps',
                          opts=self._get_opts('total_steps'), name='Episode')
        action_idxs = np.argmax(episode['action'], axis=1)
        self._plot_hist(action_idxs, win_name = 'actions', opts = self._get_opts('actions'))
        self._episode += 1

    def _plot_metric(self, step, metric, win_name, opts=None, name='Train Step', plot_avg_trace = True, avg_window = 50):
        # plots metrics, such as losses
        if self.window_id[win_name] is not None:
            self.vis.line(X=[step], Y=[metric], update='append', name=name, win=self.window_id[win_name])
        else:
            self.window_id[win_name] = self.vis.line(X=[step], Y=[metric], name=name, opts=opts)
        if plot_avg_trace:
            self.data[win_name].append(metric)
            if len(self.data[win_name]) > avg_window:
                avg = np.average(self.data[win_name][-avg_window:])
                x = step - avg_window / 2
                self.vis.line(X=[x], Y=[avg], update='append', name='Moving Average', win=self.window_id[win_name])

    def _plot_hist(self, data, win_name, opts = None):
        # plots histograms
        if self.window_id[win_name] is not None:
            self.vis.histogram(X=data, win=self.window_id[win_name], opts = opts)
        else:
            self.window_id[win_name] = self.vis.histogram(X=data, opts = opts)

    def _plot_grads(self, grads, window_name):
        # plots gradient absolute values and norms
        if self.window_id[window_name] is not None:
            for model_name in grads:
                self.vis.line(X=[self._step], Y=[grads[model_name]], update='append', name=model_name, win=self.window_id[window_name])
        else:
            for model_name in grads:
                if self.window_id[window_name] is not None:
                    self.vis.line(X=[self._step], Y=[grads[model_name]], update='replace', name=model_name, win=self.window_id[window_name])
                else:
                    opts = self._get_opts(window_name)
                    self.window_id[window_name] = self.vis.line(X=[self._step], Y=[grads[model_name]], name=model_name, opts=opts)

    def _plot_kl_min(self, kl_min, window_name):
        # plots kl min values
        if self.window_id[window_name] is not None:
            for variable_name in kl_min:
                self.vis.line(X=[self._step], Y=[kl_min[variable_name]], update='append', name=variable_name, win=self.window_id[window_name])
        else:
            for variable_name in kl_min:
                if self.window_id[window_name] is not None:
                    self.vis.line(X=[self._step], Y=[kl_min[variable_name]], update='replace', name=variable_name, win=self.window_id[window_name])
                else:
                    opts = self._get_opts(window_name)
                    self.window_id[window_name] = self.vis.line(X=[self._step], Y=[kl_min[variable_name]], name=variable_name, opts=opts)

    def plot_image(self, img, title, size=(200,200)):
        # plots an image
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

    def _get_opts(self, win_name):
        # supplies various plotting configuration settings
        xlabel = 'Train Step'
        ylabel = ''
        width = 450
        height = 320
        title = ''
        xtype = 'log'
        showlegend = True
        if win_name == 'state':
            ylabel = 'State KL (nats)'
            title = 'State KL'
        elif win_name == 'action':
            ylabel = 'Action KL (nats)'
            title = 'Action KL'
        elif win_name == 'observation':
            ylabel = 'Obs. Info Gain (nats)'
            title = 'Obs. Info Gain'
        elif win_name == 'observation_cll':
            ylabel = 'Obs. Cond. Log Likelihood (nats)'
            title = 'Obs. Cond. Log Likelihood'
        elif win_name == 'observation_mll':
            ylabel = 'Obs. Marginal Log Likelihood (nats)'
            title = 'Obs. Marginal Log Likelihood'
        elif win_name == 'reward':
            ylabel = 'Reward Info Gain (nats)'
            title = 'Reward Info Gain'
        elif win_name == 'reward_cll':
            ylabel = 'Reward Cond. Log Likelihood (nats)'
            title = 'Reward Cond. Log Likelihood'
        elif win_name == 'reward_mll':
            ylabel = 'Reward Marginal Log Likelihood (nats)'
            title = 'Reward Marginal Log Likelihood'
        elif win_name == 'optimality':
            ylabel = 'Optimality Cond. Log Likelihood (nats)'
            title = 'Optimality Cond. Log Likelihood'
        elif win_name == 'done':
            ylabel = 'Done Info Gain (nats)'
            title = 'Done Info Gain'
        elif win_name == 'done_cll':
            ylabel = 'Done Cond. Log Likelihood (nats)'
            title = 'Done Cond. Log Likelihood'
        elif win_name == 'done_mll':
            ylabel = 'Done Marginal Log Likelihood (nats)'
            title = 'Done Marginal Log Likelihood'
        elif win_name == 'value':
            ylabel = 'Squared TD Error'
            title = 'Value Loss'
        elif win_name == 'importance_weights':
            ylabel = 'Ave. Importance Weight'
            title = 'Importance Weights'
            xtype = 'line'
        elif win_name == 'policy_gradients':
            ylabel = 'Ave. Policy Gradient Loss'
            title = 'Policy Gradient Loss'
        elif win_name == 'advantages':
            ylabel = 'Ave. Estimated Advantage'
            title = 'Estimated Advantages'
            xtype = 'line'
        elif win_name == 'state_improvement':
            ylabel = 'Improvement (nats)'
            title = 'State Inf. Improvement'
            xtype = 'line'
        elif win_name == 'rollout_length':
            ylabel = 'Max. Rollout Length'
            title = 'Planning Rollout Length'
            xtype = 'line'
        elif win_name == 'action_improvement':
            ylabel = 'Improvement (nats)'
            title = 'Action Inf. Improvement'
            xtype = 'line'
        elif win_name == 'length':
            ylabel = 'Number of Steps'
            title = 'Episode Length'
            xlabel = 'Episode'
            xtype = 'line'
        elif win_name == 'env_return':
            ylabel = 'Return'
            title = 'Environment Return'
            xlabel = 'Episode'
            xtype = 'line'
        elif win_name == 'total_steps':
            ylabel = 'Total Steps'
            title = 'Total Steps'
            xlabel = 'Episode'
            xtype = 'line'
        elif win_name == 'grads':
            ylabel = 'Ave. Gradient Abs. Value'
            title = 'Gradients'
            xtype = 'line'
        elif win_name == 'grad_norms':
            ylabel = 'Gradient Norm'
            title = 'Gradient Norms'
            xtype = 'line'
        elif win_name == 'kl_min':
            ylabel = 'KL Min. Clamp (nats)'
            title = 'KL Min. Clamp'
            xtype = 'line'
        elif win_name == 'actions':
            ylabel = 'Count'
            title = 'Actions'
            xlabel = 'Action'
            xtype = 'line'
            width = 300
            height = 320
            showlegend = False

        opts = dict(xlabel=xlabel, ylabel=ylabel, title=title, width=width,
                    height=height, xtype=xtype, showlegend=showlegend)
        return opts
