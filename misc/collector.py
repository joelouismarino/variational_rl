import torch
import torch.nn as nn
import numpy as np
from torch import optim
from misc.retrace import retrace


class Collector:
    """
    Collects objectives and episodes for an agent.
    """
    def __init__(self, agent):
        # link to the corresponding agent
        self.agent = agent

        # stores the variables
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'action': [], 'log_prob': []}

        # stores the objectives during training
        self.objectives = {'optimality': [], 'action': [], 'policy_loss': [],
                           'alpha_loss': [], 'q_loss': []}
        # stores the metrics
        self.metrics = {'optimality': {'cll': []},
                        'action': {'kl': []},
                        'policy_loss': [],
                        'new_q_value': [],
                        'alpha_loss':[],
                        'alpha': []}
        # stores the distributions
        self.distributions = {'action': {'prior': {'probs': []}, 'approx_post': {'probs': []}}}
        # stores inference improvement during training
        self.inference_improvement = {'action': []}
        # stores the log probabilities during training
        self.log_probs = {'action': []}
        # stores the importance weights during training
        # self.importance_weights = {'action': [], 'state': []}
        self.importance_weights = {'action': []}
        # store the values during training
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_action_log_probs = []
        self.new_q_values = []
        self.target_q_values = []

        self.valid = []
        self.dones = []

    def _collect_likelihood(self, name, obs, variable, valid, done=0.):
        # log_importance_weights = self.agent.state_variable.log_importance_weights().detach()
        # weighted_info_gain = variable.info_gain(obs, log_importance_weights, marg_factor=self.agent.marginal_factor)
        # info_gain = variable.info_gain(obs, log_importance_weights, marg_factor=1.)
        # cll = variable.cond_log_likelihood(obs).view(self.agent.n_state_samples, -1, 1).mean(dim=0)
        cll = variable.cond_log_likelihood(obs).view(-1, 1)
        # mll = variable.marg_log_likelihood(obs, log_importance_weights)
        if self.agent._mode == 'train':
            # self.objectives[name].append(-weighted_info_gain * (1 - done) * valid)
            self.objectives[name].append(-cll * (1 - done) * valid)
        self.metrics[name]['cll'].append((-cll * (1 - done) * valid).detach())
        # self.metrics[name]['mll'].append((-mll * (1 - done) * valid).detach())
        # self.metrics[name]['info_gain'].append((-info_gain * (1 - done) * valid).detach())
        if 'probs' in dir(variable.cond_likelihood.dist):
            # self.distributions[name]['pred']['probs'].append(variable.cond_likelihood.dist.probs.detach())
            self.distributions[name]['recon']['probs'].append(variable.cond_likelihood.dist.probs.detach())
        else:
            # self.distributions[name]['pred']['loc'].append(variable.cond_likelihood.dist.loc.detach())
            # self.distributions[name]['pred']['scale'].append(variable.cond_likelihood.dist.scale.detach())
            self.distributions[name]['recon']['loc'].append(variable.cond_likelihood.dist.loc.detach())
            self.distributions[name]['recon']['scale'].append(variable.cond_likelihood.dist.scale.detach())

    def _collect_kl(self, name, variable, valid, done):
        kl = variable.kl_divergence()
        obj_kl = self.agent.kl_factor[name] * torch.clamp(kl, min=self.agent.kl_min[name]).sum(dim=1, keepdim=True)
        if variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            # discrete
            kl = kl.view(-1, 1)
            obj_kl = obj_kl.view(-1, 1)
            self.distributions[name]['prior']['probs'].append(variable.prior.dist.probs.detach())
            self.distributions[name]['approx_post']['probs'].append(variable.approx_post.dist.probs.detach())
        else:
            # continuous
            kl = kl.sum(dim=1, keepdim=True)
            obj_kl = obj_kl.sum(dim=1, keepdim=True)
            for dist_name in ['prior', 'approx_post']:
                dist = getattr(variable, dist_name)
                for param_name in dist.initial_params:
                    param = getattr(dist.dist, param_name)
                    self.distributions[name][dist_name][param_name].append(param.detach())
            # self.distributions[name]['prior']['loc'].append(variable.prior.dist.loc.detach())
            # if hasattr(variable.prior.dist, 'scale'):
            #     self.distributions[name]['prior']['scale'].append(variable.prior.dist.scale.detach())
            # self.distributions[name]['approx_post']['loc'].append(variable.approx_post.dist.loc.detach())
            # self.distributions[name]['approx_post']['scale'].append(variable.approx_post.dist.scale.detach())
        if self.agent._mode == 'train':
            self.objectives[name].append(self.agent.alpha[name] * obj_kl * (1 - done) * valid)
        self.metrics[name]['kl'].append((kl * (1 - done) * valid).detach())

    def _collect_log_probs(self, action, log_prob, valid):
        action = self.agent._convert_action(action)
        action_log_prob = self.agent.action_variable.approx_post.dist.log_prob(action)
        if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            action_log_prob = action_log_prob.view(-1, 1)
        else:
            action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
        self.log_probs['action'].append(action_log_prob * valid)
        if self.agent.state_variable is not None:
            state = self.agent.state_variable.sample()
            state_log_prob = self.agent.state_variable.approx_post.dist.log_prob(state)
            state_log_prob = state_log_prob.sum(dim=1, keepdim=True)
            self.log_probs['state'].append(state_log_prob * valid)
        action_importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
        self.importance_weights['action'].append(action_importance_weight.detach())
        # state_importance_weight = self.agent.state_variable.log_importance_weights().exp().mean(dim=0)
        # self.importance_weights['state'].append(state_importance_weight.detach())

    def _collect_episode(self, observation, reward, done, action):
        """
        Collect the variables for this step of the episode.
        """
        if not done:
            self.episode['observation'].append(observation)
            if self.agent.state_variable is not None:
                self.episode['state'].append(self.agent.state_variable.sample().detach())
            if action is None:
                action = self.agent.action_variable.sample().detach()
                action = self.agent._convert_action(self.agent.action_variable.sample().detach())
            self.episode['action'].append(action)
            action_log_prob = self.agent.action_variable.approx_post.log_prob(action)
            if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
                action_log_prob = action_log_prob.view(-1, 1)
            else:
                action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
            self.episode['log_prob'].append(action_log_prob.detach())
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            log_prob = self.episode['log_prob'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            # self.episode['observation'].append(observation)
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['log_prob'].append(log_prob.new(log_prob.shape).zero_())
            if self.agent.state_variable is not None:
                state = self.episode['state'][0]
                self.episode['state'].append(state.new(state.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

        if done:
            if self.agent.observation_variable is not None:
                self.agent.marginal_factor *= self.agent.marginal_factor_anneal_rate
                self.agent.marginal_factor = min(self.agent.marginal_factor, 1.)
            self.agent.kl_min['state'] *= self.agent.kl_min_anneal_rate['state']
            self.agent.kl_min['action'] *= self.agent.kl_min_anneal_rate['action']
            self.agent.kl_factor['state'] *= self.agent.kl_factor_anneal_rate['state']
            self.agent.kl_factor['action'] *= self.agent.kl_factor_anneal_rate['action']
            self.agent.kl_factor['state'] = min(self.agent.kl_factor['state'], 1.)
            self.agent.kl_factor['action'] = min(self.agent.kl_factor['action'], 1.)

    def collect(self, observation, reward, done, action, valid, log_prob):
        optimality_cll = reward
        if self.agent._mode == 'train':
            self.objectives['optimality'].append(-optimality_cll * valid)
        self.metrics['optimality']['cll'].append((-optimality_cll * valid).detach())

        if self.agent.done_likelihood_model is not None:
            self._collect_likelihood('done', done, self.agent.done_variable, valid)

        if self.agent.reward_likelihood_model is not None:
            self._collect_likelihood('reward', reward, self.agent.reward_variable, valid)

        if self.agent.obs_likelihood_model is not None:
            self._collect_likelihood('observation', observation, self.agent.observation_variable, valid, done)

        if self.agent.state_variable is not None:
            self._collect_kl('state', self.agent.state_variable, valid, done)

        self._collect_kl('action', self.agent.action_variable, valid, done)

        if self.agent._mode == 'train':
            self._collect_log_probs(action, log_prob, valid)
            self._get_policy_loss(valid, done)
            self._get_alpha_losses(valid, done)
        else:
            self._collect_episode(observation, reward, done, action)

        self.valid.append(valid)
        self.dones.append(done)

    def _get_policy_loss(self, valid, done):
        policy_loss = -self.new_q_values[-1] * valid * (1 - done)
        if self.agent.action_variable.approx_post.update == 'direct':
            self.objectives['policy_loss'].append(policy_loss)
        else:
            self.objectives['policy_loss'].append(policy_loss * 0.)
        self.metrics['policy_loss'].append(policy_loss.detach())
        self.metrics['new_q_value'].append(self.new_q_values[-1].detach())

    def _get_alpha_losses(self, valid, done):
        new_action_log_probs = torch.stack(self.new_action_log_probs)
        target_entropy = -self.agent.action_variable.n_variables
        alpha_loss = - (self.agent.log_alpha['action'] * (self.new_action_log_probs[-1] + target_entropy).detach()) * valid * (1 - done)
        # target_kl = 0.1
        # alpha_loss = - (self.agent.log_alpha['action'] * (self.metrics['action']['kl'][-1] - target_kl).detach()) * valid * (1 - done)
        self.objectives['alpha_loss'].append(alpha_loss)
        self.metrics['alpha_loss'].append(alpha_loss.detach())
        self.metrics['alpha'].append(self.agent.alpha['action'])

    def get_episode(self):
        """
        Concatenate each variable, objective in the episode. Put on the CPU.
        """
        results = {}
        # copy over the episode itself
        for k, v in self.episode.items():
            if len(v) > 0:
                results[k] = torch.cat(v, dim=0).detach().cpu()
            else:
                results[k] = []
        # get the metrics
        results['metrics'] = {}
        for k, v in self.metrics.items():
            if type(v) == dict:
                results['metrics'][k] = {}
                for kk, vv in v.items():
                    if len(vv) > 0:
                        results['metrics'][k][kk] = torch.cat(vv, dim=0).detach().cpu()
                    else:
                        results['metrics'][k][kk] = []
        # get the inference improvements
        results['inf_imp'] = {}
        for k, v in self.inference_improvement.items():
            if len(v) > 0:
                results['inf_imp'][k] = torch.cat(v, dim=0).detach().cpu()
            else:
                results['inf_imp'][k] = []
        # get the distribution parameters
        results['distributions'] = {}
        for k, v in self.distributions.items():
            # variable
            results['distributions'][k] = {}
            for kk, vv in v.items():
                # distribution
                results['distributions'][k][kk] = {}
                for kkk, vvv in vv.items():
                    # parameters
                    if len(vvv) > 0:
                        results['distributions'][k][kk][kkk] = torch.cat(vvv, dim=0).detach().cpu()
        # get the returns, values, advantages
        # results['value'] = torch.cat(self.values, dim=0).detach().cpu()
        return results

    def get_metrics(self):
        """
        Collect the metrics into a dictionary.
        """
        metrics = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0) - 1

        # average metrics over time and batch (for reporting)
        for variable_name, metric in self.metrics.items():
            if type(metric) == dict:
                for metric_name, met in metric.items():
                    m = torch.stack(met[:-1]).sum(dim=0).div(n_valid_steps).mean(dim=0)
                    if metric_name in ['cll', 'mll', 'info_gain']:
                        # negate for plotting
                        m = m * -1
                    metrics[variable_name + '_' + metric_name] = m.detach().cpu().item()
            else:
                if type(metric) == list:
                    metric = torch.stack(metric[:-1])
                m = metric.sum(dim=0).div(n_valid_steps).mean(dim=0)
                metrics[variable_name] = m.detach().cpu().item()

        return metrics

    def get_inf_imp(self):
        """
        Collect the inference improvement into a dictionary.
        """
        inf_imp = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0)

        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                inf_imp[name + '_improvement'] = imp.detach().cpu().item()

        return inf_imp

    def _get_q_targets(self):
        """
        Get the targets for the Q-value estimator.
        """
        dones = torch.stack(self.dones)
        valid = torch.stack(self.valid)
        rewards = -torch.stack(self.objectives['optimality'])[1:]
        # new_action_log_probs = torch.stack(self.new_action_log_probs)
        action_kl = torch.stack(self.objectives['action'])
        # alpha = self.agent.alpha['action']
        # target_values = torch.stack(self.target_q_values) - alpha * new_action_log_probs
        future_q = torch.stack(self.target_q_values) - action_kl
        future_q = future_q * valid
        # TODO: should be an hyper-parameter
        LAMBDA = 0.
        #target_values = torch.stack(self.target_q_values) - action_kl
        #old_q_targets = self.agent.reward_scale * rewards + self.agent.reward_discount * target_values[1:] * valid[1:] * (1. - dones[1:])
        #print(old_q_targets)
        rewards *= self.agent.reward_scale
        importance_weights = torch.stack(self.importance_weights['action'])
        q_targets = retrace(future_q, rewards, importance_weights, discount=self.agent.reward_discount, l=LAMBDA)
        #q_targets = self.agent.reward_scale * rewards + self.agent.reward_discount * target_values[1:] * valid[1:] * (1. - dones[1:])
        return q_targets.detach()

    def evaluate_q_loss(self):
        """
        Get the loss for the Q networks.
        """
        valid = torch.stack(self.valid)
        q_values1 = torch.stack(self.qvalues1)
        q_values2 = torch.stack(self.qvalues2)
        q_targets = self._get_q_targets()
        q_loss1 = 0.5 * (q_values1[:-1] - q_targets).pow(2) * valid[:-1]
        q_loss2 = 0.5 * (q_values2[:-1] - q_targets).pow(2) * valid[:-1]
        self.objectives['q_loss'] = q_loss1 + q_loss2
        self.metrics['q_loss1'] = q_loss1.mean()
        self.metrics['q_loss2'] = q_loss2.mean()
        self.metrics['q_values1'] = q_values1[:-1].mean()
        self.metrics['q_values2'] = q_values2[:-1].mean()
        self.metrics['q_value_targets'] = q_targets.mean()

    def evaluate(self):
        """
        Combines the objectives for training.
        """
        self.evaluate_q_loss()
        n_steps = len(self.objectives['optimality'])
        n_valid_steps = torch.stack(self.valid).sum(dim=0) - 1
        total_objective = torch.zeros(n_steps-1, self.agent.batch_size, 1).to(self.agent.device)
        for objective_name, objective in self.objectives.items():
            obj = torch.stack(objective[:-1]) if type(objective) == list else objective
            total_objective = total_objective + obj
        total_objective = total_objective.sum(dim=0).div(n_valid_steps).mean(dim=0).sum()
        return total_objective

    def get_grads(self):
        """
        Calculate the average gradient for each model.
        """
        grads_dict = {}
        grad_norm_dict = {}
        for model_name, params in self.agent.parameters().items():
            grads = [param.grad.view(-1) for param in params if param.grad is not None]
            if len(grads) > 0:
                grads = torch.cat(grads, dim=0)
                grads_dict[model_name] = grads.abs().mean().cpu().numpy().item()
                grad_norm_dict[model_name] = grads.norm().cpu().numpy().item()
        return {'grads': grads_dict, 'grad_norms': grad_norm_dict}

    def reset(self):
        """
        Reset the episode, objectives, and log probs.
        """
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'action': [], 'log_prob': []}

        self.objectives = {'optimality': [], 'action': [], 'policy_loss': [],
                           'alpha_loss': [], 'q_loss': []}
        self.metrics = {'optimality': {'cll': []},
                        'action': {'kl': []},
                        'policy_loss': [],
                        'new_q_value': [],
                        'alpha_loss':[],
                        'alpha': []}
        self.distributions = {}
        if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            self.distributions['action'] = {'prior': {'probs': []},
                                            'approx_post': {'probs': []}}
        else:
            self.distributions['action'] = {'prior': {'loc': [], 'scale': []},
                                            'approx_post': {'loc': [], 'scale': []}}

        self.inference_improvement = {'action': []}
        self.log_probs = {'action': []}

        if self.agent.state_variable is not None:
            self.episode['state'] = []
            self.objectives['state'] = []
            self.metrics['state'] = {'kl': []}
            self.distributions['state'] = {'prior': {'loc': [], 'scale': []},
                                           'approx_post': {'loc': [], 'scale': []}}
            self.log_probs['state'] = []
            self.inference_improvement['state'] = []
        if self.agent.observation_variable is not None:
            self.objectives['observation'] = []
            self.metrics['observation'] = {'cll': []}
            # self.metrics['observation'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['observation'] = {'pred': {'loc': [], 'scale': []},
                                                 'recon': {'loc': [], 'scale': []}}
        if self.agent.reward_variable is not None:
            self.objectives['reward'] = []
            self.metrics['reward'] = {'cll': []}
            # self.metrics['reward'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['reward'] = {'pred': {'loc': [], 'scale': []},
                                            'recon': {'loc': [], 'scale': []}}
        if self.agent.done_variable is not None:
            self.objectives['done'] = []
            self.metrics['done'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['done'] = {'pred': {'probs': []}, 'recon': {'probs': []}}

        # self.importance_weights = {'action': [], 'state': []}
        self.importance_weights = {'action': []}
        self.target_q_values = []
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_q_values = []
        self.new_action_log_probs = []
        self.valid = []
        self.dones = []
