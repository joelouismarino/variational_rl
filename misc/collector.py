import torch
import torch.nn as nn
import copy
import numpy as np
from torch import optim
from misc.retrace import retrace
from modules.distributions import kl_divergence


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
        self.importance_weights = {'action': []}
        # store the values during training
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_action_log_probs = []
        self.new_q_values = []
        self.target_q_values = []
        self.sample_new_q_values = []

        self.valid = []
        self.dones = []

    def _collect_likelihood(self, name, obs, variable, valid, done=0.):
        cll = variable.cond_log_likelihood(obs).view(-1, 1)
        if self.agent._mode == 'train':
            self.objectives[name].append(-cll * (1 - done) * valid)
        self.metrics[name]['cll'].append((-cll * (1 - done) * valid).detach())
        if 'probs' in dir(variable.cond_likelihood.dist):
            self.distributions[name]['recon']['probs'].append(variable.cond_likelihood.dist.probs.detach())
        else:
            self.distributions[name]['recon']['loc'].append(variable.cond_likelihood.dist.loc.detach())
            self.distributions[name]['recon']['scale'].append(variable.cond_likelihood.dist.scale.detach())

    def _collect_kl(self, name, variable, valid, done):

        sample = None

        if name == 'action' and variable.approx_post.dist is None:
            # calculate the Boltzmann approximate posterior
            action_prior_samples = self.new_actions[-1]
            sample = action_prior_samples
            # if self.agent._mode == 'train':
            #     import ipdb; ipdb.set_trace()
            # prior_log_probs = variable.prior.log_prob(action_prior_samples).detach()
            prior_log_probs = self.agent.target_action_variable.prior.log_prob(action_prior_samples).detach()
            prior_log_probs = prior_log_probs.sum(dim=2, keepdim=True)
            q_values = self.sample_new_q_values[-1].detach().view(self.agent.n_action_samples, -1, 1)
            temperature = self.agent.alphas['pi'].detach()
            variable.approx_post.step(prior_log_probs=prior_log_probs,
                                      q_values=q_values,
                                      temperature=temperature)

        if name == 'action' and self.agent.target_action_prior_model is not None:
            batch_size = variable.prior._batch_size
            # get the distribution parameters
            target_variable = self.agent.target_action_variable
            target_prior_loc = self.agent.target_action_variable.prior.dist.loc
            target_prior_scale = self.agent.target_action_variable.prior.dist.scale
            current_prior_loc = self.agent.action_variable.prior.dist.loc
            current_prior_scale = self.agent.action_variable.prior.dist.scale
            if 'loc' in dir(variable.approx_post.dist):
                current_post_loc = self.agent.action_variable.approx_post.dist.loc
                current_post_scale = self.agent.action_variable.approx_post.dist.scale
                variable.approx_post.reset(batch_size, dist_params={'loc': current_post_loc.detach(), 'scale': current_post_scale.detach()})
            # target_post_loc = self.agent.target_action_variable.approx_post.dist.loc
            # target_post_scale = self.agent.target_action_variable.approx_post.dist.scale
            # decoupled updates on the prior from the previous prior distribution
            # and decoupled updates on the prior from the approx. posterior distribution
            # evaluate separate KLs for the loc and scale
            # loc KLs
            variable.prior.reset(batch_size, dist_params={'loc': current_prior_loc, 'scale': target_prior_scale.detach()})
            kl_prev_loc = kl_divergence(target_variable.prior, variable.prior, n_samples=self.agent.n_action_samples).sum(dim=1, keepdim=True)
            kl_curr_loc = kl_divergence(variable.approx_post, variable.prior, n_samples=self.agent.n_action_samples, sample=sample)
            kl_curr_loc_obj = torch.clamp(kl_curr_loc, min=self.agent.kl_min[name]).sum(dim=1, keepdim=True)
            kl_curr_loc = kl_curr_loc.sum(dim=1, keepdim=True)
            # scale KLs
            variable.prior.reset(batch_size, dist_params={'loc': target_prior_loc.detach(), 'scale': current_prior_scale})
            kl_prev_scale = kl_divergence(target_variable.prior, variable.prior, n_samples=self.agent.n_action_samples).sum(dim=1, keepdim=True)
            kl_curr_scale = kl_divergence(variable.approx_post, variable.prior, n_samples=self.agent.n_action_samples, sample=sample)
            kl_curr_scale_obj = torch.clamp(kl_curr_scale, min=self.agent.kl_min[name]).sum(dim=1, keepdim=True)
            kl_curr_scale = kl_curr_scale.sum(dim=1, keepdim=True)

            # append the objectives and metrics
            if self.agent._mode == 'train':
                self.objectives[name + '_prev_loc'].append(self.agent.alphas['loc'] * kl_prev_loc * (1 - done) * valid)
                self.objectives[name + '_curr_loc'].append(self.agent.alphas['pi'] * kl_curr_loc_obj * (1 - done) * valid)
                self.objectives[name + '_prev_scale'].append(self.agent.alphas['scale'] * kl_prev_scale * (1 - done) * valid)
                self.objectives[name + '_curr_scale'].append(self.agent.alphas['pi'] * kl_curr_scale_obj * (1 - done) * valid)
            self.metrics[name]['kl_prev_loc'].append((kl_prev_loc * (1 - done) * valid).detach())
            self.metrics[name]['kl_curr_loc'].append((kl_curr_loc * (1 - done) * valid).detach())
            self.metrics[name]['kl_prev_scale'].append((kl_prev_scale * (1 - done) * valid).detach())
            self.metrics[name]['kl_curr_scale'].append((kl_curr_scale * (1 - done) * valid).detach())
            self.metrics[name]['prior_prev_loc'].append(target_prior_loc.detach().mean(dim=1, keepdim=True))
            self.metrics[name]['prior_prev_scale'].append(target_prior_scale.detach().mean(dim=1, keepdim=True))
            self.metrics[name]['prior_curr_loc'].append(current_prior_loc.detach().mean(dim=1, keepdim=True))
            self.metrics[name]['prior_curr_scale'].append(current_prior_scale.detach().mean(dim=1, keepdim=True))
            if 'loc' in dir(variable.approx_post.dist):
                self.metrics[name]['approx_post_loc'].append(current_post_loc.detach().mean(dim=1, keepdim=True))
                self.metrics[name]['approx_post_scale'].append(current_post_scale.detach().mean(dim=1, keepdim=True))

            # reset the prior with detached parameters and approx. post. with
            # non-detached parameters to evaluate KL for approx. post.
            variable.prior.reset(batch_size, dist_params={'loc': current_prior_loc.detach(), 'scale': current_prior_scale.detach()})
            if 'loc' in dir(variable.approx_post.dist):
                variable.approx_post.reset(batch_size, dist_params={'loc': current_post_loc, 'scale': current_post_scale})
        else:
            if 'loc' in dir(variable.approx_post.dist):
                current_post_loc = self.agent.action_variable.approx_post.dist.loc
                current_post_scale = self.agent.action_variable.approx_post.dist.scale
                self.metrics[name]['approx_post_loc'].append(current_post_loc.detach().mean(dim=1, keepdim=True))
                self.metrics[name]['approx_post_scale'].append(current_post_scale.detach().mean(dim=1, keepdim=True))

        kl = variable.kl_divergence(n_samples=self.agent.n_action_samples, sample=sample)
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
                for param_name in dist.param_names:
                    param = getattr(dist.dist, param_name)
                    self.distributions[name][dist_name][param_name].append(param.detach())
        if self.agent._mode == 'train':
            self.objectives[name].append(self.agent.alphas['pi'] * obj_kl * (1 - done) * valid)
        self.metrics[name]['kl'].append((kl * (1 - done) * valid).detach())

    def _collect_log_probs(self, action, log_prob, valid):
        action = self.agent._convert_action(action)
        if 'loc' not in dir(self.agent.action_variable.approx_post.dist):
            self.log_probs['action'].append(valid)
            self.importance_weights['action'].append(valid)
        else:
            action_log_prob = self.agent.action_variable.approx_post.dist.log_prob(action)
            if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
                action_log_prob = action_log_prob.view(-1, 1)
            else:
                action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
            self.log_probs['action'].append(action_log_prob * valid)
            action_importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
            self.importance_weights['action'].append(action_importance_weight.detach())

    def _collect_episode(self, observation, reward, done, action):
        """
        Collect the variables for this step of the episode.
        """
        if not done:
            self.episode['observation'].append(observation)
            if action is None:
                action = self.agent.action_variable.sample(n_samples=1).detach()
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
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['log_prob'].append(log_prob.new(log_prob.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

        if done:
            self.agent.kl_min['action'] *= self.agent.kl_min_anneal_rate['action']
            self.agent.kl_factor['action'] *= self.agent.kl_factor_anneal_rate['action']
            self.agent.kl_factor['action'] = min(self.agent.kl_factor['action'], 1.)

    def collect(self, observation, reward, done, action, valid, log_prob):
        optimality_cll = reward
        if self.agent._mode == 'train':
            self.objectives['optimality'].append(-optimality_cll * valid)
        self.metrics['optimality']['cll'].append((-optimality_cll * valid).detach())

        if self.agent.reward_likelihood_model is not None and self.agent.train_model:
            self._collect_likelihood('reward', reward, self.agent.reward_variable, valid)

        if self.agent.obs_likelihood_model is not None and self.agent.train_model:
            self._collect_likelihood('observation', observation, self.agent.observation_variable, valid, done)

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
        # new_action_log_probs = torch.stack(self.new_action_log_probs)
        # target_entropy = -self.agent.action_variable.n_variables
        # alpha_loss = - (self.agent.log_alphas['pi'].exp() * (self.new_action_log_probs[-1] + target_entropy).detach()) * valid * (1 - done)
        if self.agent.epsilons['pi'] is not None:
            target_kl = self.agent.epsilons['pi']
        else:
            target_kl = self.agent.action_variable.n_variables * (1. + float(np.log(2)))
        alpha_loss = - (self.agent.log_alphas['pi'].exp() * (self.metrics['action']['kl'][-1] - target_kl).detach()) * valid * (1 - done)
        self.objectives['alpha_loss_pi'].append(alpha_loss)
        self.metrics['alpha_losses']['pi'].append(alpha_loss.detach())
        self.metrics['alphas']['pi'].append(self.agent.alphas['pi'])

        if self.agent.action_prior_model is not None:
            target_loc_kl = self.agent.epsilons['loc']
            alpha_loss_loc = - (self.agent.log_alphas['loc'].exp() * (self.metrics['action']['kl_prev_loc'][-1] - target_loc_kl).detach()) * valid * (1 - done)
            # alpha_loss_loc = - (self.agent.log_alphas['loc'] * (self.metrics['action']['kl_curr_loc'][-1] - target_loc_kl).detach()) * valid * (1 - done)
            self.objectives['alpha_loss_loc'].append(alpha_loss_loc)
            self.metrics['alpha_losses']['loc'].append(alpha_loss_loc.detach())
            self.metrics['alphas']['loc'].append(self.agent.alphas['loc'])

            target_scale_kl = self.agent.epsilons['scale']
            alpha_loss_scale = - (self.agent.log_alphas['scale'].exp() * (self.metrics['action']['kl_prev_scale'][-1] - target_scale_kl).detach()) * valid * (1 - done)
            # alpha_loss_scale = - (self.agent.log_alphas['scale'] * (self.metrics['action']['kl_curr_scale'][-1] - target_scale_kl).detach()) * valid * (1 - done)
            self.objectives['alpha_loss_scale'].append(alpha_loss_scale)
            self.metrics['alpha_losses']['scale'].append(alpha_loss_scale.detach())
            self.metrics['alphas']['scale'].append(self.agent.alphas['scale'])

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
        action_kl = torch.stack(self.objectives['action'])
        # action_kl = torch.stack(self.objectives['action_curr_loc'])
        # new_action_log_probs = torch.stack(self.new_action_log_probs)
        # alpha = self.agent.alpha['action']
        # target_values = torch.stack(self.target_q_values) - alpha * new_action_log_probs
        future_q = torch.stack(self.target_q_values) - action_kl
        # future_q = torch.stack(self.target_q_values)
        future_q = future_q * valid * (1. - dones)
        rewards *= self.agent.reward_scale
        importance_weights = torch.stack(self.importance_weights['action'])
        q_targets = retrace(future_q, rewards, importance_weights, discount=self.agent.reward_discount, l=self.agent.retrace_lambda)
        # target_values = torch.stack(self.target_q_values) - action_kl
        # q_targets = self.agent.reward_scale * rewards + self.agent.reward_discount * target_values[1:] * valid[1:] * (1. - dones[1:])
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
                           'alpha_loss_pi': [], 'q_loss': []}
        self.metrics = {'optimality': {'cll': []},
                        'action': {'kl': []},
                        'policy_loss': [],
                        'new_q_value': [],
                        'alpha_losses':{'pi': []},
                        'alphas': {'pi': []}}


        if self.agent.action_prior_model is not None:
            self.objectives['action_prev_loc'] = []
            self.objectives['action_prev_scale'] = []
            self.objectives['action_curr_loc'] = []
            self.objectives['action_curr_scale'] = []
            self.objectives['alpha_loss_loc'] = []
            self.objectives['alpha_loss_scale'] = []
            self.metrics['action']['kl_prev_loc'] = []
            self.metrics['action']['kl_prev_scale'] = []
            self.metrics['action']['kl_curr_loc'] = []
            self.metrics['action']['kl_curr_scale'] = []
            self.metrics['alpha_losses']['loc'] = []
            self.metrics['alpha_losses']['scale'] = []
            self.metrics['alphas']['loc'] = []
            self.metrics['alphas']['scale'] = []

            self.metrics['action']['prior_prev_loc'] = []
            self.metrics['action']['prior_prev_scale'] = []
            self.metrics['action']['prior_curr_loc'] = []
            self.metrics['action']['prior_curr_scale'] = []

        if self.agent.action_inference_model is not None:
            self.metrics['action']['approx_post_loc'] = []
            self.metrics['action']['approx_post_scale'] = []

        self.distributions = {}
        if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            self.distributions['action'] = {'prior': {'probs': []},
                                            'approx_post': {'probs': []}}
        else:
            self.distributions['action'] = {'prior': {param_name: [] for param_name in self.agent.action_variable.prior.param_names},
                                            'approx_post': {param_name: [] for param_name in self.agent.action_variable.approx_post.param_names}}

        self.inference_improvement = {'action': []}
        self.log_probs = {'action': []}

        if self.agent.observation_variable is not None and self.agent.train_model:
            self.objectives['observation'] = []
            self.metrics['observation'] = {'cll': []}
            self.distributions['observation'] = {'pred': {'loc': [], 'scale': []},
                                                 'recon': {'loc': [], 'scale': []}}
        if self.agent.reward_variable is not None and self.agent.train_model:
            self.objectives['reward'] = []
            self.metrics['reward'] = {'cll': []}
            self.distributions['reward'] = {'pred': {'loc': [], 'scale': []},
                                            'recon': {'loc': [], 'scale': []}}

        self.importance_weights = {'action': []}
        self.target_q_values = []
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_q_values = []
        self.sample_new_q_values = []
        self.new_action_log_probs = []
        self.valid = []
        self.dones = []
