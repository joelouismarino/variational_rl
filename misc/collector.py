import torch
import torch.nn as nn
import numpy as np
from torch import optim


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
        # self.distributions = {'action': {'prior': {'probs': []}, 'approx_post': {'probs': []}}}
        # stores inference improvement during training
        self.inference_improvement = {'action': []}
        # stores planning rollout lengths during training
        self.rollout_lengths = []
        # stores the log probabilities during training
        self.log_probs = {'action': []}
        # stores the importance weights during training
        # self.importance_weights = {'action': [], 'state': []}
        self.importance_weights = {'action': []}
        # store the values during training
        # self.values = []
        # self.target_values = []
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_action_log_probs = []
        self.new_q_values = []
        self.target_q_values = []

        self.valid = []
        self.dones = []

        # self.q_criterion = nn.MSELoss()

    # def create_optimizers(self):
    #     self.alpha_optimizer = optim.Adam([self.agent.log_alpha['action']], lr=3e-4)
    #     self.policy_optimizer = optim.Adam(self.agent.parameters()['action_inference_model'], lr=3e-4)
    #     self.q_value_model_optimizer = optim.Adam(self.agent.parameters()['q_value_models'], lr=3e-4)

    # def get_future_terms(self):
    #     # get the future terms in the objective
    #     valid = torch.stack(self.valid)
    #     optimality = -self.agent.reward_scale * torch.stack(self.objectives['optimality'])
    #     future_terms = optimality[1:]
    #     # TODO: should only include these if the action distribution is not reparameterizable
    #     # if self.agent.action_prior_model is not None:
    #     #     action_kl = torch.stack(self.objectives['action']) * valid
    #     #     future_terms = future_terms - self.agent.kl_scale['action'] * action_kl[1:]
    #     # if self.agent.state_prior_model is not None:
    #     #     state_kl = torch.stack(self.objectives['state']) * valid
    #     #     future_terms = future_terms - self.agent.kl_scale['state'] *state_kl[1:]
    #     # if self.agent.obs_likelihood_model is not None:
    #     #     obs_info_gain = torch.stack(self.metrics['observation']['info_gain']) * valid
    #     #     reward_info_gain = torch.stack(self.metrics['reward']['info_gain']) * valid
    #     #     done_info_gain = torch.stack(self.metrics['done']['info_gain']) * valid
    #     #     future_terms = future_terms + obs_info_gain[1:] + reward_info_gain[1:] + done_info_gain[1:]
    #     return future_terms
    #
    # def get_v_trace(self):
    #     valid = torch.stack(self.valid)
    #     # target_values = torch.stack(self.target_values)
    #     # import ipdb; ipdb.set_trace()
    #     new_action_log_probs = torch.stack(self.new_action_log_probs)
    #     # tqv = torch.stack(self.target_q_values)
    #     # alpha = self.agent.log_alpha.exp()
    #     # print('TQV', tqv.shape)
    #     # print('new_action_log_probs', new_action_log_probs.shape)
    #     # print('alpha', alpha.shape)
    #     target_values = torch.stack(self.target_q_values) - self.agent.log_alpha.exp() * new_action_log_probs
    #     # values = torch.stack(self.values)
    #     # qvalues = torch.stack(self.qvalues)
    #     #importance_weights = torch.stack(self.importance_weights['action'])
    #     #clip_importance_weights = self.agent.v_trace['lambda']*torch.clamp(importance_weights, 0, self.agent.v_trace['iw_clip'])
    #     future_terms = self.get_future_terms()
    #     #deltas = (future_terms + self.agent.reward_discount * values[1:] * valid[1:] - qvalues[:-1])
    #     #targets = []
    #     #sequence_len = len(future_terms)
    #     #for i in range(sequence_len):
    #     #    if i < sequence_len - 1:
    #     #        discount = self.agent.reward_discount ** torch.arange(0, sequence_len-i).view(-1, 1, 1).float().to(self.agent.device)
    #     #        cum_delta_i = torch.sum(discount * torch.cumprod(clip_importance_weights[i:-1], 0) * deltas[i:], 0)
    #     #        assert values[i].shape == cum_delta_i.shape
    #     #    target_i = (qvalues[i] + cum_delta_i) * valid[i]
    #     #    targets.append(target_i)
    #     #targets.append(values[-1])
    #     #targets = torch.cat(targets, 1).t().unsqueeze(2)
    #     #advantadges = future_terms + self.agent.reward_discount * targets[1:] * valid[1:] - values[:-1]
    #     #targets = future_terms + self.agent.reward_discount * values[1:] * valid[1:]
    #     q_targets = future_terms + self.agent.reward_discount * target_values[1:] * valid[1:]
    #     #q_targets = future_terms + self.agent.reward_discount * targets[1:] * valid[1:]
    #     #assert advantadges.shape == future_terms.shape
    #     # gradient should never pass here
    #     return None, q_targets.detach(), None
    #
    # def get_policy_loss(self):
    #     # E step
    #     # sample from current policy and reweight by the exponential of the Q function
    #
    #     # return weights and actions
    #     return 0

    def _collect_likelihood(self, name, obs, variable, valid, done=0.):
        log_importance_weights = self.agent.state_variable.log_importance_weights().detach()
        weighted_info_gain = variable.info_gain(obs, log_importance_weights, marg_factor=self.agent.marginal_factor)
        info_gain = variable.info_gain(obs, log_importance_weights, marg_factor=1.)
        cll = variable.cond_log_likelihood(obs).view(self.agent.n_state_samples, -1, 1).mean(dim=0)
        mll = variable.marg_log_likelihood(obs, log_importance_weights)
        if self.agent._mode == 'train':
            self.objectives[name].append(-weighted_info_gain * (1 - done) * valid)
        self.metrics[name]['cll'].append((-cll * (1 - done) * valid).detach())
        self.metrics[name]['mll'].append((-mll * (1 - done) * valid).detach())
        self.metrics[name]['info_gain'].append((-info_gain * (1 - done) * valid).detach())
        # if 'probs' in dir(variable.cond_likelihood.dist):
        #     # self.distributions[name]['pred']['probs'].append(variable.cond_likelihood.dist.probs.detach())
        #     self.distributions[name]['recon']['probs'].append(variable.cond_likelihood.dist.probs.detach())
        # else:
        #     # self.distributions[name]['pred']['loc'].append(variable.cond_likelihood.dist.loc.detach())
        #     # self.distributions[name]['pred']['scale'].append(variable.cond_likelihood.dist.scale.detach())
        #     self.distributions[name]['recon']['loc'].append(variable.cond_likelihood.dist.loc.detach())
        #     self.distributions[name]['recon']['scale'].append(variable.cond_likelihood.dist.scale.detach())

    def _collect_kl(self, name, variable, valid, done):
        kl = variable.kl_divergence()
        obj_kl = self.agent.kl_factor[name] * torch.clamp(kl, min=self.agent.kl_min[name]).sum(dim=1, keepdim=True)
        if variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            # discrete
            kl = kl.view(-1, 1)
            obj_kl = obj_kl.view(-1, 1)
            # self.distributions[name]['prior']['probs'].append(variable.prior.dist.probs.detach())
            # self.distributions[name]['approx_post']['probs'].append(variable.approx_post.dist.probs.detach())
        else:
            # continuous
            kl = kl.sum(dim=1, keepdim=True)
            obj_kl = obj_kl.sum(dim=1, keepdim=True)
            # for dist_name in ['prior', 'approx_post']:
            #     dist = getattr(variable, dist_name)
            #     for param_name in dist.initial_params:
            #         param = getattr(dist.dist, param_name)
            #         self.distributions[name][dist_name][param_name].append(param.detach())
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
            # self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['observation'].append(observation)
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
        self.objectives['policy_loss'].append(policy_loss)
        self.metrics['policy_loss'].append(policy_loss.detach())
        self.metrics['new_q_value'].append(self.new_q_values[-1].detach())
    #
    def _get_alpha_losses(self, valid, done):
        new_action_log_probs = torch.stack(self.new_action_log_probs)
        target_entropy = -self.agent.action_variable.n_variables
        alpha_loss = - (self.agent.log_alpha['action'] * (self.new_action_log_probs[-1] + target_entropy).detach()) * valid * (1 - done)
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
        # results['distributions'] = {}
        # for k, v in self.distributions.items():
        #     # variable
        #     results['distributions'][k] = {}
        #     for kk, vv in v.items():
        #         # distribution
        #         results['distributions'][k][kk] = {}
        #         for kkk, vvv in vv.items():
        #             # parameters
        #             if len(vvv) > 0:
        #                 results['distributions'][k][kk][kkk] = torch.cat(vvv, dim=0).detach().cpu()
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
        # TODO: incorporate V-trace here
        # target_values = torch.stack(self.target_q_values) - alpha * new_action_log_probs
        target_values = torch.stack(self.target_q_values) - action_kl
        q_targets = self.agent.reward_scale * rewards + self.agent.reward_discount * target_values[1:] * valid[1:] * (1. - dones[1:])
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
        total_objective = torch.zeros(n_steps, self.agent.batch_size, 1).to(self.agent.device)
        for objective_name, objective in self.objectives.items():
            obj = torch.stack(objective[:-1]) if type(objective) == list else objective
            total_objective = total_objective + obj
        total_objective = total_objective.sum(dim=0).div(n_valid_steps).mean(dim=0).sum()
        return total_objective

        # valid = torch.stack(self.valid)
        # dones = torch.stack(self.dones)
        # rewards = -torch.stack(self.objectives['optimality'])[1:]
        # new_action_log_probs = torch.stack(self.new_action_log_probs)
        #
        # # ALPHA LOSS
        # target_entropy = -self.agent.action_variable.n_variables
        # alpha_loss = -self.agent.log_alpha['action'] * (new_action_log_probs + target_entropy).detach()
        # self.alpha_optimizer.zero_grad()
        # alpha_loss[:-1].mean().backward()
        # self.alpha_optimizer.step()
        # alpha = self.agent.log_alpha['action'].exp().detach()
        #
        # # total_objective = alpha_loss[:-1].mean()
        # total_objective = 0
        #
        # # POLICY LOSS
        # new_q_values = torch.stack(self.new_q_values)
        # policy_loss = (alpha * new_action_log_probs - new_q_values) * valid
        # # total_objective = total_objective + policy_loss[:-1].mean()
        # self.policy_optimizer.zero_grad()
        # policy_loss[:-1].mean().backward()
        # self.policy_optimizer.step()
        #
        # # VALUE LOSS
        # q_values1 = torch.stack(self.qvalues1)
        # q_values2 = torch.stack(self.qvalues2)
        # # q_targets = self.get_q_targets()
        # target_values = torch.stack(self.target_q_values) - alpha * new_action_log_probs
        # q_targets = self.agent.reward_scale * rewards + self.agent.reward_discount * target_values[1:] * valid[1:] * (1. - dones[1:])
        # q_loss1 = self.q_criterion(q_values1[:-1], q_targets.detach())
        # q_loss2 = self.q_criterion(q_values2[:-1], q_targets.detach())
        # # total_objective = total_objective + q_loss1 + q_loss2
        # self.q_value_model_optimizer.zero_grad()
        # (q_loss1 + q_loss2).backward()
        # self.q_value_model_optimizer.step()

        # calculate policy gradient terms and add them to the total objective
        #action_log_probs = torch.stack(self.log_probs['action'])
        #action_importance_weights = torch.stack(self.importance_weights['action']).detach()
        #action_reinforce_terms = - action_importance_weights[:-1] * action_log_probs[:-1] * advantages
        #clip_importance_weight = torch.clamp(action_importance_weights[:-1], 0, 1)
        #assert action_log_probs[:-1].shape == advantages.shape
        #action_reinforce_terms = - action_importance_weights[:-1] * action_log_probs[:-1] * advantages.detach()
        #entropy_term = - action_log_probs[:-1]
        #if not self.agent.action_variable.approx_post.update == 'iterative':
        #    # include the policy gradients in the total objective
        #    assert action_reinforce_terms.shape == entropy_term.shape
        #    total_objective[:-1] = total_objective[:-1] + action_reinforce_terms

        # q_loss = 0.5 * (q_values1[:-1] - q_targets).pow(2) + 0.5 * (q_values2[:-1] - q_targets).pow(2)
        # v_targets = (new_q_values - self.agent.kl_scale['action'] * new_action_log_probs).detach() * valid
        # v_targets = ((new_q_values - self.agent.kl_scale['action'] * action_kl) * valid).detach()
        # value_loss = 0.5 * (v_targets[:-1] - values[:-1]).pow(2)
        # total_objective[:-1] = total_objective[:-1] + value_loss + q_loss
        # total_objective = total_objective + value_loss + q_loss

        # total_objective = total_objective.sum(dim=0).div(n_valid_steps).mean(dim=0).sum()
        # total_objective = total_objective.sum(dim=0).mean(dim=0).sum()

        # if q_loss1 > 200 or q_loss2 > 200:
        #     import ipdb; ipdb.set_trace()

        # save value / policy gradient metrics
        # self.metrics['q_loss1'] = q_loss1
        # self.metrics['q_loss2'] = q_loss2
        # self.metrics['value_loss'] = value_loss
        # self.metrics['value'] = values.mean()
        # self.metrics['value_target'] = v_targets.mean()
        # self.metrics['target_q_parameter'] = self.agent.parameters()['target_q_value_models'][0][0, 0]
        # print(self.agent.parameters()['target_q_value_models'][0][0, 0])
        # self.metrics['q_values1'] = q_values1[:-1].mean()
        # self.metrics['q_values2'] = q_values2[:-1].mean()
        # self.metrics['value_targets'] = target_values[1:].mean()
        # self.metrics['q_value_targets'] = q_targets.mean()
        # self.metrics['new_q_value'] = new_q_values.mean()
        # self.metrics['policy_loss'] = policy_loss[:-1].mean()
        # self.metrics['entropy_policy'] = -new_action_log_probs[:-1].mean()
        # self.metrics['next_action_log_prob'] = new_action_log_probs[1:].mean()
        # self.metrics['alpha_loss'] = alpha_loss[:-1].mean()
        # self.metrics['alpha'] = alpha.mean()
        # self.metrics['rewards'] = rewards.mean()
        # state_importance_weights = torch.stack(self.importance_weights['state'])
        # self.metrics['state_importance_weights'] = state_importance_weights

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
        # self.distributions = {}
        # if self.agent.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
        #     self.distributions['action'] = {'prior': {'probs': []},
        #                                     'approx_post': {'probs': []}}
        # else:
        #     self.distributions['action'] = {'prior': {'loc': [], 'scale': []},
        #                                     'approx_post': {'loc': [], 'scale': []}}

        self.inference_improvement = {'action': []}
        self.log_probs = {'action': []}

        if self.agent.state_variable is not None:
            self.episode['state'] = []
            self.objectives['state'] = []
            self.metrics['state'] = {'kl': []}
            # self.distributions['state'] = {'prior': {'loc': [], 'scale': []},
            #                                'approx_post': {'loc': [], 'scale': []}}
            self.log_probs['state'] = []
            self.inference_improvement['state'] = []
        if self.agent.observation_variable is not None:
            self.objectives['observation'] = []
            self.metrics['observation'] = {'cll': [], 'info_gain': [], 'mll': []}
            # self.distributions['observation'] = {'pred': {'loc': [], 'scale': []},
            #                                      'recon': {'loc': [], 'scale': []}}
        if self.agent.reward_variable is not None:
            self.objectives['reward'] = []
            self.metrics['reward'] = {'cll': [], 'info_gain': [], 'mll': []}
            # self.distributions['reward'] = {'pred': {'loc': [], 'scale': []},
            #                                 'recon': {'loc': [], 'scale': []}}
        if self.agent.done_variable is not None:
            self.objectives['done'] = []
            self.metrics['done'] = {'cll': [], 'info_gain': [], 'mll': []}
            # self.distributions['done'] = {'pred': {'probs': []}, 'recon': {'probs': []}}

        self.rollout_lengths = []
        # self.importance_weights = {'action': [], 'state': []}
        self.importance_weights = {'action': []}
        # self.values = []
        # self.target_values = []
        self.target_q_values = []
        self.qvalues = []
        self.qvalues1 = []
        self.qvalues2 = []
        self.new_actions = []
        self.new_q_values = []
        self.new_action_log_probs = []
        self.valid = []
        self.dones = []
