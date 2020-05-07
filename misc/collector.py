import torch
import torch.nn as nn
import copy
import numpy as np
from torch import optim
from misc.estimators import retrace
from lib.distributions import kl_divergence


class Collector:
    """
    Collects objectives and episodes for an agent.
    """
    def __init__(self, agent):
        # link to the corresponding agent
        self.agent = agent

        # stores the variables
        self.episode = {'state': [], 'reward': [], 'done': [],
                        'action': [], 'log_prob': []}

        # stores the objectives during training
        self.objectives = {'optimality': [], 'alpha_loss': [], 'q_loss': [],
                           'log_scale': []}
        # stores the metrics
        self.metrics = {'optimality': {'cll': []},
                        'action': {'kl': []},
                        'alpha_losses': {},
                        'alphas': {},
                        'log_scale_loss': {}}
        # stores the distributions
        self.distributions = {'action': {'prior': {}, 'approx_post': {}}}
        # stores inference improvement during training
        self.inference_improvement = []
        # stores the log probabilities during training
        self.log_probs = {'action': []}
        # stores the importance weights during training
        self.importance_weights = {'action': []}
        # store the values during training
        self.q_values = []
        self.q_values1 = []
        self.q_values2 = []
        self.target_q_values = []

        self.valid = []
        self.dones = []

    def collect(self, state, action, reward, done, valid, log_prob):
        """
        Collect objectives and variables for a single time step.
        Note: behavior depends on whether in train or eval mode.
        """
        if self.agent.mode == 'train':
            # collect the objectives for training
            self._collect_train(state, action, reward, done, valid, log_prob)
        elif self.agent.mode == 'eval':
            # collect terms
            self._collect_eval(state, action, reward, done)
        else:
            raise NotImplementedError
        self.valid.append(valid)
        self.dones.append(done)

    def _collect_train(self, state, off_policy_action, reward, done, valid, log_prob):
        """
        Collects objectives and other quantities for training.
        """
        on_policy_action = self.agent.approx_post.sample(self.agent.n_action_samples)
        if self.agent.direct_approx_post is not None:
            target_on_policy_action = self.agent.direct_approx_post.sample(self.agent.n_action_samples)
        elif self.agent.target_approx_post is not None:
            target_on_policy_action = self.agent.target_approx_post.sample(self.agent.n_action_samples)
        else:
            target_on_policy_action = on_policy_action
        # collect inference optimizer objective
        self._collect_inf_opt_objective(state, on_policy_action, valid, done)
        # collect the optimality (reward)
        self._collect_optimality_objective(reward, valid, done)
        # collect KL divergence objectives
        self._collect_kl_objectives(on_policy_action.detach(), valid, done)
        # collect alpha objectives
        self._collect_alpha_objectives(valid, done)
        # collect value estimator objectives
        self._get_value_est_objectives(state, off_policy_action, target_on_policy_action.detach(), reward, valid, done)
        # collect log probabilities
        self._collect_log_probs(off_policy_action, log_prob, valid)
        # collect log-scale limit objectives
        self._collect_log_scale_lim_objectives()

    def _collect_eval(self, state, action, reward, done):
        """
        Collects terms and distributions during evaluation (episode collection).
        """
        # collect terms
        self._collect_terms(state, action, reward, done)
        # collect distributions
        self._collect_distributions()

    def _collect_terms(self, state, action, reward, done):
        """
        Collect the variables for this step of the episode.
        """
        if not done.prod():
            # get relevant variables
            self.episode['state'].append(state)
            self.episode['action'].append(action)
            # TODO: this isn't correct for non-parametric
            log_prob = self.agent.approx_post.log_prob(action).sum(dim=1, keepdim=True).detach()
            self.episode['log_prob'].append(log_prob)
        else:
            # fill in with zeros when done
            state = self.episode['state'][0]
            action = self.episode['action'][0]
            log_prob = self.episode['log_prob'][0]
            self.episode['state'].append(state.new(state.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['log_prob'].append(log_prob.new(log_prob.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

    def _collect_distributions(self):
        """
        Collect the distribution parameters.
        """
        # action prior and approximate posterior
        dist_name_list = ['prior', 'approx_post']
        if self.agent.direct_approx_post is not None:
            dist_name_list.append('direct_approx_post')
        for dist_name in dist_name_list:
            dist = getattr(self.agent, dist_name)
            for param_name in dist.param_names:
                param = getattr(dist.dist, param_name)
                self.distributions['action'][dist_name][param_name].append(param.detach())

        #  state and reward conditional likelihoods (if applicable)
        if 'state_likelihood_model' in dir(self.agent.q_value_estimator):
            if self.agent._prev_action is not None:
                # predict the current state and reward
                self.agent.q_value_estimator.generate(self.agent)
            variable = self.agent.q_value_estimator.state_variable
            self.distributions['state']['cond_like']['loc'].append(variable.cond_likelihood.dist.loc.detach())
            self.distributions['state']['cond_like']['scale'].append(variable.cond_likelihood.dist.scale.detach())
            if self.agent.q_value_estimator.reward_likelihood_model is not None:
                variable = self.agent.q_value_estimator.reward_variable
                self.distributions['reward']['cond_like']['loc'].append(variable.cond_likelihood.dist.loc.detach())
                self.distributions['reward']['cond_like']['scale'].append(variable.cond_likelihood.dist.scale.detach())

    def _collect_optimality_objective(self, reward, valid, done):
        """
        Collects the negative log optimality (reward).
        """
        self.objectives['optimality'].append(-reward * valid)
        self.metrics['optimality']['cll'].append((-reward * valid).detach())

    def _collect_inf_opt_objective(self, state, on_policy_action, valid, done):
        """
        Evaluates the inference optimizer if there are amortized parameters.
        """
        if 'parameters' in dir(self.agent.inference_optimizer):
            # detach the prior
            if self.agent.prior_model is not None:
                batch_size = self.agent.prior._batch_size
                loc = self.agent.prior.dist.loc
                scale = self.agent.prior.dist.scale
                self.agent.prior.reset(batch_size, dist_params={'loc': loc.detach(), 'scale': scale.detach()})
            # evaluate the objective
            obj = self.agent.estimate_objective(state, on_policy_action)
            obj = obj.view(self.agent.n_action_samples, -1, 1).mean(dim=0)
            if self.agent.inference_optimizer.n_inf_iters > 1:
                # append final objective, calculate inference improvement
                self.agent.inference_optimizer.estimated_objectives.append(-obj.detach())
                objectives = torch.stack(self.agent.inference_optimizer.estimated_objectives)
                inf_imp = - objectives[0] + objectives[-1]
                self.inference_improvement.append(inf_imp)
            # note: multiply by batch size because we divide later (in optimizer)
            obj = - obj * valid * (1 - done) * self.agent.batch_size
            self.objectives['inf_opt_obj'].append(obj)
            # re-attach the prior
            if self.agent.prior_model is not None:
                self.agent.prior.reset(batch_size, dist_params={'loc': loc, 'scale': scale})

        if self.agent.direct_approx_post is not None:
            # train the direct inference model using on policy actions
            # log_prob = self.agent.direct_approx_post.log_prob(on_policy_action).sum(dim=2)
            # log_prob = log_prob.view(self.agent.n_action_samples, -1, 1).mean(dim=0)
            # self.objectives['direct_inf_opt_obj'].append(-log_prob * valid * (1 - done))
            batch_size = self.agent.approx_post._batch_size
            loc = self.agent.approx_post.dist.loc
            scale = self.agent.approx_post.dist.scale
            self.agent.approx_post.reset(batch_size, dist_params={'loc': loc.detach(), 'scale': scale.detach()})
            kl = kl_divergence(self.agent.approx_post, self.agent.direct_approx_post, n_samples=self.agent.n_action_samples, sample=on_policy_action).sum(dim=1, keepdim=True)
            self.objectives['direct_inf_opt_obj'].append(kl * valid * (1 - done))
            self.agent.approx_post.reset(batch_size, dist_params={'loc': loc, 'scale': scale})
            self.metrics['action']['direct_kl'].append((kl * (1 - done) * valid).detach())

    def _collect_alpha_objectives(self, valid, done):
        """
        Collect the objectives for the Lagrange multipliers using the target epsilons.
        """
        if self.agent.epsilons['pi'] is not None:
            target_kl = self.agent.epsilons['pi']
        else:
            # SAC heuristic: target entropy <-- |A|
            target_kl = self.agent.prior.n_variables * (1. + float(np.log(2)))
        alpha_loss = - (self.agent.log_alphas['pi'].exp() * (self.metrics['action']['kl'][-1] - target_kl).detach()) * valid * (1 - done)
        self.objectives['alpha_loss_pi'].append(alpha_loss)
        self.metrics['alpha_losses']['pi'].append(alpha_loss.detach())
        self.metrics['alphas']['pi'].append(self.agent.alphas['pi'])

        if self.agent.prior_model is not None:
            target_loc_kl = self.agent.epsilons['loc']
            alpha_loss_loc = - (self.agent.log_alphas['loc'].exp() * (self.metrics['action']['kl_prev_loc'][-1] - target_loc_kl).detach()) * valid * (1 - done)
            self.objectives['alpha_loss_loc'].append(alpha_loss_loc)
            self.metrics['alpha_losses']['loc'].append(alpha_loss_loc.detach())
            self.metrics['alphas']['loc'].append(self.agent.alphas['loc'])

            target_scale_kl = self.agent.epsilons['scale']
            alpha_loss_scale = - (self.agent.log_alphas['scale'].exp() * (self.metrics['action']['kl_prev_scale'][-1] - target_scale_kl).detach()) * valid * (1 - done)
            self.objectives['alpha_loss_scale'].append(alpha_loss_scale)
            self.metrics['alpha_losses']['scale'].append(alpha_loss_scale.detach())
            self.metrics['alphas']['scale'].append(self.agent.alphas['scale'])

    def _collect_log_scale_lim_objectives(self):
        """
        Collects the objectives for the log-scale limit parameters.
        """
        objective = 0.
        # policy
        if 'scale' in self.agent.approx_post.param_names:
            min_log_scale = self.agent.approx_post.min_log_scale
            max_log_scale = self.agent.approx_post.max_log_scale
            objective = objective - 0.01 * min_log_scale.sum()
            objective = objective + 0.01 * max_log_scale.sum()
        if 'scale' in self.agent.prior.param_names:
            min_log_scale = self.agent.prior.min_log_scale
            max_log_scale = self.agent.prior.max_log_scale
            objective = objective - 0.01 * min_log_scale.sum()
            objective = objective + 0.01 * max_log_scale.sum()
        # model
        if 'state_likelihood_model' in dir(self.agent.q_value_estimator):
            min_log_scale = self.agent.q_value_estimator.state_variable.cond_likelihood.min_log_scale
            max_log_scale = self.agent.q_value_estimator.state_variable.cond_likelihood.max_log_scale
            objective = objective - 0.01 * min_log_scale.sum()
            objective = objective + 0.01 * max_log_scale.sum()

            if self.agent.q_value_estimator.reward_likelihood_model is not None:
                min_log_scale = self.agent.q_value_estimator.reward_variable.cond_likelihood.min_log_scale
                max_log_scale = self.agent.q_value_estimator.reward_variable.cond_likelihood.max_log_scale
                objective = objective - 0.01 * min_log_scale.sum()
                objective = objective + 0.01 * max_log_scale.sum()

        self.objectives['log_scale'].append(objective)
        self.metrics['log_scale_loss']['loss'].append(objective.detach())

    def _get_value_est_objectives(self, state, off_policy_action, on_policy_action,
                                    reward, valid, done):
        """
        Collects the online objectives for the value estimator(s).
        """
        # collect the q-values for the off-policy action
        off_policy_q_values = self.agent.q_value_estimator(self.agent, state, off_policy_action, both=True, direct=True)
        self.q_values1.append(off_policy_q_values[0])
        self.q_values2.append(off_policy_q_values[1])
        # collect the target state-values
        if self.agent.state_value_estimator is not None:
            state_values = self.agent.state_value_estimator(self.agent, state, both=True)
            self.state_values1.append(state_values[0])
            self.state_values2.append(state_values[1])
            target_state_values = self.agent.state_value_estimator(self.agent, state, target=True)
            self.target_state_values.append(target_state_values)
        # collect the target q-values for the on-policy actions
        expanded_state = state.repeat(self.agent.n_action_samples, 1)
        direct_targets = not self.agent.model_value_targets
        target = self.agent.state_value_estimator is None
        target_q_values = self.agent.q_value_estimator(self.agent, expanded_state, on_policy_action, target=target, direct=direct_targets)
        target_q_values = target_q_values.view(self.agent.n_action_samples, -1, 1)[:self.agent.n_q_action_samples].mean(dim=0)
        self.target_q_values.append(target_q_values)
        # other terms for model-based Q-value estimator
        if 'state_likelihood_model' in dir(self.agent.q_value_estimator):
            # generate and evaluate the conditional likelihoods
            self.agent.q_value_estimator.generate(self.agent)
            variable = self.agent.q_value_estimator.state_variable
            self._collect_likelihood('state', state, variable, valid, done)
            if self.agent.q_value_estimator.reward_likelihood_model is not None:
                variable = self.agent.q_value_estimator.reward_variable
                self._collect_likelihood('reward', reward, variable, valid, done)

    def _collect_likelihood(self, name, x, variable, valid, done=0.):
        """
        Collects the log likelihood for a state or reward prediction.
        """
        cll = variable.cond_log_likelihood(x).view(-1, 1)
        self.objectives[name].append(-cll * (1 - done) * valid)
        self.metrics[name]['cll'].append((-cll * (1 - done) * valid).detach())

    def _collect_kl_objectives(self, on_policy_action, valid, done):
        """
        Collect the KL divergence objectives to train the prior.
        """
        if self.agent.target_prior_model is not None:
            batch_size = self.agent.prior._batch_size
            # get the distribution parameters
            target_prior_loc = self.agent.target_prior.dist.loc
            target_prior_scale = self.agent.target_prior.dist.scale
            current_prior_loc = self.agent.prior.dist.loc
            current_prior_scale = self.agent.prior.dist.scale
            if 'loc' in dir(self.agent.approx_post.dist):
                post_loc = self.agent.approx_post.dist.loc
                post_scale = self.agent.approx_post.dist.scale
                self.agent.approx_post.reset(batch_size, dist_params={'loc': post_loc.detach(), 'scale': post_scale.detach()})

            # decoupled updates on the prior
            # loc KLs
            self.agent.prior.reset(batch_size, dist_params={'loc': current_prior_loc, 'scale': target_prior_scale.detach()})
            kl_prev_loc = kl_divergence(self.agent.target_prior, self.agent.prior, n_samples=self.agent.n_action_samples).sum(dim=1, keepdim=True)
            kl_curr_loc = kl_divergence(self.agent.approx_post, self.agent.prior, n_samples=self.agent.n_action_samples, sample=on_policy_action).sum(dim=1, keepdim=True)
            # scale KLs
            self.agent.prior.reset(batch_size, dist_params={'loc': target_prior_loc.detach(), 'scale': current_prior_scale})
            kl_prev_scale = kl_divergence(self.agent.target_prior, self.agent.prior, n_samples=self.agent.n_action_samples).sum(dim=1, keepdim=True)
            kl_curr_scale = kl_divergence(self.agent.approx_post, self.agent.prior, n_samples=self.agent.n_action_samples, sample=on_policy_action).sum(dim=1, keepdim=True)

            # append the KL objectives
            self.objectives['action_kl_prev_loc'].append(self.agent.alphas['loc'] * kl_prev_loc * (1 - done) * valid)
            self.objectives['action_kl_curr_loc'].append(self.agent.alphas['pi'] * kl_curr_loc * (1 - done) * valid)
            self.objectives['action_kl_prev_scale'].append(self.agent.alphas['scale'] * kl_prev_scale * (1 - done) * valid)
            self.objectives['action_kl_curr_scale'].append(self.agent.alphas['pi'] * kl_curr_scale * (1 - done) * valid)

            # report the KL divergences
            self.metrics['action']['kl_prev_loc'].append((kl_prev_loc * (1 - done) * valid).detach())
            self.metrics['action']['kl_curr_loc'].append((kl_curr_loc * (1 - done) * valid).detach())
            self.metrics['action']['kl_prev_scale'].append((kl_prev_scale * (1 - done) * valid).detach())
            self.metrics['action']['kl_curr_scale'].append((kl_curr_scale * (1 - done) * valid).detach())

            # report the prior and target prior distribution parameters
            self.metrics['action']['prior_prev_loc'].append(target_prior_loc.detach().mean(dim=1, keepdim=True))
            self.metrics['action']['prior_prev_scale'].append(target_prior_scale.detach().mean(dim=1, keepdim=True))
            self.metrics['action']['prior_curr_loc'].append(current_prior_loc.detach().mean(dim=1, keepdim=True))
            self.metrics['action']['prior_curr_scale'].append(current_prior_scale.detach().mean(dim=1, keepdim=True))

            # reset the prior with detached parameters and approx. post. with
            # non-detached parameters to evaluate KL for approx. post.
            self.agent.prior.reset(batch_size, dist_params={'loc': current_prior_loc.detach(), 'scale': current_prior_scale.detach()})
            if 'loc' in dir(self.agent.approx_post.dist):
                self.agent.approx_post.reset(batch_size, dist_params={'loc': post_loc, 'scale': post_scale})

        if 'loc' in dir(self.agent.approx_post.dist):
            # report the approx. post. distribution parameters
            current_post_loc = self.agent.approx_post.dist.loc
            current_post_scale = self.agent.approx_post.dist.scale
            self.metrics['action']['approx_post_loc'].append(current_post_loc.detach().mean(dim=1, keepdim=True))
            self.metrics['action']['approx_post_scale'].append(current_post_scale.detach().mean(dim=1, keepdim=True))

        # evaluate the KL for reporting
        kl = kl_divergence(self.agent.approx_post, self.agent.prior, n_samples=self.agent.n_action_samples, sample=on_policy_action).sum(dim=1, keepdim=True)
        self.metrics['action']['kl'].append((kl * (1 - done) * valid).detach())

    def _collect_log_probs(self, off_policy_action, log_prob, valid):
        """
        Evaluates the log probability of the action under the current policy.
        Evaluates the importance weight from the previous log probability.
        """
        # TODO: fix this
        # action_log_prob = self.agent.approx_post.dist.log_prob(off_policy_action).sum(dim=1, keepdim=True)
        # self.log_probs['action'].append(action_log_prob * valid)
        # action_importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
        # self.importance_weights['action'].append(action_importance_weight.detach())
        self.importance_weights['action'].append(torch.ones(log_prob.shape[0], 1).to(log_prob.device))

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
        results['inf_imp'] = {'action': []}
        if len(self.inference_improvement) > 0:
            results['inf_imp']['action'] = torch.cat(self.inference_improvement, dim=0).detach().cpu()
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
                    if metric_name == 'cll':
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

        if len(self.inference_improvement) > 0:
            imp = torch.stack(self.inference_improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
            inf_imp['inference_improvement'] = imp.detach().cpu().item()

        return inf_imp

    def _get_q_targets(self):
        """
        Get the targets for the Q-value estimator.
        """
        dones = torch.stack(self.dones)
        valid = torch.stack(self.valid)
        rewards = -torch.stack(self.objectives['optimality'])[1:]
        if self.agent.state_value_estimator is not None:
            state_value = torch.stack(self.target_state_values)
        else:
            action_kl = torch.stack(self.metrics['action']['kl'])
            state_value = torch.stack(self.target_q_values) - self.agent.alphas['pi'] * action_kl
        state_value = state_value * valid * (1. - dones)
        importance_weights = torch.stack(self.importance_weights['action'])
        q_targets = retrace(state_value, rewards, importance_weights, discount=self.agent.reward_discount, l=self.agent.retrace_lambda)
        return q_targets.detach()

    def _get_v_targets(self):
        """
        Get the targets for the state-value estimator.
        """
        dones = torch.stack(self.dones)
        valid = torch.stack(self.valid)
        action_kl = torch.stack(self.metrics['action']['kl'])
        state_value = torch.stack(self.target_q_values) - self.agent.alphas['pi'] * action_kl
        state_value = state_value * valid * (1. - dones)
        return state_value[:-1].detach()

    def _train_value_estimators(self):
        """
        Get the losses for the value networks.
        """
        valid = torch.stack(self.valid)
        q_values1 = torch.stack(self.q_values1)
        q_values2 = torch.stack(self.q_values2)
        q_targets = self._get_q_targets()
        q_loss1 = 0.5 * (q_values1[:-1] - q_targets).pow(2) * valid[:-1]
        q_loss2 = 0.5 * (q_values2[:-1] - q_targets).pow(2) * valid[:-1]
        self.objectives['q_loss'] = q_loss1 + q_loss2
        self.metrics['q_loss1'] = q_loss1.mean()
        self.metrics['q_loss2'] = q_loss2.mean()
        self.metrics['q_values1'] = q_values1[:-1].mean()
        self.metrics['q_values2'] = q_values2[:-1].mean()
        self.metrics['q_value_targets'] = q_targets.mean()

        if self.agent.state_value_estimator is not None:
            state_values1 = torch.stack(self.state_values1)
            state_values2 = torch.stack(self.state_values2)
            v_targets = self._get_v_targets()
            v_loss1 = 0.5 * (state_values1[:-1] - v_targets).pow(2) * valid[:-1]
            v_loss2 = 0.5 * (state_values2[:-1] - v_targets).pow(2) * valid[:-1]
            self.objectives['v_loss'] = v_loss1 + v_loss2
            self.metrics['v_loss1'] = v_loss1.mean()
            self.metrics['v_loss2'] = v_loss2.mean()
            self.metrics['state_values1'] = state_values1[:-1].mean()
            self.metrics['state_values2'] = state_values2[:-1].mean()
            self.metrics['state_value_targets'] = v_targets.mean()

    def evaluate(self):
        """
        Combines the objectives for training.
        """
        self._train_value_estimators()
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
        self.episode = {'state': [], 'reward': [], 'done': [],
                        'action': [], 'log_prob': []}

        self.objectives = {'optimality': [], 'alpha_loss_pi': [], 'q_loss': [],
                           'log_scale': []}
        self.metrics = {'optimality': {'cll': []},
                        'action': {'kl': []},
                        'alpha_losses':{'pi': []},
                        'alphas': {'pi': []},
                        'log_scale_loss':{'loss': []}}

        if 'parameters' in dir(self.agent.inference_optimizer):
            # amortized inference optimizer
            self.objectives['inf_opt_obj'] = []

        if self.agent.direct_inference_optimizer is not None:
            # direct amortized inference optimizer
            self.objectives['direct_inf_opt_obj'] = []
            self.metrics['action']['direct_kl'] = []

        if self.agent.prior_model is not None:
            self.objectives['action_kl_prev_loc'] = []
            self.objectives['action_kl_prev_scale'] = []
            self.objectives['action_kl_curr_loc'] = []
            self.objectives['action_kl_curr_scale'] = []
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

        if 'loc' in self.agent.approx_post.param_names:
            self.metrics['action']['approx_post_loc'] = []
            self.metrics['action']['approx_post_scale'] = []

        self.distributions = {}
        self.distributions['action'] = {'prior': {param_name: [] for param_name in self.agent.prior.param_names},
                                        'approx_post': {param_name: [] for param_name in self.agent.approx_post.param_names}}

        if self.agent.direct_approx_post is not None:
            param_dict = {param_name: [] for param_name in self.agent.direct_approx_post.param_names}
            self.distributions['action']['direct_approx_post'] = param_dict

        self.log_probs = {'action': []}

        if self.agent.state_value_estimator is not None:
            self.target_state_values = []
            self.state_values1 = self.state_values2 = []
            self.objectives['v_loss'] = []

        # model-based Q-value estimator distributions
        if 'state_likelihood_model' in dir(self.agent.q_value_estimator):
            if self.agent.q_value_estimator.state_likelihood_model is not None:
                self.objectives['state'] = []
                self.metrics['state'] = {'cll': []}
                self.distributions['state'] = {'cond_like': {'loc': [], 'scale': []}}
        if 'reward_likelihood_model' in dir(self.agent.q_value_estimator):
            if self.agent.q_value_estimator.reward_likelihood_model is not None:
                self.objectives['reward'] = []
                self.metrics['reward'] = {'cll': []}
                self.distributions['reward'] = {'cond_like': {'loc': [], 'scale': []}}

        self.importance_weights = {'action': []}
        self.inference_improvement = []
        self.target_q_values = []
        self.q_values = []
        self.q_values1 = []
        self.q_values2 = []
        self.valid = []
        self.dones = []
