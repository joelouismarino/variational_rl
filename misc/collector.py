import torch


class Collector:
    """
    Collects objectives and episodes for an agent.
    """
    def __init__(self, agent):
        # link to the corresponding agent
        self.agent = agent

        # stores the variables
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'log_prob': []}

        # stores the objectives during training
        self.objectives = {'optimality': [], 'state': [], 'action': []}
        # stores the metrics
        self.metrics = {'optimality': {'cll': []},
                        'state': {'kl': []},
                        'action': {'kl': []}}
        # stores the distributions
        self.distributions = {'state': {'prior': {'loc': [], 'scale': []}, 'approx_post': {'loc': [], 'scale': []}},
                              'action': {'prior': {'probs': []}, 'approx_post': {'probs': []}}}
        # stores inference improvement during training
        self.inference_improvement = {'state': [], 'action': []}
        # stores planning rollout lengths during training
        self.rollout_lengths = []
        # stores the log probabilities during training
        self.log_probs = {'action': [], 'state': []}
        # stores the importance weights during training
        self.importance_weights = {'action': []}
        # store the values during training
        self.values = []

        self.rollout_lenghts = []
        self.valid = []

    def get_future_terms(self):
        # get the future terms in the objective
        valid = torch.stack(self.valid)
        optimality = (-torch.stack(self.metrics['optimality']['cll']) + 1.) * valid
        future_terms = optimality[1:]
        # TODO: should only include these if the action distribution is not reparameterizable
        # if self.action_prior_model is not None:
        #     action_kl = torch.stack(self.metrics['action']['kl']) * valid
        #     future_terms = future_terms - action_kl[1:]
        # if self.state_prior_model is not None:
        #     state_kl = torch.stack(self.metrics['state']['kl']) * valid
        #     future_terms = future_terms - state_kl[1:]
        # if self.obs_likelihood_model is not None:
        #     obs_info_gain = torch.stack(self.metrics['observation']['info_gain']) * valid
        #     reward_info_gain = torch.stack(self.metrics['reward']['info_gain']) * valid
        #     done_info_gain = torch.stack(self.metrics['done']['info_gain']) * valid
        #     future_terms = future_terms + obs_info_gain[1:] + reward_info_gain[1:] + done_info_gain[1:]
        return future_terms

    def estimate_advantages(self, update=False):
        # estimate bootstrapped advantages
        valid = torch.stack(self.valid)
        values = torch.stack(self.values)
        future_terms = self.get_future_terms()
        if self.agent.return_normalizer:
            # normalize the future terms
            future_terms = self.agent.return_normalizer(future_terms.squeeze(-1))
            future_terms = future_terms.unsqueeze(-1)
        deltas = future_terms + self.agent.reward_discount * values[1:] * valid[1:] - values[:-1]
        advantages = deltas.detach()
        # use generalized advantage estimator
        for i in range(advantages.shape[0]-1, 0, -1):
            advantages[i-1] = advantages[i-1] + self.agent.reward_discount * self.agent.gae_lambda * advantages[i] * valid[i]
        # if self.agent.advantage_normalizer and update:
        #     self.agent.advantage_normalizer.update(advantages.squeeze(-1))
        return advantages

    def estimate_returns(self, update=False):
        # calculate the discounted Monte Carlo return
        valid = torch.stack(self.valid)
        discounted_returns = self.get_future_terms()
        for i in range(discounted_returns.shape[0]-1, 0, -1):
            discounted_returns[i-1] += self.agent.reward_discount * discounted_returns[i] * valid[i]
        if self.agent.return_normalizer and update:
            # normalize the discounted returns
            self.agent.return_normalizer.update(discounted_returns.squeeze(-1))
        return discounted_returns

    def _collect_likelihood(self, name, obs, variable, valid, done=0.):
        log_importance_weights = self.agent.state_variable.log_importance_weights().detach()
        weighted_info_gain = variable.info_gain(obs, log_importance_weights, marginal_factor=self.agent.marginal_factor)
        info_gain = variable.info_gain(obs, log_importance_weights, marginal_factor=1.)
        cll = variable.cond_log_likelihood(obs).view(self.agent.n_state_samples, -1, 1).mean(dim=0)
        mll = variable.marginal_log_likelihood(obs, log_importance_weights)
        if self.agent._mode == 'train':
            self.objectives[name].append(-weighted_info_gain * (1 - done) * valid)
        self.metrics[name]['cll'].append((-cll * (1 - done) * valid).detach())
        self.metrics[name]['mll'].append((-mll * (1 - done) * valid).detach())
        self.metrics[name]['info_gain'].append((-info_gain * (1 - done) * valid).detach())
        if 'probs' in dir(variable.likelihood_dist):
            self.distributions[name]['pred']['probs'].append(variable.likelihood_dist_pred.probs.detach())
            self.distributions[name]['recon']['probs'].append(variable.likelihood_dist.probs.detach())
        else:
            self.distributions[name]['pred']['loc'].append(variable.likelihood_dist_pred.loc.detach())
            self.distributions[name]['pred']['scale'].append(variable.likelihood_dist_pred.scale.detach())
            self.distributions[name]['recon']['loc'].append(variable.likelihood_dist.loc.detach())
            self.distributions[name]['recon']['scale'].append(variable.likelihood_dist.scale.detach())

    def _collect_kl(self, name, variable, valid, done):
        kl = variable.kl_divergence()
        obj_kl = self.agent.kl_factor[name] * torch.clamp(kl, min=self.agent.kl_min[name]).sum(dim=1, keepdim=True)
        if variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            # discrete
            kl = kl.view(-1, 1)
            obj_kl = obj_kl.view(-1, 1)
            self.distributions[name]['prior']['probs'].append(variable.prior_dist.probs.detach())
            self.distributions[name]['approx_post']['probs'].append(variable.approx_post_dist.probs.detach())
        else:
            # continuous
            kl = kl.sum(dim=1, keepdim=True)
            obj_kl = obj_kl.sum(dim=1, keepdim=True)
            self.distributions[name]['prior']['loc'].append(variable.prior_dist.loc.detach())
            if hasattr(variable.prior_dist, 'scale'):
                self.distributions[name]['prior']['scale'].append(variable.prior_dist.scale.detach())
            self.distributions[name]['approx_post']['loc'].append(variable.approx_post_dist.loc.detach())
            self.distributions[name]['approx_post']['scale'].append(variable.approx_post_dist.scale.detach())
        if self.agent._mode == 'train':
            self.objectives[name].append(obj_kl * (1 - done) * valid)
        self.metrics[name]['kl'].append((kl * (1 - done) * valid).detach())

    def _collect_log_probs(self, action, log_prob, valid):
        action = self.agent._convert_action(action)
        action_log_prob = self.agent.action_variable.approx_post_dist.log_prob(action)
        if self.agent.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action_log_prob = action_log_prob.view(-1, 1)
        else:
            action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
        self.log_probs['action'].append(action_log_prob * valid)
        state = self.agent.state_variable.sample()
        state_log_prob = self.agent.state_variable.approx_post_dist.log_prob(state)
        state_log_prob = state_log_prob.sum(dim=1, keepdim=True)
        self.log_probs['state'].append(state_log_prob * valid)
        importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
        self.importance_weights['action'].append(importance_weight.detach())

    def _collect_episode(self, observation, reward, done):
        # collect the variables for this step of the episode
        if not done:
            self.episode['observation'].append(observation)
            self.episode['action'].append(self.agent.action_variable.sample().detach())
            self.episode['state'].append(self.agent.state_variable.sample().detach())
            act = self.agent._convert_action(self.agent.action_variable.sample().detach())
            action_log_prob = self.agent.action_variable.approx_post_dist.log_prob(act)
            if self.agent.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
                action_log_prob = action_log_prob.view(-1, 1)
            else:
                action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
            self.episode['log_prob'].append(action_log_prob.detach())
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            state = self.episode['state'][0]
            log_prob = self.episode['log_prob'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['state'].append(state.new(state.shape).zero_())
            self.episode['log_prob'].append(log_prob.new(log_prob.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

        if done:
            self.agent.marginal_factor *= self.agent.marginal_factor_anneal_rate
            self.agent.marginal_factor = min(self.agent.marginal_factor, 1.)
            self.agent.kl_min['state'] *= self.agent.kl_min_anneal_rate['state']
            self.agent.kl_min['action'] *= self.agent.kl_min_anneal_rate['action']
            self.agent.kl_factor['state'] *= self.agent.kl_factor_anneal_rate['state']
            self.agent.kl_factor['action'] *= self.agent.kl_factor_anneal_rate['action']
            self.agent.kl_factor['state'] = min(self.agent.kl_factor['state'], 1.)
            self.agent.kl_factor['action'] = min(self.agent.kl_factor['action'], 1.)

    def collect(self, observation, reward, done, action, value, valid, log_prob):
        optimality_cll = self.agent.optimality_scale * (reward - 1.)
        if self.agent._mode == 'train':
            self.objectives['optimality'].append(-optimality_cll * valid)
        self.metrics['optimality']['cll'].append((-optimality_cll * valid).detach())

        if self.agent.done_likelihood_model is not None:
            self._collect_likelihood('done', done, self.agent.done_variable, valid)

        if self.agent.reward_likelihood_model is not None:
            self._collect_likelihood('reward', reward, self.agent.reward_variable, valid)

        if self.agent.obs_likelihood_model is not None:
            self._collect_likelihood('observation', observation, self.agent.observation_variable, valid, done)

        self._collect_kl('state', self.agent.state_variable, valid, done)

        self._collect_kl('action', self.agent.action_variable, valid, done)

        if self.agent._mode == 'train':
            self._collect_log_probs(action, log_prob, valid)
        else:
            self._collect_episode(observation, reward, done)

        self.valid.append(valid)

    def get_episode(self):
        """
        Concatenate each variable, objective in the episode. Put on the CPU.
        """
        results = {}
        # copy over the episode itself
        for k, v in self.episode.items():
            results[k] = torch.cat(v, dim=0).detach().cpu()
        # get the metrics
        results['metrics'] = {}
        for k, v in self.metrics.items():
            results['metrics'][k] = {}
            for kk, vv in v.items():
                results['metrics'][k][kk] = torch.cat(vv, dim=0).detach().cpu()
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
        results['return'] = self.estimate_returns(update=True).mean(dim=1).detach().cpu()
        results['value'] = torch.cat(self.values, dim=0).detach().cpu()
        results['advantage'] = torch.zeros(results['value'].shape)
        results['advantage'][:-1] = self.estimate_advantages(update=True).mean(dim=1).detach().cpu()
        return results

    def get_metrics(self):
        metrics = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0)

        # average metrics over time and batch (for reporting)
        for variable_name, metric in self.metrics.items():
            if type(metric) == dict:
                for metric_name, met in metric.items():
                    m = torch.stack(met).sum(dim=0).div(n_valid_steps).mean(dim=0)
                    if metric_name in ['cll', 'mll', 'info_gain']:
                        # negate for plotting
                        m = m * -1
                    metrics[variable_name + '_' + metric_name] = m.detach().cpu().item()
            else:
                m = metric.sum(dim=0).div(n_valid_steps).mean(dim=0)
                metrics[variable_name] = m.detach().cpu().item()

        return metrics

    def get_inf_imp(self):
        inf_imp = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0)

        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                inf_imp[name + '_improvement'] = imp.detach().cpu().item()

        return inf_imp

    def evaluate(self):
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0)

        # sum the objectives (for training)
        n_steps = len(self.objectives['optimality'])
        total_objective = torch.zeros(n_steps, self.agent.batch_size, 1).to(self.agent.device)
        for objective_name, objective in self.objectives.items():
            total_objective = total_objective + torch.stack(objective)

        values = torch.stack(self.values)
        advantages = self.estimate_advantages()

        # calculate value loss
        returns = advantages + values[:-1].detach()
        value_loss = 0.5 * (values[:-1] - returns).pow(2)
        total_objective[:-1] = total_objective[:-1] + value_loss

        # normalize the advantages
        advantages_mean = advantages.sum(dim=0).div(n_valid_steps).mean(dim=0)
        advantages_std = torch.sqrt((advantages - advantages_mean).pow(2).mul(valid[:-1]).sum(dim=0).div(n_valid_steps).mean(dim=0))
        advantages = (advantages - advantages_mean) / advantages_std

        # calculate policy gradient terms and add them to the total objective
        action_log_probs = torch.stack(self.log_probs['action'])
        action_importance_weights = torch.stack(self.importance_weights['action']).detach()
        action_reinforce_terms = - action_importance_weights[:-1] * action_log_probs[:-1] * advantages
        if not self.agent.action_variable.inference_type == 'iterative':
            # include the policy gradients in the total objective
            total_objective[:-1] = total_objective[:-1] + action_reinforce_terms

        total_objective = total_objective.sum(dim=0).div(n_valid_steps).mean(dim=0).sum()

        # save value / policy gradient metrics
        self.metrics['value'] = value_loss
        self.metrics['importance_weights'] = action_importance_weights
        self.metrics['policy_gradients'] = action_reinforce_terms
        self.metrics['advantages'] = advantages

        return total_objective

    def get_grads(self):
        # calculate the average gradient for each model (for reporting)
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
        # reset the episode, objectives, and log probs
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'log_prob': []}

        self.objectives = {'optimality': [], 'state': [], 'action': []}
        self.metrics = {'optimality': {'cll': []},
                        'state': {'kl': []},
                        'action': {'kl': []}}
        self.distributions = {'state': {'prior': {'loc': [], 'scale': []}, 'approx_post': {'loc': [], 'scale': []}}}
        if self.agent.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            self.distributions['action'] = {'prior': {'probs': []},
                                            'approx_post': {'probs': []}}
        else:
            self.distributions['action'] = {'prior': {'loc': [], 'scale': []},
                                            'approx_post': {'loc': [], 'scale': []}}

        if self.agent.observation_variable is not None:
            self.objectives['observation'] = []
            self.metrics['observation'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['observation'] = {'pred': {'loc': [], 'scale': []},
                                                 'recon': {'loc': [], 'scale': []}}
        if self.agent.reward_variable is not None:
            self.objectives['reward'] = []
            self.metrics['reward'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['reward'] = {'pred': {'loc': [], 'scale': []},
                                            'recon': {'loc': [], 'scale': []}}
        if self.agent.done_variable is not None:
            self.objectives['done'] = []
            self.metrics['done'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['done'] = {'pred': {'probs': []}, 'recon': {'probs': []}}

        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': [], 'state': []}
        self.rollout_lenghts = []
        self.importance_weights = {'action': []}
        self.values = []
        self.valid = []
