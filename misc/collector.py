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
        self.valid

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

    def _collect_objectives_and_log_probs(self, observation, reward, done, action, value, valid, log_prob):
        if self.done_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            weighted_done_info_gain = self.done_variable.info_gain(done, log_importance_weights, marginal_factor=self.marginal_factor)
            done_info_gain = self.done_variable.info_gain(done, log_importance_weights, marginal_factor=1.)
            done_cll = self.done_variable.cond_log_likelihood(done).view(self.n_state_samples, -1, 1).mean(dim=0)
            done_mll = self.done_variable.marginal_log_likelihood(done, log_importance_weights)
            if self._mode == 'train':
                self.objectives['done'].append(-weighted_done_info_gain * valid)
            self.metrics['done']['cll'].append((-done_cll * valid).detach())
            self.metrics['done']['mll'].append((-done_mll * valid).detach())
            self.metrics['done']['info_gain'].append((-done_info_gain * valid).detach())
            self.distributions['done']['pred']['probs'].append(self.done_variable.likelihood_dist_pred.probs.detach())
            self.distributions['done']['recon']['probs'].append(self.done_variable.likelihood_dist.probs.detach())

        if self.reward_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            weighted_reward_info_gain = self.reward_variable.info_gain(reward, log_importance_weights, marginal_factor=self.marginal_factor)
            reward_info_gain = self.reward_variable.info_gain(reward, log_importance_weights, marginal_factor=1.)
            reward_cll = self.reward_variable.cond_log_likelihood(reward).view(self.n_state_samples, -1, 1).mean(dim=0)
            reward_mll = self.reward_variable.marginal_log_likelihood(reward, log_importance_weights)
            if self._mode == 'train':
                self.objectives['reward'].append(-weighted_reward_info_gain * valid)
            self.metrics['reward']['cll'].append((-reward_cll * valid).detach())
            self.metrics['reward']['mll'].append((-reward_mll * valid).detach())
            self.metrics['reward']['info_gain'].append((-reward_info_gain * valid).detach())
            self.distributions['reward']['pred']['loc'].append(self.reward_variable.likelihood_dist_pred.loc.detach())
            self.distributions['reward']['pred']['scale'].append(self.reward_variable.likelihood_dist_pred.scale.detach())
            self.distributions['reward']['recon']['loc'].append(self.reward_variable.likelihood_dist.loc.detach())
            self.distributions['reward']['recon']['scale'].append(self.reward_variable.likelihood_dist.scale.detach())

        if self.obs_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            weighted_observation_info_gain = self.observation_variable.info_gain(observation, log_importance_weights, marginal_factor=self.marginal_factor)
            observation_info_gain = self.observation_variable.info_gain(observation, log_importance_weights, marginal_factor=1.)
            observation_cll = self.observation_variable.cond_log_likelihood(observation).view(self.n_state_samples, -1, 1).mean(dim=0)
            observation_mll = self.observation_variable.marginal_log_likelihood(observation, log_importance_weights)
            if self._mode == 'train':
                self.objectives['observation'].append(-weighted_observation_info_gain * (1 - done) * valid)
            self.metrics['observation']['cll'].append((-observation_cll * (1 - done) * valid).detach())
            self.metrics['observation']['mll'].append((-observation_mll * (1 - done) * valid).detach())
            self.metrics['observation']['info_gain'].append((-observation_info_gain * (1 - done) * valid).detach())
            self.distributions['observation']['pred']['loc'].append(self.observation_variable.likelihood_dist_pred.loc.detach())
            self.distributions['observation']['pred']['scale'].append(self.observation_variable.likelihood_dist_pred.scale.detach())
            self.distributions['observation']['recon']['loc'].append(self.observation_variable.likelihood_dist.loc.detach())
            self.distributions['observation']['recon']['scale'].append(self.observation_variable.likelihood_dist.scale.detach())

        optimality_cll = self.optimality_scale * (reward - 1.)
        if self._mode == 'train':
            self.objectives['optimality'].append(-optimality_cll * valid)
        self.metrics['optimality']['cll'].append((-optimality_cll * valid).detach())

        state_kl = self.state_variable.kl_divergence()
        obj_state_kl = self.kl_factor['state'] * torch.clamp(state_kl, min=self.kl_min['state']).sum(dim=1, keepdim=True)
        state_kl = state_kl.sum(dim=1, keepdim=True)
        if self._mode == 'train':
            self.objectives['state'].append(obj_state_kl * (1 - done) * valid)
        self.metrics['state']['kl'].append((state_kl * (1 - done) * valid).detach())
        self.distributions['state']['prior']['loc'].append(self.state_variable.prior_dist.loc.detach())
        if hasattr(self.state_variable.prior_dist, 'scale'):
            self.distributions['state']['prior']['scale'].append(self.state_variable.prior_dist.scale.detach())
        if self.state_variable.approx_post_dist is not None:
            self.distributions['state']['approx_post']['loc'].append(self.state_variable.approx_post_dist.loc.detach())
            self.distributions['state']['approx_post']['scale'].append(self.state_variable.approx_post_dist.scale.detach())

        action_kl = self.action_variable.kl_divergence()
        obj_action_kl = self.kl_factor['action'] * torch.clamp(action_kl, min=self.kl_min['action'])
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action_kl = action_kl.view(-1, 1)
            obj_action_kl = obj_action_kl.view(-1, 1)
            self.distributions['action']['prior']['probs'].append(self.action_variable.prior_dist.probs.detach())
            self.distributions['action']['approx_post']['probs'].append(self.action_variable.approx_post_dist.probs.detach())
        else:
            action_kl = action_kl.sum(dim=1, keepdim=True)
            obj_action_kl = obj_action_kl.sum(dim=1, keepdim=True)
            self.distributions['action']['prior']['loc'].append(self.action_variable.prior_dist.loc.detach())
            self.distributions['action']['prior']['scale'].append(self.action_variable.prior_dist.scale.detach())
            self.distributions['action']['approx_post']['loc'].append(self.action_variable.approx_post_dist.loc.detach())
            self.distributions['action']['approx_post']['scale'].append(self.action_variable.approx_post_dist.scale.detach())
        if self._mode == 'train':
            self.objectives['action'].append(obj_action_kl * (1 - done) * valid)
        self.metrics['action']['kl'].append((action_kl * (1 - done) * valid).detach())

        if self._mode == 'train':
            # if action is None:
            #     action = self.action_variable.sample()
            action = self._convert_action(action)
            action_log_prob = self.action_variable.approx_post_dist.log_prob(action)
            # if log_prob is None:
            #     log_prob = action_log_prob
            if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
                action_log_prob = action_log_prob.view(-1, 1)
            else:
                action_log_prob = action_log_prob.sum(dim=1, keepdim=True)
            self.log_probs['action'].append(action_log_prob * valid)
            state = self.state_variable.sample()
            state_log_prob = self.state_variable.approx_post_dist.log_prob(state)
            state_log_prob = state_log_prob.sum(dim=1, keepdim=True)
            self.log_probs['state'].append(state_log_prob * valid)
            # if log_prob is None:
            #     log_prob = action_log_prob
            importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
            self.importance_weights['action'].append(importance_weight.detach())

    def _collect_episode(self, observation, reward, done):
        # collect the variables for this step of the episode
        if not done:
            self.episode['observation'].append(observation)
            self.episode['action'].append(self.action_variable.sample().detach())
            self.episode['state'].append(self.state_variable.sample().detach())
            act = self._convert_action(self.action_variable.sample().detach())
            action_log_prob = self.action_variable.approx_post_dist.log_prob(act)
            if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
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
            self.marginal_factor *= self.marginal_factor_anneal_rate
            self.marginal_factor = min(self.marginal_factor, 1.)

            self.kl_min['state'] *= self.kl_min_anneal_rate['state']
            self.kl_min['action'] *= self.kl_min_anneal_rate['action']
            self.kl_factor['state'] *= self.kl_factor_anneal_rate['state']
            self.kl_factor['action'] *= self.kl_factor_anneal_rate['action']
            self.kl_factor['state'] = min(self.kl_factor['state'], 1.)
            self.kl_factor['action'] = min(self.kl_factor['action'], 1.)

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

    def reset(self):
        # reset the episode, objectives, and log probs
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'log_prob': []}

        self.objectives = {'optimality': [], 'state': [], 'action': []}
        self.metrics = {'optimality': {'cll': []},
                        'state': {'kl': []},
                        'action': {'kl': []}}
        self.distributions = {'state': {'prior': {'loc': [], 'scale': []}, 'approx_post': {'loc': [], 'scale': []}}}
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            self.distributions['action'] = {'prior': {'probs': []},
                                            'approx_post': {'probs': []}}
        else:
            self.distributions['action'] = {'prior': {'loc': [], 'scale': []},
                                            'approx_post': {'loc': [], 'scale': []}}

        if self.observation_variable is not None:
            self.objectives['observation'] = []
            self.metrics['observation'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['observation'] = {'pred': {'loc': [], 'scale': []},
                                                 'recon': {'loc': [], 'scale': []}}
        if self.reward_variable is not None:
            self.objectives['reward'] = []
            self.metrics['reward'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['reward'] = {'pred': {'loc': [], 'scale': []},
                                            'recon': {'loc': [], 'scale': []}}
        if self.done_variable is not None:
            self.objectives['done'] = []
            self.metrics['done'] = {'cll': [], 'info_gain': [], 'mll': []}
            self.distributions['done'] = {'pred': {'probs': []}, 'recon': {'probs': []}}
        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': [], 'state': []}
        self.rollout_lenghts = []
        self.importance_weights = {'action': []}
        self.values = []
