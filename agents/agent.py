import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from misc import one_hot_to_index
# from misc.collector import Collector
from misc.normalization import Normalizer


class Agent(nn.Module):
    """
    Variational RL Agent
    """
    def __init__(self, misc_args):
        super(Agent, self).__init__()

        # models
        self.state_prior_model = None
        self.action_prior_model = None
        self.obs_likelihood_model = None
        self.reward_likelihood_model = None
        self.done_likelihood_model = None
        self.state_inference_model = None
        self.action_inference_model = None
        self.value_model = None

        # variables
        self.state_variable = None
        self.action_variable = None
        self.observation_variable = None
        self.reward_variable = None
        self.done_variable = None

        # miscellaneous
        self.optimality_scale = misc_args['optimality_scale']
        self.kl_min = {'state': misc_args['kl_min']['state'],
                       'action': misc_args['kl_min']['action']}
        self.kl_min_anneal_rate = {'state': misc_args['kl_min_anneal_rate']['state'],
                                   'action': misc_args['kl_min_anneal_rate']['action']}
        self.kl_factor = {'state': misc_args['kl_factor']['state'],
                          'action': misc_args['kl_factor']['action']}
        self.kl_factor_anneal_rate = {'state': misc_args['kl_factor_anneal_rate']['state'],
                                      'action': misc_args['kl_factor_anneal_rate']['action']}

        # mode (either 'train' or 'eval')
        self._mode = 'train'

        #
        # self.collector = Collector(self)

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

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = misc_args['gae_lambda']
        self.reward_discount = misc_args['reward_discount']

        # normalizers for various quantities
        self.return_normalizer = None
        if misc_args['normalize_returns']:
            self.return_normalizer = Normalizer(shift=False, clip_value=10.)
        self.advantage_normalizer = None
        if misc_args['normalize_advantages']:
            self.advantage_normalizer = Normalizer(clip_value=10.)
        self.obs_normalizer = None
        if misc_args['normalize_observations']:
            observation_size = state_inference_args['n_input']
            # TODO: should set this in a better way, in case of image input
            self.obs_normalizer = Normalizer(shape=(observation_size), clip_value=10.)

    def act(self, observation, reward=None, done=False, action=None, valid=None, log_prob=None, random=False):
        observation, reward, action, done, valid, log_prob = self._change_device(observation, reward, action, done, valid, log_prob)
        self.step_state(observation=observation, reward=reward, done=done, valid=valid)
        self.state_inference(observation=observation, reward=reward, done=done, valid=valid)
        self.step_action(observation=observation, reward=reward, done=done, valid=valid, action=action)
        if not random:
            self.action_inference(observation=observation, reward=reward, done=done, valid=valid, action=action)
        value = self.estimate_value(observation=observation, reward=reward, done=done, valid=valid)
        self._collect_objectives_and_log_probs(observation, reward, done, action, value, valid, log_prob)
        self.valid.append(valid)
        if self._mode == 'train':
            self._prev_action = action
        else:
            self._collect_episode(observation, reward, done)
            if observation is not None:
                action = self.action_variable.sample()
                action = self._convert_action(action).cpu().numpy()
        return action

    @abstractmethod
    def state_inference(self, observation, reward, done, valid):
        pass

    @abstractmethod
    def action_inference(self, observation, reward, done, valid, action=None):
        pass

    @abstractmethod
    def step_state(self, observation, reward, done, valid):
        pass

    @abstractmethod
    def step_action(self, observation, reward, done, valid, action=None):
        pass

    def generate(self):
        self.generate_observation()
        self.generate_reward()
        self.generate_done()

    def generate_observation(self):
        # generate the conditional likelihood for the observation
        pass

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        pass

    def generate_done(self):
        # generate the conditional likelihood for episode being done
        pass

    def estimate_value(self, observation, reward, done, valid):
        # estimate the value of the current state
        pass

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
        if self.return_normalizer:
            # normalize the future terms
            future_terms = self.return_normalizer(future_terms.squeeze(-1))
            future_terms = future_terms.unsqueeze(-1)
        deltas = future_terms + self.reward_discount * values[1:] * valid[1:] - values[:-1]
        advantages = deltas.detach()
        # use generalized advantage estimator
        for i in range(advantages.shape[0]-1, 0, -1):
            advantages[i-1] = advantages[i-1] + self.reward_discount * self.gae_lambda * advantages[i] * valid[i]
        # if self.advantage_normalizer and update:
        #     self.advantage_normalizer.update(advantages.squeeze(-1))
        return advantages

    def estimate_returns(self, update=False):
        # calculate the discounted Monte Carlo return
        valid = torch.stack(self.valid)
        discounted_returns = self.get_future_terms()
        for i in range(discounted_returns.shape[0]-1, 0, -1):
            discounted_returns[i-1] += self.reward_discount * discounted_returns[i] * valid[i]
        if self.return_normalizer and update:
            # normalize the discounted returns
            self.return_normalizer.update(discounted_returns.squeeze(-1))
        return discounted_returns

    def evaluate(self):
        # evaluate the objective, averaged over the batch, backprop
        results = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0)

        # average metrics over time and batch (for reporting)
        for variable_name, metric_dict in self.metrics.items():
            for metric_name, metric in metric_dict.items():
                met = torch.stack(metric).sum(dim=0).div(n_valid_steps).mean(dim=0)
                if metric_name in ['cll', 'mll', 'info_gain']:
                    # negate for plotting
                    met = met * -1
                results[variable_name + '_' + metric_name] = met.detach().cpu().item()

        # evaluate inference improvement (for reporting)
        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                results[name + '_improvement'] = imp.detach().cpu().item()

        # report the average planning rollout lengths
        if len(self.rollout_lengths) > 0:
            results['rollout_length'] = sum(self.rollout_lengths) / len(self.rollout_lengths)

        # sum the objectives (for training)
        n_steps = len(self.objectives['optimality'])
        total_objective = torch.zeros(n_steps, self.batch_size, 1).to(self.device)
        for objective_name, objective in self.objectives.items():
            total_objective = total_objective + torch.stack(objective)

        # calculate the REINFORCE terms
        if self.value_model:
            # use a bootstrapped value estimate as a baseline
            values = torch.stack(self.values)
            advantages = self.estimate_advantages()
            # calculate value loss
            returns = advantages + values[:-1].detach()
            value_loss = 0.5 * (values[:-1] - returns).pow(2)
            total_objective[:-1] = total_objective[:-1] + value_loss
            results['value'] = value_loss.sum(dim=0).div(n_valid_steps).mean(dim=0).detach().cpu().item()
            advantages_mean = advantages.sum(dim=0).div(n_valid_steps).mean(dim=0)
            advantages_std = torch.sqrt((advantages - advantages_mean).pow(2).mul(valid[:-1]).sum(dim=0).div(n_valid_steps).mean(dim=0))
            advantages = (advantages - advantages_mean) / advantages_std
        else:
            raise NotImplementedError
        # add the REINFORCE terms to the total objective
        action_log_probs = torch.stack(self.log_probs['action'])
        action_importance_weights = torch.stack(self.importance_weights['action']).detach()
        action_reinforce_terms = - action_importance_weights[:-1] * action_log_probs[:-1] * advantages
        if not self.action_variable.inference_type == 'iterative':
            # include the policy gradients in the total objective
            total_objective[:-1] = total_objective[:-1] + action_reinforce_terms

        results['importance_weights'] = action_importance_weights.sum(dim=0).div(n_valid_steps).mean(dim=0).detach().cpu().item()
        results['policy_gradients'] = action_reinforce_terms.sum(dim=0).div(n_valid_steps-1).mean(dim=0).detach().cpu().item()
        results['advantages'] = advantages.sum(dim=0).div(n_valid_steps-1).mean(dim=0).detach().cpu().item()

        # time average, batch average, and backprop
        total_objective = total_objective.sum(dim=0).div(n_valid_steps)
        total_objective = total_objective.mean(dim=0)
        total_objective.sum().backward()

        # calculate the average gradient for each model (for reporting)
        grads_dict = {}
        grad_norm_dict = {}
        for model_name, params in self.parameters().items():
            grads = [param.grad.view(-1) for param in params if param.grad is not None]
            if len(grads) > 0:
                grads = torch.cat(grads, dim=0)
                grads_dict[model_name] = grads.abs().mean().cpu().numpy().item()
                grad_norm_dict[model_name] = grads.norm().cpu().numpy().item()
        results['grads'] = grads_dict
        results['grad_norms'] = grad_norm_dict

        results['kl_min'] = self.kl_min
        results['kl_factor'] = self.kl_factor
        results['marginal_factor'] = self.marginal_factor

        return results

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

    def _convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        # else:
        #     action = action.detach()
        return action

    def _change_device(self, observation, reward, action, done, valid, log_prob):
        if observation is None:
            observation = torch.zeros(self.episode['observation'][0].shape)
        elif type(observation) == np.ndarray:
            observation = torch.from_numpy(observation.astype('float32')).view(1, -1) # hack
        if observation.device != self.device:
            observation = observation.to(self.device)
        if type(reward) in [float, int]:
            reward = torch.tensor(reward).to(torch.float32).view(1, 1)
        elif type(reward) == np.ndarray:
            reward = torch.from_numpy(reward.astype('float32')).view(1, 1) # hack
        if reward.device != self.device:
            reward = reward.to(self.device)
        if action is not None:
            if action.device != self.device:
                action = action.to(self.device)
        if type(done) == bool:
            done = torch.tensor(done).to(torch.float32).view(1, 1)
        elif type(done) == np.ndarray:
            done = torch.from_numpy(done.astype('float32')).view(1, 1) # hack
        if done.device != self.device:
            done = done.to(self.device)
        if valid is None:
            valid = torch.ones(done.shape[0], 1)
        if valid.device != self.device:
            valid = valid.to(self.device)
        if log_prob is not None:
            log_prob = log_prob.to(self.device)
        return observation, reward, action, done, valid, log_prob

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

    def reset(self, batch_size=1):
        # reset the variables
        self.state_variable.reset(batch_size)
        self.action_variable.reset(batch_size)
        if self.observation_variable is not None:
            self.observation_variable.reset(batch_size)
        if self.reward_variable is not None:
            self.reward_variable.reset(batch_size)
        if self.done_variable is not None:
            self.done_variable.reset(batch_size)

        # reset the networks
        if self.state_prior_model is not None:
            self.state_prior_model.reset(batch_size)
        if self.action_prior_model is not None:
            self.action_prior_model.reset(batch_size)
        if self.obs_likelihood_model is not None:
            self.obs_likelihood_model.reset(batch_size)
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.reset(batch_size)
        if self.done_likelihood_model is not None:
            self.done_likelihood_model.reset(batch_size)

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

        self._prev_action = None
        self.valid = []
        self.batch_size = batch_size

    @property
    def device(self):
        return self.generative_parameters()[0].device

    def train(self, *args):
        super(Agent, self).train(*args)
        self._mode = 'train'

    def eval(self, *args):
        super(Agent, self).eval(*args)
        self._mode = 'eval'

    def parameters(self):
        param_dict = {}

        if self.state_inference_model is not None:
            param_dict['state_inference_model'] = nn.ParameterList()
            param_dict['state_inference_model'].extend(list(self.state_inference_model.parameters()))
            param_dict['state_inference_model'].extend(list(self.state_variable.inference_parameters()))

        if self.action_inference_model is not None:
            param_dict['action_inference_model'] = nn.ParameterList()
            param_dict['action_inference_model'].extend(list(self.action_inference_model.parameters()))
            param_dict['action_inference_model'].extend(list(self.action_variable.inference_parameters()))

        if self.state_prior_model is not None:
            param_dict['state_prior_model'] = nn.ParameterList()
            param_dict['state_prior_model'].extend(list(self.state_prior_model.parameters()))
            param_dict['state_prior_model'].extend(list(self.state_variable.generative_parameters()))

        if self.action_prior_model is not None:
            param_dict['action_prior_model'] = nn.ParameterList()
            param_dict['action_prior_model'].extend(list(self.action_prior_model.parameters()))
            param_dict['action_prior_model'].extend(list(self.action_variable.generative_parameters()))

        if self.obs_likelihood_model is not None:
            param_dict['obs_likelihood_model'] = nn.ParameterList()
            param_dict['obs_likelihood_model'].extend(list(self.obs_likelihood_model.parameters()))
            param_dict['obs_likelihood_model'].extend(list(self.observation_variable.parameters()))

        if self.reward_likelihood_model is not None:
            param_dict['reward_likelihood_model'] = nn.ParameterList()
            param_dict['reward_likelihood_model'].extend(list(self.reward_likelihood_model.parameters()))
            param_dict['reward_likelihood_model'].extend(list(self.reward_variable.parameters()))

        if self.done_likelihood_model is not None:
            param_dict['done_likelihood_model'] = nn.ParameterList()
            param_dict['done_likelihood_model'].extend(list(self.done_likelihood_model.parameters()))
            param_dict['done_likelihood_model'].extend(list(self.done_variable.parameters()))

        if self.value_model is not None:
            param_dict['value_model'] = nn.ParameterList()
            param_dict['value_model'].extend(list(self.value_model.parameters()))

        return param_dict

    def inference_parameters(self):
        params = nn.ParameterList()
        if self.state_inference_model is not None:
            params.extend(list(self.state_inference_model.parameters()))
            params.extend(list(self.state_variable.inference_parameters()))
        if self.action_inference_model is not None:
            params.extend(list(self.action_inference_model.parameters()))
            params.extend(list(self.action_variable.inference_parameters()))
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        if self.state_prior_model is not None:
            params.extend(list(self.state_prior_model.parameters()))
            params.extend(list(self.state_variable.generative_parameters()))
        if self.action_prior_model is not None:
            params.extend(list(self.action_prior_model.parameters()))
            params.extend(list(self.action_variable.generative_parameters()))
        if self.obs_likelihood_model is not None:
            params.extend(list(self.obs_likelihood_model.parameters()))
            params.extend(list(self.observation_variable.parameters()))
        if self.reward_likelihood_model is not None:
            params.extend(list(self.reward_likelihood_model.parameters()))
            params.extend(list(self.reward_variable.parameters()))
        if self.done_likelihood_model is not None:
            params.extend(list(self.done_likelihood_model.parameters()))
            params.extend(list(self.done_variable.parameters()))
        return params

    def inference_mode(self):
        self.state_variable.inference_mode()
        self.action_variable.inference_mode()
        if self.state_prior_model is not None:
            self.state_prior_model.detach_hidden_state()
        if self.action_prior_model is not None:
            self.action_prior_model.detach_hidden_state()
        if self.obs_likelihood_model is not None:
            self.obs_likelihood_model.detach_hidden_state()
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.detach_hidden_state()
        if self.done_likelihood_model is not None:
            self.done_likelihood_model.detach_hidden_state()

    def generative_mode(self):
        self.state_variable.generative_mode()
        self.action_variable.generative_mode()
        if self.state_prior_model is not None:
            self.state_prior_model.attach_hidden_state()
        if self.action_prior_model is not None:
            self.action_prior_model.attach_hidden_state()
        if self.obs_likelihood_model is not None:
            self.obs_likelihood_model.attach_hidden_state()
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.attach_hidden_state()
        if self.done_likelihood_model is not None:
            self.done_likelihood_model.attach_hidden_state()

    def load(self, state_dict):
        # load the state dictionary for the agent
        for k, v in state_dict.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                try:
                    attr.load_state_dict(v)
                except:
                    print('WARNING: could not load ' + k + '.')
            else:
                raise ValueError
