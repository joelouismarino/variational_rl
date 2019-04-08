import torch
import torch.nn as nn
import torch.distributions as dist
from abc import abstractmethod
from ..modules.networks import get_network
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index


class Agent(nn.Module):
    """
    Variational RL Agent
    """
    def __init__(self):
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
        self.optimality_scale = 1.
        self.kl_min = {'state': 0, 'action': 0}

        # mode (either 'train' or 'eval')
        self._mode = 'train'

        # stores the variables during an episode
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'log_prob': []}
        # stores the objectives during training
        self.objectives = {'optimality': [], 'state': [], 'action': []}
        # stores the metrics during training
        self.metrics = {'optimality': [], 'state': [], 'action': []}
        # stores inference improvement during training
        self.inference_improvement = {'state': [], 'action': []}
        # stores planning rollout lengths during training
        self.rollout_lengths = []
        # stores the log probabilities during training
        self.log_probs = {'action': []}
        # stores the importance weights during training
        self.importance_weights = {'action': []}
        # store the values during training
        self.values = []
        # store the reward predictions during training
        self.reward_predictions = []

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = 0.
        self.max_rollout_length = 1
        self.alpha = 1.

        self.obs_reconstruction = None
        self.obs_prediction = None

    def act(self, observation, reward=None, done=False, action=None, valid=None, log_prob=None):
        observation, reward, action, done, valid, log_prob = self._change_device(observation, reward, action, done, valid, log_prob)
        self.step_state(observation=observation, reward=reward, done=done, valid=valid)
        self.state_inference(observation=observation, reward=reward, done=done, valid=valid)
        self.step_action(observation=observation, reward=reward, done=done, valid=valid, action=action)
        self.action_inference(observation=observation, reward=reward, done=done, valid=valid, action=action)
        if self._mode == 'train':
            # import ipdb; ipdb.set_trace()
            value = None
            if self.value_model is not None:
                value = self.estimate_value(observation=observation, reward=reward, done=done, valid=valid)
                self._prev_value = value
            self._collect_objectives_and_log_probs(observation, reward, done, action, value, valid, log_prob)
            self._prev_action = action
            self.valid.append(valid)
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

    def free_energy(self, observation, reward, done, valid):
        observation, reward, done = self._change_device(observation, reward, done)
        cond_log_likelihood = self.cond_log_likelihood(observation, reward, done)
        kl_divergence = self.kl_divergence()
        free_energy = kl_divergence - cond_log_likelihood
        return free_energy

    def cond_log_likelihood(self, observation, reward, done, valid):
        observation, reward, done = self._change_device(observation, reward, done)
        opt_log_likelihood = self.optimality_scale * (reward - 1.)
        cond_log_likelihood = valid * opt_log_likelihood
        if self.observation_variable is not None:
            obs_log_likelihood = (1 - done) * self.observation_variable.cond_log_likelihood(observation).sum(dim=1, keepdim=True)
            cond_log_likelihood += valid * obs_log_likelihood
        if self.reward_variable is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
            cond_log_likelihood += valid * reward_log_likelihood
        if self.done_variable is not None:
            done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
            cond_log_likelihood += valid * done_log_likelihood
        return cond_log_likelihood

    def kl_divergence(self, valid):
        state_kl_divergence = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
        action_kl_divergence = self.action_variable.kl_divergence().view(-1, 1)
        kl_divergence = valid * (state_kl_divergence + action_kl_divergence)
        return kl_divergence

    def evaluate(self):
        # evaluate the objective, averaged over the batch, backprop
        results = {}
        valid = torch.stack(self.valid)
        # n_valid_steps = valid.sum(dim=0).sub(1)
        n_valid_steps = valid.sum(dim=0)

        # average objectives over time and batch (for reporting)
        for objective_name, objective in self.metrics.items():
            obj = torch.stack(objective).sum(dim=0).div(n_valid_steps).mean(dim=0)
            if objective_name in ['observation', 'reward', 'optimality', 'done',
                                  'observation_cll', 'observation_mll', 'reward_cll',
                                  'reward_mll', 'done_cll', 'done_mll']:
                # negate for plotting in log scale
                obj = obj * -1
            results[objective_name] = obj.detach().cpu().item()

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
        free_energy = torch.zeros(n_steps, self.batch_size, 1).to(self.device)
        for objective_name, objective in self.objectives.items():
            free_energy = free_energy + torch.stack(objective)

        # calculate the REINFORCE terms
        if self.value_model is not None:
            # use a bootstrapped estimate as a baseline
            # calculate TD errors
            discount = 0.99
            values = torch.stack(self.values)
            # add up the future terms in the objective
            # can we use the reward prediction (from the reward model) as a baseline?
            # optimality_predictions = - self.optimality_scale * (torch.stack(self.reward_predictions) - 1.)
            optimality = -(-torch.stack(self.objectives['optimality']) + 1.) * valid
            # optimality = torch.stack(self.objectives['optimality']) * valid
            future_terms = optimality[1:]
            # TODO: should only include these if the action distribution is not reparameterizable
            if self.action_prior_model is not None:
                # action_kl = torch.stack(self.objectives['action']) * valid
                # future_terms = future_terms + action_kl[1:]
                pass
            if self.state_prior_model is not None:
                # state_kl = torch.clamp(torch.stack(self.objectives['state']) * valid, max=30)
                # future_terms = future_terms + state_kl[1:]
                pass
            if self.obs_likelihood_model is not None:
                # TODO: include information gain terms from observation, reward, and done
                # info_gain = torch.stack(self.objectives['info_gain']) * valid
                # future_terms = future_terms + info_gain[1:]
                pass

            deltas = future_terms + discount * values[1:] * valid[1:] - values[:-1]
            advantages = deltas.detach()
            # use generalized advantage estimator
            for i in range(advantages.shape[0]-1, 0, -1):
                advantages[i-1] = advantages[i-1] + discount * self.gae_lambda * advantages[i] * valid[i]
            # calculate value loss
            returns = advantages + values[:-1].detach()
            value_loss = 0.5 * (values[:-1] - returns).pow(2)
            free_energy[:-1] = free_energy[:-1] + value_loss
            results['value'] = value_loss.sum(dim=0).div(n_valid_steps).mean(dim=0).detach().cpu().item()
        else:
            # TODO: this is hacky and doesn't work
            # use Monte Carlo baseline
            rewards = (-torch.stack(self.objectives['optimality']) + 1.) * valid
            returns = torch.flip(torch.cumsum(torch.flip(rewards.detach(), dims=[0]), dim=0), dims=[0])
            # normalize the future sums
            returns_mean = returns[1:].sum(dim=0, keepdim=True).div(n_valid_steps)
            returns_std = (returns[1:] - returns_mean).pow(2).mul(valid[1:]).sum(dim=0, keepdim=True).div(n_valid_steps-1).pow(0.5)
            advantages = (returns - returns_mean) / (returns_std + 1e-6)
        # add the REINFORCE terms to the total objective
        log_probs = torch.stack(self.log_probs['action'])
        importance_weights = torch.stack(self.importance_weights['action']).detach()
        reinforce_terms = importance_weights[:-1] * log_probs[:-1] * advantages
        # reinforce_terms = - log_probs[:-1] * advantages
        free_energy[:-1] = free_energy[:-1] + reinforce_terms

        results['importance_weights'] = importance_weights.sum(dim=0).div(n_valid_steps).mean(dim=0).detach().cpu().item()
        results['policy_gradients'] = reinforce_terms.sum(dim=0).div(n_valid_steps-1).mean(dim=0).detach().cpu().item()
        results['advantages'] = advantages.sum(dim=0).div(n_valid_steps-1).mean(dim=0).detach().cpu().item()

        # time average, batch average, and backprop
        free_energy = free_energy.sum(dim=0).div(n_valid_steps)
        # free_energy = free_energy.mean(dim=0)
        free_energy = free_energy.mean(dim=0)
        free_energy.sum().backward()

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

        return results

    def _collect_objectives_and_log_probs(self, observation, reward, done, action, value, valid, log_prob):
        if self.done_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            # mll = self.done_variable.marginal_log_likelihood(done, log_importance_weights)
            # self.metrics['done'].append(-mll * valid)
            # self.objectives['done'].append(-mll * valid)
            done_info_gain = self.done_variable.info_gain(done, log_importance_weights, alpha=self.alpha)
            self.metrics['done'].append(-done_info_gain * valid)
            self.objectives['done'].append(-done_info_gain * valid)
            done_cll = self.done_variable.cond_log_likelihood(done).view(self.n_state_samples, -1, 1).mean(dim=0)
            self.metrics['done_cll'].append(-done_cll * valid)
            # self.objectives['done'].append(-done_log_likelihood * valid)
            done_mll = self.done_variable.marginal_log_likelihood(done, log_importance_weights)
            self.metrics['done_mll'].append(-done_mll * valid)

        if self.reward_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            # mll = self.reward_variable.marginal_log_likelihood(reward, log_importance_weights)
            # self.metrics['reward'].append(-mll * valid)
            # self.objectives['reward'].append(-mll * valid)
            reward_info_gain = self.reward_variable.info_gain(reward, log_importance_weights, alpha=self.alpha)
            self.metrics['reward'].append(-reward_info_gain * valid)
            self.objectives['reward'].append(-reward_info_gain * valid)
            reward_cll = self.reward_variable.cond_log_likelihood(reward).view(self.n_state_samples, -1, 1).mean(dim=0)
            self.metrics['reward_cll'].append(-reward_cll * valid)
            # self.objectives['reward'].append(-reward_log_likelihood * valid)
            reward_mll = self.reward_variable.marginal_log_likelihood(reward, log_importance_weights)
            self.metrics['reward_mll'].append(-reward_mll * (1 - done) * valid)

        if self.obs_likelihood_model is not None:
            log_importance_weights = self.state_variable.log_importance_weights().detach()
            # mll = self.observation_variable.marginal_log_likelihood(observation, log_importance_weights)
            # self.metrics['observation'].append(-mll * valid)
            # self.objectives['observation'].append(-mll * valid)
            observation_info_gain = self.observation_variable.info_gain(observation, log_importance_weights, alpha=self.alpha)
            self.metrics['observation'].append(-observation_info_gain * (1 - done) * valid)
            self.objectives['observation'].append(-observation_info_gain * (1 - done) * valid)
            observation_cll = self.observation_variable.cond_log_likelihood(observation).view(self.n_state_samples, -1, 1).mean(dim=0)
            # self.metrics['observation'].append(-observation_log_likelihood * (1 - done) * valid)
            self.metrics['observation_cll'].append(-observation_cll * (1 - done) * valid)
            observation_mll = self.observation_variable.marginal_log_likelihood(observation, log_importance_weights)
            self.metrics['observation_mll'].append(-observation_mll * (1 - done) * valid)

        optimality_log_likelihood = self.optimality_scale * (reward - 1.)
        self.metrics['optimality'].append(-optimality_log_likelihood * valid)
        self.objectives['optimality'].append(-optimality_log_likelihood * valid)

        state_kl = self.state_variable.kl_divergence()
        clamped_state_kl = torch.clamp(state_kl, min=self.kl_min['state']).sum(dim=1, keepdim=True)
        state_kl = state_kl.sum(dim=1, keepdim=True)
        self.metrics['state'].append(state_kl * (1 - done) * valid)
        self.objectives['state'].append(clamped_state_kl * (1 - done) * valid)

        action_kl = self.action_variable.kl_divergence().view(-1, 1)
        clamped_action_kl = torch.clamp(action_kl, min=self.kl_min['action'])
        self.metrics['action'].append(action_kl * valid)
        self.objectives['action'].append(clamped_action_kl * valid)

        action_ind = self._convert_action(action)
        action_log_prob = self.action_variable.approx_post_dist.log_prob(action_ind).view(-1, 1)
        self.log_probs['action'].append(action_log_prob * valid)

        importance_weight = torch.exp(action_log_prob) / torch.exp(log_prob)
        self.importance_weights['action'].append(importance_weight)

    def _collect_episode(self, observation, reward, done):
        if not done:
            self.episode['observation'].append(observation)
            self.episode['action'].append(self.action_variable.sample())
            self.episode['state'].append(self.state_variable.sample())
            action_ind = self._convert_action(self.action_variable.sample())
            action_log_prob = self.action_variable.approx_post_dist.log_prob(action_ind).view(-1, 1)
            self.episode['log_prob'].append(action_log_prob)
            if self.obs_reconstruction is not None:
                self.episode['reconstruction'].append(self.obs_reconstruction)
            if self.obs_prediction is not None:
                self.episode['prediction'].append(self.obs_prediction)
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            state = self.episode['state'][0]
            log_prob = self.episode['log_prob'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['state'].append(state.new(state.shape).zero_())
            self.episode['log_prob'].append(log_prob.new(log_prob.shape).zero_())
            if self.obs_reconstruction is not None:
                self.episode['reconstruction'].append(obs.new(obs.shape).zero_())
            if self.obs_prediction is not None:
                self.episode['prediction'].append(obs.new(obs.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

        self.alpha += 0.001
        self.alpha = min(self.alpha, 1.)

    def _convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        return action

    def _change_device(self, observation, reward, action, done, valid, log_prob):
        if observation is None:
            observation = torch.zeros(self.episode['observation'][0].shape)
        if observation.device != self.device:
            observation = observation.to(self.device)
        if type(reward) in [float, int]:
            reward = torch.tensor(reward).to(torch.float32).view(1, 1)
        if reward.device != self.device:
            reward = reward.to(self.device)
        if action is not None:
            if action.device != self.device:
                action = action.to(self.device)
        if type(done) == bool:
            done = torch.tensor(done).to(torch.float32).view(1, 1)
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
        Concatenate each variable in the episode. Put on the CPU.
        """
        episode = {}
        for k, v in self.episode.items():
            episode[k] = torch.cat(v, dim=0).detach().cpu()
        return episode

    def reset(self, batch_size=1):
        # reset the variables
        self.state_variable.reset(batch_size)
        self.action_variable.reset(batch_size)
        if self.observation_variable is not None:
            self.observation_variable.reset()
        if self.reward_variable is not None:
            self.reward_variable.reset()
        if self.done_variable is not None:
            self.done_variable.reset()

        # reset the networks
        if self.state_prior_model is not None:
            self.state_prior_model.reset(batch_size)
        if self.action_prior_model is not None:
            self.action_prior_model.reset(batch_size)
        if self.obs_likelihood_model is not None:
            self.obs_likelihood_model.reset()
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.reset()
        if self.done_likelihood_model is not None:
            self.done_likelihood_model.reset()

        # reset the episode, objectives, and log probs
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'log_prob': []}
        if self.obs_likelihood_model is not None:
            self.episode['reconstruction'] = []
            self.episode['prediction'] = []

        self.objectives = {'optimality': [], 'state': [], 'action': []}
        self.metrics = {'optimality': [], 'state': [], 'action': []}
        if self.observation_variable is not None:
            self.objectives['observation'] = []
            self.metrics['observation'] = []
            self.metrics['observation_cll'] = []
            self.metrics['observation_mll'] = []
        if self.reward_variable is not None:
            self.objectives['reward'] = []
            self.metrics['reward'] = []
            self.metrics['reward_cll'] = []
            self.metrics['reward_mll'] = []
        if self.done_variable is not None:
            self.objectives['done'] = []
            self.metrics['done'] = []
            self.metrics['done_cll'] = []
            self.metrics['done_mll'] = []
        # if self.value_model is not None:
        #     self.objectives['value'] = []

        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': []}
        self.rollout_lenghts = []
        self.importance_weights = {'action': []}
        self.values = []
        self.reward_predictions = []

        self._prev_action = None
        self.valid = []
        self.batch_size = batch_size
        self.state_inf_free_energies = []
        self.obs_reconstruction = None
        self.obs_prediction = None

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
