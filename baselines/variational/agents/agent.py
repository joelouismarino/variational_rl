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
                        'state': [], 'action': []}
        # stores the objectives during an episode
        self.objectives = {'optimality': [], 'state': [], 'action': []}
        # stores inference improvement
        self.inference_improvement = {'state': [], 'action': []}
        # stores the log probabilities during an episode
        self.log_probs = {'action': []}
        # store the values during training
        self.values = []

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = 0.

        self.obs_reconstruction = None
        self.obs_prediction = None

    def act(self, observation, reward=None, done=False, action=None, valid=None):
        observation, reward, action, done, valid = self._change_device(observation, reward, action, done, valid)
        self.step_state(observation=observation, reward=reward, done=done, valid=valid)
        self.state_inference(observation=observation, reward=reward, done=done, valid=valid)
        self.step_action(observation=observation, reward=reward, done=done, valid=valid, action=action)
        self.action_inference(observation=observation, reward=reward, done=done, valid=valid, action=action)
        if self._mode == 'train':
            value = None
            if self.value_model is not None:
                value = self.estimate_value(observation=observation, reward=reward, done=done, valid=valid)
                self._prev_value = value
            self._collect_objectives_and_log_probs(observation, reward, done, action, value, valid)
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

    def value_loss(self):
        pass

    def evaluate(self):
        # evaluate the objective, averaged over the batch, backprop

        results = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0).sub(1)

        # average objectives over time and batch (for reporting)
        for objective_name, objective in self.objectives.items():
            obj = torch.stack(objective).sum(dim=0).div(n_valid_steps).mean(dim=0)
            if objective_name in ['observation', 'reward', 'optimality', 'done']:
                # negate for plotting in log scale
                obj = obj * -1
            results[objective_name] = obj.detach().cpu().item()

        # evaluate inference improvement (for reporting)
        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                results[name + '_improvement'] = imp.detach().cpu().item()

        # sum the objectives (for training)
        n_steps = len(self.objectives['optimality'])
        free_energy = torch.zeros(n_steps, self.batch_size, 1).to(self.device)
        for objective_name, objective in self.objectives.items():
            free_energy = free_energy + torch.stack(objective)
        free_energy = free_energy * 0.01
        # calculate the REINFORCE terms
        rewards = (-torch.stack(self.objectives['optimality']) + 1.) * valid
        if self.value_model is not None:
            # calculate TD errors
            values = torch.stack(self.values)
            deltas = rewards[1:] + values[1:] * valid[1:] - values[:-1]
            advantages = deltas.detach()
            # use generalized advantage estimator
            for i in range(advantages.shape[0]-1, 0, -1):
                advantages[i-1] = advantages[i-1] + self.gae_lambda * advantages[i] * valid[i]
            # calculate value loss
            returns = advantages + values[:-1].detach()
            value_loss = 0.5 * (values[:-1] - returns).pow(2)
            free_energy[:-1] = free_energy[:-1] + value_loss
            results['value'] = value_loss.sum(dim=0).div(n_valid_steps).mean(dim=0).detach().cpu().item()
        else:
            # TODO: this is hacky and doesn't work
            # use Monte Carlo baseline
            returns = torch.flip(torch.cumsum(torch.flip(rewards.detach(), dims=[0]), dim=0), dims=[0])
            # normalize the future sums
            returns_mean = returns[1:].sum(dim=0, keepdim=True).div(n_valid_steps)
            returns_std = (returns[1:] - returns_mean).pow(2).mul(valid[1:]).sum(dim=0, keepdim=True).div(n_valid_steps-1).pow(0.5)
            advantages = (returns - returns_mean) / (returns_std + 1e-6)
        # add the REINFORCE terms to the total objective
        log_probs = torch.stack(self.log_probs['action'])
        reinforce_terms = - log_probs[:-1] * advantages
        free_energy[:-1] = free_energy[:-1] + reinforce_terms

        # time average, batch average, and backprop
        free_energy = free_energy.sum(dim=0).div(n_valid_steps)
        free_energy = free_energy.mean(dim=0)
        free_energy.sum().backward()

        # calculate the average gradient for each model (for reporting)
        grads_dict = {}
        grad_norm_dict = {}
        for model_name, params in self.parameters().items():
            grads = [param.grad for param in params if param.grad is not None]
            grads = torch.cat([grad.view(-1) for grad in grads], dim=0)
            grads_dict[model_name] = grads.abs().mean().cpu().numpy().item()
            grad_norm_dict[model_name] = grads.norm().cpu().numpy().item()
        results['grads'] = grads_dict
        results['grad_norms'] = grad_norm_dict

        return results

    def _collect_objectives_and_log_probs(self, observation, reward, done, action, value, valid):
        if self.done_likelihood_model is not None:
            done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
            self.objectives['done'].append(-done_log_likelihood * valid)

        if self.reward_likelihood_model is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
            self.objectives['reward'].append(-reward_log_likelihood * valid)

        if self.obs_likelihood_model is not None:
            observation_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=1, keepdim=True)
            self.objectives['observation'].append(-observation_log_likelihood * (1 - done) * valid)

        optimality_log_likelihood = self.optimality_scale * (reward - 1.)
        self.objectives['optimality'].append(-optimality_log_likelihood * valid)

        state_kl = self.state_variable.kl_divergence()
        state_kl = torch.clamp(state_kl, min=self.kl_min['state']).sum(dim=1, keepdim=True)
        self.objectives['state'].append(state_kl * (1 - done) * valid)

        action_kl = self.action_variable.kl_divergence().view(-1, 1)
        self.objectives['action'].append(action_kl * valid)

        action_ind = self._convert_action(action)
        action_log_prob = self.action_variable.approx_post_dist.log_prob(action_ind).view(-1, 1)
        self.log_probs['action'].append(action_log_prob * valid)

    def _collect_episode(self, observation, reward, done):
        if not done:
            self.episode['observation'].append(observation)
            self.episode['action'].append(self.action_variable.sample())
            self.episode['state'].append(self.state_variable.sample())
            if self.obs_reconstruction is not None:
                self.episode['reconstruction'].append(self.obs_reconstruction)
            if self.obs_prediction is not None:
                self.episode['prediction'].append(self.obs_prediction)
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            state = self.episode['state'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['state'].append(state.new(state.shape).zero_())
            if self.obs_reconstruction is not None:
                self.episode['reconstruction'].append(obs.new(obs.shape).zero_())
            if self.obs_prediction is not None:
                self.episode['prediction'].append(obs.new(obs.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

    def _convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        return action

    def _change_device(self, observation, reward, action, done, valid):
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
        return observation, reward, action, done, valid

    def get_episode(self):
        """
        Concatenate each variable in the episode. Put on the CPU.
        """
        episode = {}
        for k, v in self.episode.items():
            episode[k] = torch.cat(v, dim=0).cpu()
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
                        'state': [], 'action': []}
        if self.obs_likelihood_model is not None:
            self.episode['reconstruction'] = []
            self.episode['prediction'] = []

        self.objectives = {'optimality': [], 'state': [], 'action': []}
        if self.observation_variable is not None:
            self.objectives['observation'] = []
        if self.reward_variable is not None:
            self.objectives['reward'] = []
        if self.done_variable is not None:
            self.objectives['done'] = []
        # if self.value_model is not None:
        #     self.objectives['value'] = []

        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': []}
        self.values = []

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
