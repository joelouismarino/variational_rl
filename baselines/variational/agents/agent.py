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

        # networks
        self.state_prior_model = None
        self.action_prior_model = None
        self.state_inference_model = None
        self.action_inference_model = None

        # variables
        self.state_variable = None
        self.action_variable = None

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

        self.valid = []
        self._prev_action = None
        self.batch_size = 1

    def act(self, observation, reward=None, done=False, action=None, valid=None):
        observation, reward, action, done, valid = self._change_device(observation, reward, action, done, valid)
        self.step_state()
        self.state_inference(observation, reward, done, valid)
        self.step_action()
        self.action_inference()
        if self._mode == 'train':
            self._collect_objectives_and_log_probs(observation, reward, done, action, valid)
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
    def action_inference(self, action=None):
        pass

    @abstractmethod
    def step_state(self):
        pass

    @abstractmethod
    def step_action(self):
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
        return valid * opt_log_likelihood

    def kl_divergence(self, valid):
        state_kl_divergence = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
        action_kl_divergence = self.action_variable.kl_divergence().view(-1, 1)
        kl_divergence = valid * (state_kl_divergence + action_kl_divergence)
        return kl_divergence

    def evaluate(self):
        # evaluate the objective, averaged over the batch, backprop

        results = {}
        valid = torch.stack(self.valid)
        n_valid_steps = valid.sum(dim=0).sub(1)
        # average objectives over time and batch for reporting
        for objective_name, objective in self.objectives.items():
            obj = torch.stack(objective).sum(dim=0).div(n_valid_steps).mean(dim=0)
            if objective_name in ['observation', 'reward', 'optimality', 'done']:
                obj = obj * -1
            results[objective_name] = obj.detach().cpu().item()

        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                results[name + '_improvement'] = imp.detach().cpu().item()

        # sum the objectives
        n_steps = len(self.objectives['observation'])
        free_energy = torch.zeros(n_steps, self.batch_size, 1).to(self.device)
        for objective_name, objective in self.objectives.items():
            free_energy = free_energy + torch.stack(objective)

        # calculate the reinforce terms
        optimality = torch.stack(self.objectives['optimality'])
        future_sums = torch.flip(torch.cumsum(torch.flip(optimality.detach(), dims=[0]), dim=0), dims=[0])
        # normalize the future sums
        future_sums_mean = future_sums[1:].sum(dim=0, keepdim=True).div(n_valid_steps)
        future_sums_std = (future_sums[1:] - future_sums_mean).pow(2).mul(valid[1:]).sum(dim=0, keepdim=True).div(n_valid_steps-1).pow(0.5)
        future_sums = (future_sums - future_sums_mean) / (future_sums_std + 1e-6)
        log_probs = torch.stack(self.log_probs['action'])
        reinforce_terms = - log_probs * future_sums
        free_energy = free_energy + reinforce_terms

        # time average
        free_energy = free_energy.sum(dim=0).div(n_valid_steps)

        # batch average
        free_energy = free_energy.mean(dim=0)

        # backprop
        free_energy.sum().backward()

        # calculate the average gradient for each model
        grads_dict = {}
        grad_norm_dict = {}
        for model_name, params in self.parameters().items():
            grads = torch.cat([param.grad.view(-1) for param in params], dim=0)
            grads_dict[model_name] = grads.abs().mean().cpu().numpy().item()
            grad_norm_dict[model_name] = grads.norm().cpu().numpy().item()
        results['grads'] = grads_dict
        results['grad_norms'] = grad_norm_dict

        return results


    def _collect_objectives_and_log_probs(self, observation, reward, done, action, valid):

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
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            state = self.episode['state'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['state'].append(state.new(state.shape).zero_())
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

        # reset the networks
        self.state_prior_model.reset(batch_size)
        self.action_prior_model.reset(batch_size)

        # reset the episode, objectives, and log probs
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': []}
        self.objectives = {'optimality': [], 'state': [], 'action': []}
        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': []}

        self._prev_action = None
        self.valid = []
        self.batch_size = batch_size
        self.state_inf_free_energies = []

    @property
    def device(self):
        return self.inference_parameters()[0].device

    def train(self, *args):
        super(Model, self).train(*args)
        self._mode = 'train'

    def eval(self, *args):
        super(Model, self).eval(*args)
        self._mode = 'eval'

    def parameters(self):
        param_dict = {}

        param_dict['state_inference_model'] = nn.ParameterList()
        param_dict['state_inference_model'].extend(list(self.state_inference_model.parameters()))
        param_dict['state_inference_model'].extend(list(self.state_variable.inference_parameters()))

        param_dict['action_inference_model'] = nn.ParameterList()
        param_dict['action_inference_model'].extend(list(self.action_inference_model.parameters()))
        param_dict['action_inference_model'].extend(list(self.action_variable.inference_parameters()))

        param_dict['state_prior_model'] = nn.ParameterList()
        param_dict['state_prior_model'].extend(list(self.state_prior_model.parameters()))
        param_dict['state_prior_model'].extend(list(self.state_variable.generative_parameters()))

        param_dict['action_prior_model'] = nn.ParameterList()
        param_dict['action_prior_model'].extend(list(self.action_prior_model.parameters()))
        param_dict['action_prior_model'].extend(list(self.action_variable.generative_parameters()))

        return param_dict

    def inference_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_inference_model.parameters()))
        params.extend(list(self.action_inference_model.parameters()))
        params.extend(list(self.state_variable.inference_parameters()))
        params.extend(list(self.action_variable.inference_parameters()))
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_prior_model.parameters()))
        params.extend(list(self.action_prior_model.parameters()))
        params.extend(list(self.state_variable.generative_parameters()))
        params.extend(list(self.action_variable.generative_parameters()))
        return params

    def inference_mode(self):
        self.state_variable.inference_mode()
        self.action_variable.inference_mode()
        self.state_prior_model.detach_hidden_state()
        self.action_prior_model.detach_hidden_state()

    def generative_mode(self):
        self.state_variable.generative_mode()
        self.action_variable.generative_mode()
        self.state_prior_model.attach_hidden_state()
        self.action_prior_model.attach_hidden_state()
