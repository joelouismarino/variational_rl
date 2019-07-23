import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from misc import one_hot_to_index
from misc.collector import Collector
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

        # collects relevant quantities
        self.collector = Collector(self)

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
        self.collector.collect(observation, reward, done, action, value, valid, log_prob)
        if self._mode == 'train':
            self._prev_action = action
        else:
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

    def evaluate(self):
        # evaluate the objective, collect various metrics for reporting
        objective = self.collector.evaluate()
        objective.backward()

        results = {}
        for k, v in self.collector.get_metrics().items():
            results[k] = v
        for k, v in self.collector.get_inf_imp().items():
            results[k] = v
        for k, v in self.collector.get_grads().items():
            results[k] = v

        # report the average planning rollout lengths
        rollout_lengths = self.collector.rollout_lengths
        if len(rollout_lengths) > 0:
            results['rollout_length'] = sum(rollout_lengths) / len(rollout_lengths)

        results['kl_min'] = self.kl_min
        results['kl_factor'] = self.kl_factor
        results['marginal_factor'] = self.marginal_factor

        return results

    def get_episode(self):
        return self.collector.get_episode()

    def _convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post.dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        # else:
        #     action = action.detach()
        return action

    def _change_device(self, observation, reward, action, done, valid, log_prob):
        if observation is None:
            observation = torch.zeros(self.collector.episode['observation'][0].shape)
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

        # reset the collector
        self.collector.reset()

        self._prev_action = None
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