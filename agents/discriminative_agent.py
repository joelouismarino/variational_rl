import torch
import torch.nn as nn
from .agent import Agent
from modules.models import get_model
from modules.variables import get_variable
from misc import clear_gradients, one_hot_to_index
from util.normalization_util import Normalizer


class DiscriminativeAgent(Agent):
    """
    Variational RL Agent with Discriminative State Estimation
    """
    def __init__(self, state_variable_args, action_variable_args,
                 state_prior_args, action_prior_args, state_inference_args,
                 action_inference_args, value_model_args, misc_args):
        super(DiscriminativeAgent, self).__init__()

        # models
        self.state_prior_model = get_model(state_prior_args)
        self.action_prior_model = get_model(action_prior_args)
        self.state_inference_model = get_model(state_inference_args)
        self.action_inference_model = get_model(action_inference_args)
        self.value_model = get_model(value_model_args)

        # variables
        state_variable_args['n_input'] = [None, None]
        if self.state_prior_model is not None:
            state_variable_args['n_input'][0] = self.state_prior_model.n_out
        if self.state_inference_model is not None:
            state_variable_args['n_input'][1] = self.state_inference_model.n_out
        self.state_variable = get_variable(type='latent', args=state_variable_args)

        action_variable_args['n_input'] = [None, None]
        if self.action_prior_model is not None:
            action_variable_args['n_input'][0] = self.action_prior_model.n_out
        if self.action_inference_model is not None:
           action_variable_args['n_input'][1] = self.action_inference_model.n_out
        self.action_variable = get_variable(type='latent', args=action_variable_args)

        if self.value_model is not None:
            self.value_variable = get_variable(type='value', args={'n_input': self.value_model.n_out})

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

        # stores the variables during an episode
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': []}
        # stores the objectives during an episode
        self.objectives = {'optimality': [], 'state': [], 'action': []}
        # stores inference improvement
        self.inference_improvement = {'state': [], 'action': []}
        # stores the log probabilities during an episode
        self.log_probs = {'action': [], 'state': []}
        # store the values during training
        self.values = []

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = misc_args['gae_lambda']
        self.reward_discount = misc_args['reward_discount']

        if misc_args['normalize_returns']:
            self.return_normalizer = Normalizer(shift=False, clip_value=10.)
        if misc_args['normalize_advantages']:
            self.advantage_normalizer = Normalizer(clip_value=10.)
        if misc_args['normalize_observations']:
            if state_prior_args:
                observation_size = state_prior_args['n_input']
            else:
                observation_size = state_inference_args['n_input']
            # TODO: should set this in a better way, in case of image input
            self.obs_normalizer = Normalizer(shape=(observation_size), clip_value=10.)

        self.state_variable.inference_mode()
        self.action_variable.inference_mode()

    def state_inference(self, observation, reward, **kwargs):
        # observation = observation - 0.5
        # infer the approx. posterior on the state
        if self.state_inference_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.state_variable.reinitialized:
                # use zeros as initial state and action inputs
                state = state.new_zeros(state.shape)
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            if self.obs_normalizer:
                update = self._mode=='eval' and self.state_prior_model is None
                observation = self.obs_normalizer(observation, update=update)
            inf_input = self.state_inference_model(observation=observation, reward=reward, state=state, action=action)
            self.state_variable.infer(inf_input)

    def action_inference(self, observation, reward, action=None, **kwargs):
        # observation = observation - 0.5
        # infer the approx. posterior on the action
        if self.action_inference_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            inf_input = self.action_inference_model(observation=observation, reward=reward, state=state, action=action)
            self.action_variable.infer(inf_input)

    def step_state(self, observation, reward, **kwargs):
        # observation = observation - 0.5
        # calculate the prior on the state variable
        if self.state_prior_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.state_variable.reinitialized:
                # use zeros as initial state and action inputs
                state = state.new_zeros(state.shape)
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation, update=self._mode=='eval')
            prior_input = self.state_prior_model(observation=observation, reward=reward, state=state, action=action)
            self.state_variable.step(prior_input)

    def step_action(self, observation, reward, **kwargs):
        # observation = observation - 0.5
        # calculate the prior on the action variable
        if self.action_prior_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            prior_input = self.action_prior_model(observation=observation, reward=reward, state=state, action=action)
            self.action_variable.step(prior_input)

    def estimate_value(self, done, **kwargs):
        # estimate the value of the current state
        state = self.state_variable.sample()
        value = self.value_variable(self.value_model(state=state)) * (1 - done)
        self.values.append(value)
        return value
