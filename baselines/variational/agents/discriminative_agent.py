import torch
import torch.nn as nn
from .agent import Agent
from ..modules.models import get_model
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index


class DiscriminativeAgent(Agent):
    """
    Variational RL Agent with Discriminative State Estimation
    """
    def __init__(self, state_variable_args, action_variable_args,
                 state_prior_args, action_prior_args, state_inference_args,
                 action_inference_args, value_args, misc_args):
        super(DiscriminativeAgent, self).__init__()

        # models
        self.state_prior_model = get_model('discriminative', 'state', 'prior', state_prior_args)
        self.action_prior_model = get_model('discriminative', 'action', 'prior', action_prior_args)
        self.state_inference_model = get_model('discriminative', 'state', 'inference', state_inference_args)
        self.action_inference_model = get_model('discriminative', 'action', 'inference', action_inference_args)
        self.value_model = get_model('value', value_args)

        # variables
        state_variable_args['n_input'] = [None, None]
        if self.state_prior_model is not None:
            state_variable_args['n_input'][0] = self.state_prior_model.n_out
        if self.state_inference_model is not None:
            state_variable_args['n_input'][1] = self.state_inference_model.n_out
        self.state_variable = get_variable(latent=True, args=state_variable_args)

        action_variable_args['n_input'] = [None, None]
        if self.action_prior_model is not None:
            action_variable_args['n_input'][0] = self.action_prior_model.n_out
        if self.action_inference_model is not None:
           action_variable_args['n_input'][1] = self.action_inference_model.n_out
        self.action_variable = get_variable(latent=True, args=action_variable_args)

        # miscellaneous
        self.optimality_scale = misc_args['optimality_scale']
        self.kl_min = {'state': misc_args['kl_min']['state'],
                       'action': misc_args['kl_min']['action']}

        # mode (either 'train' or 'eval')
        self._mode = 'train'

        # stores the variables during an episode
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': []}
        # stores the objectives during an episode
        self.objectives = {'optimality': [], 'state': [], 'action': [], 'value': []}
        # stores inference improvement
        self.inference_improvement = {'state': [], 'action': []}
        # stores the log probabilities during an episode
        self.log_probs = {'action': []}
        # store the temporal difference errors during training
        self.td_errors = []

        self.valid = []
        self._prev_action = None
        self._prev_value = None
        self.batch_size = 1

        self.state_variable.inference_mode()
        self.action_variable.inference_mode()

    def state_inference(self, observation, reward, **kwargs):
        observation = observation - 0.5
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
            inf_input = self.state_inference_model(observation, reward, state, action)
            self.state_variable.infer(inf_input)

    def action_inference(self, observation, reward, action=None, **kwargs):
        observation = observation - 0.5
        # infer the approx. posterior on the action
        if self.action_inference_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            inf_input = self.action_inference_model(observation, reward, state, action)
            self.action_variable.infer(inf_input)

    def step_state(self, observation, reward, **kwargs):
        observation = observation - 0.5
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
            prior_input = self.state_prior_model(observation, reward, state, action)
            self.state_variable.step(prior_input)

    def step_action(self, observation, reward, **kwargs):
        observation = observation - 0.5
        # calculate the prior on the action variable
        if self.action_prior_model is not None:
            state = self.state_variable.sample()
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            prior_input = self.action_prior_model(observation, reward, state, action)
            self.action_variable.step(prior_input)

    def estimate_value(self, reward, done, **kwargs):
        # estimate the value of the current state
        state = self.state_variable.sample()
        value = self.value_model(state)
        if self._prev_value is not None:
            td_error = value * (1 - done) + self.optimality_scale * (reward - 1.) - self._prev_value
            self.td_errors.append(td_error)
        else:
            self.td_errors.append(value.new_zeros(value.shape))
        return value
