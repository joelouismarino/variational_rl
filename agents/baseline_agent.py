import torch
import torch.nn as nn
from .agent import Agent
from modules.models import get_model
from modules.variables import get_variable
import copy


class BaselineAgent(Agent):
    """
    Baseline Variational RL Agent
    """
    def __init__(self, action_variable_args, action_prior_args,
                 action_inference_args, q_value_model_args, misc_args):
        super(BaselineAgent, self).__init__(misc_args)

        self.type = 'baseline'

        # models
        self.action_prior_model = get_model(copy.deepcopy(action_prior_args))
        self.action_inference_model = get_model(copy.deepcopy(action_inference_args))
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(q_value_model_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(q_value_model_args)) for _ in range(2)])
        self.target_action_prior_model = get_model(copy.deepcopy(action_prior_args))
        self.target_action_inference_model = get_model(copy.deepcopy(action_inference_args))

        # variables
        action_variable_args['n_input'] = [None, None]
        if self.action_prior_model is not None:
            action_variable_args['n_input'][0] = self.action_prior_model.n_out
        if self.action_inference_model is not None:
           action_variable_args['n_input'][1] = self.action_inference_model.n_out
        self.action_variable = get_variable(type='latent', args=copy.deepcopy(action_variable_args))
        self.target_action_variable = get_variable(type='latent', args=copy.deepcopy(action_variable_args))

        if self.q_value_models is not None:
            self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': self.q_value_models[0].n_out}) for _ in range(2)])
            self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': self.q_value_models[0].n_out}) for _ in range(2)])

    def state_inference(self, **kwargs):
        pass

    def action_inference(self, observation, reward, action=None,**kwargs):
        """
        Infer the approximate posterior on the action.
        """
        self.action_variable.inference_mode()
        self.action_variable.init_approx_post()
        if self.action_inference_model is not None:
            action = self._prev_action
            # if action is None:
            #     action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation, update=self._mode=='eval')
            inf_input = self.action_inference_model(observation=observation, reward=reward, action=action)
            self.action_variable.infer(inf_input)
            # get the target network output
            inf_input = self.target_action_inference_model(observation=observation, reward=reward, action=action)
            self.target_action_variable.infer(inf_input)

    def step_state(self, **kwargs):
        pass

    def step_action(self, observation, reward, **kwargs):
        """
        Generate the prior on the action.
        """
        self.action_variable.generative_mode()
        if self.action_prior_model is not None:
            if self._prev_action is not None:
                action = self._prev_action
            else:
                action = self.action_variable.sample()
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation, update=self._mode=='eval')
            prior_input = self.action_prior_model(observation=observation, reward=reward, action=action)
            self.action_variable.step(prior_input)
            # get the target network output
            prior_input = self.target_action_prior_model(observation=observation, reward=reward, action=action)
            self.target_action_variable.step(prior_input)
