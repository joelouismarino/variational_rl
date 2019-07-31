import torch
import torch.nn as nn
from .agent import Agent
from modules.models import get_model
from modules.variables import get_variable
import copy
from misc.normalization import Normalizer


class BaselineAgent(Agent):
    """
    Baseline Variational RL Agent
    """
    def __init__(self, action_variable_args, action_prior_args,
                 action_inference_args, value_model_args, q_value_model_args, misc_args):
        super(BaselineAgent, self).__init__(misc_args)

        # models
        self.action_prior_model = get_model(action_prior_args)
        self.action_inference_model = get_model(action_inference_args)
        self.value_model = get_model(value_model_args)
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(q_value_model_args)) for _ in range(2)])

        # variables
        action_variable_args['n_input'] = [None, None]
        if self.action_prior_model is not None:
            action_variable_args['n_input'][0] = self.action_prior_model.n_out
        if self.action_inference_model is not None:
           action_variable_args['n_input'][1] = self.action_inference_model.n_out
        self.action_variable = get_variable(type='latent', args=action_variable_args)

        if self.value_model is not None:
            self.value_variable = get_variable(type='value', args={'n_input': self.value_model.n_out})
            self.qvalue1_variable = get_variable(type='value', args={'n_input': self.q_value_models[0].n_out})
            self.qvalue2_variable = get_variable(type='value', args={'n_input': self.q_value_models[1].n_out})

    def state_inference(self, **kwargs):
        pass

    def action_inference(self, observation, reward, action=None,**kwargs):
        self.action_variable.inference_mode()
        # infer the approx. posterior on the action
        if self.action_inference_model is not None:
            action = self._prev_action
            if action is None:
                action = self.action_variable.sample()
            if self.action_variable.reinitialized:
                action = self.action_variable.sample()
                action = action.new_zeros(action.shape)
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation, update=self._mode=='eval')
            inf_input = self.action_inference_model(observation=observation, reward=reward, action=action)
            self.action_variable.infer(inf_input)

    def step_state(self, **kwargs):
        pass

    def step_action(self, observation, reward, **kwargs):
        self.action_variable.generative_mode()
        # calculate the prior on the action variable
        if self.action_prior_model is not None:
            if not self.action_variable.reinitialized:
                if self._prev_action is not None:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation, update=self._mode=='eval')
                prior_input = self.action_prior_model(observation=observation, reward=reward, action=action)
                self.action_variable.step(prior_input)

    def estimate_value(self, done, observation, reward, **kwargs):
        # estimate the value of the current state
        value_input = self.value_model(observation=observation, reward=reward)
        value = self.value_variable(value_input) * (1 - done)
        self.collector.values.append(value)
        return value

    def estimate_q_values(self, done, observation, reward, action, **kwargs):
        # estimate the value of the current state
        # TODO: SAC potential bug
        state = None #self.state_variable.sample()
        q_value_input = [self.q_value_models[i](state=state, observation=observation, action=action, reward=reward) for i in range(2)]
        qvalue1 = self.qvalue1_variable(q_value_input[0])
        qvalue2 = self.qvalue2_variable(q_value_input[1])
        self.collector.qvalues1.append(qvalue1)
        self.collector.qvalues2.append(qvalue2)
        self.collector.qvalues.append(torch.min(qvalue1, qvalue2))
        #print(self._mode)
        #print(self.action_variable.approx_post._detach)
        new_action = self.action_variable.approx_post.sample(1)
        new_action_log_prob = self.action_variable.approx_post.dist.log_prob(new_action)
        #new_action = self.action_variable.prior.sample(1)
        #new_action_log_prob = self.action_variable.prior.dist.log_prob(new_action)
        self.collector.new_actions.append(new_action)
        self.collector.new_action_log_probs.append(new_action_log_prob)
        new_q_value_models = copy.deepcopy(self.q_value_models)
        new_q_value_input = [new_q_value_models[i](state=state, observation=observation, action=new_action, reward=reward) for i in range(2)]
        new_qvalue1_variable = copy.deepcopy(self.qvalue1_variable)
        new_qvalue2_variable = copy.deepcopy(self.qvalue2_variable)
        new_qvalue1 = new_qvalue1_variable(new_q_value_input[0])
        new_qvalue2 = new_qvalue2_variable(new_q_value_input[1])
        new_q_value = torch.min(new_qvalue1, new_qvalue2)
        self.collector.new_q_values.append(new_q_value)
        return torch.min(qvalue1, qvalue2)
