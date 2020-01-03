import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable
from misc.retrace import retrace


class ModelBasedEstimator(nn.Module):
    """
    Estimate the Q-value using a learned model and Q network. Uses the Retrace
    estimator.

    Args:
        agent (Agent): the parent agent
        network_args (dict): arguments for the Q network
        model_args (dict): arguments for the dynamics and reward models
        horizon (int): planning horizon
        retrace_lambda (float): smoothing factor for Retrace estimator
        learn_reward (bool): whether to learn the reward function
    """
    def __init__(self, agent, network_args, model_args, horizon, learn_reward=True):
        super(ModelBasedEstimator, self).__init__()
        # self.agent = agent
        # direct Q-value model
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        q_model_output = self.q_value_models[0].n_out
        self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])
        self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])
        self.q_values = None

        # model
        self.state_likelihood_model = get_model(model_args['state_likelihood_args'])
        model_args['state_likelihood_args']['n_input'] = self.state_likelihood_model.n_out
        self.state_variable = get_variable(type='observed', args=model_args['state_variable_args'])

        self.reward_likelihood_model = None
        if learn_reward:
            self.reward_likelihood_model = get_model(model_args['reward_likelihood_args'])
            model_args['reward_likelihood_args']['n_input'] = self.reward_likelihood_model.n_out
            self.reward_variable = get_variable(type='observed', args=model_args['reward_variable_args'])
        else:
            raise NotImplementedError

        # hyper-parameters
        self.horizon = horizon
        self.retrace_lambda = agent.retrace_lambda

        self._prev_state = None

        # remove agent to prevent infinite recursion
        # del self.__dict__['_modules']['agent']

    def forward(self, agent, state, action, target=False):

        self._prev_state = state

        # roll out the model
        rewards_list = []
        q_values_list = []
        for _ in range(self.horizon):
            # estimate the Q-value at current state
            action = action.tanh() if agent.postprocess_action else action
            q_value_input = [model(state=state, action=action) for model in self.q_value_models]
            q_values = [variable(inp) for variable, inp in zip(self.q_value_variables, q_value_input)]
            q_value = torch.min(q_values[0], q_values[1])
            q_values_list.append(q_value)
            # predict state and reward
            self.generate_state(state, action)
            self.generate_reward(state, action)
            reward = self.reward_variable.sample()
            rewards_list.append(reward)
            # step the action
            state = self.state_variable.sample()
            agent.generate_prior(state)
            # TODO: give option of which distribution to sample from
            act = self.agent.prior.sample()

        # estimate Q-value at final state
        action = action.tanh() if agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in self.q_value_models]
        q_values = [variable(inp) for variable, inp in zip(self.q_value_variables, q_value_input)]
        q_value = torch.min(q_values[0], q_values[1])
        q_values_list.append(q_value)

        # add retrace Q-value estimate to the objective
        total_rewards = torch.stack(rewards_list) if len(rewards_list) > 0 else None
        total_q_values = torch.stack(q_values_list)
        retrace_estimate = retrace(total_q_values, total_rewards, None, discount=agent.reward_discount, l=self.retrace_lambda)
        retrace_estimate = retrace_estimate.view(-1, self.n_planning_samples, 1)

        return retrace_estimate

    def generate_reward(self, state, action):
        """
        Generate the conditional likelihood for the reward.
        """
        likelihood_input = self.reward_likelihood_model(state=state, action=action)
        self.reward_variable.generate(likelihood_input, action=action)

    def generate_state(self, state, action):
        """
        Generate the conditional likelihood for the state.
        """
        likelihood_input = self.state_likelihood_model(state=state, action=action)
        self.state_variable.generate(likelihood_input)

    def reset(self, batch_size, prev_action, prev_state):
        self.q_values = None
        self.state_variable.reset(batch_size, prev_state=prev_state)
        self.reward_variable.reset(batch_size)
        self.state_likelihood_model.reset(batch_size)
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.reset(batch_size)

    def parameters(self):
        param_dict = {}
        param_dict['q_value_models'] = nn.ParameterList()
        param_dict['q_value_models'].extend(list(self.q_value_models.parameters()))
        param_dict['q_value_models'].extend(list(self.q_value_variables.parameters()))
        param_dict['target_q_value_models'] = nn.ParameterList()
        param_dict['target_q_value_models'].extend(list(self.target_q_value_models.parameters()))
        param_dict['target_q_value_models'].extend(list(self.target_q_value_variables.parameters()))
        param_dict['state_likelihood_model'] = nn.ParameterList()
        param_dict['state_likelihood_model'].extend(list(self.state_likelihood_model.parameters()))
        param_dict['state_likelihood_model'].extend(list(self.state_variable.parameters()))
        if self.reward_likelihood_model is not None:
            param_dict['reward_likelihood_model'] = nn.ParameterList()
            param_dict['reward_likelihood_model'].extend(list(self.reward_likelihood_model.parameters()))
            param_dict['reward_likelihood_model'].extend(list(self.reward_variable.parameters()))
        return param_dict
