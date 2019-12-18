import copy
import torch.nn as nn
from lib.models import get_model
from modules.variables import get_variable
from misc.retrace import retrace


class ModelBasedEstimator(nn.Module):
    """
    Estimate the Q-value using a learned model. Uses the Retrace estimator.

    Args:
        agent (Agent): the parent agent
        network_args (dict): arguments for the direct Q-value network
        model_args (dict): arguments for the dynamics and reward models
        horizon (int): planning horizon
        retrace_lambda (float): smoothing factor for Retrace estimator
    """
    def __init__(self, agent, network_args, model_args, horizon, retrace_lambda):
        self.agent = agent
        # direct Q-value model
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])

        # model
        self.reward_likelihood_model = get_model(model_args['reward_likelihood_args'])
        self.state_likelihood_model = get_model(model_args['state_likelihood_args'])
        observation_variable_args['n_input'] = self.state_likelihood_model.n_out
        self.state_variable = get_variable(type='observed', args=model_args['state_variable_args'])
        reward_variable_args['n_input'] = self.reward_likelihood_model.n_out
        self.reward_variable = get_variable(type='observed', args=model_args['reward_variable_args'])

        # hyper-parameters
        self.horizon = horizon
        self.retrace_lambda = retrace_lambda

    def forward(self, state, action):

        # roll out the model
        rewards_list = []
        q_values_list = []
        for _ in range(self.horizon):
            # estimate the Q-value at current state
            act = act.tanh() if self.agent.postprocess_action else act
            q_value_input = [model(state=state, action=act) for model in q_value_models]
            q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
            q_value = torch.min(q_values[0], q_values[1])
            q_values_list.append(q_value)
            # predict state and reward
            self.generate_observation(obs, act)
            self.generate_reward(obs, act)
            reward = self.reward_variable.sample()
            rewards_list.append(reward)
            # step the action
            state = self.state_variable.sample()
            self.step_action(state)
            act = self.action_variable.sample()

        # estimate Q-value at final state
        act = act.tanh() if self.agent.postprocess_action else act
        q_value_input = [model(state=state, action=act) for model in q_value_models]
        q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
        q_value = torch.min(q_values[0], q_values[1])
        q_values_list.append(q_value)

        # add retrace Q-value estimate to the objective
        total_rewards = torch.stack(rewards_list) if len(rewards_list) > 0 else None
        total_q_values = torch.stack(q_values_list)
        retrace_estimate = retrace(total_q_values, total_rewards, None, discount=self.agent.reward_discount, l=self.retrace_lambda)
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
        self.state_variable.reset(batch_size, prev_state=prev_state)
        self.reward_variable.reset(batch_size)
        self.state_likelihood_model.reset(batch_size)
        self.reward_likelihood_model.reset(batch_size)
