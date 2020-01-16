import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable
from lib.distributions import kl_divergence
from misc.retrace import retrace


class ModelBasedEstimator(nn.Module):
    """
    Estimate the Q-value using a learned model and Q network. Uses the Retrace
    estimator.

    Args:
        network_args (dict): arguments for the Q network
        model_args (dict): arguments for the dynamics and reward models
        horizon (int): planning horizon
        learn_reward (bool): whether to learn the reward function
    """
    def __init__(self, network_args, model_args, horizon, learn_reward=True):
        super(ModelBasedEstimator, self).__init__()
        # direct Q-value model
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(network_args)) for _ in range(2)])
        q_model_output = self.q_value_models[0].n_out
        self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])
        self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': q_model_output}) for _ in range(2)])

        # model
        self.state_likelihood_model = get_model(model_args['state_likelihood_args'])
        model_args['state_variable_args']['n_input'] = self.state_likelihood_model.n_out
        self.state_variable = get_variable(type='observed', args=model_args['state_variable_args'])

        self.reward_likelihood_model = None
        if learn_reward:
            self.reward_likelihood_model = get_model(model_args['reward_likelihood_args'])
            model_args['reward_variable_args']['n_input'] = self.reward_likelihood_model.n_out
            self.reward_variable = get_variable(type='observed', args=model_args['reward_variable_args'])
        else:
            raise NotImplementedError

        # hyper-parameters and internal attributes
        self.horizon = horizon

    def forward(self, agent, state, action, target=False, both=False,
                detach_params=False, direct=False):
        """
        Estimates the Q-value using the state and action using model and Q-networks.

        Args:
            state (torch.Tensor): the state [batch_size * n_action_samples, state_dim]
            action (torch.Tensor): the action [batch_size * n_action_samples, action_dim]
            target (bool): whether to use the target networks
            both (bool): whether to return both values (or the min value)
            detach_params (bool): whether to use detached (copied) parameters
            direct (bool): whether to get the direct (network) estimate
        """
        if direct:
            return self.direct_estimate(agent, state, action, target, both, detach_params)

        self.planning_mode(agent)
        # set the previous state for residual state prediction
        self.state_variable.cond_likelihood.set_prev_x(state)
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
            # TODO: give option for deterministic sampling?
            state = self.state_variable.sample()
            # generate the action
            agent.generate_prior(state)
            if agent.prior_model is not None:
                # sample from the learned prior
                action = agent.prior.sample()
            else:
                # estimate approximate posterior
                agent.inference(state)
                # calculate KL divergence
                action = agent.approx_post.sample(agent.n_action_samples)
                kl = kl_divergence(agent.approx_post, agent.prior, n_samples=agent.n_action_samples, sample=action).sum(dim=1, keepdim=True)
                raise NotImplementedError

        # estimate Q-value at final state
        action = action.tanh() if agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in self.q_value_models]
        q_values = [variable(inp) for variable, inp in zip(self.q_value_variables, q_value_input)]
        q_value = torch.min(q_values[0], q_values[1])
        q_values_list.append(q_value)

        # calculate the retrace Q-value estimate
        total_rewards = torch.stack(rewards_list) if len(rewards_list) > 0 else None
        total_q_values = torch.stack(q_values_list)
        retrace_estimate = retrace(total_q_values, total_rewards, None, discount=agent.reward_discount, l=agent.retrace_lambda)

        self.acting_mode(agent)

        return retrace_estimate

    def direct_estimate(self, agent, state, action, target=False, both=False,
                        detach_params=False):
        """
        Estimates the Q-value using the state and action.

        Args:
            state (torch.Tensor): the state
            action (torch.Tensor): the action
            target (bool): whether to use the target networks
            both (bool): whether to return both values (or the min value)
            detach_params (bool): whether to use detached (copied) parameters
        """
        # estimate q value
        if target:
            q_value_models = self.target_q_value_models
            q_value_variables = self.target_q_value_variables
        else:
            q_value_models = self.q_value_models
            q_value_variables = self.q_value_variables
        if detach_params:
            q_value_models = copy.deepcopy(q_value_models)
            q_value_variables = copy.deepcopy(q_value_variables)
        action = action.tanh() if agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in q_value_models]
        q_value = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
        if not both:
            q_value = torch.min(q_value[0], q_value[1])
        return q_value

    def generate(self, agent):
        """
        Generate conditional likelihoods for the current state and reward.
        """
        prev_act = agent._prev_action.tanh() if agent.postprocess_action else agent._prev_action
        self.generate_state(agent._prev_state, prev_act)
        self.generate_reward(agent._prev_state, prev_act)

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

    def planning_mode(self, agent):
        """
        Puts the distributions into planning mode.
        """
        agent.prior.planning_mode(n_samples=agent.n_action_samples)
        agent.target_prior.planning_mode(n_samples=agent.n_action_samples)
        agent.approx_post.planning_mode(n_samples=agent.n_action_samples)
        self.state_variable.planning_mode(agent.n_action_samples)
        if self.reward_likelihood_model is not None:
            self.reward_variable.planning_mode(agent.n_action_samples)

    def acting_mode(self, agent):
        """
        Puts the distributions into acting mode.
        """
        agent.prior.acting_mode()
        agent.target_prior.acting_mode()
        agent.approx_post.acting_mode()
        self.state_variable.acting_mode()
        if self.reward_likelihood_model is not None:
            self.reward_variable.acting_mode()

    def reset(self, batch_size, prev_action, prev_state):
        """
        Reset the model componenets.
        """
        self.state_variable.reset(batch_size, prev_x=prev_state)
        self.reward_variable.reset(batch_size)
        self.state_likelihood_model.reset(batch_size)
        if self.reward_likelihood_model is not None:
            self.reward_likelihood_model.reset(batch_size)

    def set_prev_state(self, prev_state):
        """
        Sets the previous state in the state variable.
        """
        self.state_variable.set_prev_x(prev_state)

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
