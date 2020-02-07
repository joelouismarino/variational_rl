import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable
from lib.distributions import kl_divergence
from misc.estimators import n_step, average_n_step, exp_average_n_step, retrace_n_step


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
    def __init__(self, network_args, model_args, horizon, learn_reward=True,
                 value_estimate='retrace'):
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
        self.value_estimate = value_estimate

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

        if target:
            q_value_models = self.target_q_value_models
            q_value_variables = self.target_q_value_variables
        else:
            q_value_models = self.q_value_models
            q_value_variables = self.q_value_variables
        if detach_params:
            q_value_models = copy.deepcopy(q_value_models)
            q_value_variables = copy.deepcopy(q_value_variables)

        self.planning_mode(agent)
        # set the previous state for residual state prediction
        self.state_variable.cond_likelihood.set_prev_x(state)
        # roll out the model
        rewards_list = []
        kl_list = []
        q_values_list = []
        for _ in range(self.horizon):
            # estimate the Q-value at current state
            action = action.tanh() if agent.postprocess_action else action
            q_value_input = [model(state=state, action=action) for model in q_value_models]
            q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
            q_value = torch.min(q_values[0], q_values[1])
            q_values_list.append(q_value)
            # predict state and reward
            self.generate_state(state, action, detach_params)
            self.generate_reward(state, action, detach_params)
            reward = self.reward_variable.sample()
            rewards_list.append(reward)
            state = self.state_variable.sample()
            # generate the action
            agent.generate_prior(state, detach_params)
            if agent.prior_model is not None:
                # sample from the learned prior
                action = agent.prior.sample()
                kl_list.append(torch.zeros(action.shape[0], 1, device=action.device))
            else:
                # estimate approximate posterior
                agent.inference(state, detach_params, direct=True)
                dist = agent.direct_approx_post if agent.direct_approx_post is not None else agent.approx_post
                action = dist.sample()
                # calculate KL divergence
                kl = kl_divergence(dist, agent.prior, n_samples=1, sample=action).sum(dim=1, keepdim=True)
                kl_list.append(agent.alphas['pi'] * kl)

        # estimate Q-value at final state
        action = action.tanh() if agent.postprocess_action else action
        q_value_input = [model(state=state, action=action) for model in q_value_models]
        q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
        q_value = torch.min(q_values[0], q_values[1])
        q_values_list.append(q_value)

        # calculate the Q-value estimate
        total_rewards = torch.stack(rewards_list)
        total_kl = torch.stack(kl_list)
        total_q_values = torch.stack(q_values_list)

        if self.value_estimate == 'n_step':
            estimate = n_step(total_q_values, total_rewards, total_kl, discount=agent.reward_discount)
        elif self.value_estimate == 'average_n_step':
            estimate = average_n_step(total_q_values, total_rewards, total_kl, discount=agent.reward_discount)
        elif self.value_estimate == 'exp_average_n_step':
            estimate = exp_average_n_step(total_q_values, total_rewards, total_kl, discount=agent.reward_discount, factor=1.)
        elif self.value_estimate == 'retrace':
            estimate = retrace_n_step(total_q_values, total_rewards, total_kl, discount=agent.reward_discount, factor=agent.retrace_lambda)
        else:
            raise NotImplementedError

        self.acting_mode(agent)

        return estimate

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

    def generate(self, agent, detach_params=False):
        """
        Generate conditional likelihoods for the current state and reward.

        Args:
            agent (Agent): the agent used to generate state and reward predictions
            detach_params (bool): whether to use detached (copied) parameters
        """
        self.generate_state(agent._prev_state, agent._prev_action, detach_params)
        self.generate_reward(agent._prev_state, agent._prev_action, detach_params)

    def generate_reward(self, state, action, detach_params=False):
        """
        Generate the conditional likelihood for the reward.

        Args:
            state (torch.Tensor): the input state [batch_size, state_dim]
            action (torch.Tensor): the input action [batch_size, action_dim]
            detach_params (bool): whether to use detached (copied) parameters
        """
        if detach_params:
            reward_likelihood_model = copy.deepcopy(self.reward_likelihood_model)
        else:
            reward_likelihood_model = self.reward_likelihood_model
        likelihood_input = reward_likelihood_model(state=state, action=action)
        self.reward_variable.generate(likelihood_input, action=action, detach_params=detach_params)

    def generate_state(self, state, action, detach_params=False):
        """
        Generate the conditional likelihood for the state.

        Args:
            state (torch.Tensor): the input state [batch_size, state_dim]
            action (torch.Tensor): the input action [batch_size, action_dim]
            detach_params (bool): whether to use detached (copied) parameters
        """
        if detach_params:
            state_likelihood_model = copy.deepcopy(self.state_likelihood_model)
        else:
            state_likelihood_model = self.state_likelihood_model
        likelihood_input = state_likelihood_model(state=state, action=action)
        self.state_variable.generate(likelihood_input, detach_params=detach_params)

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
        if agent.direct_approx_post is not None:
            agent.direct_approx_post.planning_mode(n_samples=agent.n_action_samples)

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
        if agent.direct_approx_post is not None:
            agent.direct_approx_post.acting_mode()

    def reset(self, batch_size, prev_state):
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
