import copy
import torch
import torch.nn as nn
from lib.models import get_model
from lib.variables import get_variable
from torch.distributions import Normal


class GoalBasedQEstimator(nn.Module):
    """
    Estimate the goal-based objective using a learned model and 1-step horizon.

    Args:
        model_args (dict, optional): arguments for the dynamics model
    """
    def __init__(self, model_args=None):
        super(GoalBasedQEstimator, self).__init__()
        # model
        if model_args:
            self.state_likelihood_model = get_model(model_args['state_likelihood_args'])
            model_args['state_variable_args']['n_input'] = self.state_likelihood_model.n_out
            self.state_variable = get_variable(type='observed', args=model_args['state_variable_args'])
        else:
            self.state_likelihood_model = None
            self.state_variable = None
        self.reward_likelihood_model = None
        self.reward_variable = None

        # hyper-parameters and internal attributes
        self.goal_state = None
        self.goal_std = 1.
        self.horizon = 1

        self.errors = {}

        # save the results of MB planning
        # self.rollout_states = []
        # self.rollout_q_values = []
        # self.rollout_rewards = []
        # self.rollout_actions = []

    def forward(self, agent, state, action, detach_params=False, *args, **kwargs):
        """
        Estimates the Q-value using the state and action using model and Q-networks.

        Args:
            state (torch.Tensor): the state [batch_size * n_action_samples, state_dim]
            action (torch.Tensor): the action [batch_size * n_action_samples, action_dim]
            detach_params (bool): whether to use detached (copied) parameters

        Returns a Q-value estimate of shape [n_action_samples * batch_size, 1]
        """
        self.planning_mode(agent)
        # set the previous state for residual state prediction
        self.state_variable.cond_likelihood.set_prev_x(state)
        prev_state = state
        # roll out the model
        actions_list = [action]
        states_list = [state]
        rewards_list = []

        # predict the next state
        action = action.tanh() if agent.postprocess_action else action
        self.generate_state(state, action, detach_params)
        state = self.state_variable.sample()
        states_list.append(state)

        # evaluate the goal-based reward
        goal_scale = torch.ones(self.goal_state.shape)
        goal_scale[:, :8] *= self.goal_std
        goal_loc = self.goal_state.repeat(agent.n_action_samples, 1)
        goal_scale = goal_scale.repeat(agent.n_action_samples, 1)
        reward_dist = Normal(loc=goal_loc, scale=goal_scale)
        # reward_dist = reward_dist.expand(torch.Size([10]) + reward_dist.batch_shape)
        goal_reward = reward_dist.log_prob(state).sum(dim=1, keepdim=True)
        self.errors['goal'] = ((1./goal_scale) * (goal_loc - state)) ** 2
        print('     GOAL REWARD: ' + str(goal_reward.mean().item()))

        prev_state_loc = prev_state.clone().detach()
        prev_state_loc[:, 8:] *= 0
        prev_state_scale = torch.ones(prev_state_loc.shape)
        prev_state_scale[:, :8] *= (self.goal_std * 10)
        damping_reward_dist = Normal(loc=prev_state_loc, scale=prev_state_scale)
        damping_reward = damping_reward_dist.log_prob(state).sum(dim=1, keepdim=True)
        self.errors['prev_state'] = ((1./prev_state_scale) * (prev_state_loc - state)) ** 2
        print('     DAMPING REWARD: ' + str(damping_reward.mean().item()))
        # damping_reward = 0.

        action_scale = 0.1
        action_reward_dist = Normal(loc=torch.zeros(action.shape), scale=action_scale*torch.ones(action.shape))
        action_reward = action_reward_dist.log_prob(action).sum(dim=1, keepdim=True)
        self.errors['action'] = ((1./action_scale) * action) ** 2
        print('     ACTION REWARD: ' + str(action_reward.mean().item()))
        # action_reward = 0.

        reward = goal_reward + damping_reward + action_reward

        # Note: only evaluated on the position (not velocity)
        # reward_dist = Normal(loc=self.goal_state, scale=self.goal_std*torch.ones(self.goal_state.shape))
        # reward = reward_dist.log_prob(state[:, :8]).sum(dim=1, keepdim=True)
        rewards_list.append(reward)
        # print(reward)

        self.acting_mode(agent)

        # self.rollout_states.append(states_list)
        # self.rollout_rewards.append(rewards_list)
        # self.rollout_q_values.append(q_values_list)
        # self.rollout_actions.append(actions_list)

        return reward

    def get_errors(self):
        return self.errors

    def set_goal_state(self, state):
        """
        Set the goal state.

        Args:
            state (torch.Tensor): the goal state
        """
        # set zero velocity
        # state[:, 8:] *= 0.
        self.goal_state = state
        # self.goal_state = state[:, :8]

    def set_goal_std(self, std):
        """
        Set the goal state weight (std dev).

        Args:
            std (float): the goal state weight
        """
        self.goal_std = std

    def generate(self, agent, detach_params=False):
        """
        Generate conditional likelihoods for the current state and reward.

        Args:
            agent (Agent): the agent used to generate state and reward predictions
            detach_params (bool): whether to use detached (copied) parameters
        """
        self.generate_state(agent._prev_state, agent._prev_action, detach_params)

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
        if agent.direct_approx_post is not None:
            agent.direct_approx_post.acting_mode()

    def reset(self, batch_size, prev_state):
        """
        Reset the model componenets.
        """
        self.state_variable.reset(batch_size, prev_x=prev_state)
        self.state_likelihood_model.reset(batch_size)

    def set_prev_state(self, prev_state):
        """
        Sets the previous state in the state variable.
        """
        self.state_variable.set_prev_x(prev_state)

    def parameters(self):
        param_dict = {}
        param_dict['state_likelihood_model'] = nn.ParameterList()
        param_dict['state_likelihood_model'].extend(list(self.state_likelihood_model.parameters()))
        param_dict['state_likelihood_model'].extend(list(self.state_variable.parameters()))
        return param_dict
