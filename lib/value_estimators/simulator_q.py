import gym
import torch
import torch.nn as nn
from lib.distributions import kl_divergence
from misc.estimators import n_step


class SimulatorQEstimator(nn.Module):
    """
    A Q-value estimator that uses a differentiable ground-truth simulator.
    Currently only works with DroneEnv.

    Args:
        env_type (str): the environment name
        horizon (int): the rollout horizon
    """
    def __init__(self, env_type, horizon):
        super(SimulatorQEstimator, self).__init__()
        assert env_type == 'Drone-v0'
        self.env_type = env_type
        self.horizon = horizon
        self.env = None

    def forward(self, agent, state, action, detach_params=False, *args, **kwargs):
        """
        Rolls out the simulator.
        """
        self.reset(batch_size=state.shape[0], prev_state=None)
        self.env.model.to(agent.device)
        self.env.set_state(state)
        rewards_list = []
        kl_list = []
        for _ in range(self.horizon):
            action = action.tanh() if agent.postprocess_action else action
            # step the environment
            state, reward, done, _ = self.env.step(action)
            rewards_list.append(reward)
            # estimate the next action
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

        # calculate the Q-value estimate
        total_rewards = torch.stack(rewards_list)
        total_kl = torch.stack(kl_list)
        total_q_values = torch.zeros((self.horizon + 1, total_rewards.shape[1], total_rewards.shape[2])).to(agent.device)

        estimate = n_step(total_q_values, total_rewards, total_kl, discount=agent.reward_discount)
        return estimate

    def reset(self, batch_size, prev_state):
        """
        Resets the simulator.
        """
        self.env = gym.make(self.env_type, batch_size=batch_size)
        # self.env.model.to(prev_state.device)

    def set_prev_state(self, *args, **kwargs):
        pass

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

    def parameters(self):
        return {}
