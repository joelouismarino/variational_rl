import gym
from gym.spaces import Box
import numpy as np


class NormalizeAction(gym.ActionWrapper):
    """
    Normalizes the reward to [-1, 1].
    """
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self._wrapped_env = env
        ub = np.ones(env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def action(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action
