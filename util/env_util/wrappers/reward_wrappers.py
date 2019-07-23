import gym
import numpy as np
import torch


class RescaleRewardEnv(gym.RewardWrapper):
    """
    Rescales the reward to [0, 1].
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        range = 1. * (self.reward_range[1] - self.reward_range[0])
        return reward / range


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class ToTensor(gym.RewardWrapper):
    """
    Converts reward to a PyTorch tensor.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return torch.from_numpy(reward.astype('float32'))


class AddBatchDim(gym.RewardWrapper):
    """
    Adds a batch dimension to the reward.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        reward = reward.reshape(1)
        return np.expand_dims(reward, 0)
