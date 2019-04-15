import gym
import numpy as np
from .vizdoom_wrappers import ToTensor, RescaleRewardEnv, AddBatchDim
'''Not used. Currently using OpenAI Baselines Mujoco Wrappers under VecNormalize'''
class SignRewardEnv(gym.RewardWrapper):
    """
    Takes the sign of the reward to [0, 1].
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # print(f"reward: {reward} signreward: {np.sign(reward)}")
        return float(reward)

def wrap_mujoco(env, to_tensor=False, add_batch_dim=False, rescale_rewards=False, sign_rewards=False):
    if add_batch_dim:
        env = AddBatchDim(env)
    if rescale_rewards:
        env = RescaleRewardEnv(env)
    if sign_rewards:
        env = SignRewardEnv(env)
    if to_tensor:
        env = ToTensor(env)

    return env
