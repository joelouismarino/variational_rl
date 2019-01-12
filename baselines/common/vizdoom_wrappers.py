import gym
from baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, ScaledFloatFrame, ClipRewardEnv, EpisodicLifeEnv
import numpy as np
import torch

from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

def make_vizdoom(env_id):
    env = gym.make(env_id)
    return env

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height, grayscale):
        """Set to Atari conventions for now."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class Transpose(gym.ObservationWrapper):
    """
    Transposes the axes of the observation to C x H x W.
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_space = env.observation_space
        new_shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])
        self.observation_space = spaces.Box(low=obs_space.low.min(), high=obs_space.high.max(),
                                            shape=new_shape, dtype=obs_space.dtype)

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class AddBatchDim(gym.ObservationWrapper):
    """
    Adds an extra dimension to the beginning of the observation.
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_space = env.observation_space
        new_shape = tuple([1] + list(obs_space.shape))
        self.observation_space = spaces.Box(low=obs_space.low.min(), high=obs_space.high.max(),
                                            shape=new_shape, dtype=obs_space.dtype)

    def observation(self, obs):
        return np.expand_dims(obs, 0)


class ToTensor(gym.ObservationWrapper):
    """
    Converts the observation to a PyTorch tensor.
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_space = env.observation_space
        # Note the dtype is wrong here, won't accept pytorch types
        self.observation_space = spaces.Box(low=obs_space.low.min(), high=obs_space.high.max(),
                                            shape=obs_space.shape, dtype=np.float32)

    def observation(self, obs):
        return torch.from_numpy(obs.astype('float32'))


class RescaleRewardEnv(gym.RewardWrapper):
    """
    Rescales the reward to [0, 1].
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        range = 1. * (self.reward_range[1] - self.reward_range[0])
        return (reward - self.reward_range[0]) / range


def wrap_deepmind_vizdoom(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False,
                          skip_frames=False, frame_width=84, frame_height=84, grayscale=True,
                          to_tensor=False, transpose=False, add_batch_dim=False, rescale_rewards=False):
    env = WarpFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)
    if transpose:
        env = Transpose(env)
    if add_batch_dim:
        env = AddBatchDim(env)
    if episode_life:
        env = EpisodicLifeEnv(env) # this does not yet work with vizdoom cuz it requires a special parameter
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if rescale_rewards:
        env = RescaleRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 1) # in default DQN implementation that came installed with VizDoom this was 1, not 4
    if skip_frames:
        env = MaxAndSkipEnv(env, skip=4) # action repeat currently set under vizdoom_env and is 10 frames
    if to_tensor:
        env = ToTensor(env)
    return env
