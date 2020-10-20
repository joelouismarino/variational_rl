import gym
from gym import spaces
import numpy as np
import torch
# import cv2
# 
# cv2.ocl.setUseOpenCL(False)
#
#
# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env, width, height, grayscale):
#         """Set to Atari conventions for now."""
#         gym.ObservationWrapper.__init__(self, env)
#         self.width = width
#         self.height = height
#         self.grayscale = grayscale
#         if self.grayscale:
#             self.observation_space = spaces.Box(low=0, high=255,
#                 shape=(self.height, self.width, 1), dtype=np.uint8)
#         else:
#             self.observation_space = spaces.Box(low=0, high=255,
#                 shape=(self.height, self.width, 3), dtype=np.uint8)
#
#     def observation(self, frame):
#         if self.grayscale:
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
#         if self.grayscale:
#             frame = np.expand_dims(frame, -1)
#         return frame


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


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
