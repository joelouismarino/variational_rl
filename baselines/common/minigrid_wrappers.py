import gym
import gym_minigrid
import numpy as np
from .vizdoom_wrappers import ToTensor, Transpose, RescaleRewardEnv, AddBatchDim


def make_minigrid(env_id):
    env = gym.make(env_id)
    return env

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 5.


def wrap_minigrid(env, frame_stack=False, frame_width=-1, frame_height=-1,
                  grayscale=True, scale=False, to_tensor=False, transpose=False,
                  add_batch_dim=False, rescale_rewards=False):

    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    if frame_width != -1 or frame_height != -1:
        env = WarpFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)
    if transpose:
        env = Transpose(env)
    if add_batch_dim:
        env = AddBatchDim(env)
    if scale:
        env = ScaledFloatFrame(env)
    if rescale_rewards:
        env = RescaleRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 1)
    if to_tensor:
        env = ToTensor(env)

    return env
