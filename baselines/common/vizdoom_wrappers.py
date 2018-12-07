import gym
from baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, ScaledFloatFrame, ClipRewardEnv, EpisodicLifeEnv
import numpy as np

from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

def make_vizdoom(env_id):
    env = gym.make(env_id)
    return env

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Follows atari convention for now."""
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

def wrap_deepmind_vizdoom(env, episode_life=False, clip_rewards=False, frame_stack=False, scale = False,
                          skip_frames = False):
    env = WarpFrame(env)
    if episode_life:
        env = EpisodicLifeEnv(env) # this does not yet work with vizdoom cuz it requires a special parameter
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 1) # in default DQN implementation that came installed with VizDoom this was 1, not 4
    if skip_frames:
        env = MaxAndSkipEnv(env, skip=4) # action repeat currently set under vizdoom_env and is 10 frames
    return env
