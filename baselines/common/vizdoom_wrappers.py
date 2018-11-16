import gym
from baselines.common.atari_wrappers import FrameStack, ClipRewardEnv, WarpFrame


def make_vizdoom(env_id):
    env = gym.make(env_id)
    return env

def wrap_deepmind_vizdoom(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env
