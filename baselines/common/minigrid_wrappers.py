import gym
import gym_minigrid
from .vizdoom_wrappers import ToTensor, Transpose, RescaleRewardEnv, AddBatchDim


def make_minigrid(env_id):
    env = gym.make(env_id)
    return env


def wrap_minigrid(env, frame_stack=False, frame_width=-1, frame_height=-1,
                  grayscale=True, to_tensor=False, transpose=False,
                  add_batch_dim=False, rescale_rewards=False):

    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    if frame_width != -1 or frame_height != -1:
        env = WarpFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)
    if transpose:
        env = Transpose(env)
    if add_batch_dim:
        env = AddBatchDim(env)
    if rescale_rewards:
        env = RescaleRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 1)
    if to_tensor:
        env = ToTensor(env)

    return env
