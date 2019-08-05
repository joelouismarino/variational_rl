import gym
from collections import defaultdict
from .registration import register_env
from .wrappers import action_wrappers, observation_wrappers, reward_wrappers


def get_env_types():
    """
    Get the type of each environment, e.g. mujoco, atari, etc.
    """
    env_types = {}
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        env_types[env.id] = env_type
    return env_types


def create_env(env_name, seed=None):
    """
    Create an OpenAI gym environment.
    """
    register_env(env_name)
    env_types = get_env_types()
    if env_name not in env_types:
        raise KeyError('Environment not found.')
    env_type = env_types[env_name]
    env = gym.make(env_name)
    env.seed(seed)

    # wrap the environment
    if env_type == 'atari':
        raise NotImplementedError
    elif env_type == 'box2d':
        raise NotImplementedError
    elif env_type == 'classic_control':
        raise NotImplementedError
    elif env_type == 'mujoco':
        env = observation_wrappers.AddBatchDim(env)
        env = observation_wrappers.ToTensor(env)
        env = reward_wrappers.AddBatchDim(env)
        env = reward_wrappers.ToTensor(env)
    elif env_type == 'robotics':
        raise NotImplementedError
    elif env_type == 'vizdoom':
        env = observation_wrappers.WarpFrame(env, width=45, height=30, grayscale=False)
        env = observation_wrappers.Transpose(env)
        env = observation_wrappers.AddBatchDim(env)
        env = observation_wrappers.ScaledFloatFrame(env)
        env = observation_wrappers.ToTensor(env)
        env = reward_wrappers.ToTensor(env)

    return env
