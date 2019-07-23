from util.env_util import get_env_types

def get_agent_args(env):
    """
    Get the agent configuration arguments for the specific environment.

    Args:
        env (OpenAI gym environment): the environment for the model

    Return:
        dictionary containing agent configuration arguments
    """
    env_name = env.spec.id
    env_type = get_env_types()[env_name]

    if env_type == 'atari':
        raise NotImplementedError
    elif env_type == 'box2d':
        raise NotImplementedError
    elif env_type == 'classic_control':
        raise NotImplementedError
    elif env_type == 'mujoco':
        from .mujoco_config import get_mujoco_config
        return get_mujoco_config(env)
    elif env_type == 'robotics':
        raise NotImplementedError
    elif env_type == 'vizdoom':
        raise NotImplementedError
    else:
        raise NotImplementedError
