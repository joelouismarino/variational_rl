from util.env_util import get_env_types
from .postprocess_misc_args import postprocess_misc_args
from .mujoco_config import get_mujoco_config


def get_agent_args(env):
    """
    Get the agent configuration arguments for the specific environment.

    Args:
        env (OpenAI gym environment): the environment for the model

    Return:
        dictionary containing agent configuration arguments
    """
    return get_mujoco_config(env)
