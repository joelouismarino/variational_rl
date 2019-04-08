try:
    import gym_minigrid
except ImportError:
    gym_minigrid = None

def get_agent_args(env):
    """
    Get the agent configuration arguments for the specific environment.

    Args:
        env (OpenAI gym environment): the environment for the model

    Return:
        dictionary containing agent configuration arguments
    """
    env_name = env.spec.id
    # VizDoom environments
    if env_name in ['VizDoom-v0']:
        from .vizdoom_config import get_vizdoom_config
        return get_vizdoom_config(env)
    elif env_name in gym_minigrid.register.env_list:
        from .minigrid_config import get_minigrid_config
        return get_minigrid_config(env)
    # Atari environments
    elif env_name in ['SpaceInvaders-v0']:
        from .atari_config import get_atari_config
        return get_atari_config(env)
    # Box 2D environments
    elif env_name in ['BipedalWalker-v2', 'BipedalWalkerHardcore-v2',
                      'CarRacing-v0', 'LunarLander-v2', 'LunarLanderContinuous-v2']:
        raise NotImplementedError
        # from .box2d_config impot get_box2d_config
        # return get_box2d_config(env)
    # Control environments
    elif env_name in ['Acrobat-v1', 'CartPole-v1', 'MountainCar-v0',
                      'MountainCarContinuous-v0', 'Pendulum-v0']:
        from .classic_control_config import get_classic_control_config
        return get_classic_control_config(env)
    elif env_name in ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
                      'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2',
                      'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']:
        from .mujoco_config import get_mujoco_config
        return get_mujoco_config(env)
    else:
        raise NotImplementedError
