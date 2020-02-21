def get_euler_args(env):
    """
    Gets the euler integration arguements for a MuJoCo environment.

    Args:
        env (gym.env): MuJoCo environment
    """
    assert 'sim' in dir(env.unwrapped)
    arg_dict = {'n_vel': env.unwrapped.sim.data.qvel.shape[0],
                'dt': env.unwrapped.dt}
    return arg_dict
