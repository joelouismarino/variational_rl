import numpy as np


def get_euler_args(env):
    """
    Gets the euler integration arguements for a MuJoCo environment.

    Args:
        env (gym.env): MuJoCo environment
    """
    assert 'sim' in dir(env.unwrapped)

    n_pos = env.unwrapped.sim.data.qpos.shape[0]
    # MuJoCo envs typically exclude current position from state definition
    # note: this is only true for position, not velocity
    if env.spec.id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2']:
        n_pos -= 1
        angle_inds = [2]
    elif env.spec.id in ['AntTruncatedObs-v2', 'HumanoidTruncatedObs-v2',
                         'Swimmer-v2']:
        n_pos -= 2
        raise NotImplementedError
    # # TODO: the following isn't correct
    # elif env.spec.id in ['InvertedPendulum-v2', 'InvertedDoublePendulum-v2',
    #                      'Reacher-v2']:
    else:
        raise NotImplementedError

    arg_dict = {'n_pos': n_pos,
                'dt': env.unwrapped.dt,
                'angle_inds': angle_inds}
    return arg_dict
