import numpy as np


def get_euler_args(env):
    """
    Gets the euler integration arguements for a MuJoCo environment.

    Args:
        env (gym.env): MuJoCo environment
    """
    assert 'sim' in dir(env.unwrapped)

    integral_dims = env.unwrapped.sim.data.qpos.shape[0]
    # MuJoCo envs typically exclude current position from state definition
    # note: this is only true for position, not velocity
    if env.spec.id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2']:
        integral_dims -= 1
        orientation_inds = [1]
        is_3d = False
    elif env.spec.id in ['AntTruncatedObs-v2', 'HumanoidTruncatedObs-v2',
                         'Swimmer-v2']:
        integral_dims -= 2
        orientation_inds = [1, 2, 3, 4]
        is_3d = True
    else:
        raise NotImplementedError

    arg_dict = {'integral_dims': integral_dims,
                'orientation_inds': orientation_inds,
                'dt': env.unwrapped.dt,
                'is_3d': is_3d}
    return arg_dict

################################################################################
# State Space Definitions
# NOTE: gym removes first dimension from 2d environments and first 2 dimensions
#       from 3d environments.

## Hopper
# State-Space (name/joint/parameter):
#     - rootx       slider      position (m)
#     - rootz       slider      position (m)
#     - rooty       hinge       angle (rad)
#     - thigh_joint hinge       angle (rad)
#     - leg_joint   hinge       angle (rad)
#     - foot_joint  hinge       angle (rad)
#     - rootx       slider      velocity (m/s)
#     - rootz       slider      velocity (m/s)
#     - rooty       hinge       angular velocity (rad/s)
#     - thigh_joint hinge       angular velocity (rad/s)
#     - leg_joint   hinge       angular velocity (rad/s)
#     - foot_joint  hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - thigh_joint hinge       torque (N m)
#     - leg_joint   hinge       torque (N m)
#     - foot_joint  hinge       torque (N m)

## Walker2d
# State-Space (name/joint/parameter):
#     - rootx            slider      position (m)
#     - rootz            slider      position (m)
#     - rooty            hinge       angle (rad)
#     - thigh_joint      hinge       angle (rad)
#     - leg_joint        hinge       angle (rad)
#     - foot_joint       hinge       angle (rad)
#     - thigh_left_joint hinge       angle (rad)
#     - leg_left_joint   hinge       angle (rad)
#     - foot_left_joint  hinge       angle (rad)
#     - rootx            slider      velocity (m/s)
#     - rootz            slider      velocity (m/s)
#     - rooty            hinge       angular velocity (rad/s)
#     - thigh_joint      hinge       angular velocity (rad/s)
#     - leg_joint        hinge       angular velocity (rad/s)
#     - foot_joint       hinge       angular velocity (rad/s)
#     - thigh_left_joint hinge       angular velocity (rad/s)
#     - leg_left_joint   hinge       angular velocity (rad/s)
#     - foot_left_joint  hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - thigh_joint      hinge       torque (N m)
#     - leg_joint        hinge       torque (N m)
#     - foot_joint       hinge       torque (N m)
#     - thigh_left_joint hinge       torque (N m)
#     - leg_left_joint   hinge       torque (N m)
#     - foot_left_joint  hinge       torque (N m)

## HalfCheetah
# State-Space (name/joint/parameter):
#     - rootx     slider      position (m)
#     - rootz     slider      position (m)
#     - rooty     hinge       angle (rad)
#     - bthigh    hinge       angle (rad)
#     - bshin     hinge       angle (rad)
#     - bfoot     hinge       angle (rad)
#     - fthigh    hinge       angle (rad)
#     - fshin     hinge       angle (rad)
#     - ffoot     hinge       angle (rad)
#     - rootx     slider      velocity (m/s)
#     - rootz     slider      velocity (m/s)
#     - rooty     hinge       angular velocity (rad/s)
#     - bthigh    hinge       angular velocity (rad/s)
#     - bshin     hinge       angular velocity (rad/s)
#     - bfoot     hinge       angular velocity (rad/s)
#     - fthigh    hinge       angular velocity (rad/s)
#     - fshin     hinge       angular velocity (rad/s)
#     - ffoot     hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - bthigh    hinge       torque (N m)
#     - bshin     hinge       torque (N m)
#     - bfoot     hinge       torque (N m)
#     - fthigh    hinge       torque (N m)
#     - fshin     hinge       torque (N m)
#     - ffoot     hinge       torque (N m)

## Ant
# State-Space (name/joint/parameter):
#     - root      free        position (m)
#     - hip_1     hinge       angle (rad)
#     - ankle_1   hinge       angle (rad)
#     - hip_2     hinge       angle (rad)
#     - ankle_2   hinge       angle (rad)
#     - hip_3     hinge       angle (rad)
#     - ankle_3   hinge       angle (rad)
#     - hip_4     hinge       angle (rad)
#     - ankle_4   hinge       angle (rad)
#     - root      free        velocity (m/s)
#     - hip_1     hinge       angular velocity (rad/s)
#     - ankle_1   hinge       angular velocity (rad/s)
#     - hip_2     hinge       angular velocity (rad/s)
#     - ankle_2   hinge       angular velocity (rad/s)
#     - hip_3     hinge       angular velocity (rad/s)
#     - ankle_3   hinge       angular velocity (rad/s)
#     - hip_4     hinge       angular velocity (rad/s)
#     - ankle_4   hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - hip_4     hinge       torque (N m)
#     - ankle_4   hinge       torque (N m)
#     - hip_1     hinge       torque (N m)
#     - ankle_1   hinge       torque (N m)
#     - hip_2     hinge       torque (N m)
#     - ankle_2   hinge       torque (N m)
#     - hip_3     hinge       torque (N m)
#     - ankle_3   hinge       torque (N m)

## Humanoid
# State-Space (name/joint/parameter):
#     - root            free        position (m)
#     - abdomen_z       hinge       angle (rad)
#     - abdomen_y       hinge       angle (rad)
#     - abdomen_x       hinge       angle (rad)
#     - right_hip_x     hinge       angle (rad)
#     - right_hip_z     hinge       angle (rad)
#     - right_hip_y     hinge       angle (rad)
#     - right_knee      hinge       angle (rad)
#     - left_hip_x      hinge       angle (rad)
#     - left_hip_z      hinge       angle (rad)
#     - left_hip_y      hinge       angle (rad)
#     - left_knee       hinge       angle (rad)
#     - right_shoulder1 hinge       angle(rad)
#     - right_shoulder2 hinge       angle(rad)
#     - right_elbow     hinge       angle(rad)
#     - left_shoulder1  hinge       angle(rad)
#     - left_shoulder2  hinge       angle(rad)
#     - left_elbow      hinge       angle(rad)
#     - root            free        velocity (m/s)
#     - abdomen_z       hinge       angular velocity (rad/s)
#     - abdomen_y       hinge       angular velocity (rad/s)
#     - abdomen_x       hinge       angular velocity (rad/s)
#     - right_hip_x     hinge       angular velocity (rad/s)
#     - right_hip_z     hinge       angular velocity (rad/s)
#     - right_hip_y     hinge       angular velocity (rad/s)
#     - right_knee      hinge       angular velocity (rad/s)
#     - left_hip_x      hinge       angular velocity (rad/s)
#     - left_hip_z      hinge       angular velocity (rad/s)
#     - left_hip_y      hinge       angular velocity (rad/s)
#     - left_knee       hinge       angular velocity (rad/s)
#     - right_shoulder1 hinge       angular velocity (rad/s)
#     - right_shoulder2 hinge       angular velocity (rad/s)
#     - right_elbow     hinge       angular velocity (rad/s)
#     - left_shoulder1  hinge       angular velocity (rad/s)
#     - left_shoulder2  hinge       angular velocity (rad/s)
#     - left_elbow      hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - abdomen_z       hinge       torque (N m)
#     - abdomen_y       hinge       torque (N m)
#     - abdomen_x       hinge       torque (N m)
#     - right_hip_x     hinge       torque (N m)
#     - right_hip_z     hinge       torque (N m)
#     - right_hip_y     hinge       torque (N m)
#     - right_knee      hinge       torque (N m)
#     - left_hip_x      hinge       torque (N m)
#     - left_hip_z      hinge       torque (N m)
#     - left_hip_y      hinge       torque (N m)
#     - left_knee       hinge       torque (N m)
#     - right_shoulder1 hinge       torque (N m)
#     - right_shoulder2 hinge       torque (N m)
#     - right_elbow     hinge       torque (N m)
#     - left_shoulder1  hinge       torque (N m)
#     - left_shoulder2  hinge       torque (N m)
#     - left_elbow      hinge       torque (N m)
