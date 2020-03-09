import numpy as np
import torch

# HalfCheetah-v3
# STATE_DIMS = 18
# INTEGRAL_DIMS = STATE_DIMS // 2
# ACTIONS_DIMS = 6
# ORIENTATION_INDICES = [2]
# IS_3D_ENV = False
# DT = 0.05

# Hopper-v3
# STATE_DIMS = 12
# INTEGRAL_DIMS = STATE_DIMS // 2
# ACTIONS_DIMS = 3
# ORIENTATION_INDICES = [2]
# IS_3D_ENV = False
# DT = 0.008

# Humanoid-v3
STATE_DIMS = 24 + 23
INTEGRAL_DIMS = 24
ACTIONS_DIMS = 17
ORIENTATION_INDICES = [3, 4, 5, 6]
IS_3D_ENV = True
DT = 0.015


def custom_prediction(models, inputs, euler_integration=True, stop_gradients=True, dt=DT, is_3D_env=IS_3D_ENV):
    stop_gradient = lambda x: x.detach() if stop_gradients else lambda x: x

    if is_3D_env:
        # An option is to only give knowledge of the "up"-vector but not absolute orientation.
        # Not a good idea for environments where reward is only given for movement along a specific axis.
        # The next three lines would be the way to do it:
        # world_z_axis = quat_to_rmat(normalize_quaternion(inputs[..., ORIENTATION_INDICES]))
        # world_z_axis = world_z_axis[..., -1, :]
        # partial_inputs = torch.cat([inputs[..., [2]], world_z_axis, inputs[..., 7:]], -1)

        partial_inputs = inputs[..., 2:]
    else:
        partial_inputs = inputs[..., 1:]

    if isinstance(models, list):
        raw_preds = [m(partial_inputs) for m in models]
        raw_preds = torch.cat(raw_preds, -1)
    else:
        raw_preds = models(partial_inputs)

    if is_3D_env:
        num_joints = INTEGRAL_DIMS - 3 - 4
        position, orientation, joints_pos, root_vel, root_gyro, joints_vel = \
            torch.split(inputs[..., :STATE_DIMS],
                        [3, 4, num_joints, 3, 3, num_joints], dim=-1)
        delta_position, delta_orientation, delta_joints_pos, delta_root_vel, delta_root_gyro, delta_joints_vel = \
            torch.split(raw_preds,
                        [3, 3, num_joints, 3, 3, num_joints], dim=-1)

        pred_root_vel = root_vel + delta_root_vel
        pred_root_gyro = root_gyro + delta_root_gyro
        pred_joints_vel = joints_vel + delta_joints_vel

        pred_position = position + delta_position
        adjusted_gyro = delta_orientation
        pred_joints_pos = joints_pos + delta_joints_pos

        if euler_integration:
            adjusted_gyro += stop_gradient(pred_root_gyro)
            pred_joints_pos += dt * stop_gradient(pred_joints_vel)

        quat_zero_leading_dim = torch.zeros(pred_root_gyro.shape[:-1])[..., None]
        quat_gyro = torch.cat([quat_zero_leading_dim, adjusted_gyro], -1)
        orientation = normalize_quaternion(orientation)
        pred_orientation = orientation + dt * 0.5 * quat_mul(orientation, quat_gyro)
        pred_orientation = normalize_quaternion(pred_orientation)

        if euler_integration:
            pred_ori_rmat = quat_to_rmat(pred_orientation)
            pos_vel_delta = torch.sum(
                pred_ori_rmat * pred_root_vel[..., None, :], -1)
            pred_position += dt * stop_gradient(pos_vel_delta)

        preds = torch.cat([pred_position, pred_orientation, pred_joints_pos,
                           pred_root_vel, pred_root_gyro, pred_joints_vel], -1)

    else:
        preds = inputs[..., :-ACTIONS_DIMS] + raw_preds

        if euler_integration:
            preds[..., :INTEGRAL_DIMS] += dt * stop_gradient(preds[..., INTEGRAL_DIMS:])

        angle_pred_clone = preds[:, ORIENTATION_INDICES].clone()
        preds[:, ORIENTATION_INDICES] = torch.atan2(torch.sin(angle_pred_clone),
                                                    torch.cos(angle_pred_clone))
    return preds


def multi_step_prediction(pred_fn, models, inputs, actions, dt=DT):
    T, N = actions.size()[:2]
    preds = torch.zeros(T, N, STATE_DIMS)
    for t in range(-1, T - 1):
        inputs_t = torch.cat([preds[t] if t > -1 else inputs,
                              actions[t + 1]], -1)
        preds[t + 1] = pred_fn(models, inputs_t, dt)
    return preds


def custom_loss_no_aggregation(predictions, targets, is_3D_env=IS_3D_ENV):
    mask = set(range(STATE_DIMS)) - set(ORIENTATION_INDICES)
    mask = np.sort(list(mask))
    se_loss = (targets[..., mask] - predictions[..., mask]) ** 2
    if is_3D_env:
        orientation_loss = quat_distance(targets[..., ORIENTATION_INDICES],
                                         predictions[..., ORIENTATION_INDICES]
                                         ) ** 2.
    else:
        orientation_loss = 1. - torch.cos(
            targets[..., ORIENTATION_INDICES] - predictions[..., ORIENTATION_INDICES])
    return torch.cat([se_loss, orientation_loss], -1)


def custom_loss_per_dimension(predictions, targets):
    return torch.mean(
        custom_loss_no_aggregation(predictions, targets), -2)


def custom_loss(predictions, targets):
    return torch.sum(
        custom_loss_per_dimension(predictions, targets), -1)


def quat_distance(target_quats, pred_quats, eps=1e-6, sign_invariant=True):
    """Angle of the "difference rotation" between target_quats and pred_quats.

    Though arguments `target_quats` and `pred_quats` are suggestively named,
    the function is invariant to the actual ordering of these arguments.
    Args:
      target_quats: the ground truth quaternion orientations: shape=(..., 4).
      pred_quats: the predicted quaternion orientations: shape=(..., 4).
      eps: tolerance parameter to ensure arccos evaluates correctly.
      sign_invariant: quaternions q and -q specify equivalent orientations,
        however their geodesic distance is not 0. This argument specifies whether
        we consider the sign information or whether we pick the sign which
        gives us the shortest distance between true and predicted quaternions.

    Returns:
      shape=(..., 1). Angle subtended by true and predicted quaternions along a
      great arc of the S^3 sphere (also equivalent to double the geodesic
      distance).
      If sign_invariant=True then the minimum such angle/distance is returned by
      ignoring the sign of the quaternions.
    """
    quat_dot = torch.sum(target_quats * pred_quats, -1, keepdim=True)
    if sign_invariant:
        quat_dot = torch.abs(quat_dot) - eps
    else:
        quat_dot = quat_dot - eps * torch.sign(quat_dot)
    return 2 * torch.acos(quat_dot)


def quat_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r (Hamilton product).
    The quaternions should be in (w, x, y, z) format.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def normalize_quaternion(quaternion: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=eps)


def quat_to_rmat(quaternion: torch.Tensor) -> torch.Tensor:
    """Converts quaternion(s) to rotation matrix.
    The quaternion should be in (w, x, y, z) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape (*, 4).
    Return:
        torch.Tensor: the rotation matrix of shape (*, 3, 3).
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.)

    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1)

    shape = quaternion.shape[:-1] + (3, 3)
    return matrix.view(shape)
