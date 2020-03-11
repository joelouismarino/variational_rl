import numpy as np
import torch


def euler_integration(prev_state, input, integral_dims, orientation_inds, dt,
                      is_3d, detach):
    """
    Performs euler integration for MuJoCo environments. Used in state estimation
    models for model-based value estimation.

    Args:
        prev_state (torch.Tensor): the previous state variable or estimate
        input (torch.Tensor): the output of the state prediction network
        integral_dims (int): the number of dimensions to integrate
        orientation_inds (list): list of angle indices
        dt (float): the temporal step size
        is_3d (bool): whether the environment is three dimensional
        detach (bool): whether to detach the velocity estimate
    """
    stop_gradient = lambda x: x
    if detach:
        stop_gradient = lambda x: x.detach()

    if is_3d:
        num_joints = integral_dims - 5
        # get the previous values
        position, orientation, joints_pos, root_vel, root_gyro, joints_vel = \
            torch.split(prev_state, [1, 4, num_joints, 3, 3, num_joints], dim=-1)
        # get the predicted changes
        delta_position, delta_orientation, delta_joints_pos, delta_root_vel, delta_root_gyro, delta_joints_vel = \
            torch.split(input, [1, 3, num_joints, 3, 3, num_joints], dim=-1)

        # add the changes to the previous values
        pred_root_vel = root_vel + delta_root_vel
        pred_root_gyro = root_gyro + delta_root_gyro
        pred_joints_vel = joints_vel + delta_joints_vel
        pred_position = position + delta_position
        adjusted_gyro = delta_orientation
        pred_joints_pos = joints_pos + delta_joints_pos

        # euler integration
        adjusted_gyro += stop_gradient(pred_root_gyro)
        pred_joints_pos += dt * stop_gradient(pred_joints_vel)

        quat_zero_leading_dim = torch.zeros(pred_root_gyro.shape[:-1], device=input.device)[..., None]
        quat_gyro = torch.cat([quat_zero_leading_dim, adjusted_gyro], -1)
        orientation = normalize_quaternion(orientation)
        pred_orientation = orientation + dt * 0.5 * quat_mul(orientation, quat_gyro)
        pred_orientation = normalize_quaternion(pred_orientation)

        pred_ori_rmat = quat_to_rmat(pred_orientation)
        pos_vel_delta = torch.sum(pred_ori_rmat * pred_root_vel[..., None, :], -1)
        pred_position += dt * stop_gradient(pos_vel_delta[:, 2:])

        preds = torch.cat([pred_position, pred_orientation, pred_joints_pos,
                           pred_root_vel, pred_root_gyro, pred_joints_vel], -1)
    else:
        # add changes to previous state
        preds = prev_state + input
        # euler integration
        preds[..., :integral_dims] += dt * stop_gradient(preds[..., -integral_dims:])
        # angle consistency
        angle_pred_clone = preds[:, orientation_inds].clone()
        preds[:, orientation_inds] = torch.atan2(torch.sin(angle_pred_clone),
                                                 torch.cos(angle_pred_clone))
    return preds


def euler_loss(predictions, targets, orientation_inds, is_3d):
    """
    Calculates squared error loss for non-orientation dimensions and a custom
    quaternion loss for orientation dimensions.

    Args:
        predictions (torch.Tensor): predicted state [batch_size, n_dims]
        targets (torch.Tensor): target state [batch_size, n_dims]
        orientation_inds (list): list of angle indices
        is_3d (bool): whether the environment is three dimensional
    """
    mask = set(range(predictions.shape[-1])) - set(orientation_inds)
    mask = np.sort(list(mask))
    se_loss = (targets[..., mask] - predictions[..., mask]) ** 2
    target_orientations = targets[..., orientation_inds]
    pred_orientations = predictions[..., orientation_inds]
    if is_3d:
        orientation_loss = quat_distance(target_orientations, pred_orientations) ** 2.
    else:
        orientation_loss = 1. - torch.cos(target_orientations - pred_orientations)
    return torch.cat([se_loss, orientation_loss], -1)

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
