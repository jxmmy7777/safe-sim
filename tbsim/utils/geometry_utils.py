import numpy as np

import torch
from tbsim.utils.tensor_utils import round_2pi
from enum import IntEnum

##Paste from l5kit.geometry
def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a set of 2D/3D points using the given transformation matrix.
    Assumes row major ordering of the input points. The transform function has 3 modes:
    - points (N, F), transf_matrix (F+1, F+1)
    all points are transformed using the matrix and the output points have shape (N, F).
    - points (B, N, F), transf_matrix (F+1, F+1)
    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
    transf_matrix is broadcasted.
    - points (B, N, F), transf_matrix (B, F+1, F+1)
    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    :param points: Input points of shape (N, F) or (B, N, F)
        with F = 2 or 3 depending on input points are 2D or 3D points.
    :param transf_matrix: Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
    :return: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
    """
    points_log = f" received points with shape {points.shape} "
    matrix_log = f" received matrices with shape {transf_matrix.shape} "

    assert points.ndim in [2, 3], f"points should have ndim in [2,3],{points_log}"
    assert transf_matrix.ndim in [2, 3], f"matrix should have ndim in [2,3],{matrix_log}"
    assert points.ndim >= transf_matrix.ndim, f"points ndim should be >= than matrix,{points_log},{matrix_log}"

    points_feat = points.shape[-1]
    assert points_feat in [2, 3], f"last points dimension must be 2 or 3,{points_log}"
    assert transf_matrix.shape[-1] == transf_matrix.shape[-2], f"matrix should be a square matrix,{matrix_log}"

    matrix_feat = transf_matrix.shape[-1]
    assert matrix_feat in [3, 4], f"last matrix dimension must be 3 or 4,{matrix_log}"
    assert points_feat == matrix_feat - 1, f"points last dim should be one less than matrix,{points_log},{matrix_log}"

    def _transform(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = np.transpose(transf_matrix, (0, 2, 1))
        return points @ transf_matrix[:, :num_dims, :num_dims] + transf_matrix[:, -1:, :num_dims]

    if points.ndim == transf_matrix.ndim == 2:
        points = np.expand_dims(points, 0)
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif points.ndim == transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unsupported case!{points_log},{matrix_log}")


from typing import cast, Union

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """angle_between_vectors returns the angle in radians between two vectors.

    Args:
        v1 (np.ndarray): Vector 1 of shape (N)
        v2 (np.ndarray): Vector 2 of same shape as ``v1``

    Returns:
        float: angle in radians
    """
    cos_ang = np.dot(v1, v2)
    sin_ang = np.linalg.norm(np.cross(v1, v2))
    return cast(float, np.arctan2(sin_ang, cos_ang))


def compute_yaw_around_north_from_direction(direction_vector: np.ndarray) -> float:
    """compute_yaw_from_direction computes the yaw as angle between a 2D input direction vector and
the y-axis direction vector (0, 1).

    Args:
        direction_vector (np.ndarray): Vector of shape (2,)

    Returns:
        float: angle to (0,1) vector in radians
    """
    return angle_between_vectors(direction_vector, np.array([0.0, 1.0]))


def angular_distance(angle_a: Union[float, np.ndarray], angle_b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """ A function that takes two arrays of angles in radian and compute the angular distance, wrap the angular
    distance such that they are always in the [-pi, pi) range.

    Args:
        angle_a (np.ndarray, float): first array of angles in radians
        angle_b (np.ndarray, float): second array of angles in radians

    Returns:
        angular distance in radians between two arrays of angles
    """

    return (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi



def get_box_agent_coords(pos, yaw, extent):
    corners = (torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5).to(pos.device) * (
        extent.unsqueeze(-2)
    )
    s = torch.sin(yaw).unsqueeze(-1)
    c = torch.cos(yaw).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    rotated_corners = (corners + pos.unsqueeze(-2)) @ rotM
    return rotated_corners


def get_box_world_coords(pos, yaw, extent):
    corners = (torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5).to(pos.device) * (
        extent.unsqueeze(-2)
    )
    s = torch.sin(yaw).unsqueeze(-1)
    c = torch.cos(yaw).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    rotated_corners = corners @ rotM + pos.unsqueeze(-2)
    return rotated_corners


def get_box_agent_coords_np(pos, yaw, extent):
    corners = (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5) * (
        extent[..., None, :]
    )
    s = np.sin(yaw)[..., None]
    c = np.cos(yaw)[..., None]
    rotM = np.concatenate((np.concatenate((c, s), axis=-1), np.concatenate((-s, c), axis=-1)), axis=-2)
    rotated_corners = (corners + pos[..., None, :]) @ rotM
    return rotated_corners


def get_box_world_coords_np(pos, yaw, extent):
    corners = (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5) * (
        extent[..., None, :]
    )
    s = np.sin(yaw)[..., None]
    c = np.cos(yaw)[..., None]
    rotM = np.concatenate((np.concatenate((c, s), axis=-1), np.concatenate((-s, c), axis=-1)), axis=-2)
    rotated_corners = corners @ rotM + pos[..., None, :]
    return rotated_corners


def get_upright_box(pos, extent):
    yaws = torch.zeros(*pos.shape[:-1], 1).to(pos.device)
    boxes = get_box_world_coords(pos, yaws, extent)
    upright_boxes = boxes[..., [0, 2], :]
    return upright_boxes


def batch_nd_transform_points(points, Mat):
    if points.dtype != Mat.dtype:
        Mat = Mat.to(points.dtype)
    ndim = Mat.shape[-1] - 1
    Mat = Mat.transpose(-1, -2)
    return (points.unsqueeze(-2) @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
        ..., -1:, :ndim
    ].squeeze(-2)

def batch_nd_transform_points_np(points, Mat):
    if points.dtype != Mat.dtype:
        points = points.astype(Mat.dtype)
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim-2))+[Mat.ndim-1]+[Mat.ndim-2]
    Mat = np.transpose(Mat,batch)
    if points.ndim==Mat.ndim-1:
        return (points[...,np.newaxis,:] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim==Mat.ndim:
        return ((points[...,np.newaxis,:] @ Mat[...,np.newaxis, :ndim, :ndim]) + Mat[
            ...,np.newaxis, -1:, :ndim]).squeeze(-2)
    else:
        raise Exception("wrong shape")


def transform_points_tensor(
    points: torch.Tensor, transf_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Transform a set of 2D/3D points using the given transformation matrix.
    Assumes row major ordering of the input points. The transform function has 3 modes:
    - points (N, F), transf_matrix (F+1, F+1)
    all points are transformed using the matrix and the output points have shape (N, F).
    - points (B, N, F), transf_matrix (F+1, F+1)
    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
    transf_matrix is broadcasted.
    - points (B, N, F), transf_matrix (B, F+1, F+1)
    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    :param points: Input points of shape (N, F) or (B, N, F)
        with F = 2 or 3 depending on input points are 2D or 3D points.
    :param transf_matrix: Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
    :return: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
    """
    points_log = f" received points with shape {points.shape} "
    matrix_log = f" received matrices with shape {transf_matrix.shape} "

    assert points.ndim in [2, 3], f"points should have ndim in [2,3],{points_log}"
    assert transf_matrix.ndim in [
        2,
        3,
    ], f"matrix should have ndim in [2,3],{matrix_log}"
    assert (
        points.ndim >= transf_matrix.ndim
    ), f"points ndim should be >= than matrix,{points_log},{matrix_log}"

    points_feat = points.shape[-1]
    assert points_feat in [2, 3], f"last points dimension must be 2 or 3,{points_log}"
    assert (
        transf_matrix.shape[-1] == transf_matrix.shape[-2]
    ), f"matrix should be a square matrix,{matrix_log}"

    matrix_feat = transf_matrix.shape[-1]
    assert matrix_feat in [3, 4], f"last matrix dimension must be 3 or 4,{matrix_log}"
    assert (
        points_feat == matrix_feat - 1
    ), f"points last dim should be one less than matrix,{points_log},{matrix_log}"

    def _transform(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = torch.permute(transf_matrix, (0, 2, 1))
        return (
            points @ transf_matrix[:, :num_dims, :num_dims]
            + transf_matrix[:, -1:, :num_dims]
        )

    if points.ndim == transf_matrix.ndim == 2:
        points = torch.unsqueeze(points, 0)
        transf_matrix = torch.unsqueeze(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif points.ndim == transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        transf_matrix = torch.unsqueeze(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unsupported case!{points_log},{matrix_log}")


def PED_PED_collision(p1, p2, S1, S2):
    if isinstance(p1, torch.Tensor):

        return (
            torch.linalg.norm(p1[..., 0:2] - p2[..., 0:2], dim=-1)
            - (S1[..., 0] + S2[..., 0]) / 2
        )

    elif isinstance(p1, np.ndarray):

        return (
            np.linalg.norm(p1[..., 0:2] - p2[..., 0:2], axis=-1)
            - (S1[..., 0] + S2[..., 0]) / 2
        )
    else:
        raise NotImplementedError


def batch_rotate_2D(xy, theta):
    if isinstance(xy, torch.Tensor):
        x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
        y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
        return torch.stack([x1, y1], dim=-1)
    elif isinstance(xy, np.ndarray):
        x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
        y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
        return np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=-1)

        


def VEH_VEH_collision(
    p1, p2, S1, S2, offsetX=0.0, offsetY=0.0
):
    if isinstance(p1, torch.Tensor):
        cornersX = torch.kron(
            S1[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(p1.device)
        )
        cornersY = torch.kron(
            S1[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(p1.device)
        )
        corners = torch.stack([cornersX, cornersY], dim=-1)
        theta1 = p1[..., 2]
        theta2 = p2[..., 2]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat_interleave(4, dim=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat_interleave(4, dim=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat_interleave(4, dim=-1))
        dis = torch.maximum(
            torch.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat_interleave(4, dim=-1),
            torch.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat_interleave(4, dim=-1),
        ).view(*S1.shape[:-1], 4)
        min_dis, _ = torch.min(dis, dim=-1)

        return min_dis

    elif isinstance(p1, np.ndarray):
        cornersX = np.kron(S1[..., 0] + offsetX, np.array([0.5, 0.5, -0.5, -0.5]))
        cornersY = np.kron(S1[..., 1] + offsetY, np.array([0.5, -0.5, 0.5, -0.5]))
        corners = np.concatenate((cornersX, cornersY), axis=-1)
        theta1 = p1[..., 2]
        theta2 = p2[..., 2]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat(4, axis=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat(4, axis=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat(4, axis=-1))
        dis = np.maximum(
            np.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat(4, axis=-1),
            np.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat(4, axis=-1),
        ).reshape(*S1.shape[:-1], 4)
        min_dis = np.min(dis, axis=-1)
        return min_dis
    else:
        raise NotImplementedError


def VEH_PED_collision(p1, p2, S1, S2):
    if isinstance(p1, torch.Tensor):

        mask = torch.logical_or(
            torch.abs(p1[..., 2]) > 0.1, torch.linalg.norm(p2[..., 2:4], dim=-1) > 0.1
        ).detach()
        theta = p1[..., 2]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)

        return torch.maximum(
            torch.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
            torch.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
        )
    elif isinstance(p1, np.ndarray):

        theta = p1[..., 2]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)
        return np.maximum(
            np.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
            np.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
        )
    else:
        raise NotImplementedError


def PED_VEH_collision(p1, p2, S1, S2):
    return VEH_PED_collision(p2, p1, S2, S1)

def batch_get_distance(x, line):
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta, axis=-1)
    return dis

def batch_proj(x, line):
    # x:[batch,3], line:[batch,N,3]
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta, axis=-1)
        idx0 = torch.argmin(dis, dim=-1)
        idx = idx0.view(*line.shape[:-2], 1, 1).repeat(
            *([1] * (batch_dim + 1)), line.shape[-1]
        )
        line_min = torch.squeeze(torch.gather(line, -2, idx), dim=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        #transfer to nt coordinates
        delta_y = -dx * torch.sin(line_min[..., None, 2]) + dy * torch.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * torch.cos(line_min[..., None, 2]) + dy * torch.sin(
            line_min[..., None, 2]
        )
        # ref_pts = torch.stack(
        #     [
        #         line_min[..., 0] + delta_x * torch.cos(line_min[..., 2]),
        #         line_min[..., 1] + delta_x * torch.sin(line_min[..., 2]),
        #         line_min[..., 2],
        #     ],
        #     dim=-1,
        # )
        delta_psi = round_2pi(x[..., 2] - line_min[..., 2])
        #hack
        delta_y = dis
        return (
            delta_x,
            delta_y,
            torch.unsqueeze(delta_psi, dim=-1),
        )
    elif isinstance(x, np.ndarray):
        delta = line[..., 0:2] - np.repeat(
            x[..., np.newaxis, 0:2], line_length, axis=-2
        )
        dis = np.linalg.norm(delta, axis=-1)
        idx0 = np.argmin(dis, axis=-1)
        idx = idx0.reshape(*line.shape[:-2], 1, 1).repeat(line.shape[-1], axis=-1)
        line_min = np.squeeze(np.take_along_axis(line, idx, axis=-2), axis=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * np.sin(line_min[..., None, 2]) + dy * np.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * np.cos(line_min[..., None, 2]) + dy * np.sin(
            line_min[..., None, 2]
        )
        # line_min[..., 0] += delta_x * np.cos(line_min[..., 2])
        # line_min[..., 1] += delta_x * np.sin(line_min[..., 2])
        delta_psi = round_2pi(x[..., 2] - line_min[..., 2])
        return (
            delta_x,
            delta_y,
            np.expand_dims(delta_psi, axis=-1),
        )



class CollisionType(IntEnum):
    """This enum defines the three types of collisions: front, rear and side."""
    FRONT = 0
    REAR = 1
    SIDE = 2

def detect_collision(
        ego_pos: np.ndarray,
        ego_yaw: np.ndarray,
        ego_extent: np.ndarray,
        other_pos: np.ndarray,
        other_yaw: np.ndarray,
        other_extent: np.ndarray,
):
    """
    Computes whether a collision occured between ego and any another agent.
    Also computes the type of collision: rear, front, or side.
    For this, we compute the intersection of ego's four sides with a target
    agent and measure the length of this intersection. A collision
    is classified into a class, if the corresponding length is maximal,
    i.e. a front collision exhibits the longest intersection with
    egos front edge.

    .. note:: please note that this funciton will stop upon finding the first
              colision, so it won't return all collisions but only the first
              one found.

    :param ego_pos: predicted centroid
    :param ego_yaw: predicted yaw
    :param ego_extent: predicted extent
    :param other_pos: target agents
    :return: None if not collision was found, and a tuple with the
             collision type and the agent track_id
    """
    # from l5kit.planning import utils, directly pasted it down
    ego_bbox = _get_bounding_box(centroid=ego_pos, yaw=ego_yaw, extent=ego_extent)
    
    # within_range_mask = within_range(ego_pos, ego_extent, other_pos, other_extent)
    # TBD: hack to ignore shapely warning when no intersection is found.
    import warnings
    warnings.filterwarnings('ignore')
    for i in range(other_pos.shape[0]):
        agent_bbox = _get_bounding_box(other_pos[i], other_yaw[i], other_extent[i])
        if ego_bbox.intersects(agent_bbox):
            front_side, rear_side, left_side, right_side = _get_sides(ego_bbox)
            front_int = agent_bbox.intersection(front_side)
            rear_int = agent_bbox.intersection(rear_side)
            left_int = agent_bbox.intersection(left_side)
            right_int = agent_bbox.intersection(right_side)

            intersection_length_per_side = np.asarray(
                [
                    front_int.length,
                    rear_int.length,
                    left_int.length,
                    right_int.length,
                ]
            )
            argmax_side = np.argmax(intersection_length_per_side)

            # Remap here is needed because there are two sides that are
            # mapped to the same collision type CollisionType.SIDE
            max_collision_types = max(CollisionType).value
            remap_argmax = min(argmax_side, max_collision_types)
            collision_type = CollisionType(remap_argmax)
            return collision_type, i
    # TBD: restore warning filter
    warnings.filterwarnings('default')
    return None



def calc_distance_map(road_flag,max_dis = 10,mode="L1"):
    """mark the image with manhattan distance to the drivable area

    Args:
        road_flag (torch.Tensor[B,W,H]): an image with 1 channel, 1 for drivable area, 0 for non-drivable area
        max_dis (int, optional): maximum distance that the result saturates to. Defaults to 10.
    """
    out = torch.zeros_like(road_flag,dtype=torch.float)
    out[road_flag==0] = max_dis 
    out[road_flag==1] = 0
    if mode=="L1":
        for i in range(max_dis-1):
            out[...,1:,:] = torch.min(out[...,1:,:],out[...,:-1,:]+1)
            out[...,:-1,:] = torch.min(out[...,:-1,:],out[...,1:,:]+1)
            out[...,:,1:] = torch.min(out[...,:,1:],out[...,:,:-1]+1)
            out[...,:,:-1] = torch.min(out[...,:,:-1],out[...,:,1:]+1)
    elif mode=="Linf":
        for i in range(max_dis-1):
            out[...,1:,:] = torch.min(out[...,1:,:],out[...,:-1,:]+1)
            out[...,:-1,:] = torch.min(out[...,:-1,:],out[...,1:,:]+1)
            out[...,:,1:] = torch.min(out[...,:,1:],out[...,:,:-1]+1)
            out[...,:,:-1] = torch.min(out[...,:,:-1],out[...,:,1:]+1)
            out[...,1:,1:] = torch.min(out[...,1:,1:],out[...,:-1,:-1]+1)
            out[...,1:,:-1] = torch.min(out[...,1:,:-1],out[...,:-1,1:]+1)
            out[...,:-1,:-1] = torch.min(out[...,:-1,:-1],out[...,1:,1:]+1)
            out[...,:-1,1:] = torch.min(out[...,:-1,1:],out[...,1:,:-1]+1)

    return out


def transform_matrices(angles: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    """Creates a 3x3 transformation matrix for each angle and translation in the input.

    Args:
        angles (Tensor): The (N,)-shaped angles tensor to rotate points by.
        translations (Tensor): The (N,2)-shaped translations to shift points by.

    Returns:
        Tensor: The Nx3x3 transformation matrices.
    """
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    last_rows = torch.tensor(
        [[0.0, 0.0, 1.0]], dtype=angles.dtype, device=angles.device
    ).expand((angles.shape[0], -1))
    return torch.stack(
        [
            torch.stack([cos_vals, -sin_vals, translations[:, 0]], dim=-1),
            torch.stack([sin_vals, cos_vals, translations[:, 1]], dim=-1),
            last_rows,
        ],
        dim=-2,
    )


def transform_agents_to_world(pos_pred, yaw_pred, batch_world_from_agent):
    '''
    Converts local agent poses to global frame.
    Input:
        pos_pred: (num_agents, num_samp, time_steps, 2)
        yaw_pred: (num_agents, num_samp, time_steps, 1)
        batch_world_from_agent: (num_agent, 3, 3) (agent-centric) or (3, 3) (scene-centric)
    Output:
        agents_fut_pos: (num_agents, num_samp, time_steps, 2)
        agents_fut_yaw: (num_agents, num_samp, time_steps, 1)
    '''
    # hacky way to handle scene-centric matrix
    if len(batch_world_from_agent.shape) == 2:
        batch_world_from_agent = batch_world_from_agent.unsqueeze(0)
    bsize, num_samp, t, _ = pos_pred.size()
    pos_pred = pos_pred.reshape((bsize*num_samp, t, 2))
    batch_world_from_agent = batch_world_from_agent.unsqueeze(1).expand((bsize, num_samp, 3, 3)).reshape((bsize*num_samp, 3, 3))

    agents_fut_pos = transform_points_tensor(pos_pred, batch_world_from_agent).reshape((bsize, num_samp, t, 2))
    hvec = torch.cat([torch.cos(yaw_pred), torch.sin(yaw_pred)], dim=-1).reshape((bsize*num_samp, t, 2))
    world_hvec = transform_points_tensor(hvec, batch_world_from_agent).reshape((bsize, num_samp, t, 2))
    world_origin = batch_world_from_agent[:,:2,2].reshape((bsize, num_samp, 1, 2))
    world_hvec = world_hvec - world_origin
    agents_fut_yaw = torch.atan2(world_hvec[..., 1], world_hvec[..., 0]).unsqueeze(-1)
    
    return agents_fut_pos, agents_fut_yaw

##paste from l5kit.planning.utils

from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

# TODO(perone):  The functions _ego_agent_within_range, _get_bounding_box,
# _get_sides would ideally be moved to an abstraction of the Agent
# later. Such as in the example below:
#
#  ego = Agent(...)
#  agent = Agent(...)
#  bbox = ego.get_bounding_box()
#  within_range = ego.within_range(agent)
#  sides = ego.get_sides()


def _get_bounding_box(centroid: np.ndarray, yaw: np.ndarray,
                      extent: np.ndarray,) -> Polygon:
    """This function will get a shapely Polygon representing the bounding box
    with an optional buffer around it.

    :param centroid: centroid of the agent
    :param yaw: the yaw of the agent
    :param extent: the extent of the agent
    :return: a shapely Polygon
    """
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


# TODO(perone): this should probably return a namedtuple as otherwise it
# would have to depend on the correct ordering of the front/rear/left/right
def _get_sides(bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    """This function will get the sides of a bounding box.

    :param bbox: the bounding box
    :return: a tuple with the four sides of the bounding box as LineString,
             representing front/rear/left/right.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    return (
        LineString([(x1, y1), (x2, y2)]),
        LineString([(x3, y3), (x4, y4)]),
        LineString([(x1, y1), (x4, y4)]),
        LineString([(x2, y2), (x3, y3)]),
    )


def within_range(ego_centroid: np.ndarray, ego_extent: np.ndarray,
                 agent_centroid: np.ndarray, agent_extent: np.ndarray) -> np.ndarray:
    """This function will check if the agent is within range of the ego. It accepts
    as input a vectorized form with shapes N,D or a flat vector as well with shapes just D.

    :param ego_centroid: the ego centroid (shape: 2)
    :param ego_extent: the ego extent (shape: 3)
    :param agent_centroid: the agent centroid (shape: N, 2)
    :param agent_extent: the agent extent (shape: N, 3)
    :return: array with True if within range, False otherwise (shape: N)
    """
    distance = np.linalg.norm(ego_centroid - agent_centroid, axis=-1)
    norm_ego = np.linalg.norm(ego_extent[:2])
    norm_agent = np.linalg.norm(agent_extent[:, :2], axis=-1)
    max_range = 0.5 * (norm_ego + norm_agent)
    return distance < max_range

# def batch_get_distance(x, line):
#     line_length = line.shape[-2]
#     batch_dim = x.ndim - 1
#     if isinstance(x, torch.Tensor):
#         delta = line[:,None,:,0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
#             *([1] * batch_dim), line_length, 1
#         )
#         dis = torch.linalg.norm(delta, axis=-1)
#     return dis
