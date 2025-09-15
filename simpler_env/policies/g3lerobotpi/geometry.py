import math

import numpy as np
import torch

def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Extract the first two columns of a 3x3 rotation matrix => 6D representation.
    R: shape [B, 3, 3]
    returns: shape [B, 6]
    """
    # columns 0 and 1 => shape [B, 3, 2]
    col0 = R[..., :, 0]  # [B, 3]
    col1 = R[..., :, 1]  # [B, 3]
    return torch.cat([col0, col1], dim=-1)  # [B, 6]

def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to a rotation matrix.
    Args:
        quat: [B,4] tensor with order (w, x, y, z)
    Returns:
        [B, 3, 3] rotation matrix.
    """
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    B = quat.shape[0]
    R = torch.zeros(B, 3, 3, device=quat.device, dtype=quat.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R

def quat_to_rot6d(tcp_pose: torch.Tensor) -> torch.Tensor:
    """
    Convert tcp_pose from quaternion representation to 6D rotation representation.
    
    Args:
        tcp_pose: [B, 8] tensor with format [x, y, z, w, x, y, z, gripper]
                  (i.e. position (3) + quaternion (4) + gripper (1)).
    Returns:
        [B, 10] tensor with format [x, y, z, r6d0, r6d1, r6d2, r6d3, r6d4, r6d5, gripper].
    """
    xyz   = tcp_pose[:, 0:3]          # [B,3]
    quat  = tcp_pose[:, 3:7]          # [B,4] assumed order: (w, x, y, z)
    gripper = tcp_pose[:, 7]          # [B]
    
    # quaternion -> rotation matrix
    R = quaternion_to_matrix(quat)    # [B,3,3]
    # Extract 6D representation from the first two columns
    r6d = matrix_to_6d(R)             # [B,6]
    
    # Concatenate: [x,y,z] + [r6d (6)] + [gripper] => [B,10]
    state_10d = torch.cat([xyz, r6d, gripper.unsqueeze(-1)], dim=-1)
    return state_10d

def euler_to_matrix(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) in ZYX convention to a 3x3 rotation matrix.
    roll, pitch, yaw: shape [B] (batch)
    Returns: shape [B, 3, 3]
    """
    sr, cr = torch.sin(roll),  torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw),   torch.cos(yaw)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr

    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    # Stack into [B, 3, 3]
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)
    return R

def rpy_to_rot6d(tcp_pose: torch.Tensor) -> torch.Tensor:
    """
    tcp_pose: shape [B, 8] => [x, y, z, roll, pitch, yaw, pad, gripper]
    returns: shape [B, 10] => 
             [x, y, z, r6d0, r6d1, r6d2, r6d3, r6d4, r6d5, gripper]
    """
    x     = tcp_pose[:, 0]
    y     = tcp_pose[:, 1]
    z     = tcp_pose[:, 2]
    roll  = tcp_pose[:, 3]
    pitch = tcp_pose[:, 4]
    yaw   = tcp_pose[:, 5]
    pad   = tcp_pose[:, 6]
    gr    = tcp_pose[:, 7]  # gripper

    # Convert RPY -> 3x3 matrix -> 6D
    R = euler_to_matrix(roll, pitch, yaw)   # [B, 3, 3]
    r6d = matrix_to_6d(R)                   # [B, 6]

    # Combine into 10D => [x, y, z, r6d(6), gripper]
    xyz = tcp_pose[:, 0:3]                 # [B, 3]
    xyz_r6d = torch.cat([xyz, r6d], dim=-1)  # [B, 9]
    state_10d = torch.cat([xyz_r6d, gr.unsqueeze(-1)], dim=-1)  # [B, 10]
    return state_10d