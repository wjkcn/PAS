"""Shim: pointnet2_ops using local PyTorch implementations (no CUDA build needed)."""
import torch
from models.pointnet2_utils import farthest_point_sample, index_points


def furthest_point_sample(data, number):
    """Compatibility wrapper: CUDA furthest_point_sample → local farthest_point_sample.
    Args: data [B, N, 3], number: int
    Returns: indices [B, number]
    """
    return farthest_point_sample(data, number)


def gather_operation(data, idx):
    """Compatibility wrapper: CUDA gather_operation → local index_points.
    Args: data [B, C, N], idx [B, S]
    Returns: gathered [B, C, S]
    """
    B, C, N = data.shape
    data_transposed = data.transpose(1, 2).contiguous()  # [B, N, C]
    gathered = index_points(data_transposed, idx)  # [B, S, C]
    return gathered.transpose(1, 2).contiguous()  # [B, C, S]
