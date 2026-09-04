"""Voxel Downsampling (VDS) — pure geometric, independent of FPS.

Partitions 3D space into voxels and picks one representative per occupied voxel.
Unlike Grid Sampling (uniform grid cells), VDS adapts to the point distribution
by only considering occupied voxels.

Key property for the ablation study: VDS is a genuinely different algorithm from
FPS/k-Center Greedy, making it a valid independent geometric baseline.
"""

import torch
import torch.nn as nn


def voxel_downsampling(xyz, npoint):
    """Voxel Downsampling: partition points by 3D voxel grid, pick one
    representative per occupied voxel (closest to voxel centroid).

    xyz: [B, N, 3]
    npoint: target number of points

    Returns: indices [B, npoint]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

    all_indices = []
    for b in range(B):
        p = xyz[b]  # [N, 3]
        p_min = p.min(dim=0)[0]
        p_max = p.max(dim=0)[0]
        extent = p_max - p_min

        # Voxel size: target ~npoint occupied voxels
        volume = extent.prod().clamp(min=1e-9)
        voxel_size = (volume / npoint) ** (1.0 / 3.0)
        voxel_size = max(voxel_size.item(), 1e-6)

        # Quantize to voxel coordinates
        vc = ((p - p_min.unsqueeze(0)) / voxel_size).long()  # [N, 3]

        # Hash voxel coordinates to 1D
        max_c = vc.max(dim=0)[0].clamp(min=0) + 1
        hash_val = vc[:, 0] + vc[:, 1] * max_c[0] + vc[:, 2] * max_c[0] * max_c[1]

        unique_h, inverse = torch.unique(hash_val, return_inverse=True)
        n_voxels = len(unique_h)

        # For each voxel, pick the point closest to the centroid
        representatives = []
        for v in range(n_voxels):
            mask = inverse == v
            pt_idx = torch.where(mask)[0]
            if len(pt_idx) == 1:
                representatives.append(pt_idx[0].item())
            else:
                voxel_pts = p[pt_idx]
                centroid = voxel_pts.mean(dim=0)
                dists = torch.sum((voxel_pts - centroid.unsqueeze(0)) ** 2, dim=-1)
                representatives.append(pt_idx[torch.argmin(dists)].item())

        # Adjust to exact npoint
        if len(representatives) > npoint:
            perm = torch.randperm(len(representatives), device=device)[:npoint]
            representatives = [representatives[i] for i in perm.tolist()]
        elif len(representatives) < npoint:
            selected_set = set(representatives)
            remaining = [i for i in range(N) if i not in selected_set]
            if remaining:
                extra_n = min(npoint - len(representatives), len(remaining))
                extra_idx = torch.randperm(len(remaining), device=device)[:extra_n]
                for idx in extra_idx:
                    representatives.append(remaining[idx.item()])

        all_indices.append(torch.tensor(representatives[:npoint], dtype=torch.long, device=device))

    return torch.stack(all_indices)


class VoxelSampler(nn.Module):
    """Voxel Downsampling — pure geometric baseline independent of FPS."""

    def __init__(self):
        super().__init__()

    def forward(self, xyz, npoint):
        return voxel_downsampling(xyz, npoint)
