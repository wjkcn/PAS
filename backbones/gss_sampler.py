"""Non-random sampling baselines for comparison with PAS.

Implements two non-random sampling methods that serve as controls:
- k-Center Greedy (GSS): iteratively picks the point farthest from all selected points
- Grid Sampling: divides 3D bounding box into grid cells, picks one point per cell

These are geometric-only methods (no 2D semantics), isolating the contribution
of PAS's cross-modal anomaly guidance.
"""

import torch
import torch.nn as nn


def kcenter_greedy_sampling(xyz, npoint):
    """k-Center Greedy sampling: iteratively selects points to maximize coverage.

    A simpler variant of FPS: start from a random point, then greedily add
    the point farthest from all currently selected points.

    xyz: [B, N, 3]
    npoint: int

    Returns: indices [B, npoint]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

    indices = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Seed: random first point per batch
    first = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    indices[:, 0] = first
    batch_idx = torch.arange(B, device=device)

    for i in range(npoint):
        cur = indices[:, i]
        cur_xyz = xyz[batch_idx, cur, :].view(B, 1, 3)
        dist = torch.sum((xyz - cur_xyz) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        if i < npoint - 1:
            indices[:, i + 1] = torch.argmax(distance, dim=-1)

    return indices


def grid_sampling(xyz, npoint):
    """Uniform Grid Sampling: divide 3D space into grid cells, pick closest to cell center.

    Number of grid cells ≈ npoint. For dimensions where the bounding box is degenerate
    (max == min), use single slice.

    xyz: [B, N, 3]
    npoint: int

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

        # Determine grid resolution: target ~npoint cells
        # grid_x * grid_y * grid_z ≈ npoint, with aspect ratio preserved
        total_extent = extent.sum().clamp(min=1e-6)
        ratios = extent / total_extent
        # Allocate cells per dimension proportionally
        cells_per_dim = (ratios * npoint).clamp(min=1).int()
        # Scale so product ≈ npoint
        product = (cells_per_dim[0] * cells_per_dim[1] * cells_per_dim[2]).item()
        while product > npoint * 2:
            # Reduce largest dim
            d = cells_per_dim.argmax().item()
            cells_per_dim[d] = max(1, cells_per_dim[d] - 1)
            product = (cells_per_dim[0] * cells_per_dim[1] * cells_per_dim[2]).item()
        while product < npoint // 2:
            d = cells_per_dim.argmin().item()
            cells_per_dim[d] += 1
            product = (cells_per_dim[0] * cells_per_dim[1] * cells_per_dim[2]).item()

        collected_indices = []
        # Generate grid cells
        for gx in range(cells_per_dim[0].item()):
            for gy in range(cells_per_dim[1].item()):
                for gz in range(cells_per_dim[2].item()):
                    # Cell boundaries
                    low = torch.stack([
                        p_min[0] + extent[0] * gx / max(1, cells_per_dim[0] - 1) if cells_per_dim[
                                                                                          0] > 1 else p_min[0],
                        p_min[1] + extent[1] * gy / max(1, cells_per_dim[1] - 1) if cells_per_dim[
                                                                                          1] > 1 else p_min[1],
                        p_min[2] + extent[2] * gz / max(1, cells_per_dim[2] - 1) if cells_per_dim[
                                                                                          2] > 1 else p_min[2],
                    ])
                    high = torch.stack([
                        p_min[0] + extent[0] * (gx + 1) / cells_per_dim[0] if cells_per_dim[0] > 1 else p_max[0],
                        p_min[1] + extent[1] * (gy + 1) / cells_per_dim[1] if cells_per_dim[1] > 1 else p_max[1],
                        p_min[2] + extent[2] * (gz + 1) / cells_per_dim[2] if cells_per_dim[2] > 1 else p_max[2],
                    ])
                    center = (low + high) / 2.0
                    # Find point closest to cell center
                    dist_to_center = torch.sum((p - center.unsqueeze(0)) ** 2, dim=-1)
                    closest = torch.argmin(dist_to_center)
                    collected_indices.append(closest)

        if len(collected_indices) == 0:
            # Fallback: random
            collected_indices = torch.randperm(N, device=device)[:npoint].tolist()

        # Deduplicate and subsample to exact npoint
        collected = torch.tensor(collected_indices, dtype=torch.long, device=device)
        indices_b = torch.unique(collected)
        if indices_b.size(0) > npoint:
            perm = torch.randperm(indices_b.size(0), device=device)[:npoint]
            indices_b = indices_b[perm]
        elif indices_b.size(0) < npoint:
            selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
            selected_mask[indices_b] = True
            remaining = torch.nonzero(~selected_mask).squeeze(-1)
            if remaining.numel() > 0:
                extra = remaining[torch.randperm(len(remaining), device=device)[:npoint - indices_b.size(0)]]
                indices_b = torch.cat([indices_b, extra])

        all_indices.append(indices_b[:npoint])

    return torch.stack(all_indices)


class GSSSampler(nn.Module):
    """k-Center Greedy sampler — non-random geometric baseline."""

    def __init__(self):
        super().__init__()

    def forward(self, xyz, npoint):
        return kcenter_greedy_sampling(xyz, npoint)


class GridSampler(nn.Module):
    """Uniform Grid sampler — non-random geometric baseline."""

    def __init__(self):
        super().__init__()

    def forward(self, xyz, npoint):
        return grid_sampling(xyz, npoint)
