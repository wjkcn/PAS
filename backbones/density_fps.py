"""Density-Aware FPS — pure geometric sampling with local density compensation.

Modifies FPS to account for local point density variations.
In dense regions, effective distance is reduced to avoid over-sampling.
In sparse regions, effective distance is amplified to ensure coverage.

Zero parameters, zero training, O(N) density estimation via voxel binning.
"""

import torch


def _compute_density_voxel(xyz, voxel_size=0.05):
    """Compute per-point density via voxel binning (O(N)).

    xyz: [B, N, 3], range roughly -1 to 1.
    Returns density [B, N] — higher = more points in same voxel.
    """
    B, N, D = xyz.shape
    device = xyz.device

    # Quantize to voxel grid
    voxel_idx = (xyz / voxel_size).long()  # [B, N, 3]
    # Shift to non-negative
    vmin = voxel_idx.amin(dim=1, keepdim=True)  # [B, 1, 3]
    voxel_idx = voxel_idx - vmin
    vmax = voxel_idx.amax(dim=1, keepdim=True) + 1  # [B, 1, 3]

    # Flatten voxel coords to 1D index
    stride_x = vmax[:, 0, 1] * vmax[:, 0, 2]  # [B]
    stride_y = vmax[:, 0, 2]  # [B]
    # Broadcast [B] * [B, N] safely by unsqueezing to [B, 1]
    flat_idx = (voxel_idx[:, :, 0].float() * stride_x.unsqueeze(1) +
                voxel_idx[:, :, 1].float() * stride_y.unsqueeze(1) +
                voxel_idx[:, :, 2].float()).long()  # [B, N]

    # Count per voxel
    density = torch.zeros(B, N, device=device)
    for b in range(B):
        _, inv, counts = flat_idx[b].unique(return_inverse=True, return_counts=True)
        density[b] = counts[inv].float()

    return density


def _native_fps_with_density(xyz, npoint, density, alpha=0.5):
    """FPS modified by density weight. Higher density → shorter effective distance.

    xyz: [B, N, 3], density: [B, N], alpha in [0, 1].
    Returns [B, npoint] indices.
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Normalize density to [0.2, 1.8] range so alpha modulates around 1.0
    d_min = density.amin(dim=1, keepdim=True).clamp(min=1)
    d_max = density.amax(dim=1, keepdim=True).clamp(min=d_min + 1)
    d_norm = (density - d_min) / (d_max - d_min)  # [0, 1]
    # Rescale: at alpha=0 → all 1.0 (vanilla FPS); at alpha=1 → 0.3 to 1.7
    weight = 1.0 + alpha * (d_norm * 1.4 - 0.7)  # [0.3, 1.7] at alpha=1

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        # Apply density weight: dense regions appear "closer" → FPS avoids them
        weighted_dist = distance / weight
        farthest = torch.max(weighted_dist, -1)[1]

    return centroids


def density_aware_fps(xyz, npoint, voxel_size=0.05, alpha=0.5):
    """Density-Aware FPS: accounts for local point density in farthest-point selection.

    Standard FPS treats Euclidean distance as the only selection criterion.
    This biases sampling toward dense point clusters (e.g., complex surfaces with
    many scan points). DA-FPS compensates by making dense regions appear "closer"
    (smaller effective distance), encouraging FPS to also cover sparser regions.

    Args:
        xyz: [B, N, 3] point coordinates
        npoint: target number of points
        voxel_size: size of voxel bins for density estimation (default 0.05)
        alpha: density influence strength (0 = vanilla FPS, 1 = full compensation)

    Returns:
        indices [B, npoint]
    """
    N = xyz.size(1)
    if N <= npoint:
        return torch.arange(N, device=xyz.device).unsqueeze(0).expand(xyz.size(0), N)

    density = _compute_density_voxel(xyz, voxel_size=voxel_size)
    return _native_fps_with_density(xyz, npoint, density, alpha=alpha)
