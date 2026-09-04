"""Curvature-Based Sampling — pure geometric surface variation guided sampling.

Estimates local surface curvature via eigenvalue decomposition of the local
covariance matrix. Points with high curvature (edges, corners, shape transitions)
are preferentially selected.

This is a pure 3D geometric method — independent of both FPS and 2D semantics.
It serves as a stronger geometric baseline than VDS/Grid for isolating the
contribution of 2D cross-modal guidance in PAS.

Zero parameters, zero training.
"""

import torch


def compute_curvature(xyz, k=32, n_anchors=2048):
    """Estimate per-point curvature from local covariance eigenvalues (vectorized).

    curvature = λ_min / (λ₁ + λ₂ + λ₃) where λ are eigenvalues of the local
    covariance matrix. High curvature → surface edge/corner.

    Uses random anchor points with KNN, then nearest-anchor interpolation.
    All covariance + eigendecomposition is batched for GPU efficiency.

    xyz: [B, N, 3]
    Returns curvature [B, N] in [0, 1/3] (planar → curved).
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= k:
        return torch.zeros(B, N, device=device)

    n_anchors = min(n_anchors, N)
    results = []

    for b in range(B):
        anchor_idx = torch.randperm(N, device=device)[:n_anchors]
        points = xyz[b]  # [N, 3]
        anchors = points[anchor_idx]  # [A, 3]

        # KNN from anchors to all points: [A, N] distances
        dist = torch.cdist(anchors.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
        _, nn_idx = dist.topk(k, dim=-1, largest=False)  # [A, k]

        # Gather all neighbors: [A, k, 3]
        neighbors = points[nn_idx]  # [A, k, 3]

        # Batched covariance: center → [A, k, 3] → cov [A, 3, 3]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # [A, k, 3]
        centered_t = centered.transpose(1, 2)  # [A, 3, k]
        cov = (centered_t @ centered) / (k - 1)  # [A, 3, 3]

        # Ensure symmetry (numerical)
        cov = (cov + cov.transpose(1, 2)) / 2

        # Batched eigenvalue decomposition (ascending order)
        eigvals = torch.linalg.eigvalsh(cov)  # [A, 3]

        # curvature = λ_min / (λ₁ + λ₂ + λ₃), in [0, 1/3]
        curvatures = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-8)  # [A]

        # Nearest-anchor interpolation to all points
        min_dist_anchor = dist.argmin(dim=0)  # [N]
        results.append(curvatures[min_dist_anchor])  # [N]

    return torch.stack(results, dim=0)  # [B, N]


def curvature_guided_sampling(xyz, npoint, k=32, tau=0.5):
    """Curvature-guided hybrid sampling.

    Selects npoint centers by combining:
      - Top-(tau * npoint) highest-curvature points (geometric interest)
      - Remaining (1-tau) * npoint by FPS for spatial coverage

    This mirrors PAS's hybrid structure but replaces DINO semantic guidance
    with pure 3D geometric curvature, isolating the cross-modal contribution.

    Args:
        xyz: [B, N, 3]
        npoint: target number of points
        k: neighbors for curvature estimation
        tau: fraction of points selected by curvature (0 = pure FPS, 1 = pure curvature)

    Returns:
        indices [B, npoint]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

    n_curv = int(npoint * tau)
    n_fps = npoint - n_curv

    curvature = compute_curvature(xyz, k=k)  # [B, N]

    selected = torch.zeros(B, npoint, dtype=torch.long, device=device)
    taken_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        c = curvature[b]  # [N]

        if n_curv > 0:
            # Select top curvature points
            _, top_idx = c.topk(n_curv)
            selected[b, :n_curv] = top_idx
            taken_mask[b, top_idx] = True

        if n_fps > 0:
            # FPS from remaining points
            remaining = (~taken_mask[b]).nonzero(as_tuple=False).squeeze(-1)
            if len(remaining) <= n_fps:
                selected[b, n_curv:n_curv + len(remaining)] = remaining
            else:
                xyz_rem = xyz[b, remaining]
                # Simplified FPS: farthest from already-selected points
                dist_to_selected = torch.ones(len(remaining), device=device) * 1e10
                if n_curv > 0:
                    selected_xyz = xyz[b, selected[b, :n_curv]]
                    d_curv = torch.cdist(xyz_rem.unsqueeze(0), selected_xyz.unsqueeze(0)).squeeze(0)
                    dist_to_selected = d_curv.min(dim=1)[0]

                fps_selected = []
                farthest = dist_to_selected.argmax().item()
                for _ in range(n_fps):
                    fps_selected.append(remaining[farthest])
                    d_new = torch.sum((xyz_rem - xyz_rem[farthest].unsqueeze(0)) ** 2, dim=1)
                    dist_to_selected = torch.min(dist_to_selected, d_new)
                    farthest = dist_to_selected.argmax().item()

                selected[b, n_curv:] = torch.tensor(fps_selected, dtype=torch.long, device=device)

    return selected
