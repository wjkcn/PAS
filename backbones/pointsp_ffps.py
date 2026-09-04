"""PointSP: Filtered FPS (FFPS) — filters outliers before FPS via isolation rate.

From Li et al. (IJCAI 2025): "Enhancing Sampling Protocol for Point Cloud
Classification Against Corruptions."

Key idea: FPS is vulnerable to outliers because Euclidean-distance-based
selection naturally picks them. FFPS computes per-point "isolation rate" and
filters out the most isolated points (potential outliers) before running FPS.

Isolation rate w'_i = fraction of k neighbors whose distance ≥ median radius.
Binary mask: keep if w'_i ≤ omega (default omega=0.95, filters top 5%).

For anomaly detection, this is the OPPOSITE of PAS: FFPS removes outliers
(anomalous points), while PAS preferentially samples them.
"""

import torch


def compute_isolation_rate(xyz, k=20, n_anchors=2048):
    """Per-point isolation rate w'_i via anchor-based KNN.

    Computes the fraction of each point's k nearest neighbors whose
    distance exceeds the global median neighborhood radius.

    Uses random anchor points with KNN then nearest-anchor interpolation
    for O(A*N) efficiency instead of O(N²).

    Args:
        xyz: [B, N, 3]
        k: number of nearest neighbors (paper default 20)
        n_anchors: number of random anchor points for efficiency

    Returns:
        isolation_rate: [B, N] in [0, 1] (higher = more isolated)
        binary_mask: [B, N] bool (True = keep, False = filter out)
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= k:
        return torch.zeros(B, N, device=device), torch.ones(B, N, dtype=torch.bool, device=device)

    n_anchors = min(n_anchors, N)
    results_rate = []
    results_mask = []

    for b in range(B):
        anchor_idx = torch.randperm(N, device=device)[:n_anchors]
        points = xyz[b]
        anchors = points[anchor_idx]

        # KNN from anchors to all points: [A, N]
        dist = torch.cdist(anchors.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
        nn_dist, _ = dist.topk(k, dim=-1, largest=False)  # [A, k]

        # r_i = max distance to k neighbors per anchor: [A]
        r_i = nn_dist.max(dim=1)[0]

        # r_bar = median of all radii: scalar
        r_bar = r_i.median()

        # w'_i per anchor = fraction of neighbors with dist >= r_bar: [A]
        w_prime = (nn_dist >= r_bar).float().mean(dim=1)  # [A]

        # Nearest-anchor interpolation to all points
        min_dist_anchor = dist.argmin(dim=0)  # [N]
        iso_rate = w_prime[min_dist_anchor]  # [N]

        results_rate.append(iso_rate)

    iso_rate = torch.stack(results_rate, dim=0)  # [B, N]
    return iso_rate


def filtered_fps(xyz, npoint, k=20, omega=0.95, n_anchors=2048):
    """Filtered FPS: remove top-(1-omega) most isolated points, then FPS.

    From PointSP (Li et al., IJCAI 2025), Equation 3.

    Args:
        xyz: [B, N, 3]
        npoint: target number of points to sample
        k: neighbors for isolation rate (paper default 20)
        omega: quantile threshold (paper default 0.95)
        n_anchors: random anchors for efficiency

    Returns:
        indices [B, npoint] — selected point indices in original xyz
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

    iso_rate = compute_isolation_rate(xyz, k=k, n_anchors=n_anchors)  # [B, N]

    selected = torch.zeros(B, npoint, dtype=torch.long, device=device)

    for b in range(B):
        ir = iso_rate[b]  # [N]

        # Binary mask: keep if isolation_rate <= omega
        # omega=0.95 → keep bottom 95%, filter top 5% most isolated
        threshold = ir.quantile(omega)
        keep_mask = ir <= threshold
        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)

        n_keep = len(keep_indices)

        if n_keep <= npoint:
            # Not enough points after filtering — take all kept + pad with
            # lowest-isolation-rate filtered points
            selected[b, :n_keep] = keep_indices
            if n_keep < npoint:
                filtered = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
                # Sort filtered by isolation rate (ascending = least isolated first)
                filtered_sorted = filtered[ir[filtered].argsort()]
                pad_n = min(npoint - n_keep, len(filtered))
                selected[b, n_keep:n_keep + pad_n] = filtered_sorted[:pad_n]
        else:
            # Run FPS on kept points
            xyz_kept = xyz[b, keep_indices]  # [n_keep, 3]

            # Standard FPS
            fps_indices = _fps(xyz_kept, npoint)  # [npoint] indices into keep_indices
            selected[b] = keep_indices[fps_indices]

    return selected


def _fps(xyz, npoint):
    """Standard Farthest Point Sampling on single batch.

    xyz: [N, 3]
    Returns indices [npoint]
    """
    N = xyz.shape[0]
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device)

    selected = torch.zeros(npoint, dtype=torch.long, device=device)
    dist = torch.ones(N, device=device) * 1e10

    farthest = 0  # start from first point
    for i in range(npoint):
        selected[i] = farthest
        centroid = xyz[farthest].unsqueeze(0)  # [1, 3]
        d = torch.sum((xyz - centroid) ** 2, dim=1)  # [N]
        dist = torch.min(dist, d)
        farthest = dist.argmax().item()

    return selected
