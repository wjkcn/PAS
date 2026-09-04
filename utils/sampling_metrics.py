"""
Four sampling quality metrics for point cloud sampling evaluation.
Inspired by SAMBLE (CVPR 2025) experimental methodology.

Metrics:
  1. Coverage      — fraction of surface area covered by sampled centers
  2. Uniformity    — NUC (Normalized Uniformity Coefficient), closer to 1 = more uniform
  3. Edge Preserv. — ratio of high-curvature points captured vs. full cloud
  4. Anomaly Hit   — % of anomaly-guided centers falling in GT anomalous regions
"""

import numpy as np
from scipy.spatial import KDTree, cKDTree


def compute_sampling_metrics(sampled_xyz, full_xyz, gt_mask_2d=None,
                             center_yx=None, H=224, W=224, anomaly_weights=None):
    """
    Compute all four sampling quality metrics.

    Args:
        sampled_xyz:    [N, 3] sampled center positions (numpy)
        full_xyz:       [M, 3] full point cloud (numpy)
        gt_mask_2d:     [H, W] ground-truth anomaly mask (0=normal, 1=anomaly)
        center_yx:      [N, 2] (row, col) pixel positions of centers in 224x224 grid
        H, W:           image dimensions
        anomaly_weights:[N] per-center anomaly scores, higher = more anomalous

    Returns:
        dict with keys: coverage, uniformity, edge_preservation, anomaly_hit_rate
    """
    metrics = {}

    # 1. Coverage
    metrics['coverage'] = _compute_coverage(sampled_xyz, full_xyz)

    # 2. Uniformity (NUC)
    metrics['uniformity'] = _compute_uniformity(sampled_xyz, full_xyz)

    # 3. Edge Preservation
    metrics['edge_preservation'] = _compute_edge_preservation(sampled_xyz, full_xyz)

    # 4. Anomaly Hit Rate
    if gt_mask_2d is not None and center_yx is not None and anomaly_weights is not None:
        metrics['anomaly_hit_rate'] = _compute_anomaly_hit_rate(
            center_yx, gt_mask_2d, anomaly_weights, H, W)
    else:
        metrics['anomaly_hit_rate'] = None

    return metrics


def _compute_coverage(sampled_xyz, full_xyz, voxel_size=0.02):
    """
    Coverage: fraction of occupied surface voxels that contain >=1 sampled point.
    Range: [0, 1], higher = better coverage.

    Uses voxel grid on normalized point cloud to estimate surface coverage.
    """
    if len(full_xyz) < 10:
        return 0.0

    # Normalize to unit cube
    all_pts = np.concatenate([sampled_xyz, full_xyz], axis=0)
    center = all_pts.mean(axis=0)
    scale = np.max(np.linalg.norm(all_pts - center, axis=1)) + 1e-8
    norm_full = (full_xyz - center) / scale
    norm_sampled = (sampled_xyz - center) / scale

    vs = voxel_size
    full_voxels = set()
    for pt in norm_full:
        vx = tuple((pt / vs).astype(np.int32))
        full_voxels.add(vx)

    sampled_voxels = set()
    for pt in norm_sampled:
        vx = tuple((pt / vs).astype(np.int32))
        sampled_voxels.add(vx)

    if len(full_voxels) == 0:
        return 0.0

    covered = len(sampled_voxels & full_voxels)
    return covered / len(full_voxels)


def _compute_uniformity(sampled_xyz, full_xyz):
    """
    NUC: ratio of actual avg NN distance to ideal uniform NN distance.
    Range: [0, inf), 1.0 = perfectly uniform, <1 = clustered, >1 = sparse.
    """
    N = len(sampled_xyz)
    if N < 3:
        return 0.0

    # Normalize
    all_pts = np.concatenate([sampled_xyz, full_xyz], axis=0)
    center = all_pts.mean(axis=0)
    scale = np.max(np.linalg.norm(all_pts - center, axis=1)) + 1e-8
    norm_pts = (sampled_xyz - center) / scale

    # Actual average NN distance
    tree = cKDTree(norm_pts)
    dists, _ = tree.query(norm_pts, k=2)
    nn_dists = dists[:, 1]  # skip self
    actual_nn = np.mean(nn_dists)

    # Ideal NN distance for uniform sampling on sphere of radius ~1
    # Surface area of unit sphere = 4*pi. Average area per point = 4*pi/N
    # Average NN distance ≈ sqrt(area_per_point / pi) = sqrt(4/N) = 2/sqrt(N)
    ideal_nn = 2.0 / np.sqrt(N)

    if ideal_nn < 1e-8:
        return 0.0

    nuc = actual_nn / (ideal_nn + 1e-8)

    # Normalize to [0, 1] where 1 is best (nuc close to 1)
    # score = exp(-|nuc - 1|) → 1 when nuc=1, decays when far from 1
    uniformity_score = np.exp(-abs(nuc - 1.0))
    return float(uniformity_score)


def _compute_edge_preservation(sampled_xyz, full_xyz, k=10):
    """
    Edge Preservation: ratio of curvature-weighted sampled points vs full cloud.
    Estimates local curvature via PCA eigenvalues.

    Range: [0, inf), 1.0 = sampled points capture same level of geometric detail
    as the full point cloud. >1.0 = sampled points are biased toward edges.
    """
    if len(sampled_xyz) < k + 1 or len(full_xyz) < k + 1:
        return 1.0

    def mean_curvature_estimate(pts, k=k):
        """Estimate mean curvature at each point using PCA eigenvalues."""
        tree = cKDTree(pts)
        curvatures = []
        sample_size = min(500, len(pts))
        indices = np.random.choice(len(pts), sample_size, replace=False)
        for i in indices:
            _, knn_idx = tree.query(pts[i], k=k+1)
            knn_idx = knn_idx[1:]  # skip self
            neighbors = pts[knn_idx]
            centered = neighbors - neighbors.mean(axis=0)
            if len(centered) < 3:
                curvatures.append(0.0)
                continue
            cov = centered.T @ centered / (len(centered) - 1)
            eigenvalues = np.linalg.svd(cov, compute_uv=False)
            if len(eigenvalues) >= 3:
                # curvature ≈ λ3 / (λ1 + λ2 + λ3)
                curv = eigenvalues[2] / (eigenvalues.sum() + 1e-8)
                curvatures.append(float(curv))
            else:
                curvatures.append(0.0)
        return np.mean(curvatures) if curvatures else 0.0

    full_curv = mean_curvature_estimate(full_xyz)
    sampled_curv = mean_curvature_estimate(sampled_xyz)

    if full_curv < 1e-8:
        return 1.0

    return float(sampled_curv / full_curv)


def _compute_anomaly_hit_rate(center_yx, gt_mask, anomaly_weights, H=224, W=224):
    """
    Anomaly Hit Rate: among top-K (by anomaly weight) sampled centers,
    what fraction fall inside GT anomalous regions?

    Range: [0, 1], higher = better anomaly localization.
    """
    if center_yx is None or gt_mask is None or anomaly_weights is None:
        return None

    N = len(center_yx)
    if N < 10:
        return None

    # Top 20% most anomalous centers
    K = max(5, int(N * 0.2))
    top_indices = np.argsort(anomaly_weights)[-K:]
    top_yx = center_yx[top_indices].astype(np.int32)

    # Clamp to valid range
    top_yx[:, 0] = np.clip(top_yx[:, 0], 0, H - 1)
    top_yx[:, 1] = np.clip(top_yx[:, 1], 0, W - 1)

    hits = 0
    for y, x in top_yx:
        if gt_mask[y, x] > 0:
            hits += 1

    return hits / K
