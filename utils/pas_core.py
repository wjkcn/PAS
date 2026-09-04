"""
PAS Core: k-center greedy coreset for memory bank compression.
Unified anomaly prior computation for PAS sampling.

Paper definition (per_image_mean):
    f_mean = mean of all spatial features in the current image
    score_i = 1 - cosine_similarity(f_i, f_mean)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def determine_prior_type(global_center):
    """Return the effective prior type based on actual computation mode.

    Returns:
        "per_image_mean" if global_center is None (paper-compliant).
        "global_normal_center" if global_center is provided (DEPRECATED).
    """
    return "global_normal_center" if global_center is not None else "per_image_mean"


def compute_feature_anomaly_score(features, tau=0.5, global_center=None,
                                   paper_protocol=False):
    """Legacy anomaly-prior implementation. Not used for final paper experiments.

    Paper results use compute_anomaly_scores from backbones/pas_sampler.py instead,
    which has no tau parameter and always uses per_image_mean.

    This function supports both per_image_mean (if global_center=None) and
    global_normal_center (if global_center is provided) modes.

    NOTE — the 'tau' parameter here is a legacy score-clipping factor
    (tau=0.5 means clamp(raw/0.5, 0, 1)). It is NOT the same as the
    PAS sampling tau=0.6 in PASSampler, which controls the fraction of
    anomaly-guided centers in hybrid sampling.

    Args:
        features: [B, N, C] feature tensor, or [B, C, H, W] for 2D features
        tau: legacy score normalization via clamp(raw/tau, 0, 1).
            Default 0.5. NOT related to PAS sampling tau (which is 0.6).
        global_center: [1, 1, C] normal-set mean center.
            DEPRECATED for paper results. Paper results use None (per_image_mean).
        paper_protocol: if True, raises ValueError when global_center is provided.

    Returns:
        anomaly_scores: [B, N] per-point anomaly scores (0-1, higher = anomalous)
        raw_scores: [B, N] un-normalized cosine distance scores
    """
    if paper_protocol and global_center is not None:
        raise ValueError(
            "Paper experiments must use per_image_mean prior. "
            "Do not pass global_center when paper_protocol=True."
        )

    if features.dim() == 4:
        B, C, H, W = features.shape
        features = features.reshape(B, C, -1).permute(0, 2, 1)

    B, N, C_feat = features.shape

    if global_center is not None:
        global_center = global_center.to(features.device)
        cos_sim = F.cosine_similarity(features, global_center, dim=-1)
    else:
        mean_feat = features.mean(dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(features, mean_feat, dim=-1)

    raw_scores = 1.0 - cos_sim  # [B, N]

    if tau > 0:
        anomaly_scores = torch.clamp(raw_scores / tau, 0, 1)
    else:
        anomaly_scores = raw_scores

    return anomaly_scores, raw_scores


def k_center_greedy_coreset(features, fraction, device='cuda'):
    """
    K-Center Greedy coreset selection.
    
    Args:
        features: [N, C] feature matrix on CPU
        fraction: float, target fraction of features to keep
        device: torch device
    
    Returns:
        coreset_indices: [K] long tensor of selected indices
    """
    n_samples = features.size(0)
    target_size = max(1, int(round(n_samples * fraction)))
    
    if target_size >= n_samples:
        return features.to(device)
    
    features = features.to(device)
    
    # Initialize with first point
    selected = torch.zeros(target_size, dtype=torch.long, device=device)
    min_dist = torch.full((n_samples,), float('inf'), device=device)
    
    # Pick first center randomly
    perm = torch.randperm(n_samples, device=device)
    selected[0] = perm[0]
    min_dist = torch.min(min_dist, torch.norm(
        features - features[selected[0]].unsqueeze(0), dim=1)**2)
    
    pbar = tqdm(range(1, target_size), desc='  Coreset Greedy 采样中', leave=False)
    for i in pbar:
        # Pick farthest point from current set
        _, farthest = torch.max(min_dist, dim=0)
        selected[i] = farthest
        # Update distances
        new_dist = torch.norm(
            features - features[farthest].unsqueeze(0), dim=1)**2
        min_dist = torch.min(min_dist, new_dist)
    
    coreset = features[selected]
    print(f"✅ Coreset 构建完毕，最终保留点数: {target_size}")
    return selected.cpu()
