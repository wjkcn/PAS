"""PAS (Reverse Thought Sampling) — unified sampler module.

Replaces FPS calls in any 3D backbone with anomaly-guided hybrid sampling.
Zero additional parameters, zero training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_is_cuda_ops_available = None


def _detect_cuda_ops():
    global _is_cuda_ops_available
    if _is_cuda_ops_available is None:
        try:
            from pointnet2_ops import pointnet2_utils as _
            _is_cuda_ops_available = True
        except ImportError:
            _is_cuda_ops_available = False
    return _is_cuda_ops_available


def _get_fps_fn():
    if _detect_cuda_ops():
        from pointnet2_ops import pointnet2_utils
        def cuda_fps(xyz, n):
            return pointnet2_utils.furthest_point_sample(xyz, n)
        return cuda_fps
    else:
        return _native_fps


def _native_fps(xyz, n):
    """Pure PyTorch FPS. xyz: [B, N, 3], returns [B, n] indices."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(n):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def compute_anomaly_scores(features_2d, global_center=None):
    """Compute per-point anomaly score via cosine distance.

    features_2d: [B, N, C] 2D features (e.g. DINO patches mapped to points)
    global_center: [1, 1, C] or None (uses batch mean as fallback)

    Returns: distances [B, N] (higher = more anomalous)
    """
    if features_2d.dim() == 4:
        B, C, H, W = features_2d.shape
        features_2d = features_2d.view(B, C, -1).permute(0, 2, 1)

    if global_center is not None:
        global_center = global_center.to(features_2d.device)
        cos_sim = F.cosine_similarity(features_2d, global_center, dim=-1)
    else:
        mean_feat = features_2d.mean(dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(features_2d, mean_feat, dim=-1)

    return 1.0 - cos_sim


class PASSampler(nn.Module):
    """PAS hybrid sampler — drop-in replacement for FPS.

    tau: fraction for anomaly-guided local FPS (default 0.6)
    rand_frac: fraction for random diversity (default 0.1)
    pool_mult: candidate pool size = n_feat * pool_mult (default 3.0)

    The remaining (1 - tau - rand_frac) goes to global FPS for geometry coverage.
    """

    def __init__(self, tau=0.6, rand_frac=0.1, pool_mult=3.0):
        super().__init__()
        self.tau = tau
        self.rand_frac = rand_frac
        self.pool_mult = pool_mult
        self._fps_fn = _get_fps_fn()

    def forward(self, xyz, npoint, anomaly_scores, return_papas=False):
        """
        xyz: [B, N, 3]
        npoint: int, number of centers to output
        anomaly_scores: [B, N], higher = more anomalous
        return_papas: if True, also return (fps_idx, feat_idx, rand_idx) per batch

        Returns: indices [B, npoint]
                 if return_papas: (indices, fps_idx_list, feat_idx_list, rand_idx_list)
        """
        B, N, _ = xyz.shape
        device = xyz.device

        n_feat = min(int(npoint * self.tau), npoint)
        n_rand = min(int(npoint * self.rand_frac), npoint - n_feat)
        n_fps = npoint - n_feat - n_rand

        if N <= npoint:
            idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            if return_papas:
                return idx, [torch.arange(N, device=device)] * B, \
                       [torch.arange(N, device=device)] * B, \
                       [torch.arange(N, device=device)] * B
            return idx

        collected = []
        fps_list, feat_list, rand_list = [], [], []
        for b in range(B):
            available = torch.ones(N, dtype=torch.bool, device=device)
            papas = []

            # A: global FPS for geometric coverage
            if n_fps > 0:
                fps_idx = self._fps_fn(xyz[b:b + 1], n_fps).squeeze(0).long()
                available[fps_idx] = False
                papas.append(fps_idx)
                if return_papas:
                    fps_list.append(fps_idx)

            # B: top-K anomaly pool → local FPS within pool
            scores = anomaly_scores[b].clone()
            scores[~available] = -float('inf')
            pool_size = min(n_feat * int(self.pool_mult), int(available.sum().item()))

            if n_feat > 0:
                if pool_size > n_feat:
                    _, candidates = torch.topk(scores, pool_size)
                    cand_xyz = xyz[b:b + 1, candidates, :]
                    local_sub = self._fps_fn(cand_xyz, n_feat).squeeze(0).long()
                    feat_idx = candidates[local_sub]
                else:
                    _, feat_idx = torch.topk(scores, n_feat)

                available[feat_idx] = False
                papas.append(feat_idx)
                if return_papas:
                    feat_list.append(feat_idx)

            # C: random diversity
            if n_rand > 0:
                remaining = torch.nonzero(available).squeeze(-1)
                rand_idx = remaining[torch.randperm(len(remaining), device=device)[:n_rand]]
                papas.append(rand_idx)
                if return_papas:
                    rand_list.append(rand_idx)

            collected.append(torch.cat(papas, dim=0))

        if return_papas:
            return torch.stack(collected), fps_list, feat_list, rand_list
        return torch.stack(collected)
