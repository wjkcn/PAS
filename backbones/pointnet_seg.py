"""Vanilla PointNet segmentation backbone with PAS sampling support.

The most basic 3D deep learning architecture — shared MLP per-point + global max pool.
Integrates PAS sampling at the bottleneck: after the first shared MLP stage, points are
selected via PAS (or FPS) before the deeper per-point MLP, then global pooling is tiled
back to all N points and concatenated with early per-point features.

Architecture:
  Shared MLP1 (all N points): 3→64→128
  PAS/FPS sampling: N→512 centers
  Shared MLP2 (512 centers): 128→256→512→1024
  Global Max Pool: 1024-dim
  Tile + Concat(early 128, global 1024) → Conv1d → [B, 64, N]

Interface matches PointNet2SegBackbone for drop-in use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.pas_sampler import PASSampler, _native_fps


def _gather(xyz, idx):
    """Gather points [B, N, 3] by indices [B, K] → [B, K, 3]. Pure PyTorch, no CUDA ops."""
    B, N, _ = xyz.shape
    K = idx.size(1)
    batch_indices = torch.arange(B, device=xyz.device).view(-1, 1).expand(-1, K)
    return xyz[batch_indices, idx.long(), :]


def _gather_features(features, idx):
    """Gather features [B, C, N] by indices [B, K] → [B, C, K]. Pure PyTorch."""
    B, C, N = features.shape
    feat_t = features.permute(0, 2, 1)  # [B, N, C]
    batch_idx = torch.arange(B, device=features.device).view(B, 1)
    return feat_t[batch_idx, idx.long()].permute(0, 2, 1)  # [B, C, K]


class PointNetSegBackbone(nn.Module):
    """Vanilla PointNet adapted for per-point anomaly detection with PAS.

    Args:
        tau: PAS anomaly-guided fraction (default 0.6)
        ncenter: number of centers after sampling (default 512)
    """

    def __init__(self, tau=0.6, ncenter=512):
        super().__init__()
        self.ncenter = ncenter
        self.pas_sampler = PASSampler(tau=tau)
        self._sampling_mode = None

        # Stage 1: per-point MLP (all N points)
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)

        # Stage 2: per-point MLP (K centers only)
        self.conv3 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)

        # Output: combine early per-point + tiled global
        self.out_conv = nn.Conv1d(128 + 1024, 64, 1, bias=False)
        self.out_bn = nn.BatchNorm1d(64)

    def set_sampling_mode(self, mode):
        self._sampling_mode = mode

    @property
    def output_dim(self):
        return 64

    def forward(self, xyz, anomaly_scores=None):
        """Forward pass.

        Args:
            xyz: [B, N, 3] point coordinates
            anomaly_scores: [B, N] anomaly scores, higher = more anomalous

        Returns:
            per_point_feat: [B, 64, N]
            sa_xyzs: [original_xyz, center_xyz]
            sa_feats: [input_feat, stage1_feat, stage2_feat, global_feat]
            sa_indices: [center_indices, None]
        """
        B, N, _ = xyz.shape
        mode = self._sampling_mode or 'auto'
        device = xyz.device

        # Stage 1: per-point MLP on all N points
        input_feat = xyz.transpose(1, 2)  # [B, 3, N]
        f1 = F.relu(self.bn1(self.conv1(input_feat)))   # [B, 64, N]
        f2 = F.relu(self.bn2(self.conv2(f1)))            # [B, 128, N]

        # Sampling
        ncenter = min(self.ncenter, N)

        if mode in ('vds', 'grid', 'dafps', 'curv', 'ffps'):
            if mode == 'vds':
                from backbones.voxel_sampler import voxel_downsampling
                center_idx = voxel_downsampling(xyz, ncenter)
            elif mode == 'grid':
                from backbones.gss_sampler import grid_sampling
                center_idx = grid_sampling(xyz, ncenter)
            elif mode == 'dafps':
                from backbones.density_fps import density_aware_fps
                center_idx = density_aware_fps(xyz, ncenter, voxel_size=0.05, alpha=0.5)
            elif mode == 'curv':
                from backbones.curvature_sampler import curvature_guided_sampling
                center_idx = curvature_guided_sampling(xyz, ncenter, k=32, tau=0.5)
            else:  # ffps
                from backbones.pointsp_ffps import filtered_fps
                center_idx = filtered_fps(xyz, ncenter, k=20, omega=0.95)
        elif mode == 'rs':
            center_idx = torch.stack([
                torch.randperm(N, device=device)[:ncenter] for _ in range(B)
            ], dim=0)
        elif mode == 'pas' or (mode == 'auto' and anomaly_scores is not None):
            center_idx = self.pas_sampler(xyz, ncenter, anomaly_scores)
        else:
            center_idx = _native_fps(xyz, ncenter)

        center_xyz = _gather(xyz, center_idx)
        center_feat = _gather_features(f2, center_idx)  # [B, 128, K]

        # Stage 2: per-point MLP on K centers
        c3 = F.relu(self.bn3(self.conv3(center_feat)))  # [B, 256, K]
        c4 = F.relu(self.bn4(self.conv4(c3)))            # [B, 512, K]
        c5 = F.relu(self.bn5(self.conv5(c4)))            # [B, 1024, K]

        # Global max pool over centers
        global_feat = c5.max(dim=-1, keepdim=True)[0]     # [B, 1024, 1]

        # Combine: tile global to all N points, concat with stage-1 features
        global_tiled = global_feat.expand(-1, -1, N)     # [B, 1024, N]
        concat = torch.cat([f2, global_tiled], dim=1)    # [B, 128+1024, N]
        out = F.relu(self.out_bn(self.out_conv(concat)))  # [B, 64, N]

        sa_xyzs = [xyz, center_xyz]
        sa_feats = [input_feat, f2, center_feat, c5]
        sa_indices = [center_idx, None]

        return out, sa_xyzs, sa_feats, sa_indices
