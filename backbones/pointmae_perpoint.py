"""Point-MAE backbone with Feature Propagation for per-point output.

Wraps Point-MAE's Group + Encoder modules and adds Feature Propagation (FP)
layers to upsample from center points to all original points.

This allows Point-MAE to serve as the 3D backbone in the per-point anomaly
detection pipeline (same scoring head as PointNet++), enabling cross-architecture
comparison of PAS vs FPS.

Architecture:
  Input [B, N, 3]
    → Group (FPS/PAS, K=1024) → grouped [B, 1024, 128, 3] + centers [B, 1024, 3]
    → Encoder → center_features [B, 1024, 384]
    → FP2 (1024→allN): interpolate + refine
    → FP1 (N→N): final refinement
  Output: [B, 64, N] per-point features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import Group, Encoder
from backbones.pointnet2_seg import PointNetFP
from backbones.pas_sampler import PASSampler


class PointMAEPerPointBackbone(nn.Module):
    """Point-MAE backbone adapted for per-point feature extraction.

    Uses Point-MAE's Group + Encoder (center-point features), then Feature
    Propagation to upsample to all original points.

    Sampling modes: 'fps' (pure geometric), 'pas' (anomaly-guided hybrid).
    """

    def __init__(self, group_size=128, num_group=1024, encoder_channel=384, tau=0.6):
        super().__init__()

        self.num_group = num_group
        self.group_size = group_size
        self.encoder_channel = encoder_channel

        # Point-MAE modules
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_channel)

        # Feature Propagation: upsample from G centers → all N original points
        # FP2: G → N (3-NN interpolation from centers + Conv1d refine)
        self.fp2 = PointNetFP(mlp=[encoder_channel + 3, 256, 128])
        # Refine: per-point Conv1d refinement with xyz skip connection
        self.refine = nn.Sequential(
            nn.Conv1d(128 + 3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self._sampling_mode = None

    def set_sampling_mode(self, mode):
        self._sampling_mode = mode
        self.group_divider._sampling_mode = mode

    @property
    def output_dim(self):
        return 64

    def forward(self, xyz, anomaly_scores=None):
        """
        Args:
            xyz: [B, N, 3] point cloud
            anomaly_scores: [B, N] optional anomaly scores for PAS sampling

        Returns:
            per_point_feat: [B, 64, N]
            center_xyz: [B, G, 3] — sampled centers (for density weighting)
            sa_xyzs: [xyz_original, xyz_centers] (for compatibility)
            sa_indices: [center_indices] (for compatibility)
        """
        B, N, _ = xyz.shape

        # Step 1: Group (FPS or PAS sampling)
        neighborhood, center_xyz, ori_idx, center_idx = self.group_divider(xyz, anomaly_scores)

        # Step 2: Encoder → per-center features
        center_features = self.encoder(neighborhood)  # [B, G, encoder_channel]

        # Step 3: Feature Propagation G → all N
        center_feat_t = center_features.transpose(1, 2)  # [B, 384, G]
        xyz_feat_input = xyz.transpose(1, 2).contiguous()  # [B, 3, N]

        # FP2: 3-NN interpolate from G centers to all N points + Conv1d refine
        fp2_feat = self.fp2(xyz, center_xyz, xyz_feat_input, center_feat_t)  # [B, 128, N]

        # Refine: per-point Conv1d with xyz skip connection
        refine_input = torch.cat([fp2_feat, xyz_feat_input], dim=1)  # [B, 128+3, N]
        per_point_feat = self.refine(refine_input)  # [B, 64, N]

        sa_xyzs = [xyz, center_xyz]
        sa_indices = [center_idx]

        return per_point_feat, sa_xyzs, [fp2_feat, per_point_feat], sa_indices

    def _interpolate_to_all(self, unknown_xyz, known_xyz, known_feat):
        """3-NN interpolation from known points (G centers) to unknown points (all N).

        unknown_xyz: [B, n, 3]
        known_xyz: [B, m, 3]
        known_feat: [B, C, m]

        Returns: [B, C, n]
        """
        B, n, _ = unknown_xyz.shape
        C = known_feat.size(1)
        m = known_xyz.size(1)

        # Compute pairwise distances and find 3 nearest neighbors
        dist = torch.cdist(unknown_xyz, known_xyz)  # [B, n, m]
        dist3, idx3 = torch.topk(dist, 3, dim=-1, largest=False)
        dist3 = torch.clamp(dist3, min=1e-10)
        weight = 1.0 / dist3
        weight = weight / weight.sum(dim=-1, keepdim=True)  # [B, n, 3]

        # Gather features
        idx3_flat = idx3 + torch.arange(B, device=idx3.device).view(-1, 1, 1) * m
        known_flat = known_feat.transpose(1, 2).reshape(-1, C)
        gathered = known_flat[idx3_flat]  # [B, n, 3, C]
        interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # [B, n, C]
        return interpolated.transpose(1, 2)  # [B, C, n]
