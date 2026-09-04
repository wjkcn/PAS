"""PCT (Point Cloud Transformer) segmentation backbone with PAS sampling.

Offset-Attention transformer blocks from Guo et al. 2021 (Computational Visual Media).
PAS sampling inserted after input embedding: selects K centers → transformer operates on
centers only → global pool → tiled back to all N points.

Architecture:
  Shared MLP (all N): 3→64→128
  PAS/FPS sampling: N→512 centers
  Transformer on 512 centers: 128→256, 4×OffsetAttention + FFN blocks
  Global Max Pool: 256→1024
  Tile + Concat(early 128, global 1024) → Conv1d → [B, 64, N]

Interface matches PointNet2SegBackbone for drop-in use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.pas_sampler import PASSampler, _native_fps


def _gather(xyz, idx):
    B, N, _ = xyz.shape
    K = idx.size(1)
    batch_indices = torch.arange(B, device=xyz.device).view(-1, 1).expand(-1, K)
    return xyz[batch_indices, idx.long(), :]


def _gather_features(features, idx):
    B, C, N = features.shape
    feat_t = features.permute(0, 2, 1)  # [B, N, C]
    batch_idx = torch.arange(B, device=features.device).view(B, 1)
    return feat_t[batch_idx, idx.long()].permute(0, 2, 1)  # [B, C, K]


class OffsetAttention(nn.Module):
    """Offset-Attention from PCT (Guo et al. 2021).

    Standard SA:  F_attn = Attention(Q, K, V)
    Offset SA:    F_out  = x + LBR(F_attn - x)

    The subtraction F_attn - x makes the block model the "offset" from identity,
    which the authors show improves robustness to rigid transformations.
    """

    def __init__(self, dim, head_dim=64):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv1d(dim, dim, 1, bias=False)

        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: [B, dim, N] → [B, dim, N]."""
        B, C, N = x.shape
        H = self.num_heads

        q = self.q_conv(x).view(B, H, self.head_dim, N)
        k = self.k_conv(x).view(B, H, self.head_dim, N)
        v = self.v_conv(x).view(B, H, self.head_dim, N)

        # Scaled dot-product attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = torch.einsum('bhnm,bhdm->bhdn', attn, v)

        attn_out = attn.reshape(B, C, N)  # F_attn

        # Offset: LBR(F_attn - x) + x
        offset = self.proj(attn_out - x)
        return x + offset


class TransformerBlock(nn.Module):
    """PCT transformer block: OffsetAttention + FFN, pre-normalization style."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.attn = OffsetAttention(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim * expansion, 1, bias=False),
            nn.BatchNorm1d(dim * expansion),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim * expansion, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Offset-Attention with residual (built into OffsetAttention)
        x = self.attn(x)
        # FFN with residual
        return x + self.ffn(x)


class PCTSegBackbone(nn.Module):
    """PCT per-point segmentation backbone with PAS sampling.

    Args:
        tau: PAS anomaly-guided fraction (default 0.6)
        ncenter: number of transformer centers (default 512)
        dim: transformer feature dimension (default 256)
        num_blocks: number of transformer blocks (default 4)
    """

    def __init__(self, tau=0.6, ncenter=512, dim=256, num_blocks=4):
        super().__init__()
        self.ncenter = ncenter
        self.dim = dim
        self.pas_sampler = PASSampler(tau=tau)
        self._sampling_mode = None

        # Input embedding (all N points)
        self.embed_conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.embed_bn1 = nn.BatchNorm1d(64)
        self.embed_conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.embed_bn2 = nn.BatchNorm1d(128)

        # Project to transformer dim
        self.proj_in = nn.Conv1d(128, dim, 1, bias=False)

        # Transformer blocks (on K centers)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_blocks)
        ])

        # Global feature
        self.global_conv = nn.Conv1d(dim, 1024, 1, bias=False)
        self.global_bn = nn.BatchNorm1d(1024)

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
            anomaly_scores: [B, N] anomaly scores

        Returns:
            per_point_feat: [B, 64, N]
            sa_xyzs: [original_xyz, center_xyz]
            sa_feats: [input_feat, embed_feat, trans_feat, global_feat]
            sa_indices: [center_indices, None]
        """
        B, N, _ = xyz.shape
        mode = self._sampling_mode or 'auto'
        device = xyz.device

        # Stage 1: input embedding on all N points
        input_feat = xyz.transpose(1, 2)  # [B, 3, N]
        e1 = F.relu(self.embed_bn1(self.embed_conv1(input_feat)))  # [B, 64, N]
        e2 = F.relu(self.embed_bn2(self.embed_conv2(e1)))          # [B, 128, N]

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
        center_feat = _gather_features(e2, center_idx)  # [B, 128, K]

        # Stage 2: project to transformer dim + transformer blocks
        x = self.proj_in(center_feat)  # [B, dim, K]
        for block in self.blocks:
            x = block(x)               # [B, dim, K]

        # Stage 3: global pooling
        x_max = x.max(dim=-1, keepdim=True)[0]  # [B, dim, 1]
        global_feat = F.relu(self.global_bn(self.global_conv(x_max)))  # [B, 1024, 1]

        # Stage 4: tile global to all N points, concat with early embedding
        global_tiled = global_feat.expand(-1, -1, N)     # [B, 1024, N]
        concat = torch.cat([e2, global_tiled], dim=1)    # [B, 128+1024, N]
        out = F.relu(self.out_bn(self.out_conv(concat)))  # [B, 64, N]

        sa_xyzs = [xyz, center_xyz]
        sa_feats = [input_feat, e2, x, global_feat]
        sa_indices = [center_idx, None]

        return out, sa_xyzs, sa_feats, sa_indices
