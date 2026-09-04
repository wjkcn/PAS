"""Lightweight PO3AD — 3D Conv offset prediction without MinkowskiEngine.

Core idea from PO3AD (CVPR 2025): anomaly score = per-point offset magnitude.
Implementation uses standard PyTorch Conv3d for dense voxel processing.

Pipeline:
  point cloud → voxelize to dense 48³ grid → 3D U-Net → trilinear sample at points
  → MLP predicts offset [N,3] → anomaly score = |offset|

Designed to be compatible with PAS: can use PAS-sampled or FPS-sampled point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def points_to_voxel(xyz, grid_size=48):
    """Convert point cloud to dense voxel grid.

    xyz: [B, N, 3] in arbitrary range
    Returns:
        voxel:     [B, 3, G, G, G] mean xyz per voxel
        mask:      [B, 1, G, G, G] occupancy
        xyz_01:    [B, N, 3] normalized coordinates in [0, 1] (for grid_sample)
        flat_idx:  [B, N] voxel index per point
    """
    B, N, _ = xyz.shape
    G = grid_size
    device = xyz.device

    # Normalize to [0, 1] per batch
    xyz_min = xyz.amin(dim=1, keepdim=True)  # [B, 1, 3]
    xyz_max = xyz.amax(dim=1, keepdim=True)  # [B, 1, 3]
    extent = (xyz_max - xyz_min).clamp(min=1e-6)
    xyz_01 = (xyz - xyz_min) / extent  # [0, 1]

    # Quantize to voxel indices
    xyz_idx = (xyz_01 * (G - 1)).round().long().clamp(0, G - 1)  # [B, N, 3]

    # Flatten
    flat_idx = (xyz_idx[:, :, 0] * G * G +
                xyz_idx[:, :, 1] * G +
                xyz_idx[:, :, 2])  # [B, N]

    total = G * G * G
    voxel = torch.zeros(B, 3, total, device=device)
    count = torch.zeros(B, 1, total, device=device)

    for c in range(3):
        voxel[:, c].scatter_add_(1, flat_idx, xyz[:, :, c])
    count[:, 0].scatter_add_(1, flat_idx, torch.ones(B, N, device=device))

    count = count.clamp(min=1)
    voxel = voxel / count
    mask = (count > 1e-6).float()

    return (voxel.reshape(B, 3, G, G, G),
            mask.reshape(B, 1, G, G, G),
            xyz_01,
            flat_idx)


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Tiny3DUNet(nn.Module):
    """Lightweight 3D U-Net for voxel feature extraction."""

    def __init__(self, in_ch=3, base_ch=32, grid_size=48):
        super().__init__()
        G = grid_size
        assert G % 8 == 0, "grid_size must be divisible by 8"

        self.enc1 = _conv_block(in_ch, base_ch)       # G
        self.enc2 = _conv_block(base_ch, base_ch * 2) # G/2
        self.enc3 = _conv_block(base_ch * 2, base_ch * 4)  # G/4
        self.enc4 = _conv_block(base_ch * 4, base_ch * 8)  # G/8

        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = _conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _conv_block(base_ch * 2, base_ch)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return d1


class PO3ADLite(nn.Module):
    """Lightweight PO3AD: voxelize → 3D U-Net → offset prediction.

    Output: per-point offset [B, N, 3]. Anomaly score = |offset| (L1 norm).
    """

    def __init__(self, grid_size=48, base_ch=32):
        super().__init__()
        self.grid_size = grid_size
        self.unet = Tiny3DUNet(in_ch=3, base_ch=base_ch, grid_size=grid_size)
        self.offset_head = nn.Sequential(
            nn.Linear(base_ch, base_ch // 2, bias=False),
            nn.BatchNorm1d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch // 2, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3, bias=True),
        )

    def forward(self, xyz, return_scores=False):
        """xyz: [B, N, 3] → offset [B, N, 3] or scores [B, N]."""
        B, N, _ = xyz.shape

        voxel, mask, xyz_01, _ = points_to_voxel(xyz, self.grid_size)
        voxel = voxel * mask

        voxel_feat = self.unet(voxel)  # [B, base_ch, G, G, G]

        # Trilinear sample at point locations (grid_sample uses [-1, 1])
        grid = xyz_01 * 2 - 1  # [0,1] → [-1,1]
        grid = grid[:, None, None, :, :]  # [B, 1, 1, N, 3]
        point_feat = F.grid_sample(voxel_feat, grid, mode='bilinear',
                                   align_corners=True, padding_mode='border')
        point_feat = point_feat[:, :, 0, 0, :]  # [B, C, D, H, W] → [B, C, N]

        point_feat_t = point_feat.transpose(1, 2)  # [B, N, C]

        offset = self.offset_head(point_feat_t.reshape(-1, point_feat_t.size(-1)))
        offset = offset.reshape(B, N, 3)

        if return_scores:
            scores = offset.abs().sum(dim=-1)  # L1 norm
            return scores

        return offset

    def anomaly_score(self, xyz):
        """Per-point anomaly score from offset magnitude."""
        return self.forward(xyz, return_scores=True)


def generate_pseudo_anomaly(xyz, noise_ratio=0.05, noise_std=0.02):
    """Generate pseudo-anomalies for PO3AD training.

    Randomly selects noise_ratio fraction of points, adds Gaussian noise.
    Returns modified xyz and ground truth offsets (non-zero only for modified points).

    xyz: [B, N, 3]
    Returns: (noisy_xyz [B, N, 3], gt_offset [B, N, 3])
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Normalize noise_std by point cloud extent
    extent = (xyz.amax(dim=1, keepdim=True) - xyz.amin(dim=1, keepdim=True)).amax(dim=-1, keepdim=True)  # [B,1,1]
    noise = torch.randn(B, N, 3, device=device) * noise_std * extent

    # Random mask: noise_ratio of points get perturbed
    mask = torch.rand(B, N, 1, device=device) < noise_ratio
    noisy_xyz = xyz + noise * mask.float()

    gt_offset = noise * mask.float()

    return noisy_xyz, gt_offset
