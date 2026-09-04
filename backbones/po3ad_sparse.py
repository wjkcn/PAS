"""PO3AD with spconv — sparse 3D convolution offset prediction.

Proper implementation using spconv (SparseConv3d + SubMConv3d + SparseInverseConv3d)
instead of MinkowskiEngine. Maintains sharp boundary features and efficient memory usage
by only computing on occupied voxels.

PAS-compatible: uses FPS/PAS-selected points as input.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class SparseResBlock(nn.Module):
    """Residual block with SubMConv3d + BN + ReLU."""

    def __init__(self, channels, indice_key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        out = self.relu(self.bn1(self.conv1(x).features))
        out = self.bn2(self.conv2(x.replace_feature(out)).features)
        return x.replace_feature(self.relu(out + identity))


class SparsePO3AD(nn.Module):
    """Sparse 3D U-Net for point cloud offset prediction.

    Voxelize → Sparse U-Net (encoder-decoder with skip connections) → v2p → MLP → offset.
    Anomaly score = L1 norm of predicted offset.

    Args:
        voxel_size: grid cell size (in point cloud coordinate units)
        base_ch: base channel count
    """

    def __init__(self, voxel_size=0.005, base_ch=32):
        super().__init__()
        self.voxel_size = voxel_size

        # Encoder
        self.enc_conv1 = spconv.SubMConv3d(3, base_ch, 3, padding=1,
                                           bias=False, indice_key='enc1')
        self.enc_bn1 = nn.BatchNorm1d(base_ch)

        self.enc_down1 = spconv.SparseConv3d(base_ch, base_ch * 2, 3, stride=2,
                                             padding=1, bias=False, indice_key='down1')
        self.enc_bn2 = nn.BatchNorm1d(base_ch * 2)
        self.enc_block1 = SparseResBlock(base_ch * 2, 'enc_block1')

        self.enc_down2 = spconv.SparseConv3d(base_ch * 2, base_ch * 4, 3, stride=2,
                                             padding=1, bias=False, indice_key='down2')
        self.enc_bn3 = nn.BatchNorm1d(base_ch * 4)
        self.enc_block2 = SparseResBlock(base_ch * 4, 'enc_block2')

        self.enc_down3 = spconv.SparseConv3d(base_ch * 4, base_ch * 8, 3, stride=2,
                                             padding=1, bias=False, indice_key='down3')
        self.enc_bn4 = nn.BatchNorm1d(base_ch * 8)
        self.enc_block3 = SparseResBlock(base_ch * 8, 'enc_block3')

        # Decoder
        self.dec_up3 = spconv.SparseInverseConv3d(base_ch * 8, base_ch * 4, 3,
                                                  indice_key='down3')
        self.dec_bn3 = nn.BatchNorm1d(base_ch * 4)
        self.dec_block3 = SparseResBlock(base_ch * 4, 'dec_block3')

        self.dec_up2 = spconv.SparseInverseConv3d(base_ch * 4, base_ch * 2, 3,
                                                  indice_key='down2')
        self.dec_bn2 = nn.BatchNorm1d(base_ch * 2)
        self.dec_block2 = SparseResBlock(base_ch * 2, 'dec_block2')

        self.dec_up1 = spconv.SparseInverseConv3d(base_ch * 2, base_ch, 3,
                                                  indice_key='down1')
        self.dec_bn1 = nn.BatchNorm1d(base_ch)
        self.dec_conv1 = spconv.SubMConv3d(base_ch, base_ch, 3, padding=1,
                                           bias=False, indice_key='dec1')
        self.dec_out_bn = nn.BatchNorm1d(base_ch)

        self.relu = nn.ReLU(inplace=True)

        # Offset prediction head
        self.offset_head = nn.Sequential(
            nn.Linear(base_ch, base_ch // 2, bias=False),
            nn.BatchNorm1d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch // 2, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3, bias=True),
        )

    def _voxelize(self, xyz):
        """Convert point cloud to SparseConvTensor.

        xyz: [B, N, 3]
        Returns: (sparse_tensor, v2p_indices)
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # Normalize per batch
        xyz_min = xyz.amin(dim=1, keepdim=True)
        xyz_max = xyz.amax(dim=1, keepdim=True)
        extent = (xyz_max - xyz_min).clamp(min=1e-6)
        xyz_norm = (xyz - xyz_min) / extent  # [0, 1]

        # Determine grid size from extent and voxel_size
        grid_dim = (extent[0, 0] / self.voxel_size).ceil().long()
        # Clamp to reasonable range
        grid_dim = grid_dim.clamp(min=16, max=256)

        # Quantize
        xyz_q = (xyz_norm[0] * (grid_dim.float() - 1).to(device)).round().long()
        xyz_q = xyz_q.clamp(torch.zeros(3, device=device).long(),
                            (grid_dim - 1).to(device))

        # Hash voxel coordinates
        stride = torch.tensor([1, grid_dim[0], grid_dim[0] * grid_dim[1]],
                              device=device, dtype=torch.long)
        voxel_hash = (xyz_q * stride).sum(dim=1)  # [N]

        # Unique voxels and inverse mapping
        unique_hash, inverse = torch.unique(voxel_hash, return_inverse=True)

        # For each unique voxel, compute mean xyz
        n_voxels = len(unique_hash)
        voxel_feat = torch.zeros(n_voxels, 3, device=device)
        voxel_count = torch.zeros(n_voxels, 1, device=device)
        voxel_feat = voxel_feat.scatter_add(0, inverse.unsqueeze(-1).expand(-1, 3), xyz[0])
        voxel_count = voxel_count.scatter_add(0, inverse.unsqueeze(-1), torch.ones(N, 1, device=device))
        voxel_feat = voxel_feat / voxel_count.clamp(min=1)

        # Reconstruct quantized coordinates from hash
        x = unique_hash % grid_dim[0]
        y = (unique_hash // grid_dim[0]) % grid_dim[1]
        z = unique_hash // (grid_dim[0] * grid_dim[1])
        voxel_indices = torch.stack([torch.zeros(n_voxels, dtype=torch.int32, device=device),
                                      x.int(), y.int(), z.int()], dim=1)

        sparse_tensor = spconv.SparseConvTensor(
            voxel_feat, voxel_indices, grid_dim.tolist(), 1
        )

        return sparse_tensor, inverse

    def forward(self, xyz, return_scores=False):
        """xyz: [B, N, 3] → offset [B, N, 3] or scores [B, N]."""
        B, N, _ = xyz.shape
        assert B == 1, "SparsePO3AD currently suppopas batch_size=1"

        sparse_tensor, v2p = self._voxelize(xyz)

        # Encoder
        e1 = sparse_tensor.replace_feature(
            self.relu(self.enc_bn1(self.enc_conv1(sparse_tensor).features)))

        d1 = self.enc_down1(e1)
        d1 = d1.replace_feature(self.relu(self.enc_bn2(d1.features)))
        d1 = self.enc_block1(d1)

        d2 = self.enc_down2(d1)
        d2 = d2.replace_feature(self.relu(self.enc_bn3(d2.features)))
        d2 = self.enc_block2(d2)

        d3 = self.enc_down3(d2)
        d3 = d3.replace_feature(self.relu(self.enc_bn4(d3.features)))
        d3 = self.enc_block3(d3)

        # Decoder
        u3 = self.dec_up3(d3)
        u3 = u3.replace_feature(self.relu(self.dec_bn3(u3.features)))
        u3 = self.dec_block3(u3)

        u2 = self.dec_up2(u3)
        u2 = u2.replace_feature(self.relu(self.dec_bn2(u2.features)))
        u2 = self.dec_block2(u2)

        u1 = self.dec_up1(u2)
        u1 = u1.replace_feature(self.relu(self.dec_bn1(u1.features)))
        u1_feat = self.relu(self.dec_out_bn(self.dec_conv1(u1).features))

        # Map voxel features to points
        point_feat = u1_feat[v2p]  # [N, base_ch]
        offset = self.offset_head(point_feat)  # [N, 3]

        if return_scores:
            return offset.abs().sum(dim=-1).unsqueeze(0)  # [1, N]
        return offset.unsqueeze(0)  # [1, N, 3]

    def anomaly_score(self, xyz):
        """Per-point anomaly score from offset magnitude."""
        return self.forward(xyz, return_scores=True)


def generate_pseudo_anomaly(xyz, noise_ratio=0.05, noise_std=0.02):
    """Generate pseudo-anomalies for PO3AD training.

    Randomly perturbs noise_ratio fraction of points with Gaussian noise.
    Returns (noisy_xyz, gt_offset, anomaly_mask).

    xyz: [B, N, 3]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    extent = (xyz.amax(dim=1, keepdim=True) - xyz.amin(dim=1, keepdim=True))
    extent_max = extent.amax(dim=-1, keepdim=True).clamp(min=1e-6)

    noise = torch.randn(B, N, 3, device=device) * noise_std * extent_max
    mask = torch.rand(B, N, 1, device=device) < noise_ratio
    noisy_xyz = xyz + noise * mask.float()
    gt_offset = noise * mask.float()

    return noisy_xyz, gt_offset
