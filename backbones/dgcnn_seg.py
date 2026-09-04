"""DGCNN segmentation backbone with PAS sampling support.

Uses EdgeConv (dynamic k-NN graph + edge features + MLP) as the feature extraction
operator instead of PointNet++'s ball query. FPS/PAS controls downsampling in the
Set Abstraction modules — same interface as PointNet2SegBackbone for drop-in comparison.

Architecture:
  SA1 (N→512) → SA2 (512→128) → SA3 (128→1)
  FP3 (1→128)  → FP2 (128→512)  → FP1 (512→N)
  Output: [B, 64, N] per-point features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.pas_sampler import PASSampler
from backbones.pointnet2_backbone import _fps, _gather


def _gather_features(features, idx):
    """Gather features [B, N, C] by indices [B, K] → [B, K, C]."""
    B, N, C = features.shape
    K = idx.size(1)
    batch_idx = torch.arange(B, device=features.device).view(-1, 1) * N
    flat_idx = (idx + batch_idx).reshape(-1)
    return features.reshape(-1, C)[flat_idx].reshape(B, K, C)


class EdgeConv(nn.Module):
    """EdgeConv block: k-NN graph in xyz space → edge features → Conv2d MLP → max pool."""

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        mlp_spec = [in_channels * 2] + [out_channels, out_channels]
        layers = []
        for i in range(1, len(mlp_spec)):
            layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, xyz, features):
        """xyz: [B, N, 3], features: [B, Cin, N] → [B, Cout, N]."""
        B, N_pts, _ = xyz.shape
        k = min(self.k, N_pts)

        dist = torch.cdist(xyz, xyz)
        _, nn_idx = torch.topk(dist, k, dim=-1, largest=False)

        feat_t = features.permute(0, 2, 1)
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, N_pts, k)
        feat_neighbors = feat_t[batch_idx, nn_idx]
        feat_center = feat_t.unsqueeze(2)

        edge = torch.cat([feat_center.expand(-1, -1, k, -1),
                          feat_neighbors - feat_center], dim=-1)

        edge = edge.permute(0, 3, 1, 2)
        out = self.mlp(edge)
        return out.max(dim=-1)[0]


class DGCNNSA(nn.Module):
    """Set Abstraction: EdgeConv feature extraction + FPS/PAS downsampling.

    Same interface as PointNet2SegSA.
    """

    def __init__(self, npoint, k, mlp, tau=0.6):
        super().__init__()
        self.npoint = npoint
        in_ch = mlp[0]
        out_ch = mlp[-1]
        self.edge_conv = EdgeConv(in_ch, out_ch, k)
        self.pas_sampler = PASSampler(tau=tau) if npoint is not None else None

    def forward(self, xyz, features, anomaly_scores=None, sampling_mode='auto'):
        B, N, _ = xyz.shape

        if self.npoint is None:
            feat = self.edge_conv(xyz, features)
            feat = feat.max(dim=-1, keepdim=True)[0]
            return None, feat, None

        feat = self.edge_conv(xyz, features)
        npoint = min(self.npoint, N)

        if sampling_mode in ('vds', 'grid', 'dafps', 'curv', 'ffps'):
            # Geometric alternative sampling methods
            if sampling_mode == 'vds':
                from backbones.voxel_sampler import voxel_downsampling
                center_idx = voxel_downsampling(xyz, npoint)
            elif sampling_mode == 'grid':
                from backbones.gss_sampler import grid_sampling
                center_idx = grid_sampling(xyz, npoint)
            elif sampling_mode == 'dafps':
                from backbones.density_fps import density_aware_fps
                center_idx = density_aware_fps(xyz, npoint, voxel_size=0.05, alpha=0.5)
            elif sampling_mode == 'curv':
                from backbones.curvature_sampler import curvature_guided_sampling
                center_idx = curvature_guided_sampling(xyz, npoint, k=32, tau=0.5)
            else:  # ffps
                from backbones.pointsp_ffps import filtered_fps
                center_idx = filtered_fps(xyz, npoint, k=20, omega=0.95)
        elif sampling_mode == 'auto':
            use_pas = anomaly_scores is not None
            use_rs = False
        elif sampling_mode == 'pas':
            use_pas = anomaly_scores is not None
            use_rs = False
        elif sampling_mode == 'rs':
            use_pas = False
            use_rs = True
        else:
            use_pas = False
            use_rs = False

        if sampling_mode in ('vds', 'grid', 'dafps', 'curv', 'ffps'):
            pass  # center_idx already computed above
        elif use_pas:
            center_idx = self.pas_sampler(xyz, npoint, anomaly_scores)
        elif use_rs:
            center_idx = torch.stack([
                torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)
            ], dim=0)
        else:
            center_idx = _fps(xyz, npoint)

        new_xyz = _gather(xyz, center_idx)
        new_feat = _gather_features(feat.permute(0, 2, 1), center_idx).permute(0, 2, 1)

        return new_xyz, new_feat, center_idx


class DGCNNSegBackbone(nn.Module):
    """DGCNN segmentation backbone for PAS benchmarking.

    SA1 (N→512) → SA2 (512→128) → SA3 (128→1)
    FP3 (1→128)  → FP2 (128→512)  → FP1 (512→N)
    Output: [B, 64, N]
    """

    def __init__(self, k=20, tau=0.6):
        super().__init__()

        self.sa1 = DGCNNSA(npoint=512, k=k, mlp=[3, 64, 64, 128], tau=tau)
        self.sa2 = DGCNNSA(npoint=128, k=k, mlp=[128, 128, 128, 256], tau=tau)
        self.sa3 = DGCNNSA(npoint=None, k=k, mlp=[256, 256, 512, 1024])

        from backbones.pointnet2_seg import PointNetFP
        self.fp3 = PointNetFP(mlp=[1024 + 256, 512, 256])
        self.fp2 = PointNetFP(mlp=[256 + 128, 256, 128])
        self.fp1 = PointNetFP(mlp=[128 + 3, 128, 64])

        self._sampling_mode = None

    def set_sampling_mode(self, mode):
        self._sampling_mode = mode

    @property
    def output_dim(self):
        return 64

    def forward(self, xyz, anomaly_scores=None):
        mode = self._sampling_mode or 'auto'
        input_features = xyz.transpose(1, 2).contiguous()

        sa1_xyz, sa1_feat, idx1 = self.sa1(xyz, input_features, anomaly_scores, mode)

        if anomaly_scores is not None and idx1 is not None:
            B = anomaly_scores.size(0)
            scores_512 = anomaly_scores[
                torch.arange(B, device=anomaly_scores.device).view(-1, 1), idx1]
        else:
            scores_512 = None

        # PAS is applied only at the first sampling bottleneck.
        # Subsequent Set Abstraction layers retain standard FPS.
        sa2_xyz, sa2_feat, idx2 = self.sa2(sa1_xyz, sa1_feat, scores_512, "fps")

        _, sa3_feat, _ = self.sa3(sa2_xyz, sa2_feat, None, "fps")

        fp3_feat = self.fp3(sa2_xyz, None, sa2_feat, sa3_feat)
        fp2_feat = self.fp2(sa1_xyz, sa2_xyz, sa1_feat, fp3_feat)
        fp1_feat = self.fp1(xyz, sa1_xyz, input_features, fp2_feat)

        sa_xyzs = [xyz, sa1_xyz, sa2_xyz]
        sa_feats = [input_features, sa1_feat, sa2_feat, sa3_feat]
        sa_indices = [idx1, idx2]

        return fp1_feat, sa_xyzs, sa_feats, sa_indices
