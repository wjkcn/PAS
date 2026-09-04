"""PointNet++ segmentation backbone with PAS sampling support.

Outputs PER-POINT features (not global) via Set Abstraction + Feature Propagation.
This enables point-level anomaly detection with memory bank comparison.

PAS replaces only the first sampling bottleneck (SA1).
Subsequent PointNet2 abstraction layers (SA2, SA3) retain standard FPS
to preserve the original backbone architecture.

Architecture:
  SA1 (N→512) → SA2 (512→128) → SA3 (128→1)
  FP3 (1→128) → FP2 (128→512) → FP1 (512→N)
  Output: [B, C_out, N] per-point features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointnet2_ops import pointnet2_modules, pointnet2_utils
    _has_cuda = True
except ImportError:
    _has_cuda = False

from backbones.pas_sampler import PASSampler
from backbones.pointnet2_backbone import _gather as pas_gather, _fps as pas_fps


class PointNet2SegSA(nn.Module):
    """Set Abstraction module with PAS support for segmentation backbone."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, use_xyz=True, tau=0.6, rand_frac=0.1):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pas_sampler = PASSampler(tau=tau, rand_frac=rand_frac) if npoint is not None else None

        if _has_cuda:
            self._sa = pointnet2_modules.PointnetSAModule(
                npoint=npoint, radius=radius, nsample=nsample,
                mlp=list(mlp), use_xyz=use_xyz,
            )
            self._sa_mlp_in = mlp[0] + (3 if use_xyz else 0)
            self._has_cuda_sa = True
        else:
            self._has_cuda_sa = False
            self._use_xyz = use_xyz
            self._build_mlp(mlp, use_xyz)

    def _build_mlp(self, mlp, use_xyz):
        layers = []
        spec = list(mlp)
        if use_xyz:
            spec[0] += 3
        for i in range(1, len(spec)):
            layers.append(nn.Conv2d(spec[i-1], spec[i], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(spec[i]))
            layers.append(nn.ReLU(inplace=True))
        self._mlp = nn.Sequential(*layers)

    def forward(self, xyz, features, anomaly_scores=None, sampling_mode='auto'):
        B, N, _ = xyz.shape

        if self.npoint is None:
            if self._has_cuda_sa:
                _, new_features = self._sa(xyz, features)
                return None, new_features, None
            else:
                grouped = xyz.transpose(1, 2).unsqueeze(2)
                if features is not None:
                    gf = features.unsqueeze(2)
                    grouped = torch.cat([grouped, gf], dim=1)
                out = self._mlp(grouped)
                out = F.max_pool2d(out, [1, out.size(3)]).squeeze(-1)
                return None, out, None

        npoint = min(self.npoint, N)

        # Sampling
        if sampling_mode in ('vds', 'grid', 'dafps', 'curv', 'ffps'):
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
            pass  # center_idx already computed
        elif use_pas:
            center_idx = self.pas_sampler(xyz, npoint, anomaly_scores)
        elif use_rs:
            center_idx = torch.stack([
                torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)
            ], dim=0)
        else:
            center_idx = pas_fps(xyz, npoint)

        new_xyz = pas_gather(xyz, center_idx)

        if self._has_cuda_sa:
            xyz_c = xyz.contiguous()
            new_xyz_c = new_xyz.contiguous()
            idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz_c, new_xyz_c)
            grouped_xyz = pointnet2_utils.grouping_operation(
                xyz_c.transpose(1, 2).contiguous(), idx)
            grouped_xyz = grouped_xyz - new_xyz_c.transpose(1, 2).unsqueeze(-1)

            if features is not None:
                features_c = features.contiguous()
                grouped_feat = pointnet2_utils.grouping_operation(features_c, idx)
                grouped = torch.cat([grouped_xyz, grouped_feat], dim=1)
            else:
                grouped = grouped_xyz

            new_features = self._sa.mlps[0](grouped)
            new_features = F.max_pool2d(new_features, [1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
        else:
            new_features = self._naive_group_mlp(xyz, new_xyz, features)

        return new_xyz, new_features, center_idx

    def _naive_group_mlp(self, xyz, new_xyz, features):
        B, N, _ = xyz.shape
        K = new_xyz.size(1)
        n_sample = min(self.nsample, N)

        dist = torch.cdist(new_xyz, xyz)
        _, nn_idx = torch.topk(dist, n_sample, dim=-1, largest=False)

        batch_idx = torch.arange(B, device=xyz.device).view(-1, 1, 1)
        center_idx = torch.arange(K, device=xyz.device).view(1, -1, 1)
        grouped_xyz = xyz[batch_idx, nn_idx] - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_feat = features.transpose(1, 2)[batch_idx, nn_idx]
            if self._use_xyz:
                grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
            else:
                grouped = grouped_feat
        else:
            grouped = grouped_xyz

        grouped = grouped.permute(0, 3, 1, 2)
        out = self._mlp(grouped)
        return out.max(dim=-1)[0]


class PointNet2SegBackbone(nn.Module):
    """PointNet++ segmentation backbone with PAS support.

    PAS replaces only the first sampling bottleneck (SA1).
    Subsequent PointNet2 abstraction layers (SA2, SA3) retain standard FPS
    to preserve the original backbone architecture.

    SA1(N→512) → SA2(512→128) → SA3(128→1)
    FP3(1→128) → FP2(128→512) → FP1(512→N)

    Output: per-point features [B, 64, N]
    """

    def __init__(self, tau=0.6, rand_frac=0.1):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = PointNet2SegSA(
            npoint=512, radius=0.2, nsample=64,
            in_channel=3, mlp=[3, 64, 64, 128], use_xyz=True, tau=tau, rand_frac=rand_frac
        )
        self.sa2 = PointNet2SegSA(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128, mlp=[128, 128, 128, 256], use_xyz=True, tau=tau, rand_frac=rand_frac
        )
        self.sa3 = PointNet2SegSA(
            npoint=None, radius=None, nsample=None,
            in_channel=256, mlp=[256, 256, 512, 1024], use_xyz=True
        )

        # Feature Propagation layers
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
        """
        Args:
            xyz: [B, N, 3] point cloud
            anomaly_scores: [B, N] optional anomaly scores for PAS

        Returns:
            per_point_feat: [B, 64, N]
            sa_xyzs: list of xyz at each SA level
            sa_feats: list of features at each SA level
            sa_indices: list of center indices at each SA level
        """
        mode = self._sampling_mode or 'auto'
        input_features = xyz.transpose(1, 2).contiguous()  # [B, 3, N]

        # SA1: N → 512
        # PAS replaces only the first sampling bottleneck (SA1).
        # Anomaly scores propagate here to guide PAS sampling.
        sa1_xyz, sa1_feat, idx1 = self.sa1(xyz, input_features, anomaly_scores, mode)

        # Propagate anomaly scores to SA2 (for verification purposes only;
        # SA2 does NOT use them for sampling by design — see below).
        if anomaly_scores is not None and idx1 is not None:
            B = anomaly_scores.size(0)
            scores_512 = anomaly_scores[
                torch.arange(B, device=anomaly_scores.device).view(-1, 1), idx1]
        else:
            scores_512 = None

        # SA2: 512 → 128
        # PAS replaces only the first sampling bottleneck (SA1).
        # Subsequent PointNet2 abstraction layers retain standard FPS
        # to preserve the original backbone architecture.
        sa2_xyz, sa2_feat, idx2 = self.sa2(sa1_xyz, sa1_feat, scores_512, 'fps')

        # SA3: 128 → 1 (global, always FPS)
        _, sa3_feat, _ = self.sa3(sa2_xyz, sa2_feat, None, 'fps')

        # FP3: SA3(1) → SA2(128)
        fp3_feat = self.fp3(sa2_xyz, None, sa2_feat, sa3_feat)

        # FP2: SA2(128) → SA1(512)
        fp2_feat = self.fp2(sa1_xyz, sa2_xyz, sa1_feat, fp3_feat)

        # FP1: SA1(512) → original N
        fp1_feat = self.fp1(xyz, sa1_xyz, input_features, fp2_feat)

        sa_xyzs = [xyz, sa1_xyz, sa2_xyz]
        sa_feats = [input_features, sa1_feat, sa2_feat, sa3_feat]
        sa_indices = [idx1, idx2]

        return fp1_feat, sa_xyzs, sa_feats, sa_indices


class PointNetFP(nn.Module):
    """Feature Propagation module (simplified, without CUDA dependency)."""

    def __init__(self, mlp):
        super().__init__()
        layers = []
        for i in range(1, len(mlp)):
            layers.append(nn.Conv1d(mlp[i-1], mlp[i], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(mlp[i]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, unknown_xyz, known_xyz, unknown_feat, known_feat):
        """
        unknown_xyz: [B, n, 3]  target points
        known_xyz:   [B, m, 3]  source points (or None for global)
        unknown_feat:[B, C1, n]  features at target points
        known_feat:  [B, C2, m]  features at source points

        Returns: [B, mlp[-1], n]
        """
        B, n, _ = unknown_xyz.shape

        if known_xyz is not None:
            # 3-NN interpolation
            dist = torch.cdist(unknown_xyz, known_xyz)  # [B, n, m]
            dist3, idx3 = torch.topk(dist, 3, dim=-1, largest=False)  # [B, n, 3]
            dist3 = torch.clamp(dist3, min=1e-10)
            weight = 1.0 / dist3
            weight = weight / weight.sum(dim=-1, keepdim=True)  # [B, n, 3]

            # Gather features
            m = known_xyz.size(1)
            idx3_flat = idx3 + torch.arange(B, device=idx3.device).view(-1, 1, 1) * m
            known_flat = known_feat.transpose(1, 2).reshape(-1, known_feat.size(1))
            gathered = known_flat[idx3_flat]  # [B, n, 3, C2]
            interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # [B, n, C2]
            interpolated = interpolated.transpose(1, 2)  # [B, C2, n]
        else:
            interpolated = known_feat.expand(-1, -1, n)  # [B, C2, n]

        if unknown_feat is not None:
            new_feat = torch.cat([interpolated, unknown_feat], dim=1)  # [B, C2+C1, n]
        else:
            new_feat = interpolated

        return self.mlp(new_feat)
