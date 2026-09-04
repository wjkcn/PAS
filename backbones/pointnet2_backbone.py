"""PointNet++ SSG backbone with PAS sampling support.

Wraps the SA (Set Abstraction) modules from pointnet2_ops to accept
anomaly_scores for PAS-guided FPS replacement.
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


def _fps(xyz, npoint):
    if _has_cuda:
        idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
        return idx
    else:
        from backbones.pas_sampler import _native_fps
        return _native_fps(xyz, npoint)


def _gather(xyz, idx):
    """xyz: [B, N, 3], idx: [B, K], returns [B, K, 3]"""
    if _has_cuda:
        out = pointnet2_utils.gather_operation(
            xyz.transpose(1, 2).contiguous(), idx.int()
        ).transpose(1, 2).contiguous()
    else:
        B = xyz.size(0)
        idx_base = torch.arange(B, device=xyz.device).view(-1, 1) * xyz.size(1)
        out = xyz.view(-1, 3)[(idx + idx_base).view(-1)].view(B, -1, 3)
    return out


class PointNet2SAT(nn.Module):
    """Set Abstraction module with PAS support.

    Wraps a standard SA module, intercepting FPS to use PAS.
    """

    def __init__(self, npoint, radius, nsample, mlp, use_xyz=True):
        super().__init__()
        self.npoint = npoint
        self.pas_sampler = PASSampler(tau=0.6) if npoint is not None else None
        self.radius = radius
        self.nsample = nsample

        if _has_cuda:
            self._sa = pointnet2_modules.PointnetSAModule(
                npoint=npoint,
                radius=radius,
                nsample=nsample,
                mlp=mlp,
                use_xyz=use_xyz,
            )
            self._has_cuda_sa = True
        else:
            self._has_cuda_sa = False

    def _build_mlp(self, mlp_spec):
        layers = []
        for i in range(1, len(mlp_spec)):
            layers.append(nn.Conv2d(mlp_spec[i-1], mlp_spec[i], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, xyz, features, anomaly_scores=None, sampling_mode='auto'):
        """
        xyz: [B, N, 3]
        features: [B, C, N] or None
        anomaly_scores: [B, N] optional
        sampling_mode: 'auto' (pas if scores given), 'fps', 'pas'

        Returns: new_xyz [B, K, 3], new_features [B, C', K]
        """
        if self.npoint is None:
            # Global SA: no sampling, group all
            if self._has_cuda_sa:
                return self._sa(xyz, features)
            else:
                return self._global_sa(xyz, features)

        # Determine sampling strategy
        npoint = min(self.npoint, xyz.size(1))
        if sampling_mode == 'auto':
            use_pas = anomaly_scores is not None
        elif sampling_mode == 'pas':
            use_pas = anomaly_scores is not None
        else:
            use_pas = False

        # Select center indices
        if use_pas:
            center_idx = self.pas_sampler(xyz, npoint, anomaly_scores)
        else:
            center_idx = _fps(xyz, npoint)

        # Gather centers
        new_xyz = _gather(xyz, center_idx)  # [B, K, 3]

        # Ball query + group + MLP (standard PointNet++ ops)
        if self._has_cuda_sa:
            xyz = xyz.contiguous()
            new_xyz = new_xyz.contiguous()
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            idx = pointnet2_utils.ball_query(
                self.radius, self.nsample, xyz, new_xyz
            )
            grouped_xyz = pointnet2_utils.grouping_operation(
                xyz_flipped, idx
            )
            grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)

            if features is not None:
                features = features.contiguous()
                grouped_feat = pointnet2_utils.grouping_operation(features, idx)
                grouped = torch.cat([grouped_xyz, grouped_feat], dim=1)
            else:
                grouped = grouped_xyz

            # Use the original SA module's MLP (hijack it)
            new_features = self._sa.mlps[0](grouped)
            new_features = F.max_pool2d(new_features, [1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
        else:
            new_features = self._naive_group_mlp(xyz, new_xyz, features)

        return new_xyz, new_features, center_idx

    def _naive_group_mlp(self, xyz, new_xyz, features):
        """Fallback grouping + MLP when CUDA ops unavailable."""
        B, N, _ = xyz.shape
        K = new_xyz.size(1)
        n_sample = min(self.nsample, N)

        dist = torch.cdist(new_xyz, xyz)
        _, nn_idx = torch.topk(dist, n_sample, dim=-1, largest=False)

        grouped = xyz.unsqueeze(1).expand(B, K, N, 3)[
            torch.arange(B).view(-1, 1, 1),
            torch.arange(K).view(1, -1, 1),
            nn_idx
        ] - new_xyz.unsqueeze(2)

        grouped = grouped.permute(0, 3, 1, 2)
        if hasattr(self, '_mlp'):
            return self._mlp(grouped).max(dim=-1)[0]
        return grouped.mean(dim=-1).transpose(1, 2)

    def _global_sa(self, xyz, features):
        B = xyz.size(0)
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            grouped = torch.cat([grouped_xyz, grouped_features], dim=1)
        else:
            grouped = grouped_xyz
        out = self._sa.mlps[0](grouped)
        out = F.max_pool2d(out, [1, out.size(3)]).squeeze(-1)
        return None, out


class PointNet2Backbone(nn.Module):
    """PointNet++ SSG feature extractor that suppopas PAS sampling.

    Architecture: SA(512) -> SA(128) -> SA(global)
    Output: multi-scale features [B, 1024] (classification-style)
    """

    def __init__(self, use_xyz=True):
        super().__init__()

        self.sa1 = PointNet2SAT(
            npoint=512, radius=0.2, nsample=64,
            mlp=[3, 64, 64, 128], use_xyz=use_xyz
        )
        self.sa2 = PointNet2SAT(
            npoint=128, radius=0.4, nsample=64,
            mlp=[128, 128, 128, 256], use_xyz=use_xyz
        )
        self.sa3 = PointNet2SAT(
            npoint=None, radius=None, nsample=None,
            mlp=[256, 256, 512, 1024], use_xyz=use_xyz
        )

        self._sampling_mode = None

    def set_sampling_mode(self, mode):
        self._sampling_mode = mode

    def output_dim(self):
        return 1024

    def forward(self, xyz, anomaly_scores=None):
        """
        xyz: [B, N, 3]
        anomaly_scores: [B, N] optional

        Returns:
            features: [B, 1024, 1]
            center_indices: list of [B, K] per SA level
            xyz_list: list of xyz tensors per SA level
        """
        mode = self._sampling_mode or 'auto'
        features = xyz.transpose(1, 2).contiguous()  # [B, 3, N] — PointNet++ uses xyz coordinates as initial features
        center_indices = []

        # SA1: N → 512
        new_xyz, features, idx1 = self.sa1(xyz, features, anomaly_scores, mode)
        center_indices.append(idx1)

        # Propagate anomaly scores to SA1 centers
        if anomaly_scores is not None and idx1 is not None:
            B = anomaly_scores.size(0)
            scores_512 = anomaly_scores[
                torch.arange(B, device=anomaly_scores.device).view(-1, 1), idx1
            ]
        else:
            scores_512 = None

        # SA2: 512 → 128
        # PAS is applied only at the first sampling bottleneck.
        # Subsequent Set Abstraction layers retain standard FPS.
        new_xyz, features, idx2 = self.sa2(new_xyz, features, scores_512, "fps")
        center_indices.append(idx2)

        # SA3: 128 → 1 (global)
        # PAS is applied only at the first sampling bottleneck.
        # Subsequent Set Abstraction layers retain standard FPS.
        _, features = self.sa3(new_xyz, features, None, "fps")

        return features, center_indices, [xyz, new_xyz]


class PointNet2AD(nn.Module):
    """PointNet++ based anomaly detection pipeline.

    Combines PointNet2Backbone with PatchCore-style memory bank and scoring.
    Designed to be comparable to the Point-MAE (M3DM) pipeline but without
    the interpolation bottleneck.
    """

    def __init__(self, memory_size=400, f_coreset=0.1):
        super().__init__()
        self.backbone = PointNet2Backbone(use_xyz=True)
        self.memory_bank = []
        self.memory_size = memory_size
        self.f_coreset = f_coreset
        self.memory = None
        self.memory_mean = None
        self.memory_std = None

    def set_sampling_mode(self, mode):
        self.backbone.set_sampling_mode(mode)

    def output_dim(self):
        return self.backbone.output_dim()

    @torch.no_grad()
    def forward(self, xyz, anomaly_scores=None):
        """Returns global feature vector [B, 1024]."""
        features, _, _ = self.backbone(xyz, anomaly_scores)
        return features.squeeze(-1)  # [B, 1024]

    @torch.no_grad()
    def add_to_memory(self, xyz, anomaly_scores=None):
        """Add a normal sample's feature to the memory bank."""
        feat = self(xyz, anomaly_scores)  # [1, 1024]
        self.memory_bank.append(feat.cpu())

    def build_memory(self):
        """Build coreset memory bank from collected features."""
        all_feat = torch.cat(self.memory_bank, dim=0)  # [M, 1024]
        if self.f_coreset < 1.0 and all_feat.size(0) > self.memory_size:
            all_feat = all_feat[:self.memory_size]
        self.memory_mean = all_feat.mean()
        self.memory_std = all_feat.std()
        self.memory = (all_feat - self.memory_mean) / (self.memory_std + 1e-8)
        self.memory_bank = []

    def score(self, xyz, anomaly_scores=None):
        """Score a test sample. Returns image-level anomaly score."""
        feat = self(xyz, anomaly_scores)
        feat = (feat - self.memory_mean.to(feat.device)) / (
            self.memory_std.to(feat.device) + 1e-8
        )
        mem = self.memory.to(feat.device)
        dist = torch.cdist(feat, mem)
        min_dist, _ = torch.min(dist, dim=1)
        s_star = torch.max(min_dist)
        return s_star.cpu().item()
