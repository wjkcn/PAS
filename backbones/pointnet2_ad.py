"""PointNet++ anomaly detection pipeline with PAS-guided sampling.

Complete pipeline:
  DINO → anomaly scores → PAS-guided PointNet++ → Memory Bank → Anomaly Score

Usable as a drop-in alternative to the M3DM Point-MAE pipeline.
No interpolation bottleneck (PointNet++ SA modules use direct FPS/PAS selection).
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from backbones.pointnet2_backbone import PointNet2AD as PN2Core
from backbones.pas_sampler import compute_anomaly_scores
from utils.mvtec3d_util import organized_pc_to_unorganized_pc


class PointNet2ADPipeline:
    """Full AD pipeline: data → features → memory → scoring.

    Designed for MVTec-3D organized point cloud format (224x224 grid).
    Convepas organized PC to unorganized, extracts DINO anomaly scores,
    feeds into PAS-guided PointNet++, builds PatchCore memory bank.
    """

    def __init__(self, args, dino_extractor=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = args.img_size

        # 2D anomaly signal source (DINO)
        self.dino_extractor = dino_extractor

        # 3D backbone
        self.backbone = PN2Core()
        self.backbone.to(self.device)
        self.backbone.eval()

        # Memory bank
        self.memory = None
        self.memory_mean = 0.0
        self.memory_std = 1.0
        self.patch_lib = []

        # Metrics
        self.image_preds = []
        self.image_labels = []
        self.pixel_preds = []
        self.pixel_labels = []
        self.predictions = []
        self.gts = []

        # Sampling config
        self.sampling_mode = None  # None=auto, 'fps', 'pas', 'rs', 'anomaly'
        self._sample_count = 0
        self._max_samples = args.max_sample

    def set_sampling_mode(self, mode):
        self.sampling_mode = mode
        self.backbone.set_sampling_mode(mode)

    def _sample_to_pc(self, sample):
        """Convert organized PC sample to unorganized tensor."""
        organized_pc = sample[1].squeeze().permute(1, 2, 0).numpy()
        unorganized = organized_pc_to_unorganized_pc(organized_pc)
        nonzero = np.nonzero(np.all(unorganized != 0, axis=1))[0]
        clean = torch.tensor(unorganized[nonzero, :], dtype=torch.float32)
        return clean.unsqueeze(0).to(self.device), nonzero  # [B, N, 3]

    def _compute_dino_scores(self, rgb_img):
        """Compute per-point anomaly scores from DINO features."""
        with torch.no_grad():
            feat = self.dino_extractor.forward_rgb_features(rgb_img.to(self.device))
            # feat: [1, C, 16, 16]
            feat_up = F.interpolate(feat, size=(self.image_size, self.image_size),
                                     mode='bilinear', align_corners=False)
            feat_flat = feat_up.squeeze(0).view(feat_up.size(1), -1).T  # [N, C]
            scores = compute_anomaly_scores(feat_flat.unsqueeze(0))
            return scores.squeeze(0)  # [N]

    def fit(self, class_name, train_loader):
        """Extract features from normal training samples and build memory bank."""
        print(f"[PN2-AD] Fitting class '{class_name}'...")
        self.backbone.eval()

        count = 0
        for sample, _ in tqdm(train_loader, desc=f"  Extracting train features"):
            xyz_t, nonzero = self._sample_to_pc(sample)

            scores = None
            if self.dino_extractor is not None and self.sampling_mode != 'fps':
                scores_full = self._compute_dino_scores(sample[0])
                scores = scores_full[nonzero].unsqueeze(0)

            feat = self.backbone(xyz_t, scores)  # [1, 1024]
            self.patch_lib.append(feat.cpu())

            count += 1
            if count >= self._max_samples:
                break

        self.patch_lib = torch.cat(self.patch_lib, dim=0)
        self.memory_mean = self.patch_lib.mean()
        self.memory_std = self.patch_lib.std()
        self.memory = (self.patch_lib - self.memory_mean) / (self.memory_std + 1e-8)
        self.patch_lib = []

        print(f"  Memory bank built: {self.memory.shape[0]} features × {self.memory.shape[1]} dim")

    def predict(self, sample, mask, label):
        """Score a single test sample."""
        self.backbone.eval()

        xyz_t, nonzero = self._sample_to_pc(sample)

        with torch.no_grad():
            scores = None
            if self.dino_extractor is not None and self.sampling_mode != 'fps':
                scores_full = self._compute_dino_scores(sample[0])
                scores = scores_full[nonzero].unsqueeze(0)

            feat = self.backbone(xyz_t, scores)  # [1, 1024]
            feat_norm = (feat - self.memory_mean.to(feat.device)) / (
                self.memory_std.to(feat.device) + 1e-8
            )

            mem = self.memory.to(feat.device)
            dist = torch.cdist(feat_norm, mem)
            min_val, _ = torch.min(dist, dim=1)

            # Image-level anomaly score
            D = math.sqrt(feat_norm.size(1))
            s_star = torch.max(min_val).cpu().item()
            s_star = s_star / 1000.0  # scaling (matching M3DM convention)

        self.image_preds.append(s_star)
        self.image_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    def evaluate(self, class_name, test_loader):
        """Evaluate on test set, compute AUC."""
        print(f"[PN2-AD] Evaluating class '{class_name}'...")
        for sample, mask, label, _ in tqdm(test_loader, desc=f"  Testing"):
            self.predict(sample, mask, label)

        from sklearn.metrics import roc_auc_score
        img_auc = roc_auc_score(self.image_labels, self.image_preds)
        return img_auc
