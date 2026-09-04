"""K_in Sensitivity Experiment (Fixed).

Correctly varies K_in (number of points entering SA1) while keeping
K_mem (memory bank size) fixed at 100.

Previous benchmark_kin_sensitivity.py incorrectly varied K_mem (max_sample)
instead of K_in. This script fixes that.

Usage:
    python benchmark_kin_sensitivity_fixed.py --kin_values 50,100,200,512,1024
    python benchmark_kin_sensitivity_fixed.py --kin_values 50,100,200,512,1024 --classes bagel,carrot,rope
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
from backbones.pointnet2_seg import PointNet2SegBackbone, PointNetFP
from backbones.pas_sampler import PASSampler
from backbones.pointnet2_backbone import _gather as pas_gather, _fps as pas_fps

try:
    from pointnet2_ops import pointnet2_modules, pointnet2_utils
    _has_cuda = True
except ImportError:
    _has_cuda = False


def _batched_cdist_min(query, ref, chunk_size=4096):
    """Compute min distance from each query to ref in chunks to save memory."""
    M = query.size(0)
    mins = []
    for i in range(0, M, chunk_size):
        chunk = query[i:i + chunk_size]
        dist = torch.cdist(chunk, ref)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins)


def compute_pro(score_map, mask, n_thresholds=200):
    """Compute PRO (Per-Region Overlap) score."""
    from skimage import measure
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0.0
    labels = measure.label(mask_bool)
    if labels.max() == 0:
        return 0.0
    thresholds = np.linspace(score_map.min(), score_map.max(), n_thresholds)
    pro_values = []
    for t in thresholds:
        pred = score_map > t
        for region_id in range(1, labels.max() + 1):
            region = labels == region_id
            if region.sum() == 0:
                continue
            intersection = (pred & region).sum()
            union = region.sum()
            pro_values.append(intersection / union)
    return np.mean(pro_values) if pro_values else 0.0


def find_best_threshold(score_map, mask, n_thresholds=200):
    """Find best IoU threshold."""
    thresholds = np.linspace(score_map.min(), score_map.max(), n_thresholds)
    best_iou = 0.0
    for t in thresholds:
        pred = score_map > t
        inter = (pred & mask).sum()
        union = (pred | mask).sum()
        if union > 0:
            iou = inter / union
            best_iou = max(best_iou, iou)
    return best_iou


class KinAwarePipeline:
    """Pipeline that correctly controls K_in (points entering SA1).

    For FPS/PAS strategies, bypasses backbone's fixed SA1(npoint=512)
    by externally sampling K_in points, then running SA1(K_in) → SA2 → SA3 → FP.
    """

    def __init__(self, dino_model, kin, tau=0.6, density_alpha=0.5,
                 density_K=16, max_samples=100, img_size=224):
        self.dino = dino_model
        self.device = dino_model.device
        self.kin = kin
        self.tau = tau
        self.density_alpha = density_alpha
        self.density_K = density_K
        self.max_samples = max_samples
        self.img_size = img_size
        self.use_amp = True

        # Backbone components (reuse from dino_model's Point_MAE)
        # We need SA1/SA2/SA3/FP layers — use PointNet2SegBackbone's layers
        self.bb = PointNet2SegBackbone(tau=tau)
        self.bb.to(self.device).eval()

        self._strategy = 'fps'
        self.memory = None
        self.memory_mean = 0.0
        self.memory_std = 1.0
        self._reset_metrics()

    def _reset_metrics(self):
        self.point_scores_list = []
        self.point_labels_list = []
        self.score_maps = []
        self.mask_maps = []
        self.image_preds = []
        self.image_labels = []

    def set_strategy(self, strategy):
        self._strategy = strategy

    def _sample_to_pc(self, sample):
        organized_pc = sample[1].squeeze().permute(1, 2, 0).numpy()
        unorganized = organized_pc_to_unorganized_pc(organized_pc)
        nonzero = np.nonzero(np.all(unorganized != 0, axis=1))[0]
        clean = torch.tensor(unorganized[nonzero, :], dtype=torch.float32)
        return clean.unsqueeze(0).to(self.device), nonzero

    def _compute_dino_scores(self, rgb_img):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            feat = self.dino.forward_rgb_features(rgb_img.to(self.device))
            feat_up = F.interpolate(feat, size=(self.img_size, self.img_size),
                                     mode='bilinear', align_corners=False)
            feat_flat = feat_up.squeeze(0).view(feat_up.size(1), -1).T
            from backbones.pas_sampler import compute_anomaly_scores
            scores = compute_anomaly_scores(feat_flat.unsqueeze(0))
            return scores.squeeze(0)

    @staticmethod
    def _compute_density_weight(target_xyz, center_xyz, K=16, sigma=None):
        M = center_xyz.size(0)
        k = min(K, M)
        dist = torch.cdist(target_xyz, center_xyz)
        nn_dist, _ = torch.topk(dist, k, dim=-1, largest=False)
        mean_dist = nn_dist.mean(dim=-1)
        if sigma is None:
            sigma = mean_dist.median().clamp(min=1e-6) * 2.0
        density = torch.exp(-mean_dist / sigma)
        density = density / (density.mean() + 1e-8)
        return density

    def _extract_features_with_kin(self, xyz, anomaly_scores):
        """Extract features with SA1 outputting K_in points.

        Pipeline: sample K_in centers from N → SA1(N→K_in via ball query on N) → SA2(K_in→min(128,K_in)) → SA3(global) → FP
        """
        B, N, _ = xyz.shape
        device = xyz.device
        init_features = xyz.transpose(1, 2).contiguous()  # [B, 3, N]

        # Step 1: Sample K_in centers from N points
        kin = min(self.kin, N)
        if self._strategy == 'fps':
            center_idx = pas_fps(xyz, kin)
        elif self._strategy == 'pas':
            from backbones.pas_sampler import PASSampler
            sampler = PASSampler(tau=self.tau)
            center_idx = sampler(xyz, kin, anomaly_scores)
        elif self._strategy == 'rs':
            center_idx = torch.stack([
                torch.randperm(N, device=device)[:kin] for _ in range(B)
            ], dim=0)
        else:
            center_idx = pas_fps(xyz, kin)

        # Gather K_in center coordinates
        kin_xyz = pas_gather(xyz, center_idx)  # [B, kin, 3]
        if anomaly_scores is not None:
            scores_kin = anomaly_scores[
                torch.arange(B, device=device).view(-1, 1), center_idx]
        else:
            scores_kin = None

        # Step 2: SA1 — from N points, select K_in centers via ball query + MLP
        # This matches the original SA1 behavior: ball query finds nsample neighbors
        # from the FULL point cloud (N points) around each of the K_in centers.
        sa1 = self.bb.sa1
        if _has_cuda:
            xyz_c = xyz.contiguous()            # [B, N, 3] — full cloud for neighbor search
            new_xyz_c = kin_xyz.contiguous()     # [B, kin, 3] — centers
            idx = pointnet2_utils.ball_query(sa1.radius, sa1.nsample, xyz_c, new_xyz_c)
            grouped_xyz = pointnet2_utils.grouping_operation(
                xyz_c.transpose(1, 2).contiguous(), idx)
            grouped_xyz = grouped_xyz - new_xyz_c.transpose(1, 2).unsqueeze(-1)
            # Include initial xyz features (sa1's use_xyz=True expects features)
            grouped_feat = pointnet2_utils.grouping_operation(init_features, idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=1)
            sa1_feat = sa1._sa.mlps[0](grouped)
            sa1_feat = F.max_pool2d(sa1_feat, [1, sa1_feat.size(3)]).squeeze(-1)
        else:
            sa1_feat = sa1._naive_group_mlp(xyz, kin_xyz, init_features)

        sa1_xyz = kin_xyz  # [B, kin, 3]

        # Step 3: SA2 — kin → min(128, kin) (always FPS for geometric levels)
        sa2_xyz, sa2_feat, idx2 = self.bb.sa2(sa1_xyz, sa1_feat, scores_kin, 'fps')

        # Step 4: SA3 — global pooling
        _, sa3_feat, _ = self.bb.sa3(sa2_xyz, sa2_feat, None, 'fps')

        # Step 5: FP layers — upsample back to N
        fp3_feat = self.bb.fp3(sa2_xyz, None, sa2_feat, sa3_feat)
        fp2_feat = self.bb.fp2(sa1_xyz, sa2_xyz, sa1_feat, fp3_feat)
        fp1_feat = self.bb.fp1(xyz, sa1_xyz, init_features, fp2_feat)

        return fp1_feat, sa1_xyz

    @torch.no_grad()
    def build_memory(self, class_name, train_loader):
        print(f"  Building memory bank for '{class_name}' (K_in={self.kin}, strategy={self._strategy})...")
        self.bb.eval()
        all_features = []
        count = 0
        for sample, _ in tqdm(train_loader, desc="    Extracting"):
            xyz_t, nonzero = self._sample_to_pc(sample)
            anomaly_scores = None
            if self._strategy == 'pas':
                scores_full = self._compute_dino_scores(sample[0])
                anomaly_scores = scores_full[nonzero].unsqueeze(0)

            feat, _ = self._extract_features_with_kin(xyz_t, anomaly_scores)
            all_features.append(feat.squeeze(0).T.cpu())
            count += 1
            if count >= self.max_samples:
                break

        all_features = torch.cat(all_features, dim=0)
        n_coreset = min(20000, all_features.size(0))
        if all_features.size(0) > n_coreset:
            idx = torch.randperm(all_features.size(0))[:n_coreset]
            all_features = all_features[idx]

        self.memory_mean = all_features.mean()
        self.memory_std = all_features.std() + 1e-8
        self.memory = (all_features - self.memory_mean) / self.memory_std
        print(f"    Memory bank: {self.memory.shape[0]} x {self.memory.shape[1]}")

    @torch.no_grad()
    def predict(self, sample, mask_sample, label):
        self.bb.eval()
        xyz_t, nonzero = self._sample_to_pc(sample)
        anomaly_scores = None
        if self._strategy == 'pas':
            scores_full = self._compute_dino_scores(sample[0])
            anomaly_scores = scores_full[nonzero].unsqueeze(0)

        per_pt_feat, sa1_xyz = self._extract_features_with_kin(xyz_t, anomaly_scores)

        feat_norm = (per_pt_feat.squeeze(0).T - self.memory_mean.to(self.device))
        feat_norm = feat_norm / (self.memory_std.to(self.device) + 1e-8)
        mem = self.memory.to(self.device)
        min_dist = _batched_cdist_min(feat_norm, mem, chunk_size=4096)

        density_w = self._compute_density_weight(
            xyz_t.squeeze(0), sa1_xyz.squeeze(0), K=self.density_K)
        weighted_score = min_dist * (1.0 + self.density_alpha * density_w)

        img_score = weighted_score.max().item()
        H = W = self.img_size
        score_map = np.zeros(H * W, dtype=np.float32)
        score_map[nonzero] = weighted_score.cpu().numpy()
        score_map = score_map.reshape(H, W)

        mask_np = mask_sample.squeeze().cpu().numpy()
        self.image_preds.append(img_score)
        self.image_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        self.point_scores_list.append(weighted_score.cpu().numpy())
        self.point_labels_list.append(mask_np.reshape(-1)[nonzero])
        self.score_maps.append(score_map)
        self.mask_maps.append(mask_np)

    @torch.no_grad()
    def evaluate(self, class_name, test_loader):
        for sample, mask, label, _ in tqdm(test_loader, desc=f"    Testing"):
            self.predict(sample, mask, label)

        all_scores = np.concatenate(self.point_scores_list)
        all_labels = np.concatenate(self.point_labels_list)
        point_auc = roc_auc_score(all_labels, all_scores)

        img_auc = roc_auc_score(self.image_labels, self.image_preds)

        return {'point_auc': point_auc, 'pro': 0.0, 'iou': 0.0, 'image_auc': img_auc}


def main():
    parser = argparse.ArgumentParser(description='K_in Sensitivity Experiment (Fixed)')
    parser.add_argument('--kin_values', default='50,100,200,512,1024',
                        help='Comma-separated K_in values')
    parser.add_argument('--classes', default='all',
                        help='Comma-separated class names or "all"')
    parser.add_argument('--k_mem', default=100, type=int,
                        help='Memory bank size (fixed across all K_in)')
    parser.add_argument('--tau', default=0.6, type=float)
    parser.add_argument('--density_alpha', default=0.5, type=float)
    parser.add_argument('--output_json', default='results/kin_sensitivity_fixed.json')
    parser.add_argument('--dataset_path', default='datasets/mvtec3d')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    kin_values = [int(x) for x in args.kin_values.split(',')]
    if args.classes == 'all':
        classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    else:
        classes = args.classes.split(',')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seeds(42)

    # Load DINO model (only need the RGB backbone)
    from models.models import Model
    dino = Model(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                 xyz_backbone_name='Point_MAE', group_size=128, num_group=1024)
    dino.to(device).eval()
    print("DINO loaded.")

    from dataset import get_data_loader

    all_results = {}
    start_time = time.time()

    for kin in kin_values:
        print(f"\n{'='*70}")
        print(f"  K_in = {kin}")
        print(f"{'='*70}")

        for cls in classes:
            print(f"\n  Class: {cls}")
            train_loader = get_data_loader('train', cls, 224, args)
            test_loader = get_data_loader('test', cls, 224, args)

            for strategy in ['fps', 'pas']:
                pipe = KinAwarePipeline(
                    dino, kin=kin, tau=args.tau,
                    density_alpha=args.density_alpha,
                    max_samples=args.k_mem, img_size=224)
                pipe.set_strategy(strategy)
                pipe.build_memory(cls, train_loader)
                results = pipe.evaluate(cls, test_loader)
                pipe._reset_metrics()

                key = f"K={kin}/{cls}/{strategy}"
                all_results[key] = {
                    'point_auc': results['point_auc'],
                    'pro': results['pro'],
                    'iou': results['iou'],
                    'image_auc': results['image_auc'],
                    'kin': kin,
                    'class': cls,
                    'strategy': strategy,
                }

                torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: K_in Sensitivity (K_mem fixed={args.k_mem})")
    print(f"{'='*70}")
    print(f"  Time: {elapsed/60:.1f} min\n")

    summary = {}
    for kin in kin_values:
        fps_aucs, pas_aucs = [], []
        for cls in classes:
            fps_key = f"K={kin}/{cls}/fps"
            pas_key = f"K={kin}/{cls}/pas"
            if fps_key in all_results and pas_key in all_results:
                fps_aucs.append(all_results[fps_key]['point_auc'])
                pas_aucs.append(all_results[pas_key]['point_auc'])

        if fps_aucs:
            fps_mean = float(np.mean(fps_aucs))
            pas_mean = float(np.mean(pas_aucs))
            delta = pas_mean - fps_mean
            summary[f"K={kin}"] = {
                'fps_mean': fps_mean,
                'pas_mean': pas_mean,
                'delta': delta,
                'n_classes': len(fps_aucs),
            }
            print(f"  K={kin:>5}: FPS={fps_mean:.4f}, PAS={pas_mean:.4f}, "
                  f"Δ={delta:+.4f} ({delta/fps_mean*100:+.2f}%)")

    # Per-class detail
    print(f"\n  Per-class detail:")
    header = f"  {'Class':<14}"
    for kin in kin_values:
        header += f"  K={kin} FPS  K={kin} PAS"
    print(header)
    print("  " + "-" * len(header))
    for cls in classes:
        line = f"  {cls:<14}"
        for kin in kin_values:
            fps_key = f"K={kin}/{cls}/fps"
            pas_key = f"K={kin}/{cls}/pas"
            fps_v = all_results.get(fps_key, {}).get('point_auc', 0)
            pas_v = all_results.get(pas_key, {}).get('point_auc', 0)
            line += f"  {fps_v:>8.4f}  {pas_v:>8.4f}"
        print(line)

    # Save
    output = {
        '_metadata': {
            'description': 'K_in sensitivity experiment (K_mem fixed)',
            'kin_values': kin_values,
            'k_mem': args.k_mem,
            'classes': classes,
            'tau': args.tau,
            'density_alpha': args.density_alpha,
            'elapsed_seconds': elapsed,
        },
        'per_class': all_results,
        'summary': summary,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
