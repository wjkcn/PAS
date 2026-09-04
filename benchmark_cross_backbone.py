"""Cross-backbone sampling comparison: N backbones × M methods.

Runs RS, FPS, PAS (and optionally more strategies) across multiple 3D backbones.
Produces a backbone × method matrix for the paper's cross-backbone table.

Usage:
    # 5 backbone × 3 method core experiment
    python benchmark_cross_backbone.py --backbones pn2,dgcnn,pointmae,pointnet,pct --strategies fps,pas,rs

    # Single backbone all methods
    python benchmark_cross_backbone.py --backbones pct --strategies rs,fps,ffps,pas,vds,grid,dafps,curv

Backbone key → class mapping:
    pn2       → PointNet2SegBackbone (hierarchical)
    dgcnn     → DGCNNSegBackbone (graph)
    pointmae  → PointMAEPerPointBackbone (transformer/MAE)
    pointnet  → PointNetSegBackbone (vanilla MLP)
    pct       → PCTSegBackbone (transformer/offset-attention)
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from backbones.pas_sampler import compute_anomaly_scores
from utils.mvtec3d_util import organized_pc_to_unorganized_pc


# ── Backbone registry ──────────────────────────────────────────────

def _make_pn2(tau):
    from backbones.pointnet2_seg import PointNet2SegBackbone
    return PointNet2SegBackbone(tau=tau)


def _make_dgcnn(tau):
    from backbones.dgcnn_seg import DGCNNSegBackbone
    return DGCNNSegBackbone(tau=tau)


def _make_pointmae(tau):
    from backbones.pointmae_perpoint import PointMAEPerPointBackbone
    return PointMAEPerPointBackbone(tau=tau)


def _make_pointnet(tau):
    from backbones.pointnet_seg import PointNetSegBackbone
    return PointNetSegBackbone(tau=tau, ncenter=512)


def _make_pct(tau):
    from backbones.pct_seg import PCTSegBackbone
    return PCTSegBackbone(tau=tau, ncenter=512, dim=256, num_blocks=4)


BACKBONE_REGISTRY = {
    'pn2':      _make_pn2,
    'dgcnn':    _make_dgcnn,
    'pointmae': _make_pointmae,
    'pointnet': _make_pointnet,
    'pct':      _make_pct,
}


# ── Metric helpers ──────────────────────────────────────────────────

def _batched_cdist_min(x, y, chunk_size=4096):
    N = x.size(0)
    mins = []
    for i in range(0, N, chunk_size):
        chunk = x[i:i + chunk_size]
        d = torch.cdist(chunk, y)
        mins.append(d.min(dim=1)[0])
    return torch.cat(mins, dim=0)


def compute_iou(mask_gt, mask_pred):
    intersection = (mask_gt & mask_pred).sum()
    union = (mask_gt | mask_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_pro(mask_gt, score_map):
    from skimage import measure
    mask_gt = mask_gt.astype(bool)
    if mask_gt.sum() == 0:
        return 0.0
    labeled = measure.label(mask_gt, connectivity=2)
    regions = np.unique(labeled)
    regions = regions[regions > 0]
    if len(regions) == 0:
        return 0.0
    best_pro = 0.0
    for pct in range(50, 100):
        thresh = np.percentile(score_map, pct)
        mask_pred = score_map > thresh
        overlaps = []
        for r in regions:
            region_mask = labeled == r
            region_pred = mask_pred[region_mask]
            if region_pred.sum() > 0:
                overlaps.append(region_pred.sum() / region_mask.sum())
            else:
                overlaps.append(0.0)
        pro_val = np.mean(overlaps)
        if pro_val > best_pro:
            best_pro = pro_val
    return best_pro


def find_best_threshold(scores, gt_mask):
    best_iou = 0.0
    for pct in range(50, 100):
        thresh = np.percentile(scores, pct)
        pred = scores > thresh
        iou = compute_iou(gt_mask, pred)
        if iou > best_iou:
            best_iou = iou
    return best_iou


# ── Generic Cross-Backbone Pipeline ─────────────────────────────────

class CrossBackbonePipeline:
    """Per-point AD pipeline that works with any backbone in the registry.

    Uses backbone.forward(xyz, anomaly_scores) for feature extraction.
    Sets sampling mode via backbone.set_sampling_mode() before forward.
    """

    def __init__(self, args, backbone, dino_extractor=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = args.img_size
        self.use_amp = getattr(args, 'amp', True) and self.device.type == 'cuda'
        self.dino = dino_extractor
        self.backbone = backbone
        self.backbone.to(self.device).eval()
        self.memory = None
        self.memory_mean = 0.0
        self.memory_std = 1.0
        self.density_alpha = getattr(args, 'density_alpha', 0.5)
        self.density_K = getattr(args, 'density_K', 16)
        self._max_samples = args.max_sample
        self.point_scores_list = []
        self.point_labels_list = []
        self.score_maps = []
        self.mask_maps = []
        self.image_preds = []
        self.image_labels = []
        self.max_points = getattr(args, 'max_points', None)
        self._strategy = 'fps'

    def set_strategy(self, strategy):
        self._strategy = strategy
        # Map strategy to backbone sampling mode
        # fps/pas/rs and the 5 alternative geometric methods
        mode_map = {'fps': 'fps', 'pas': 'pas', 'rs': 'rs',
                    'vds': 'vds', 'grid': 'grid', 'dafps': 'dafps',
                    'curv': 'curv', 'ffps': 'ffps'}
        if strategy in mode_map:
            self.backbone.set_sampling_mode(mode_map[strategy])

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

    def _sample_to_pc(self, sample):
        organized_pc = sample[1].squeeze().permute(1, 2, 0).numpy()
        unorganized = organized_pc_to_unorganized_pc(organized_pc)
        nonzero = np.nonzero(np.all(unorganized != 0, axis=1))[0]
        clean = torch.tensor(unorganized[nonzero, :], dtype=torch.float32)

        # Cap points for backbones with O(N²) ops (e.g. DGCNN EdgeConv)
        if self.max_points is not None and len(clean) > self.max_points:
            perm = torch.randperm(len(clean))[:self.max_points]
            clean = clean[perm]
            nonzero = nonzero[perm.numpy()]

        return clean.unsqueeze(0).to(self.device), nonzero

    def _compute_dino_scores(self, rgb_img):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            feat = self.dino.forward_rgb_features(rgb_img.to(self.device))
            feat_up = F.interpolate(feat, size=(self.image_size, self.image_size),
                                     mode='bilinear', align_corners=False)
            feat_flat = feat_up.squeeze(0).view(feat_up.size(1), -1).T
            scores = compute_anomaly_scores(feat_flat.unsqueeze(0))
            return scores.squeeze(0)

    @torch.no_grad()
    def build_memory(self, class_name, train_loader):
        print(f"  Building memory bank...")
        self.backbone.eval()
        all_features = []
        count = 0
        for sample, _ in tqdm(train_loader, desc="  Extracting"):
            xyz_t, nonzero = self._sample_to_pc(sample)
            anomaly_scores = None
            if self._strategy == 'pas':
                scores_full = self._compute_dino_scores(sample[0])
                anomaly_scores = scores_full[nonzero].unsqueeze(0)

            feat, _, _, _ = self.backbone(xyz_t, anomaly_scores)
            all_features.append(feat.squeeze(0).T.cpu())
            count += 1
            if count >= self._max_samples:
                break

        all_features = torch.cat(all_features, dim=0)
        n_coreset = min(20000, all_features.size(0))
        if all_features.size(0) > n_coreset:
            idx = torch.randperm(all_features.size(0))[:n_coreset]
            all_features = all_features[idx]

        self.memory_mean = all_features.mean()
        self.memory_std = all_features.std() + 1e-8
        self.memory = (all_features - self.memory_mean) / self.memory_std
        print(f"  Memory bank: {self.memory.shape[0]} x {self.memory.shape[1]}")

    @torch.no_grad()
    def predict(self, sample, mask_sample, label):
        self.backbone.eval()
        rgb_img = sample[0]
        xyz_t, nonzero = self._sample_to_pc(sample)
        B, N, _ = xyz_t.shape

        scores_full = self._compute_dino_scores(rgb_img)
        scores_nz = scores_full[nonzero].unsqueeze(0)

        # Extract features using backbone's current sampling mode
        guide = scores_nz if self._strategy == 'pas' else None
        per_pt_feat, sa_xyzs, _, sa_indices = self.backbone(xyz_t, guide)
        sa1_centers = sa_xyzs[1]
        sa1_idx = sa_indices[0]  # [B, K] for retention tracking

        feat_norm = (per_pt_feat.squeeze(0).T - self.memory_mean.to(self.device))
        feat_norm = feat_norm / (self.memory_std.to(self.device) + 1e-8)

        mem = self.memory.to(self.device)
        min_dist = _batched_cdist_min(feat_norm, mem, chunk_size=4096)

        if self.density_alpha > 0:
            density_w = self._compute_density_weight(
                xyz_t.squeeze(0), sa1_centers.squeeze(0), K=self.density_K)
            weighted_score = min_dist * (1.0 + self.density_alpha * density_w)
        else:
            weighted_score = min_dist

        img_score = weighted_score.detach().max().item()

        H = W = self.image_size
        score_map = np.zeros(H * W, dtype=np.float32)
        score_map[nonzero] = weighted_score.detach().cpu().numpy()
        score_map = score_map.reshape(H, W)

        mask_np = mask_sample.squeeze().cpu().numpy()

        self.image_preds.append(img_score)
        self.image_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        self.point_scores_list.append(weighted_score.detach().cpu().numpy())
        self.point_labels_list.append(mask_np.reshape(-1)[nonzero])
        self.score_maps.append(score_map)
        self.mask_maps.append(mask_np)

        # Retention tracking
        if not hasattr(self, '_sa1_indices_list'):
            self._sa1_indices_list = []
            self._fg_labels_list = []
        if sa1_idx is not None:
            self._sa1_indices_list.append(sa1_idx.cpu().numpy())
            fg_labels = mask_np.reshape(-1)[nonzero]
            self._fg_labels_list.append(fg_labels)


# ── DINO model loader ───────────────────────────────────────────────

def load_dino_extractor(device):
    from models.models import Model
    dino = Model(
        device=device,
        rgb_backbone_name='vit_base_patch14_dinov2',
        xyz_backbone_name='Point_MAE',
        group_size=128, num_group=1024,
    )
    dino.to(device).eval()
    return dino


# ── Main ────────────────────────────────────────────────────────────

class Args:
    img_size = 224
    dataset_path = 'datasets/mvtec3d'
    max_sample = 100
    tau = 0.6
    density_alpha = 0.5
    amp = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', default='pn2,dgcnn,pointmae,pointnet,pct',
                        help='Comma-separated backbone keys')
    parser.add_argument('--strategies', default='fps,pas,rs',
                        help='Comma-separated strategies (fps,pas,rs)')
    parser.add_argument('--classes', default='all',
                        help='Classes to evaluate')
    parser.add_argument('--max_sample', type=int, default=100)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--output_json', default='results/cross_backbone_results.json')
    parser.add_argument('--include_retention', action='store_true', default=True)
    args_p = parser.parse_args()

    backbone_keys = args_p.backbones.split(',')
    strategies = args_p.strategies.split(',')
    if args_p.classes == 'all':
        classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    else:
        classes = args_p.classes.split(',')

    base_args = Args()
    base_args.max_sample = args_p.max_sample
    base_args.tau = args_p.tau

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seeds(42)

    print("=" * 80)
    print(f"Cross-Backbone Benchmark: {len(backbone_keys)} backbones × {len(strategies)} strategies")
    print(f"Backbones: {backbone_keys}")
    print(f"Strategies: {strategies}")
    print(f"Classes: {len(classes)}")
    print("=" * 80)

    # Load DINO once
    print("\nLoading DINO backbone...")
    dino = load_dino_extractor(device)
    print("DINO loaded.\n")

    from dataset import get_data_loader

    all_results = {}

    prev_backbone = None
    for backbone_key in backbone_keys:
        bb_name = backbone_key.upper()
        print(f"\n{'#' * 80}")
        print(f"# Backbone: {bb_name}")
        print(f"{'#' * 80}")

        # Clean up previous backbone
        if prev_backbone is not None:
            del prev_backbone
            torch.cuda.empty_cache()

        make_fn = BACKBONE_REGISTRY[backbone_key]
        backbone = make_fn(tau=args_p.tau)

        for strategy in strategies:
            print(f"\n  [{bb_name} / {strategy.upper()}]")

            # Reload backbone per strategy to avoid state pollution
            if backbone is not None:
                del backbone
                torch.cuda.empty_cache()
            backbone = make_fn(tau=args_p.tau)

            # DGCNN EdgeConv uses torch.cdist O(N²) but fits in ~10 GB for 50k pts
            base_args.max_points = None

            pipeline = CrossBackbonePipeline(base_args, backbone, dino_extractor=dino)
            pipeline.set_strategy(strategy)

            for class_name in classes:
                print(f"    Class: {class_name}")

                train_loader = get_data_loader('train', class_name, base_args.img_size, base_args)
                test_loader = get_data_loader('test', class_name, base_args.img_size, base_args)

                # Build memory bank
                pipeline.build_memory(class_name, train_loader)

                # Evaluate
                pipeline.point_scores_list = []
                pipeline.point_labels_list = []
                pipeline.score_maps = []
                pipeline.mask_maps = []
                pipeline.image_preds = []
                pipeline.image_labels = []
                if hasattr(pipeline, '_sa1_indices_list'):
                    pipeline._sa1_indices_list = []
                    pipeline._fg_labels_list = []

                for sample, mask_sample, label, _ in tqdm(test_loader, desc=f"    Testing"):
                    pipeline.predict(sample, mask_sample, label)

                # Compute metrics
                point_scores = np.concatenate(pipeline.point_scores_list)
                point_labels = np.concatenate(pipeline.point_labels_list)

                if point_labels.sum() > 0 and point_labels.sum() < len(point_labels):
                    point_auc = roc_auc_score(point_labels, point_scores)
                else:
                    point_auc = 0.5

                pro_vals = [compute_pro(gt.reshape(224, 224), sm.reshape(224, 224))
                           for gt, sm in zip(pipeline.mask_maps, pipeline.score_maps)]
                iou_vals = [find_best_threshold(s, m.astype(bool))
                           for s, m in zip(pipeline.score_maps, pipeline.mask_maps)]

                image_labels_np = np.array([l.item() if hasattr(l, 'item') else l
                                            for l in pipeline.image_labels])
                image_preds_np = np.array(pipeline.image_preds)
                if len(np.unique(image_labels_np)) > 1:
                    image_auc = roc_auc_score(image_labels_np, image_preds_np)
                else:
                    image_auc = 0.5

                # Retention ratio
                retention = None
                if hasattr(pipeline, '_sa1_indices_list') and pipeline._sa1_indices_list:
                    from backbones.metrics_extended import compute_retention_ratio_batched
                    mean_ret, mean_base, _ = compute_retention_ratio_batched(
                        pipeline._fg_labels_list, pipeline._sa1_indices_list)
                    retention = float(mean_ret)

                key = f"{backbone_key}/{strategy}/{class_name}"
                all_results[key] = {
                    'point_auc': float(point_auc),
                    'pro': float(np.mean(pro_vals)) if pro_vals else 0.0,
                    'iou': float(np.mean(iou_vals)) if iou_vals else 0.0,
                    'image_auc': float(image_auc),
                    'retention': retention,
                }

                print(f"      PtAUC={point_auc:.4f}  PRO={all_results[key]['pro']:.4f}  "
                      f"IoU={all_results[key]['iou']:.4f}  ImgAUC={image_auc:.4f}"
                      + (f"  Ret={retention:.3f}" if retention else ""))

                torch.cuda.empty_cache()

        prev_backbone = backbone
        backbone = None

    # ── Summary tables ──
    print(f"\n{'=' * 80}")
    print("  CROSS-BACKBONE SUMMARY")
    print(f"{'=' * 80}")

    # Per-backbone × strategy mean PtAUC
    print(f"\n  Mean PtAUC — {len(backbone_keys)} backbones × {len(strategies)} strategies:")
    header = f"  {'Backbone':<12}"
    for s in strategies:
        header += f" {s.upper():>10}"
    print(header)
    print(f"  {'-' * (14 + 11 * len(strategies))}")
    for bb in backbone_keys:
        row = f"  {bb.upper():<12}"
        for s in strategies:
            vals = [all_results[f"{bb}/{s}/{c}"]['point_auc'] for c in classes
                    if f"{bb}/{s}/{c}" in all_results]
            if vals:
                row += f" {np.mean(vals):>10.4f}"
            else:
                row += f" {'---':>10}"
        print(row)

    # PAS vs FPS delta per backbone
    print(f"\n  PAS − FPS Δ:")
    for bb in backbone_keys:
        fps_vals = [all_results[f"{bb}/fps/{c}"]['point_auc'] for c in classes
                    if f"{bb}/fps/{c}" in all_results]
        pas_vals = [all_results[f"{bb}/pas/{c}"]['point_auc'] for c in classes
                    if f"{bb}/pas/{c}" in all_results]
        if fps_vals and pas_vals:
            delta = np.mean(pas_vals) - np.mean(fps_vals)
            wins = sum(1 for c in classes
                      if f"{bb}/fps/{c}" in all_results and f"{bb}/pas/{c}" in all_results
                      and all_results[f"{bb}/pas/{c}"]['point_auc'] > all_results[f"{bb}/fps/{c}"]['point_auc'])
            print(f"    {bb.upper():<12} Δ=+{delta:.4f}  wins={wins}/{len(classes)}")

    # Save
    os.makedirs(os.path.dirname(args_p.output_json) if os.path.dirname(args_p.output_json) else '.', exist_ok=True)
    with open(args_p.output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {args_p.output_json}")


if __name__ == '__main__':
    main()
