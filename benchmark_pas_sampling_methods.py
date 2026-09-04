"""PAS vs VDS vs Grid — Independent geometric baselines comparison.

Compares 4 sampling strategies on the same PN2 backbone + per-point scoring head:
  - FPS: pure Farthest Point Sampling (geometric baseline)
  - PAS: Reverse Thought Sampling (ours, 2D-semantic guided)
  - VDS: Voxel Downsampling (geometric, independent of FPS)
  - Grid: Uniform 3D Grid Sampling (geometric, independent of FPS)

The key claim: PAS's advantage comes from cross-modal (2D→3D) guidance.
FPS, VDS, and Grid are three genuinely distinct purely-geometric methods,
ruling out any "non-randomness" alternative explanation.

Output: Point AUC, PRO, IoU, Image AUC per class per method.
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from backbones.pas_sampler import compute_anomaly_scores
from backbones.pointnet2_seg import PointNet2SegBackbone
from backbones.gss_sampler import grid_sampling
from backbones.voxel_sampler import voxel_downsampling
from utils.mvtec3d_util import organized_pc_to_unorganized_pc


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


def _fps_on_pc(xyz, npoint):
    """FPS sampling returning indices. xyz: [B, N, 3], returns [B, npoint]."""
    from backbones.pointnet2_backbone import _fps
    return _fps(xyz, npoint)


def _gather_features(xyz, idx):
    """Gather points by indices. xyz: [B, N, 3], idx: [B, K], returns [B, K, 3]."""
    from backbones.pointnet2_backbone import _gather
    return _gather(xyz, idx)


class SamplingComparisonPipeline:
    """Per-point AD pipeline that can use different sampling strategies on PN2 backbone."""

    def __init__(self, args, dino_extractor=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = args.img_size
        self.use_amp = getattr(args, 'amp', True) and self.device.type == 'cuda'
        self.dino = dino_extractor
        self.backbone = PointNet2SegBackbone(tau=getattr(args, 'tau', 0.6))
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
        self._strategy = 'fps'

    def set_strategy(self, strategy):
        """Set sampling strategy: 'fps', 'pas', 'vds', 'grid'."""
        self._strategy = strategy

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
        print(f"[SamplingCmp] Building memory bank for '{class_name}'...")
        self.backbone.eval()
        all_features = []
        count = 0
        for sample, _ in tqdm(train_loader, desc="  Extracting features"):
            xyz_t, nonzero = self._sample_to_pc(sample)
            B, N, _ = xyz_t.shape
            anomaly_scores = None
            if self._strategy == 'pas':
                scores_full = self._compute_dino_scores(sample[0])
                anomaly_scores = scores_full[nonzero].unsqueeze(0)

            # For VDS/Grid: manually select centers then use PN2 forward with modified sampling
            feat = self._extract_features(xyz_t, anomaly_scores)
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
    def _extract_features(self, xyz, anomaly_scores):
        """Extract per-point features using the current sampling strategy.

        Returns features [B, 64, N] for build_memory compatibility.
        """
        if self._strategy in ('vds', 'grid'):
            feat, _ = self._extract_features_alt_sampling(xyz, anomaly_scores)
            return feat
        else:
            guide = anomaly_scores if self._strategy == 'pas' else None
            feat, _, _, _ = self.backbone(xyz, guide)
            return feat

    @torch.no_grad()
    def _extract_features_alt_sampling(self, xyz, anomaly_scores, precomputed_centers=None):
        """Extract features with VDS or Grid sampling replacing SA1's FPS.

        Returns: (fp1_feat [B, 64, N], sa1_centers [B, 512, 3])
        """
        B, N, _ = xyz.shape
        device = xyz.device

        init_features = xyz.transpose(1, 2).contiguous()  # [B, 3, N]
        npoint_sa1 = min(512, N)

        if precomputed_centers is not None:
            center_idx = precomputed_centers
        elif self._strategy == 'vds':
            center_idx = voxel_downsampling(xyz, npoint_sa1)
        elif self._strategy == 'grid':
            center_idx = grid_sampling(xyz, npoint_sa1)
        else:
            center_idx = _fps_on_pc(xyz, npoint_sa1)

        sa1 = self.backbone.sa1
        new_xyz = _gather_features(xyz, center_idx)  # [B, 512, 3]

        if sa1._has_cuda_sa:
            import pointnet2_ops.pointnet2_utils as pn2u
            xyz_c = xyz.contiguous()
            new_xyz_c = new_xyz.contiguous()
            idx = pn2u.ball_query(sa1.radius, sa1.nsample, xyz_c, new_xyz_c)
            grouped_xyz = pn2u.grouping_operation(xyz_c.transpose(1, 2).contiguous(), idx)
            grouped_xyz = grouped_xyz - new_xyz_c.transpose(1, 2).unsqueeze(-1)
            # Include initial xyz features (sa1's use_xyz=True expects features)
            grouped_feat = pn2u.grouping_operation(init_features, idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=1)
            sa1_feat = sa1._sa.mlps[0](grouped)
            sa1_feat = F.max_pool2d(sa1_feat, [1, sa1_feat.size(3)]).squeeze(-1)
        else:
            sa1_feat = sa1._naive_group_mlp(xyz, new_xyz, init_features)

        sa1_xyz = new_xyz  # [B, 512, 3]

        # === SA2: FPS from sa1_xyz to 128 (no PAS — geometric only for VDS/Grid) ===
        if anomaly_scores is not None:
            scores_512 = anomaly_scores[
                torch.arange(B, device=device).view(-1, 1), center_idx]
        else:
            scores_512 = None
        sa2_xyz, sa2_feat, idx2 = self.backbone.sa2(sa1_xyz, sa1_feat, scores_512, 'fps')

        # === SA3: global ===
        _, sa3_feat, _ = self.backbone.sa3(sa2_xyz, sa2_feat, None, 'fps')

        # === FP3: SA3 → SA2 ===
        fp3_feat = self.backbone.fp3(sa2_xyz, None, sa2_feat, sa3_feat)

        # === FP2: SA2 → SA1 ===
        fp2_feat = self.backbone.fp2(sa1_xyz, sa2_xyz, sa1_feat, fp3_feat)

        # === FP1: SA1 → original N ===
        input_feat = init_features
        fp1_feat = self.backbone.fp1(xyz, sa1_xyz, input_feat, fp2_feat)

        return fp1_feat, sa1_xyz

    @torch.no_grad()
    def predict(self, sample, mask_sample, label):
        self.backbone.eval()
        rgb_img = sample[0]
        xyz_t, nonzero = self._sample_to_pc(sample)
        B, N, _ = xyz_t.shape

        scores_full = self._compute_dino_scores(rgb_img)
        scores_nz = scores_full[nonzero].unsqueeze(0)

        # Extract per-point features and SA1 centers based on strategy
        if self._strategy in ('fps', 'pas'):
            guide = scores_nz if self._strategy == 'pas' else None
            per_pt_feat, sa_xyzs, _, _ = self.backbone(xyz_t, guide)
        else:
            per_pt_feat, sa1_centers_t = self._extract_features_alt_sampling(xyz_t, scores_nz)
            sa_xyzs = [xyz_t, sa1_centers_t]

        sa1_centers = sa_xyzs[1]

        feat_norm = (per_pt_feat.squeeze(0).T - self.memory_mean.to(self.device))
        feat_norm = feat_norm / (self.memory_std.to(self.device) + 1e-8)

        mem = self.memory.to(self.device)
        min_dist = _batched_cdist_min(feat_norm, mem, chunk_size=4096)

        density_w = self._compute_density_weight(
            xyz_t.squeeze(0), sa1_centers.squeeze(0), K=self.density_K)
        weighted_score = min_dist * (1.0 + self.density_alpha * density_w)

        img_score = weighted_score.max().item()

        H = W = self.image_size
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
        print(f"[SamplingCmp] Evaluating '{class_name}' with '{self._strategy}'...")
        for sample, mask, label, _ in tqdm(test_loader, desc=f"  Testing"):
            self.predict(sample, mask, label)

        all_scores = np.concatenate(self.point_scores_list)
        all_labels = np.concatenate(self.point_labels_list)
        point_auc = roc_auc_score(all_labels, all_scores)

        pro_values = []
        for smap, mmap in zip(self.score_maps, self.mask_maps):
            if mmap.sum() > 0:
                pro_values.append(compute_pro(mmap, smap))
        pro = np.mean(pro_values) if pro_values else 0.0

        iou_values = []
        for smap, mmap in zip(self.score_maps, self.mask_maps):
            if mmap.sum() > 0:
                iou_values.append(find_best_threshold(smap, mmap.astype(bool)))
        iou = np.mean(iou_values) if iou_values else 0.0

        img_auc = roc_auc_score(self.image_labels, self.image_preds)

        print(f"\n  Point AUC: {point_auc:.4f}")
        print(f"  PRO:       {pro:.4f}")
        print(f"  IoU:       {iou:.4f}")
        print(f"  Image AUC: {img_auc:.4f}")

        return {'point_auc': point_auc, 'pro': pro, 'iou': iou, 'image_auc': img_auc}

    def reset_metrics(self):
        self.point_scores_list = []
        self.point_labels_list = []
        self.score_maps = []
        self.mask_maps = []
        self.image_preds = []
        self.image_labels = []


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='PAS vs VDS vs Grid — Sampling Methods Comparison')
    parser.add_argument('--dataset', default='mvtec3d', choices=['mvtec3d', 'eyecandies'])
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--max_sample', default=100, type=int)
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch14_dinov2', type=str)
    parser.add_argument('--group_size', default=128, type=int)
    parser.add_argument('--num_group', default=1024, type=int)
    parser.add_argument('--tau', default=0.6, type=float)
    parser.add_argument('--density_alpha', default=0.5, type=float)
    parser.add_argument('--density_K', default=16, type=int)
    parser.add_argument('--strategies', nargs='+', default=['fps', 'pas', 'vds', 'grid'],
                        type=str, help='Sampling strategies to compare')
    parser.add_argument('--classes', nargs='+', default=None, type=str)
    parser.add_argument('--amp', default=True, action='store_true')
    parser.add_argument('--no_amp', dest='amp', action='store_false')
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = 'datasets/mvtec3d' if args.dataset == 'mvtec3d' else 'datasets/eyecandies_preprocessed'
    return args


def run_benchmark(args):
    from dataset import get_data_loader, mvtec3d_classes, eyecandies_classes
    from models.models import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seeds(0)

    print("Loading DINO backbone...")
    dino_model = Model(
        device=device,
        rgb_backbone_name=args.rgb_backbone_name,
        xyz_backbone_name='Point_MAE',
        group_size=args.group_size,
        num_group=args.num_group,
    )
    dino_model.to(device).eval()

    classes = args.classes if args.classes else (
        mvtec3d_classes() if args.dataset == 'mvtec3d' else eyecandies_classes())
    strategies = args.strategies

    all_results = {s: {} for s in strategies}

    for cls_name in classes:
        print(f"\n{'=' * 60}")
        print(f"  Sampling Methods Comparison: {cls_name}")
        print(f"{'=' * 60}")

        train_loader = get_data_loader('train', cls_name, args.img_size, args)
        test_loader = get_data_loader('test', cls_name, args.img_size, args)

        for strategy in strategies:
            print(f"\n  [{strategy.upper()}] Sampling...")
            pipe = SamplingComparisonPipeline(args, dino_extractor=dino_model)
            pipe.set_strategy(strategy)
            pipe.build_memory(cls_name, train_loader)
            res = pipe.evaluate(cls_name, test_loader)
            all_results[strategy][cls_name] = res
            pipe.reset_metrics()
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"  Sampling Methods Comparison — Summary")
    print(f"{'=' * 90}")

    try:
        import pandas as pd
        rows = []
        for cls_name in classes:
            row = {'Class': cls_name}
            base = all_results[strategies[0]][cls_name]['point_auc']
            for s in strategies:
                r = all_results[s][cls_name]
                row[f'{s.upper()}-PtAUC'] = f"{r['point_auc']:.4f}"
                row[f'{s.upper()}-PRO'] = f"{r['pro']:.4f}"
                row[f'{s.upper()}-IoU'] = f"{r['iou']:.4f}"
            rows.append(row)

        # Mean
        mean_row = {'Class': 'MEAN'}
        for s in strategies:
            mean_ptauc = np.mean([all_results[s][c]['point_auc'] for c in classes])
            mean_pro = np.mean([all_results[s][c]['pro'] for c in classes])
            mean_iou = np.mean([all_results[s][c]['iou'] for c in classes])
            mean_row[f'{s.upper()}-PtAUC'] = f"{mean_ptauc:.4f}"
            mean_row[f'{s.upper()}-PRO'] = f"{mean_pro:.4f}"
            mean_row[f'{s.upper()}-IoU'] = f"{mean_iou:.4f}"
        rows.append(mean_row)

        df = pd.DataFrame(rows)
        print(df.to_markdown(index=False))

        # Statistical tests
        print(f"\n  Pairwise vs PAS (PtAUC):")
        for s in strategies:
            if s == 'pas':
                continue
            deltas = [all_results['pas'][c]['point_auc'] - all_results[s][c]['point_auc']
                      for c in classes]
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
            print(f"    PAS vs {s.upper()}: delta={np.mean(deltas):+.4f}, p={p_val:.4f}")

    except ImportError:
        for cls_name in classes:
            print(f"  {cls_name}:")
            for s in strategies:
                r = all_results[s][cls_name]
                print(f"    {s.upper()} — PtAUC={r['point_auc']:.4f} PRO={r['pro']:.4f} "
                      f"IoU={r['iou']:.4f} ImgAUC={r['image_auc']:.4f}")

    return all_results


if __name__ == '__main__':
    args = get_args()
    run_benchmark(args)
