"""BTF-style FPFH Memory Bank Plug-in Experiment — Complete Pipeline.

Follows the BTF (Back to the Feature) paper's complete pipeline:
  Training: FPFH on ALL points → sampling selects which features enter memory bank → coreset
  Testing:  FPFH on ALL points → every point scored by nn distance to memory bank

This is the CORRECT plug-in experiment: sampling only affects "what's in the
memory bank", not "which points get scored". If PAS improves here, it proves
PAS selects better features for the memory bank, not just better spatial coverage.

Uses Open3D's FPFH implementation for speed and accuracy.

Metrics: PtAUC, PRO, IoU, ImgAUC, Retention
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from utils.mvtec3d_util import (
    organized_pc_to_unorganized_pc, read_tiff_organized_pc, resize_organized_pc
)
from backbones.pas_sampler import PASSampler, _native_fps
from backbones.curvature_sampler import curvature_guided_sampling
from backbones.random_sampler import random_sampling


# ─────────────────────────────────────────────────────────────────────────────
# Open3D-based FPFH
# ─────────────────────────────────────────────────────────────────────────────

def xyz_to_o3d_pointcloud(xyz):
    """Convert numpy [N,3] to Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def compute_normals_o3d(xyz, radius=None, k=32):
    """Estimate normals using Open3D.

    Args:
        xyz: [N, 3] numpy array
        radius: search radius (None = use kNN)
        k: number of neighbors if radius is None

    Returns:
        normals: [N, 3] numpy array
    """
    pcd = xyz_to_o3d_pointcloud(xyz)
    if radius is not None:
        o3d.geometry.PointCloud.estimate_normals(
            pcd, search_param=o3d.geometry.KDTreeSearchParamRadius(radius))
    else:
        o3d.geometry.PointCloud.estimate_normals(
            pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(k))
    # Orient normals consistently
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0, 0, 0]))
    return np.asarray(pcd.normals)


def compute_fpfh_o3d(xyz, normals=None, voxel_size=None, k=32, radius=None):
    """Compute FPFH features using Open3D.

    Args:
        xyz: [N, 3]
        normals: [N, 3] (computed if None)
        voxel_size: for Open3D's radius estimation (None = use k/radius directly)
        k: neighbors for normal estimation
        radius: search radius for FPFH (if None, estimated from data)

    Returns:
        fpfh: Open3d FPFHFeature object, use np.asarray() to get [N, 33] array
    """
    pcd = xyz_to_o3d_pointcloud(xyz)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        o3d.geometry.PointCloud.estimate_normals(
            pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(k))
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0, 0, 0]))

    # Estimate radius if not provided
    if radius is None:
        # Use mean distance to k-th neighbor as radius
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=k)
        radius = float(np.mean(dists[:, -1])) * 2.5  # 2.5x mean kNN distance

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
    )
    return fpfh


# ─────────────────────────────────────────────────────────────────────────────
# Sampling wrappers
# ─────────────────────────────────────────────────────────────────────────────

def fps_sampling(xyz, npoint):
    return _native_fps(xyz, npoint)


def pas_sampling(xyz, npoint, anomaly_scores, tau=0.6):
    sampler = PASSampler(tau=tau)
    return sampler(xyz, npoint, anomaly_scores)


def curv_sampling(xyz, npoint):
    return curvature_guided_sampling(xyz, npoint, k=32, tau=0.5)


def rs_sampling(xyz, npoint):
    return random_sampling(xyz, npoint)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Bank
# ─────────────────────────────────────────────────────────────────────────────

class FPFHMemoryBank:
    """PatchCore-style memory bank for FPFH features.

    Training: add features from sampled points, then build coreset.
    Testing: score ALL points via nn distance to coreset.
    """

    def __init__(self, f_coreset=1.0):
        self.f_coreset = f_coreset
        self.memory_bank = None
        self.all_features = []

    def add(self, features):
        """Add features. features: [K, D] (from sampled points)"""
        self.all_features.append(features.astype(np.float64))

    def build(self):
        """Build coreset from collected features."""
        all_feat = np.concatenate(self.all_features, axis=0)
        print(f"  Memory bank: {all_feat.shape[0]} features, dim={all_feat.shape[1]}")

        if self.f_coreset < 1.0:
            n_select = max(1, int(len(all_feat) * self.f_coreset))
            idx = self._greedy_coreset(all_feat, n_select)
            self.memory_bank = all_feat[idx]
        else:
            self.memory_bank = all_feat

        self._tree = cKDTree(self.memory_bank)
        print(f"  Coreset: {self.memory_bank.shape[0]} features retained")

    def _greedy_coreset(self, features, n_select):
        n = len(features)
        if n <= n_select:
            return np.arange(n)
        rp = SparseRandomProjection(n_components=min(128, features.shape[1]), random_state=42)
        projected = rp.fit_transform(features)
        selected = [np.random.randint(n)]
        min_dist = np.full(n, np.inf)
        for _ in range(n_select - 1):
            last = projected[selected[-1]]
            d = np.sqrt(np.sum((projected - last) ** 2, axis=1))
            min_dist = np.minimum(min_dist, d)
            selected.append(np.argmax(min_dist))
        return np.array(selected)

    def score(self, features):
        """Score features via nn distance. features: [M, D] → scores: [M]"""
        dists, _ = self._tree.query(features, k=1)
        return dists.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / max(union, 1)


def find_best_threshold(score_map, gt_mask, n_thresh=100):
    thresholds = np.linspace(score_map.min(), score_map.max(), n_thresh)
    best_iou = 0
    for t in thresholds:
        iou = compute_iou(score_map > t, gt_mask)
        if iou > best_iou:
            best_iou = iou
    return best_iou


def compute_pro(score_map, gt_mask, n_thresh=200):
    from skimage import measure
    if gt_mask.sum() == 0:
        return 0.0
    gt_labels = measure.label(gt_mask.astype(int))
    n_regions = gt_labels.max()
    if n_regions == 0:
        return 0.0
    thresholds = np.linspace(0, 1, n_thresh)
    pro_curve = []
    for t in thresholds:
        pred = score_map > t
        overlaps = []
        for r in range(1, n_regions + 1):
            rm = gt_labels == r
            sz = rm.sum()
            if sz > 0:
                overlaps.append((pred & rm).sum() / sz)
        pro_curve.append(np.mean(overlaps) if overlaps else 0.0)
    fpr = np.linspace(0, 1, n_thresh)
    mask = fpr <= 0.3
    return float(np.trapz(np.array(pro_curve)[mask], fpr[mask])) if mask.sum() >= 2 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_point_clouds(class_name, split, dataset_path='datasets/mvtec3d', img_size=224):
    import glob
    from PIL import Image
    cls_path = os.path.join(dataset_path, class_name, split)
    samples = []

    def _load(xyz_path, gt_path=None, label=0):
        org = read_tiff_organized_pc(xyz_path)
        resized = resize_organized_pc(org, target_height=img_size, target_width=img_size, tensor_out=False)
        xyz = organized_pc_to_unorganized_pc(resized)
        valid = np.abs(xyz).sum(axis=1) > 1e-6
        xyz_valid = xyz[valid]
        gt_mask = None
        if gt_path and os.path.exists(gt_path):
            gt = np.array(Image.open(gt_path).convert('L').resize((img_size, img_size), resample=0))
            gt_mask = gt > 127
        return (xyz_valid, resized, gt_mask, label, xyz_path, valid)

    if split == 'train':
        for p in sorted(glob.glob(os.path.join(cls_path, 'good', 'xyz', '*.tiff'))):
            samples.append(_load(p, None, 0))
    else:
        for dt in os.listdir(cls_path):
            xyz_paths = sorted(glob.glob(os.path.join(cls_path, dt, 'xyz', '*.tiff')))
            if dt == 'good':
                for p in xyz_paths:
                    samples.append(_load(p, None, 0))
            else:
                gt_paths = sorted(glob.glob(os.path.join(cls_path, dt, 'gt', '*.png')))
                for i, p in enumerate(xyz_paths):
                    gt_p = gt_paths[i] if i < len(gt_paths) else None
                    samples.append(_load(p, gt_p, 1))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline — Complete BTF-style
# ─────────────────────────────────────────────────────────────────────────────

def run_btf_experiment(class_name, strategy, args, dino_model=None):
    """Run complete BTF-style experiment for one class and one strategy.

    Pipeline:
      Train: for each training sample → compute FPFH on ALL points →
             sample K points (using strategy) → add their FPFH to memory bank → coreset
      Test:  for each test sample → compute FPFH on ALL points →
             score ALL points against memory bank → per-point anomaly map
    """
    npoint = args.npoint
    tau = args.tau
    img_size = 224

    # Load data
    train_samples = load_point_clouds(class_name, 'train', args.dataset_path, img_size)
    test_samples = load_point_clouds(class_name, 'test', args.dataset_path, img_size)
    if not train_samples or not test_samples:
        return None

    # Estimate FPFH radius from first training sample
    first_xyz = train_samples[0][0]
    tree0 = cKDTree(first_xyz)
    dists0, _ = tree0.query(first_xyz, k=32)
    fpfh_radius = float(np.mean(dists0[:, -1])) * 2.5
    print(f"  FPFH radius: {fpfh_radius:.4f} (from k=32 mean distance)")

    # ── Training Phase ──
    # Each strategy gets its own memory bank (sampling affects what enters the bank)
    mem_bank = FPFHMemoryBank(f_coreset=args.f_coreset)

    # For PAS training: compute DINO anomaly scores on training samples
    train_anomaly_scores = {}
    if strategy == 'pas' and dino_model is not None:
        print(f"  Computing DINO anomaly scores for PAS training...")
        from dataset import get_data_loader
        train_loader = get_data_loader('train', class_name, img_size, args)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="    DINO train", leave=False)):
            (rgb, _, _), _ = batch  # train returns (data_tuple, label)
            rgb = rgb.to(args.device)
            with torch.no_grad():
                feat = dino_model.forward_rgb_features(rgb)
                B, C, H, W = feat.shape
                flat = feat.view(B, C, -1).permute(0, 2, 1)
                mean = flat.mean(dim=1, keepdim=True)
                cos = torch.nn.functional.cosine_similarity(flat, mean, dim=-1)
                train_anomaly_scores[batch_idx] = (1.0 - cos).view(B, H, W).cpu().numpy()

    print(f"  Building memory bank from {len(train_samples)} training samples...")
    for sample_idx, (xyz, org_pc, _, _, path, valid_mask) in enumerate(
            tqdm(train_samples, desc="    Train", leave=False)):
        if len(xyz) < 33:
            continue

        normals = compute_normals_o3d(xyz, k=32)
        fpfh_obj = compute_fpfh_o3d(xyz, normals=normals, k=32, radius=fpfh_radius)
        fpfh_all = np.asarray(fpfh_obj.data).T

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0).to(device)
        K = min(npoint, len(xyz))

        if strategy == 'pas' and sample_idx in train_anomaly_scores:
            # PAS: use DINO anomaly scores even for training
            scores_2d = train_anomaly_scores[sample_idx][0]
            H_org, W_org = org_pc.shape[:2]
            if scores_2d.shape != (H_org, W_org):
                from scipy.ndimage import zoom
                scores_2d = zoom(scores_2d, (H_org / scores_2d.shape[0], W_org / scores_2d.shape[1]))
            scores_flat = scores_2d.reshape(-1)
            scores_valid = scores_flat[valid_mask]
            a_scores = scores_valid[:len(xyz)] if len(scores_valid) >= len(xyz) else np.zeros(len(xyz))
            a_tensor = torch.from_numpy(a_scores).float().unsqueeze(0).to(device)
            idx = pas_sampling(xyz_tensor, K, a_tensor, tau=tau).squeeze(0).cpu().numpy()
        elif strategy == 'curv':
            idx = curv_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()
        elif strategy == 'rs':
            idx = rs_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()
        else:
            idx = fps_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()

        idx = np.clip(idx, 0, len(xyz) - 1)
        mem_bank.add(fpfh_all[idx])

    mem_bank.build()

    # ── Pre-compute DINO anomaly scores for PAS (test only) ──
    anomaly_scores_map = {}
    if strategy == 'pas' and dino_model is not None:
        print(f"  Computing DINO anomaly scores for PAS...")
        from dataset import get_data_loader
        test_loader = get_data_loader('test', class_name, img_size, args)
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="    DINO", leave=False)):
            (rgb, _, _), _, _, _ = batch
            rgb = rgb.to(args.device)
            with torch.no_grad():
                feat = dino_model.forward_rgb_features(rgb)
                B, C, H, W = feat.shape
                flat = feat.view(B, C, -1).permute(0, 2, 1)
                mean = flat.mean(dim=1, keepdim=True)
                cos = torch.nn.functional.cosine_similarity(flat, mean, dim=-1)
                anomaly_scores_map[batch_idx] = (1.0 - cos).view(B, H, W).cpu().numpy()

    # ── Testing Phase ──
    all_point_scores = []
    all_point_labels = []
    all_score_maps = []
    all_gt_maps = []
    all_image_preds = []
    all_image_labels = []
    all_retention = []

    for sample_idx, (xyz, org_pc, gt_mask, label, path, valid_mask) in enumerate(
            tqdm(test_samples, desc=f"    Test [{strategy}]", leave=False)):
        if len(xyz) < 33:
            continue

        # Compute FPFH on ALL test points
        normals = compute_normals_o3d(xyz, k=32)
        fpfh_obj = compute_fpfh_o3d(xyz, normals=normals, k=32, radius=fpfh_radius)
        fpfh_all = np.asarray(fpfh_obj.data).T  # [N, 33]

        # Score ALL points against memory bank
        all_scores = mem_bank.score(fpfh_all)  # [N]

        # Normalize scores to [0, 1]
        s_min, s_max = all_scores.min(), all_scores.max()
        if s_max - s_min > 1e-10:
            all_scores = (all_scores - s_min) / (s_max - s_min)

        # Image-level score: max of all point scores
        image_score = float(all_scores.max())

        # For retention: also get sampled indices (to measure sampling quality)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0).to(device)
        K = min(npoint, len(xyz))

        if strategy == 'pas' and sample_idx in anomaly_scores_map:
            scores_2d = anomaly_scores_map[sample_idx][0]
            H_org, W_org = org_pc.shape[:2]
            if scores_2d.shape != (H_org, W_org):
                from scipy.ndimage import zoom
                scores_2d = zoom(scores_2d, (H_org / scores_2d.shape[0], W_org / scores_2d.shape[1]))
            scores_flat = scores_2d.reshape(-1)
            scores_valid = scores_flat[valid_mask]
            a_scores = scores_valid[:len(xyz)] if len(scores_valid) >= len(xyz) else np.zeros(len(xyz))
            a_tensor = torch.from_numpy(a_scores).float().unsqueeze(0).to(device)
            idx = pas_sampling(xyz_tensor, K, a_tensor, tau=tau).squeeze(0).cpu().numpy()
        elif strategy == 'fps':
            idx = fps_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()
        elif strategy == 'curv':
            idx = curv_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()
        elif strategy == 'rs':
            idx = rs_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()
        else:
            idx = fps_sampling(xyz_tensor, K).squeeze(0).cpu().numpy()

        idx = np.clip(idx, 0, len(xyz) - 1)

        # Map scores back to organized grid for pixel-level evaluation
        score_map_full = np.zeros(valid_mask.shape)
        score_map_full[valid_mask] = all_scores
        score_organized = score_map_full.reshape(org_pc.shape[0], org_pc.shape[1])

        # Collect metrics
        if gt_mask is not None:
            gt_flat = gt_mask.reshape(-1).astype(int)
            score_flat = score_organized.reshape(-1)
            all_point_scores.extend(score_flat.tolist())
            all_point_labels.extend(gt_flat.tolist())
            all_score_maps.append(score_organized)
            all_gt_maps.append(gt_mask.astype(float))
            all_image_labels.append(1)

            # Retention: how many GT defect points are in our sampled set?
            if label == 1 and gt_mask.sum() > 0:
                gt_at_xyz = gt_mask.reshape(-1)[valid_mask][:len(xyz)]
                defect_count = gt_at_xyz.sum()
                if defect_count > 0:
                    all_retention.append(float(gt_at_xyz[idx].sum() / defect_count))
        else:
            all_image_labels.append(0)

        all_image_preds.append(image_score)

    # Compute metrics
    results = {}
    ps = np.array(all_point_scores)
    pl = np.array(all_point_labels)

    results['point_auc'] = float(roc_auc_score(pl, ps)) if len(np.unique(pl)) > 1 else 0.5
    results['image_auc'] = float(roc_auc_score(all_image_labels, all_image_preds)) if len(np.unique(all_image_labels)) > 1 else 0.5

    pro_vals = [compute_pro(s, g) for s, g in zip(all_score_maps, all_gt_maps) if g.sum() > 0]
    results['pro'] = float(np.mean(pro_vals)) if pro_vals else 0.0

    iou_vals = [find_best_threshold(s, g.astype(bool)) for s, g in zip(all_score_maps, all_gt_maps) if g.sum() > 0]
    results['iou'] = float(np.mean(iou_vals)) if iou_vals else 0.0

    results['retention'] = float(np.mean(all_retention)) if all_retention else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description='BTF-style FPFH Plug-in Experiment (Complete Pipeline)')
    parser.add_argument('--strategies', default='fps,pas,curv,rs')
    parser.add_argument('--classes', default='all')
    parser.add_argument('--npoint', type=int, default=512)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--f_coreset', type=float, default=1.0)
    parser.add_argument('--dataset_path', default='datasets/mvtec3d')
    parser.add_argument('--output_json', default='results/btf_fpfh_plugin.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--use_dino', action='store_true', default=True)
    args = parser.parse_args()

    strategies = args.strategies.split(',')
    if args.classes == 'all':
        classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    else:
        classes = args.classes.split(',')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seeds(42)

    print("=" * 80)
    print("BTF-style FPFH Plug-in Experiment (Complete Pipeline)")
    print(f"Strategies: {strategies}")
    print(f"Classes: {len(classes)}, npoint={args.npoint}, tau={args.tau}")
    print("Pipeline: Train=FPFH(all)→sample→bank, Test=FPFH(all)→score(all)")
    print("=" * 80)

    # Load DINO for PAS
    dino_model = None
    if 'pas' in strategies and args.use_dino:
        print("\nLoading DINO backbone...")
        from models.models import Model
        dino_model = Model(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                           xyz_backbone_name='Point_MAE', group_size=128, num_group=1024)
        dino_model.to(device).eval()
        print("DINO loaded.")

    all_results = {}

    for cls in classes:
        print(f"\n{'='*60}")
        print(f"Class: {cls}")
        print(f"{'='*60}")

        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")
            t0 = time.time()
            results = run_btf_experiment(cls, strategy, args, dino_model)
            elapsed = time.time() - t0
            if results is not None:
                key = f"{cls}/{strategy}"
                all_results[key] = results
                results['time_sec'] = elapsed
                print(f"    PtAUC={results['point_auc']:.4f}  PRO={results['pro']:.4f}  "
                      f"IoU={results['iou']:.4f}  ImgAUC={results['image_auc']:.4f}  "
                      f"Ret={results['retention']:.4f}  ({elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = {}
    for strategy in strategies:
        vals = {m: [] for m in ['point_auc', 'pro', 'iou', 'image_auc', 'retention']}
        for cls in classes:
            key = f"{cls}/{strategy}"
            if key in all_results:
                for m in vals:
                    vals[m].append(all_results[key][m])
        if vals['point_auc']:
            summary[strategy] = {f'mean_{m}': float(np.mean(v)) for m, v in vals.items()}
            summary[strategy]['n_classes'] = len(vals['point_auc'])

    print(f"\n{'Strategy':<10} {'PtAUC':>8} {'PRO':>8} {'IoU':>8} {'ImgAUC':>8} {'Retention':>10}")
    print("-" * 60)
    for strategy in strategies:
        if strategy in summary:
            s = summary[strategy]
            print(f"{strategy:<10} {s['mean_point_auc']:>8.4f} {s['mean_pro']:>8.4f} "
                  f"{s['mean_iou']:>8.4f} {s['mean_image_auc']:>8.4f} {s['mean_retention']:>10.4f}")

    # Save
    output = {
        '_metadata': {
            'description': 'BTF-style FPFH memory bank plug-in (complete pipeline)',
            'pipeline': 'Train: FPFH(all)→sample→bank, Test: FPFH(all)→score(all)',
            'strategies': strategies, 'classes': classes,
            'npoint': args.npoint, 'tau': args.tau, 'f_coreset': args.f_coreset,
        },
        'per_class': all_results,
        'summary': summary,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
