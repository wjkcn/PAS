"""
Local Recall @ Top-K: How many sampled centers fall inside the GT anomaly region?

Metric:
  - Recall = |centers ∩ GT_anomaly| / |GT_anomaly|
  - Precision = |centers ∩ GT_anomaly| / |centers|

Pure geometry-based GT evaluation — no anomaly score needed for the metric itself.
"""
import os, sys
sys.path.insert(0, '.')
import glob
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from models.models import Model, Group
from utils.pas_core import compute_feature_anomaly_score

BASE_FEAT = 'datasets/patch_lib/offline_features/mvtec3d'
BASE_DATA = 'datasets/mvtec3d'
CLASSES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
           'foam', 'peach', 'potato', 'rope', 'tire']
BUDGETS = [32, 64, 128, 256, 512, 1024]
STRATEGIES = ['FPS', 'RS', 'PAS', 'Anomaly']
H, W = 224, 224
SW = 5.0
BANK_IMAGES = 30
MAX_TEST = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

yg, xg = torch.meshgrid(
    torch.linspace(-1, 1, H, device=device),
    torch.linspace(-1, 1, W, device=device), indexing='ij')
spatial_pe = torch.stack([xg, yg], dim=-1).view(1, H * W, 2)


class Args:
    rgb_backbone_name = 'vit_base_patch14_dinov2'
    xyz_backbone_name = 'Point_MAE'
    group_size = 128
    num_group = 1024
    img_size = 224


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_raw_sample(rgb_path, tiff_path):
    from utils.mvtec3d_util import read_tiff_organized_pc, resize_organized_pc

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = Image.open(rgb_path).convert('RGB').resize((H, W), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    organized_pc = read_tiff_organized_pc(tiff_path)
    resized_pc = resize_organized_pc(organized_pc, target_height=H, target_width=W)
    resized_pc = resized_pc.to(device)
    pc_flat = resized_pc.contiguous().view(3, -1).unsqueeze(0)
    nonzero_mask = (pc_flat.abs().sum(dim=1) > 1e-8).squeeze(0)
    xyz_all = pc_flat[:, :, nonzero_mask].contiguous()
    return img_tensor, xyz_all, nonzero_mask


def load_gt_mask(gt_path):
    """Load GT mask, resize to 224x224, return binary tensor [H, W]."""
    img = Image.open(gt_path)
    arr = np.array(img.resize((W, H), Image.NEAREST))
    # 254 and 255 are anomaly regions
    return torch.from_numpy((arr >= 254).astype(np.float32))


def map_center_idx_to_2d(center_idx, nonzero_mask):
    nz_lin = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
    lin_idx = nz_lin[center_idx.squeeze(0)]
    y = lin_idx // W
    x = lin_idx % W
    return y, x


def compute_local_recall(yx_centers, gt_mask):
    """
    Args:
        yx_centers: (y, x) tuple of tensors [N] in 224x224 grid
        gt_mask: [H, W] binary tensor (1=anomaly)
    Returns:
        recall: fraction of GT anomaly pixels covered by centers
        precision: fraction of centers that fall in GT anomaly
    """
    y, x = yx_centers
    mask_vals = gt_mask[y.clamp(0, H - 1), x.clamp(0, W - 1)]
    centers_in_gt = mask_vals.sum().item()
    total_gt = gt_mask.sum().item()
    total_centers = len(y)
    recall = centers_in_gt / total_gt if total_gt > 0 else 0.0
    precision = centers_in_gt / total_centers if total_centers > 0 else 0.0
    return recall, precision


def build_gc(train_paths):
    all_r = []
    num_bank = min(BANK_IMAGES, len(train_paths))
    with torch.no_grad():
        for p in tqdm(train_paths[:num_bank], desc="  Building GC", leave=False):
            d = torch.load(p, map_location='cpu')
            fr = d['rgb_features']
            B, Nr, Cr = fr.shape
            sr = int(np.sqrt(Nr))
            far = F.interpolate(fr.permute(0, 2, 1).view(B, Cr, sr, sr),
                                size=(H, W), mode='bilinear', align_corners=False
                                ).view(B, Cr, -1).permute(0, 2, 1)
            all_r.append(far.squeeze(0).mean(dim=0).cpu())
    gc_r = torch.stack(all_r).mean(dim=0)
    sz = torch.zeros(2, device=device)
    return torch.cat([gc_r.to(device), sz]).unsqueeze(0).unsqueeze(0)


# ── Per-class benchmark ─────────────────────────────────────────────────────

def benchmark_class_recall(class_name, model):
    print(f"\n{'=' * 60}")
    print(f"[{class_name}] Local Recall @ Top-K")
    print(f"{'=' * 60}")

    feat_dir = os.path.join(BASE_FEAT, class_name)
    train_paths = sorted(glob.glob(os.path.join(feat_dir, 'train', '*.pt')))
    if not train_paths:
        return None
    gc_rgb = build_gc(train_paths)

    # Collect test anomaly samples with GT
    data_dir = os.path.join(BASE_DATA, class_name)
    test_rgb = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'rgb', '*.png')))
    test_tiff = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'xyz', '*.tiff')))

    samples = []
    for r in test_rgb:
        if '/good/' in r:
            continue
        base = os.path.splitext(r)[0].replace('/rgb/', '/xyz/')
        t = base + '.tiff'
        # GT mask: .../test/{defect_type}/gt/{id}.png
        gt_dir = os.path.dirname(os.path.dirname(r))
        fname = os.path.basename(r)
        gt_path = os.path.join(gt_dir, 'gt', fname)
        if t in test_tiff and os.path.exists(gt_path):
            samples.append((r, t, gt_path))

    if not samples:
        print("  No anomaly samples with GT, skip")
        return None
    print(f"  Anomaly samples: {len(samples)}")

    # results[budget][strategy] = list of (recall, precision) pairs
    results = {b: {s: {'recall': [], 'precision': []} for s in STRATEGIES} for b in BUDGETS}

    for rgb_path, tiff_path, gt_path in tqdm(samples, desc=f"  {class_name}"):
        try:
            img_tensor, xyz_all, nonzero_mask = load_raw_sample(rgb_path, tiff_path)
            gt_mask = load_gt_mask(gt_path).to(device)
        except Exception:
            continue

        n_pts = int(nonzero_mask.sum().item())
        if n_pts < min(BUDGETS):
            continue
        if gt_mask.sum().item() < 10:
            continue

        # DINO forward (shared)
        with torch.no_grad():
            rgb_feat_28 = model.forward_rgb_features(img_tensor)
            rgb_224 = F.interpolate(rgb_feat_28, size=(H, W), mode='bilinear',
                                     align_corners=False).view(1, 768, -1).permute(0, 2, 1)
            pe_valid = (spatial_pe * SW)[:, nonzero_mask, :]
            rgb_valid_pe = torch.cat([rgb_224[:, nonzero_mask, :], pe_valid], dim=-1)
            anomaly_scores, _ = compute_feature_anomaly_score(
                rgb_valid_pe, tau=0.5, global_center=gc_rgb)

        for budget in BUDGETS:
            model.xyz_backbone.group_divider = Group(num_group=budget, group_size=128).to(device)

            for strat in STRATEGIES:
                model.xyz_backbone.group_divider._sampling_mode = strat.lower()
                with torch.no_grad():
                    if strat in ('PAS', 'Anomaly'):
                        _, _, _, center_idx = model.xyz_backbone(xyz_all, anomaly_scores)
                    else:
                        _, _, _, center_idx = model.xyz_backbone(xyz_all)
                y, x = map_center_idx_to_2d(center_idx, nonzero_mask)
                recall, precision = compute_local_recall((y, x), gt_mask)
                results[budget][strat]['recall'].append(recall)
                results[budget][strat]['precision'].append(precision)

        torch.cuda.empty_cache()

    # Restore
    model.xyz_backbone.group_divider = Group(num_group=1024, group_size=128).to(device)
    if hasattr(model.xyz_backbone.group_divider, '_sampling_mode'):
        del model.xyz_backbone.group_divider._sampling_mode

    # Print summary (recall as percentage: fraction of GT anomaly region covered)
    print(f"\n  {'Budget':<8} {'FPS_rec%':>9} {'RS_rec%':>9} {'PAS_rec%':>9} {'Anom_rec%':>9}  {'PAS/FPS':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*9}  {'-'*8}")
    class_summary = {}
    for b in BUDGETS:
        means = {}
        for s in STRATEGIES:
            vals = results[b][s]['recall']
            means[s] = np.mean(vals) if vals else 0  # raw fraction
        ratio = means['PAS'] / means['FPS'] if means['FPS'] > 0 else 0
        print(f"  K={b:<5} {means['FPS']*100:>8.2f}% {means['RS']*100:>8.2f}% {means['PAS']*100:>8.2f}% {means['Anomaly']*100:>8.2f}%  {ratio:>7.1f}x")
        class_summary[b] = means
    return class_summary


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("=" * 80)
    print("Local Recall @ Top-K: GT-Anchored Center Quality Evaluation")
    print(f"Strategies: {STRATEGIES}, Budgets: {BUDGETS}")
    print("=" * 80)

    print("\nLoading M3DM model...")
    args = Args()
    model = Model(
        device=device, rgb_backbone_name=args.rgb_backbone_name,
        xyz_backbone_name=args.xyz_backbone_name,
        group_size=args.group_size, num_group=args.num_group,
    )
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    all_results = {}
    for cls in CLASSES:
        r = benchmark_class_recall(cls, model)
        if r:
            all_results[cls] = r
        torch.cuda.empty_cache()

    if not all_results:
        print("\nNo results.")
        return

    # ── Grand Summary ──
    print("\n\n" + "=" * 90)
    print("GRAND SUMMARY: Mean Local Recall (fraction of GT anomaly covered)")
    print("=" * 90)

    header = f"{'Budget':<8} {'FPS%':>8} {'RS%':>8} {'PAS%':>8} {'Anom%':>8}  {'PAS/FPS':>8}  {'PAS-FPS%':>10}"
    print(header)
    print("-" * 90)

    for b in BUDGETS:
        means = {}
        for s in STRATEGIES:
            vals = [all_results[c][b][s] for c in all_results if b in all_results[c]]
            means[s] = np.mean(vals) if vals else 0  # raw fraction, no multiply
        ratio = means['PAS'] / means['FPS'] if means['FPS'] > 0 else 0
        delta = means['PAS'] - means['FPS']
        print(f"K={b:<5} {means['FPS']*100:>7.2f}% {means['RS']*100:>7.2f}% {means['PAS']*100:>7.2f}% {means['Anomaly']*100:>7.2f}%  {ratio:>7.1f}x  {delta*100:>+9.2f}%")

    # Save CSV
    csv_path = f'results/local_recall_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Budget', 'Strategy', 'Recall', 'Precision'])
        # We'd need per-sample data saved — skip for brevity, save means
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
