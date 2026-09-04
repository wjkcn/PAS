"""
Budget-aware 4-strategy center sampling comparison.
Core hypothesis: fewer centers → selection strategy matters more.
All strategies bypass interpolation — center features scored directly against bank.

Strategies: FPS (geometry), RS (random), PAS (hybrid), Anomaly (pure anomaly-guided)
Budgets: 32, 64, 128, 256, 512, 1024
"""
import os
import sys
sys.path.insert(0, '.')
import glob
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from models.models import Model, Group
from utils.pas_core import compute_feature_anomaly_score, k_center_greedy_coreset

# ── Config ────────────────────────────────────────────────────────────────
BASE_FEAT = 'datasets/patch_lib/offline_features/mvtec3d'
BASE_DATA = 'datasets/mvtec3d'
CLASSES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
           'foam', 'peach', 'potato', 'rope', 'tire']
BUDGETS = [32, 64, 128, 256, 512, 1024]
STRATEGIES = ['FPS', 'RS', 'PAS', 'Anomaly']
BANK_IMAGES = 30
BANK_STRIDE = 5
CORESET_FRAC = 0.02
H, W = 224, 224
SW = 5.0
MAX_TEST = None  # Use all test samples

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


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════

def entropy_weight(d, T=1.0):
    p = F.softmax(d / T, dim=0)
    return 1.0 / (-(p * torch.log(p + 1e-8)).sum() + 1e-5)


def score_centers(sfr_centers, sfp_centers, br, bp):
    """Score center features directly (no interpolation)."""
    dr = torch.cdist(sfr_centers.unsqueeze(0), br.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    dp = torch.cdist(sfp_centers.unsqueeze(0), bp.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    wr = entropy_weight(dr)
    wp = entropy_weight(dp)
    tot = wr + wp
    fused = (wr / tot) * (dr / (dr.mean() + 1e-5)) + (wp / tot) * (dp / (dp.mean() + 1e-5))
    k = min(10, len(fused))
    return fused.topk(k).values.mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Memory bank
# ═══════════════════════════════════════════════════════════════════════════

def build_bank_and_gc(train_paths):
    bank_r, bank_p = [], []
    all_r, all_p = [], []
    num_bank = min(BANK_IMAGES, len(train_paths))

    with torch.no_grad():
        for p in tqdm(train_paths[:num_bank], desc="  Building bank", leave=False):
            d = torch.load(p, map_location='cpu')
            fr, fp = d['rgb_features'], d['pts_features']
            if 'nonzero_indices' in d:
                nz = d['nonzero_indices'].to(device).view(-1)
            else:
                nz = torch.arange(0, H * W, max(1, (H * W) // 4096), device=device)

            B, Nr, Cr = fr.shape
            sr = int(np.sqrt(Nr))
            far = F.interpolate(fr.permute(0, 2, 1).view(B, Cr, sr, sr),
                                size=(H, W), mode='bilinear', align_corners=False
                                ).view(B, Cr, -1).permute(0, 2, 1)
            fr_f = torch.cat([far.to(device), (spatial_pe * SW).to(device)], dim=-1)

            B, Np, Cp = fp.shape
            sp = int(np.sqrt(Np))
            fap = F.interpolate(fp.permute(0, 2, 1).view(B, Cp, sp, sp),
                                size=(H, W), mode='bilinear', align_corners=False
                                ).view(B, Cp, -1).permute(0, 2, 1)
            fp_f = torch.cat([fap.to(device), (spatial_pe * SW).to(device)], dim=-1)

            bank_r.append(fr_f[:, nz, :].squeeze(0).cpu())
            bank_p.append(fp_f[:, nz, :].squeeze(0).cpu())
            all_r.append(far.squeeze(0).mean(dim=0).cpu())
            all_p.append(fap.squeeze(0).mean(dim=0).cpu())

    gc_r = torch.stack(all_r).mean(dim=0)
    gc_p = torch.stack(all_p).mean(dim=0)
    sz = torch.zeros(2, device=device)
    gc = torch.cat([gc_r.to(device), sz, gc_p.to(device), sz]).unsqueeze(0).unsqueeze(0)
    gc_rgb = gc[:, :, :768 + 2].contiguous()

    br = torch.cat(bank_r, dim=0)
    bp = torch.cat(bank_p, dim=0)
    if len(br) > 100000:
        br = br[::BANK_STRIDE]
        bp = bp[::BANK_STRIDE]

    br = k_center_greedy_coreset(br, fraction=CORESET_FRAC, device=device).to(device)
    bp = k_center_greedy_coreset(bp, fraction=CORESET_FRAC, device=device).to(device)
    return br, bp, gc_rgb


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def map_center_idx_to_2d(center_idx, nonzero_mask, H=224, W=224):
    nz_lin = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
    lin_idx = nz_lin[center_idx]
    y = lin_idx // W
    x = lin_idx % W
    yn = (y.float() / (H - 1)) * 2 - 1
    xn = (x.float() / (W - 1)) * 2 - 1
    return torch.stack([xn, yn], dim=-1)


def load_raw_sample(rgb_path, tiff_path):
    from PIL import Image
    from torchvision import transforms
    from utils.mvtec3d_util import read_tiff_organized_pc, resize_organized_pc

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    img = Image.open(rgb_path).convert('RGB')
    img_tensor = rgb_transform(img).unsqueeze(0).to(device)

    organized_pc = read_tiff_organized_pc(tiff_path)
    resized_pc = resize_organized_pc(organized_pc, target_height=224, target_width=224)
    resized_pc = resized_pc.to(device)
    pc_flat = resized_pc.contiguous().view(3, -1).unsqueeze(0)
    nonzero_mask = (pc_flat.abs().sum(dim=1) > 1e-8).squeeze(0)
    xyz_all = pc_flat[:, :, nonzero_mask].contiguous()

    return img_tensor, xyz_all, nonzero_mask


def extract_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask):
    """Extract RGB and PTS features at selected center positions."""
    center_idx_1d = center_idx.squeeze(0)
    sfr = rgb_valid_pe.squeeze(0)[center_idx_1d]
    pts_center_feat = pts_feat.squeeze(0).permute(1, 0)
    pe = map_center_idx_to_2d(center_idx_1d, nonzero_mask)
    sfp = torch.cat([pts_center_feat, pe * SW], dim=-1)
    return sfr, sfp


# ═══════════════════════════════════════════════════════════════════════════
# Per-class benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_class(class_name, model):
    print(f"\n{'=' * 60}")
    print(f"[{class_name}] 4-Strategy Budget Sweep")
    print(f"{'=' * 60}")

    feat_dir = os.path.join(BASE_FEAT, class_name)
    train_paths = sorted(glob.glob(os.path.join(feat_dir, 'train', '*.pt')))
    if not train_paths:
        print(f"  No training features, skip")
        return None

    br, bp, gc_rgb = build_bank_and_gc(train_paths)
    print(f"  Bank: rgb={br.shape}, pts={bp.shape}")

    data_dir = os.path.join(BASE_DATA, class_name)
    test_rgb = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'rgb', '*.png')))
    test_tiff = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'xyz', '*.tiff')))

    test_pairs = []
    for r in test_rgb:
        base = os.path.splitext(r)[0].replace('/rgb/', '/xyz/')
        t = base + '.tiff'
        if t in test_tiff:
            label = 0 if '/good/' in r else 1
            test_pairs.append((r, t, label))

    if MAX_TEST:
        test_pairs = test_pairs[:MAX_TEST]
    if not test_pairs:
        return None

    print(f"  Test samples: {len(test_pairs)}")

    # ── Sample-outer loop: DINO once per sample ──
    # Structure: dict[budget][strategy] = list of scores
    all_labels = []
    all_scores = {b: {s: [] for s in STRATEGIES} for b in BUDGETS}

    for rgb_path, tiff_path, label in tqdm(test_pairs, desc=f"  {class_name}"):
        try:
            img_tensor, xyz_all, nonzero_mask = load_raw_sample(rgb_path, tiff_path)
        except Exception:
            continue

        n_pts = int(nonzero_mask.sum().item())
        min_budget = min(BUDGETS)
        if n_pts < min_budget:
            continue

        # DINO forward — once per sample
        with torch.no_grad():
            rgb_feat_28 = model.forward_rgb_features(img_tensor)
            rgb_224 = F.interpolate(rgb_feat_28, size=(H, W), mode='bilinear',
                                     align_corners=False).view(1, 768, -1).permute(0, 2, 1)
            pe_valid = (spatial_pe * SW)[:, nonzero_mask, :]
            rgb_valid_pe = torch.cat([rgb_224[:, nonzero_mask, :], pe_valid], dim=-1)
            anomaly_scores, _ = compute_feature_anomaly_score(
                rgb_valid_pe, tau=0.5, global_center=gc_rgb)

        all_labels.append(label)

        for budget in BUDGETS:
            # Swap group_divider for this budget
            model.xyz_backbone.group_divider = Group(num_group=budget, group_size=128).to(device)

            # ── FPS ──
            model.xyz_backbone.group_divider._sampling_mode = 'fps'
            with torch.no_grad():
                pts_feat, _, _, center_idx = model.xyz_backbone(xyz_all)
            sfr, sfp = extract_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask)
            all_scores[budget]['FPS'].append(score_centers(sfr, sfp, br, bp))

            # ── RS ──
            model.xyz_backbone.group_divider._sampling_mode = 'rs'
            with torch.no_grad():
                pts_feat, _, _, center_idx = model.xyz_backbone(xyz_all)
            sfr, sfp = extract_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask)
            all_scores[budget]['RS'].append(score_centers(sfr, sfp, br, bp))

            # ── PAS ──
            model.xyz_backbone.group_divider._sampling_mode = 'pas'
            with torch.no_grad():
                pts_feat, _, _, center_idx = model.xyz_backbone(xyz_all, anomaly_scores)
            sfr, sfp = extract_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask)
            all_scores[budget]['PAS'].append(score_centers(sfr, sfp, br, bp))

            # ── Anomaly ──
            model.xyz_backbone.group_divider._sampling_mode = 'anomaly'
            with torch.no_grad():
                pts_feat, _, _, center_idx = model.xyz_backbone(xyz_all, anomaly_scores)
            sfr, sfp = extract_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask)
            all_scores[budget]['Anomaly'].append(score_centers(sfr, sfp, br, bp))

        torch.cuda.empty_cache()

    # Restore original divider
    model.xyz_backbone.group_divider = Group(num_group=1024, group_size=128).to(device)
    if hasattr(model.xyz_backbone.group_divider, '_sampling_mode'):
        del model.xyz_backbone.group_divider._sampling_mode

    # ── Compute AUCs ──
    labels_arr = np.array(all_labels)
    if len(np.unique(labels_arr)) < 2:
        return None

    results = {}
    for budget in BUDGETS:
        results[budget] = {}
        for strat in STRATEGIES:
            results[budget][strat] = roc_auc_score(labels_arr, all_scores[budget][strat])

    # Per-class summary
    print(f"  {'Budget':<8}", end='')
    for s in STRATEGIES:
        print(f" {s:>8}", end='')
    print(f"  {'PAS-FPS':>10}  {'Anom-FPS':>10}")
    print(f"  {'-' * 8}", end='')
    print(f" {'-' * (10 * len(STRATEGIES) + 22)}")

    for budget in BUDGETS:
        print(f"  K={budget:<5}", end='')
        for s in STRATEGIES:
            print(f" {results[budget][s]:>8.4f}", end='')
        delta_pas = results[budget]['PAS'] - results[budget]['FPS']
        delta_anom = results[budget]['Anomaly'] - results[budget]['FPS']
        print(f"  {delta_pas:>+10.4f}  {delta_anom:>+10.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("=" * 80)
    print("4-Strategy Budget Sweep: FPS vs RS vs PAS vs Anomaly")
    print("Center-direct scoring (no interpolation)")
    print(f"Budgets: {BUDGETS}, Classes: {len(CLASSES)}")
    print("=" * 80)

    print("\nLoading M3DM model...")
    args = Args()
    model = Model(
        device=device,
        rgb_backbone_name=args.rgb_backbone_name,
        xyz_backbone_name=args.xyz_backbone_name,
        group_size=args.group_size,
        num_group=args.num_group,
    )
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    all_class_results = {}
    for cls in CLASSES:
        r = benchmark_class(cls, model)
        if r:
            all_class_results[cls] = r
        torch.cuda.empty_cache()

    if not all_class_results:
        print("\nNo results.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Grand Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("GRAND SUMMARY: Mean AUC across all classes")
    print("=" * 100)

    # Mean AUC table
    header = f"{'Budget':<8}"
    for s in STRATEGIES:
        header += f" {s:>8}"
    header += f"  {'PAS-FPS':>10}  {'RS-FPS':>10}  {'Anom-FPS':>10}"
    print(header)
    print("-" * 100)

    for budget in BUDGETS:
        means = {}
        for s in STRATEGIES:
            vals = [all_class_results[c][budget][s] for c in all_class_results if budget in all_class_results[c]]
            means[s] = np.mean(vals) if vals else float('nan')
        line = f"K={budget:<5}"
        for s in STRATEGIES:
            line += f" {means[s]:>8.4f}"
        d_pas = means['PAS'] - means['FPS']
        d_rs = means['RS'] - means['FPS']
        d_anom = means['Anomaly'] - means['FPS']
        line += f"  {d_pas:>+10.4f}  {d_rs:>+10.4f}  {d_anom:>+10.4f}"
        print(line)

    # Win-count table
    print("\n" + "=" * 100)
    print("Win Counts (vs FPS baseline)")
    print("=" * 100)
    print(f"{'Budget':<8} {'PAS Wins':>10} {'RS Wins':>10} {'Anomaly Wins':>10}  {'Best Strategy':>16}")
    print("-" * 65)
    for budget in BUDGETS:
        pas_wins = sum(1 for c in all_class_results
                       if budget in all_class_results[c]
                       and all_class_results[c][budget]['PAS'] > all_class_results[c][budget]['FPS'])
        rs_wins = sum(1 for c in all_class_results
                      if budget in all_class_results[c]
                      and all_class_results[c][budget]['RS'] > all_class_results[c][budget]['FPS'])
        anom_wins = sum(1 for c in all_class_results
                        if budget in all_class_results[c]
                        and all_class_results[c][budget]['Anomaly'] > all_class_results[c][budget]['FPS'])
        # Find best strategy by mean
        means_at_budget = {}
        for s in STRATEGIES:
            vals = [all_class_results[c][budget][s] for c in all_class_results if budget in all_class_results[c]]
            means_at_budget[s] = np.mean(vals) if vals else 0
        best = max(means_at_budget, key=means_at_budget.get)
        print(f"K={budget:<5} {pas_wins:>10} {rs_wins:>10} {anom_wins:>10}  {best:>16}")

    # Per-class detail: best budget for PAS
    print("\n" + "=" * 100)
    print("Per-Class PAS-FPS Delta at Each Budget")
    print("=" * 100)
    header = f"{'Class':<16}"
    for b in BUDGETS:
        header += f"  K={b:>8}"
    print(header)
    print("-" * 100)
    for cls in sorted(all_class_results.keys()):
        line = f"{cls:<16}"
        for b in BUDGETS:
            if b in all_class_results[cls]:
                d = all_class_results[cls][b]['PAS'] - all_class_results[cls][b]['FPS']
                line += f"  {d:>+8.4f}"
            else:
                line += f"  {'N/A':>8}"
        print(line)

    # ═══════════════════════════════════════════════════════════════════════
    # Save CSV
    # ═══════════════════════════════════════════════════════════════════════
    csv_path = f'results/budget_sweep_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Budget', 'Strategy', 'AUC'])
        for cls in sorted(all_class_results.keys()):
            for budget in BUDGETS:
                if budget in all_class_results[cls]:
                    for strat in STRATEGIES:
                        writer.writerow([cls, budget, strat, all_class_results[cls][budget][strat]])
    print(f"\nResults saved to {csv_path}")

    # ── Auto-log to RESULTS_LOG.md ──
    try:
        from utils.result_logger import log_experiment

        # Main table: mean AUC at each budget
        cols = ["Budget"] + STRATEGIES + ["PAS-FPS", "Anom-FPS"]
        rows = []
        for budget in BUDGETS:
            row = [f"K={budget}"]
            means = {}
            for s in STRATEGIES:
                vals = [all_class_results[c][budget][s] for c in all_class_results if budget in all_class_results[c]]
                means[s] = np.mean(vals) if vals else float('nan')
            for s in STRATEGIES:
                row.append(means[s])
            row.append(means['PAS'] - means['FPS'])
            row.append(means['Anomaly'] - means['FPS'])
            rows.append(row)

        mean_of_means = {}
        for s in STRATEGIES:
            vals = [np.mean([all_class_results[c][b][s] for c in all_class_results if b in all_class_results[c]])
                    for b in BUDGETS]
            mean_of_means[s] = np.mean(vals)
        mean_row = ["MEAN"] + [mean_of_means[s] for s in STRATEGIES] + [
            mean_of_means['PAS'] - mean_of_means['FPS'],
            mean_of_means['Anomaly'] - mean_of_means['FPS'],
        ]

        # Per-class delta table
        pcols = ["Class"] + [f"K={b}" for b in BUDGETS]
        prows = []
        for cls in sorted(all_class_results.keys()):
            row = [cls]
            for b in BUDGETS:
                if b in all_class_results[cls]:
                    row.append(all_class_results[cls][b]['PAS'] - all_class_results[cls][b]['FPS'])
                else:
                    row.append(float('nan'))
            prows.append(row)

        # Win counts
        win_counts = {}
        for budget in BUDGETS:
            pas_w = sum(1 for c in all_class_results
                        if budget in all_class_results[c]
                        and all_class_results[c][budget]['PAS'] > all_class_results[c][budget]['FPS'])
            anom_w = sum(1 for c in all_class_results
                         if budget in all_class_results[c]
                         and all_class_results[c][budget]['Anomaly'] > all_class_results[c][budget]['FPS'])
            win_counts[budget] = (pas_w, anom_w)

        conclusion_papas = []
        for budget in BUDGETS:
            rw, aw = win_counts[budget]
            conclusion_papas.append(f"K={budget}: PAS wins {rw}/10, Anomaly wins {aw}/10")
        conclusion = "PAS vs FPS win counts across budgets. " + "; ".join(conclusion_papas)

        log_experiment(
            name="Budget Sweep: FPS vs RS vs PAS vs Anomaly",
            script="benchmark_pas_budget.py",
            description="中心直评 (无插值), 6预算×4策略对比, 核心假设: 低预算下PAS>FPS",
            columns=cols,
            rows=rows,
            mean_row=mean_row,
            conclusion=conclusion,
            extra_sections={
                "Per-Class PAS-FPS Delta": {
                    "columns": pcols,
                    "rows": prows,
                }
            },
        )
    except Exception as e:
        print(f"[results] 自动日志失败: {e}")

    print("=" * 100)


if __name__ == "__main__":
    main()
