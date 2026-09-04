"""
DEPRECATED: This script is an old auxiliary benchmark using global_normal_center
anomaly prior. It is NOT used for current paper results.

Paper experiments use per_image_mean prior (see compute_feature_anomaly_score
in utils/pas_core.py). Do NOT mix results from this script with current results.

For paper results, use: benchmark_pas_pointlevel_v2.py (point-level, component ablation).

---

PAS-M3DM Full Pipeline: Center-Direct Scoring + Reweighting + OCSVM + Anomaly Weighting.
Integrates all 6 PAS components into a single evaluation script.

Compares 4 variants:
  Baseline:  FPS selection + no anomaly weighting
  PAS-S:     PAS-FPS selection + no anomaly weighting
  PAS-W:     FPS selection + anomaly weighting
  PAS-Full:  PAS-FPS selection + anomaly weighting

All variants use: dual_mask_cleaning, center-direct scoring, reweighting, OCSVM fusion.
"""
import os
import sys
sys.path.insert(0, '.')
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDOneClassSVM

from models.models import Model
from utils.pas_core import compute_feature_anomaly_score, k_center_greedy_coreset

# ── Config ────────────────────────────────────────────────────────────────
BASE_FEAT = 'datasets/patch_lib/offline_features/mvtec3d'
BASE_DATA = 'datasets/mvtec3d'
CLASSES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
           'foam', 'peach', 'potato', 'rope', 'tire']
BANK_IMAGES = 30
BANK_STRIDE = 5
CORESET_FRAC = 0.02
H, W = 224, 224
SW = 5.0          # spatial PE weight
MAX_TEST = None  # Use all test samples
N_REWEIGHT = 3
OCSVM_NU = 0.5
OCSVM_MAXITER = 1000

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
# Component 1: dual_mask_cleaning
# ═══════════════════════════════════════════════════════════════════════════

def dual_mask_cleaning(organized_pc_np, rgb_img_tensor):
    """
    Physical + color adaptive mask for foreground/background separation.
    Ported from tools/cache_features_offline.py.
    """
    unorganized_pc = organized_pc_np.reshape(-1, 3)

    # Physical: non-zero and z > 0
    mask_not_zero = np.any(unorganized_pc != 0, axis=1)
    mask_z_positive = unorganized_pc[:, 2] > 0
    physical_mask = mask_not_zero & mask_z_positive

    # Color: adaptive threshold
    if rgb_img_tensor.dim() == 4:
        rgb_flat = rgb_img_tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    else:
        rgb_flat = rgb_img_tensor.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    if rgb_flat.max() > 50:
        color_mask = np.any(rgb_flat > 10, axis=1)
    else:
        color_mask = np.any(rgb_flat > -2.0, axis=1)

    final_mask = physical_mask & color_mask
    valid_indices = np.nonzero(final_mask)[0]
    return unorganized_pc[valid_indices, :]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_sample_cleaned(rgb_path, tiff_path):
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
    organized_pc_np = resized_pc.squeeze().permute(1, 2, 0).cpu().numpy()

    # ── dual_mask_cleaning ──
    clean_xyz = dual_mask_cleaning(organized_pc_np, img_tensor)

    # Build nonzero_mask from same criteria
    organized_flat = organized_pc_np.reshape(-1, 3)
    phys = np.any(organized_flat != 0, axis=1) & (organized_flat[:, 2] > 0)
    rgb_flat = img_tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    if rgb_flat.max() > 50:
        col = np.any(rgb_flat > 10, axis=1)
    else:
        col = np.any(rgb_flat > -2.0, axis=1)
    nonzero_mask = torch.tensor(phys & col, device=device)

    xyz_tensor = torch.tensor(clean_xyz, dtype=torch.float32, device=device)
    xyz_all = xyz_tensor.T.unsqueeze(0)

    return img_tensor, xyz_all, nonzero_mask


# ═══════════════════════════════════════════════════════════════════════════
# Memory bank construction (from benchmark_pas_budget.py)
# ═══════════════════════════════════════════════════════════════════════════

def entropy_weight(d, T=1.0):
    p = F.softmax(d / T, dim=0)
    return 1.0 / (-(p * torch.log(p + 1e-8)).sum() + 1e-5)


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
# Center position mapping
# ═══════════════════════════════════════════════════════════════════════════

def map_center_idx_to_2d(center_idx, nonzero_mask):
    nz_lin = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
    lin_idx = nz_lin[center_idx]
    y = lin_idx // W
    x = lin_idx % W
    yn = (y.float() / (H - 1)) * 2 - 1
    xn = (x.float() / (W - 1)) * 2 - 1
    return torch.stack([xn, yn], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Component 3: Reweighting
# ═══════════════════════════════════════════════════════════════════════════

def reweighted_score_from_dist(dist, patch, bank, n_reweight=3):
    """
    Reweighted anomaly score from pre-computed min-distances.
    dist: [K] — min-distance from each test patch to bank
    patch: [K, C] — test features
    bank: [M, C] — memory bank
    """
    s_idx = torch.argmax(dist)
    s_star = dist[s_idx]

    m_test = patch[s_idx].unsqueeze(0)                 # [1, C]
    full_dist = torch.cdist(m_test, bank)               # [1, M]
    _, min_full_idx = torch.min(full_dist, dim=1)
    m_star = bank[min_full_idx[0]].unsqueeze(0)         # [1, C] — closest normal neighbour

    w_dist = torch.cdist(m_star, bank)                  # [1, M]
    _, nn_idx = torch.topk(w_dist, k=n_reweight, largest=False)

    if nn_idx.shape[1] > 1:
        m_star_knn = torch.linalg.norm(m_test - bank[nn_idx[0, 1:]], dim=1)
    else:
        m_star_knn = torch.zeros(1, device=patch.device)

    D = torch.sqrt(torch.tensor(patch.shape[1], device=patch.device))
    w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
    return (w * s_star).item()


# ═══════════════════════════════════════════════════════════════════════════
# Component 4+5+6: OCSVM training + Unified scoring
# ═══════════════════════════════════════════════════════════════════════════

def train_ocsvm_fuser(train_paths, br, bp, model):
    """
    Train SGDOneClassSVM on (s_xyz, s_rgb) pairs from training images.
    Uses approximate RGB alignment via uniform subsampling.
    """
    s_lib = []
    num_train = min(BANK_IMAGES, len(train_paths))

    with torch.no_grad():
        for p in tqdm(train_paths[:num_train], desc="  Training OCSVM", leave=False):
            d = torch.load(p, map_location='cpu')
            fr = d['rgb_features']          # [1, 256, 768]
            fp = d['pts_features']          # [1, 1024, 1152]

            # RGB: interpolate DINO 16x16 to 224x224, then subsample uniformly
            B, Nr, Cr = fr.shape
            sr = int(np.sqrt(Nr))
            far = F.interpolate(fr.permute(0, 2, 1).view(B, Cr, sr, sr),
                                size=(H, W), mode='bilinear', align_corners=False
                                ).view(B, Cr, -1).permute(0, 2, 1)
            nz = torch.arange(0, H * W, max(1, (H * W) // 1024))
            rgb_center = far[:, nz, :].squeeze(0).to(device)

            # PTS: use pts_features directly
            pts_center = fp.squeeze(0).to(device)

            # Spatial PE
            y = nz // W
            x = nz % W
            yn = (y.float() / (H - 1)) * 2 - 1
            xn = (x.float() / (W - 1)) * 2 - 1
            pe = torch.stack([xn, yn], dim=-1).to(device)

            rgb_with_pe = torch.cat([rgb_center, pe * SW], dim=-1)
            pts_with_pe = torch.cat([pts_center, pe * SW], dim=-1)

            # Compute per-modality reweighted scores
            dr = torch.cdist(rgb_with_pe.unsqueeze(0), br.unsqueeze(0)).squeeze(0).min(dim=1)[0]
            dp = torch.cdist(pts_with_pe.unsqueeze(0), bp.unsqueeze(0)).squeeze(0).min(dim=1)[0]

            s_rgb = reweighted_score_from_dist(dr, rgb_with_pe, br, n_reweight=N_REWEIGHT)
            s_xyz = reweighted_score_from_dist(dp, pts_with_pe, bp, n_reweight=N_REWEIGHT)

            s_lib.append([s_xyz, s_rgb])

    s_lib = np.array(s_lib)

    detect_fuser = SGDOneClassSVM(random_state=42, nu=OCSVM_NU, max_iter=OCSVM_MAXITER)
    detect_fuser.fit(s_lib)
    return detect_fuser


def score_sample_full(sfr_centers, sfp_centers, br, bp, detect_fuser, anomaly_weights=None):
    """
    Full PAS-M3DM scoring: reweighting + OCSVM fusion + optional anomaly weighting.

    Two independent scoring paths, combined when anomaly_weights is provided:
      Path A: Reweighting + OCSVM fusion (same as baseline)
      Path B: Per-center entropy-weighted scoring with (1 + aw * 2.0) boost
    Final score = average of Path A and Path B.

    Path B is kept separate because anomaly weighting distopas the distance
    distribution, causing the reweighting formula's w to go negative when
    s_star is artificially inflated.
    """
    # Per-center min-distances to bank
    dr = torch.cdist(sfr_centers.unsqueeze(0), br.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    dp = torch.cdist(sfp_centers.unsqueeze(0), bp.unsqueeze(0)).squeeze(0).min(dim=1)[0]

    # Path A: Reweighting + OCSVM (always computed, no anomaly weighting)
    s_xyz = reweighted_score_from_dist(dp, sfp_centers, bp, n_reweight=N_REWEIGHT)
    s_rgb = reweighted_score_from_dist(dr, sfr_centers, br, n_reweight=N_REWEIGHT)
    s_pair = np.array([[s_xyz, s_rgb]])
    fused_ocsvm = float(detect_fuser.score_samples(s_pair)[0])

    if anomaly_weights is not None:
        # Path B: Per-center entropy-weighted scoring with anomaly boost
        aw = anomaly_weights[:len(dr)]
        wr = entropy_weight(dr)
        wp = entropy_weight(dp)
        tot = wr + wp
        fused_pc = (wr / tot) * (dr / (dr.mean() + 1e-5)) + (wp / tot) * (dp / (dp.mean() + 1e-5))
        boost = 1.0 + aw * 2.0
        fused_pc = fused_pc * boost
        k = min(10, len(fused_pc))
        fused_weighted = fused_pc.topk(k).values.mean().item()
        # Combine both paths: average of two independent anomaly scores
        return (fused_ocsvm + fused_weighted) / 2.0

    return fused_ocsvm


# ═══════════════════════════════════════════════════════════════════════════
# Center feature extraction helper
# ═══════════════════════════════════════════════════════════════════════════

def get_center_features(rgb_valid_pe, pts_feat, center_idx, nonzero_mask):
    """
    Extract RGB and PTS features at center positions.
    """
    center_idx_1d = center_idx.squeeze(0)
    sfr = rgb_valid_pe.squeeze(0)[center_idx_1d]
    pts_center_feat = pts_feat.squeeze(0).permute(1, 0)
    pe = map_center_idx_to_2d(center_idx_1d, nonzero_mask)
    sfp = torch.cat([pts_center_feat, pe * SW], dim=-1)
    return sfr, sfp


# ═══════════════════════════════════════════════════════════════════════════
# Per-class evaluation
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_class_full(class_name, model):
    print(f"\n{'=' * 60}")
    print(f"[{class_name}] PAS-M3DM Full Pipeline")
    print(f"{'=' * 60}")

    feat_dir = os.path.join(BASE_FEAT, class_name)
    train_paths = sorted(glob.glob(os.path.join(feat_dir, 'train', '*.pt')))
    if not train_paths:
        return None

    # ── Build memory bank ──
    br, bp, gc_rgb = build_bank_and_gc(train_paths)
    print(f"  Bank: rgb={br.shape}, pts={bp.shape}")

    # ── Train OCSVM ──
    detect_fuser = train_ocsvm_fuser(train_paths, br, bp, model)
    print(f"  OCSVM trained on {min(BANK_IMAGES, len(train_paths))} samples")

    # ── Collect test pairs ──
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
        normal_pairs = [(r, t, l) for r, t, l in test_pairs if l == 0]
        anomaly_pairs = [(r, t, l) for r, t, l in test_pairs if l == 1]
        n_norm = min(MAX_TEST // 2, len(normal_pairs))
        n_anom = min(MAX_TEST - n_norm, len(anomaly_pairs))
        test_pairs = normal_pairs[:n_norm] + anomaly_pairs[:n_anom]
    if not test_pairs:
        return None

    n_normal = sum(1 for _, _, l in test_pairs if l == 0)
    n_anomaly = len(test_pairs) - n_normal
    print(f"  Test: {len(test_pairs)} ({n_normal} normal, {n_anomaly} anomaly)")

    # ── Evaluate 4 variants per sample ──
    labels = []
    scores_base = []     # Baseline: FPS + no weighting
    scores_pas_s = []    # PAS-S:   PAS-FPS + no weighting
    scores_pas_w = []    # PAS-W:   FPS + anomaly weighting
    scores_pas_full = [] # PAS-Full: PAS-FPS + anomaly weighting

    for rgb_path, tiff_path, label in tqdm(test_pairs, desc=f"  {class_name}"):
        try:
            img_tensor, xyz_all, nonzero_mask = load_raw_sample_cleaned(rgb_path, tiff_path)
        except Exception as e:
            continue

        n_pts = int(nonzero_mask.sum().item())
        if n_pts < 100:
            continue

        # ── DINO forward (shared) ──
        with torch.no_grad():
            rgb_feat_28 = model.forward_rgb_features(img_tensor)
            rgb_224 = F.interpolate(rgb_feat_28, size=(H, W), mode='bilinear',
                                     align_corners=False).view(1, 768, -1).permute(0, 2, 1)
            pe_valid = (spatial_pe * SW)[:, nonzero_mask, :]
            rgb_valid_pe = torch.cat([rgb_224[:, nonzero_mask, :], pe_valid], dim=-1)

        # ── Anomaly scores from DINO ──
        with torch.no_grad():
            anomaly_scores, _ = compute_feature_anomaly_score(
                rgb_valid_pe, tau=0.5, global_center=gc_rgb)  # [1, N]

        # ---- Baseline: FPS + no weighting ----
        with torch.no_grad():
            pts_feat_base, _, _, center_idx_base = model.xyz_backbone(xyz_all)
        sfr_base, sfp_base = get_center_features(
            rgb_valid_pe, pts_feat_base, center_idx_base, nonzero_mask)
        score_base = score_sample_full(sfr_base, sfp_base, br, bp, detect_fuser)

        # ---- PAS-S: PAS-FPS + no weighting ----
        with torch.no_grad():
            pts_feat_pas, _, _, center_idx_pas = model.xyz_backbone(xyz_all, anomaly_scores)
        sfr_pas, sfp_pas = get_center_features(
            rgb_valid_pe, pts_feat_pas, center_idx_pas, nonzero_mask)
        score_pas_s_val = score_sample_full(sfr_pas, sfp_pas, br, bp, detect_fuser)

        # ---- PAS-W: FPS + anomaly weighting ----
        aw_base = anomaly_scores.squeeze(0)[center_idx_base.squeeze(0)]
        score_pas_w_val = score_sample_full(
            sfr_base, sfp_base, br, bp, detect_fuser, anomaly_weights=aw_base)

        # ---- PAS-Full: PAS-FPS + anomaly weighting ----
        aw_pas = anomaly_scores.squeeze(0)[center_idx_pas.squeeze(0)]
        score_pas_full_val = score_sample_full(
            sfr_pas, sfp_pas, br, bp, detect_fuser, anomaly_weights=aw_pas)

        labels.append(label)
        scores_base.append(score_base)
        scores_pas_s.append(score_pas_s_val)
        scores_pas_w.append(score_pas_w_val)
        scores_pas_full.append(score_pas_full_val)

    if len(labels) < 2 or len(set(labels)) < 2:
        return None

    # ── Compute AUCs ──
    auc_base = roc_auc_score(labels, scores_base)
    auc_pas_s = roc_auc_score(labels, scores_pas_s)
    auc_pas_w = roc_auc_score(labels, scores_pas_w)
    auc_pas_full = roc_auc_score(labels, scores_pas_full)

    d_s = auc_pas_s - auc_base
    d_w = auc_pas_w - auc_base
    d_f = auc_pas_full - auc_base

    print(f"  Baseline:  {auc_base:.4f}")
    print(f"  PAS-S:     {auc_pas_s:.4f}  (Δ={d_s:+.4f})")
    print(f"  PAS-W:     {auc_pas_w:.4f}  (Δ={d_w:+.4f})")
    print(f"  PAS-Full:  {auc_pas_full:.4f}  (Δ={d_f:+.4f})")

    return {
        'class': class_name,
        'auc_base': auc_base,
        'auc_pas_s': auc_pas_s,
        'auc_pas_w': auc_pas_w,
        'auc_pas_full': auc_pas_full,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAS-M3DM Full Pipeline: Image-Level AUC Comparison")
    print("Center-Direct + Dual-Mask + Reweighting + OCSVM + Anomaly Weighting")
    print("=" * 70)

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

    results = []
    for cls in CLASSES:
        r = benchmark_class_full(cls, model)
        if r:
            results.append(r)
        torch.cuda.empty_cache()

    if not results:
        print("\nNo results.")
        return

    # ── Summary ──
    print("\n\n" + "=" * 95)
    print("Summary: PAS-M3DM Full Pipeline — Image-Level AUC")
    print("=" * 95)
    hdr = f"{'Class':<16} {'Base':>8} {'PAS-S':>8} {'PAS-W':>8} {'PAS-Full':>8}  {'D_S':>8} {'D_W':>8} {'D_F':>8}"
    print(hdr)
    print("-" * 95)
    for r in results:
        d_s = r['auc_pas_s'] - r['auc_base']
        d_w = r['auc_pas_w'] - r['auc_base']
        d_f = r['auc_pas_full'] - r['auc_base']
        print(f"{r['class']:<16} {r['auc_base']:>8.4f} {r['auc_pas_s']:>8.4f} "
              f"{r['auc_pas_w']:>8.4f} {r['auc_pas_full']:>8.4f}  "
              f"{d_s:>+8.4f} {d_w:>+8.4f} {d_f:>+8.4f}")
    print("-" * 95)
    for key in ['auc_base', 'auc_pas_s', 'auc_pas_w', 'auc_pas_full']:
        mean_val = np.mean([r[key] for r in results])
        print(f"  MEAN {key}: {mean_val:.4f}")

    # Win counts
    s_wins = sum(1 for r in results if r['auc_pas_s'] > r['auc_base'])
    w_wins = sum(1 for r in results if r['auc_pas_w'] > r['auc_base'])
    f_wins = sum(1 for r in results if r['auc_pas_full'] > r['auc_base'])
    print(f"\n  PAS-S wins:    {s_wins}/{len(results)}")
    print(f"  PAS-W wins:    {w_wins}/{len(results)}")
    print(f"  PAS-Full wins: {f_wins}/{len(results)}")
    print("=" * 95)

    # ── Auto-log ──
    try:
        from utils.result_logger import log_experiment

        cols = ["Class", "Base", "PAS-S", "PAS-W", "PAS-Full", "D_S", "D_W", "D_F"]
        rows = []
        for r in results:
            rows.append([
                r['class'], r['auc_base'], r['auc_pas_s'], r['auc_pas_w'], r['auc_pas_full'],
                r['auc_pas_s'] - r['auc_base'], r['auc_pas_w'] - r['auc_base'],
                r['auc_pas_full'] - r['auc_base']
            ])
        means = [np.mean([r[k] for r in results]) for k in
                 ['auc_base', 'auc_pas_s', 'auc_pas_w', 'auc_pas_full']]
        mean_row = ["MEAN"] + means + [
            means[1] - means[0], means[2] - means[0], means[3] - means[0]
        ]

        log_experiment(
            name="PAS-M3DM 完整管线",
            script="benchmark_pas_full.py",
            description="中心直评 + 双掩码 + Reweighting + OCSVM + 异常加权, 4种变体对比",
            columns=cols,
            rows=rows,
            mean_row=mean_row,
            conclusion=f"PAS-S win {s_wins}/{len(results)}, PAS-W win {w_wins}/{len(results)}, PAS-Full win {f_wins}/{len(results)}",
        )
    except Exception as e:
        print(f"[results] 自动日志失败: {e}")


if __name__ == "__main__":
    main()
