"""
实验 A (端到端): M3DM vs M3DM+PAS — 包含 Point-MAE 特征提取的完整管线
测量: 各组件耗时 (ms) + 显存峰值 (MB)

优化: DINO 特征在 Full 和 PAS 管线间共享，避免重复计算
"""
import os, sys
sys.path.insert(0, '.')
import glob, torch, torch.nn.functional as F, numpy as np, time
from tqdm import tqdm

from models.models import Model
from models.pointnet2_utils import interpolating_points
from utils.pas_core import feature_aware_hybrid_sampling, k_center_greedy_coreset

BASE_FEAT = 'datasets/patch_lib/offline_features/mvtec3d'
BASE_DATA = 'datasets/mvtec3d'
CLASSES = ['bagel', 'cookie', 'potato']  # 3 representative classes
BANK_IMAGES = 30
BANK_STRIDE = 5
CORESET_FRAC = 0.02
NUM_WARMUP = 2
NUM_MEASURE = 10
H, W = 224, 224
SW = 5.0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Args:
    rgb_backbone_name = 'vit_base_patch14_dinov2'
    xyz_backbone_name = 'Point_MAE'
    group_size = 128
    num_group = 1024
    img_size = 224

yg, xg = torch.meshgrid(
    torch.linspace(-1, 1, H, device=device),
    torch.linspace(-1, 1, W, device=device), indexing='ij')
spatial_pe = torch.stack([xg, yg], dim=-1).view(1, H * W, 2)


def entropy_w(d, T=1.0):
    p = F.softmax(d / T, dim=0)
    return 1.0 / (-(p * torch.log(p + 1e-8)).sum() + 1e-5)


def score_full(sfr, sfp, br, bp):
    dr = torch.cdist(sfr.unsqueeze(0), br.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    dp = torch.cdist(sfp.unsqueeze(0), bp.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    wr, wp = entropy_w(dr), entropy_w(dp)
    tot = wr + wp
    fused = (wr / tot) * (dr / (dr.mean() + 1e-5)) + (wp / tot) * (dp / (dp.mean() + 1e-5))
    return fused.topk(10).values.mean().item()


def build_bank_and_gc(train_paths):
    bank_r, bank_p = [], []
    all_r, all_p = [], []
    num_bank = min(BANK_IMAGES, len(train_paths))

    with torch.no_grad():
        for p in tqdm(train_paths[:num_bank], desc="  Building bank", leave=False):
            d = torch.load(p, map_location='cpu')
            fr, fp = d['rgb_features'], d['pts_features']
            nz = d['nonzero_indices'].to(device).view(-1)

            B, Nr, Cr = fr.shape; sr = int(np.sqrt(Nr))
            far = F.interpolate(fr.permute(0, 2, 1).view(B, Cr, sr, sr),
                                size=(224, 224), mode='bilinear', align_corners=False
                                ).view(B, Cr, -1).permute(0, 2, 1)
            fr_f = torch.cat([far.to(device), (spatial_pe * SW).to(device)], dim=-1)

            B, Np, Cp = fp.shape; sp = int(np.sqrt(Np))
            fap = F.interpolate(fp.permute(0, 2, 1).view(B, Cp, sp, sp),
                                size=(224, 224), mode='bilinear', align_corners=False
                                ).view(B, Cp, -1).permute(0, 2, 1)
            fp_f = torch.cat([fap.to(device), (spatial_pe * SW).to(device)], dim=-1)

            bank_r.append(fr_f[:, nz, :].squeeze(0).cpu())
            bank_p.append(fp_f[:, nz, :].squeeze(0).cpu())
            all_r.append(far.squeeze(0).mean(dim=0).cpu())
            all_p.append(fap.squeeze(0).mean(dim=0).cpu())

    gc_r = torch.stack(all_r).mean(dim=0); gc_p = torch.stack(all_p).mean(dim=0)
    sz = torch.zeros(2, device=device)
    gc = torch.cat([gc_r.to(device), sz, gc_p.to(device), sz]).unsqueeze(0).unsqueeze(0)
    gc_rgb = gc[:, :, :768 + 2].contiguous()

    br = torch.cat(bank_r, dim=0); bp = torch.cat(bank_p, dim=0)
    if len(br) > 100000:
        br = br[::BANK_STRIDE]; bp = bp[::BANK_STRIDE]

    br = k_center_greedy_coreset(br, fraction=CORESET_FRAC, device=device).to(device)
    bp = k_center_greedy_coreset(bp, fraction=CORESET_FRAC, device=device).to(device)
    return br, bp, gc, gc_rgb


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


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark_class_e2e(class_name, model):
    print(f"\n{'=' * 60}")
    print(f"[{class_name}] E2E 效率对比 (含 Point-MAE)")
    print(f"{'=' * 60}")

    feat_dir = os.path.join(BASE_FEAT, class_name)
    data_dir = os.path.join(BASE_DATA, class_name)
    train_paths = sorted(glob.glob(os.path.join(feat_dir, 'train', '*.pt')))

    br, bp, gc, gc_rgb = build_bank_and_gc(train_paths)
    print(f"  Bank: rgb={br.shape}, pts={bp.shape}")

    test_rgb = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'rgb', '*.png')))
    test_tiff = sorted(glob.glob(os.path.join(data_dir, 'test', '*', 'xyz', '*.tiff')))

    test_pairs = []
    for r in test_rgb:
        base = os.path.splitext(r)[0].replace('/rgb/', '/xyz/')
        t = base + '.tiff'
        if t in test_tiff:
            test_pairs.append((r, t))
    test_pairs = test_pairs[:NUM_MEASURE + NUM_WARMUP]
    if not test_pairs:
        print(f"  No test pairs, skip"); return None

    t_dino_list, t_pmae_full_list, t_interp_full_list, t_score_full_list = [], [], [], []
    t_pmae_pas_list, t_interp_pas_list, t_score_pas_list = [], [], []
    t_pas_sample_list, t_feat_prep_list = [], []
    mem_full_list, mem_pas_list = [], []
    n_pts_list = []

    for i, (rgb_path, tiff_path) in enumerate(tqdm(test_pairs, desc=f"  {class_name}")):
        try:
            img_tensor, xyz_all, nonzero_mask = load_raw_sample(rgb_path, tiff_path)
        except Exception as e:
            print(f"  Load error: {e}"); continue

        n_pts = int(nonzero_mask.sum().item())
        if n_pts < 100:
            continue
        n_pts_list.append(n_pts)
        is_warmup = i < NUM_WARMUP

        # ═══════════════════════════════════════════════════
        # DINO (shared) — run once, used by both pipelines
        # ═══════════════════════════════════════════════════
        torch.cuda.empty_cache()
        with torch.no_grad():
            t0 = sync_time()
            rgb_feat_28 = model.forward_rgb_features(img_tensor)
            t_dino = (sync_time() - t0) * 1000

        # Prepare RGB features at valid points (shared)
        with torch.no_grad():
            t0 = sync_time()
            rgb_224 = F.interpolate(rgb_feat_28, size=(224, 224), mode='bilinear',
                                     align_corners=False).view(1, 768, -1).permute(0, 2, 1)
            pe_valid = (spatial_pe * SW)[:, nonzero_mask, :]
            rgb_valid_pe = torch.cat([rgb_224[:, nonzero_mask, :], pe_valid], dim=-1)  # [1, N, 770]
            xyz_for_pas = xyz_all.transpose(1, 2).contiguous()  # [1, 3, N] → [1, N, 3]
            t_feat_prep = (sync_time() - t0) * 1000

        # ═══════════════════════════════════════════════════
        # FULL Pipeline: Point-MAE(N) + Interp + Score
        # ═══════════════════════════════════════════════════
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            t0 = sync_time()
            xyz_feat_full, center_full, _, _ = model.xyz_backbone(xyz_all)
            t_pmae_full = (sync_time() - t0) * 1000

            t0 = sync_time()
            interp_full = interpolating_points(xyz_all, center_full.permute(0, 2, 1), xyz_feat_full)
            t_interp_full = (sync_time() - t0) * 1000

            # Build scoring features
            pts_valid = interp_full.permute(0, 2, 1)
            sfr_full = rgb_valid_pe.squeeze(0)
            sfp_full = torch.cat([pts_valid.squeeze(0), pe_valid.squeeze(0)], dim=-1)

            t0 = sync_time()
            _score_full = score_full(sfr_full, sfp_full, br, bp)
            t_score_full = (sync_time() - t0) * 1000

        mem_full = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        # ═══════════════════════════════════════════════════
        # PAS Pipeline: PAS(1024) + Point-MAE(1024) + Score
        # (reuses rgb_valid_pe, xyz_for_pas from above — no DINO re-run)
        # ═══════════════════════════════════════════════════
        # Note: Don't empty_cache — we want to keep DINO/rgb_valid_pe alive
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            # PAS sampling
            t0 = sync_time()
            sidx = feature_aware_hybrid_sampling(
                xyz_for_pas, rgb_valid_pe, 1024, tau=0.6, global_center=gc_rgb)
            t_pas_sample = (sync_time() - t0) * 1000

            # Gather sampled xyz [1, 3, 1024]
            xyz_sampled = xyz_all[:, :, sidx[0]].contiguous()

            # Point-MAE on 1024 points
            t0 = sync_time()
            xyz_feat_pas, center_pas, _, _ = model.xyz_backbone(xyz_sampled)
            t_pmae_pas = (sync_time() - t0) * 1000

            # Interpolate
            t0 = sync_time()
            interp_pas = interpolating_points(xyz_sampled, center_pas.permute(0, 2, 1), xyz_feat_pas)
            t_interp_pas = (sync_time() - t0) * 1000

            # Build scoring features
            sfr_pas = rgb_valid_pe[0, sidx[0], :]
            sfp_pas = torch.cat([
                interp_pas.squeeze(0).permute(1, 0),
                pe_valid[0, sidx[0], :]], dim=-1)

            t0 = sync_time()
            _score_pas = score_full(sfr_pas, sfp_pas, br, bp)
            t_score_pas = (sync_time() - t0) * 1000

        mem_pas = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        if not is_warmup:
            t_dino_list.append(t_dino)
            t_feat_prep_list.append(t_feat_prep)
            t_pmae_full_list.append(t_pmae_full)
            t_interp_full_list.append(t_interp_full)
            t_score_full_list.append(t_score_full)
            t_pas_sample_list.append(t_pas_sample)
            t_pmae_pas_list.append(t_pmae_pas)
            t_interp_pas_list.append(t_interp_pas)
            t_score_pas_list.append(t_score_pas)
            mem_full_list.append(mem_full)
            mem_pas_list.append(mem_pas)

    if not t_dino_list:
        print(f"  No valid samples"); return None

    t_d = np.mean(t_dino_list)
    t_pf = np.mean(t_pmae_full_list); t_if = np.mean(t_interp_full_list); t_sf = np.mean(t_score_full_list)
    t_rs = np.mean(t_pas_sample_list); t_fp = np.mean(t_feat_prep_list)
    t_pr = np.mean(t_pmae_pas_list); t_ir = np.mean(t_interp_pas_list); t_sr = np.mean(t_score_pas_list)

    # Total: DINO (shared) + per-pipeline costs
    t_full_total = t_d + t_fp + t_pf + t_if + t_sf
    t_pas_total = t_d + t_fp + t_rs + t_pr + t_ir + t_sr
    mn_pts = np.mean(n_pts_list)
    mf = np.mean(mem_full_list); mr = np.mean(mem_pas_list)

    print(f"  Avg valid pts: {mn_pts:.0f}")
    print(f"  ┌──────────────────┬──────────┬──────────┬──────────┐")
    print(f"  │ Component        │   Full   │   PAS    │  Speedup │")
    print(f"  ├──────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ DINO             │ {t_d:>8.2f} │ {t_d:>8.2f} │   1.0x   │")
    print(f"  │ Feat Prep        │ {t_fp:>8.2f} │ {t_fp:>8.2f} │   1.0x   │")
    print(f"  │ PAS Sampling     │     -    │ {t_rs:>8.2f} │    -     │")
    print(f"  │ Point-MAE        │ {t_pf:>8.2f} │ {t_pr:>8.2f} │ {t_pf/t_pr:>7.1f}x  │")
    print(f"  │ Interpolate      │ {t_if:>8.2f} │ {t_ir:>8.2f} │ {t_if/t_ir if t_ir>0 else 0:>7.1f}x  │")
    print(f"  │ Score (cdist)    │ {t_sf:>8.2f} │ {t_sr:>8.2f} │ {t_sf/t_sr if t_sr>0 else 0:>7.1f}x  │")
    print(f"  ├──────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ TOTAL            │ {t_full_total:>8.2f} │ {t_pas_total:>8.2f} │ {t_full_total/t_pas_total:>7.1f}x  │")
    print(f"  ├──────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Peak Mem (MB)    │ {mf:>8.1f} │ {mr:>8.1f} │ {mf/mr if mr>0 else 0:>7.1f}x  │")
    print(f"  └──────────────────┴──────────┴──────────┴──────────┘")

    return {'class': class_name,
            't_dino': t_d, 't_feat_prep': t_fp,
            't_pmae_full': t_pf, 't_pmae_pas': t_pr,
            't_interp_full': t_if, 't_interp_pas': t_ir,
            't_score_full': t_sf, 't_score_pas': t_sr,
            't_pas_sample': t_rs,
            't_total_full': t_full_total, 't_total_pas': t_pas_total,
            'mem_full': mf, 'mem_pas': mr,
            'n_pts_mean': mn_pts}


def main():
    print("=" * 70)
    print("实验 A (端到端): M3DM vs M3DM+PAS — 含 Point-MAE 特征提取")
    print("=" * 70)

    print("\nLoading M3DM model (DINO + Point-MAE)...")
    args = Args()
    model = Model(device=device, rgb_backbone_name=args.rgb_backbone_name,
                  xyz_backbone_name=args.xyz_backbone_name,
                  group_size=args.group_size, num_group=args.num_group)
    model.to(device); model.eval()
    print("Model loaded.\n")

    results = []
    for cls in CLASSES:
        r = benchmark_class_e2e(cls, model)
        if r: results.append(r)
        torch.cuda.empty_cache()

    print("\n\n" + "=" * 110)
    print("实验 A 汇总: M3DM 原版 vs M3DM+PAS (端到端, 含特征提取)")
    print("=" * 110)
    print(f"{'Class':<16} {'Full(ms)':>10} {'PAS(ms)':>10} {'Speedup':>8}  "
          f"{'Full(MB)':>10} {'PAS(MB)':>10} {'MemSave':>8}  "
          f"{'PM(Full)':>10} {'PM(PAS)':>10}  {'N_pts':>8}")
    print("-" * 110)

    for r in results:
        sp = r['t_total_full'] / r['t_total_pas'] if r['t_total_pas'] > 0 else 0
        ms = r['mem_full'] / r['mem_pas'] if r['mem_pas'] > 0 else 0
        print(f"{r['class']:<16} {r['t_total_full']:>10.2f} {r['t_total_pas']:>10.2f} {sp:>7.1f}x  "
              f"{r['mem_full']:>10.1f} {r['mem_pas']:>10.1f} {ms:>7.1f}x  "
              f"{r['t_pmae_full']:>10.2f} {r['t_pmae_pas']:>10.2f}  {r['n_pts_mean']:>8.0f}")

    if results:
        mf = np.mean([r['t_total_full'] for r in results])
        mr = np.mean([r['t_total_pas'] for r in results])
        mmf = np.mean([r['mem_full'] for r in results])
        mmr = np.mean([r['mem_pas'] for r in results])
        pmf = np.mean([r['t_pmae_full'] for r in results])
        pmr = np.mean([r['t_pmae_pas'] for r in results])
        print("-" * 110)
        print(f"{'MEAN':<16} {mf:>10.2f} {mr:>10.2f} {mf/mr if mr>0 else 0:>7.1f}x  "
              f"{mmf:>10.1f} {mmr:>10.1f} {mmf/mmr if mmr>0 else 0:>7.1f}x  "
              f"{pmf:>10.2f} {pmr:>10.2f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
