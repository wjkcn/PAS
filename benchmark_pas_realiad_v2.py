"""
PAS-FPS 泛化实验：Real-IAD D3 数据集
直接读取官方目录结构，FPS vs PAS 完整对比。

用法:
    python3 benchmark_pas_realiad_v2.py --data_root datasets/Real-IAD-D3 --classes knob_cap
    python3 benchmark_pas_realiad_v2.py --xyz_backbone PointNet2 --classes knob_cap
"""

import os
import sys
import glob
import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import tifffile
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.utils import set_seeds
from utils.pas_core import k_center_greedy_coreset
from utils.au_pro_util import calculate_au_pro

import warnings
warnings.filterwarnings('ignore')

H, W = 224, 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CORESET_FRAC = 0.02
MEMORY_MAX = 20000
MEMORY_PRE_PER_SAMPLE = 20000
NORMAL_NAMES = {'OK', 'ok', 'good', 'Good', 'normal', 'Normal'}


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载：兼容两种 Real-IAD 目录结构
# ═══════════════════════════════════════════════════════════════════════════

def find_first(sample_dir, patterns):
    """按优先级查找匹配文件。"""
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        if matches:
            return matches[0]
    return None


def make_sample(sample_dir, label, defect_type='good'):
    """从样本目录构建样本字典。"""
    rgb = find_first(sample_dir, ['*RGBL01*.jpg', '*RGBL01*.png', '*rgb*.jpg', '*rgb*.png'])
    xyz = find_first(sample_dir, ['*XYZ*.tiff', '*XYZ*.tif', '*xyz*.tiff'])
    gt = find_first(sample_dir, ['*mask*.png', '*gt*.png', '*RGBL05*.png'])
    pcd = find_first(sample_dir, ['*PCD*.txt', '*pcd*.txt'])

    # 正式实验：缺少必需文件时报错，不允许静默漏样本
    if rgb is None or xyz is None:
        raise FileNotFoundError(
            f"Missing RGB or XYZ file in: {sample_dir}"
        )

    # 异常样本必须具有点级GT
    if label == 1 and gt is None:
        raise FileNotFoundError(
            f"Missing GT mask for abnormal sample: {sample_dir}"
        )

    return {
        'rgb_path': rgb,
        'xyz_path': xyz,
        'pcd_path': pcd,
        'gt_path': gt if label == 1 else None,
        'label': label,
        'defect_type': defect_type,
    }


def scan_realiad_class(data_root, class_name):
    """
    扫描 Real-IAD D3 单个类别的目录结构。
    兼容两种结构：
      A: class/OK/S*/ + class/NG/defect_type/S*/
      B: class/OK/S*/ + class/defect_type/S*/  (defect_type 与 OK 同级)
    """
    class_dir = os.path.join(data_root, class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    # 查找 OK 目录
    ok_dir = None
    for name in NORMAL_NAMES:
        d = os.path.join(class_dir, name)
        if os.path.isdir(d):
            ok_dir = d
            break
    if ok_dir is None:
        raise FileNotFoundError(f"No OK/normal directory found in {class_dir}")

    # 收集 OK 样本
    ok_samples = []
    for sdir in sorted(glob.glob(os.path.join(ok_dir, 'S*'))):
        if not os.path.isdir(sdir):
            continue
        s = make_sample(sdir, 0, 'good')
        if s:
            ok_samples.append(s)

    # 收集 NG 样本：尝试两种结构
    ng_samples = []

    # 结构 A: class/NG/defect_type/S*
    ng_dir = None
    for name in ['NG', 'ng', 'defect', 'Defect', 'anomaly', 'Anomaly']:
        d = os.path.join(class_dir, name)
        if os.path.isdir(d):
            ng_dir = d
            break

    if ng_dir is not None:
        for defect_type in sorted(os.listdir(ng_dir)):
            defect_dir = os.path.join(ng_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            for sdir in sorted(glob.glob(os.path.join(defect_dir, 'S*'))):
                if not os.path.isdir(sdir):
                    continue
                s = make_sample(sdir, 1, defect_type)
                if s:
                    ng_samples.append(s)

    # 结构 B: class/defect_type/S* (defect_type 与 OK 同级)
    if not ng_samples:
        for entry in sorted(os.listdir(class_dir)):
            if entry in NORMAL_NAMES or entry.stapaswith('.'):
                continue
            defect_dir = os.path.join(class_dir, entry)
            if not os.path.isdir(defect_dir):
                continue
            # 检查是否直接包含 S* 子目录
            sdirs = sorted(glob.glob(os.path.join(defect_dir, 'S*')))
            if not sdirs:
                continue
            for sdir in sdirs:
                if not os.path.isdir(sdir):
                    continue
                s = make_sample(sdir, 1, entry)
                if s:
                    ng_samples.append(s)

    return ok_samples, ng_samples


# ═══════════════════════════════════════════════════════════════════════════
# 2. 点云加载
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_xyz(tiff_path):
    """打印 XYZ TIFF 各通道统计，用于判断格式。"""
    tiff_img = tifffile.imread(tiff_path)
    if tiff_img.ndim == 3 and tiff_img.shape[0] == 3:
        tiff_img = tiff_img.transpose(1, 2, 0)
    print(f"  [DIAG] {os.path.basename(tiff_path)} shape={tiff_img.shape}")
    for c in range(tiff_img.shape[-1]):
        arr = tiff_img[:, :, c]
        valid = arr[np.isfinite(arr) & (arr != 0)]
        if len(valid) > 0:
            n_unique = len(np.unique(valid[:10000]))
            print(f"    ch{c}: min={valid.min():.6f} max={valid.max():.6f} unique(sample)={n_unique}")
        else:
            print(f"    ch{c}: all zero/NaN")


def _check_size_alignment(ok_sample, ng_sample=None):
    """检查RGB、GT和TIFF的尺寸是否对齐，并检查GT像素值。"""
    print("\n  [DIAG] Size alignment check:")
    
    # 检查OK样本
    with Image.open(ok_sample['rgb_path']) as im:
        rgb_size = im.size
    print(f"    OK RGB size: {rgb_size}")
    
    xyz_raw = tifffile.imread(ok_sample['xyz_path'])
    xyz_shape = xyz_raw.shape
    print(f"    OK XYZ shape: {xyz_shape}")
    
    if ok_sample['gt_path'] and os.path.exists(ok_sample['gt_path']):
        with Image.open(ok_sample['gt_path']) as im:
            gt_size = im.size
        print(f"    OK GT size:  {gt_size}")
    
    # 检查NG样本
    if ng_sample:
        with Image.open(ng_sample['rgb_path']) as im:
            rgb_size = im.size
        print(f"    NG RGB size: {rgb_size}")
        
        xyz_raw = tifffile.imread(ng_sample['xyz_path'])
        xyz_shape = xyz_raw.shape
        print(f"    NG XYZ shape: {xyz_shape}")
        
        if ng_sample['gt_path'] and os.path.exists(ng_sample['gt_path']):
            with Image.open(ng_sample['gt_path']) as im:
                gt_size = im.size
            print(f"    NG GT size:  {gt_size}")
            
            # 检查GT像素值
            gt_arr = np.array(Image.open(ng_sample['gt_path']).convert("L"))
            unique_vals = np.unique(gt_arr)
            print(f"    NG GT unique values: {unique_vals}")
            if len(unique_vals) == 2 and set(unique_vals) == {0, 255}:
                print(f"    GT format: binary [0, 255] -> threshold > 128 is correct")
            elif len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                print(f"    GT format: binary [0, 1] -> threshold should be > 0")
            else:
                print(f"    WARNING: GT has non-binary values! Check if this is correct mask.")
    
    print("    WARNING: Ensure RGB, GT, and XYZ have same FOV and orientation before scaling!")
    print("    Save an overlay check image to verify alignment.\n")


def load_pointcloud(xyz_path, pcd_path=None, target_size=H):
    """加载点云，自动检测 Real-IAD 格式（X,Y 常数或全零）。
    
    当检测到X/Y通道为常数时，使用TIFF第三通道(Z)构造pseudo-3D点云，
    而非读取稀疏PCD文件（PCD点数通常不等于像素数）。
    """
    tiff_img = tifffile.imread(xyz_path)
    if tiff_img.ndim == 3 and tiff_img.shape[0] == 3:
        tiff_img = tiff_img.transpose(1, 2, 0)

    H_orig, W_orig = tiff_img.shape[:2]

    # 检测 Real-IAD 格式：X,Y 通道是否为常数或全零
    x_ch, y_ch = tiff_img[:, :, 0], tiff_img[:, :, 1]
    x_valid = x_ch[np.isfinite(x_ch) & (x_ch != 0)]
    y_valid = y_ch[np.isfinite(y_ch) & (y_ch != 0)]

    x_unique = len(np.unique(x_valid)) if len(x_valid) > 0 else 0
    y_unique = len(np.unique(y_valid)) if len(y_valid) > 0 else 0

    xy_invalid = (
        len(x_valid) == 0
        or len(y_valid) == 0
        or x_unique <= 3
        or y_unique <= 3
    )

    if xy_invalid:
        # X/Y 通道为常数，使用TIFF Z通道构造pseudo-3D点云
        # 这是Real-IAD D3提供的pseudo-3D模态，与稀疏PCD不同
        z_ch = tiff_img[:, :, 2].astype(np.float32)
        
        # 从TIFF X/Y通道提取采样间距（如果非零）
        # 注意：需要确认0.166等值确实代表采样间距
        sx = abs(float(np.median(x_valid))) if len(x_valid) else 1.0
        sy = abs(float(np.median(y_valid))) if len(y_valid) else 1.0
        sx = sx if sx > 1e-8 else 1.0
        sy = sy if sy > 1e-8 else 1.0
        
        # 构造像素网格坐标
        x = np.arange(W_orig, dtype=np.float32) * sx
        y = np.arange(H_orig, dtype=np.float32) * sy
        xx, yy = np.meshgrid(x, y)
        
        # 组织成 [H, W, 3] 的有序点云
        organized_pc = np.stack([xx, yy, z_ch], axis=-1)
    else:
        organized_pc = tiff_img.astype(np.float32)

    # 调整大小
    pc_tensor = torch.from_numpy(organized_pc).float()
    pc_tensor = pc_tensor.permute(2, 0, 1).unsqueeze(0)
    pc_resized = F.interpolate(pc_tensor, size=(target_size, target_size), mode='nearest')
    organized_pc = pc_resized.squeeze(0).permute(1, 2, 0).numpy()

    pc_flat = organized_pc.reshape(-1, 3)
    
    # 有效点掩膜：基于Z通道判断（修复问题2）
    # 使用Z通道的有限性和非零性判断，而非X/Y（因为网格构造的X/Y始终非零）
    valid_mask = (
        np.all(np.isfinite(pc_flat), axis=1)
        & (np.abs(pc_flat[:, 2]) > 1e-8)
    )
    nonzero = np.nonzero(valid_mask)[0]

    return organized_pc, valid_mask, nonzero



# ═══════════════════════════════════════════════════════════════════════════
# 3. 特征提取
# ═══════════════════════════════════════════════════════════════════════════

def set_sampling_mode(backbone, mode):
    """强制设置采样模式，失败时报错。"""

    if hasattr(backbone, 'set_sampling_mode'):
        backbone.set_sampling_mode(mode)
        return

    candidates = [
        backbone,
        getattr(backbone, 'group_divider', None),
    ]

    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, '_sampling_mode') or hasattr(obj, 'forward'):
            obj._sampling_mode = mode
            return

    raise RuntimeError(
        f"Cannot set sampling mode to '{mode}'. "
        "Check the actual xyz_backbone sampling interface."
    )


def extract_rgb_features(model, img_tensor):
    """提取 DINO 2D 特征: [1, 768, 16, 16]。"""
    return model.forward_rgb_features(img_tensor)


def compute_anomaly_prior(features_2d, nonzero):
    """
    论文先验：上采样到 224x224 → 完整 50176 位置 → 计算当前图像均值偏离 → 提取有效点。
    """
    feat_up = F.interpolate(features_2d, size=(H, W), mode='bilinear', align_corners=False)
    feat_flat = feat_up.squeeze(0).permute(1, 2, 0).reshape(-1, features_2d.shape[1])
    mean_feat = feat_flat.mean(dim=0, keepdim=True)
    scores_all = 1.0 - F.cosine_similarity(feat_flat, mean_feat, dim=-1)
    return scores_all[nonzero].unsqueeze(0).float()  # [1, N_valid]


def extract_3d_features(model, xyz_valid, scores_nz, pn2_backbone=None):
    """
    提取 3D 特征。支持 Point_MAE 和 PointNet2。
    xyz_valid: [1, N, 3], scores_nz: [1, N] 或 None

    Point_MAE:  返回 group_feat [1, 1152, G], center_idx [1, G]
    PointNet2:  返回 per_pt_feat [1, 64, N], center_idx [1, 512] (SA1)
                 特征已经逐点，不需要 nearest-center 映射

    Note: PAS replaces only the first sampling bottleneck (SA1).
          The anomaly prior is passed only to SA1; SA2 and subsequent
          layers retain standard FPS to preserve the original backbone
          architecture.
    """

    if pn2_backbone is not None:
        mode = 'pas' if scores_nz is not None else 'fps'
        pn2_backbone.set_sampling_mode(mode)
        per_pt_feat, sa_xyzs, sa_feats, sa_indices = pn2_backbone(xyz_valid, scores_nz)
        return per_pt_feat, None, None, sa_indices[0]

    xyz_input = xyz_valid.permute(0, 2, 1).contiguous()  # [1, 3, N]

    mode = 'pas' if scores_nz is not None else 'fps'
    set_sampling_mode(model.xyz_backbone, mode)

    return model.xyz_backbone(xyz_input, scores_nz)


def _normalize_pointcloud(xyz_np):
    """
    对点云进行中心化和尺度归一化（与MVTec主实验一致）。
    
    Args:
        xyz_np: [N, 3] numpy array
    
    Returns:
        xyz_normalized: [N, 3] numpy array, centered and scaled to [-1, 1]
    """
    # 中心化
    centroid = np.mean(xyz_np, axis=0)
    xyz_centered = xyz_np - centroid
    
    # 尺度归一化（缩放到单位球）
    max_dist = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
    if max_dist > 1e-6:
        xyz_normalized = xyz_centered / max_dist
    else:
        xyz_normalized = xyz_centered
    
    return xyz_normalized


# ═══════════════════════════════════════════════════════════════════════════
# 4. 记忆银行构建
# ═══════════════════════════════════════════════════════════════════════════

def build_memory_bank(model, samples, device, use_pas=True, max_samples=50, pn2_backbone=None):
    """
    构建 FPS 或 PAS 记忆银行。
    use_pas=False: 纯 FPS 采样（scores_nz=None）
    use_pas=True:  PAS（先验引导采样）
    """
    all_features = []
    skipped_count = 0
    sample_list = samples[:max_samples]

    for s in tqdm(sample_list, desc=f"  Building {'PAS' if use_pas else 'FPS'} bank", leave=False):
        img = Image.open(s['rgb_path']).convert('RGB').resize((H, W), Image.BILINEAR)
        img_tensor = (torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                      - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

        pc_np, _, nonzero = load_pointcloud(s['xyz_path'], s.get('pcd_path'))
        if len(nonzero) < 100:
            skipped_count += 1
            continue

        pc_flat = pc_np.reshape(-1, 3)
        xyz = pc_flat[nonzero].copy().astype(np.float32)
        
        # 点云归一化（与MVTec主实验一致）
        xyz = _normalize_pointcloud(xyz)
        xyz_t = torch.from_numpy(xyz).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_2d = extract_rgb_features(model, img_tensor)
            scores_nz = compute_anomaly_prior(feat_2d, nonzero) if use_pas else None
            group_feat, _, _, center_idx = extract_3d_features(model, xyz_t, scores_nz, pn2_backbone)
            all_features.append(group_feat.squeeze(0).T.cpu())  # [G, C] or [N, C]

    # 正式实验：训练样本不足时报错
    if skipped_count > 0:
        raise RuntimeError(
            f"Memory bank building: {skipped_count}/{len(sample_list)} training samples "
            f"were skipped due to insufficient valid points (< 100). "
            "This will affect the memory bank quality."
        )
    
    if not all_features:
        return None, None, None

    all_features = torch.cat(all_features, dim=0)

    # Pre-sample large feature sets for PointNet2 to keep coreset tractable
    if pn2_backbone is not None:
        limit = max(20000, len(sample_list) * MEMORY_PRE_PER_SAMPLE)
        if all_features.size(0) > limit:
            perm = torch.randperm(all_features.size(0))[:limit]
            all_features = all_features[perm]

    # Coreset 下采样
    num_features = all_features.size(0)
    target_size = min(MEMORY_MAX, max(1, int(round(num_features * CORESET_FRAC))))

    if target_size < num_features:
        fraction = target_size / num_features
        coreset_out = k_center_greedy_coreset(all_features, fraction=fraction, device=device)

        # 安全检查：确认返回的是特征还是索引
        if coreset_out.ndim == 1:
            # 返回的是索引
            coreset = all_features[coreset_out.long().cpu()]
        elif (coreset_out.ndim == 2 and coreset_out.shape[1] == all_features.shape[1]):
            # 返回的是特征
            coreset = coreset_out
        else:
            raise ValueError(
                f"Unexpected coreset output shape: {tuple(coreset_out.shape)}"
            )
    else:
        coreset = all_features

    # 按特征维度归一化（与 MVTec 3D-AD 主实验一致）
    mem_mean = coreset.mean(dim=0, keepdim=True)
    mem_std = coreset.std(dim=0, keepdim=True).clamp(min=1e-8)
    memory = (coreset - mem_mean) / mem_std
    return memory, mem_mean, mem_std


# ═══════════════════════════════════════════════════════════════════════════
# 5. 密度加权（与主实验一致）
# ═══════════════════════════════════════════════════════════════════════════

def _compute_center_density(target_xyz, center_xyz, K=16):
    """
    计算每个有效点在采样中心集合中的局部密度，与主实验统一。
    
    target_xyz: [N, 3] 所有有效点
    center_xyz: [M, 3] SA1 采样中心
    返回: [N] 归一化密度权重（高密度区域权重 > 1）
    """
    dist = torch.cdist(target_xyz, center_xyz)  # [N, M]
    k = min(K, center_xyz.size(0))
    nn_dist = torch.topk(dist, k, dim=-1, largest=False).values  # [N, k]
    mean_dist = nn_dist.mean(dim=-1)  # [N]

    # 密度：距离越小越密集
    sigma = mean_dist.median().clamp(min=1e-6) * 2.0
    density = torch.exp(-mean_dist / sigma)
    density = density / (density.mean() + 1e-8)
    return density


# ═══════════════════════════════════════════════════════════════════════════
# 6. 测试评分
# ═══════════════════════════════════════════════════════════════════════════

def score_sample(model, memory, mem_mean, mem_std, img_tensor, pc_np, nonzero,
                 device, use_pas=True, use_density=True, density_alpha=0.5, pn2_backbone=None):
    """
    计算单个样本的图像级和点级异常分数。
    FPS: 标准距离，不使用密度加权。
    PAS-Sampling: 先验引导采样，不使用密度加权。
    PAS-Full: 先验引导采样 + 密度加权。
    返回: img_score (max-pooling), point_scores [N_valid]
    """
    pc_flat = pc_np.reshape(-1, 3)
    xyz = pc_flat[nonzero].copy().astype(np.float32)
    
    # 点云归一化（与MVTec主实验一致）
    xyz = _normalize_pointcloud(xyz)
    xyz_t = torch.from_numpy(xyz).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_2d = extract_rgb_features(model, img_tensor)
        scores_nz = compute_anomaly_prior(feat_2d, nonzero) if use_pas else None
        group_feat, _, _, center_idx = extract_3d_features(model, xyz_t, scores_nz, pn2_backbone)

    feat_norm = (group_feat.squeeze(0).T.float() - mem_mean.to(device)) / (mem_std.to(device) + 1e-8)
    min_dist = torch.cdist(feat_norm, memory.to(device)).min(dim=1)[0]  # [G] or [N]

    if pn2_backbone is not None:
        center_xyz = xyz_t[0, center_idx.squeeze(0).long()]  # [512, 3]
        point_min_dist = min_dist  # already per-point
    else:
        center_idx_flat = center_idx.squeeze(0).long()
        center_xyz = xyz_t[0, center_idx_flat]  # [G, 3]
        nearest_center = torch.cdist(xyz_t[0], center_xyz).argmin(dim=1)  # [N]
        point_min_dist = min_dist[nearest_center]  # [N]

    if use_pas and use_density:
        density_w = _compute_center_density(xyz_t[0], center_xyz, K=16)  # [N]
        point_min_dist = point_min_dist * (
            1.0 + density_alpha * density_w
        )  # alpha=0.5 matches paper default

    point_scores = point_min_dist.cpu().numpy()
    img_score = float(point_min_dist.max().item())

    return img_score, point_scores


# ═══════════════════════════════════════════════════════════════════════════
# 7. 评估
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_class(class_name, model, data_root, device, max_train=50, diagnose=False, use_density=True,
                   density_alpha=0.5, sampling_seed=42, bank_seed=None, test_seed=0, fixed_splits=None, pn2_backbone=None):
    """
    Args:
        sampling_seed: 控制FPS起点及其他模型外随机过程（固定为42）
        bank_seed: 只控制正常训练对象抽取（必需）
        test_seed: 只控制测试集划分（固定为0）
    """
    print(f"\n{'='*60}")
    print(f"  {class_name}")
    print(f"{'='*60}")

    ok_samples, ng_samples = scan_realiad_class(data_root, class_name)
    n_ok = len(ok_samples)
    n_ng = len(ng_samples)
    print(f"  OK samples: {n_ok}, NG samples: {n_ng}")

    if n_ok < 10:
        print(f"  [SKIP] Too few OK samples: {n_ok}")
        return None
    if n_ng == 0:
        print(f"  [SKIP] No NG samples found")
        return None

    # 诊断：打印第一个样本的 XYZ 通道信息和尺寸对齐检查
    if diagnose and ok_samples:
        diagnose_xyz(ok_samples[0]['xyz_path'])
        # 打印路径信息便于验证
        print(f"  [DIAG] Sample paths (first NG sample):")
        if ng_samples:
            s = ng_samples[0]
            print(f"    RGB: {s['rgb_path']}")
            print(f"    XYZ: {s['xyz_path']}")
            print(f"    PCD: {s.get('pcd_path', 'N/A')}")
            print(f"    GT : {s['gt_path']}")
        
        # 尺寸对齐检查
        _check_size_alignment(ok_samples[0], ng_samples[0] if ng_samples else None)

    # 固定测试集划分（使用 test_seed 的独立 RNG，所有种子共享同一测试集）
    rng_test = np.random.default_rng(test_seed)
    ok_indices = rng_test.permutation(n_ok)
    n_test_ok = min(n_ok // 3, 50)
    test_idx = set(ok_indices[:n_test_ok].tolist())
    train_pool_idx = ok_indices[n_test_ok:].tolist()

    # 银行样本选择（使用 bank_seed 的独立 RNG，不同种子选择不同的建库样本）
    if fixed_splits and class_name in fixed_splits:
        # 使用预计算的固定划分
        fs = fixed_splits[class_name]
        bank_train_ok_paths = set(fs['bank_train_ok'])
        test_ok_paths = set(fs['test_ok'])
        bank_train_ok = [s for s in ok_samples if os.path.relpath(s['rgb_path'], data_root) in bank_train_ok_paths]
        test_ok = [s for s in ok_samples if os.path.relpath(s['rgb_path'], data_root) in test_ok_paths]
    else:
        # 从训练池中用 bank_seed 的独立 RNG 采样建库样本
        rng_bank = np.random.default_rng(bank_seed)
        train_pool = [ok_samples[i] for i in train_pool_idx]
        bank_perm = rng_bank.permutation(len(train_pool))
        bank_train_ok = [train_pool[i] for i in bank_perm[:max_train]]
        test_ok = [ok_samples[i] for i in sorted(test_idx)]

    test_all = test_ok + ng_samples

    print(f"  Bank train: {len(bank_train_ok)}, Test OK: {len(test_ok)}, Test NG: {n_ng}")

    # 构建 FPS 和 PAS 记忆银行
    print("  Building FPS memory bank...")
    fps_mem, fps_mean, fps_std = build_memory_bank(model, bank_train_ok, device, use_pas=False, max_samples=len(bank_train_ok), pn2_backbone=pn2_backbone)
    if fps_mem is None:
        print("  [SKIP] Failed to build FPS bank")
        return None

    print("  Building PAS memory bank...")
    pas_mem, pas_mean, pas_std = build_memory_bank(model, bank_train_ok, device, use_pas=True, max_samples=len(bank_train_ok), pn2_backbone=pn2_backbone)
    if pas_mem is None:
        print("  [SKIP] Failed to build PAS bank")
        return None

    fps_bank_size = fps_mem.shape[0]
    pas_bank_size = pas_mem.shape[0]
    print(f"  FPS bank: {fps_bank_size} features, PAS bank: {pas_bank_size} features")

    # 测试
    labels = []
    img_scores_fps, img_scores_pas = [], []
    all_gt, all_fg_fps, all_fg_pas = [], [], []
    score_maps_fps, score_maps_pas, gt_maps, valid_masks = [], [], [], []
    used_ok, used_ng, skipped = 0, 0, 0

    for s in tqdm(test_all, desc=f"  Evaluating {class_name}", leave=False):
        try:
            img = Image.open(s['rgb_path']).convert('RGB').resize((H, W), Image.BILINEAR)
            img_tensor = (torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                          - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            pc_np, _, nonzero = load_pointcloud(s['xyz_path'], s.get('pcd_path'))
        except Exception as e:
            print(f"  [ERROR] {s['rgb_path']}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        if len(nonzero) < 100:
            skipped += 1
            continue

        try:
            # FPS: 不使用先验采样，不使用密度校准
            img_fps, pts_fps = score_sample(model, fps_mem, fps_mean, fps_std,
                                            img_tensor, pc_np, nonzero, device, 
                                            use_pas=False, use_density=False, density_alpha=density_alpha, pn2_backbone=pn2_backbone)
            # PAS: 使用先验采样，密度校准由参数控制
            img_pas, pts_pas = score_sample(model, pas_mem, pas_mean, pas_std,
                                            img_tensor, pc_np, nonzero, device, 
                                            use_pas=True, use_density=use_density, density_alpha=density_alpha, pn2_backbone=pn2_backbone)
        except Exception as e:
            print(f"  [ERROR] scoring {s['rgb_path']}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        # GT mask（自动适配[0,1]和[0,255]两种格式）
        if s['gt_path'] and os.path.exists(s['gt_path']):
            gt = Image.open(s['gt_path']).convert('L').resize((H, W), Image.NEAREST)
            gt_arr = np.asarray(gt)
            unique_vals = np.unique(gt_arr)
            
            if np.all(np.isin(unique_vals, [0, 1])):
                gt_map_2d = (gt_arr > 0).astype(np.float32)
            elif np.all(np.isin(unique_vals, [0, 255])):
                gt_map_2d = (gt_arr > 128).astype(np.float32)
            else:
                raise ValueError(
                    f"Unexpected GT values in {s['gt_path']}: {unique_vals[:20]}"
                )
            
            gt_mask = gt_map_2d.reshape(-1)
            gt_fg = gt_mask[nonzero]
        else:
            gt_fg = np.zeros(len(nonzero), dtype=np.float32)
            gt_map_2d = np.zeros((H, W), dtype=np.float32)

        # 收集2D score maps用于AUPRO计算
        # 创建有效区域mask，用于屏蔽无效背景
        valid_mask_2d = np.zeros((H, W), dtype=bool)
        score_map_fps_2d = np.zeros((H, W), dtype=np.float32)
        score_map_pas_2d = np.zeros((H, W), dtype=np.float32)
        # 将点级分数映射回2D图像
        for i, idx in enumerate(nonzero):
            row, col = divmod(idx, W)
            if row < H and col < W:
                score_map_fps_2d[row, col] = pts_fps[i]
                score_map_pas_2d[row, col] = pts_pas[i]
                valid_mask_2d[row, col] = True

        labels.append(s['label'])
        img_scores_fps.append(img_fps)
        img_scores_pas.append(img_pas)
        all_gt.append(gt_fg)
        all_fg_fps.append(pts_fps)
        all_fg_pas.append(pts_pas)
        score_maps_fps.append(score_map_fps_2d)
        score_maps_pas.append(score_map_pas_2d)
        gt_maps.append(gt_map_2d)
        valid_masks.append(valid_mask_2d)
        
        # 统计实际成功样本数
        if s['label'] == 0:
            used_ok += 1
        else:
            used_ng += 1

    if len(labels) < 2 or len(set(labels)) < 2:
        print("  [SKIP] Not enough samples for AUC")
        return None

    # 正式运行时不允许静默跳过测试样本
    if skipped > 0:
        raise RuntimeError(
            f"{class_name}: {skipped} test samples were skipped due to errors. "
            "Fix the errors before running final experiments."
        )

    # 计算指标
    img_auc_fps = float(roc_auc_score(labels, img_scores_fps))
    img_auc_pas = float(roc_auc_score(labels, img_scores_pas))

    gt_all = np.concatenate(all_gt)
    fg_fps = np.nan_to_num(np.concatenate(all_fg_fps), nan=0.0)
    fg_pas = np.nan_to_num(np.concatenate(all_fg_pas), nan=0.0)

    if len(np.unique(gt_all)) > 1:
        ptauc_fps = float(roc_auc_score(gt_all, fg_fps))
        ptauc_pas = float(roc_auc_score(gt_all, fg_pas))
    else:
        ptauc_fps = ptauc_pas = 0.5

    # 计算 AUPRO（使用valid_masks真正排除无效背景）
    try:
        if gt_maps and score_maps_fps:
            aupro_fps, _ = calculate_au_pro(gt_maps, score_maps_fps, integration_limit=0.3)
            aupro_pas, _ = calculate_au_pro(gt_maps, score_maps_pas, integration_limit=0.3)
            aupro_fps = float(aupro_fps)
            aupro_pas = float(aupro_pas)
        else:
            raise RuntimeError("No valid GT maps or score maps for AUPRO calculation")
    except Exception as e:
        raise RuntimeError(f"AUPRO calculation failed for {class_name}: {e}") from e

    print(f"  ImgAUC:  FPS={img_auc_fps:.4f}  PAS={img_auc_pas:.4f}  Δ={img_auc_pas-img_auc_fps:+.4f}")
    print(f"  PtAUC:   FPS={ptauc_fps:.4f}  PAS={ptauc_pas:.4f}  Δ={ptauc_pas-ptauc_fps:+.4f}")
    print(f"  AUPRO:   FPS={aupro_fps:.4f}  PAS={aupro_pas:.4f}  Δ={aupro_pas-aupro_fps:+.4f}")
    print(f"  Used samples: OK={used_ok}, NG={used_ng}, Skipped={skipped}")

    # 保存划分信息便于复现（使用相对路径避免文件名重复）
    def rel_sample_path(sample):
        return os.path.relpath(sample['rgb_path'], data_root)
    
    split_info = {
        'bank_train_ok': [rel_sample_path(s) for s in bank_train_ok],
        'test_ok': [rel_sample_path(s) for s in test_ok],
        'test_ng': [rel_sample_path(s) for s in ng_samples],
        'used_ok': used_ok,
        'used_ng': used_ng,
        'skipped': skipped,
        'sampling_seed': sampling_seed,
        'bank_seed': bank_seed,
        'test_seed': test_seed,
    }

    return {
        'class': class_name,
        'n_train': len(bank_train_ok),
        'n_test_ok': used_ok,
        'n_test_ng': used_ng,
        'n_skipped': skipped,
        'img_auc_fps': img_auc_fps,
        'img_auc_pas': img_auc_pas,
        'ptauc_fps': ptauc_fps,
        'ptauc_pas': ptauc_pas,
        'aupro_fps': aupro_fps,
        'aupro_pas': aupro_pas,
        'fps_bank_size': fps_bank_size,
        'pas_bank_size': pas_bank_size,
        'split': split_info,
    }


def main():
    parser = argparse.ArgumentParser(description='PAS-FPS 泛化实验: Real-IAD D3')
    parser.add_argument('--data_root', default='datasets/Real-IAD-D3', type=str,
                        help='Real-IAD D3 数据集根目录')
    parser.add_argument('--classes', nargs='+', default=None, type=str)
    parser.add_argument('--max_train', default=50, type=int)
    parser.add_argument('--diagnose', action='store_true', help='打印 XYZ 通道诊断信息')
    parser.add_argument('--output_json', default='results/pas_realiad_v2.json')
    parser.add_argument('--no_density', action='store_true', help='禁用密度校准（用于消融实验）')
    parser.add_argument('--density_alpha', type=float, default=0.5,
                        help='Strength of density-aware point-score calibration.')
    parser.add_argument('--sampling_seed', type=int, default=42,
                        help='控制FPS起点、随机探索等模型外随机过程（固定为42）')
    parser.add_argument('--bank_seed', type=int, required=True,
                        help='建库样本选择种子（控制正常训练对象抽取）')
    parser.add_argument('--test_seed', type=int, default=0,
                        help='测试集划分种子（固定测试集用，默认0）')
    parser.add_argument('--fixed_splits', default=None, type=str, help='固定划分JSON文件路径（可选）')
    parser.add_argument('--xyz_backbone', default='Point_MAE', type=str, choices=['Point_MAE', 'PointNet2'],
                        help='3D backbone: Point_MAE (default) or PointNet2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"sampling_seed: {args.sampling_seed} (FPS起点、随机探索等模型外随机)")
    print(f"bank_seed: {args.bank_seed} (建库样本选择)")
    print(f"test_seed: {args.test_seed} (测试集划分)")
    print(f"3D Backbone: {args.xyz_backbone}")

    # sampling_seed 控制 FPS 起点、随机探索及全局 PyTorch/numpy 随机状态
    set_seeds(args.sampling_seed)

    pn2_backbone = None
    if args.xyz_backbone == 'PointNet2':
        # PointNet2 模式下只加载 DINOv2（PAS需要二维先验），不加载 Point_MAE
        print("Loading DINOv2 model (RGB features only, PointNet2 mode)...")
        from models.models import Model as M3DMModel
        model = M3DMModel(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                          xyz_backbone_name='Point_MAE', group_size=128, num_group=1024,
                          load_xyz=False)
        model.to(device).eval()
        print("Loading PointNet2 backbone...")
        from backbones.pointnet2_seg import PointNet2SegBackbone
        pn2_backbone = PointNet2SegBackbone(tau=0.6)
        pn2_backbone.to(device).eval()
    else:
        print("Loading M3DM model (DINOv2 + PointTransformer + Point-MAE)...")
        from models.models import Model as M3DMModel
        model = M3DMModel(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                          xyz_backbone_name='Point_MAE', group_size=128, num_group=1024)
        model.to(device).eval()

    print("Model loaded.\n")

    # 扫描可用类别
    if args.classes:
        classes = args.classes
    else:
        classes = [d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)) and not d.stapaswith('.')]

    # 密度校准开关
    use_density = not args.no_density
    print(f"Density calibration: {'ON' if use_density else 'OFF'}")
    print(f"Density alpha: {args.density_alpha}")

    # 加载固定划分（如果提供）
    fixed_splits = None
    if args.fixed_splits and os.path.exists(args.fixed_splits):
        with open(args.fixed_splits, 'r') as f:
            fixed_splits = json.load(f)
        print(f"Loaded fixed splits from {args.fixed_splits}")

    results = []
    for cls in sorted(classes):
        try:
            r = evaluate_class(cls, model, args.data_root, device, args.max_train,
                               diagnose=args.diagnose, use_density=use_density,
                               density_alpha=args.density_alpha,
                               sampling_seed=args.sampling_seed,
                               test_seed=args.test_seed, bank_seed=args.bank_seed,
                               fixed_splits=fixed_splits,
                               pn2_backbone=pn2_backbone)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n[{cls}] ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # 正式实验：任一类别失败立即终止，避免生成不完整结果
            raise RuntimeError(
                f"Class '{cls}' failed. Aborting to avoid incomplete results."
            ) from e
        torch.cuda.empty_cache()

    if not results:
        print("\nNo results.")
        return

    # 汇总
    print(f"\n{'='*100}")
    print(f"  PAS-FPS 泛化实验 — Real-IAD D3 Summary (Backbone: {args.xyz_backbone})")
    print(f"{'='*100}")
    print(f"{'Class':<20} {'ImgAUC':>12} {'PtAUC':>12} {'AUPRO':>12}")
    print(f"{'':20} {'FPS':>6} {'PAS':>6} {'FPS':>6} {'PAS':>6} {'FPS':>6} {'PAS':>6} {'Δ_img':>7} {'Δ_pt':>7} {'Δ_pro':>7}")
    print("-" * 100)
    for r in results:
        d_img = r['img_auc_pas'] - r['img_auc_fps']
        d_pt = r['ptauc_pas'] - r['ptauc_fps']
        d_pro = r['aupro_pas'] - r['aupro_fps']
        print(f"{r['class']:<20} {r['img_auc_fps']:>6.4f} {r['img_auc_pas']:>6.4f} "
              f"{r['ptauc_fps']:>6.4f} {r['ptauc_pas']:>6.4f} "
              f"{r['aupro_fps']:>6.4f} {r['aupro_pas']:>6.4f} "
              f"{d_img:>+7.4f} {d_pt:>+7.4f} {d_pro:>+7.4f}")
    print("-" * 100)
    mean_img_fps = np.mean([r['img_auc_fps'] for r in results])
    mean_img_pas = np.mean([r['img_auc_pas'] for r in results])
    mean_pt_fps = np.mean([r['ptauc_fps'] for r in results])
    mean_pt_pas = np.mean([r['ptauc_pas'] for r in results])
    mean_pro_fps = np.mean([r['aupro_fps'] for r in results])
    mean_pro_pas = np.mean([r['aupro_pas'] for r in results])
    print(f"{'MEAN':<20} {mean_img_fps:>6.4f} {mean_img_pas:>6.4f} "
          f"{mean_pt_fps:>6.4f} {mean_pt_pas:>6.4f} "
          f"{mean_pro_fps:>6.4f} {mean_pro_pas:>6.4f} "
          f"{mean_img_pas-mean_img_fps:>+7.4f} {mean_pt_pas-mean_pt_fps:>+7.4f} {mean_pro_pas-mean_pro_fps:>+7.4f}")

    # 保存
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    # 不保存 split 信息到主结果文件（太大）
    save_results = [{k: v for k, v in r.items() if k != 'split'} for r in results]
    # 单独保存划分信息
    split_path = args.output_json.replace('.json', '_split.json')
    with open(split_path, 'w') as f:
        json.dump({r['class']: r['split'] for r in results}, f, indent=2)

    with open(args.output_json, 'w') as f:
        json.dump({'results': save_results, 'summary': {
            'mean_img_auc_fps': mean_img_fps, 'mean_img_auc_pas': mean_img_pas,
            'mean_ptauc_fps': mean_pt_fps, 'mean_ptauc_pas': mean_pt_pas,
            'mean_aupro_fps': mean_pro_fps, 'mean_aupro_pas': mean_pro_pas,
            'sampling_seed': args.sampling_seed,
            'bank_seed': args.bank_seed,
            'test_seed': args.test_seed,
            'xyz_backbone': args.xyz_backbone,
            'density_alpha': args.density_alpha,
            'prior_type': 'per_image_mean',
        }}, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
    print(f"Split info saved to {split_path}")

    # 保存固定划分供后续种子复用
    fixed_path = args.output_json.replace('.json', '_fixed_splits.json')
    fixed_out = {}
    for r in results:
        cls = r['class']
        fixed_out[cls] = r['split']
    with open(fixed_path, 'w') as f:
        json.dump(fixed_out, f, indent=2)
    print(f"Fixed splits saved to {fixed_path} (reuse with --fixed_splits)")


if __name__ == "__main__":
    main()
