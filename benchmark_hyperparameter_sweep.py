#!/usr/bin/env python
"""Hyperparameter sensitivity sweep — τ (hybrid ratio) and K (memory bank size).

τ ∈ {0.2, 0.4, 0.6, 0.8} with α=0.5, max_sample=100
K ∈ {50, 200} with τ=0.6, α=0.5

Combined with existing data (τ=0.0=FPS, τ=1.0=DINO, K=30, K=100) → complete curves.

Usage:
    python benchmark_hyperparameter_sweep.py
    python benchmark_hyperparameter_sweep.py --classes bagel,cookie  # quick test
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from backbones.pointnet2_seg import PointNet2SegBackbone
from benchmark_pas_sampling_methods import (
    SamplingComparisonPipeline, compute_iou, compute_pro, find_best_threshold
)

SWEEP_CONFIGS = [
    # τ sweep
    {'name': 'τ=0.2', 'tau': 0.2, 'density_alpha': 0.5, 'max_sample': 100, 'sweep': 'tau'},
    {'name': 'τ=0.4', 'tau': 0.4, 'density_alpha': 0.5, 'max_sample': 100, 'sweep': 'tau'},
    {'name': 'τ=0.6', 'tau': 0.6, 'density_alpha': 0.5, 'max_sample': 100, 'sweep': 'tau'},
    {'name': 'τ=0.8', 'tau': 0.8, 'density_alpha': 0.5, 'max_sample': 100, 'sweep': 'tau'},
    # K sweep
    {'name': 'K=50',  'tau': 0.6, 'density_alpha': 0.5, 'max_sample': 50,  'sweep': 'K'},
    {'name': 'K=200', 'tau': 0.6, 'density_alpha': 0.5, 'max_sample': 200, 'sweep': 'K'},
]


class Args:
    img_size = 224
    dataset_path = 'datasets/mvtec3d'
    max_sample = 100
    density_alpha = 0.5
    tau = 0.6
    amp = True


def run_config(config, classes, dino, device, args_base):
    args = copy.deepcopy(args_base)

    from dataset import get_data_loader

    config_name = config['name']
    class_results = {}

    for class_name in classes:
        print(f"\n  [{config_name}] {class_name}")

        backbone = PointNet2SegBackbone(tau=config['tau']).to(device)
        pipeline = SamplingComparisonPipeline(args, dino_extractor=dino)
        pipeline.backbone = backbone
        pipeline.set_strategy('pas')
        pipeline.density_alpha = config['density_alpha']
        pipeline._max_samples = config['max_sample']

        train_loader = get_data_loader('train', class_name, 224, args)
        test_loader = get_data_loader('test', class_name, 224, args)

        pipeline.build_memory(class_name, train_loader)

        pipeline.point_scores_list = []
        pipeline.point_labels_list = []
        pipeline.score_maps = []
        pipeline.mask_maps = []
        pipeline.image_preds = []
        pipeline.image_labels = []

        for sample, mask_sample, label, _ in tqdm(test_loader, desc=f"  Testing"):
            pipeline.predict(sample, mask_sample, label)

        point_scores = np.concatenate(pipeline.point_scores_list)
        point_labels = np.concatenate(pipeline.point_labels_list)

        if point_labels.sum() > 0 and point_labels.sum() < len(point_labels):
            point_auc = roc_auc_score(point_labels, point_scores)
        else:
            point_auc = 0.5

        pro_values = []
        for smap, mmap in zip(pipeline.score_maps, pipeline.mask_maps):
            if mmap.sum() > 0:
                pro_values.append(compute_pro(mmap, smap))
        pro = float(np.mean(pro_values)) if pro_values else 0.0

        iou_values = []
        for smap, mmap in zip(pipeline.score_maps, pipeline.mask_maps):
            if mmap.sum() > 0:
                iou_values.append(find_best_threshold(smap, mmap.astype(bool)))
        iou = float(np.mean(iou_values)) if iou_values else 0.0

        image_labels_np = np.array([l.item() if hasattr(l, 'item') else l
                                     for l in pipeline.image_labels])
        image_preds_np = np.array(pipeline.image_preds)
        image_auc = roc_auc_score(image_labels_np, image_preds_np) if len(np.unique(image_labels_np)) > 1 else 0.5

        class_results[class_name] = {
            'point_auc': float(point_auc), 'pro': pro,
            'iou': iou, 'image_auc': float(image_auc),
        }
        print(f"    PtAUC={point_auc:.4f}  PRO={pro:.4f}  IoU={iou:.4f}  ImgAUC={image_auc:.4f}")
        torch.cuda.empty_cache()

    mean_ptauc = float(np.mean([class_results[c]['point_auc'] for c in classes]))
    print(f"\n  [{config_name}] Mean PtAUC = {mean_ptauc:.4f}")
    return {'per_class': class_results, 'mean_ptauc': mean_ptauc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default='all')
    parser.add_argument('--output_json', default='results/hyperparameter_sweep.json')
    args_p = parser.parse_args()

    if args_p.classes == 'all':
        classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    else:
        classes = args_p.classes.split(',')

    from models.models import Model

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seeds(42)

    print("=" * 80)
    print("Hyperparameter Sensitivity Sweep (τ + K)")
    print(f"Classes: {len(classes)}, Configs: {len(SWEEP_CONFIGS)}")
    print("=" * 80)

    print("\nLoading DINO backbone...")
    dino = Model(
        device=device, rgb_backbone_name='vit_base_patch14_dinov2',
        xyz_backbone_name='Point_MAE', group_size=128, num_group=1024,
    )
    dino.to(device).eval()
    print("DINO loaded.")

    args_base = Args()

    all_results = {}
    if os.path.exists(args_p.output_json):
        with open(args_p.output_json) as f:
            all_results = json.load(f)
        print(f"Resumed checkpoint: {list(all_results.keys())}")

    for config in SWEEP_CONFIGS:
        if config['name'] in all_results:
            print(f"\n  [{config['name']}] already done, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Config: {config['name']} (τ={config['tau']}, α={config['density_alpha']}, K={config['max_sample']})")
        print(f"{'=' * 80}")

        result = run_config(config, classes, dino, device, args_base)
        all_results[config['name']] = result

        os.makedirs('results', exist_ok=True)
        with open(args_p.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Checkpoint saved to {args_p.output_json}")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("  Sensitivity Sweep — Summary")
    print(f"{'=' * 80}")

    for sweep_type in ['tau', 'K']:
        configs = [c for c in SWEEP_CONFIGS if c['sweep'] == sweep_type]
        print(f"\n  {sweep_type} sweep:")
        for c in configs:
            if c['name'] in all_results:
                print(f"    {c['name']}: {all_results[c['name']]['mean_ptauc']:.4f}")
        if sweep_type == 'tau':
            print(f"    τ=0.0 (FPS, from Exp I): 0.7856")
            print(f"    τ=1.0 (DINO, from Exp I): 0.7111")
        else:
            print(f"    K=30 (from Exp G): ~0.79")
            print(f"    K=100 (Full PAS, from Exp I): 0.8185")

    print(f"\n  Results saved to {args_p.output_json}")


if __name__ == '__main__':
    main()
