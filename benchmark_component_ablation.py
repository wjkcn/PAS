#!/usr/bin/env python
"""Component ablation for PAS — runs all configs needed for Table 2.

Configs (all use PN2 backbone, max_sample=100):
  Config  | Description                  | tau | density_alpha | strategy
  --------|------------------------------|-----|---------------|---------
  RS      | Random Sampling              | —   | 0.0           | rs
  FPS     | Pure FPS (no DINO at all)    | —   | 0.0           | fps
  DINO    | DINO score only (no hybrid)  | 1.0 | 0.0           | pas
  DINO+D  | DINO + Density weighting     | 1.0 | 0.5           | pas
  DINO+H  | DINO + Hybrid tau            | 0.6 | 0.0           | pas
  FullPAS | DINO + Hybrid + Density      | 0.6 | 0.5           | pas

Usage:
    python benchmark_component_ablation.py
    python benchmark_component_ablation.py --classes bagel,cookie  # quick test
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from backbones.pointnet2_seg import PointNet2SegBackbone
from backbones.pas_sampler import compute_anomaly_scores
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
from benchmark_pas_sampling_methods import (
    SamplingComparisonPipeline, compute_iou, compute_pro, find_best_threshold
)


# ── Configs ──────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {'name': 'RS',       'strategy': 'rs',  'tau': 0.6, 'density_alpha': 0.0},
    {'name': 'FPS',      'strategy': 'fps', 'tau': 0.6, 'density_alpha': 0.0},
    {'name': 'DINO',     'strategy': 'pas', 'tau': 1.0, 'density_alpha': 0.0},
    {'name': 'DINO+D',   'strategy': 'pas', 'tau': 1.0, 'density_alpha': 0.5},
    {'name': 'DINO+H',   'strategy': 'pas', 'tau': 0.6, 'density_alpha': 0.0},
    {'name': 'Full PAS', 'strategy': 'pas', 'tau': 0.6, 'density_alpha': 0.5},
]


class Args:
    img_size = 224
    dataset_path = 'datasets/mvtec3d'
    max_sample = 100
    density_alpha = 0.5
    tau = 0.6
    amp = True


def run_config(config, classes, dino, device, args_base):
    """Run one ablation config across all classes and return per-class + mean results."""
    args = copy.deepcopy(args_base)
    args.tau = config['tau']
    args.density_alpha = config['density_alpha']

    from dataset import get_data_loader

    strategy = config['strategy']
    config_name = config['name']

    class_results = {}

    for class_name in classes:
        print(f"\n  [{config_name}] {class_name}")

        # Build fresh pipeline with correct tau/density_alpha
        backbone = PointNet2SegBackbone(tau=args.tau).to(device)
        pipeline = SamplingComparisonPipeline(args, dino_extractor=dino)
        pipeline.backbone = backbone
        pipeline.set_strategy(strategy)
        pipeline.density_alpha = args.density_alpha
        pipeline._max_samples = args.max_sample

        # Monkey-patch _extract_features for 'rs' strategy
        orig_extract = pipeline._extract_features

        def make_extract(p_self, orig):
            def _extract(_self, xyz, anomaly_scores):
                if p_self._strategy in ('rs', 'vds', 'grid', 'dafps', 'curv'):
                    feat, _ = p_self._extract_features_alt_sampling(xyz, anomaly_scores)
                    return feat
                else:
                    guide = anomaly_scores if p_self._strategy == 'pas' else None
                    feat, _, _, _ = p_self.backbone(xyz, guide)
                    return feat
            return _extract
        pipeline._extract_features = make_extract(pipeline, orig_extract).__get__(pipeline)

        # Patch _extract_features_alt_sampling to handle 'rs'
        from benchmark_pas_sampling_extended import _extract_features_alt_sampling_extended
        SamplingComparisonPipeline._extract_features_alt_sampling = \
            _extract_features_alt_sampling_extended

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

        # Compute metrics
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

        if len(np.unique(image_labels_np)) > 1:
            image_auc = roc_auc_score(image_labels_np, image_preds_np)
        else:
            image_auc = 0.5

        class_results[class_name] = {
            'point_auc': float(point_auc), 'pro': pro,
            'iou': iou, 'image_auc': float(image_auc),
        }

        print(f"    PtAUC={point_auc:.4f}  PRO={pro:.4f}  IoU={iou:.4f}  ImgAUC={image_auc:.4f}")
        torch.cuda.empty_cache()

    # Mean
    mean_ptauc = float(np.mean([class_results[c]['point_auc'] for c in classes]))
    print(f"\n  [{config_name}] Mean PtAUC = {mean_ptauc:.4f}")

    return {'per_class': class_results, 'mean_ptauc': mean_ptauc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default='all',
                        help='Classes to evaluate (comma-separated or "all")')
    parser.add_argument('--output_json', default='results/component_ablation.json')
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
    print("PAS Component Ablation Study")
    print(f"Classes: {len(classes)}, Configs: {len(ABLATION_CONFIGS)}")
    print("=" * 80)

    print("\nLoading DINO backbone...")
    dino = Model(
        device=device, rgb_backbone_name='vit_base_patch14_dinov2',
        xyz_backbone_name='Point_MAE', group_size=128, num_group=1024,
    )
    dino.to(device).eval()
    print("DINO loaded.")

    args_base = Args()

    # Resume from checkpoint if available
    all_results = {}
    if os.path.exists(args_p.output_json):
        with open(args_p.output_json) as f:
            all_results = json.load(f)
        print(f"Resumed checkpoint: {list(all_results.keys())}")

    for config in ABLATION_CONFIGS:
        if config['name'] in all_results:
            print(f"\n  [{config['name']}] already done (mean PtAUC={all_results[config['name']].get('mean_ptauc', '?'):.4f}), skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Config: {config['name']} (strategy={config['strategy']}, tau={config['tau']}, α={config['density_alpha']})")
        print(f"{'=' * 80}")

        result = run_config(config, classes, dino, device, args_base)
        all_results[config['name']] = result

        # Save checkpoint after each config
        os.makedirs('results', exist_ok=True)
        with open(args_p.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Checkpoint saved to {args_p.output_json}")

    # ── Final Summary ──
    print(f"\n{'=' * 80}")
    print("  Component Ablation — Summary")
    print(f"{'=' * 80}")

    config_names = [c['name'] for c in ABLATION_CONFIGS]
    header = f"  {'Class':<14}"
    for name in config_names:
        header += f" {name:>10}"
    print(header)
    print(f"  {'-' * (14 + 11 * len(config_names))}")

    for class_name in classes:
        row = f"  {class_name:<14}"
        for name in config_names:
            row += f" {all_results[name]['per_class'][class_name]['point_auc']:>10.4f}"
        print(row)

    row = f"  {'MEAN':<14}"
    for name in config_names:
        row += f" {all_results[name]['mean_ptauc']:>10.4f}"
    print(f"  {'-' * (14 + 11 * len(config_names))}")
    print(row)

    # Component contributions
    print(f"\n  Component contributions (Δ from previous):")
    prev = None
    for i, name in enumerate(config_names):
        m = all_results[name]['mean_ptauc']
        if prev is not None:
            delta = m - all_results[config_names[prev]]['mean_ptauc']
            print(f"    +{name:<8}: {m:.4f} (Δ={delta:+.4f})")
        else:
            print(f"    {name:<10}: {m:.4f} (baseline)")
        prev = i

    print(f"\n  Results saved to {args_p.output_json}")


if __name__ == '__main__':
    main()
