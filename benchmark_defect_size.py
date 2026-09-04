"""Defect Size Grouping Experiment.

Uses the existing SamplingComparisonPipeline, patches predict() to save
per-sample PtAUC and GT defect ratio, then groups by defect size.

Hypothesis: PAS improvement is largest on small defects.
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from benchmark_pas_sampling_methods import SamplingComparisonPipeline, compute_pro, find_best_threshold

import warnings
warnings.filterwarnings('ignore')


def compute_gt_defect_ratio(gt_mask):
    """Compute ratio of defect pixels to total pixels."""
    if gt_mask is None or gt_mask.sum() == 0:
        return 0.0
    return float(gt_mask.sum()) / gt_mask.size


def classify_defect_size(ratio, thresholds=(0.005, 0.05)):
    """Classify defect into small/medium/large based on area ratio."""
    if ratio < thresholds[0]:
        return 'small'
    elif ratio < thresholds[1]:
        return 'medium'
    else:
        return 'large'


def main():
    parser = argparse.ArgumentParser(description='Defect Size Grouping Experiment')
    parser.add_argument('--strategies', default='fps,pas')
    parser.add_argument('--classes', default='all')
    parser.add_argument('--output_json', default='results/defect_size_grouping.json')
    parser.add_argument('--device', default='cuda:0')
    args_p = parser.parse_args()

    strategies = args_p.strategies.split(',')
    if args_p.classes == 'all':
        classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    else:
        classes = args_p.classes.split(',')

    device = torch.device(args_p.device if torch.cuda.is_available() else 'cpu')
    set_seeds(42)

    # Load DINO
    from models.models import Model
    dino = Model(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                 xyz_backbone_name='Point_MAE', group_size=128, num_group=1024)
    dino.to(device).eval()
    print("DINO loaded.")

    # Patch pipeline for extended sampling support
    from benchmark_pas_sampling_extended import _extract_features_alt_sampling_extended
    SamplingComparisonPipeline._extract_features_alt_sampling = _extract_features_alt_sampling_extended

    # Args for pipeline
    class Args:
        img_size = 224
        dataset_path = 'datasets/mvtec3d'
        max_sample = 100
        density_alpha = 0.5
        tau = 0.6
        amp = True
    args = Args()
    args.device = device

    all_results = {}

    for cls in classes:
        print(f"\n{'='*60}")
        print(f"Class: {cls}")
        print(f"{'='*60}")

        from dataset import get_data_loader
        train_loader = get_data_loader('train', cls, args.img_size, args)
        test_loader = get_data_loader('test', cls, args.img_size, args)

        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")

            pipeline = SamplingComparisonPipeline(args, dino_extractor=dino)
            pipeline.build_memory(cls, train_loader)
            pipeline.backbone.set_sampling_mode(strategy)

            # Run predict on all test samples, collect per-sample results
            sample_results = []
            for sample, mask, label, _ in tqdm(test_loader, desc=f"    Test [{strategy}]", leave=False):
                # Skip normal samples
                if label.item() == 0:
                    continue

                # Get GT mask
                gt_mask = mask.squeeze().cpu().numpy()  # [H, W]
                if gt_mask.sum() == 0:
                    continue

                # Run predict
                pipeline.reset_metrics()
                pipeline.predict(sample, mask, label)

                # Get per-sample score map and compute PtAUC
                score_map = pipeline.score_maps[0]  # [H, W]
                gt_flat = gt_mask.reshape(-1).astype(int)
                score_flat = score_map.reshape(-1)

                if gt_flat.sum() > 0 and gt_flat.sum() < len(gt_flat):
                    point_auc = float(roc_auc_score(gt_flat, score_flat))
                else:
                    continue

                defect_ratio = compute_gt_defect_ratio(gt_mask)
                defect_group = classify_defect_size(defect_ratio)

                sample_results.append({
                    'point_auc': point_auc,
                    'defect_ratio': defect_ratio,
                    'defect_group': defect_group,
                })

            # Group by defect size
            groups = {'small': [], 'medium': [], 'large': []}
            for r in sample_results:
                groups[r['defect_group']].append(r)

            # Compute per-group metrics
            group_metrics = {}
            for group_name, group_data in groups.items():
                if len(group_data) == 0:
                    continue
                pt_aucs = [r['point_auc'] for r in group_data]
                ratios = [r['defect_ratio'] for r in group_data]
                group_metrics[group_name] = {
                    'n_samples': len(group_data),
                    'mean_ptauc': float(np.mean(pt_aucs)),
                    'std_ptauc': float(np.std(pt_aucs)),
                    'mean_defect_ratio': float(np.mean(ratios)),
                    'min_defect_ratio': float(np.min(ratios)),
                    'max_defect_ratio': float(np.max(ratios)),
                }

            key = f"{cls}/{strategy}"
            all_results[key] = {
                'n_total': len(sample_results),
                'groups': group_metrics,
            }

            # Print summary
            for g in ['small', 'medium', 'large']:
                if g in group_metrics:
                    m = group_metrics[g]
                    print(f"    {g:>6}: n={m['n_samples']:>3}, "
                          f"PtAUC={m['mean_ptauc']:.4f}±{m['std_ptauc']:.4f}, "
                          f"ratio=[{m['min_defect_ratio']:.4f}, {m['max_defect_ratio']:.4f}]")

    # ── Summary: PAS vs FPS per defect size ──
    print("\n" + "=" * 80)
    print("SUMMARY: PAS (PAS) vs FPS by Defect Size")
    print("=" * 80)

    summary = {}
    for group in ['small', 'medium', 'large']:
        fps_aucs, pas_aucs = [], []
        for cls in classes:
            fps_key = f"{cls}/fps"
            pas_key = f"{cls}/pas"
            if fps_key in all_results and pas_key in all_results:
                fps_g = all_results[fps_key]['groups'].get(group, {})
                pas_g = all_results[pas_key]['groups'].get(group, {})
                if fps_g.get('n_samples', 0) > 0 and pas_g.get('n_samples', 0) > 0:
                    fps_aucs.append(fps_g['mean_ptauc'])
                    pas_aucs.append(pas_g['mean_ptauc'])

        if fps_aucs:
            fps_mean = float(np.mean(fps_aucs))
            pas_mean = float(np.mean(pas_aucs))
            delta = pas_mean - fps_mean
            summary[group] = {
                'fps_mean': fps_mean,
                'pas_mean': pas_mean,
                'delta': delta,
                'n_classes': len(fps_aucs),
            }
            print(f"\n  {group:>6}: FPS={fps_mean:.4f}, PAS={pas_mean:.4f}, "
                  f"Δ={delta:+.4f} ({len(fps_aucs)} classes)")

    # Save
    output = {
        '_metadata': {
            'description': 'Defect size grouping experiment',
            'hypothesis': 'PAS improvement is largest on small defects',
            'strategies': strategies,
            'classes': classes,
            'thresholds': {'small': '<0.5%', 'medium': '0.5-5%', 'large': '>5%'},
        },
        'per_class': all_results,
        'summary': summary,
    }
    os.makedirs(os.path.dirname(args_p.output_json), exist_ok=True)
    with open(args_p.output_json, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args_p.output_json}")


if __name__ == '__main__':
    main()
