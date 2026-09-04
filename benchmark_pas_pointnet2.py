"""Benchmark: PointNet++ backbone + MVTec-3D, FPS vs PAS comparison.

Tests PAS as a truly plug-and-play sampling module on a non-M3DM backbone.
PointNet++ has no interpolating_points bottleneck — PAS effects are directly measurable.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Heavy impopas deferred to run_benchmark() to allow --help without CUDA/deps
from utils.utils import set_seeds


def get_args():
    parser = argparse.ArgumentParser(description='PointNet++ + PAS Benchmark')
    parser.add_argument('--dataset_path', default='datasets/mvtec3d', type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--max_sample', default=400, type=int)
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch14_dinov2', type=str)
    parser.add_argument('--group_size', default=128, type=int)
    parser.add_argument('--num_group', default=1024, type=int)
    parser.add_argument('--f_coreset', default=0.1, type=float)
    parser.add_argument('--tau', default=0.6, type=float,
                        help='PAS anomaly fraction (0.6 = 60% anomaly-guided)')
    return parser.parse_args()


def run_benchmark(args):
    from dataset import get_data_loader, mvtec3d_classes
    from backbones.pointnet2_ad import PointNet2ADPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seeds(0)

    # DINO extractor for anomaly score computation (lazy import)
    print("Loading DINO backbone for anomaly score computation...")
    from models.models import Model
    dino_model = Model(
        device=device,
        rgb_backbone_name=args.rgb_backbone_name,
        xyz_backbone_name='Point_MAE',
        group_size=args.group_size,
        num_group=args.num_group,
    )
    dino_model.to(device)
    dino_model.eval()

    classes = mvtec3d_classes()
    results_fps, results_pas = {}, {}

    for cls_name in classes:
        print(f"\n{'='*60}")
        print(f"  Class: {cls_name}")
        print(f"{'='*60}")

        train_loader = get_data_loader('train', cls_name, args.img_size, args)
        test_loader = get_data_loader('test', cls_name, args.img_size, args)

        # --- FPS (no anomaly guidance) ---
        print(f"  [FPS] Pure geometric sampling...")
        pipe_fps = PointNet2ADPipeline(args, dino_extractor=dino_model)
        pipe_fps.set_sampling_mode('fps')
        pipe_fps.fit(cls_name, train_loader)
        auc_fps = pipe_fps.evaluate(cls_name, test_loader)
        results_fps[cls_name] = auc_fps

        # --- PAS (anomaly-guided hybrid sampling) ---
        print(f"  [PAS] Anomaly-guided hybrid sampling...")
        pipe_pas = PointNet2ADPipeline(args, dino_extractor=dino_model)
        pipe_pas.set_sampling_mode('pas')
        pipe_pas.fit(cls_name, train_loader)
        auc_pas = pipe_pas.evaluate(cls_name, test_loader)
        results_pas[cls_name] = auc_pas

        delta = auc_pas - auc_fps
        print(f"  → {cls_name}: FPS={auc_fps:.4f}  PAS={auc_pas:.4f}  Δ={delta:+.4f}")

        del pipe_fps, pipe_pas
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  PointNet++ + MVTec-3D: FPS vs PAS")
    print(f"{'='*60}")

    rows = []
    for cls_name in classes:
        rows.append({
            'Class': cls_name,
            'FPS': round(results_fps[cls_name], 4),
            'PAS': round(results_pas[cls_name], 4),
            'Delta': round(results_pas[cls_name] - results_fps[cls_name], 4),
        })

    fps_mean = np.mean(list(results_fps.values()))
    pas_mean = np.mean(list(results_pas.values()))
    rows.append({
        'Class': 'MEAN',
        'FPS': round(fps_mean, 4),
        'PAS': round(pas_mean, 4),
        'Delta': round(pas_mean - fps_mean, 4),
    })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))

    # Win count
    wins = sum(1 for c in classes if results_pas[c] > results_fps[c])
    print(f"\nPAS Wins: {wins}/{len(classes)}")

    return df


if __name__ == '__main__':
    args = get_args()
    run_benchmark(args)
