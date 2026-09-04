"""Benchmark: Dual-backbone + Dual-dataset, FPS vs PAS comparison.

Backbones: PointNet++ SSG
Datasets:  MVTec-3D + Eyecandies
Metrics:   Image-level AUC

This is the core Route A experiment table.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Heavy impopas deferred to run functions (avoid --help triggering CUDA/deps)
from utils.utils import set_seeds


def get_args():
    parser = argparse.ArgumentParser(description='Dual-backbone/Dual-dataset Benchmark')
    parser.add_argument('--dataset_base', default='datasets', type=str,
                        help='Parent directory containing mvtec3d/ and eyecandies/')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--max_sample', default=400, type=int)
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch14_dinov2', type=str)
    parser.add_argument('--group_size', default=128, type=int)
    parser.add_argument('--num_group', default=1024, type=int)
    parser.add_argument('--tau', default=0.6, type=float)
    return parser.parse_args()


def run_pn2(args, dataset_type, dino_model):
    """Run PointNet++ FPS vs PAS on one dataset."""
    from dataset import get_data_loader, mvtec3d_classes, eyecandies_classes
    from backbones.pointnet2_ad import PointNet2ADPipeline
    classes = {
        'mvtec3d': mvtec3d_classes,
        'eyecandies': eyecandies_classes,
    }[dataset_type]()
    fps_res, pas_res = {}, {}

    for cls_name in classes:
        print(f"\n  [{dataset_type}] {cls_name}")

        loader_args = argparse.Namespace(
            dataset_path=os.path.join(args.dataset_base, dataset_type),
            img_size=args.img_size,
            max_sample=args.max_sample,
        )

        train_loader = get_data_loader('train', cls_name, args.img_size, loader_args)
        test_loader = get_data_loader('test', cls_name, args.img_size, loader_args)

        for mode, label, results in [('fps', 'FPS', fps_res), ('pas', 'PAS', pas_res)]:
            pipe = PointNet2ADPipeline(args, dino_extractor=dino_model)
            pipe.set_sampling_mode(mode)
            pipe.fit(cls_name, train_loader)
            results[cls_name] = pipe.evaluate(cls_name, test_loader)
            del pipe
            torch.cuda.empty_cache()

        delta = pas_res[cls_name] - fps_res[cls_name]
        print(f"    FPS={fps_res[cls_name]:.4f}  PAS={pas_res[cls_name]:.4f}  Δ={delta:+.4f}")

    return fps_res, pas_res


def result_table(fps, pas, classes, label_fps, label_pas):
    import pandas as pd
    rows = []
    for c in classes:
        rows.append({
            'Class': c, label_fps: round(fps[c], 4),
            label_pas: round(pas[c], 4), 'Delta': round(pas[c] - fps[c], 4),
        })
    rows.append({
        'Class': 'MEAN', label_fps: round(np.mean(list(fps.values())), 4),
        label_pas: round(np.mean(list(pas.values())), 4),
        'Delta': round(np.mean(list(pas.values())) - np.mean(list(fps.values())), 4),
    })
    return pd.DataFrame(rows)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(0)
    print(f"Device: {device}")

    print("Loading DINO backbone...")
    from models.models import Model
    dino_model = Model(
        device=device, rgb_backbone_name=args.rgb_backbone_name,
        xyz_backbone_name='Point_MAE', group_size=args.group_size,
        num_group=args.num_group,
    )
    dino_model.to(device)
    dino_model.eval()

    all_fps, all_pas = {}, {}

    for ds in ['mvtec3d', 'eyecandies']:
        print(f"\n{'='*60}")
        print(f"  {ds} | PointNet++ | FPS vs PAS")
        print(f"{'='*60}")

        fps, pas = run_pn2(args, ds, dino_model)
        all_fps[ds], all_pas[ds] = fps, pas

        cls = mvtec3d_classes() if ds == 'mvtec3d' else eyecandies_classes()
        print(result_table(fps, pas, cls, f'FPS ({ds})', f'PAS ({ds})').to_markdown(index=False))
        print(f"PAS Wins: {sum(1 for c in cls if pas[c] > fps[c])}/{len(cls)}")

    print(f"\n{'='*60}")
    print("  Cross-Dataset Summary")
    print(f"{'='*60}")
    sr = []
    for ds, fps in all_fps.items():
        pas = all_pas[ds]
        cls = mvtec3d_classes() if ds == 'mvtec3d' else eyecandies_classes()
        sr.append({
            'Dataset': ds, 'Backbone': 'PointNet++',
            'FPS': round(np.mean(list(fps.values())), 4),
            'PAS': round(np.mean(list(pas.values())), 4),
            'Δ': round(np.mean(list(pas.values())) - np.mean(list(fps.values())), 4),
            'Wins': f"{sum(1 for c in cls if pas[c] > fps[c])}/{len(cls)}",
        })
    print(pd.DataFrame(sr).to_markdown(index=False))


if __name__ == '__main__':
    main()
