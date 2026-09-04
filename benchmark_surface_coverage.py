"""Surface Coverage Experiment.

Computes coverage, uniformity, edge_preservation for all 8 samplers
on PN2 backbone using the sampling_metrics.py functions.
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
from utils.sampling_metrics import compute_sampling_metrics
from backbones.pas_sampler import PASSampler, _native_fps, compute_anomaly_scores
from backbones.curvature_sampler import curvature_guided_sampling
from backbones.random_sampler import random_sampling
from backbones.voxel_sampler import voxel_downsampling
from backbones.gss_sampler import grid_sampling
from backbones.density_fps import density_aware_fps
from backbones.pointsp_ffps import filtered_fps

import warnings
warnings.filterwarnings('ignore')


def sample_points(xyz, strategy, npoint, anomaly_scores=None, tau=0.6):
    """Sample points using specified strategy. Returns indices."""
    xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0)
    N = xyz.shape[0]
    K = min(npoint, N)

    if strategy == 'fps':
        idx = _native_fps(xyz_tensor, K).squeeze(0).numpy()
    elif strategy == 'pas':
        scores_t = torch.from_numpy(anomaly_scores).float().unsqueeze(0)
        sampler = PASSampler(tau=tau)
        idx = sampler(xyz_tensor, K, scores_t).squeeze(0).numpy()
    elif strategy == 'rs':
        idx = random_sampling(xyz_tensor, K).squeeze(0).numpy()
    elif strategy == 'curv':
        idx = curvature_guided_sampling(xyz_tensor, K, k=32, tau=0.5).squeeze(0).numpy()
    elif strategy == 'vds':
        idx = voxel_downsampling(xyz_tensor, K).squeeze(0).numpy()
    elif strategy == 'grid':
        idx = grid_sampling(xyz_tensor, K).squeeze(0).numpy()
    elif strategy == 'dafps':
        idx = density_aware_fps(xyz_tensor, K, voxel_size=0.05, alpha=0.5).squeeze(0).numpy()
    elif strategy == 'ffps':
        idx = filtered_fps(xyz_tensor, K, k=20, omega=0.95).squeeze(0).numpy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.clip(idx, 0, N - 1).astype(int)


def main():
    parser = argparse.ArgumentParser(description='Surface Coverage Experiment')
    parser.add_argument('--strategies', default='rs,fps,ffps,pas,vds,grid,dafps,curv')
    parser.add_argument('--classes', default='all')
    parser.add_argument('--npoint', type=int, default=512)
    parser.add_argument('--output_json', default='results/surface_coverage.json')
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

    # Load DINO for PAS anomaly scores
    from models.models import Model
    dino = Model(device=device, rgb_backbone_name='vit_base_patch14_dinov2',
                 xyz_backbone_name='Point_MAE', group_size=128, num_group=1024)
    dino.to(device).eval()
    print("DINO loaded.")

    all_results = {}

    for cls in classes:
        print(f"\n{'='*60}")
        print(f"Class: {cls}")
        print(f"{'='*60}")

        from dataset import get_data_loader
        test_loader = get_data_loader('test', cls, 224, type('Args', (), {
            'dataset_path': 'datasets/mvtec3d',
            'device': device,
        })())

        # Load test data
        test_data = []
        for batch_idx, (data, gt, label, path) in enumerate(
                tqdm(test_loader, desc="    Loading", leave=False)):
            rgb, xyz_org, depth = data
            xyz_np = xyz_org[0].permute(1, 2, 0).numpy()
            xyz_flat = xyz_np.reshape(-1, 3)
            valid = np.abs(xyz_flat).sum(axis=1) > 1e-6
            xyz_valid = xyz_flat[valid].astype(np.float32)

            # DINO scores for PAS
            rgb_dev = rgb.to(device)
            with torch.no_grad():
                feat = dino.forward_rgb_features(rgb_dev)
                B, C, H, W = feat.shape
                feat_flat = feat.view(B, C, -1).permute(0, 2, 1)
                mean_feat = feat_flat.mean(dim=1, keepdim=True)
                cos_sim = torch.nn.functional.cosine_similarity(feat_flat, mean_feat, dim=-1)
                scores_2d = (1.0 - cos_sim).view(H, W).cpu().numpy()

            # Map 2D scores to 3D points
            scores_flat = scores_2d.reshape(-1)
            if len(scores_flat) == len(xyz_flat):
                scores_valid = scores_flat[valid]
            else:
                scores_valid = np.zeros(len(xyz_valid))

            test_data.append({
                'xyz': xyz_valid,
                'scores': scores_valid,
                'gt': gt[0, 0].numpy(),
            })

        # Compute metrics for each strategy
        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")

            coverage_vals = []
            uniformity_vals = []
            edge_vals = []

            for sample in tqdm(test_data, desc=f"    {strategy}", leave=False):
                xyz = sample['xyz']
                scores = sample['scores']

                if len(xyz) < args_p.npoint:
                    continue

                # Sample
                idx = sample_points(xyz, strategy, args_p.npoint,
                                     anomaly_scores=scores, tau=0.6)
                sampled_xyz = xyz[idx]

                # Compute metrics
                metrics = compute_sampling_metrics(sampled_xyz, xyz)
                coverage_vals.append(metrics['coverage'])
                uniformity_vals.append(metrics['uniformity'])
                edge_vals.append(metrics['edge_preservation'])

            key = f"{cls}/{strategy}"
            all_results[key] = {
                'mean_coverage': float(np.mean(coverage_vals)) if coverage_vals else 0,
                'mean_uniformity': float(np.mean(uniformity_vals)) if uniformity_vals else 0,
                'mean_edge_preservation': float(np.mean(edge_vals)) if edge_vals else 0,
                'n_samples': len(coverage_vals),
            }
            print(f"    Coverage={all_results[key]['mean_coverage']:.4f}, "
                  f"Uniformity={all_results[key]['mean_uniformity']:.4f}, "
                  f"Edge={all_results[key]['mean_edge_preservation']:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Surface Coverage")
    print("=" * 80)

    summary = {}
    for strategy in strategies:
        cov, uni, edge = [], [], []
        for cls in classes:
            key = f"{cls}/{strategy}"
            if key in all_results:
                cov.append(all_results[key]['mean_coverage'])
                uni.append(all_results[key]['mean_uniformity'])
                edge.append(all_results[key]['mean_edge_preservation'])
        if cov:
            summary[strategy] = {
                'mean_coverage': float(np.mean(cov)),
                'mean_uniformity': float(np.mean(uni)),
                'mean_edge_preservation': float(np.mean(edge)),
            }

    print(f"\n{'Strategy':<10} {'Coverage':>10} {'Uniformity':>12} {'EdgePres':>10}")
    print("-" * 45)
    for strategy in strategies:
        if strategy in summary:
            s = summary[strategy]
            print(f"{strategy:<10} {s['mean_coverage']:>10.4f} {s['mean_uniformity']:>12.4f} {s['mean_edge_preservation']:>10.4f}")

    # Save
    output = {
        '_metadata': {
            'description': 'Surface coverage experiment',
            'strategies': strategies,
            'classes': classes,
            'npoint': args_p.npoint,
        },
        'per_class': all_results,
        'summary': summary,
    }
    os.makedirs(os.path.dirname(args_p.output_json), exist_ok=True)
    with open(args_p.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args_p.output_json}")


if __name__ == '__main__':
    main()
