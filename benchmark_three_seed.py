"""3-seed variance: PN2 FPS/PAS × seeds {42, 123, 456} on all 10 MVTec-3D classes.

Repopas mean ± std for statistical significance in paper tables.
"""
import torch, numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seeds
from backbones.pointnet2_seg import PointNet2SegBackbone
from benchmark_cross_backbone import (
    CrossBackbonePipeline, load_dino_extractor, Args, compute_pro, find_best_threshold
)
from sklearn.metrics import roc_auc_score
from dataset import get_data_loader
from tqdm import tqdm

CLASSES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']
SEEDS = [42, 123, 456]
STRATEGIES = ['fps', 'pas']
OUTPUT = 'results/three_seed_variance.json'

def run_one(strategy, seed, classes, dino, device, args_base):
    set_seeds(seed)
    class_results = {}

    for cls in classes:
        backbone = PointNet2SegBackbone(tau=args_base.tau).to(device)
        pipeline = CrossBackbonePipeline(args_base, backbone, dino_extractor=dino)
        pipeline.set_strategy(strategy)

        train_loader = get_data_loader('train', cls, 224, args_base)
        test_loader = get_data_loader('test', cls, 224, args_base)
        pipeline.build_memory(cls, train_loader)

        pipeline.point_scores_list = []
        pipeline.point_labels_list = []
        pipeline.score_maps = []
        pipeline.mask_maps = []
        pipeline.image_preds = []
        pipeline.image_labels = []

        for sample, mask_sample, label, _ in tqdm(test_loader, desc=f"  seed={seed} {strategy} {cls}", leave=False):
            pipeline.predict(sample, mask_sample, label)

        point_scores = np.concatenate(pipeline.point_scores_list)
        point_labels = np.concatenate(pipeline.point_labels_list)
        point_auc = roc_auc_score(point_labels, point_scores) if point_labels.sum() > 0 and point_labels.sum() < len(point_labels) else 0.5

        pro_vals = [compute_pro(gt.reshape(224,224), sm.reshape(224,224))
                    for gt, sm in zip(pipeline.mask_maps, pipeline.score_maps)]
        iou_vals = [find_best_threshold(s, m.astype(bool))
                    for s, m in zip(pipeline.score_maps, pipeline.mask_maps)]

        image_labels_np = np.array([l.item() if hasattr(l,'item') else l for l in pipeline.image_labels])
        image_preds_np = np.array(pipeline.image_preds)
        image_auc = roc_auc_score(image_labels_np, image_preds_np) if len(np.unique(image_labels_np)) > 1 else 0.5

        class_results[cls] = {
            'point_auc': float(point_auc), 'pro': float(np.mean(pro_vals)),
            'iou': float(np.mean(iou_vals)), 'image_auc': float(image_auc),
        }
        del backbone, pipeline
        torch.cuda.empty_cache()

    return class_results


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading DINO...")
    dino = load_dino_extractor(device)
    print("DINO loaded.")

    args_base = Args()
    all_results = {}

    for seed in SEEDS:
        for strategy in STRATEGIES:
            key = f'{strategy}/seed{seed}'
            print(f"\n{'='*60}\n  {key}\n{'='*60}")
            result = run_one(strategy, seed, CLASSES, dino, device, args_base)
            mean_ptauc = float(np.mean([result[c]['point_auc'] for c in CLASSES]))
            all_results[key] = {'per_class': result, 'mean_ptauc': mean_ptauc}
            print(f"  Mean PtAUC = {mean_ptauc:.4f}")

    # Aggregate
    summary = {}
    for strategy in STRATEGIES:
        per_class = {}
        for cls in CLASSES:
            vals = [all_results[f'{strategy}/seed{s}']['per_class'][cls]['point_auc'] for s in SEEDS]
            per_class[cls] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                              'values': vals}
        mean_of_means = float(np.mean([per_class[c]['mean'] for c in CLASSES]))
        # Pooled std: sqrt(mean of variances + var of means)... use simple mean of per-class std
        mean_std = float(np.mean([per_class[c]['std'] for c in CLASSES]))
        summary[strategy] = {'per_class': per_class, 'mean_ptauc': mean_of_means, 'mean_per_class_std': mean_std}

    all_results['_summary'] = summary

    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final table
    print(f"\n{'='*60}")
    print("3-Seed Variance Summary")
    print(f"{'='*60}")
    for strategy in STRATEGIES:
        s = summary[strategy]
        print(f"\n{strategy.upper()}: {s['mean_ptauc']:.4f} ± {s['mean_per_class_std']:.4f} (per-class std)")
        for cls in CLASSES:
            pc = s['per_class'][cls]
            print(f"  {cls:<15}: {pc['mean']:.4f} ± {pc['std']:.4f}  [{', '.join(f'{v:.4f}' for v in pc['values'])}]")

    print(f"\nSaved to {OUTPUT}")

if __name__ == '__main__':
    main()
