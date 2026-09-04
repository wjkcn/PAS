"""Extended anomaly detection metrics beyond PtAUC.

P0 (must-have): ImgAUC (already computed), PRO (already computed)
P1 (strongly recommended): F1-max, AUPRC (imbalance-robust)
P2 (paper elevation): Defect Retention Ratio (mechanism), Neighborhood Density (topology)
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_f1_max(point_labels, point_scores):
    """F1-max: best F1 across all thresholds (point-level).

    Returns (best_f1, best_threshold, precision_at_best, recall_at_best).
    """
    if point_labels.sum() == 0 or point_labels.sum() == len(point_labels):
        return 0.0, 0.5, 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(point_labels, point_scores)
    # precision/recall are n+1 length; thresholds is n length
    # Compute F1 at each threshold
    f1_scores = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    best_prec = float(precision[best_idx])
    best_rec = float(recall[best_idx])
    return best_f1, best_thresh, best_prec, best_rec


def compute_auprc(point_labels, point_scores):
    """Area Under Precision-Recall Curve (point-level)."""
    if point_labels.sum() == 0 or point_labels.sum() == len(point_labels):
        return 0.5
    return float(average_precision_score(point_labels, point_scores))


def compute_defect_retention(gt_labels, selected_indices):
    """Fraction of GT defect points retained after sampling.

    Args:
        gt_labels: [N] binary ground truth (1 = defect)
        selected_indices: [K] indices of selected points (0-indexed into N)

    Returns:
        retention: selected_defect_points / total_defect_points
        baseline: expected retention if random sampling (defect_ratio)
    """
    N = len(gt_labels)
    total_defects = gt_labels.sum()
    if total_defects == 0:
        return 1.0, 0.0
    defect_ratio = total_defects / N
    selected = np.asarray(selected_indices).astype(int)
    selected = selected[selected < N]  # clamp
    retained_defects = gt_labels[selected].sum()
    retention = retained_defects / total_defects
    return float(retention), float(defect_ratio)


def compute_retention_ratio_batched(all_gt_labels, all_selected_indices):
    """Compute mean defect retention ratio across all test samples.

    Args:
        all_gt_labels: list of [N_i] arrays (foreground GT labels per sample)
        all_selected_indices: list of [K_i] arrays (SA1 selected indices per sample)

    Returns:
        mean_retention, mean_baseline, per_sample_retentions
    """
    retentions = []
    baselines = []
    for gt, sel in zip(all_gt_labels, all_selected_indices):
        r, b = compute_defect_retention(gt, sel)
        retentions.append(r)
        baselines.append(b)
    return float(np.mean(retentions)), float(np.mean(baselines)), retentions


def compute_f1_max_image(image_labels, image_preds):
    """F1-max at image level."""
    if len(np.unique(image_labels)) < 2:
        return 0.0, 0.5, 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(image_labels, image_preds)
    f1_scores = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)
    best_idx = np.argmax(f1_scores)
    return (float(f1_scores[best_idx]),
            float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0,
            float(precision[best_idx]),
            float(recall[best_idx]))


def compute_auprc_image(image_labels, image_preds):
    """AUPRC at image level."""
    if len(np.unique(image_labels)) < 2:
        return 0.5
    return float(average_precision_score(image_labels, image_preds))


def compute_neighborhood_stats(xyz, selected_indices, gt_labels, k=32, radius=0.1):
    """Statistics about the neighborhood around selected points.

    Args:
        xyz: [N, 3] point cloud
        selected_indices: [K] indices of SA1 centers
        gt_labels: [N] binary GT
        k: number of neighbors for ball query
        radius: ball query radius

    Returns:
        dict with mean_neighbor_count, defect_neighbor_ratio, etc.
    """
    import torch
    xyz_t = torch.as_tensor(xyz).float()
    sel_t = torch.as_tensor(selected_indices).long()
    n_sel = len(sel_t)

    # Compute pairwise distances from selected points to all points
    sel_xyz = xyz_t[sel_t]  # [K, 3]
    dist = torch.cdist(sel_xyz.unsqueeze(0), xyz_t.unsqueeze(0)).squeeze(0)  # [K, N]
    in_radius = (dist <= radius)  # [K, N]
    neighbor_count = in_radius.sum(dim=1).float()  # [K]

    gt_t = torch.as_tensor(gt_labels).float()
    defect_in_neighborhood = (in_radius.float() * gt_t.unsqueeze(0)).sum(dim=1)  # [K]
    neighbor_defect_ratio = defect_in_neighborhood / neighbor_count.clamp(min=1)

    return {
        'mean_neighbors': float(neighbor_count.mean()),
        'std_neighbors': float(neighbor_count.std()),
        'mean_defect_ratio': float(neighbor_defect_ratio.mean()),
        'neighbor_count_lt_5_pct': float((neighbor_count < 5).float().mean()),
    }


# ── Convenience aggregator ──

def compute_all_metrics(point_scores, point_labels, score_maps, mask_maps,
                        image_preds, image_labels,
                        retained_indices_list=None, gt_labels_list=None):
    """Compute full metric suite from collected raw data.

    Returns dict with all metrics.
    """
    results = {}

    # P0: PtAUC
    if point_labels.sum() > 0 and point_labels.sum() < len(point_labels):
        results['point_auc'] = float(roc_auc_score(point_labels, point_scores))
    else:
        results['point_auc'] = 0.5

    # P0: PRO (per connected component)
    pro_vals = []
    for sm, mm in zip(score_maps, mask_maps):
        if mm.sum() > 0:
            from benchmark_pas_sampling_methods import compute_pro
            pro_vals.append(compute_pro(mm, sm))
    results['pro'] = float(np.mean(pro_vals)) if pro_vals else 0.0

    # P0: IoU (best threshold)
    iou_vals = []
    for sm, mm in zip(score_maps, mask_maps):
        if mm.sum() > 0:
            from benchmark_pas_sampling_methods import find_best_threshold
            iou_vals.append(find_best_threshold(sm, mm.astype(bool)))
    results['iou'] = float(np.mean(iou_vals)) if iou_vals else 0.0

    # P0: ImgAUC
    if len(np.unique(image_labels)) > 1:
        results['image_auc'] = float(roc_auc_score(image_labels, image_preds))
    else:
        results['image_auc'] = 0.5

    # P1: F1-max (point-level)
    f1, thresh, prec, rec = compute_f1_max(point_labels, point_scores)
    results['f1_max_point'] = f1
    results['f1_threshold'] = thresh

    # P1: AUPRC (point-level)
    results['auprc_point'] = compute_auprc(point_labels, point_scores)

    # P1: F1-max (image-level)
    img_f1, img_thresh, _, _ = compute_f1_max_image(image_labels, image_preds)
    results['f1_max_image'] = img_f1

    # P1: AUPRC (image-level)
    results['auprc_image'] = compute_auprc_image(image_labels, image_preds)

    # P2: Defect Retention Ratio
    if retained_indices_list is not None and gt_labels_list is not None:
        mean_ret, mean_base, _ = compute_retention_ratio_batched(
            gt_labels_list, retained_indices_list)
        results['retention_ratio'] = mean_ret
        results['retention_baseline'] = mean_base
        results['retention_over_random'] = mean_ret - mean_base

    return results
