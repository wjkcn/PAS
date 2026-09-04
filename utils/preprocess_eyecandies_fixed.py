import os
from shutil import copyfile
import cv2
import numpy as np
import tifffile
import yaml
import imageio.v3 as iio
import math
import argparse

FOCAL_LENGTH = 711.11


def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]
    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg


def depth_to_pointcloud(depth_img, info_depth, pose_txt, focal_length):
    depth_mt = load_and_convert_depth(depth_img, info_depth)
    pose = np.loadtxt(pose_txt)
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )
    camera_proj = intrinsics_4x4 @ pose
    jj, ii = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    ones = np.ones(height * width, dtype=np.float64)
    inv_depth = 1.0 / depth_mt.ravel().astype(np.float64)
    camera_vectors = np.column_stack([ii.ravel().astype(np.float64), jj.ravel().astype(np.float64), ones, inv_depth])
    hom_3d_pts = np.linalg.inv(camera_proj) @ camera_vectors.T
    pcd = depth_mt.ravel()[:, np.newaxis] * hom_3d_pts.T
    return pcd[:, :3]


def remove_point_cloud_background(pc):
    dz = pc[256, 1] - pc[-256, 1]
    dy = pc[256, 2] - pc[-256, 2]
    norm = math.sqrt(dz ** 2 + dy ** 2)
    start_points = np.array([0, pc[-256, 1], pc[-256, 2]])
    cos_theta = dy / norm
    sin_theta = dz / norm
    rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])
    processed_pc = (rotation_matrix @ (pc - start_points).T).T
    cond = (processed_pc[:, 1] > -0.02) | (processed_pc[:, 2] > 1.8) | (processed_pc[:, 0] > 1) | (processed_pc[:, 0] < -1)
    processed_pc[cond] = -start_points
    processed_pc = (rotation_matrix.T @ processed_pc.T).T + start_points
    index = [0, 2, 1]
    processed_pc = processed_pc[:, index]
    return processed_pc * [0.1, -0.1, 0.1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Eyecandies dataset.')
    parser.add_argument('--dataset_path', default='datasets/eyecandies_raw/eyecandies', type=str,
                        help="Original Eyecandies dataset path (root with category dirs).")
    parser.add_argument('--target_dir', default='datasets/eyecandies_preprocessed', type=str,
                        help="Processed Eyecandies dataset path")
    parser.add_argument('--max_train', type=int, default=200,
                        help="Max training samples per class (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    categories_list = sorted([
        d for d in os.listdir(args.dataset_path)
        if os.path.isdir(os.path.join(args.dataset_path, d))
    ])
    print(f"Found {len(categories_list)} categories: {categories_list}")

    for category_dir in categories_list:
        category_root_path = os.path.join(args.dataset_path, category_dir)
        # FIXED: removed leading '/' from path components
        category_train_path = os.path.join(category_root_path, 'train', 'data')
        category_test_path = os.path.join(category_root_path, 'test_public', 'data')

        print(f"\nProcessing {category_dir}...")

        category_target_path = os.path.join(args.target_dir, category_dir)
        os.makedirs(category_target_path, exist_ok=True)

        # ── Training data ──
        os.makedirs(os.path.join(category_target_path, 'train'), exist_ok=True)
        category_target_train_good_path = os.path.join(category_target_path, 'train', 'good')
        category_target_train_good_rgb_path = os.path.join(category_target_train_good_path, 'rgb')
        category_target_train_good_xyz_path = os.path.join(category_target_train_good_path, 'xyz')
        os.makedirs(category_target_train_good_rgb_path, exist_ok=True)
        os.makedirs(category_target_train_good_xyz_path, exist_ok=True)

        train_depth_files = sorted([f for f in os.listdir(category_train_path) if f.endswith('_depth.png')])
        num_train_files = len(train_depth_files)
        num_train = min(num_train_files, args.max_train)
        print(f"  Train: {num_train_files} samples available, processing {num_train}")

        for idx, depth_file in enumerate(train_depth_files[:num_train]):
            prefix = depth_file.replace('_depth.png', '')
            pc = depth_to_pointcloud(
                os.path.join(category_train_path, prefix + '_depth.png'),
                os.path.join(category_train_path, prefix + '_info_depth.yaml'),
                os.path.join(category_train_path, prefix + '_pose.txt'),
                FOCAL_LENGTH,
            )
            pc = remove_point_cloud_background(pc)
            pc = pc.reshape(512, 512, 3)
            tifffile.imwrite(os.path.join(category_target_train_good_xyz_path, prefix + '.tiff'), pc)
            copyfile(
                os.path.join(category_train_path, prefix + '_image_4.png'),
                os.path.join(category_target_train_good_rgb_path, prefix + '.png')
            )
            if idx % 50 == 0:
                print(f"    Train {idx}/{num_train}")

        # ── Test data ──
        os.makedirs(os.path.join(category_target_path, 'test'), exist_ok=True)
        category_target_test_good_path = os.path.join(category_target_path, 'test', 'good')
        category_target_test_good_rgb_path = os.path.join(category_target_test_good_path, 'rgb')
        category_target_test_good_xyz_path = os.path.join(category_target_test_good_path, 'xyz')
        category_target_test_good_gt_path = os.path.join(category_target_test_good_path, 'gt')
        os.makedirs(category_target_test_good_rgb_path, exist_ok=True)
        os.makedirs(category_target_test_good_xyz_path, exist_ok=True)
        os.makedirs(category_target_test_good_gt_path, exist_ok=True)

        category_target_test_bad_path = os.path.join(category_target_path, 'test', 'bad')
        category_target_test_bad_rgb_path = os.path.join(category_target_test_bad_path, 'rgb')
        category_target_test_bad_xyz_path = os.path.join(category_target_test_bad_path, 'xyz')
        category_target_test_bad_gt_path = os.path.join(category_target_test_bad_path, 'gt')
        os.makedirs(category_target_test_bad_rgb_path, exist_ok=True)
        os.makedirs(category_target_test_bad_xyz_path, exist_ok=True)
        os.makedirs(category_target_test_bad_gt_path, exist_ok=True)

        test_depth_files = sorted([f for f in os.listdir(category_test_path) if f.endswith('_depth.png')])
        num_test_files = len(test_depth_files)
        print(f"  Test: {num_test_files} samples")

        good_count = 0
        bad_count = 0
        for depth_file in test_depth_files:
            prefix_2d = depth_file.replace('_depth.png', '')
            prefix_3d = prefix_2d.zfill(3) if len(prefix_2d) == 2 else prefix_2d
            mask_path = os.path.join(category_test_path, prefix_2d + '_mask.png')
            mask = cv2.imread(mask_path)

            pc = depth_to_pointcloud(
                os.path.join(category_test_path, prefix_2d + '_depth.png'),
                os.path.join(category_test_path, prefix_2d + '_info_depth.yaml'),
                os.path.join(category_test_path, prefix_2d + '_pose.txt'),
                FOCAL_LENGTH,
            )
            pc = remove_point_cloud_background(pc)
            pc = pc.reshape(512, 512, 3)

            if np.any(mask):
                tifffile.imwrite(os.path.join(category_target_test_bad_xyz_path, prefix_3d + '.tiff'), pc)
                cv2.imwrite(os.path.join(category_target_test_bad_gt_path, prefix_3d + '.png'), mask)
                src_rgb = os.path.join(category_test_path, prefix_2d + '_image_4.png')
                dst_rgb = os.path.join(category_target_test_bad_rgb_path, prefix_3d + '.png')
                copyfile(src_rgb, dst_rgb)
                bad_count += 1
            else:
                tifffile.imwrite(os.path.join(category_target_test_good_xyz_path, prefix_3d + '.tiff'), pc)
                cv2.imwrite(os.path.join(category_target_test_good_gt_path, prefix_3d + '.png'), mask)
                src_rgb = os.path.join(category_test_path, prefix_2d + '_image_4.png')
                dst_rgb = os.path.join(category_target_test_good_rgb_path, prefix_3d + '.png')
                copyfile(src_rgb, dst_rgb)
                good_count += 1

        print(f"    Good: {good_count}, Bad: {bad_count}")

    print(f"\nPreprocessing complete. Output: {args.target_dir}")
