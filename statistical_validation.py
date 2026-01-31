"""
Statistical Validation for Corrosion Segmentation Measurements
==============================================================
This script provides experimental evidence for:
1. Measurement accuracy with statistical validation (mean ± std, confidence intervals)
2. Thickness measurement demonstration
3. Robustness to occlusion testing

Author: [Abbas Touqeer]
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, disk
import segmentation_models_pytorch as smp
from torchvision import transforms
import pandas as pd
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

# Set Arial font for all plots
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


# =============================================================================
# Configuration
# =============================================================================
CHECKPOINT_PATH = "./checkpoints/20241111_172413 Unet+resnet50+transform/best_model.pth"
IMAGES_DIR = "./Dataset/train/imgs"
MASKS_DIR = "./Dataset/train/masks"
PIXEL_SIZE_MM = 0.03
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./validation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Load Model
# =============================================================================
def load_model():
    """Load the trained segmentation model."""
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation="sigmoid"
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Checkpoint Dice: {checkpoint['dice_score']:.4f}, IoU: {checkpoint['iou_score']:.4f}")

    return model


# =============================================================================
# SECTION 1: Statistical Validation of Measurement Accuracy
# =============================================================================
def compute_length_from_skeleton(mask, pixel_size_mm):
    """Compute crack length from binary mask using skeletonization."""
    skeleton = skeletonize(mask > 0.5)
    length_pixels = np.sum(skeleton)
    # Account for diagonal connections (approximate)
    length_mm = length_pixels * pixel_size_mm * 1.1  # 1.1 factor for diagonal paths
    return length_mm, skeleton


def compute_measurement_statistics(model, images_dir, masks_dir, pixel_size_mm):
    """
    Compute measurement errors with full statistical analysis.

    Returns:
        dict: Statistical metrics including mean, std, confidence intervals
    """
    print("\n" + "="*60)
    print("SECTION 1: Statistical Validation of Measurement Accuracy")
    print("="*60)

    transform = transforms.Compose([transforms.ToTensor()])

    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    predicted_lengths = []
    ground_truth_lengths = []
    relative_errors = []
    absolute_errors = []
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for img_file, mask_file in zip(image_files, mask_files):
            # Load image and mask
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            image = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

            # Predict
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = model(img_tensor)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

            # Compute lengths
            pred_length, _ = compute_length_from_skeleton(pred_mask, pixel_size_mm)
            gt_length, _ = compute_length_from_skeleton(gt_mask, pixel_size_mm)

            if gt_length > 0:
                predicted_lengths.append(pred_length)
                ground_truth_lengths.append(gt_length)

                # Compute errors
                abs_error = abs(pred_length - gt_length)
                rel_error = (abs_error / gt_length) * 100
                absolute_errors.append(abs_error)
                relative_errors.append(rel_error)

                # Compute segmentation metrics
                intersection = np.sum(pred_mask * gt_mask)
                dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
                iou = intersection / (np.sum(pred_mask) + np.sum(gt_mask) - intersection + 1e-8)
                dice_scores.append(dice)
                iou_scores.append(iou)

    # Convert to numpy arrays
    relative_errors = np.array(relative_errors)
    absolute_errors = np.array(absolute_errors)
    predicted_lengths = np.array(predicted_lengths)
    ground_truth_lengths = np.array(ground_truth_lengths)

    # Statistical analysis
    n = len(relative_errors)
    mean_error = np.mean(relative_errors)
    std_error = np.std(relative_errors, ddof=1)
    median_error = np.median(relative_errors)

    # 95% Confidence Interval
    ci_95 = stats.t.interval(0.95, df=n-1, loc=mean_error, scale=std_error/np.sqrt(n))

    # 99% Confidence Interval
    ci_99 = stats.t.interval(0.99, df=n-1, loc=mean_error, scale=std_error/np.sqrt(n))

    # Percentiles
    percentile_25 = np.percentile(relative_errors, 25)
    percentile_75 = np.percentile(relative_errors, 75)
    percentile_95 = np.percentile(relative_errors, 95)

    # Results dictionary
    results = {
        'n_samples': n,
        'mean_relative_error': mean_error,
        'std_relative_error': std_error,
        'median_relative_error': median_error,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'ci_99_lower': ci_99[0],
        'ci_99_upper': ci_99[1],
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'percentile_95': percentile_95,
        'min_error': np.min(relative_errors),
        'max_error': np.max(relative_errors),
        'mean_absolute_error_mm': np.mean(absolute_errors),
        'std_absolute_error_mm': np.std(absolute_errors, ddof=1),
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores, ddof=1),
        'mean_iou': np.mean(iou_scores),
        'std_iou': np.std(iou_scores, ddof=1),
    }

    # Print results
    print(f"\nNumber of samples: {n}")
    print(f"\nRelative Error Statistics:")
    print(f"  Mean ± Std: {mean_error:.2f}% ± {std_error:.2f}%")
    print(f"  Median: {median_error:.2f}%")
    print(f"  95% CI: [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]")
    print(f"  99% CI: [{ci_99[0]:.2f}%, {ci_99[1]:.2f}%]")
    print(f"  IQR: [{percentile_25:.2f}%, {percentile_75:.2f}%]")
    print(f"  95th Percentile: {percentile_95:.2f}%")
    print(f"  Range: [{np.min(relative_errors):.2f}%, {np.max(relative_errors):.2f}%]")

    print(f"\nAbsolute Error Statistics:")
    print(f"  Mean ± Std: {np.mean(absolute_errors):.4f} mm ± {np.std(absolute_errors, ddof=1):.4f} mm")

    print(f"\nSegmentation Metrics:")
    print(f"  Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores, ddof=1):.4f}")
    print(f"  IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores, ddof=1):.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error distribution histogram
    axes[0, 0].hist(relative_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}%')
    axes[0, 0].axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}%')
    axes[0, 0].set_xlabel('Relative Error (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Measurement Errors')
    axes[0, 0].legend()

    # Predicted vs Ground Truth scatter plot
    axes[0, 1].scatter(ground_truth_lengths, predicted_lengths, alpha=0.6)
    max_val = max(np.max(ground_truth_lengths), np.max(predicted_lengths))
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement')
    axes[0, 1].set_xlabel('Ground Truth Length (mm)')
    axes[0, 1].set_ylabel('Predicted Length (mm)')
    axes[0, 1].set_title('Predicted vs Ground Truth Lengths')
    axes[0, 1].legend()

    # Box plot of errors
    axes[1, 0].boxplot(relative_errors)
    axes[1, 0].set_ylabel('Relative Error (%)')
    axes[1, 0].set_title('Error Distribution (Box Plot)')

    # Bland-Altman plot
    mean_lengths = (predicted_lengths + ground_truth_lengths) / 2
    diff_lengths = predicted_lengths - ground_truth_lengths
    mean_diff = np.mean(diff_lengths)
    std_diff = np.std(diff_lengths, ddof=1)

    axes[1, 1].scatter(mean_lengths, diff_lengths, alpha=0.6)
    axes[1, 1].axhline(mean_diff, color='r', linestyle='-', label=f'Mean: {mean_diff:.3f}')
    axes[1, 1].axhline(mean_diff + 1.96*std_diff, color='g', linestyle='--', label=f'+1.96 SD')
    axes[1, 1].axhline(mean_diff - 1.96*std_diff, color='g', linestyle='--', label=f'-1.96 SD')
    axes[1, 1].set_xlabel('Mean of Predicted and GT (mm)')
    axes[1, 1].set_ylabel('Difference (Predicted - GT) (mm)')
    axes[1, 1].set_title('Bland-Altman Plot')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'statistical_validation.png'), dpi=300)
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'measurement_statistics.csv'), index=False)

    return results


# =============================================================================
# SECTION 2: Thickness Measurement
# =============================================================================
def compute_thickness(mask, pixel_size_mm):
    """
    Compute crack thickness using distance transform.

    The thickness at each point is computed as twice the distance to the nearest
    boundary (diameter of the largest inscribed circle).
    """
    binary_mask = mask > 0.5

    # Distance transform gives distance to nearest background pixel
    distance = distance_transform_edt(binary_mask)

    # Thickness = 2 * distance (diameter)
    thickness_map = 2 * distance * pixel_size_mm

    # Get skeleton for centerline thickness measurements
    skeleton = skeletonize(binary_mask)

    # Thickness along the centerline
    centerline_thickness = thickness_map[skeleton]

    if len(centerline_thickness) == 0:
        return None, None, thickness_map

    stats = {
        'mean_thickness': np.mean(centerline_thickness),
        'std_thickness': np.std(centerline_thickness),
        'min_thickness': np.min(centerline_thickness),
        'max_thickness': np.max(centerline_thickness),
        'median_thickness': np.median(centerline_thickness),
    }

    return stats, centerline_thickness, thickness_map


def demonstrate_thickness_measurement(model, images_dir, masks_dir, pixel_size_mm, num_samples=5):
    """
    Demonstrate thickness measurement on sample images.
    """
    print("\n" + "="*60)
    print("SECTION 2: Thickness Measurement Demonstration")
    print("="*60)

    transform = transforms.Compose([transforms.ToTensor()])

    image_files = sorted(os.listdir(images_dir))[:num_samples]

    all_thickness_stats = []

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(images_dir, img_file)
            image = Image.open(img_path).convert("RGB")

            # Predict
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = model(img_tensor)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

            # Compute thickness
            stats, centerline_thickness, thickness_map = compute_thickness(pred_mask, pixel_size_mm)

            if stats is not None:
                stats['image'] = img_file
                all_thickness_stats.append(stats)

                # Visualizations
                axes[idx, 0].imshow(image)
                axes[idx, 0].set_title(f'Original Image')
                axes[idx, 0].axis('off')

                axes[idx, 1].imshow(pred_mask, cmap='gray')
                axes[idx, 1].set_title('Predicted Mask')
                axes[idx, 1].axis('off')

                im = axes[idx, 2].imshow(thickness_map, cmap='jet')
                axes[idx, 2].set_title('Thickness Map (mm)')
                axes[idx, 2].axis('off')
                plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)

                axes[idx, 3].hist(centerline_thickness, bins=20, edgecolor='black', alpha=0.7)
                axes[idx, 3].axvline(stats['mean_thickness'], color='r', linestyle='--',
                                     label=f"Mean: {stats['mean_thickness']:.3f} mm")
                axes[idx, 3].set_xlabel('Thickness (mm)')
                axes[idx, 3].set_ylabel('Frequency')
                axes[idx, 3].set_title('Thickness Distribution')
                axes[idx, 3].legend()

                print(f"\n{img_file}:")
                print(f"  Mean Thickness: {stats['mean_thickness']:.4f} ± {stats['std_thickness']:.4f} mm")
                print(f"  Min: {stats['min_thickness']:.4f} mm, Max: {stats['max_thickness']:.4f} mm")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'thickness_measurement.png'), dpi=300)
    plt.show()

    # Save thickness statistics
    if all_thickness_stats:
        thickness_df = pd.DataFrame(all_thickness_stats)
        thickness_df.to_csv(os.path.join(OUTPUT_DIR, 'thickness_statistics.csv'), index=False)

        print(f"\nOverall Thickness Statistics (n={len(all_thickness_stats)}):")
        print(f"  Mean: {thickness_df['mean_thickness'].mean():.4f} ± {thickness_df['mean_thickness'].std():.4f} mm")

    return all_thickness_stats


# =============================================================================
# SECTION 3: Robustness to Occlusion Testing
# =============================================================================
def apply_occlusion(mask, occlusion_ratio, occlusion_type='random'):
    """
    Apply synthetic occlusion to a mask.

    Args:
        mask: Binary mask
        occlusion_ratio: Fraction of mask to occlude (0.0 to 1.0)
        occlusion_type: 'random', 'block', or 'scattered'

    Returns:
        Occluded mask
    """
    occluded = mask.copy()
    h, w = mask.shape

    if occlusion_type == 'random':
        # Random pixel occlusion
        num_pixels = int(np.sum(mask > 0) * occlusion_ratio)
        foreground_coords = np.argwhere(mask > 0)
        if len(foreground_coords) > 0:
            indices = np.random.choice(len(foreground_coords),
                                       min(num_pixels, len(foreground_coords)),
                                       replace=False)
            for idx in indices:
                y, x = foreground_coords[idx]
                occluded[y, x] = 0

    elif occlusion_type == 'block':
        # Block occlusion
        foreground_coords = np.argwhere(mask > 0)
        if len(foreground_coords) > 0:
            center_idx = np.random.randint(len(foreground_coords))
            cy, cx = foreground_coords[center_idx]
            block_size = int(np.sqrt(np.sum(mask > 0) * occlusion_ratio))
            y1, y2 = max(0, cy - block_size//2), min(h, cy + block_size//2)
            x1, x2 = max(0, cx - block_size//2), min(w, cx + block_size//2)
            occluded[y1:y2, x1:x2] = 0

    elif occlusion_type == 'scattered':
        # Multiple small scattered occlusions
        num_blocks = 5
        block_ratio = occlusion_ratio / num_blocks
        foreground_coords = np.argwhere(mask > 0)
        if len(foreground_coords) > 0:
            for _ in range(num_blocks):
                center_idx = np.random.randint(len(foreground_coords))
                cy, cx = foreground_coords[center_idx]
                block_size = int(np.sqrt(np.sum(mask > 0) * block_ratio))
                y1, y2 = max(0, cy - block_size//2), min(h, cy + block_size//2)
                x1, x2 = max(0, cx - block_size//2), min(w, cx + block_size//2)
                occluded[y1:y2, x1:x2] = 0

    return occluded


def test_occlusion_robustness(model, images_dir, masks_dir, pixel_size_mm):
    """
    Test model robustness to various levels of occlusion.
    """
    print("\n" + "="*60)
    print("SECTION 3: Robustness to Occlusion Testing")
    print("="*60)

    transform = transforms.Compose([transforms.ToTensor()])

    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    occlusion_types = ['random', 'block', 'scattered']

    results = {otype: {level: {'dice': [], 'iou': [], 'length_error': []}
                       for level in occlusion_levels}
               for otype in occlusion_types}

    with torch.no_grad():
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            image = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = model(img_tensor)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

            gt_length, _ = compute_length_from_skeleton(gt_mask, pixel_size_mm)

            if gt_length == 0:
                continue

            for otype in occlusion_types:
                for level in occlusion_levels:
                    if level == 0:
                        occluded_gt = gt_mask
                    else:
                        occluded_gt = apply_occlusion(gt_mask, level, otype)

                    # Compute metrics on occluded ground truth
                    intersection = np.sum(pred_mask * occluded_gt)
                    dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(occluded_gt) + 1e-8)
                    iou = intersection / (np.sum(pred_mask) + np.sum(occluded_gt) - intersection + 1e-8)

                    occluded_length, _ = compute_length_from_skeleton(occluded_gt, pixel_size_mm)
                    if occluded_length > 0:
                        length_error = abs(occluded_length - gt_length) / gt_length * 100
                    else:
                        length_error = 100.0

                    results[otype][level]['dice'].append(dice)
                    results[otype][level]['iou'].append(iou)
                    results[otype][level]['length_error'].append(length_error)

    # Compile statistics
    summary = []
    for otype in occlusion_types:
        for level in occlusion_levels:
            dice_vals = results[otype][level]['dice']
            iou_vals = results[otype][level]['iou']
            error_vals = results[otype][level]['length_error']

            if len(dice_vals) > 0:
                summary.append({
                    'occlusion_type': otype,
                    'occlusion_level': level,
                    'dice_mean': np.mean(dice_vals),
                    'dice_std': np.std(dice_vals),
                    'iou_mean': np.mean(iou_vals),
                    'iou_std': np.std(iou_vals),
                    'length_error_mean': np.mean(error_vals),
                    'length_error_std': np.std(error_vals),
                })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'occlusion_robustness.csv'), index=False)

    # Print results
    print("\nOcclusion Robustness Results:")
    print("-" * 80)
    for otype in occlusion_types:
        print(f"\n{otype.upper()} Occlusion:")
        print(f"{'Level':<10} {'Dice':<20} {'IoU':<20} {'Length Error':<20}")
        print("-" * 70)
        for level in occlusion_levels:
            row = summary_df[(summary_df['occlusion_type'] == otype) &
                            (summary_df['occlusion_level'] == level)].iloc[0]
            print(f"{level:<10.1f} {row['dice_mean']:.4f} ± {row['dice_std']:.4f}   "
                  f"{row['iou_mean']:.4f} ± {row['iou_std']:.4f}   "
                  f"{row['length_error_mean']:.2f}% ± {row['length_error_std']:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for otype in occlusion_types:
        type_data = summary_df[summary_df['occlusion_type'] == otype]
        axes[0].errorbar(type_data['occlusion_level'], type_data['dice_mean'],
                        yerr=type_data['dice_std'], marker='o', label=otype, capsize=3)
        axes[1].errorbar(type_data['occlusion_level'], type_data['iou_mean'],
                        yerr=type_data['iou_std'], marker='o', label=otype, capsize=3)
        axes[2].errorbar(type_data['occlusion_level'], type_data['length_error_mean'],
                        yerr=type_data['length_error_std'], marker='o', label=otype, capsize=3)

    axes[0].set_xlabel('Occlusion Ratio')
    axes[0].set_ylabel('Dice Score')
    axes[0].set_title('Dice vs Occlusion Level')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Occlusion Ratio')
    axes[1].set_ylabel('IoU Score')
    axes[1].set_title('IoU vs Occlusion Level')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Occlusion Ratio')
    axes[2].set_ylabel('Length Error (%)')
    axes[2].set_title('Length Error vs Occlusion Level')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occlusion_robustness.png'), dpi=300)
    plt.show()

    # Visualize occlusion examples
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Get a sample mask for visualization
    sample_mask_path = os.path.join(masks_dir, mask_files[0])
    sample_mask = np.array(Image.open(sample_mask_path).convert("L")) / 255.0

    for i, level in enumerate([0.0, 0.2, 0.4]):
        for j, otype in enumerate(['random', 'block', 'scattered']):
            ax_idx = i
            if level == 0:
                occluded = sample_mask
            else:
                occluded = apply_occlusion(sample_mask, level, otype)

            axes[j, i+1].imshow(occluded, cmap='gray')
            axes[j, i+1].set_title(f'{otype.capitalize()}\n{int(level*100)}% Occlusion')
            axes[j, i+1].axis('off')

    # Original mask in first column
    for j in range(3):
        axes[j, 0].imshow(sample_mask, cmap='gray')
        axes[j, 0].set_title('Original Mask')
        axes[j, 0].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occlusion_examples.png'), dpi=300)
    plt.show()

    return summary_df


# =============================================================================
# SECTION 4: Statistical Significance Test (Model Comparison)
# =============================================================================
# Model checkpoints for comparison
MODEL_CHECKPOINTS = {
    'UNet_ResNet50': './checkpoints/20241111_172413 Unet+resnet50+transform/best_model.pth',
    'UNet_ResNet101': './checkpoints/20241111_175300 Unet+resnet101+transform/best_model.pth',
    'UNet_ResNet34': './checkpoints/20241116_212445 Unet+resnet34+transform/best_model.pth',
    'DeepLabV3_ResNet34': './checkpoints/20241116_215529 DeepLabV3+resnet34+transform/best_model.pth',
    'DeepLabV3_ResNet50': './checkpoints/20241116_225151 DeepLabV3+resnet50+transform/best_model.pth',
    'FPN_ResNet34': './checkpoints/20241118_124808 FPN+resnet34+transform/best_model.pth',
    'FPN_ResNet50': './checkpoints/20241118_145046 FPN+resnet50+transform/best_model.pth',
    'LinkNet_ResNet34': './checkpoints/20241118_163456 Linknet+resnet34+transform/best_model.pth',
}

MODEL_ARCHITECTURES = {
    'UNet': smp.Unet,
    'DeepLabV3': smp.DeepLabV3,
    'FPN': smp.FPN,
    'LinkNet': smp.Linknet,
}


def load_model_by_config(arch_name, encoder_name, checkpoint_path):
    """Load a model with specific architecture and encoder."""
    arch_class = MODEL_ARCHITECTURES[arch_name]
    model = arch_class(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid"
    ).to(DEVICE)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    return None


def compute_model_scores(model, images_dir, masks_dir):
    """Compute Dice and IoU scores for a model on all images."""
    transform = transforms.Compose([transforms.ToTensor()])
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            image = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = model(img_tensor)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

            intersection = np.sum(pred_mask * gt_mask)
            dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
            iou = intersection / (np.sum(pred_mask) + np.sum(gt_mask) - intersection + 1e-8)

            dice_scores.append(dice)
            iou_scores.append(iou)

    return np.array(dice_scores), np.array(iou_scores)


def statistical_significance_test(images_dir, masks_dir):
    """
    Perform statistical significance tests comparing different models.
    Uses paired t-test and Wilcoxon signed-rank test.
    """
    print("\n" + "="*60)
    print("SECTION 4: Statistical Significance Test")
    print("="*60)

    model_scores = {}

    # Collect scores for each available model
    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_name}: checkpoint not found")
            continue

        parts = model_name.split('_')
        arch_name = parts[0]
        encoder_name = parts[1].lower()

        print(f"Evaluating {model_name}...")
        model = load_model_by_config(arch_name, encoder_name, checkpoint_path)
        if model is not None:
            dice, iou = compute_model_scores(model, images_dir, masks_dir)
            model_scores[model_name] = {'dice': dice, 'iou': iou}
            print(f"  Dice: {np.mean(dice):.4f} ± {np.std(dice):.4f}")
            print(f"  IoU: {np.mean(iou):.4f} ± {np.std(iou):.4f}")

    if len(model_scores) < 2:
        print("Not enough models for comparison")
        return None

    # Create comparison table
    model_names = list(model_scores.keys())
    comparison_results = []

    # Summary statistics table
    summary_table = []
    for name in model_names:
        summary_table.append({
            'Model': name,
            'Dice_Mean': np.mean(model_scores[name]['dice']),
            'Dice_Std': np.std(model_scores[name]['dice']),
            'IoU_Mean': np.mean(model_scores[name]['iou']),
            'IoU_Std': np.std(model_scores[name]['iou']),
        })

    summary_df = pd.DataFrame(summary_table)
    summary_df = summary_df.sort_values('Dice_Mean', ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_summary.csv'), index=False)

    print("\n" + "-"*60)
    print("Model Performance Summary (sorted by Dice):")
    print("-"*60)
    print(summary_df.to_string(index=False))

    # Pairwise statistical tests (compare best model with others)
    best_model = summary_df.iloc[0]['Model']
    best_dice = model_scores[best_model]['dice']
    best_iou = model_scores[best_model]['iou']

    print(f"\n" + "-"*60)
    print(f"Statistical Significance Tests (vs {best_model}):")
    print("-"*60)
    print(f"{'Model':<25} {'t-test p-value':<18} {'Wilcoxon p-value':<18} {'Significant?':<12}")
    print("-"*60)

    for name in model_names:
        if name == best_model:
            continue

        other_dice = model_scores[name]['dice']

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(best_dice, other_dice)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = stats.wilcoxon(best_dice, other_dice)
        except:
            w_pvalue = 1.0

        significant = "Yes" if t_pvalue < 0.05 and w_pvalue < 0.05 else "No"

        comparison_results.append({
            'Model_A': best_model,
            'Model_B': name,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_pvalue': w_pvalue,
            'significant_at_0.05': significant,
        })

        print(f"{name:<25} {t_pvalue:<18.6f} {w_pvalue:<18.6f} {significant:<12}")

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_significance_tests.csv'), index=False)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot for Dice scores
    dice_data = [model_scores[name]['dice'] for name in model_names]
    bp1 = axes[0].boxplot(dice_data, labels=model_names, patch_artist=True)
    axes[0].set_ylabel('Dice Score')
    axes[0].set_title('Model Comparison - Dice Score')
    axes[0].tick_params(axis='x', rotation=45)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')

    # Box plot for IoU scores
    iou_data = [model_scores[name]['iou'] for name in model_names]
    bp2 = axes[1].boxplot(iou_data, labels=model_names, patch_artist=True)
    axes[1].set_ylabel('IoU Score')
    axes[1].set_title('Model Comparison - IoU Score')
    axes[1].tick_params(axis='x', rotation=45)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_boxplot.png'), dpi=300)
    plt.show()

    return comparison_df, summary_df


# =============================================================================
# SECTION 5: Small Crack Detection Analysis
# =============================================================================
def analyze_crack_size_performance(model, images_dir, masks_dir, pixel_size_mm):
    """
    Analyze model performance stratified by crack size.
    Categorizes cracks as small, medium, or large based on area.
    """
    print("\n" + "="*60)
    print("SECTION 5: Small Crack Detection Analysis")
    print("="*60)

    transform = transforms.Compose([transforms.ToTensor()])
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    results = []

    with torch.no_grad():
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            image = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = model(img_tensor)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

            # Label connected components in ground truth
            gt_labels = label(gt_mask > 0.5)
            props = regionprops(gt_labels)

            for prop in props:
                area_pixels = prop.area
                area_mm2 = area_pixels * (pixel_size_mm ** 2)

                # Get bounding box for this component
                minr, minc, maxr, maxc = prop.bbox

                # Extract component masks
                component_gt = (gt_labels == prop.label).astype(float)
                component_pred = pred_mask * component_gt  # Intersection with GT region

                # Expand region slightly for prediction evaluation
                pad = 5
                minr_pad = max(0, minr - pad)
                maxr_pad = min(gt_mask.shape[0], maxr + pad)
                minc_pad = max(0, minc - pad)
                maxc_pad = min(gt_mask.shape[1], maxc + pad)

                region_gt = gt_mask[minr_pad:maxr_pad, minc_pad:maxc_pad]
                region_pred = pred_mask[minr_pad:maxr_pad, minc_pad:maxc_pad]

                # Compute metrics for this component
                intersection = np.sum(region_pred * region_gt)
                dice = (2 * intersection) / (np.sum(region_pred) + np.sum(region_gt) + 1e-8)
                iou = intersection / (np.sum(region_pred) + np.sum(region_gt) - intersection + 1e-8)

                # Detection: consider detected if IoU > 0.1
                detected = iou > 0.1

                # Length estimation
                gt_length, _ = compute_length_from_skeleton(region_gt, pixel_size_mm)
                pred_length, _ = compute_length_from_skeleton(region_pred * (region_gt > 0), pixel_size_mm)

                if gt_length > 0:
                    length_error = abs(pred_length - gt_length) / gt_length * 100
                else:
                    length_error = np.nan

                results.append({
                    'image': img_file,
                    'component_id': prop.label,
                    'area_pixels': area_pixels,
                    'area_mm2': area_mm2,
                    'length_mm': gt_length,
                    'dice': dice,
                    'iou': iou,
                    'detected': detected,
                    'length_error': length_error,
                })

    results_df = pd.DataFrame(results)

    # Categorize by size (using area percentiles)
    area_33 = results_df['area_mm2'].quantile(0.33)
    area_66 = results_df['area_mm2'].quantile(0.66)

    def categorize_size(area):
        if area < area_33:
            return 'Small'
        elif area < area_66:
            return 'Medium'
        else:
            return 'Large'

    results_df['size_category'] = results_df['area_mm2'].apply(categorize_size)

    # Compute statistics by size category
    print("\nPerformance by Crack Size:")
    print("-"*80)
    print(f"{'Category':<10} {'Count':<8} {'Area (mm²)':<20} {'Dice':<18} {'IoU':<18} {'Detection Rate':<15}")
    print("-"*80)

    size_stats = []
    for category in ['Small', 'Medium', 'Large']:
        cat_data = results_df[results_df['size_category'] == category]
        if len(cat_data) == 0:
            continue

        stats_row = {
            'category': category,
            'count': len(cat_data),
            'area_mean': cat_data['area_mm2'].mean(),
            'area_std': cat_data['area_mm2'].std(),
            'dice_mean': cat_data['dice'].mean(),
            'dice_std': cat_data['dice'].std(),
            'iou_mean': cat_data['iou'].mean(),
            'iou_std': cat_data['iou'].std(),
            'detection_rate': cat_data['detected'].mean() * 100,
            'length_error_mean': cat_data['length_error'].dropna().mean(),
            'length_error_std': cat_data['length_error'].dropna().std(),
        }
        size_stats.append(stats_row)

        print(f"{category:<10} {len(cat_data):<8} "
              f"{stats_row['area_mean']:.3f}±{stats_row['area_std']:.3f}      "
              f"{stats_row['dice_mean']:.4f}±{stats_row['dice_std']:.4f}  "
              f"{stats_row['iou_mean']:.4f}±{stats_row['iou_std']:.4f}  "
              f"{stats_row['detection_rate']:.1f}%")

    size_stats_df = pd.DataFrame(size_stats)
    size_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'crack_size_analysis.csv'), index=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'crack_size_details.csv'), index=False)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Dice by size category
    categories = ['Small', 'Medium', 'Large']
    dice_by_cat = [results_df[results_df['size_category'] == cat]['dice'].values for cat in categories]
    bp1 = axes[0, 0].boxplot([d for d in dice_by_cat if len(d) > 0],
                             labels=[c for c, d in zip(categories, dice_by_cat) if len(d) > 0],
                             patch_artist=True)
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].set_title('Dice Score by Crack Size')
    for patch, color in zip(bp1['boxes'], ['#ff9999', '#99ff99', '#9999ff']):
        patch.set_facecolor(color)

    # IoU by size category
    iou_by_cat = [results_df[results_df['size_category'] == cat]['iou'].values for cat in categories]
    bp2 = axes[0, 1].boxplot([d for d in iou_by_cat if len(d) > 0],
                             labels=[c for c, d in zip(categories, iou_by_cat) if len(d) > 0],
                             patch_artist=True)
    axes[0, 1].set_ylabel('IoU Score')
    axes[0, 1].set_title('IoU Score by Crack Size')
    for patch, color in zip(bp2['boxes'], ['#ff9999', '#99ff99', '#9999ff']):
        patch.set_facecolor(color)

    # Detection rate by size
    detection_rates = [results_df[results_df['size_category'] == cat]['detected'].mean() * 100
                       for cat in categories]
    colors = ['#ff9999', '#99ff99', '#9999ff']
    axes[1, 0].bar(categories, detection_rates, color=colors)
    axes[1, 0].set_ylabel('Detection Rate (%)')
    axes[1, 0].set_title('Detection Rate by Crack Size')
    axes[1, 0].set_ylim(0, 105)
    for i, v in enumerate(detection_rates):
        axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center')

    # Scatter plot: Area vs Dice with legend
    color_map = {'Small': '#ff9999', 'Medium': '#99ff99', 'Large': '#9999ff'}
    for category in ['Small', 'Medium', 'Large']:
        cat_data = results_df[results_df['size_category'] == category]
        axes[1, 1].scatter(cat_data['area_mm2'], cat_data['dice'],
                          c=color_map[category], label=category, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Crack Area (mm²)')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_title('Dice Score vs Crack Area')
    axes[1, 1].legend(title='Crack Size', loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'crack_size_analysis.png'), dpi=300)
    plt.show()

    return results_df, size_stats_df


# =============================================================================
# SECTION 6: Noise Robustness Testing
# =============================================================================
def add_gaussian_noise(image, sigma=25):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(image, kernel_size=5):
    """Add Gaussian blur to image."""
    from scipy.ndimage import gaussian_filter
    blurred = np.zeros_like(image)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma=kernel_size/3)
    return blurred.astype(np.uint8)


def adjust_brightness(image, factor=1.0):
    """Adjust image brightness."""
    adjusted = image.astype(float) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def test_noise_robustness(model, images_dir, masks_dir):
    """
    Test model robustness to various image degradations.
    """
    print("\n" + "="*60)
    print("SECTION 6: Noise Robustness Testing")
    print("="*60)

    transform = transforms.Compose([transforms.ToTensor()])
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    # Test conditions
    noise_levels = [0, 10, 25, 50, 75]
    blur_levels = [0, 3, 5, 7, 9]
    brightness_levels = [0.5, 0.75, 1.0, 1.25, 1.5]

    results = {
        'gaussian_noise': {level: {'dice': [], 'iou': []} for level in noise_levels},
        'blur': {level: {'dice': [], 'iou': []} for level in blur_levels},
        'brightness': {level: {'dice': [], 'iou': []} for level in brightness_levels},
    }

    with torch.no_grad():
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            image = np.array(Image.open(img_path).convert("RGB"))
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

            # Test Gaussian noise
            for sigma in noise_levels:
                if sigma == 0:
                    degraded = image
                else:
                    degraded = add_gaussian_noise(image, sigma)

                img_tensor = transform(Image.fromarray(degraded)).unsqueeze(0).to(DEVICE)
                pred = model(img_tensor)
                pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

                intersection = np.sum(pred_mask * gt_mask)
                dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
                iou = intersection / (np.sum(pred_mask) + np.sum(gt_mask) - intersection + 1e-8)

                results['gaussian_noise'][sigma]['dice'].append(dice)
                results['gaussian_noise'][sigma]['iou'].append(iou)

            # Test blur
            for kernel in blur_levels:
                if kernel == 0:
                    degraded = image
                else:
                    degraded = add_blur(image, kernel)

                img_tensor = transform(Image.fromarray(degraded)).unsqueeze(0).to(DEVICE)
                pred = model(img_tensor)
                pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

                intersection = np.sum(pred_mask * gt_mask)
                dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
                iou = intersection / (np.sum(pred_mask) + np.sum(gt_mask) - intersection + 1e-8)

                results['blur'][kernel]['dice'].append(dice)
                results['blur'][kernel]['iou'].append(iou)

            # Test brightness
            for factor in brightness_levels:
                degraded = adjust_brightness(image, factor)

                img_tensor = transform(Image.fromarray(degraded)).unsqueeze(0).to(DEVICE)
                pred = model(img_tensor)
                pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

                intersection = np.sum(pred_mask * gt_mask)
                dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
                iou = intersection / (np.sum(pred_mask) + np.sum(gt_mask) - intersection + 1e-8)

                results['brightness'][factor]['dice'].append(dice)
                results['brightness'][factor]['iou'].append(iou)

    # Compile results
    summary = []

    print("\nGaussian Noise Robustness:")
    print("-"*50)
    print(f"{'Sigma':<10} {'Dice':<25} {'IoU':<25}")
    print("-"*50)
    for sigma in noise_levels:
        dice_mean = np.mean(results['gaussian_noise'][sigma]['dice'])
        dice_std = np.std(results['gaussian_noise'][sigma]['dice'])
        iou_mean = np.mean(results['gaussian_noise'][sigma]['iou'])
        iou_std = np.std(results['gaussian_noise'][sigma]['iou'])
        print(f"{sigma:<10} {dice_mean:.4f} ± {dice_std:.4f}      {iou_mean:.4f} ± {iou_std:.4f}")
        summary.append({
            'degradation_type': 'gaussian_noise',
            'level': sigma,
            'dice_mean': dice_mean,
            'dice_std': dice_std,
            'iou_mean': iou_mean,
            'iou_std': iou_std,
        })

    print("\nBlur Robustness:")
    print("-"*50)
    print(f"{'Kernel':<10} {'Dice':<25} {'IoU':<25}")
    print("-"*50)
    for kernel in blur_levels:
        dice_mean = np.mean(results['blur'][kernel]['dice'])
        dice_std = np.std(results['blur'][kernel]['dice'])
        iou_mean = np.mean(results['blur'][kernel]['iou'])
        iou_std = np.std(results['blur'][kernel]['iou'])
        print(f"{kernel:<10} {dice_mean:.4f} ± {dice_std:.4f}      {iou_mean:.4f} ± {iou_std:.4f}")
        summary.append({
            'degradation_type': 'blur',
            'level': kernel,
            'dice_mean': dice_mean,
            'dice_std': dice_std,
            'iou_mean': iou_mean,
            'iou_std': iou_std,
        })

    print("\nBrightness Robustness:")
    print("-"*50)
    print(f"{'Factor':<10} {'Dice':<25} {'IoU':<25}")
    print("-"*50)
    for factor in brightness_levels:
        dice_mean = np.mean(results['brightness'][factor]['dice'])
        dice_std = np.std(results['brightness'][factor]['dice'])
        iou_mean = np.mean(results['brightness'][factor]['iou'])
        iou_std = np.std(results['brightness'][factor]['iou'])
        print(f"{factor:<10.2f} {dice_mean:.4f} ± {dice_std:.4f}      {iou_mean:.4f} ± {iou_std:.4f}")
        summary.append({
            'degradation_type': 'brightness',
            'level': factor,
            'dice_mean': dice_mean,
            'dice_std': dice_std,
            'iou_mean': iou_mean,
            'iou_std': iou_std,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'noise_robustness.csv'), index=False)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gaussian noise
    noise_data = summary_df[summary_df['degradation_type'] == 'gaussian_noise']
    axes[0].errorbar(noise_data['level'], noise_data['dice_mean'],
                     yerr=noise_data['dice_std'], marker='o', label='Dice', capsize=3)
    axes[0].errorbar(noise_data['level'], noise_data['iou_mean'],
                     yerr=noise_data['iou_std'], marker='s', label='IoU', capsize=3)
    axes[0].set_xlabel('Noise Sigma')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Gaussian Noise Robustness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Blur
    blur_data = summary_df[summary_df['degradation_type'] == 'blur']
    axes[1].errorbar(blur_data['level'], blur_data['dice_mean'],
                     yerr=blur_data['dice_std'], marker='o', label='Dice', capsize=3)
    axes[1].errorbar(blur_data['level'], blur_data['iou_mean'],
                     yerr=blur_data['iou_std'], marker='s', label='IoU', capsize=3)
    axes[1].set_xlabel('Blur Kernel Size')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Blur Robustness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Brightness
    bright_data = summary_df[summary_df['degradation_type'] == 'brightness']
    axes[2].errorbar(bright_data['level'], bright_data['dice_mean'],
                     yerr=bright_data['dice_std'], marker='o', label='Dice', capsize=3)
    axes[2].errorbar(bright_data['level'], bright_data['iou_mean'],
                     yerr=bright_data['iou_std'], marker='s', label='IoU', capsize=3)
    axes[2].set_xlabel('Brightness Factor')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Brightness Robustness')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_robustness.png'), dpi=300)
    plt.show()

    # Visualize degradation examples
    sample_img_path = os.path.join(images_dir, image_files[0])
    sample_image = np.array(Image.open(sample_img_path).convert("RGB"))

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Noise examples
    for i, sigma in enumerate(noise_levels):
        if sigma == 0:
            axes[0, i].imshow(sample_image)
        else:
            axes[0, i].imshow(add_gaussian_noise(sample_image, sigma))
        axes[0, i].set_title(f'σ={sigma}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Gaussian Noise', fontsize=12)

    # Blur examples
    for i, kernel in enumerate(blur_levels):
        if kernel == 0:
            axes[1, i].imshow(sample_image)
        else:
            axes[1, i].imshow(add_blur(sample_image, kernel))
        axes[1, i].set_title(f'k={kernel}')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Blur', fontsize=12)

    # Brightness examples
    for i, factor in enumerate(brightness_levels):
        axes[2, i].imshow(adjust_brightness(sample_image, factor))
        axes[2, i].set_title(f'×{factor}')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Brightness', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_examples.png'), dpi=300)
    plt.show()

    return summary_df


# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("="*60)
    print("Statistical Validation for Corrosion Segmentation")
    print("="*60)

    # Load model
    model = load_model()

    # Section 1: Statistical validation of measurement accuracy
    measurement_stats = compute_measurement_statistics(
        model, IMAGES_DIR, MASKS_DIR, PIXEL_SIZE_MM
    )

    # Section 2: Thickness measurement demonstration
    thickness_stats = demonstrate_thickness_measurement(
        model, IMAGES_DIR, MASKS_DIR, PIXEL_SIZE_MM, num_samples=5
    )

    # Section 3: Occlusion robustness testing
    occlusion_results = test_occlusion_robustness(
        model, IMAGES_DIR, MASKS_DIR, PIXEL_SIZE_MM
    )

    # Section 4: Statistical significance test
    significance_results = statistical_significance_test(
        IMAGES_DIR, MASKS_DIR
    )

    # Section 5: Small crack detection analysis
    crack_size_results = analyze_crack_size_performance(
        model, IMAGES_DIR, MASKS_DIR, PIXEL_SIZE_MM
    )

    # Section 6: Noise robustness testing
    noise_results = test_noise_robustness(
        model, IMAGES_DIR, MASKS_DIR
    )

    print("\n" + "="*60)
    print("Validation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

    print("\nGenerated Files:")
    print("-"*40)
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")

    return {
        'measurement_stats': measurement_stats,
        'thickness_stats': thickness_stats,
        'occlusion_results': occlusion_results,
        'significance_results': significance_results,
        'crack_size_results': crack_size_results,
        'noise_results': noise_results,
    }


if __name__ == "__main__":
    main()
