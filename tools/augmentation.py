"""
Data Augmentation Pipeline for Underwater Waste Detection

This module provides augmentation pipelines specifically designed for
underwater marine debris detection tasks. Includes transformations that
simulate real-world underwater conditions.

Key Features:
- Underwater-specific augmentations (turbidity, color shifts, distortion)
- Bounding box-aware transformations
- Multiple strength levels
- Compatible with Albumentations library

Usage:
    from tools.augmentation import get_underwater_augmentation

    aug = get_underwater_augmentation(strength='medium')
    augmented = aug(image=image, bboxes=boxes, class_labels=labels)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal


def get_underwater_augmentation(
    strength: Literal['light', 'medium', 'strong'] = 'medium',
    image_size: tuple = (512, 512)
) -> A.Compose:
    """
    Get augmentation pipeline optimized for underwater imagery.

    This pipeline simulates real-world underwater conditions:
    - Water turbidity (fog, rain effects)
    - Variable lighting (depth-dependent)
    - Color shifts (blue/green dominance)
    - Optical distortion (water refraction)
    - Sensor noise

    Args:
        strength: Augmentation intensity level
                  - 'light': Conservative (p=0.2-0.3 per transform)
                  - 'medium': Balanced (p=0.3-0.5)  # RECOMMENDED
                  - 'strong': Aggressive (p=0.5-0.7)
        image_size: Target image size (height, width)
                   Default: (512, 512) for detection tasks

    Returns:
        Albumentations Compose object with bbox support

    Example:
        >>> aug = get_underwater_augmentation(strength='medium')
        >>> image = cv2.imread('underwater_debris.jpg')
        >>> boxes = [[0.1, 0.2, 0.3, 0.4]]  # [x_min, y_min, x_max, y_max] normalized
        >>> labels = ['Bottle']
        >>>
        >>> augmented = aug(image=image, bboxes=boxes, class_labels=labels)
        >>> aug_image = augmented['image']
        >>> aug_boxes = augmented['bboxes']
    """

    # Probability multiplier based on strength
    if strength == 'light':
        p_mult = 0.6  # 60% of default probabilities
    elif strength == 'medium':
        p_mult = 1.0  # 100% - balanced
    elif strength == 'strong':
        p_mult = 1.4  # 140% - more aggressive
    else:
        raise ValueError(f"strength must be 'light', 'medium', or 'strong', got: {strength}")

    return A.Compose([
        # =====================================================================
        # Underwater Turbidity Simulation
        # =====================================================================

        # Simulate particulate matter in water
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.4,
            alpha_coef=0.1,
            p=0.3 * p_mult
        ),

        # Simulate current-induced particles
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=10,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            p=0.3 * p_mult
        ),

        # =====================================================================
        # Underwater Lighting Conditions
        # =====================================================================

        # Depth-dependent brightness/contrast variation
        A.RandomBrightnessContrast(
            brightness_limit=0.4,  # ±40%
            contrast_limit=0.4,
            p=0.5 * p_mult
        ),

        # Color shifts (blue/green dominance at depth)
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,  # Slight blue/green shift
            p=0.5 * p_mult
        ),

        # Gamma correction (lighting variations)
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3 * p_mult
        ),

        # Hue-Saturation-Value adjustments
        A.HueSaturationValue(
            hue_shift_limit=15,     # Slight color shift
            sat_shift_limit=30,     # Saturation changes
            val_shift_limit=20,     # Brightness changes
            p=0.3 * p_mult
        ),

        # =====================================================================
        # Optical Distortion (Water Refraction)
        # =====================================================================

        # Lens/water interface distortion
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.05,
            p=0.3 * p_mult
        ),

        # Grid-based distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=0.3 * p_mult
        ),

        # Elastic transformation (current effects)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.2 * p_mult
        ),

        # =====================================================================
        # Sensor Noise & Artifacts
        # =====================================================================

        # Gaussian noise (sensor noise)
        A.GaussNoise(
            var_limit=(10, 50),
            mean=0,
            p=0.2 * p_mult
        ),

        # ISO noise (low-light conditions)
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            p=0.2 * p_mult
        ),

        # Multiplicative noise
        A.MultiplicativeNoise(
            multiplier=(0.9, 1.1),
            per_channel=True,
            p=0.1 * p_mult
        ),

        # =====================================================================
        # Geometric Transformations (Bbox-aware)
        # =====================================================================

        # Horizontal flip (natural for underwater scenes)
        A.HorizontalFlip(p=0.5),

        # Vertical flip (objects at different orientations in water)
        A.VerticalFlip(p=0.3 * p_mult),

        # Small rotations (natural object orientation in water)
        A.Rotate(
            limit=15,  # ±15 degrees
            border_mode=0,
            p=0.3 * p_mult
        ),

        # Perspective transformation (camera angle variation)
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            p=0.3 * p_mult
        ),

        # Random scale (object size variation)
        A.RandomScale(
            scale_limit=0.2,  # ±20%
            p=0.3 * p_mult
        ),

        # =====================================================================
        # Image Quality Degradation
        # =====================================================================

        # Blur effects (water clarity)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2 * p_mult),

        # Sharpen (occasionally)
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.1 * p_mult
        ),

        # =====================================================================
        # Final Normalization & Resize
        # =====================================================================

        # Ensure target size
        A.Resize(height=image_size[0], width=image_size[1]),

        # Normalize pixel values to [0, 1]
        # Note: Remove this if your model expects [0, 255]
        # A.Normalize(mean=0, std=1, max_pixel_value=255.0),

    ], bbox_params=A.BboxParams(
        format='albumentations',  # [x_min, y_min, x_max, y_max] normalized [0,1]
        min_area=0.05,            # Minimum 5% of image area
        min_visibility=0.3,       # At least 30% of box must be visible
        label_fields=['class_labels']  # Field name for class labels
    ))


def get_basic_augmentation(image_size: tuple = (512, 512)) -> A.Compose:
    """
    Get basic augmentation pipeline (minimal transforms).

    Use this for validation/testing or when you want minimal augmentation.
    Only includes safe geometric transforms.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Resize(height=image_size[0], width=image_size[1]),
    ], bbox_params=A.BboxParams(
        format='albumentations',
        min_area=0.05,
        min_visibility=0.3,
        label_fields=['class_labels']
    ))


def visualize_augmentation(
    image_path: str,
    boxes: list,
    labels: list,
    strength: str = 'medium',
    num_samples: int = 5
):
    """
    Visualize augmentation effects on an image.

    Useful for debugging and tuning augmentation parameters.

    Args:
        image_path: Path to input image
        boxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
        labels: List of class labels
        strength: Augmentation strength
        num_samples: Number of augmented samples to generate

    Example:
        >>> visualize_augmentation(
        ...     'image.jpg',
        ...     [[0.1, 0.2, 0.3, 0.4]],
        ...     ['Bottle'],
        ...     strength='medium',
        ...     num_samples=5
        ... )
    """
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get augmentation pipeline
    aug = get_underwater_augmentation(strength=strength)

    # Create subplot grid
    cols = 3
    rows = (num_samples + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Plot original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Draw original boxes
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    axes[0].imshow(image)

    # Generate and plot augmented samples
    for i in range(1, num_samples + 1):
        augmented = aug(image=image, bboxes=boxes, class_labels=labels)
        aug_image = augmented['image']
        aug_boxes = augmented['bboxes']

        # Draw augmented boxes
        for box in aug_boxes:
            x_min, y_min, x_max, y_max = box
            h, w = aug_image.shape[:2]
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
            cv2.rectangle(aug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        axes[i].imshow(aug_image)
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved visualization to: augmentation_samples.png")


# Example usage:
if __name__ == "__main__":
    # Get augmentation pipeline
    aug_pipeline = get_underwater_augmentation(strength='medium')

    print("Underwater Augmentation Pipeline:")
    print(f"  - Total transforms: {len(aug_pipeline.transforms)}")
    print(f"  - Bbox parameters: {aug_pipeline.processors}")
    print(f"\nAugmentation pipeline ready!")
    print(f"\nUsage:")
    print(f"  from tools.augmentation import get_underwater_augmentation")
    print(f"  aug = get_underwater_augmentation(strength='medium')")
    print(f"  augmented = aug(image=image, bboxes=boxes, class_labels=labels)")
