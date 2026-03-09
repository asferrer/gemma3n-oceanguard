"""
Custom Loss Functions for Gemma3n Object Detection Fine-tuning

This module implements advanced loss functions optimized for:
1. Extreme class imbalance (Focal Loss)
2. Better bounding box regression (DIoU/CIoU Loss)
3. Class weighting for balanced training

References:
- Focal Loss: https://arxiv.org/abs/1708.02002
- DIoU Loss: https://arxiv.org/abs/1911.08287
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import json
from collections import Counter
from pathlib import Path


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Particularly effective for object detection with rare classes.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha (float): Weighting factor for positive class (0-1)
                      Default: 0.25 (25% weight on positive class)
        gamma (float): Focusing parameter (0-5)
                      Higher gamma = more focus on hard examples
                      Default: 2.5 (more aggressive than standard 2.0)
        reduction (str): Specifies reduction: 'mean' | 'sum' | 'none'

    Example:
        >>> focal_loss = FocalLoss(alpha=0.25, gamma=2.5)
        >>> logits = model(inputs)  # [batch, vocab_size]
        >>> loss = focal_loss(logits, targets)  # [batch]
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [batch_size, num_classes] or [batch_size, seq_len, vocab_size]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]

        Returns:
            loss: Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Flatten if needed (for sequence models)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Compute p_t: probability of correct class
        p = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting
        focal_loss = self.alpha * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DIoULoss(nn.Module):
    """
    Distance-IoU Loss for bounding box regression.

    DIoU improves upon IoU and GIoU by directly minimizing the normalized distance
    between centers of predicted and ground truth boxes, in addition to IoU.

    Advantages over IoU/GIoU:
    - Faster convergence (2-3x faster in practice)
    - Better for overlapping and non-overlapping boxes
    - Stable gradients

    Formula: DIoU = IoU - (ρ² / c²)
    where ρ is center distance and c is diagonal of smallest enclosing box

    Args:
        eps (float): Small epsilon to prevent division by zero

    Example:
        >>> diou_loss = DIoULoss()
        >>> pred_boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # [x_min, y_min, x_max, y_max]
        >>> target_boxes = torch.tensor([[0.15, 0.25, 0.35, 0.45]])
        >>> loss = diou_loss(pred_boxes, target_boxes)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 4] in format (x_min, y_min, x_max, y_max)
                       All values should be normalized to [0, 1]
            target_boxes: Ground truth boxes [N, 4] in same format

        Returns:
            loss: DIoU loss (mean over batch)
        """
        # Ensure boxes are in correct format
        assert pred_boxes.shape == target_boxes.shape
        assert pred_boxes.size(-1) == 4, "Boxes must be in format (x_min, y_min, x_max, y_max)"

        # Compute intersection area
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Compute union area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + self.eps

        # Compute IoU
        iou = inter_area / union_area

        # Compute center points
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

        # Compute squared center distance
        center_distance_sq = ((pred_center_x - target_center_x) ** 2 +
                             (pred_center_y - target_center_y) ** 2)

        # Compute diagonal of smallest enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_diagonal_sq = ((enclose_x2 - enclose_x1) ** 2 +
                              (enclose_y2 - enclose_y1) ** 2) + self.eps

        # Compute DIoU
        diou = iou - (center_distance_sq / enclose_diagonal_sq)

        # Loss is 1 - DIoU
        loss = 1 - diou

        return loss.mean()


class CIoULoss(nn.Module):
    """
    Complete-IoU Loss for bounding box regression.

    CIoU extends DIoU by additionally considering the aspect ratio consistency
    between predicted and ground truth boxes.

    Formula: CIoU = DIoU - α * v
    where v measures aspect ratio consistency and α is a trade-off parameter

    Use CIoU when:
    - Objects have varying aspect ratios
    - Precise shape matching is important

    Args:
        eps (float): Small epsilon to prevent division by zero

    Example:
        >>> ciou_loss = CIoULoss()
        >>> pred_boxes = torch.tensor([[0.1, 0.2, 0.3, 0.5]])
        >>> target_boxes = torch.tensor([[0.15, 0.25, 0.35, 0.55]])
        >>> loss = ciou_loss(pred_boxes, target_boxes)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 4] (x_min, y_min, x_max, y_max)
            target_boxes: Ground truth boxes [N, 4]

        Returns:
            loss: CIoU loss (mean over batch)
        """
        # First compute DIoU components
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # Center distance
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

        center_distance_sq = ((pred_center_x - target_center_x) ** 2 +
                             (pred_center_y - target_center_y) ** 2)

        # Enclosing box diagonal
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_diagonal_sq = ((enclose_x2 - enclose_x1) ** 2 +
                              (enclose_y2 - enclose_y1) ** 2) + self.eps

        # DIoU term
        diou_term = center_distance_sq / enclose_diagonal_sq

        # Aspect ratio consistency
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0] + self.eps
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1] + self.eps
        target_w = target_boxes[:, 2] - target_boxes[:, 0] + self.eps
        target_h = target_boxes[:, 3] - target_boxes[:, 1] + self.eps

        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )

        # Trade-off parameter alpha
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # CIoU = IoU - (center distance term + aspect ratio term)
        ciou = iou - diou_term - alpha * v

        # Loss is 1 - CIoU
        loss = 1 - ciou

        return loss.mean()


def compute_class_weights(
    dataset_path: str,
    smoothing: float = 0.1
) -> Dict[str, float]:
    """
    Compute class weights from training dataset for balanced sampling.

    Uses inverse frequency weighting with optional smoothing to prevent
    extreme weight values.

    Args:
        dataset_path: Path to training JSONL dataset
        smoothing: Smoothing factor (0-1) to prevent extreme weights
                  0 = no smoothing, 1 = uniform weights

    Returns:
        dict: Mapping from class name to weight

    Example:
        >>> weights = compute_class_weights('/path/to/train.jsonl')
        >>> print(weights)
        {'Plastic_Debris': 0.5, 'Bottle': 1.2, ..., 'Mask': 2.0}
    """
    class_counts = Counter()
    total_detections = 0

    # Count occurrences of each class
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            response_text = sample['messages'][1]['content'][0]['text']

            try:
                detections = json.loads(response_text)
                if isinstance(detections, list):
                    for det in detections:
                        class_counts[det['debris_type']] += 1
                        total_detections += 1
            except (json.JSONDecodeError, KeyError):
                continue

    # Compute inverse frequency weights
    num_classes = len(class_counts)
    class_weights = {}

    for class_name, count in class_counts.items():
        # Inverse frequency with smoothing
        weight = (total_detections / (count * num_classes))
        # Apply smoothing
        weight = (1 - smoothing) * weight + smoothing * 1.0
        class_weights[class_name] = weight

    # Normalize to [0.5, 2.0] range to prevent extreme values
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())

    if max_weight > min_weight:
        normalized_weights = {
            cls: 0.5 + 1.5 * (w - min_weight) / (max_weight - min_weight)
            for cls, w in class_weights.items()
        }
    else:
        normalized_weights = {cls: 1.0 for cls in class_weights}

    return normalized_weights


def print_class_distribution(dataset_path: str) -> None:
    """
    Print class distribution statistics for a dataset.

    Useful for diagnosing class imbalance issues.

    Args:
        dataset_path: Path to dataset JSONL file
    """
    class_counts = Counter()
    total_detections = 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            response_text = sample['messages'][1]['content'][0]['text']

            try:
                detections = json.loads(response_text)
                if isinstance(detections, list):
                    for det in detections:
                        class_counts[det['debris_type']] += 1
                        total_detections += 1
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"\n{'='*60}")
    print(f"Class Distribution Analysis")
    print(f"{'='*60}")
    print(f"Dataset: {Path(dataset_path).name}")
    print(f"Total detections: {total_detections}")
    print(f"\n{'Class':<20} {'Count':<10} {'Percentage':<12} {'Weight':<10}")
    print(f"{'-'*60}")

    # Sort by count (descending)
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_detections) * 100
        weight = total_detections / (count * len(class_counts))
        print(f"{class_name:<20} {count:<10} {percentage:>6.2f}%      {weight:>6.2f}x")

    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count

    print(f"\n{'='*60}")
    print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"  (Most common class is {imbalance_ratio:.1f}x more frequent than rarest)")

    if imbalance_ratio > 10:
        print(f"\n⚠️  SEVERE IMBALANCE DETECTED!")
        print(f"   Recommend using Focal Loss with gamma=2.5-3.0")
    elif imbalance_ratio > 5:
        print(f"\n⚠️  Moderate imbalance detected")
        print(f"   Recommend using Focal Loss with gamma=2.0-2.5")
    else:
        print(f"\n✓  Reasonable class balance")

    print(f"{'='*60}\n")


# Example usage in train.py:
"""
from tools.custom_losses import FocalLoss, DIoULoss, compute_class_weights, print_class_distribution

# Print class distribution
print_class_distribution('/workspace/project/prepared_dataset_docker/train_dataset.jsonl')

# Compute class weights
class_weights = compute_class_weights('/workspace/project/prepared_dataset_docker/train_dataset.jsonl')
print("Class weights:", class_weights)

# Initialize loss functions
focal_loss = FocalLoss(alpha=0.25, gamma=2.5)
diou_loss = DIoULoss()

# Use in training loop
# (Integration depends on your trainer implementation)
"""
