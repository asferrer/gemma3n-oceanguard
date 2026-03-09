"""
metrics.py

Advanced metrics calculation for object detection evaluation,
including COCO-style metrics (mAP) and standard precision/recall/F1.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA: [x_min, y_min, x_max, y_max] normalized coordinates
        boxB: [x_min, y_min, x_max, y_max] normalized coordinates

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Validate box format (min < max)
    if not (boxA[0] < boxA[2] and boxA[1] < boxA[3]):
        return 0.0
    if not (boxB[0] < boxB[2] and boxB[1] < boxB[3]):
        return 0.0

    # Calculate intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Calculate union
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)

    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_giou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate Generalized Intersection over Union (GIoU).
    GIoU is better than IoU for non-overlapping boxes and optimization.

    Args:
        boxA: [x_min, y_min, x_max, y_max] normalized coordinates
        boxB: [x_min, y_min, x_max, y_max] normalized coordinates

    Returns:
        GIoU score (-1.0 to 1.0)
    """
    # Validate box format
    if not (boxA[0] < boxA[2] and boxA[1] < boxA[3]):
        return -1.0
    if not (boxB[0] < boxB[2] and boxB[1] < boxB[3]):
        return -1.0

    # Calculate IoU
    iou = calculate_iou(boxA, boxB)

    # Calculate smallest enclosing box
    xA_min = min(boxA[0], boxB[0])
    yA_min = min(boxA[1], boxB[1])
    xB_max = max(boxA[2], boxB[2])
    yB_max = max(boxA[3], boxB[3])

    enclosing_area = (xB_max - xA_min) * (yB_max - yA_min)

    # Calculate areas
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Union area
    union_area = boxA_area + boxB_area - (iou * min(boxA_area, boxB_area))

    # GIoU formula: IoU - (enclosing - union) / enclosing
    if enclosing_area > 0:
        giou = iou - (enclosing_area - union_area) / enclosing_area
        return giou
    else:
        return iou


def categorize_object_size(bbox: List[float]) -> str:
    """
    Categorize object by bounding box area (COCO-style).

    Args:
        bbox: [x_min, y_min, x_max, y_max] normalized coordinates

    Returns:
        Size category: 'small', 'medium', or 'large'
    """
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    if area < 0.01:  # < 1% of image
        return "small"
    elif area < 0.05:  # < 5% of image
        return "medium"
    else:
        return "large"


class DetectionMetrics:
    """
    Calculate comprehensive detection metrics including:
    - Precision, Recall, F1 per class and overall
    - True Positives, False Positives, False Negatives
    - IoU-based matching
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def update(self, predictions: List[Dict], ground_truths: List[Dict]):
        """
        Update metrics with predictions and ground truths for a single image.

        Args:
            predictions: List of dicts with 'debris_type' and 'bounding_box'
            ground_truths: List of dicts with 'class_name' and 'bounding_box'
        """
        gt_matched = [False] * len(ground_truths)

        for pred in predictions:
            pred_box = pred.get("bounding_box")
            pred_class = pred.get("debris_type", "Unknown")

            if not isinstance(pred_box, list) or len(pred_box) != 4:
                self.stats[pred_class]["fp"] += 1
                continue

            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(ground_truths):
                if gt["class_name"].lower() == pred_class.lower():
                    iou = calculate_iou(pred_box, gt["bounding_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= self.iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                self.stats[ground_truths[best_gt_idx]["class_name"]]["tp"] += 1
                gt_matched[best_gt_idx] = True
            else:
                self.stats[pred_class]["fp"] += 1

        for j, gt in enumerate(ground_truths):
            if not gt_matched[j]:
                self.stats[gt["class_name"]]["fn"] += 1

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 for each class.

        Returns:
            Dict mapping class names to metrics
        """
        metrics = {}
        for class_name, s in self.stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        return metrics

    def get_overall_metrics(self) -> Dict[str, float]:
        """Get overall (macro-averaged) metrics."""
        total_tp = sum(s["tp"] for s in self.stats.values())
        total_fp = sum(s["fp"] for s in self.stats.values())
        total_fn = sum(s["fn"] for s in self.stats.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }


class ClassificationMetrics:
    """
    Calculate classification metrics based on class presence (not localization).
    """

    def __init__(self):
        self.stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def update(self, predictions: List[str], ground_truths: List[Dict]):
        """
        Update metrics with predictions and ground truths for a single image.

        Args:
            predictions: List of predicted class names
            ground_truths: List of dicts with 'class_name'
        """
        gt_classes = set(ann['class_name'] for ann in ground_truths)
        pred_classes = set(cls.strip().title() for cls in predictions if isinstance(cls, str))

        tp_classes = gt_classes.intersection(pred_classes)
        fp_classes = pred_classes.difference(gt_classes)
        fn_classes = gt_classes.difference(pred_classes)

        for cls in tp_classes:
            self.stats[cls]["tp"] += 1
        for cls in fp_classes:
            self.stats[cls]["fp"] += 1
        for cls in fn_classes:
            self.stats[cls]["fn"] += 1

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 for each class."""
        metrics = {}
        for class_name, s in self.stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        return metrics

    def get_overall_metrics(self) -> Dict[str, float]:
        """Get overall (macro-averaged) metrics."""
        total_tp = sum(s["tp"] for s in self.stats.values())
        total_fp = sum(s["fp"] for s in self.stats.values())
        total_fn = sum(s["fn"] for s in self.stats.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }


class COCOMetrics:
    """
    Calculate COCO-style metrics including mAP@[0.5, 0.75, 0.5:0.95].
    """

    def __init__(self, iou_thresholds: List[float] = None):
        if iou_thresholds is None:
            # Standard COCO IoU thresholds: 0.5 to 0.95 in steps of 0.05
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

        # Store all predictions and ground truths across images
        self.all_predictions = []
        self.all_ground_truths = []

    def update(self, predictions: List[Dict], ground_truths: List[Dict], image_id: int = None):
        """
        Store predictions and ground truths for an image.

        Args:
            predictions: List of dicts with 'debris_type', 'bounding_box', 'confidence_score'
            ground_truths: List of dicts with 'class_name', 'bounding_box'
            image_id: Optional image identifier
        """
        for pred in predictions:
            if isinstance(pred.get("bounding_box"), list) and len(pred["bounding_box"]) == 4:
                self.all_predictions.append({
                    'class': pred.get('debris_type', 'Unknown'),
                    'bbox': pred['bounding_box'],
                    'score': pred.get('confidence_score', 1.0),
                    'image_id': image_id
                })

        for gt in ground_truths:
            if isinstance(gt.get("bounding_box"), list) and len(gt["bounding_box"]) == 4:
                self.all_ground_truths.append({
                    'class': gt['class_name'],
                    'bbox': gt['bounding_box'],
                    'image_id': image_id
                })

    def calculate_ap(self, class_name: str, iou_threshold: float) -> float:
        """
        Calculate Average Precision (AP) for a single class at a given IoU threshold.

        Args:
            class_name: The class to evaluate
            iou_threshold: IoU threshold for matching

        Returns:
            Average Precision score
        """
        # Filter predictions and ground truths for this class
        class_preds = [p for p in self.all_predictions if p['class'] == class_name]
        class_gts = [g for g in self.all_ground_truths if g['class'] == class_name]

        if len(class_gts) == 0:
            return 0.0

        if len(class_preds) == 0:
            return 0.0

        # Sort predictions by confidence score (descending)
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)

        # Track which ground truths have been matched
        gt_matched = {i: False for i in range(len(class_gts))}

        tp = []
        fp = []

        for pred in class_preds:
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt in enumerate(class_gts):
                # Only match ground truths from the same image
                if pred.get('image_id') != gt.get('image_id'):
                    continue

                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # Calculate precision and recall at each threshold
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        return ap

    def get_map(self) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) at different IoU thresholds.

        Returns:
            Dict with mAP@0.5, mAP@0.75, mAP@0.5:0.95
        """
        # Get all unique classes
        all_classes = set(p['class'] for p in self.all_predictions)
        all_classes.update(g['class'] for g in self.all_ground_truths)

        results = {}

        # Calculate AP for each class at each IoU threshold
        ap_per_class_per_iou = defaultdict(dict)
        for class_name in all_classes:
            for iou_thresh in self.iou_thresholds:
                ap = self.calculate_ap(class_name, iou_thresh)
                ap_per_class_per_iou[class_name][iou_thresh] = ap

        # mAP@0.5
        if 0.5 in self.iou_thresholds:
            ap_50 = [ap_per_class_per_iou[cls].get(0.5, 0) for cls in all_classes]
            results['mAP@0.5'] = np.mean(ap_50) if ap_50 else 0.0

        # mAP@0.75
        if 0.75 in self.iou_thresholds:
            ap_75 = [ap_per_class_per_iou[cls].get(0.75, 0) for cls in all_classes]
            results['mAP@0.75'] = np.mean(ap_75) if ap_75 else 0.0

        # mAP@0.5:0.95 (average over all IoU thresholds)
        all_aps = []
        for class_name in all_classes:
            class_aps = [ap_per_class_per_iou[class_name].get(iou, 0) for iou in self.iou_thresholds]
            all_aps.append(np.mean(class_aps))
        results['mAP@0.5:0.95'] = np.mean(all_aps) if all_aps else 0.0

        # Per-class AP at IoU 0.5
        results['per_class_ap'] = {cls: ap_per_class_per_iou[cls].get(0.5, 0) for cls in all_classes}

        return results


class PerformanceMetrics:
    """
    Track inference performance metrics (latency, throughput).
    """

    def __init__(self):
        self.inference_times = []
        self.parsing_successes = 0
        self.parsing_failures = 0

    def add_inference_time(self, time_seconds: float):
        """Record inference time for a single image."""
        self.inference_times.append(time_seconds)

    def add_parsing_result(self, success: bool):
        """Record whether JSON parsing was successful."""
        if success:
            self.parsing_successes += 1
        else:
            self.parsing_failures += 1

    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance statistics."""
        if not self.inference_times:
            return {
                'mean_latency_ms': 0,
                'median_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0,
                'fps': 0,
                'parsing_success_rate': 0
            }

        times_ms = [t * 1000 for t in self.inference_times]
        total_parsing = self.parsing_successes + self.parsing_failures

        return {
            'mean_latency_ms': np.mean(times_ms),
            'median_latency_ms': np.median(times_ms),
            'min_latency_ms': np.min(times_ms),
            'max_latency_ms': np.max(times_ms),
            'fps': 1.0 / np.mean(self.inference_times),
            'parsing_success_rate': self.parsing_successes / total_parsing if total_parsing > 0 else 0
        }
