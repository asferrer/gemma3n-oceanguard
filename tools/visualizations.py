"""
visualizations.py

Advanced visualization utilities for model evaluation results,
including confusion matrices, PR curves, comparison tables, and HTML reports.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import json
from PIL import Image, ImageDraw, ImageFont


# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str,
    output_path: str,
    metric_names: List[str] = ['precision', 'recall', 'f1']
):
    """
    Plot bar chart comparing metrics across classes.

    Args:
        metrics_dict: Dict mapping class names to metrics
        title: Chart title
        output_path: Path to save figure
        metric_names: List of metric names to plot
    """
    classes = sorted(metrics_dict.keys())
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, metric_name in enumerate(metric_names):
        values = [metrics_dict[cls].get(metric_name, 0) for cls in classes]
        ax.bar(x + i * width, values, width, label=metric_name.capitalize())

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix for classification results.

    Args:
        predictions: List of predicted class names
        ground_truths: List of ground truth class names
        class_names: List of all possible class names
        output_path: Path to save figure
        title: Chart title
    """
    # Create confusion matrix
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for pred, gt in zip(predictions, ground_truths):
        if pred in class_to_idx and gt in class_to_idx:
            confusion[class_to_idx[gt], class_to_idx[pred]] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(
    ap_per_class: Dict[str, float],
    output_path: str,
    title: str = "Average Precision per Class"
):
    """
    Plot Average Precision as a bar chart for each class.

    Args:
        ap_per_class: Dict mapping class names to AP scores
        output_path: Path to save figure
        title: Chart title
    """
    classes = sorted(ap_per_class.keys())
    ap_values = [ap_per_class[cls] for cls in classes]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = sns.color_palette("rocket", len(classes))
    bars = ax.barh(classes, ap_values, color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, ap_values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    ax.set_xlabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(
    base_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
    output_path: str,
    title: str = "Model Performance Comparison"
):
    """
    Create side-by-side comparison of base vs fine-tuned model.

    Args:
        base_metrics: Metrics from base model
        finetuned_metrics: Metrics from fine-tuned model
        output_path: Path to save figure
        title: Chart title
    """
    metric_names = ['precision', 'recall', 'f1', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
    base_values = [base_metrics.get(m, 0) for m in metric_names]
    ft_values = [finetuned_metrics.get(m, 0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model', color='lightcoral')
    bars2 = ax.bar(x + width/2, ft_values, width, label='Fine-tuned Model', color='skyblue')

    # Add improvement percentage on top
    for i, (base, ft) in enumerate(zip(base_values, ft_values)):
        if base > 0:
            improvement = ((ft - base) / base) * 100
            y_pos = max(base, ft) + 0.03
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, y_pos, f'{improvement:+.1f}%', ha='center', fontsize=10,
                   color=color, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_metrics(
    perf_metrics: Dict[str, float],
    output_path: str,
    title: str = "Inference Performance"
):
    """
    Plot performance metrics (latency, throughput, etc.).

    Args:
        perf_metrics: Dict with performance metrics
        output_path: Path to save figure
        title: Chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Latency plot
    latency_metrics = {
        'Mean': perf_metrics.get('mean_latency_ms', 0),
        'Median': perf_metrics.get('median_latency_ms', 0),
        'Min': perf_metrics.get('min_latency_ms', 0),
        'Max': perf_metrics.get('max_latency_ms', 0)
    }

    axes[0].bar(latency_metrics.keys(), latency_metrics.values(), color='teal')
    axes[0].set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Inference Latency', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Throughput and parsing success
    other_metrics = {
        'FPS': perf_metrics.get('fps', 0),
        'Parsing\nSuccess': perf_metrics.get('parsing_success_rate', 0) * 100
    }

    colors = ['orange', 'green']
    axes[1].bar(other_metrics.keys(), other_metrics.values(), color=colors)
    axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Throughput & Parsing Success', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontweight='bold')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_evaluation_dashboard(
    detection_metrics: Dict,
    classification_metrics: Dict,
    coco_metrics: Dict,
    performance_metrics: Dict,
    class_distribution: Dict[str, int],
    output_path: str
):
    """
    Create comprehensive evaluation dashboard with multiple subplots.

    Args:
        detection_metrics: Detection task metrics
        classification_metrics: Classification task metrics
        coco_metrics: COCO mAP metrics
        performance_metrics: Performance metrics
        class_distribution: Ground truth class distribution
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. Detection metrics by class
    ax1 = fig.add_subplot(gs[0, 0])
    classes = sorted(detection_metrics.keys())
    precision = [detection_metrics[c]['precision'] for c in classes]
    recall = [detection_metrics[c]['recall'] for c in classes]
    f1 = [detection_metrics[c]['f1'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    ax1.bar(x - width, precision, width, label='Precision', color='royalblue')
    ax1.bar(x, recall, width, label='Recall', color='limegreen')
    ax1.bar(x + width, f1, width, label='F1-Score', color='tomato')
    ax1.set_ylabel('Score')
    ax1.set_title('Detection Performance (IoU-based)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Classification metrics by class
    ax2 = fig.add_subplot(gs[0, 1])
    cls_classes = sorted(classification_metrics.keys())
    cls_precision = [classification_metrics[c]['precision'] for c in cls_classes]
    cls_recall = [classification_metrics[c]['recall'] for c in cls_classes]
    cls_f1 = [classification_metrics[c]['f1'] for c in cls_classes]

    x2 = np.arange(len(cls_classes))
    ax2.bar(x2 - width, cls_precision, width, label='Precision', color='royalblue')
    ax2.bar(x2, cls_recall, width, label='Recall', color='limegreen')
    ax2.bar(x2 + width, cls_f1, width, label='F1-Score', color='tomato')
    ax2.set_ylabel('Score')
    ax2.set_title('Classification Performance (Presence-based)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cls_classes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Class distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_dist = sorted(class_distribution.items(), key=lambda x: x[1])
    dist_classes = [x[0] for x in sorted_dist]
    dist_counts = [x[1] for x in sorted_dist]
    ax3.barh(dist_classes, dist_counts, color='skyblue')
    ax3.set_xlabel('Number of Instances')
    ax3.set_title('Ground Truth Class Distribution', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', linestyle='--', alpha=0.7)

    # 4. COCO mAP metrics
    ax4 = fig.add_subplot(gs[1, 1])
    map_names = ['mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
    map_values = [coco_metrics.get(m, 0) for m in map_names]
    bars = ax4.bar(map_names, map_values, color=['gold', 'orange', 'orangered'])
    ax4.set_ylabel('Score')
    ax4.set_title('COCO mAP Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels
    for bar, val in zip(bars, map_values):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold')

    # 5. Per-class AP
    ax5 = fig.add_subplot(gs[2, :])
    if 'per_class_ap' in coco_metrics:
        ap_classes = sorted(coco_metrics['per_class_ap'].keys())
        ap_values = [coco_metrics['per_class_ap'][c] for c in ap_classes]
        colors = sns.color_palette("rocket", len(ap_classes))
        bars = ax5.barh(ap_classes, ap_values, color=colors)
        # Add value labels
        for bar, val in zip(bars, ap_values):
            ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)
        ax5.set_xlabel('Average Precision (AP)')
        ax5.set_title('Per-Class Average Precision @ IoU 0.5', fontsize=14, fontweight='bold')
        ax5.set_xlim(0, 1.0)
        ax5.grid(axis='x', linestyle='--', alpha=0.7)

    # 6. Performance metrics
    ax6 = fig.add_subplot(gs[3, 0])
    perf_labels = ['Mean\nLatency\n(ms)', 'Median\nLatency\n(ms)', 'FPS']
    perf_values = [
        performance_metrics.get('mean_latency_ms', 0),
        performance_metrics.get('median_latency_ms', 0),
        performance_metrics.get('fps', 0)
    ]
    ax6.bar(perf_labels, perf_values, color=['teal', 'turquoise', 'orange'])
    ax6.set_ylabel('Value')
    ax6.set_title('Inference Performance', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(perf_values):
        ax6.text(i, v + max(perf_values)*0.02, f'{v:.2f}', ha='center', fontweight='bold')

    # 7. Overall metrics table
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')
    ax7.set_title('Overall Metrics Summary', fontsize=14, fontweight='bold')

    table_data = []
    # Calculate overall metrics
    det_tp = sum(detection_metrics[c]['tp'] for c in detection_metrics)
    det_fp = sum(detection_metrics[c]['fp'] for c in detection_metrics)
    det_fn = sum(detection_metrics[c]['fn'] for c in detection_metrics)
    det_p = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0
    det_r = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0

    cls_tp = sum(classification_metrics[c]['tp'] for c in classification_metrics)
    cls_fp = sum(classification_metrics[c]['fp'] for c in classification_metrics)
    cls_fn = sum(classification_metrics[c]['fn'] for c in classification_metrics)
    cls_p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
    cls_r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
    cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0

    table_data = [
        ['Detection', f"{det_p:.2%}", f"{det_r:.2%}", f"{det_f1:.3f}"],
        ['Classification', f"{cls_p:.2%}", f"{cls_r:.2%}", f"{cls_f1:.3f}"],
        ['mAP@0.5', '', '', f"{coco_metrics.get('mAP@0.5', 0):.3f}"],
        ['mAP@0.5:0.95', '', '', f"{coco_metrics.get('mAP@0.5:0.95', 0):.3f}"]
    ]

    table = ax7.table(cellText=table_data,
                      colLabels=['Task/Metric', 'Precision', 'Recall', 'F1/Score'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    fig.suptitle('Gemma3n Model Evaluation - Comprehensive Report',
                 fontsize=20, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive dashboard saved to: {output_path}")


def draw_bounding_boxes(
    image_path: str,
    predictions: List[Dict],
    ground_truths: List[Dict],
    output_path: str
):
    """
    Draw side-by-side comparison of predictions vs ground truth.

    Args:
        image_path: Path to original image
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        output_path: Path to save comparison image
    """
    original_image = Image.open(image_path).convert("RGB")

    def draw_boxes(image, annotations, color, label_key):
        draw = ImageDraw.Draw(image)
        width, height = image.size
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for ann in annotations:
            box = ann.get('bounding_box')
            label = ann.get(label_key, 'Unknown')
            if not isinstance(box, list) or len(box) != 4:
                continue

            xmin, ymin, xmax, ymax = box
            draw.rectangle(
                [(xmin * width, ymin * height), (xmax * width, ymax * height)],
                outline=color,
                width=4
            )
            text_bbox = draw.textbbox((xmin * width, ymin * height), label, font=font)
            draw.rectangle(
                (text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2),
                fill=color
            )
            draw.text((xmin * width, ymin * height), label, fill="black", font=font)
        return image

    pred_img = draw_boxes(original_image.copy(), predictions, 'cyan', 'debris_type')
    gt_img = draw_boxes(original_image.copy(), ground_truths, 'lime', 'class_name')

    width, height = original_image.size
    title_height = 40
    comparison_img = Image.new('RGB', (width * 2, height + title_height), 'black')

    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 5), "Model Predictions (Cyan)", font=font, fill="cyan")
    draw.text((width + 10, 5), "Ground Truth (Green)", font=font, fill="lime")

    comparison_img.paste(pred_img, (0, title_height))
    comparison_img.paste(gt_img, (width, title_height))

    comparison_img.save(output_path)


def generate_html_report(
    model_name: str,
    results: Dict[str, Any],
    output_path: str
):
    """
    Generate an HTML report with all evaluation results.

    Args:
        model_name: Name of the evaluated model
        results: Dictionary containing all evaluation results
        output_path: Path to save HTML report
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Report - {model_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-box {{
                display: inline-block;
                padding: 15px 25px;
                margin: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                margin-top: 5px;
            }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .medium {{ color: #f39c12; font-weight: bold; }}
            .poor {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌊 OceanGuard AI - Model Evaluation Report</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Date:</strong> {results.get('timestamp', 'N/A')}</p>

            <h2>📊 Overall Performance</h2>
            <div>
    """

    # Add overall metrics boxes
    overall = results.get('overall_metrics', {})
    for metric_name, value in overall.items():
        html_content += f"""
                <div class="metric-box">
                    <div class="metric-label">{metric_name.upper()}</div>
                    <div class="metric-value">{value:.3f}</div>
                </div>
        """

    html_content += """
            </div>

            <h2>🎯 Detection Metrics (IoU-based)</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                </tr>
    """

    # Add detection metrics
    detection = results.get('detection_metrics', {})
    for class_name, metrics in sorted(detection.items()):
        p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        p_class = 'good' if p > 0.7 else 'medium' if p > 0.4 else 'poor'
        r_class = 'good' if r > 0.7 else 'medium' if r > 0.4 else 'poor'
        f1_class = 'good' if f1 > 0.7 else 'medium' if f1 > 0.4 else 'poor'

        html_content += f"""
                <tr>
                    <td><strong>{class_name}</strong></td>
                    <td class="{p_class}">{p:.2%}</td>
                    <td class="{r_class}">{r:.2%}</td>
                    <td class="{f1_class}">{f1:.3f}</td>
                    <td>{metrics['tp']}</td>
                    <td>{metrics['fp']}</td>
                    <td>{metrics['fn']}</td>
                </tr>
        """

    html_content += """
            </table>

            <h2>🏷️ Classification Metrics (Presence-based)</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
    """

    # Add classification metrics
    classification = results.get('classification_metrics', {})
    for class_name, metrics in sorted(classification.items()):
        p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        p_class = 'good' if p > 0.7 else 'medium' if p > 0.4 else 'poor'
        r_class = 'good' if r > 0.7 else 'medium' if r > 0.4 else 'poor'
        f1_class = 'good' if f1 > 0.7 else 'medium' if f1 > 0.4 else 'poor'

        html_content += f"""
                <tr>
                    <td><strong>{class_name}</strong></td>
                    <td class="{p_class}">{p:.2%}</td>
                    <td class="{r_class}">{r:.2%}</td>
                    <td class="{f1_class}">{f1:.3f}</td>
                </tr>
        """

    html_content += """
            </table>

            <h2>⚡ Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
    """

    # Add performance metrics
    performance = results.get('performance_metrics', {})
    perf_display = {
        'mean_latency_ms': 'Mean Latency (ms)',
        'median_latency_ms': 'Median Latency (ms)',
        'fps': 'Frames Per Second',
        'parsing_success_rate': 'JSON Parsing Success Rate'
    }

    for key, label in perf_display.items():
        value = performance.get(key, 0)
        display_value = f"{value:.2f}" if 'rate' not in key else f"{value:.2%}"
        html_content += f"""
                <tr>
                    <td><strong>{label}</strong></td>
                    <td>{display_value}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {output_path}")
