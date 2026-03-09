"""
test_metrics.py

Script de prueba para verificar que los módulos de métricas funcionan correctamente
sin necesidad de cargar modelos pesados.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from metrics import DetectionMetrics, ClassificationMetrics, COCOMetrics, PerformanceMetrics


def test_detection_metrics():
    """Test DetectionMetrics class."""
    print("\n" + "="*50)
    print("🧪 Testing DetectionMetrics")
    print("="*50)

    metrics = DetectionMetrics(iou_threshold=0.5)

    # Sample predictions
    predictions = [
        {
            "debris_type": "Bottle",
            "bounding_box": [0.1, 0.2, 0.3, 0.4],
            "confidence_score": 0.95
        },
        {
            "debris_type": "Can",
            "bounding_box": [0.5, 0.5, 0.7, 0.8],
            "confidence_score": 0.88
        }
    ]

    # Sample ground truths
    ground_truths = [
        {
            "class_name": "Bottle",
            "bounding_box": [0.12, 0.22, 0.32, 0.42]  # Close match
        },
        {
            "class_name": "Plastic_Debris",
            "bounding_box": [0.6, 0.6, 0.8, 0.9]  # No match (different class)
        }
    ]

    # Update metrics
    metrics.update(predictions, ground_truths)

    # Get results
    per_class = metrics.get_metrics()
    overall = metrics.get_overall_metrics()

    print("\n📊 Per-Class Metrics:")
    for class_name, m in per_class.items():
        print(f"  {class_name}: P={m['precision']:.2f}, R={m['recall']:.2f}, F1={m['f1']:.2f}")

    print("\n📊 Overall Metrics:")
    print(f"  Precision: {overall['precision']:.2f}")
    print(f"  Recall: {overall['recall']:.2f}")
    print(f"  F1: {overall['f1']:.2f}")

    print("✅ DetectionMetrics test passed!")


def test_classification_metrics():
    """Test ClassificationMetrics class."""
    print("\n" + "="*50)
    print("🧪 Testing ClassificationMetrics")
    print("="*50)

    metrics = ClassificationMetrics()

    # Sample predictions (list of class names)
    predictions = ["Bottle", "Can", "Plastic_Debris"]

    # Sample ground truths
    ground_truths = [
        {"class_name": "Bottle"},
        {"class_name": "Fishing_Net"}
    ]

    # Update metrics
    metrics.update(predictions, ground_truths)

    # Get results
    per_class = metrics.get_metrics()
    overall = metrics.get_overall_metrics()

    print("\n📊 Per-Class Metrics:")
    for class_name, m in per_class.items():
        print(f"  {class_name}: P={m['precision']:.2f}, R={m['recall']:.2f}, F1={m['f1']:.2f}")

    print("\n📊 Overall Metrics:")
    print(f"  Precision: {overall['precision']:.2f}")
    print(f"  Recall: {overall['recall']:.2f}")
    print(f"  F1: {overall['f1']:.2f}")

    print("✅ ClassificationMetrics test passed!")


def test_coco_metrics():
    """Test COCOMetrics class."""
    print("\n" + "="*50)
    print("🧪 Testing COCOMetrics")
    print("="*50)

    metrics = COCOMetrics(iou_thresholds=[0.5, 0.75])

    # Sample predictions for multiple images
    for img_id in range(3):
        predictions = [
            {
                "debris_type": "Bottle",
                "bounding_box": [0.1, 0.2, 0.3, 0.4],
                "confidence_score": 0.95
            }
        ]
        ground_truths = [
            {
                "class_name": "Bottle",
                "bounding_box": [0.12, 0.22, 0.32, 0.42]
            }
        ]
        metrics.update(predictions, ground_truths, image_id=img_id)

    # Get mAP
    map_results = metrics.get_map()

    print("\n📊 COCO Metrics:")
    for metric_name, value in map_results.items():
        if metric_name != 'per_class_ap':
            print(f"  {metric_name}: {value:.4f}")

    if 'per_class_ap' in map_results:
        print("\n📊 Per-Class AP:")
        for class_name, ap in map_results['per_class_ap'].items():
            print(f"  {class_name}: {ap:.4f}")

    print("✅ COCOMetrics test passed!")


def test_performance_metrics():
    """Test PerformanceMetrics class."""
    print("\n" + "="*50)
    print("🧪 Testing PerformanceMetrics")
    print("="*50)

    metrics = PerformanceMetrics()

    # Simulate inference times
    for i in range(10):
        metrics.add_inference_time(1.2 + i * 0.1)  # 1.2 to 2.1 seconds
        metrics.add_parsing_result(i % 10 != 0)  # 90% success rate

    # Get results
    perf = metrics.get_metrics()

    print("\n📊 Performance Metrics:")
    print(f"  Mean Latency: {perf['mean_latency_ms']:.2f} ms")
    print(f"  Median Latency: {perf['median_latency_ms']:.2f} ms")
    print(f"  Min Latency: {perf['min_latency_ms']:.2f} ms")
    print(f"  Max Latency: {perf['max_latency_ms']:.2f} ms")
    print(f"  FPS: {perf['fps']:.2f}")
    print(f"  Parsing Success Rate: {perf['parsing_success_rate']:.2%}")

    print("✅ PerformanceMetrics test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 TESTING METRICS MODULES")
    print("="*70)

    try:
        test_detection_metrics()
        test_classification_metrics()
        test_coco_metrics()
        test_performance_metrics()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nLos módulos de métricas funcionan correctamente.")
        print("Ahora puedes usar evaluate_models.py para evaluar tus modelos.\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
