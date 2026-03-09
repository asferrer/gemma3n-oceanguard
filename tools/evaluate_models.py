"""
evaluate_models.py

Comprehensive evaluation script for Gemma3n marine debris detection models.
Supports evaluation of fine-tuned models with advanced metrics including:
- Detection metrics (IoU-based Precision/Recall/F1)
- Classification metrics (Presence-based)
- COCO-style metrics (mAP@0.5, mAP@0.75, mAP@0.5:0.95)
- Performance metrics (latency, throughput)
- Model comparison (base vs fine-tuned)

Usage:
    python tools/evaluate_models.py --model-path outputs/gemma3n/runs/best_model
    python tools/evaluate_models.py --compare --base-model unsloth/gemma-3n-e2b-it
    python tools/evaluate_models.py --test-set prepared_dataset/test_dataset.jsonl
"""

import torch
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from PIL import Image
from unsloth import FastLanguageModel
from datetime import datetime

# Import custom modules
from metrics import DetectionMetrics, ClassificationMetrics, COCOMetrics, PerformanceMetrics
from visualizations import (
    create_evaluation_dashboard,
    plot_model_comparison,
    draw_bounding_boxes,
    generate_html_report
)

# Disable torch dynamo for stability
torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999


class ModelEvaluator:
    """Main evaluator class for Gemma3n models."""

    def __init__(
        self,
        model_path: str,
        test_set_path: str = None,
        iou_thresholds: List[float] = None,
        output_dir: str = "evaluation_results",
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.test_set_path = test_set_path or "prepared_dataset/test_dataset.jsonl"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize metrics
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.iou_thresholds = iou_thresholds

        self.detection_metrics = DetectionMetrics(iou_threshold=0.5)
        self.classification_metrics = ClassificationMetrics()
        self.coco_metrics = COCOMetrics(iou_thresholds=self.iou_thresholds)
        self.performance_metrics = PerformanceMetrics()

        # Load model
        print(f"🚀 Loading model from: {model_path}")
        self.model, self.tokenizer = self._load_model()

        # Valid debris classes
        self.valid_classes = [
            "Bottle", "Can", "Fishing_Net", "Glove",
            "Mask", "Metal_Debris", "Plastic_Debris", "Tire"
        ]

    def _load_model(self) -> Tuple:
        """Load the Gemma3n model and tokenizer."""
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=4096,
                dtype=torch.float16,
                load_in_4bit=True,
                device_map={'': 0}
            )
            FastLanguageModel.for_inference(model)
            print("✅ Model loaded successfully")
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _get_detection_prompt(self) -> str:
        """Get the detection task prompt."""
        class_list_str = ", ".join([f"'{cls}'" for cls in self.valid_classes])
        return (
            "You are a precise, pixel-perfect marine debris detection system. "
            "Analyze the image and identify ALL man-made waste items. "
            "Return a JSON list of objects. Each object must have:\n"
            f"1. 'debris_type': The specific type of object. It MUST be one of the following: [{class_list_str}].\n"
            "2. 'material': The likely material (e.g., 'Plastic', 'Metal').\n"
            "3. 'confidence_score': A float from 0.0 to 1.0 indicating your certainty.\n"
            "4. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max]. "
            "CRITICAL: The bounding box must be extremely precise, tightly enclosing the object with no extra padding. "
            "Do not include background, water, or shadows. Aim for pixel-perfect accuracy.\n"
            "Your response MUST be ONLY the JSON list."
        )

    def _get_classification_prompt(self) -> str:
        """Get the classification task prompt."""
        class_list_str = ", ".join([f"'{cls}'" for cls in self.valid_classes])
        return (
            "You are an expert marine debris classifier. "
            "Analyze the image and identify all types of man-made waste present. "
            "Return a JSON list containing the names of the debris types you see. "
            f"Each name MUST be one of the following: [{class_list_str}].\n"
            "Example: [\"Bottle\", \"Fishing_Net\", \"Can\"]\n"
            "Your response MUST be ONLY the JSON list."
        )

    @torch.no_grad()
    def _run_inference(self, image_path: str, prompt_text: str) -> Tuple[Any, str, bool]:
        """
        Run inference on a single image.

        Returns:
            (predictions, raw_response, parsing_success)
        """
        start_time = time.time()
        raw_response = "Error during image processing."
        parsing_success = False

        try:
            image = Image.open(image_path).convert("RGB")
            prompt_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
            messages = [{"role": "user", "content": prompt_content}]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1
            )

            raw_response = self.tokenizer.batch_decode(
                output_tokens[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            # Parse JSON response
            match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = raw_response.strip().replace("```json", "").replace("```", "")

            predictions = json.loads(json_str)
            parsing_success = True

        except Exception as e:
            raw_response += f"\n[PARSING ERROR]: {e}"
            predictions = []
            parsing_success = False

        inference_time = time.time() - start_time
        self.performance_metrics.add_inference_time(inference_time)
        self.performance_metrics.add_parsing_result(parsing_success)

        return predictions, raw_response, parsing_success

    def _load_test_set(self) -> List[Dict]:
        """Load test dataset from JSONL file."""
        print(f"📖 Loading test set from: {self.test_set_path}")

        test_samples = []
        with open(self.test_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                test_samples.append(sample)

        print(f"✅ Loaded {len(test_samples)} test samples")
        return test_samples

    def _parse_ground_truth(self, sample: Dict) -> Tuple[str, List[Dict]]:
        """
        Parse ground truth from test sample.

        Returns:
            (image_path, list of ground truth annotations)
        """
        messages = sample.get('messages', [])

        # Extract image path from user message
        image_path = None
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg.get('content', []):
                    if content.get('type') == 'image':
                        image_path = content.get('image')
                        break

        # Extract ground truth from assistant message
        ground_truths = []
        for msg in messages:
            if msg['role'] == 'assistant':
                for content in msg.get('content', []):
                    if content.get('type') == 'text':
                        try:
                            gt_data = json.loads(content['text'])
                            if isinstance(gt_data, list):
                                for item in gt_data:
                                    ground_truths.append({
                                        'class_name': item.get('debris_type', 'Unknown'),
                                        'bounding_box': item.get('bounding_box', []),
                                        'material': item.get('material', 'Unknown')
                                    })
                        except:
                            pass

        return image_path, ground_truths

    def evaluate(self, max_samples: int = None, save_visuals: bool = False) -> Dict[str, Any]:
        """
        Run complete evaluation on test set.

        Args:
            max_samples: Maximum number of samples to evaluate (None for all)
            save_visuals: Whether to save visual comparisons

        Returns:
            Dictionary containing all evaluation results
        """
        print("\n" + "="*60)
        print("🔬 Starting Comprehensive Model Evaluation")
        print("="*60)

        test_samples = self._load_test_set()
        if max_samples:
            test_samples = test_samples[:max_samples]
            print(f"⚠️  Evaluating only first {max_samples} samples")

        # Track class distribution
        class_distribution = defaultdict(int)

        # Create visuals directory if needed
        if save_visuals:
            visuals_dir = self.output_dir / "visual_comparisons"
            visuals_dir.mkdir(exist_ok=True)
            print(f"🎨 Visual comparisons will be saved to: {visuals_dir}")

        print(f"\n🔍 Processing {len(test_samples)} test samples...")

        for idx, sample in enumerate(test_samples):
            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx + 1}/{len(test_samples)} samples")

            # Parse ground truth
            image_path, ground_truths = self._parse_ground_truth(sample)

            if not image_path or not os.path.exists(image_path):
                continue

            # Update class distribution
            for gt in ground_truths:
                class_distribution[gt['class_name']] += 1

            # Run detection inference
            detection_prompt = self._get_detection_prompt()
            detection_preds, _, _ = self._run_inference(image_path, detection_prompt)

            # Normalize prediction format
            for det in detection_preds:
                if isinstance(det, dict) and 'debris_type' in det:
                    det['debris_type'] = det['debris_type'].strip().title()

            # Run classification inference
            classification_prompt = self._get_classification_prompt()
            classification_preds, _, _ = self._run_inference(image_path, classification_prompt)

            # Update metrics
            self.detection_metrics.update(detection_preds, ground_truths)
            self.classification_metrics.update(classification_preds, ground_truths)
            self.coco_metrics.update(detection_preds, ground_truths, image_id=idx)

            # Save visual comparison
            if save_visuals and idx < 20:  # Save first 20 for inspection
                output_path = visuals_dir / f"comparison_{idx:04d}.jpg"
                try:
                    draw_bounding_boxes(image_path, detection_preds, ground_truths, str(output_path))
                except Exception as e:
                    print(f"  ⚠️  Could not save visual for sample {idx}: {e}")

        print("✅ Evaluation complete!\n")

        # Compile all results
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'test_set_path': self.test_set_path,
            'num_samples': len(test_samples),
            'detection_metrics': self.detection_metrics.get_metrics(),
            'detection_overall': self.detection_metrics.get_overall_metrics(),
            'classification_metrics': self.classification_metrics.get_metrics(),
            'classification_overall': self.classification_metrics.get_overall_metrics(),
            'coco_metrics': self.coco_metrics.get_map(),
            'performance_metrics': self.performance_metrics.get_metrics(),
            'class_distribution': dict(class_distribution)
        }

        # Add to overall metrics for easy access
        results['overall_metrics'] = {
            'detection_f1': results['detection_overall']['f1'],
            'classification_f1': results['classification_overall']['f1'],
            'mAP@0.5': results['coco_metrics']['mAP@0.5'],
            'mAP@0.75': results['coco_metrics']['mAP@0.75'],
            'mAP@0.5:0.95': results['coco_metrics']['mAP@0.5:0.95']
        }

        return results

    def save_results(self, results: Dict[str, Any], prefix: str = ""):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        json_path = self.output_dir / f"{prefix}evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {json_path}")

        # Save markdown summary
        md_path = self.output_dir / f"{prefix}evaluation_summary_{timestamp}.md"
        self._save_markdown_summary(results, md_path)
        print(f"📝 Summary saved to: {md_path}")

        # Generate comprehensive dashboard
        dashboard_path = self.output_dir / f"{prefix}evaluation_dashboard_{timestamp}.png"
        create_evaluation_dashboard(
            detection_metrics=results['detection_metrics'],
            classification_metrics=results['classification_metrics'],
            coco_metrics=results['coco_metrics'],
            performance_metrics=results['performance_metrics'],
            class_distribution=results['class_distribution'],
            output_path=str(dashboard_path)
        )

        # Generate HTML report
        html_path = self.output_dir / f"{prefix}evaluation_report_{timestamp}.html"
        generate_html_report(
            model_name=self.model_path,
            results=results,
            output_path=str(html_path)
        )

    def _save_markdown_summary(self, results: Dict[str, Any], output_path: Path):
        """Save a markdown summary of results."""
        content = f"""# Gemma3n Model Evaluation Report

**Model:** `{results['model_path']}`
**Test Set:** `{results['test_set_path']}`
**Samples Evaluated:** {results['num_samples']}
**Timestamp:** {results['timestamp']}

---

## 📊 Overall Metrics

| Metric | Value |
|--------|-------|
| Detection F1 | {results['overall_metrics']['detection_f1']:.4f} |
| Classification F1 | {results['overall_metrics']['classification_f1']:.4f} |
| mAP@0.5 | {results['overall_metrics']['mAP@0.5']:.4f} |
| mAP@0.75 | {results['overall_metrics']['mAP@0.75']:.4f} |
| mAP@0.5:0.95 | {results['overall_metrics']['mAP@0.5:0.95']:.4f} |

---

## 🎯 Detection Metrics (IoU-based)

**Overall:**
- Precision: {results['detection_overall']['precision']:.2%}
- Recall: {results['detection_overall']['recall']:.2%}
- F1-Score: {results['detection_overall']['f1']:.4f}

**Per-Class:**

| Class | Precision | Recall | F1-Score | TP | FP | FN |
|-------|-----------|--------|----------|----|----|---- |
"""
        for class_name, metrics in sorted(results['detection_metrics'].items()):
            content += f"| {class_name} | {metrics['precision']:.2%} | {metrics['recall']:.2%} | {metrics['f1']:.4f} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} |\n"

        content += f"""
---

## 🏷️ Classification Metrics (Presence-based)

**Overall:**
- Precision: {results['classification_overall']['precision']:.2%}
- Recall: {results['classification_overall']['recall']:.2%}
- F1-Score: {results['classification_overall']['f1']:.4f}

**Per-Class:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""
        for class_name, metrics in sorted(results['classification_metrics'].items()):
            content += f"| {class_name} | {metrics['precision']:.2%} | {metrics['recall']:.2%} | {metrics['f1']:.4f} |\n"

        content += f"""
---

## 🎖️ COCO Metrics (mAP)

| Metric | Value |
|--------|-------|
| mAP@0.5 | {results['coco_metrics']['mAP@0.5']:.4f} |
| mAP@0.75 | {results['coco_metrics']['mAP@0.75']:.4f} |
| mAP@0.5:0.95 | {results['coco_metrics']['mAP@0.5:0.95']:.4f} |

**Per-Class AP @ IoU 0.5:**

| Class | AP |
|-------|----|
"""
        for class_name, ap in sorted(results['coco_metrics']['per_class_ap'].items()):
            content += f"| {class_name} | {ap:.4f} |\n"

        content += f"""
---

## ⚡ Performance Metrics

| Metric | Value |
|--------|-------|
| Mean Latency | {results['performance_metrics']['mean_latency_ms']:.2f} ms |
| Median Latency | {results['performance_metrics']['median_latency_ms']:.2f} ms |
| FPS | {results['performance_metrics']['fps']:.2f} |
| Parsing Success Rate | {results['performance_metrics']['parsing_success_rate']:.2%} |

---

## 📈 Class Distribution (Ground Truth)

| Class | Count |
|-------|-------|
"""
        for class_name, count in sorted(results['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            content += f"| {class_name} | {count} |\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for Gemma3n marine debris detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a fine-tuned model
  python tools/evaluate_models.py --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1

  # Evaluate with custom test set
  python tools/evaluate_models.py --model-path my_model --test-set custom_test.jsonl

  # Limit samples and save visual comparisons
  python tools/evaluate_models.py --model-path my_model --max-samples 100 --save-visuals

  # Compare two models
  python tools/evaluate_models.py --model-path fine_tuned_model --compare --base-model unsloth/gemma-3n-e2b-it
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model to evaluate (local path or HuggingFace model ID)'
    )

    parser.add_argument(
        '--test-set',
        type=str,
        default='prepared_dataset_docker/test_dataset.jsonl',
        help='Path to test dataset JSONL file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    parser.add_argument(
        '--save-visuals',
        action='store_true',
        help='Save visual comparisons of predictions vs ground truth'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with a base model'
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default='unsloth/gemma-3n-e2b-it',
        help='Base model for comparison (used with --compare)'
    )

    parser.add_argument(
        '--iou-thresholds',
        type=str,
        default='0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95',
        help='Comma-separated IoU thresholds for COCO metrics'
    )

    args = parser.parse_args()

    # Parse IoU thresholds
    iou_thresholds = [float(x) for x in args.iou_thresholds.split(',')]

    print("\n" + "="*70)
    print("🌊 GEMMA3N MODEL EVALUATION - OCEANGUARD AI")
    print("="*70)

    # Evaluate main model
    print("\n📍 Evaluating main model...")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        test_set_path=args.test_set,
        iou_thresholds=iou_thresholds,
        output_dir=args.output_dir
    )

    results = evaluator.evaluate(
        max_samples=args.max_samples,
        save_visuals=args.save_visuals
    )

    evaluator.save_results(results, prefix="main_")

    # Print summary
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"Detection F1:      {results['overall_metrics']['detection_f1']:.4f}")
    print(f"Classification F1: {results['overall_metrics']['classification_f1']:.4f}")
    print(f"mAP@0.5:          {results['overall_metrics']['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95:     {results['overall_metrics']['mAP@0.5:0.95']:.4f}")
    print(f"Mean Latency:      {results['performance_metrics']['mean_latency_ms']:.2f} ms")
    print(f"FPS:               {results['performance_metrics']['fps']:.2f}")
    print("="*70)

    # Compare with base model if requested
    if args.compare:
        print(f"\n📍 Evaluating base model for comparison: {args.base_model}")
        base_evaluator = ModelEvaluator(
            model_path=args.base_model,
            test_set_path=args.test_set,
            iou_thresholds=iou_thresholds,
            output_dir=args.output_dir
        )

        base_results = base_evaluator.evaluate(
            max_samples=args.max_samples,
            save_visuals=False
        )

        base_evaluator.save_results(base_results, prefix="base_")

        # Generate comparison plot
        print("\n📊 Generating comparison plot...")
        comparison_path = Path(args.output_dir) / "model_comparison.png"
        plot_model_comparison(
            base_metrics=base_results['overall_metrics'],
            finetuned_metrics=results['overall_metrics'],
            output_path=str(comparison_path),
            title="Fine-tuned vs Base Model Performance"
        )

        # Print comparison
        print("\n" + "="*70)
        print("🔍 MODEL COMPARISON")
        print("="*70)
        print(f"{'Metric':<20} | {'Base Model':>15} | {'Fine-tuned':>15} | {'Improvement':>15}")
        print("-"*70)

        for metric in ['detection_f1', 'classification_f1', 'mAP@0.5', 'mAP@0.5:0.95']:
            base_val = base_results['overall_metrics'][metric]
            ft_val = results['overall_metrics'][metric]
            improvement = ((ft_val - base_val) / base_val * 100) if base_val > 0 else 0
            print(f"{metric:<20} | {base_val:>15.4f} | {ft_val:>15.4f} | {improvement:>14.2f}%")

        print("="*70)

    print("\n✅ Evaluation complete! Check the output directory for detailed results.")
    print(f"📁 Output directory: {args.output_dir}\n")


if __name__ == "__main__":
    main()
