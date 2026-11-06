"""Marine debris detection and counting with Gemma 3n.

Modes:
  detect  Identify debris items, their material and bounding boxes, and save an annotated image.
  count   Count debris items per type and print a JSON summary.

Usage:
  python scripts/inference_gemma.py --mode detect --path image.jpg
  python scripts/inference_gemma.py --mode count --path image1.jpg image2.jpg
"""
import argparse
import torch
from unsloth import FastLanguageModel
from PIL import Image, ImageDraw, ImageFont
import json
import os
from collections import Counter

# Torch settings
torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

def setup_model():
    """Loads Gemma 3n and its tokenizer from Hugging Face."""
    print("Loading Gemma 3n (this can take a few minutes)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

def load_image_from_path(path: str) -> Image.Image:
    """Loads an image from a local path."""
    try:
        image = Image.open(path).convert("RGB")
        print(f"Image loaded from: {path}")
        return image
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def visualize_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draws bounding boxes and labels on an image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    color_map = {
        "plastic": "#FF1493", "metal": "#1E90FF", "fabric": "#32CD32",
        "textile": "#32CD32", "rubber": "#FFD700", "glass": "#00CED1",
        "nylon": "#9400D3", "synthetic": "#9400D3", "fishing": "#FF8C00",
        "default": "#FFFFFF"
    }
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get('bounding_box')
        label = det.get('debris_type', 'Unknown')
        material = det.get('material', 'default').lower()
        if not box or len(box) != 4: continue
        
        color_key = next((key for key in color_map if key in material), "default")
        color = color_map[color_key]

        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        text_bbox = draw.textbbox((xmin * width, ymin * height - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * width, ymin * height - 22), label, fill="black", font=font)
        
    return img_draw

def analyze_seabed_image(model, tokenizer, image_path: str, mode: str = 'detect'):
    """Analyzes a seabed image in the given mode ('detect' or 'count')."""
    print("-" * 50)
    print(f"Analyzing image: {image_path} (mode: {mode})")

    image = load_image_from_path(image_path)
    if image is None:
        return None

    if mode == 'detect':
        prompt_text = (
            "You are a precise, expert marine debris detection system. "
            "Analyze the image and identify ALL man-made waste items. "
            "For each item, provide a TIGHT bounding box. "
            "Return a JSON list of objects. Each object must have 'debris_type', 'material', and 'bounding_box' ([x_min, y_min, x_max, y_max]). "
            "Your response MUST be ONLY the JSON list, without any other text or markdown."
        )
    elif mode == 'count':
        prompt_text = (
            "You are an efficient marine debris counter. Your task is to analyze the image and count all man-made waste items. "
            "Group the items by their type and return a single JSON object where keys are the 'debris_type' and values are the integer count. "
            "Example: {\"Plastic Bottle\": 2, \"Metal Can\": 1, \"Fishing Net\": 1}. "
            "Your response MUST be ONLY the JSON object, without any other text or markdown."
        )
    else:
        print(f"Error: unknown mode '{mode}'. Use 'detect' or 'count'.")
        return None

    prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": prompt_content}]

    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")

        output_tokens = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        print("\n Respuesta del Modelo (JSON crudo):")
        print(response_text)
        
        json_str = response_text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_str)

        if mode == 'detect':
            print(f"\nDetected {len(parsed_json)} objects.")
            annotated_image = visualize_bounding_boxes(image, parsed_json)
            base, _ = os.path.splitext(image_path)
            output_path = f"{base}_analyzed.jpg"
            annotated_image.save(output_path)
            print(f"Annotated image saved to: {output_path}")
            return parsed_json
        
        elif mode == 'count':
            print("\n Resumen de Conteo para esta imagen:")
            if not parsed_json:
                print("No debris found.")
            else:
                for debris_type, count in parsed_json.items():
                    print(f"  - {debris_type}: {count}")
            return parsed_json

    except json.JSONDecodeError:
        print("Error: the model response is not valid JSON.")
    except Exception as e:
        print(f"Inference error: {e}")
    finally:
        torch.cuda.empty_cache()
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Marine debris detection/counting with Gemma 3n.")
    parser.add_argument("--mode", choices=["detect", "count"], default="detect")
    parser.add_argument("--path", nargs="+", required=True, help="One or more image paths.")
    args = parser.parse_args()

    model, tokenizer = setup_model()
    total_counts = Counter()
    for path in args.path:
        result = analyze_seabed_image(model, tokenizer, path, mode=args.mode)
        if args.mode == "count" and result:
            total_counts.update(result)

    if args.mode == "count":
        print("Aggregated counts over all images:")
        for debris_type, count in total_counts.items():
            print(f"  - {debris_type}: {count}")


if __name__ == "__main__":
    main()
