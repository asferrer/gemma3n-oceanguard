# inference_bbox.py
# Robust VLM inference for text-JSON bounding boxes with Gemma 3N Vision (Unsloth).
# - Supports merged full model OR base + adapters (LoRA).
# - Batched or single-image inference.
# - Safe JSON parsing, schema validation, clamping to [0,1].
# - Visualization overlays + per-image results.json.

import os
import re
import json
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

# Unsloth APIs
from unsloth import FastLanguageModel
try:
    # Optional helper for PIL loading and platform specifics
    from unsloth import to_image as unsloth_to_image
except Exception:
    unsloth_to_image = None


# ---------- Utilities ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_pil(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def draw_detections(img: Image.Image,
                    detections: List[Dict[str, Any]],
                    out_path: str,
                    class_color_map: Optional[Dict[str, str]] = None) -> None:
    """Draw normalized [0..1] boxes onto a copy and save."""
    im = img.copy()
    W, H = im.size
    d = ImageDraw.Draw(im)

    # consistent colors
    palette = [
        "#3b82f6", "#22c55e", "#ef4444", "#a855f7", "#eab308",
        "#06b6d4", "#f97316", "#10b981", "#f43f5e", "#94a3b8"
    ]
    def color_for(cls: str) -> str:
        if class_color_map and cls in class_color_map:
            return class_color_map[cls]
        idx = (abs(hash(cls)) % len(palette))
        return palette[idx]

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bounding_box")
        label = det.get("debris_type", "Unknown")
        mat   = det.get("material", "")
        conf  = det.get("confidence_score", None)

        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = bbox
        x1p, y1p, x2p, y2p = x1 * W, y1 * H, x2 * W, y2 * H
        col = color_for(label)

        d.rectangle([(x1p, y1p), (x2p, y2p)], outline=col, width=3)
        tag = f"{label}"
        if mat:
            tag += f" | {mat}"
        if isinstance(conf, (int, float)):
            tag += f" | {conf:.2f}"

        tw, th = d.textlength(tag, font=font), 18
        tx, ty = int(x1p), int(max(0, y1p - th - 4))
        d.rectangle([(tx, ty), (tx + tw + 6, ty + th)], fill=col)
        d.text((tx + 3, ty), tag, fill="black", font=font)

    im.save(out_path)


def safe_float(x) -> Optional[float]:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def validate_and_clamp_detection(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ensure required keys, clamp bbox to [0,1], discard malformed entries."""
    if not isinstance(d, dict):
        return None
    bbox = d.get("bounding_box")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None

    coords = []
    for c in bbox:
        cf = safe_float(c)
        if cf is None:
            return None
        coords.append(clamp01(cf))

    x1, y1, x2, y2 = coords
    # enforce x1<=x2, y1<=y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    out = {
        "debris_type": d.get("debris_type", "Unknown"),
        "material": d.get("material", "Unknown"),
        "confidence_score": float(d.get("confidence_score", 1.0)) if isinstance(d.get("confidence_score", 1.0), (int, float)) else 1.0,
        "bounding_box": [x1, y1, x2, y2],
    }
    return out


def extract_first_json_array(text: str) -> Optional[List[Any]]:
    """
    Extract the first top-level JSON array using a bracket balancing approach,
    then json.loads it. Robust against extra prose or code fences.
    """
    if not text:
        return None

    # Quick regex for a JSON array; fallback to bracket scan if fails
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Bracket balancing (first array)
    open_idx = text.find("[")
    while open_idx != -1:
        depth = 0
        for i in range(open_idx, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[open_idx:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        open_idx = text.find("[", open_idx + 1)
    return None


# ---------- Model I/O ----------
def load_model(
    merged_dir: Optional[str],
    base: Optional[str],
    adapters: Optional[str],
    dtype: str = "bf16",
    load_in_4bit: bool = True,
    attn_impl: str = "auto",
    max_seq_len: int = 4096,
) -> Tuple[Any, Any, str]:
    """
    Load either a merged model (preferred for inference) or base + adapters.
    Returns (model, tokenizer, mode_str).
    """
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "eager"

    torch_dtype = None
    if dtype:
        d = dtype.lower()
        if d in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif d in ("fp16", "float16", "half"):
            torch_dtype = torch.float16
        elif d in ("fp32", "float32"):
            torch_dtype = torch.float32

    if merged_dir:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = merged_dir,
            max_seq_length = max_seq_len,
            dtype          = torch_dtype,
            load_in_4bit   = False,            # merged full-precision weights typically
            fast_inference = True,
            attn_implementation = attn_impl,
        )
        return model, tokenizer, "merged"
    else:
        if not base or not adapters:
            raise ValueError("If --merged_dir is not provided, you must pass --base and --adapters.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = base,
            max_seq_length = max_seq_len,
            dtype          = torch_dtype,
            load_in_4bit   = load_in_4bit,
            fast_inference = False,
            attn_implementation = attn_impl,
        )
        # attach adapters (LoRA) from training run dir
        model = FastLanguageModel.load_peft_adapter(model, adapters)
        return model, tokenizer, "adapters"


def build_messages(image_path: str, instruction: str) -> List[Dict[str, Any]]:
    """
    Messages structure aligned with your training data.
    """
    # Use Unsloth's to_image if available (helps device conversion)
    img_field = image_path
    if unsloth_to_image is not None:
        try:
            img_field = unsloth_to_image(image_path)
        except Exception:
            img_field = image_path

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": img_field},
            ],
        }
    ]
    return msgs


@torch.inference_mode()
def generate_for_image(
    model,
    tokenizer,
    image_path: str,
    instruction: str,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """
    Build inputs for VLM and run .generate to get text.
    Attempts robust handling of Unsloth vision encoding.
    """
    messages = build_messages(image_path, instruction)
    # Tokenize with chat template; Unsloth will inspect 'content' for images
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Move to device and collect vision tensors if present
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # inputs may contain pixel tensors or separate image refs internally;
    # send everything we got to device.
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
    else:
        inputs = inputs.to(device)

    model = model.to(device)
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0.0),
    )

    # decode new tokens only
    text = tokenizer.decode(gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def parse_and_validate(text: str) -> List[Dict[str, Any]]:
    """
    Returns a cleaned list of detections (each with debris_type, material, confidence_score, bounding_box).
    """
    arr = extract_first_json_array(text)
    if not isinstance(arr, list):
        return []
    cleaned = []
    for d in arr:
        vd = validate_and_clamp_detection(d)
        if vd:
            cleaned.append(vd)
    return cleaned


# ---------- CLI ----------
DEFAULT_INSTRUCTION = (
    "You are a precise, pixel-perfect marine debris detection system. "
    "Analyze the image and identify ALL man-made waste items. "
    "Return ONLY a JSON list of detections. Each detection must include:\n"
    "  - 'debris_type' (one of: ['Basket','Bottle','Can','CarBumper','FishingNet','Glove',"
    "'MetalDebris','PackagingBag','Pipe','PlasticBag','PlasticDebris','Rope','Shoe','SquaredCan','Tire','Towel','Wood'])\n"
    "  - 'material' (e.g., 'Plastic', 'Metal')\n"
    "  - 'confidence_score' (0.0–1.0)\n"
    "  - 'bounding_box' ([x_min, y_min, x_max, y_max] normalized in [0,1])\n"
    "Boxes must be extremely tight."
)

def main():
    ap = argparse.ArgumentParser(description="Inference: Gemma3N VLM → JSON bounding boxes")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--merged_dir", type=str, help="Path to merged model (preferred for inference).")
    m.add_argument("--base", type=str, help="Base model ID (e.g., unsloth/gemma-3n-E4B-it). Use with --adapters.")
    ap.add_argument("--adapters", type=str, help="PEFT adapter dir (training run_dir). Required if --base is set.")
    ap.add_argument("--images", type=str, nargs="+", help="Image path(s) to run inference on.", required=True)
    ap.add_argument("--out_dir", type=str, default="./inference_out", help="Output directory.")
    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Prompt instruction.")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"], help="Model dtype.")
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading when using base+adapters.")
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto","flash_attention_2","eager"], help="Attention backend.")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Load model
    model, tokenizer, mode = load_model(
        merged_dir=args.merged_dir,
        base=args.base,
        adapters=args.adapters,
        dtype=args.dtype,
        load_in_4bit=(not args.no_4bit),
        attn_impl=args.attn_impl,
        max_seq_len=args.max_seq_len,
    )

    all_results = []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        try:
            raw_text = generate_for_image(
                model=model,
                tokenizer=tokenizer,
                image_path=img_path,
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            print(f"❌ Generation failed for {img_path}: {e}")
            continue

        detections = parse_and_validate(raw_text)

        # Save per-image JSON
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_out = os.path.join(args.out_dir, f"{base}_detections.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "image": img_path,
                "mode": mode,
                "raw_text": raw_text,
                "detections": detections,
            }, f, ensure_ascii=False, indent=2)

        # Render overlay
        try:
            img = load_pil(img_path)
            vis_out = os.path.join(args.out_dir, f"{base}_vis.jpg")
            draw_detections(img, detections, vis_out)
        except Exception as e:
            print(f"⚠️ Visualization failed for {img_path}: {e}")

        all_results.append({
            "image": img_path,
            "json": json_out,
            "vis": os.path.join(args.out_dir, f"{base}_vis.jpg"),
            "num_detections": len(detections),
        })
        print(f"✔ Done: {img_path} | {len(detections)} detections")

    # Index file
    with open(os.path.join(args.out_dir, "results_index.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
