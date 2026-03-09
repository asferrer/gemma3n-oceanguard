# -*- coding: utf-8 -*-
"""
COCO → multimodal chat dataset (Gemma3N/Unsloth) with:
- append & dedupe
- robust image resolution across multiple search roots
- single-file or directory-of-JSONs processing

Produces JSONL like:
  {"messages":[
      {"role":"user","content":[{"type":"text","text":"..."},{"type":"image","image":"..."}]},
      {"role":"assistant","content":[{"type":"text","text":"[...]"}]}
  ]}
Compatible with Unsloth vision fine-tuning format and TRL’s SFTTrainer.
"""

import json
import os
import argparse
import random
import hashlib
from glob import glob
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional, Union
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Prompt template - Optimized for precise object detection
USER_PROMPT_TEMPLATE = (
    "You are an expert marine debris detection system trained on underwater imagery. "
    "Analyze the image and identify all visible man-made waste items with high precision.\n\n"
    "For each detection, provide:\n"
    "1. debris_type: Must be one of [{class_list_str}]\n"
    "2. material: Likely composition (Plastic, Metal, Fabric, Rubber, Glass, etc.)\n"
    "3. confidence_score: Your certainty level from 0.0 to 1.0 based on visibility and clarity\n"
    "4. bounding_box: [x_min, y_min, x_max, y_max] in normalized coordinates [0,1]\n"
    "   - Draw tight boxes around objects, excluding background water\n"
    "   - Ensure all coordinates are within valid bounds: 0 ≤ coordinate ≤ 1\n"
    "   - x_min < x_max and y_min < y_max must always be true\n\n"
    "Return ONLY a valid JSON array with no additional text.\n\n"
    "Example output:\n"
    '[{{"debris_type": "Bottle", "material": "Plastic", "confidence_score": 0.95, '
    '"bounding_box": [0.123456, 0.234567, 0.345678, 0.456789]}}]'
)

CATEGORY_TO_MATERIAL_MAPPING = {
    # Primary marine debris categories
    "Bottle": "Plastic",
    "Can": "Metal",
    "Fishing_Net": "Plastic",
    "Glove": "Rubber",  # Fixed: Added missing mapping
    "Mask": "Fabric",  # Fixed: Added missing mapping
    "Metal_Debris": "Metal",  # Fixed: Added missing mapping
    "Plastic_Debris": "Plastic",  # Fixed: Was mapping to "Unknown"
    "Tire": "Rubber",

    # Generic categories
    "Plastic_Bag": "Plastic",
    "Food_Wrapper": "Plastic",
    "Rope": "Fabric",
    "Plastic": "Plastic",
    "Metal": "Metal",
    "Glass": "Glass",
    "Fabric": "Fabric",
    "Rubber": "Rubber",

    # Fallback
    "Default": "Unknown",
}

def visualize_bounding_boxes(image, detections):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    color_map = {
        "plastic": "#FF1493", "metal": "#1E90FF", "fabric": "#32CD32",
        "textile": "#32CD32", "rubber": "#FFD700", "glass": "#00CED1",
        "nylon": "#9400D3", "synthetic": "#9400D3", "fishing": "#FF8C00",
        "net": "#FF8C00", "default": "#FFFFFF"
    }
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    for det in detections:
        box = det.get('bounding_box')
        label = det.get('debris_type', 'Unknown')
        material = det.get('material', 'default').lower()
        if not isinstance(box, list) or len(box) != 4:
            continue
        color_key = next((k for k in color_map if k in material or k in label.lower()), "default")
        color = color_map[color_key]
        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height),
                        (xmax * width, ymax * height)],
                       outline=color, width=4)
        text = f"{label}"
        tw, th = draw.textlength(text, font=font), 20
        tx, ty = int(xmin * width), max(0, int(ymin * height) - th - 2)
        draw.rectangle([(tx, ty), (tx + tw + 6, ty + th)], fill=color)
        draw.text((tx + 3, ty), text, fill="black", font=font)
    return img_draw

# ---------- JSONL helpers ----------
def jsonl_read(path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def jsonl_write(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _extract_image_from_user_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Supports legacy string content with '[Image: path]' AND structured list content with {'type':'image','image':...}.
    """
    if isinstance(content, str):
        marker = "[Image:"
        idx = content.find(marker)
        if idx != -1:
            end = content.find("]", idx)
            if end != -1:
                return content[idx + len(marker):end].strip()
        return ""
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image" and isinstance(c.get("image"), str):
                return c["image"]
    return ""

def messages_image_path(sample: Dict[str, Any]) -> str:
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            return _extract_image_from_user_content(m.get("content", ""))
    return ""

def messages_hash(sample: Dict[str, Any]) -> str:
    mbytes = json.dumps(sample.get("messages", []), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(mbytes).hexdigest()

def build_dedupe_sets(existing: List[Dict[str, Any]], dedupe_on: str) -> Tuple[Set[str], Set[str]]:
    by_image, by_hash = set(), set()
    if dedupe_on in ("image", "messages_hash"):
        for s in existing:
            if dedupe_on == "image":
                p = messages_image_path(s)
                if p:
                    by_image.add(p)
            elif dedupe_on == "messages_hash":
                by_hash.add(messages_hash(s))
    return by_image, by_hash

def is_duplicate(sample: Dict[str, Any], dedupe_on: str, by_image: Set[str], by_hash: Set[str]) -> bool:
    if dedupe_on == "none":
        return False
    if dedupe_on == "image":
        return messages_image_path(sample) in by_image
    if dedupe_on == "messages_hash":
        return messages_hash(sample) in by_hash
    return False

# ---------- Image search ----------
def _norm(p: str) -> str:
    return os.path.normpath(p)

def build_image_index(search_dirs: List[str], exts: List[str], case_insensitive: bool = True) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    exts_norm = {e.lower().lstrip(".") for e in exts}
    for root in search_dirs:
        if not root:
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
                if exts_norm and ext not in exts_norm:
                    continue
                key = fn.lower() if case_insensitive else fn
                full = _norm(os.path.join(dirpath, fn))
                index.setdefault(key, []).append(full)
    return index

def resolve_image_path(rel_path_from_coco: str,
                       images_dir: str,
                       image_index: Dict[str, List[str]],
                       on_missing: str = "warn",
                       path_mode: str = "absolute",
                       relative_base: Optional[str] = None,
                       case_insensitive: bool = True) -> Optional[str]:
    joined = _norm(os.path.join(images_dir, rel_path_from_coco)) if images_dir else _norm(rel_path_from_coco)
    if os.path.exists(joined):
        path_to_save = joined
    else:
        base = os.path.basename(rel_path_from_coco)
        key = base.lower() if case_insensitive else base
        candidates = image_index.get(key, [])
        if len(candidates) > 1:
            tail = _norm(rel_path_from_coco).replace("\\", "/").split("/")
            matched = []
            for cand in candidates:
                c_parts = _norm(cand).replace("\\", "/").split("/")
                for n in range(min(len(tail), len(c_parts)), 0, -1):
                    if tail[-n:] == c_parts[-n:]:
                        matched.append(cand)
                        break
            if matched:
                candidates = matched
        if candidates:
            path_to_save = candidates[0]
        else:
            msg = f"[image-missing] Could not resolve: '{rel_path_from_coco}'"
            if on_missing == "error":
                raise FileNotFoundError(msg)
            elif on_missing == "warn":
                print("⚠️", msg)
            else:
                return None
            path_to_save = joined
    if path_mode == "relative_to" and relative_base:
        try:
            path_to_save = os.path.relpath(path_to_save, start=relative_base)
        except Exception:
            pass
    return path_to_save

# ---------- Main conversion ----------
def process_coco_to_multimodal_messages(
    annotations_path: str,
    images_dir: str,
    output_dir: str,
    split_ratio: float = 0.9,
    double_user_format: bool = False,
    keep_images_without_annotations: bool = False,
    visualize: bool = False,
    max_visualizations: int = 24,
    append: bool = False,
    split_strategy: str = "append_split_new",
    dedupe_on: str = "image",
    train_filename: str = "train_dataset.jsonl",
    val_filename: str = "validation_dataset.jsonl",
    image_search_dirs: Optional[List[str]] = None,
    image_exts: Optional[List[str]] = None,
    on_missing_image: str = "warn",
    path_mode: str = "absolute",
    relative_to: Optional[str] = None,
):
    print(f"🔄 Loading COCO annotations from: {annotations_path}")
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}
    categories_info = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    annotations_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    print(f"✅ {len(images_info)} images | {len(categories_info)} categories")

    class_list = sorted(list(categories_info.values()))
    class_list_str = ", ".join([f"'{c}'" for c in class_list]) if class_list else "'object'"
    instruction = USER_PROMPT_TEMPLATE.format(class_list_str=class_list_str)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    val_path   = os.path.join(output_dir, val_filename)

    existing_train = jsonl_read(train_path) if append else []
    existing_val   = jsonl_read(val_path)   if append else []
    print(f"📥 Existing: train={len(existing_train)} | val={len(existing_val)} (append={append})")

    dedupe_train_img, dedupe_train_hash = build_dedupe_sets(existing_train, dedupe_on)
    dedupe_val_img,   dedupe_val_hash   = build_dedupe_sets(existing_val,   dedupe_on)

    default_exts = ["png", "jpg", "jpeg", "bmp", "webp", "tiff"]
    exts = image_exts if image_exts else default_exts

    search_roots = [images_dir] + (image_search_dirs or [])
    search_roots = [r for r in dict.fromkeys([os.path.normpath(r) for r in search_roots if r])]
    print(f"🔎 Building image index on {len(search_roots)} roots with exts={exts} ...")
    image_index = build_image_index(search_roots, exts, case_insensitive=True)
    total_indexed = sum(len(v) for v in image_index.values())
    print(f"📚 Indexed {total_indexed} image paths")

    samples_new: List[Dict[str, Any]] = []
    vis_count = 0
    vis_dir = os.path.join(output_dir, "vis")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    print("🔄 Converting to multimodal messages and normalizing boxes...")
    for image_id, image_info in tqdm(images_info.items(), desc="Images"):
        anns = annotations_by_image.get(image_id, [])
        if not keep_images_without_annotations and not anns:
            continue

        w, h = image_info["width"], image_info["height"]
        rel_path = image_info["file_name"]
        resolved_path = resolve_image_path(
            rel_path_from_coco=rel_path,
            images_dir=images_dir,
            image_index=image_index,
            on_missing=on_missing_image,
            path_mode=path_mode,
            relative_base=relative_to,
            case_insensitive=True,
        )
        if resolved_path is None:
            continue

        detections = []
        for a in anns:
            cat_name = categories_info.get(a["category_id"], "object")
            x, y, bw, bh = a["bbox"]

            # Validate bbox dimensions (COCO format: [x, y, width, height])
            if bw <= 0 or bh <= 0:
                logging.warning(f"Invalid bbox dimensions in {img_info['file_name']}: {a['bbox']}")
                continue

            # Normalize coordinates with higher precision (6 decimals instead of 4)
            # This preserves ~1 pixel accuracy for images up to 1000x1000
            x_min = x / w
            y_min = y / h
            x_max = (x + bw) / w
            y_max = (y + bh) / h

            # Clamp to valid range [0, 1] to handle edge cases
            x_min = max(0.0, min(x_min, 1.0))
            y_min = max(0.0, min(y_min, 1.0))
            x_max = max(0.0, min(x_max, 1.0))
            y_max = max(0.0, min(y_max, 1.0))

            # Final validation: ensure min < max
            if x_min >= x_max or y_min >= y_max:
                logging.warning(f"Invalid bbox after normalization in {img_info['file_name']}: [{x_min}, {y_min}, {x_max}, {y_max}]")
                continue

            detections.append({
                "debris_type": cat_name,
                "material": CATEGORY_TO_MATERIAL_MAPPING.get(cat_name, "Unknown"),
                "confidence_score": 1.0,
                "bounding_box": [x_min, y_min, x_max, y_max],
            })

        # Structured messages (Unsloth vision format)
        user_block = {
            "role": "user",
            "content": [
                {"type": "text",  "text": instruction},
                {"type": "image", "image": resolved_path}
            ]
        }
        assistant_block = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": json.dumps(detections, ensure_ascii=False)}
            ]
        }

        messages = [user_block, assistant_block]
        if double_user_format:
            messages = [user_block, {"role": "user", "content": user_block["content"]}]

        sample = {"messages": messages}

        if dedupe_on != "none":
            if is_duplicate(sample, dedupe_on, dedupe_train_img, dedupe_train_hash):
                continue
            if is_duplicate(sample, dedupe_on, dedupe_val_img,   dedupe_val_hash):
                continue

        samples_new.append(sample)

        if visualize and vis_count < max_visualizations and os.path.exists(resolved_path) and len(detections) > 0:
            try:
                img = Image.open(resolved_path).convert("RGB")
                rendered = visualize_bounding_boxes(img, detections)
                base = os.path.basename(resolved_path)
                out_path = os.path.join(vis_dir, f"{os.path.splitext(base)[0]}_vis.jpg")
                rendered.save(out_path)
                vis_count += 1
            except Exception as e:
                print(f"⚠️ Visualization failed for {resolved_path}: {e}")

    print(f"🆕 New samples (pre-split): {len(samples_new)}")
    if visualize:
        print(f"🖼️ Visualization (new only): {vis_count} saved → {vis_dir}")

    # Split / Append logic
    if append:
        existing_train = jsonl_read(train_path)
        existing_val   = jsonl_read(val_path)
        if split_strategy == "resplit_all":
            all_samples = existing_train + existing_val + samples_new
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * split_ratio)
            train_data = all_samples[:split_idx]
            val_data   = all_samples[split_idx:]
        elif split_strategy == "all_to_train":
            train_data = existing_train + samples_new
            val_data   = existing_val
        elif split_strategy == "all_to_val":
            train_data = existing_train
            val_data   = existing_val + samples_new
        else:  # append_split_new
            random.shuffle(samples_new)
            split_idx = int(len(samples_new) * split_ratio)
            train_data = existing_train + samples_new[:split_idx]
            val_data   = existing_val   + samples_new[split_idx:]
    else:
        random.shuffle(samples_new)
        split_idx = int(len(samples_new) * split_ratio)
        train_data = samples_new[:split_idx]
        val_data   = samples_new[split_idx:]

    jsonl_write(train_path, train_data)
    jsonl_write(val_path,   val_data)

    print("\n--- Results ---")
    print(f"📁 Output dir: {output_dir}")
    print(f"🔵 Train: {train_path} ({len(train_data)} samples)")
    print(f"🟢 Val:   {val_path}   ({len(val_data)} samples)")
    if visualize:
        print(f"🖼️ Vis (new): {vis_dir} ({vis_count} images)")
    print("🎉 Ready for PEFT with Gemma3N/Unsloth")

def process_dir_of_coco(
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    **kwargs
):
    """
    Lee todos los .json en `annotations_dir` y enruta su contenido al split
    correcto en `output_dir` según el nombre del fichero:
      - *train*.json                  -> train_dataset.jsonl
      - *val|valid|validation*.json   -> validation_dataset.jsonl
      - *test*.json                   -> test_dataset.jsonl
    Corrige el bug de "multiple values for keyword argument 'train_filename'".
    """
    os.makedirs(output_dir, exist_ok=True)
    coco_files = sorted(glob(os.path.join(annotations_dir, "*.json")))
    if not coco_files:
        raise FileNotFoundError(f"No JSON files found in {annotations_dir}")

    # Copia base de kwargs y extrae/elim. filenames para no reenviarlos duplicados
    base_kwargs = dict(kwargs)

    # Saca (y quita) posibles valores que vengan en kwargs para evitar duplicados
    default_train_filename = base_kwargs.pop("train_filename", "train_dataset.jsonl")
    default_val_filename   = base_kwargs.pop("val_filename", "validation_dataset.jsonl")
    # (si quieres parametrizar test también, añade pop de "test_filename" aquí)
    default_test_filename  = "test_dataset.jsonl"

    # Helper para detectar split por nombre
    def detect_split(name: str) -> str:
        n = name.lower()
        if "train" in n:
            return "train"
        if "test" in n:
            return "test"
        if ("validation" in n) or ("valid" in n) or ("val" in n):
            return "val"
        return "train"

    print(f"🗂️  Found {len(coco_files)} JSON files in: {annotations_dir}")
    print(f"📦 Output dir: {output_dir}")

    for jf in coco_files:
        name = os.path.splitext(os.path.basename(jf))[0]
        split = detect_split(name)

        # Clona kwargs base y fuerza append=True para acumular por split
        k = dict(base_kwargs)
        k["append"] = True

        if split == "train":
            # TODO a train
            k["split_strategy"] = "all_to_train"
            print(f"\n=== Processing {jf} -> split=train (file={default_train_filename}) ===")
            process_coco_to_multimodal_messages(
                annotations_path=jf,
                images_dir=images_dir,
                output_dir=output_dir,
                train_filename=default_train_filename,   # << se pasa solo aquí
                val_filename=default_val_filename,       # << y NO vuelve dentro de **k
                **k
            )

        elif split == "val":
            # TODO a val
            k["split_strategy"] = "all_to_val"
            print(f"\n=== Processing {jf} -> split=val (file={default_val_filename}) ===")
            process_coco_to_multimodal_messages(
                annotations_path=jf,
                images_dir=images_dir,
                output_dir=output_dir,
                train_filename=default_train_filename,
                val_filename=default_val_filename,
                **k
            )

        elif split == "test":
            # TODO a test: reutilizamos 'all_to_train' escribiendo en test_filename
            k["split_strategy"] = "all_to_train"
            print(f"\n=== Processing {jf} -> split=test (file={default_test_filename}) ===")
            process_coco_to_multimodal_messages(
                annotations_path=jf,
                images_dir=images_dir,
                output_dir=output_dir,
                train_filename=default_test_filename,    # << aquí escribimos test
                val_filename=default_val_filename,
                **k
            )

        else: continue  # nunca debería pasar

    print("\n✅ Completed directory processing with split routing by filename.")
    print(f"  - Train JSONL:      {os.path.join(output_dir, default_train_filename)}")
    print(f"  - Validation JSONL: {os.path.join(output_dir, default_val_filename)}")
    test_path = os.path.join(output_dir, default_test_filename)
    if os.path.exists(test_path):
        print(f"  - Test JSONL:       {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO → multimodal chat dataset with recursive image search & append.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--annotations_path", type=str, help="Path to a single COCO JSON (e.g., train.json).")
    grp.add_argument("--annotations_dir",  type=str, default= "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/split_v8", help="Directory with multiple COCO JSONs (e.g., train/val/test).")

    parser.add_argument("--images_dir", type=str, default="C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset",
                        help="Primary base directory (first direct attempt).")
    parser.add_argument("--output_dir", type=str, default="./prepared_dataset", help="Output directory.")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train ratio (e.g., 0.9 => 90/10 split).")

    parser.add_argument("--double_user_format", action="store_true",
                        help="Emit two consecutive 'user' messages (text only).")
    parser.add_argument("--keep_images_without_annotations", action="store_true",
                        help="Keep images without annotations (assistant returns an empty list).")

    parser.add_argument("--visualize", action="store_true",
                        help="Save a subset of NEW samples with normalized boxes to <output>/vis/.")
    parser.add_argument("--max_visualizations", type=int, default=24,
                        help="Maximum number of visualization images to save.")

    parser.add_argument("--append", action="store_true", help="Append to existing train/val JSONLs.")
    parser.add_argument("--split_strategy", type=str, default="append_split_new",
                        choices=["append_split_new", "resplit_all", "all_to_train", "all_to_val"],
                        help="Distribution strategy for NEW samples when appending.")
    parser.add_argument("--dedupe_on", type=str, default="image",
                        choices=["image", "messages_hash", "none"], help="Deduplication key.")
    parser.add_argument("--train_filename", type=str, default="train_dataset.jsonl", help="Train JSONL name.")
    parser.add_argument("--val_filename", type=str, default="validation_dataset.jsonl", help="Val JSONL name.")

    parser.add_argument("--image_search_dirs", type=str, nargs="*", default=[
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/cleansea_dataset/CocoFormatDataset/train_coco/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Ocean_garbage/train/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Ocean_garbage/valid/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Neural_Ocean/train/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Neural_Ocean/valid/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/cleansea_dataset/CocoFormatDataset/test_coco/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Ocean_garbage/test/",
        "C:/Users/aleja/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/Neural_Ocean/test/",
    ], help="Additional root directories to recursively search for images.")
    parser.add_argument("--image_exts", type=str, nargs="*", default=None,
                        help="Image extensions to index (default: png jpg jpeg bmp webp tiff).")
    parser.add_argument("--on_missing_image", type=str, default="warn",
                        choices=["skip", "warn", "error"], help="When an image cannot be found.")
    parser.add_argument("--path_mode", type=str, default="absolute",
                        choices=["absolute", "relative_to"], help="How to store image paths in messages.")
    parser.add_argument("--relative_to", type=str, default=None,
                        help="If path_mode=relative_to, make image paths relative to this directory.")

    args = parser.parse_args()

    common_kwargs = dict(
        split_ratio=args.split_ratio,
        double_user_format=args.double_user_format,
        keep_images_without_annotations=args.keep_images_without_annotations,
        visualize=args.visualize,
        max_visualizations=args.max_visualizations,
        append=args.append,
        split_strategy=args.split_strategy,
        dedupe_on=args.dedupe_on,
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        image_search_dirs=args.image_search_dirs,
        image_exts=args.image_exts,
        on_missing_image=args.on_missing_image,
        path_mode=args.path_mode,
        relative_to=args.relative_to,
    )

    if args.annotations_path:
        process_coco_to_multimodal_messages(
            annotations_path=args.annotations_path,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            **common_kwargs
        )
    else:
        process_dir_of_coco(
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            **common_kwargs
        )
