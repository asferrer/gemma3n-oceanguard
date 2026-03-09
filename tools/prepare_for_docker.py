#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Path Converter for Docker Deployment
=============================================
Converts absolute Windows paths in existing datasets to relative paths
suitable for Docker container deployment.

This script:
1. Reads existing JSONL datasets with absolute Windows paths
2. Converts image paths to relative paths from project root
3. Validates that all images are accessible
4. Creates new Docker-compatible dataset files

Usage:
    python tools/prepare_for_docker.py \
        --input-dir ./prepared_dataset \
        --output-dir ./prepared_dataset_docker \
        --project-root /workspace/project
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def convert_path_to_relative(
    absolute_path: str,
    windows_project_root: str,
    container_project_root: str = "/workspace/project"
) -> str:
    """
    Convert Windows absolute path to container-relative path.

    Maneja dos casos:
    1. Imágenes dentro del proyecto: C:\\Users\\...\\gemma3n\\prepared_dataset\\...
       → /workspace/project/prepared_dataset/...

    2. Imágenes externas (dataset RT-DETRv2):
       C:\\Users\\...\\RT-DETRv2-Densea\\rtdetrv2_pytorch\\dataset\\...
       → /workspace/project/dataset/...

       Esto asume que el dataset externo está montado como volumen en docker-compose.yml

    Args:
        absolute_path: Windows path like C:\\Users\\...\\dataset\\image.jpg
        windows_project_root: Windows path to project root
        container_project_root: Container path to project root

    Returns:
        Container path like /workspace/project/dataset/...
    """
    # Normalize paths for comparison
    abs_path_norm = os.path.normpath(absolute_path)
    win_root_norm = os.path.normpath(windows_project_root)

    # Caso 1: Try to make relative to project root (imágenes dentro del proyecto)
    try:
        relative_path = os.path.relpath(abs_path_norm, start=win_root_norm)

        # Si la ruta relativa no comienza con "..", está dentro del proyecto
        if not relative_path.startswith(".."):
            # Convert Windows separators to Unix
            relative_path = relative_path.replace('\\', '/')
            # Join with container root
            container_path = os.path.join(container_project_root, relative_path).replace('\\', '/')
            return container_path
    except ValueError:
        pass  # Diferentes drives, continuar al caso 2

    # Caso 2: Imagen externa (dataset RT-DETRv2 u otros)
    # Extraer la parte relevante del path
    # De: C:\Users\aleja\Desktop\RT-DETRv2-Densea\rtdetrv2_pytorch\dataset\cleansea_dataset\...
    # A: /workspace/project/dataset/cleansea_dataset/...

    # Buscar la palabra "dataset" en el path y tomar todo desde ahí
    path_parts = abs_path_norm.split(os.sep)

    # Encontrar el índice de "dataset" en el path
    try:
        dataset_index = None
        for i, part in enumerate(path_parts):
            if part.lower() == 'dataset':
                dataset_index = i
                break

        if dataset_index is not None:
            # Tomar todo desde "dataset" en adelante
            relative_parts = path_parts[dataset_index:]
            relative_path = '/'.join(relative_parts)
            container_path = f"{container_project_root}/{relative_path}"
            return container_path
    except:
        pass

    # Caso 3: No se pudo convertir - advertir y retornar path original
    print(f"⚠️ Warning: Cannot convert path for {absolute_path}")
    print(f"   Path might not be accessible in container")
    return absolute_path


def validate_image_exists(image_path: str, windows_project_root: str) -> bool:
    """Check if image file exists relative to project root."""
    if os.path.isabs(image_path):
        return os.path.exists(image_path)
    else:
        full_path = os.path.join(windows_project_root, image_path)
        return os.path.exists(full_path)


def process_sample(
    sample: Dict[str, Any],
    windows_project_root: str,
    container_project_root: str,
    validate: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample, converting image paths.

    Args:
        sample: Dataset sample with messages
        windows_project_root: Windows path to project root
        container_project_root: Container path to project root
        validate: Whether to validate image existence

    Returns:
        Processed sample with converted paths, or None if validation fails
    """
    processed_sample = {"messages": []}

    for message in sample.get("messages", []):
        processed_message = {
            "role": message["role"],
            "content": []
        }

        for content_item in message.get("content", []):
            if isinstance(content_item, dict) and content_item.get("type") == "image":
                original_path = content_item.get("image", "")

                # Validate image exists before conversion
                if validate and not validate_image_exists(original_path, windows_project_root):
                    print(f"⚠️ Image not found: {original_path}")
                    return None

                # Convert path
                container_path = convert_path_to_relative(
                    original_path,
                    windows_project_root,
                    container_project_root
                )

                processed_message["content"].append({
                    "type": "image",
                    "image": container_path
                })
            else:
                # Keep text and other content as-is
                processed_message["content"].append(content_item)

        processed_sample["messages"].append(processed_message)

    return processed_sample


def convert_dataset(
    input_jsonl: str,
    output_jsonl: str,
    windows_project_root: str,
    container_project_root: str = "/workspace/project",
    validate: bool = True
):
    """
    Convert a JSONL dataset file from Windows paths to container paths.

    Args:
        input_jsonl: Input JSONL file path
        output_jsonl: Output JSONL file path
        windows_project_root: Windows path to project root
        container_project_root: Container path to project root
        validate: Whether to validate image existence
    """
    if not os.path.exists(input_jsonl):
        print(f"❌ Input file not found: {input_jsonl}")
        return

    print(f"📖 Reading {input_jsonl}...")

    # Read input
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse line: {e}")

    print(f"✅ Loaded {len(samples)} samples")

    # Process samples
    processed_samples = []
    skipped = 0

    for sample in tqdm(samples, desc="Converting paths"):
        processed = process_sample(
            sample,
            windows_project_root,
            container_project_root,
            validate=validate
        )

        if processed:
            processed_samples.append(processed)
        else:
            skipped += 1

    print(f"✅ Processed {len(processed_samples)} samples (skipped {skipped})")

    # Write output
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"💾 Saved to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset paths for Docker deployment"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./prepared_dataset',
        help='Input directory containing JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./prepared_dataset_docker',
        help='Output directory for converted JSONL files'
    )
    parser.add_argument(
        '--windows-root',
        type=str,
        default=None,
        help='Windows project root path (auto-detected if not provided)'
    )
    parser.add_argument(
        '--container-root',
        type=str,
        default='/workspace/project',
        help='Container project root path'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip image existence validation'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=['train_dataset.jsonl', 'validation_dataset.jsonl', 'test_dataset.jsonl'],
        help='JSONL files to convert'
    )

    args = parser.parse_args()

    # Auto-detect Windows root if not provided
    if args.windows_root is None:
        # Get script directory and go up to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        windows_root = os.path.dirname(script_dir)  # Go up from tools/ to project root
    else:
        windows_root = args.windows_root

    windows_root = os.path.normpath(windows_root)
    print(f"🔧 Windows project root: {windows_root}")
    print(f"🐳 Container project root: {args.container_root}")
    print(f"📂 Input directory: {args.input_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print()

    # Process each file
    for filename in args.files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        if not os.path.exists(input_path):
            print(f"⚠️ Skipping {filename} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

        convert_dataset(
            input_jsonl=input_path,
            output_jsonl=output_path,
            windows_project_root=windows_root,
            container_project_root=args.container_root,
            validate=not args.no_validate
        )

    print(f"\n{'='*60}")
    print("✅ All datasets converted successfully!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Update your training config to use: {args.output_dir}")
    print(f"2. Build Docker image: docker-compose build")
    print(f"3. Start training: docker-compose up gemma3n-training")


if __name__ == "__main__":
    main()
