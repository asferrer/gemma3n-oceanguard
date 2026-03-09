# -*- coding: utf-8 -*-
"""
Config-driven PEFT (4-bit LoRA) fine-tuning for Gemma3N Vision (Unsloth + TRL) on Windows / single GPU.
- Preserves your original features (env snapshot, manifest, early stopping, merge, GGUF export).
- Adds optional UnslothVisionDataCollator for official vision dataset format.
Usage:
  python train.py --config config.yaml
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth  # important: import before transformers to enable patches
import sys
import json
import yaml
import shutil
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import functools

import torch
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.suppress_errors = True

from datasets import load_dataset
from transformers import set_seed, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

# ----------------------------
# Utilities
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yml", ".yaml")):
        return yaml.safe_load(text)
    return json.loads(text)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def setup_run_dir(output_root: str, run_name: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"run-{ts}"
    run_dir = os.path.join(output_root, "runs", run_name)
    ensure_dir(run_dir)
    return run_dir

def copy_config_to_run(config_path: str, run_dir: str):
    dst = os.path.join(run_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    return dst

def setup_logging(run_dir: str, verbose: bool = True):
    log_path = os.path.join(run_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    logging.info(f"Logging to {log_path}")
    return log_path

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----------------------------
# Dataset Validation Utilities
def validate_sample_format(sample: Dict[str, Any], sample_idx: int = -1) -> tuple[bool, Optional[str]]:
    """
    Validates that a sample has correct format for object detection.
    Returns: (is_valid, error_message)
    """
    try:
        messages = sample.get("messages", [])
        if not messages:
            return False, "No messages found"

        # Find assistant message with detections
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "text":
                        text = content_item.get("text", "")
                        try:
                            detections = json.loads(text)
                            if not isinstance(detections, list):
                                return False, f"Detections must be a list, got {type(detections)}"

                            # Validate each detection
                            for idx, det in enumerate(detections):
                                # Check required fields
                                if "bounding_box" not in det:
                                    return False, f"Detection {idx} missing 'bounding_box'"

                                bbox = det["bounding_box"]
                                if not isinstance(bbox, list) or len(bbox) != 4:
                                    return False, f"Detection {idx}: bbox must be [x_min, y_min, x_max, y_max]"

                                # Validate normalized coordinates
                                if not all(isinstance(coord, (int, float)) for coord in bbox):
                                    return False, f"Detection {idx}: bbox coords must be numeric"

                                if not all(0 <= coord <= 1 for coord in bbox):
                                    return False, f"Detection {idx}: bbox coords out of range [0,1]: {bbox}"

                                # Validate min < max
                                x_min, y_min, x_max, y_max = bbox
                                if x_min >= x_max or y_min >= y_max:
                                    return False, f"Detection {idx}: invalid bbox geometry (min >= max): {bbox}"

                        except json.JSONDecodeError as e:
                            return False, f"Invalid JSON in assistant response: {e}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_dataset_sample(dataset, num_samples: int = 100):
    """
    Validates a random sample of the dataset and logs statistics.
    """
    import random

    logging.info(f"🔍 Validating {num_samples} random samples from dataset...")

    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    valid_count = 0
    invalid_count = 0
    errors = []

    for idx in sample_indices:
        sample = dataset[idx]
        is_valid, error_msg = validate_sample_format(sample, idx)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors.append(f"Sample {idx}: {error_msg}")
            if len(errors) <= 5:  # Only log first 5 errors
                logging.warning(f"⚠ Invalid sample {idx}: {error_msg}")

    logging.info(f"✓ Dataset validation complete: {valid_count}/{num_samples} valid, {invalid_count} invalid")

    if invalid_count > 0:
        logging.warning(f"⚠ Found {invalid_count} invalid samples. Training may fail on these samples.")
        if invalid_count > num_samples * 0.1:  # More than 10% invalid
            logging.error("❌ High number of invalid samples detected! Consider regenerating the dataset.")

    return valid_count, invalid_count, errors

def load_jsonl_dataset(train_path: str, eval_path: str):
    data_files = {"train": train_path, "validation": eval_path}
    ds = load_dataset("json", data_files=data_files)

    # Simple schema check
    if len(ds["train"]) == 0:
        raise ValueError("Empty training dataset.")
    first = ds["train"][0]
    if "messages" not in first:
        raise ValueError("Each JSONL line must have a 'messages' key with user/assistant blocks.")

    # Enhanced validation: Check format of training samples
    logging.info(f"📊 Dataset loaded: {len(ds['train'])} train, {len(ds['validation'])} validation samples")
    validate_dataset_sample(ds["train"], num_samples=100)

    return ds

def collect_env_snapshot() -> Dict[str, Any]:
    import platform
    snap = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [],
        "packages": {},
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            snap["devices"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i)
            })
    try:
        import transformers, datasets, accelerate, trl, unsloth, bitsandbytes, pillow
        snap["packages"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "trl": trl.__version__,
            "unsloth": getattr(unsloth, "__version__", "unknown"),
            "bitsandbytes": getattr(bitsandbytes, "__version__", "unknown"),
            "pillow": getattr(pillow, "__version__", "unknown"),
        }
    except Exception:
        pass
    return snap

def write_manifest(run_dir: str,
                   config_copied_path: str,
                   logs_path: str,
                   trainer_state_path: str,
                   log_history_path: str,
                   final_metrics_path: str,
                   adapter_dir: str,
                   merged_dir: Optional[str] = None,
                   gguf_dir: Optional[str] = None,
                   best_model_checkpoint: Optional[str] = None):
    manifest = {
        "run_dir": run_dir,
        "artifacts": {
            "config_copy": config_copied_path,
            "logs": logs_path,
            "trainer_state": trainer_state_path,
            "trainer_log_history": log_history_path,
            "final_metrics": final_metrics_path,
            "adapter_dir": adapter_dir,
            "merged_model_dir": merged_dir,
            "gguf_dir": gguf_dir,
            "best_model_checkpoint": best_model_checkpoint,
        },
        "environment": collect_env_snapshot(),
    }
    save_json(manifest, os.path.join(run_dir, "manifest.json"))
    logging.info(f"Artifact manifest written at: {os.path.join(run_dir, 'manifest.json')}")

def formatting_prompts_func(examples, processor):
    # Use the model’s chat template to pack messages when not using the vision collator
    is_batch = isinstance(examples["messages"][0], list)
    if is_batch:
        convos = examples["messages"]
        texts = [processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True) for convo in convos]
        return texts
    else:
        convo = examples["messages"]
        text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return [text]

# ----------------------------
# Main
def main():
    parser = argparse.ArgumentParser(description="Config-driven PEFT training for Gemma3N Vision (Unsloth + TRL).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training    = cfg.get("training", {})
    data        = cfg.get("data", {})
    peft        = cfg.get("peft", {})
    system      = cfg.get("system", {})
    logging_cfg = cfg.get("logging", {})
    export_cfg  = cfg.get("export", {})
    extras      = cfg.get("extras", {})

    output_root = training.get("output_root", "./outputs/gemma3n_peft")
    run_name    = training.get("run_name", None)
    run_dir     = setup_run_dir(output_root, run_name)
    config_copy_path = copy_config_to_run(args.config, run_dir)
    logs_path = setup_logging(run_dir, verbose=logging_cfg.get("console", True))

    logging.info("===== PEFT Training (Gemma3N Vision) | Unsloth + TRL =====")
    logging.info(f"Run dir: {run_dir}")

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    seed = int(training.get("seed", 42))
    set_seed(seed)
    if system.get("disable_wandb", True):
        os.environ["WANDB_DISABLED"] = "true"

    # precision
    use_bf16 = bool(training.get("bf16", False)) and is_bfloat16_supported()
    use_fp16 = bool(training.get("fp16", False))
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    load_in_4bit = bool(training.get("use_4bit", False))

    model_name   = training.get("model_name", "unsloth/gemma-3n-E4B-it")
    max_seq_len  = int(training.get("max_seq_len", 4096))
    attn_impl    = extras.get("attn_implementation", "auto")
    use_vision_collator = bool(extras.get("use_vision_data_collator", True))

    logging.info(f"Loading base model: {model_name} | max_seq_len={max_seq_len} | 4bit={load_in_4bit} | dtype={dtype}")
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa
            from packaging import version

            # Verify Flash Attention version
            if hasattr(flash_attn, '__version__'):
                fa_version = flash_attn.__version__
                logging.info(f"Flash Attention version: {fa_version}")

                if version.parse(fa_version) >= version.parse("2.7.0"):
                    attn_impl = "flash_attention_2"
                    logging.info("✓ Flash-Attention 2 (v2.7.0+) detected and enabled.")
                else:
                    attn_impl = "eager"
                    logging.warning(f"⚠ Flash Attention {fa_version} is outdated (need >=2.7.0). Using eager mode.")
            else:
                attn_impl = "flash_attention_2"
                logging.info("Flash-Attention 2 detected (version unknown), attempting to use it.")
        except ImportError:
            attn_impl = "eager"
            logging.info("Flash-Attention not installed, using eager mode.")
        except Exception as e:
            attn_impl = "eager"
            logging.warning(f"Flash-Attention check failed: {e}. Using eager mode.")

    # Load model with resume_download=True to handle interrupted downloads
    logging.info("Loading model (with automatic download resume if needed)...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            model, processor = FastLanguageModel.from_pretrained(
                model_name     = model_name,
                max_seq_length = max_seq_len,
                dtype          = dtype,
                load_in_4bit   = load_in_4bit,
                fast_inference = False,
                attn_implementation = attn_impl,
                resume_download = True,  # Resume interrupted downloads
            )
            logging.info("Model loaded successfully")
            break
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"Model loading failed (attempt {attempt}/{max_retries}): {e}")
                logging.info(f"Retrying in 10 seconds...")
                import time
                time.sleep(10)
            else:
                logging.error(f"Model loading failed after {max_retries} attempts")
                raise

    # Optional torch.compile
    if extras.get("torch_compile", False):
        try:
            model = torch.compile(model, mode=extras.get("torch_compile_mode", "reduce-overhead"))
            logging.info("torch.compile enabled.")
        except Exception as e:
            logging.warning(f"torch.compile not available: {e}")

    # ---- PEFT setup (accept both old/new config keys) ----
    lora_r        = int(peft.get("r", 16))
    lora_alpha    = int(peft.get("lora_alpha", 16))
    lora_dropout  = float(peft.get("lora_dropout", 0.05))
    use_rslora    = bool(peft.get("use_rslora", False))

    cfg_tune_language_layers = bool(peft.get("tune_language_layers", peft.get("tune_language_model", False)))
    cfg_tune_vision_layers   = bool(peft.get("tune_vision_layers",   peft.get("tune_vision_model", False)))
    cfg_tune_attention_modules = bool(peft.get("tune_attention_modules", False))
    cfg_tune_mlp_modules     = bool(peft.get("tune_mlp_modules", False))

    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora_r,
        lora_alpha     = lora_alpha,
        lora_dropout   = lora_dropout,
        target_modules = "all-linear",
        finetune_vision_layers     = cfg_tune_vision_layers,
        finetune_language_layers   = cfg_tune_language_layers,
        finetune_attention_modules = cfg_tune_attention_modules,
        finetune_mlp_modules       = cfg_tune_mlp_modules,
        bias           = "none",
        use_rslora     = use_rslora,
        loftq_config   = None,
    )

    # gradient checkpointing
    if training.get("gradient_checkpointing", True):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled (use_reentrant=False).")

    # ---- Load dataset ----
    train_jsonl = data.get("train_jsonl")
    eval_jsonl  = data.get("eval_jsonl")
    if not train_jsonl or not eval_jsonl:
        raise ValueError("data.train_jsonl and data.eval_jsonl must be specified in config.")

    ds = load_jsonl_dataset(train_jsonl, eval_jsonl)
    logging.info(f"Dataset loaded: train={len(ds['train'])} | val={len(ds['validation'])}")

    # Save a tiny schema/sample for auditability
    sample_out = {
        "train_first": ds["train"][0],
        "val_first": ds["validation"][0] if len(ds["validation"]) > 0 else None
    }
    save_json(sample_out, os.path.join(run_dir, "dataset_first_samples.json"))

    # ---- Trainer args ----
    eval_strategy = training.get("evaluation_strategy", "steps")
    save_strategy = training.get("save_strategy", "steps")
    if eval_strategy != save_strategy and bool(training.get("load_best_model_at_end", False)):
        logging.warning(f"evaluation_strategy ({eval_strategy}) != save_strategy ({save_strategy}); overriding eval_strategy to match save_strategy.")
        eval_strategy = save_strategy

    sft_args = SFTConfig(
        output_dir                      = run_dir,
        num_train_epochs                = int(training.get("epochs", 3)),
        per_device_train_batch_size     = int(training.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size      = int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps     = int(training.get("gradient_accumulation_steps", 8)),
        learning_rate                   = float(training.get("lr", 5e-5)),
        weight_decay                    = float(training.get("weight_decay", 0.0)),
        max_grad_norm                   = float(training.get("max_grad_norm", 1.0)),
        warmup_ratio                    = float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type               = training.get("lr_scheduler_type", "cosine"),
        bf16                            = use_bf16,
        fp16                            = use_fp16,
        seed                            = seed,
        report_to                       = training.get("report_to", "none"),
        logging_first_step              = bool(training.get("logging_first_step", False)),
        do_eval                         = True,
        eval_strategy                   = eval_strategy,
        save_strategy                   = save_strategy,
        optim                           = training.get("optim", "paged_adamw_8bit"),
        dataset_num_proc                = int(training.get("dataset_num_proc", 1)),
        eval_steps                      = int(training.get("eval_steps", 200)),
        save_steps                      = int(training.get("save_steps", 200)),
        save_total_limit                = int(training.get("save_total_limit", 3)),
        metric_for_best_model           = training.get("metric_for_best_model", "eval_loss"),
    )
    sft_args.load_best_model_at_end = bool(training.get("load_best_model_at_end", False))

    # Save training args to disk
    args_out_dir = os.path.join(run_dir, "trainer_args")
    ensure_dir(args_out_dir)
    save_json(sft_args.to_dict(), os.path.join(args_out_dir, "training_args.json"))

    callbacks = []
    if training.get("early_stopping", {}).get("enabled", False):
        es_cfg = training["early_stopping"]
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience  = int(es_cfg.get("patience", 3)),
            early_stopping_threshold = float(es_cfg.get("threshold", 0.0)),
        ))

    from unsloth import get_chat_template
    processor = get_chat_template(processor, "gemma-3n")
    # ---- Choose pipeline: Vision Collator (preferred) or Chat Template formatting ----
    trainer_kwargs = dict(
        model         = model,
        processor     = processor,
        args          = sft_args,
        train_dataset = ds["train"],
        eval_dataset  = ds["validation"],
        callbacks     = callbacks,
    )

    if use_vision_collator:
        try:
            from unsloth.trainer import UnslothVisionDataCollator
            data_collator = UnslothVisionDataCollator(
                model           = model,
                processor       = processor,
                resize          = "min",
            )
            trainer_kwargs["data_collator"] = data_collator
            # no formatting_func needed; the collator parses messages with type=image|text
            logging.info("Using UnslothVisionDataCollator.")
        except Exception as e:
            logging.warning(f"Falling back to chat template formatting (vision collator unavailable): {e}")
            formatting_func_with_processor = functools.partial(formatting_prompts_func, processor=processor)
            trainer_kwargs["formatting_func"] = formatting_func_with_processor
            trainer_kwargs["dataset_text_field"] = None
    else:
        formatting_func_with_processor = functools.partial(formatting_prompts_func, processor=processor)
        trainer_kwargs["formatting_func"] = formatting_func_with_processor
        trainer_kwargs["dataset_text_field"] = None
        logging.info("Using chat-template formatting pipeline (extras.use_vision_data_collator=False).")

    trainer = SFTTrainer(**trainer_kwargs)

    # ---- Train (supports resume) ----
    resume_flag = training.get("resume_from_checkpoint", False)
    resume_path = training.get("resume_checkpoint_path", None)
    if isinstance(resume_flag, str) and resume_flag.strip():
        resume_path = resume_flag
        resume_flag = True

    if resume_flag:
        logging.info(f"Resuming training from checkpoint: {resume_path or 'auto-last'}")
        train_result = trainer.train(resume_from_checkpoint=resume_path or True)
    else:
        logging.info("Starting training from scratch…")
        train_result = trainer.train()

    logging.info("Training finished.")
    trainer.save_model(run_dir)

    # ---- Save trainer state & metrics ----
    trainer_state_path    = os.path.join(run_dir, "trainer_state.json")
    trainer_log_hist_path = os.path.join(run_dir, "trainer_log_history.json")
    final_metrics_path    = os.path.join(run_dir, "final_metrics.json")

    trainer.state.save_to_json(trainer_state_path)
    save_json(trainer.state.log_history, trainer_log_hist_path)
    final_metrics = getattr(train_result, "metrics", {}) or {}
    save_json(final_metrics, final_metrics_path)

    adapter_dir = run_dir
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

    # ---- Merge adapters into base ----
    merged_dir = None
    if export_cfg.get("merge_and_save", False):
        merged_dir = export_cfg.get("merged_output_dir") or os.path.join(run_dir, "merged_model")
        ensure_dir(merged_dir)

        merged_ok = False
        try:
            logging.info("Merging adapters into 16-bit base via save_pretrained_merged(...)")
            # This writes a full 16-bit HF folder (recommended before GGUF)
            model.save_pretrained_merged(
                merged_dir,
                processor,
                save_method = "merged_16bit",
                # reduce peak mem during save if needed:
                maximum_memory_usage = 0.5,
            )
            merged_ok = True
            logging.info(f"Merged (16-bit) model saved to: {merged_dir}")
        except Exception as e:
            logging.warning(f"save_pretrained_merged failed: {e}")

        # Fallbacks if the official path fails:
        if not merged_ok:
            try:
                logging.info("Fallback: merge_and_unload() -> save_pretrained()")
                merged = model.merge_and_unload(safe_merge=True)
                merged.save_pretrained(merged_dir, safe_serialization=True)
                processor.save_pretrained(merged_dir)
                merged_ok = True
                logging.info(f"Merged model saved to: {merged_dir}")
            except Exception as e:
                logging.warning(f"PEFT merge_and_unload fallback failed: {e}")

        if not merged_ok:
            try:
                from unsloth.merge_utils import merge_lora_weights
                logging.info("Fallback: unsloth.merge_utils.merge_lora_weights(...)")
                merge_lora_weights(model, merged_dir, save_base_model=False, dtype=torch.bfloat16)
                processor.save_pretrained(merged_dir)
                merged_ok = True
                logging.info(f"Merged (merge_utils) saved to: {merged_dir}")
            except Exception as e:
                logging.warning(f"merge_lora_weights fallback failed: {e}")
                merged_dir = None

        if not merged_ok:
            logging.warning("Merging not completed. Adapter-only remains available in run_dir.")

    # ---- Optional GGUF export (multimodal, best-effort) ----
    gguf_dir = None
    if export_cfg.get("export_gguf", False):
        gguf_dir = export_cfg.get("gguf_output_dir", os.path.join(run_dir, "gguf"))
        ensure_dir(gguf_dir)
        quant = export_cfg.get("gguf_quant", "q4_k_m").lower()

        try:
            # Prefer exporting from the 16-bit merged folder for correctness
            export_source_dir = merged_dir if merged_dir is not None else run_dir

            logging.info(f"Loading for GGUF export from: {export_source_dir}")
            export_model, export_processor = FastLanguageModel.from_pretrained(
                model_name     = export_source_dir,
                max_seq_length = int(training.get("max_seq_len", 4096)),
                dtype          = torch.bfloat16,       # f16/bf16 are fine for conversion step
                load_in_4bit   = False,                # GGUF conversion expects full precision state
                fast_inference = False,
            )

            logging.info(f"Exporting GGUF -> dir={gguf_dir}, quant={quant}")
            export_model.save_pretrained_gguf(
                gguf_dir,
                export_processor,
                quantization_method = quant,
                # You can also set maximum_memory_usage to reduce OOM risks while saving:
                maximum_memory_usage = 0.6,
            )
            logging.info(f"GGUF exported to directory: {gguf_dir}")

        except Exception as e:
            logging.warning(f"GGUF export failed: {e}")
            logging.warning("Tip: make sure you're on latest Unsloth. See GGUF docs for supported quants.")
            gguf_dir = None

    write_manifest(
        run_dir              = run_dir,
        config_copied_path   = config_copy_path,
        logs_path            = logs_path,
        trainer_state_path   = trainer_state_path,
        log_history_path     = trainer_log_hist_path,
        final_metrics_path   = final_metrics_path,
        adapter_dir          = adapter_dir,
        merged_dir           = merged_dir,
        gguf_dir             = gguf_dir,
        best_model_checkpoint= best_ckpt,
    )

    logging.info("All done.")
    logging.info(f"Artifacts in: {run_dir}")

if __name__ == "__main__":
    main()
