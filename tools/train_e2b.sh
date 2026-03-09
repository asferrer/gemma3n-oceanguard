#!/bin/bash
# ==============================================================================
# Training Script - Gemma3n E2B Training - Mobile Optimized
# ==============================================================================
# This script handles the complete training workflow:
#   1. Environment verification (GPU, CUDA, dataset)
#   2. Dataset verification (Docker and local paths)
#   3. Configuration check
#   4. TensorBoard startup
#   5. Training execution
#
# Usage:
#   docker exec -it gemma3n-training bash /workspace/project/tools/train_e2b.sh
#   docker exec -it gemma3n-training bash /workspace/project/tools/train_e2b.sh --gpu rtx5090
#   docker exec -it gemma3n-training bash /workspace/project/tools/train_e2b.sh --config /path/to/config.yaml
#   docker exec -it gemma3n-training bash /workspace/project/tools/train_e2b.sh --skip-tensorboard
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# CLI Argument Parsing
# ==============================================================================

CONFIG_OVERRIDE=""
GPU_MODE="auto"
SKIP_TENSORBOARD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --gpu)
            GPU_MODE="$2"
            shift 2
            ;;
        --skip-tensorboard)
            SKIP_TENSORBOARD=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config PATH] [--gpu rtx5090|rtx2070|auto] [--skip-tensorboard]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT="/workspace/project"
TRAIN_SCRIPT="${PROJECT_ROOT}/tools/train.py"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
TENSORBOARD_PORT=6006

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo "=================================="
    echo "$1"
    echo "=================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ==============================================================================
# Step 1: Environment Verification
# ==============================================================================

print_header "Gemma3n E2B Training - Mobile Optimized"

print_info "Step 1/5: Verifying environment..."

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found!"
    exit 1
fi
print_success "Python: $(python --version)"

# Check CUDA availability
print_info "Checking CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    print_error "CUDA is not available!"
    print_info "Make sure Docker has GPU access:"
    print_info "  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi"
    exit 1
fi
print_success "CUDA is available"

# Detect GPU and select config
print_info "Detecting GPU..."
GPU_INFO=$(python -c "import torch; print(f'{torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}')")
GPU_CAPABILITY=$(python -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'{cap[0]},{cap[1]}')")
print_info "Detected: $GPU_INFO"

# Resolve GPU mode (CLI override takes priority over auto-detection)
if [ "$GPU_MODE" = "auto" ]; then
    if [ "$GPU_CAPABILITY" = "12,0" ]; then
        GPU_MODE="rtx5090"
    elif [ "$GPU_CAPABILITY" = "7,5" ]; then
        GPU_MODE="rtx2070"
    else
        print_warning "Unknown compute capability: $GPU_CAPABILITY"
        print_warning "Unknown GPU, using RTX 2070 config (conservative)"
        GPU_MODE="rtx2070"
    fi
fi

# Select config file based on resolved GPU mode (unless --config was provided)
if [ -n "$CONFIG_OVERRIDE" ]; then
    CONFIG_FILE="$CONFIG_OVERRIDE"
    print_info "Config override provided via --config: $CONFIG_FILE"
else
    case "$GPU_MODE" in
        rtx5090)
            CONFIG_FILE="${PROJECT_ROOT}/tools/config.e2b.rtx5090.yaml"
            print_success "RTX 5090 (Blackwell sm_120) selected"
            print_info "Expected training time: 3-5 hours, effective batch: 64"
            ;;
        rtx2070)
            CONFIG_FILE="${PROJECT_ROOT}/tools/config.e2b.rtx2070.yaml"
            print_success "RTX 2070 selected"
            print_info "Expected training time: 15-20 hours, effective batch: 16"
            ;;
        *)
            print_error "Invalid --gpu value: $GPU_MODE. Use rtx5090, rtx2070, or auto."
            exit 1
            ;;
    esac
fi

# Check Flash Attention
print_info "Checking Flash Attention 2..."
if python -c "import flash_attn" 2>/dev/null; then
    FA_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)")
    print_success "Flash Attention $FA_VERSION installed"
else
    print_warning "Flash Attention not available (will use eager attention)"
fi

# Check Unsloth
print_info "Checking Unsloth..."
if python -c "import unsloth" 2>/dev/null; then
    print_success "Unsloth installed"
else
    print_error "Unsloth not found!"
    exit 1
fi

# ==============================================================================
# Step 2: Dataset Verification
# ==============================================================================

print_header "Step 2/5: Verifying dataset"

# Auto-select dataset location: Docker path takes priority, then local paths
DATASET_DIR=""

DOCKER_DATASET_DIR="${PROJECT_ROOT}/prepared_dataset_docker"
LOCAL_DATASET_DIR_A="./prepared_dataset"
LOCAL_DATASET_DIR_B="${PROJECT_ROOT}/prepared_dataset"

if [ -f "${DOCKER_DATASET_DIR}/train_dataset.jsonl" ]; then
    DATASET_DIR="$DOCKER_DATASET_DIR"
    print_info "Using Docker dataset path: $DATASET_DIR"
elif [ -f "${LOCAL_DATASET_DIR_B}/train_dataset.jsonl" ]; then
    DATASET_DIR="$LOCAL_DATASET_DIR_B"
    print_info "Using local dataset path: $DATASET_DIR"
elif [ -f "${LOCAL_DATASET_DIR_A}/train_dataset.jsonl" ]; then
    DATASET_DIR="$LOCAL_DATASET_DIR_A"
    print_info "Using local dataset path: $DATASET_DIR"
else
    print_error "Training dataset not found in any expected location:"
    print_error "  Docker: ${DOCKER_DATASET_DIR}/train_dataset.jsonl"
    print_error "  Local:  ${LOCAL_DATASET_DIR_B}/train_dataset.jsonl"
    print_error "  Local:  ${LOCAL_DATASET_DIR_A}/train_dataset.jsonl"
    print_info "Make sure you ran prepare_for_docker.py first"
    exit 1
fi

TRAIN_DATASET="${DATASET_DIR}/train_dataset.jsonl"
VAL_DATASET="${DATASET_DIR}/validation_dataset.jsonl"

if [ ! -f "$VAL_DATASET" ]; then
    print_error "Validation dataset not found: $VAL_DATASET"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_DATASET")
VAL_COUNT=$(wc -l < "$VAL_DATASET")

print_success "Training samples: $TRAIN_COUNT"
print_success "Validation samples: $VAL_COUNT"

# Quick format check
print_info "Validating dataset format..."
if python -c "
import json
with open('$TRAIN_DATASET', 'r') as f:
    sample = json.loads(f.readline())
    assert 'messages' in sample
    assert len(sample['messages']) == 2
    assert sample['messages'][0]['role'] == 'user'
    assert sample['messages'][1]['role'] == 'assistant'
print('OK')
"; then
    print_success "Dataset format is valid"
else
    print_error "Dataset format validation failed"
    exit 1
fi

# ==============================================================================
# Step 3: Configuration Check
# ==============================================================================

print_header "Step 3/5: Checking configuration"

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_success "Config file: $CONFIG_FILE"

# Display key config parameters
print_info "Key configuration parameters:"
python - <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(f"  - Model: {config['training']['model_name']}")
print(f"  - Batch size: {config['training']['per_device_train_batch_size']}")
print(f"  - Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
print(f"  - Effective batch: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
print(f"  - Learning rate: {config['training']['lr']}")
print(f"  - LoRA rank: {config['peft']['r']}")
print(f"  - Max epochs: {config['training']['epochs']}")
print(f"  - Max seq length: {config['training']['max_seq_len']}")
EOF

# ==============================================================================
# Step 4: Start TensorBoard
# ==============================================================================

print_header "Step 4/5: Starting TensorBoard"

if [ "$SKIP_TENSORBOARD" = true ]; then
    print_info "Skipping TensorBoard startup (--skip-tensorboard flag set)"
    TENSORBOARD_PID=""
else
    # Kill any existing TensorBoard processes
    pkill -f tensorboard 2>/dev/null || true

    # Start TensorBoard in background
    print_info "Starting TensorBoard on port $TENSORBOARD_PORT..."
    tensorboard --logdir="$OUTPUTS_DIR" --host=0.0.0.0 --port=$TENSORBOARD_PORT \
        --reload_interval=30 > /tmp/tensorboard.log 2>&1 &
    TENSORBOARD_PID=$!

    sleep 2  # Give it time to start

    if ps -p $TENSORBOARD_PID > /dev/null; then
        print_success "TensorBoard started (PID: $TENSORBOARD_PID)"
        print_info "Access TensorBoard at: http://localhost:$TENSORBOARD_PORT"
    else
        print_warning "TensorBoard failed to start (training will continue)"
        TENSORBOARD_PID=""
    fi
fi

# ==============================================================================
# Step 5: Run Training
# ==============================================================================

print_header "Step 5/5: Starting training"

print_info "Training will run with the following specs:"
case "$GPU_MODE" in
    rtx5090)
        print_info "  - GPU: RTX 5090 (32GB GDDR7)"
        print_info "  - Expected training time: 3-5 hours"
        print_info "  - Effective batch size: 64"
        ;;
    rtx2070)
        print_info "  - GPU: RTX 2070 (8GB GDDR6)"
        print_info "  - Expected training time: 15-20 hours"
        print_info "  - Effective batch size: 16"
        ;;
esac
print_info "  - Checkpoints every 150 steps"
print_info "  - Early stopping enabled (patience=5)"
print_info ""
print_info "Press Ctrl+C to stop training gracefully"
print_info ""

# Record start time
START_TIME=$(date +%s)

# Run training (with error handling)
if python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"; then
    TRAIN_EXIT_CODE=0
    TRAIN_STATUS="SUCCESS"
else
    TRAIN_EXIT_CODE=$?
    TRAIN_STATUS="FAILED"
fi

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# ==============================================================================
# Cleanup & Final Report
# ==============================================================================

print_header "Training Complete"

# Stop TensorBoard
if [ -n "${TENSORBOARD_PID:-}" ]; then
    print_info "Stopping TensorBoard (PID: $TENSORBOARD_PID)..."
    kill $TENSORBOARD_PID 2>/dev/null || true
fi

# Print summary
echo ""
echo "=================================="
echo "Training Summary"
echo "=================================="
echo ""
echo "Status: $TRAIN_STATUS"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Exit code: $TRAIN_EXIT_CODE"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    print_success "Training completed successfully!"
    echo ""
    print_info "Next steps:"
    print_info "  1. Check outputs directory:"
    print_info "     ls -lh $OUTPUTS_DIR/gemma3n/runs/"
    print_info ""
    print_info "  2. Evaluate model performance:"
    print_info "     python tools/evaluate_models.py \\"
    print_info "       --model-path $OUTPUTS_DIR/gemma3n/runs/<run_name> \\"
    print_info "       --save-visuals \\"
    print_info "       --compare"
    print_info ""

    # Find and display latest run directory
    LATEST_RUN=$(ls -td $OUTPUTS_DIR/gemma3n/runs/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        print_info "Latest run directory:"
        print_info "  $LATEST_RUN"

        # Show size
        RUN_SIZE=$(du -sh "$LATEST_RUN" | cut -f1)
        print_info "  Size: $RUN_SIZE"

        # Check for merged model
        if [ -d "${LATEST_RUN}/merged_model" ]; then
            print_success "Merged model available at: ${LATEST_RUN}/merged_model"
        fi
    fi

    echo ""
    echo "=================================="
    echo "Next steps for mobile deployment:"
    echo "=================================="
    echo ""
    print_info "  1. Convert to LiteRT:"
    print_info "     python mobile/conversion/convert_base_model.py"
    print_info ""
    print_info "  2. Convert LoRA adapter:"
    print_info "     python mobile/conversion/convert_lora_adapter.py"
    echo ""
else
    print_error "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo ""
    print_info "Troubleshooting steps:"
    print_info "  1. Check logs in: $OUTPUTS_DIR/gemma3n/runs/"
    print_info "  2. Verify dataset format"
    print_info "  3. Check VRAM usage (reduce batch size if OOM)"
    print_info "  4. Review error messages above"
    echo ""
fi

echo "=================================="
echo ""

exit $TRAIN_EXIT_CODE
