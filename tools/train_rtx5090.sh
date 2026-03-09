#!/bin/bash
# ==============================================================================
# Training Script Optimized for RTX 5090
# ==============================================================================
# This script handles the complete training workflow:
#   1. Environment verification (GPU, CUDA, dataset)
#   2. TensorBoard startup
#   3. Training execution
#   4. Post-training cleanup and reporting
#
# Usage:
#   docker exec -it gemma3n-training-rtx5090 bash /workspace/project/tools/train_rtx5090.sh
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT="/workspace/project"
CONFIG_FILE="${PROJECT_ROOT}/tools/config.rtx5090.yaml"
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

print_header "Gemma3n Training - RTX 5090 Optimized"

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

# Verify RTX 5090 (Blackwell sm_120)
print_info "Verifying RTX 5090 (Blackwell)..."
GPU_INFO=$(python -c "import torch; print(f'{torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}')")
GPU_CAPABILITY=$(python -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'{cap[0]},{cap[1]}')")

if [ "$GPU_CAPABILITY" != "12,0" ]; then
    print_error "Wrong compute capability: $GPU_CAPABILITY"
    print_error "Expected: (12, 0) for Blackwell architecture"
    print_warning "Current GPU: $GPU_INFO"
    print_info ""
    print_info "This means the GPU is falling back to an older architecture."
    print_info "Possible causes:"
    print_info "  1. CUDA version is not 12.8+ (check: nvcc --version)"
    print_info "  2. PyTorch version is not 2.6+ (check: python -c 'import torch; print(torch.__version__)')"
    print_info "  3. Docker image needs to be rebuilt with Dockerfile.rtx5090"
    print_info ""
    print_warning "Training will proceed but with 30-40% performance loss"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "RTX 5090 detected correctly: $GPU_INFO"
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

TRAIN_DATASET="${PROJECT_ROOT}/prepared_dataset_docker/train_dataset.jsonl"
VAL_DATASET="${PROJECT_ROOT}/prepared_dataset_docker/validation_dataset.jsonl"

if [ ! -f "$TRAIN_DATASET" ]; then
    print_error "Training dataset not found: $TRAIN_DATASET"
    print_info "Expected location: /workspace/project/prepared_dataset_docker/"
    print_info "Make sure you ran prepare_for_docker.py first"
    exit 1
fi

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

# ==============================================================================
# Step 5: Run Training
# ==============================================================================

print_header "Step 5/5: Starting training"

print_info "Training will run with the following specs:"
print_info "  - GPU: RTX 5090 (32GB GDDR7)"
print_info "  - Expected duration: 5-7 hours"
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
if [ ! -z "$TENSORBOARD_PID" ]; then
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
    print_info "  3. Export to ONNX for faster inference:"
    print_info "     python tools/inference_optimized.py \\"
    print_info "       --model-path $OUTPUTS_DIR/gemma3n/runs/<run_name>/merged_model \\"
    print_info "       --export-onnx"
    echo ""

    # Find and display latest run directory
    LATEST_RUN=$(ls -td $OUTPUTS_DIR/gemma3n/runs/*/ 2>/dev/null | head -1)
    if [ ! -z "$LATEST_RUN" ]; then
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
