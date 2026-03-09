# ==============================================================================
# Dockerfile for Gemma3N Fine-tuning with Official Unsloth Image
# Base: unsloth/unsloth (includes all dependencies pre-configured)
# Supports: Gemma 3n, latest Unsloth, Flash Attention 2, CUDA 12.x
# ==============================================================================

FROM unsloth/unsloth:latest

# Metadata
LABEL maintainer="asanc.tech@gmail.com"
LABEL description="Gemma3N Fine-tuning with Unsloth - Latest Version"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Hugging Face cache directories
    HF_HOME=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    # Optimize for CUDA
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0" \
    FORCE_CUDA=1

# Switch to root for system installations
USER root

# ==============================================================================
# Update system packages and install additional tools
# ==============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Development tools
    vim \
    nano \
    htop \
    tmux \
    tree \
    # Utilities
    zip \
    unzip \
    wget \
    curl \
    git \
    # For monitoring
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Upgrade Unsloth to absolute latest version (important for Gemma3n)
# ==============================================================================

RUN pip install --upgrade --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ==============================================================================
# Update core dependencies to latest compatible versions
# OPTIMIZED: Updated datasets version and added packaging
# ==============================================================================

RUN pip install --upgrade --no-cache-dir \
    # Core ML libraries (versiones compatibles con Unsloth)
    "transformers>=4.47.0,<4.50.0" \
    "datasets>=4.5.0,<5.0.0" \
    accelerate>=1.2.0 \
    peft>=0.13.2 \
    trl>=0.12.0 \
    bitsandbytes>=0.44.0 \
    # Utilities
    pyyaml \
    omegaconf \
    hydra-core \
    packaging \
    # Progress and monitoring
    tqdm \
    rich \
    gpustat \
    nvitop \
    # File handling
    pandas \
    pyarrow \
    # Development tools
    ipython \
    ipywidgets

# ==============================================================================
# Install Flash Attention 2 for Linux (critical for performance)
# OPTIMIZED: Better error handling and verification
# ==============================================================================

RUN pip install --no-cache-dir ninja packaging wheel && \
    (pip install --no-cache-dir flash-attn --no-build-isolation && \
     python -c "import flash_attn; print(f'✓ Flash Attention {flash_attn.__version__} installed successfully')" || \
     echo "⚠ Warning: Flash Attention installation failed, will use eager attention")

# ==============================================================================
# Install/update additional helpful packages
# ==============================================================================

RUN pip install --upgrade --no-cache-dir \
    # For better logging
    tensorboard \
    # For vision tasks
    pillow>=10.0.0 \
    opencv-python-headless \
    # Code formatting (optional)
    black \
    isort \
    # For evaluation and visualization
    matplotlib>=3.8.0 \
    seaborn>=0.12.0 \
    plotly>=5.17.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0

# ==============================================================================
# Setup workspace structure
# ==============================================================================

WORKDIR /workspace/project

RUN mkdir -p \
    /workspace/project/outputs \
    /workspace/project/prepared_dataset \
    /workspace/project/configs \
    /workspace/project/scripts \
    /workspace/project/notebooks \
    /workspace/.cache/huggingface/transformers \
    /workspace/.cache/huggingface/datasets \
    /workspace/.cache/huggingface/hub

# ==============================================================================
# Set permissions for unsloth user
# ==============================================================================

RUN chown -R unsloth:unsloth /workspace

# Switch back to unsloth user (non-root for security)
USER unsloth

# ==============================================================================
# Verify installations and print versions (sin GPU durante build)
# ==============================================================================

# Verificaciones que NO requieren GPU (imports básicos)
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Version:', torch.version.cuda)" && \
    python -c "import transformers; print('Transformers:', transformers.__version__)" && \
    python -c "import trl; print('TRL:', trl.__version__)" && \
    python -c "import peft; print('PEFT:', peft.__version__)" && \
    python -c "import bitsandbytes; print('BitsAndBytes:', bitsandbytes.__version__)" && \
    echo "Flash Attention: Will be verified at runtime"

# Nota:
# - Unsloth requiere GPU, se verificará cuando el contenedor inicie
# - Flash Attention se verificará en runtime
# - CUDA available se verifica en runtime (requiere GPU)

# ==============================================================================
# Healthcheck - OPTIMIZED: Verify GPU availability
# ==============================================================================

# Enhanced healthcheck: Verifies Python, PyTorch, and GPU availability
# This ensures the container is ready for training
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('OK')" || exit 1

# ==============================================================================
# Default command
# ==============================================================================

# Keep container running for interactive use
CMD ["/bin/bash"]