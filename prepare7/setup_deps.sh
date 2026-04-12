#!/bin/bash
# Create a persistent conda env for ProtoMotions + Newton on shared storage.
# Lives at /mnt/vita/scratch so it survives container restarts.
#
# Usage (inside runai bash intintermask):
#   bash prepare7/setup_deps.sh          # create env + install deps
#   bash prepare7/setup_deps.sh --force  # recreate from scratch
#
# To activate manually:
#   source /mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh
#   conda activate protomotions

set -e

CONDA_SH="/mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="protomotions"

# Fix RunAI container missing passwd entry (PyTorch 2.11+ needs getpass.getuser())
export USER="${USER:-$(id -u)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$$}"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh not found at $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"

# Check if env already exists and is working
if conda env list | grep -q "$ENV_NAME" && [ "${1:-}" != "--force" ]; then
    echo "Conda env '$ENV_NAME' already exists. Checking..."
    conda activate "$ENV_NAME"
    if python -c "import lightning, newton, warp, torch" 2>/dev/null; then
        echo "All imports OK. Python: $(which python)"
        exit 0
    fi
    echo "Some imports failed. Will fix by installing missing packages..."
fi

if [ "${1:-}" = "--force" ]; then
    echo "Removing existing env '$ENV_NAME'..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# Create env if it doesn't exist
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "=== Creating conda env: $ENV_NAME (Python 3.11) ==="
    conda create -n "$ENV_NAME" python=3.11 -y
fi

echo "=== Activating $ENV_NAME ==="
conda activate "$ENV_NAME"

echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== Installing ProtoMotions dependencies ==="
pip install \
    lightning \
    "omegaconf==2.3.0" \
    dm_control \
    rich \
    typer \
    termcolor \
    trimesh \
    easydict \
    tensordict \
    warp-lang \
    mujoco-warp \
    pandas \
    tensorboardX \
    scipy \
    tqdm \
    scikit-image \
    wandb \
    transformers

echo "=== Installing Newton simulator (commit e7a737c) ==="
pip install git+https://github.com/newton-physics/newton.git@e7a737c

# Fix RunAI container missing passwd entry (PyTorch 2.11+ needs getpass.getuser())
export USER="${USER:-$(id -u)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$$}"

echo "=== Installing EGL/GLVND for headless MuJoCo rendering ==="
conda install -c conda-forge libegl libglvnd -y 2>/dev/null || echo "WARN: conda libegl install failed (non-fatal)"

# Set up NVIDIA EGL vendor JSON so the GLVND dispatcher finds the GPU driver
mkdir -p "${CONDA_PREFIX}/share/glvnd/egl_vendor.d"
cat > "${CONDA_PREFIX}/share/glvnd/egl_vendor.d/10_nvidia.json" << 'EJSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
EJSON
echo "  EGL vendor JSON: ${CONDA_PREFIX}/share/glvnd/egl_vendor.d/10_nvidia.json"

echo "=== Installing video recording deps ==="
pip install moviepy matplotlib 2>/dev/null || echo "WARN: moviepy/matplotlib install failed (non-fatal)"

echo "=== Verifying imports ==="
python -c "
import lightning; print(f'  lightning {lightning.__version__}')
import newton; print(f'  newton OK')
import warp; print(f'  warp {warp.__version__}')
import torch; print(f'  torch {torch.__version__} CUDA={torch.cuda.is_available()}')
import dm_control; print(f'  dm_control OK')
"

echo ""
echo "=== Done! Env '$ENV_NAME' is ready ==="
echo "Activate with: source $CONDA_SH && conda activate $ENV_NAME"
