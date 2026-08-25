#!/bin/bash
#SBATCH --job-name=soil_dl_sec16
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/section16_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/section16_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "========================================"
echo "  SLURM Job : $SLURM_JOB_ID"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "========================================"

# ── FIX 1: Load modules with correct names ───────────────────
module purge
# Try different CUDA module names — at least one should work
module load cuda 2>/dev/null || module load cuda/12.8 2>/dev/null || module load cuda/11.8 2>/dev/null || module load CUDA 2>/dev/null || echo "WARNING: No CUDA module loaded — using system CUDA"

# Show what loaded
module list 2>&1
nvidia-smi

# ── FIX 2: Use direct Python path — skip conda activate ──────
# Find the correct Python directly
CONDA_PYTHON="/home/emmanuel.keku/miniconda3/envs/soil_dl/bin/python"

# Verify it exists and has packages
if [ -f "$CONDA_PYTHON" ]; then
    echo "Using conda Python: $CONDA_PYTHON"
    PY="$CONDA_PYTHON"
else
    # Search for it
    PY=$(find /home/emmanuel.keku /cm/shared/apps         -name python3         -path "*soil_dl*"         2>/dev/null | head -1)
    if [ -z "$PY" ]; then
        echo "ERROR: Cannot find soil_dl Python"
        echo "Trying system Python3..."
        PY=$(which python3)
    fi
fi

echo "Python path: $PY"
$PY --version

# ── FIX 3: Verify packages exist ─────────────────────────────
$PY -c "
import sys
print('Python:', sys.executable)
try:
    import pandas
    print('pandas     :', pandas.__version__)
except ImportError:
    print('pandas     : MISSING — installing...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip',
        'install', '--user', 'pandas',
        'numpy', 'scikit-learn', 'matplotlib',
        'einops', '-q'])

try:
    import numpy
    print('numpy      :', numpy.__version__)
except ImportError:
    print('numpy      : MISSING')

import torch
print('torch      :', torch.__version__)
print('CUDA       :', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        n = torch.cuda.get_device_name(i)
        m = torch.cuda.get_device_properties(i).total_memory/1e9
        print(f'GPU {i}     : {n} | {m:.1f} GB')
"

# ── FIX 4: Install CUDA PyTorch if still CPU-only ────────────
$PY -c "import torch; assert torch.cuda.is_available()" || {
    echo "Installing CUDA PyTorch..."
    $PY -m pip install --user         torch torchvision torchaudio         --index-url         https://download.pytorch.org/whl/cu118 -q
}

# ── FIX 5: Skip mamba-ssm — use pure PyTorch Mamba ──────────
# mamba-ssm requires nvcc compiler — not available here
# Our soil_dl_models.py implements Mamba in pure PyTorch
# so mamba-ssm package is NOT needed
echo "Skipping mamba-ssm install (not needed — pure PyTorch Mamba)"

# ── FIX 6: Install only safe packages ────────────────────────
$PY -m pip install --user     einops -q

cd /home/emmanuel.keku
export PYTHONPATH=/home/emmanuel.keku:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Starting training at $(date)..."
$PY /home/emmanuel.keku/logs/section16_train.py     2>&1 | tee /home/emmanuel.keku/logs/section16_training.log

echo "========================================"
echo "  DONE : $(date)"
echo "========================================"
