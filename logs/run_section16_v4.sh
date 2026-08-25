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

echo "===================================="
echo "  Job : $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "===================================="

# Correct modules for this cluster
module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

module list 2>&1
nvidia-smi

# Python from loaded module
PY=$(which python3)
echo "Python: $PY"
$PY --version

# Upgrade numpy BEFORE loading scalers.pkl
echo "Upgrading numpy..."
$PY -m pip install --user "numpy>=1.24" -q
$PY -c "import numpy; print('numpy:', numpy.__version__)"

# Verify all imports
$PY -c "
import torch
print('torch :', torch.__version__)
print('CUDA  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU   :', torch.cuda.get_device_name(0))
import pandas; print('pandas:', pandas.__version__)
import numpy;  print('numpy :', numpy.__version__)
print('ALL OK')
"

# Install extras
$PY -m pip install --user einops scikit-learn -q

cd /home/emmanuel.keku
export PYTHONPATH=/home/emmanuel.keku:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Training start: $(date)"
$PY /home/emmanuel.keku/logs/section16_train.py     2>&1 | tee     /home/emmanuel.keku/logs/section16_training.log

echo "Done: $(date)"
