#!/bin/bash
#SBATCH --job-name=soil_spatial_v2
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/spatial_v2_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/spatial_v2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "====================================="
echo "  Job  : $SLURM_JOB_ID"
echo "  Node : $(hostname)"
echo "  Start: $(date)"
echo "====================================="

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
echo "Python: $PY"
$PY --version
nvidia-smi

# Upgrade numpy to avoid _core import errors
$PY -m pip install --user "numpy>=1.24" -q
$PY -m pip install --user einops scipy scikit-learn PyWavelets xgboost lightgbm shap -q

$PY -c "
import torch
print('torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU    :', torch.cuda.get_device_name(0))
import numpy; print('numpy  :', numpy.__version__)
print('ALL OK')
"

cd /home/emmanuel.keku
export PYTHONPATH=/home/emmanuel.keku:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Training start: $(date)"
$PY /home/emmanuel.keku/train_soil_spatial.py \
    2>&1 | tee /home/emmanuel.keku/logs/soil_spatial_v2_training.log

echo "Done: $(date)"
