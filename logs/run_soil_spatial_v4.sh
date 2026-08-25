#!/bin/bash
#SBATCH --job-name=soil_spatial_v4
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/soil_spatial_v4_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/soil_spatial_v4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  SOIL SPATIAL v4 — DISTRIBUTED AI"
echo "  11 Models | Spatial Holdout | DeepESN | GraphSAGE"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname)"
echo "  Start: $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
echo "Python: $PY"; $PY --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY -m pip install --user "numpy==1.24.4" -q --no-deps
$PY -m pip install --user "PyWavelets>=1.3" "scikit-learn>=1.0" "xgboost>=1.6" "seaborn>=0.12" -q

$PY -c "
import numpy,scipy,torch,pandas,pywt,sklearn,seaborn
print('numpy :', numpy.__version__)
print('torch :', torch.__version__)
print('CUDA  :', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU   :', torch.cuda.get_device_name(0))
print('ALL IMPORTS OK')
" || { echo "FATAL: import check failed"; exit 1; }

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/models_v4/dl
mkdir -p /home/emmanuel.keku/results_v4
mkdir -p /home/emmanuel.keku/figures_v4

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "======================================================"
echo "  TRAINING START: $(date)"
echo "======================================================"

$PY /home/emmanuel.keku/train_soil_spatial_v4.py \
    2>&1 | tee /home/emmanuel.keku/logs/soil_training_v4.log

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

find /home/emmanuel.keku/models_v4/dl -name "*.pt"  -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/results_v4   -name "*.csv" -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/figures_v4   -name "*.png" -printf "  %p (%k KB)\n" | sort

exit $EXIT
