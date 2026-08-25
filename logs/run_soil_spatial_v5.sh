#!/bin/bash
# ==============================================================================
# run_soil_spatial_v5.sh — SLURM batch script for v5 training
#
# v5 additions vs v4:
#   - Regularisation: DP=0.15, WD=5e-4, L1=1e-5
#   - MC-Dropout uncertainty quantification (N=30)
#   - Magnitude-based weight pruning (20% sparsity)
#   - Stability benchmark mode (--mode stability)
#
# USAGE:
#   sbatch ~/logs/run_soil_spatial_v5.sh             # full training
#   sbatch ~/logs/run_soil_spatial_v5.sh stability   # 10-run stability bench
#   sbatch ~/logs/run_soil_spatial_v5.sh uncertainty # MC-Dropout on checkpoints
# ==============================================================================
#SBATCH --job-name=soil_spatial_v5
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/soil_spatial_v5_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/soil_spatial_v5_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

MODE=${1:-train}   # train | stability | uncertainty

echo "======================================================"
echo "  SOIL SPATIAL v5 — REGULARISATION + UNCERTAINTY"
echo "  Mode: $MODE"
echo "  11 Models | Spatial Holdout | DP=0.15 WD=5e-4 L1=1e-5 Prune=20%"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname)"
echo "  Start: $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
$PY --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY -m pip install --user "numpy==1.24.4" -q --no-deps
$PY -m pip install --user "PyWavelets>=1.3" "scikit-learn>=1.0" \
    "xgboost>=1.6" "seaborn>=0.12" -q

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/models_v5/dl
mkdir -p /home/emmanuel.keku/results_v5
mkdir -p /home/emmanuel.keku/figures_v5

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "======================================================"
echo "  START: $(date)"
echo "======================================================"

if [ "$MODE" = "stability" ]; then
    $PY /home/emmanuel.keku/train_soil_spatial_v5.py \
        --mode stability --n_runs 10 \
        2>&1 | tee /home/emmanuel.keku/logs/soil_v5_stability.log

elif [ "$MODE" = "uncertainty" ]; then
    $PY /home/emmanuel.keku/train_soil_spatial_v5.py \
        --mode uncertainty --mc_samples 30 \
        2>&1 | tee /home/emmanuel.keku/logs/soil_v5_uncertainty.log
    $PY /home/emmanuel.keku/uncertainty_analysis_v5.py \
        2>&1 | tee -a /home/emmanuel.keku/logs/soil_v5_uncertainty.log

else
    # Full training pipeline
    $PY /home/emmanuel.keku/train_soil_spatial_v5.py \
        --mode train --mc_samples 30 \
        2>&1 | tee /home/emmanuel.keku/logs/soil_training_v5.log

    # Post-process: uncertainty figures
    $PY /home/emmanuel.keku/uncertainty_analysis_v5.py \
        2>&1 | tee -a /home/emmanuel.keku/logs/soil_training_v5.log
fi

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

find /home/emmanuel.keku/models_v5/dl -name "*.pt"  -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/results_v5   -name "*.csv" -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/figures_v5   -name "*.png" -printf "  %p (%k KB)\n" | sort

exit $EXIT
