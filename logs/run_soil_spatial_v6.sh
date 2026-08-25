#!/bin/bash
# ==============================================================================
# run_soil_spatial_v6.sh — v6 SLURM batch script
#
# MODES:
#   sbatch run_soil_spatial_v6.sh train      — full training (default)
#   sbatch run_soil_spatial_v6.sh tune       — 50-trial random search tuning
#   sbatch run_soil_spatial_v6.sh ablation   — full ablation study
#   sbatch run_soil_spatial_v6.sh figures    — generate all figures
#
# RECOMMENDED ORDER:
#   1. tune      (~8h)  — find best hparams per model
#   2. train     (~15h) — full training with best hparams
#   3. ablation  (~6h)  — component removal study
#   4. figures   (~1h)  — all visualisations
# ==============================================================================
#SBATCH --job-name=soil_v6
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/soil_v6_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/soil_v6_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

MODE=${1:-train}

echo "======================================================"
echo "  SOIL SPATIAL v6 — FULL REDESIGN"
echo "  Mode: $MODE"
echo "  13 Models (added SpatialTransformer + SpatialInformer)"
echo "  3 test sets: unseen-space, unseen-time, unseen-both"
echo "  Heteroscedastic output: μ + σ² simultaneously"
echo "  Metrics: R², KGE, ubRMSE, CRPS, DTW, KL Div, NLL"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
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
    "xgboost>=1.6" "seaborn>=0.12" "scipy>=1.9" -q

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/models_v6/dl
mkdir -p /home/emmanuel.keku/results_v6
mkdir -p /home/emmanuel.keku/figures_v6

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "======================================================"
echo "  START: $(date)"
echo "======================================================"

$PY /home/emmanuel.keku/train_soil_spatial_v6.py --mode $MODE \
    2>&1 | tee /home/emmanuel.keku/logs/soil_${MODE}_v6.log

# Auto-generate figures after training
if [ "$MODE" = "train" ]; then
    echo "Generating figures..."
    $PY /home/emmanuel.keku/figures_v6.py \
        2>&1 | tee -a /home/emmanuel.keku/logs/soil_train_v6.log
fi

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

find /home/emmanuel.keku/models_v6/dl -name "*.pt"  -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/results_v6   -name "*.csv" -printf "  %p (%k KB)\n" | sort
find /home/emmanuel.keku/figures_v6   -name "*.png" -printf "  %p (%k KB)\n" | sort
exit $EXIT
