#!/bin/bash
# ==============================================================================
# run_baseline_v6.sh — SLURM script for v6 baseline comparison
# Runs after v6 DL training completes for full DL vs ML comparison figure.
# Can also run standalone to get ML baseline results only.
# ==============================================================================
#SBATCH --job-name=baseline_v6
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --output=/home/emmanuel.keku/logs/baseline_v6_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/baseline_v6_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  v6 BASELINE COMPARISON"
echo "  Same features as v6 DL: no cyclical, wavelet approx"
echo "  3 test sets: std, unseen-space, unseen-time"
echo "  Metrics: R², KGE, ubRMSE, CRPS, DTW, KL Div"
echo "  Job: $SLURM_JOB_ID | $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
$PY -m pip install --user "xgboost>=1.6" "lightgbm" "seaborn>=0.12" "scipy>=1.9" -q

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1

$PY /home/emmanuel.keku/baseline_comparison_v6.py \
    2>&1 | tee /home/emmanuel.keku/logs/baseline_v6.log

echo "Done: $(date)"
