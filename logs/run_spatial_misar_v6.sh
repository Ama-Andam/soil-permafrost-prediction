#!/bin/bash
# ==============================================================================
# run_spatial_misar_v6.sh — SpatialMISAR Phase 2
# Parallel Weight Space Exploration
# ==============================================================================
#SBATCH --job-name=misar_v6
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/misar_v6_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/misar_v6_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  SpatialMISAR v6 — Parallel Weight Space Exploration"
echo "  5 models × 4 N configs × 4 strategies × 3 targets"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/results_v6
mkdir -p /home/emmanuel.keku/figures_v6/manuscript
mkdir -p /home/emmanuel.keku/models_v6/misar

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1

python3 /home/emmanuel.keku/spatial_misar_v6.py \
    2>&1 | tee /home/emmanuel.keku/logs/spatial_misar_v6.log

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

# List output files
ls -lh /home/emmanuel.keku/results_v6/spatial_misar*.csv 2>/dev/null
ls /home/emmanuel.keku/figures_v6/manuscript/MISAR_*.png 2>/dev/null | wc -l

exit $EXIT
