#!/bin/bash
#SBATCH --job-name=misar_v7
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/home/emmanuel.keku/logs/misar_v7_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/misar_v7_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  SpatialMISAR v7 — Behavior-Guided Distributed AI"
echo "  STGCN | Soil Temperature | T=18 rounds | N=4 workers"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/results_v7
mkdir -p /home/emmanuel.keku/figures_v7

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1

python3 /home/emmanuel.keku/spatial_misar_v7.py \
    2>&1 | tee /home/emmanuel.keku/logs/spatial_misar_v7.log

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"
ls -lh /home/emmanuel.keku/figures_v7/*.gif 2>/dev/null
ls -lh /home/emmanuel.keku/figures_v7/*.png 2>/dev/null
exit $EXIT
