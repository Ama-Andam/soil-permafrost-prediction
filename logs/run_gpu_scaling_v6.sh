#!/bin/bash
# ==============================================================================
# run_gpu_scaling_v6.sh
# GPU SCALING EXPERIMENT — nn.DataParallel (1,2,4,8 GPUs)
# Requests 8 GPUs, runs all 4 configs sequentially on same node
# ==============================================================================
#SBATCH --job-name=gpu_scale_v6
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/gpu_scaling_v6_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/gpu_scaling_v6_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  GPU SCALING v6 — nn.DataParallel"
echo "  Testing: 1, 2, 4, 8 GPUs"
echo "  13 models | 3 targets | 15 epochs per config"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
$PY --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/results_v6
mkdir -p /home/emmanuel.keku/figures_v6/publication

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1

# Remove old scaling results to force fresh run with fixed script
rm -f /home/emmanuel.keku/results_v6/v6_scaling_results.csv
echo "Cleared old scaling results — running fresh with GraphAwareWrapper fix"

echo ""
echo "======================================================"
echo "  Running all targets: temp, smap, moist"
echo "  GPU configs: 1,2,4,8"
echo "  GraphAwareWrapper: A replicated (not scattered) to each GPU"
echo "======================================================"

for TARGET in temp smap moist; do
    echo ""
    echo "------------------------------------------------------"
    echo "  TARGET: $TARGET | $(date)"
    echo "------------------------------------------------------"
    $PY /home/emmanuel.keku/gpu_scaling_v6.py \
        --gpus 1,2,4,8 \
        --epochs 30 \
        --target $TARGET \
        2>&1 | tee -a /home/emmanuel.keku/logs/gpu_scaling_v6.log
done

echo ""
echo "======================================================"
echo "  Generating publication figures..."
echo "======================================================"
$PY /home/emmanuel.keku/pub_figures_v6.py \
    2>&1 | tee -a /home/emmanuel.keku/logs/gpu_scaling_v6.log

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

find /home/emmanuel.keku/results_v6 -name "v6_scaling*" -printf "  %p (%k KB)\n"
find /home/emmanuel.keku/figures_v6/publication -name "*.png" -printf "  %p\n" | sort

exit $EXIT
