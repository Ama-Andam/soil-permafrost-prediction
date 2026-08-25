#!/bin/bash
# ==============================================================================
# run_model_parallel_v5.sh
# Submits model-level Ray parallelism scaling experiment: 1→2→4→8 GPUs
#
# Each sbatch request is for the target GPU count.
# Results accumulate in results_v5/v5_scaling_results.csv
# Figure auto-generated once ≥2 data points exist.
#
# USAGE:
#   bash ~/logs/run_model_parallel_v5.sh
# ==============================================================================

set -e

LOG=/home/emmanuel.keku/logs
SCRIPT=/home/emmanuel.keku/ray_model_parallel_v5.py

echo "Submitting model-level parallelism scaling experiment..."
echo "Target: temp | 11 models | 4 GPU configs"
echo ""

# 1 GPU baseline
jid1=$(sbatch --job-name=v5_par_1gpu \
    --account=hpcusers --partition=talon-gpu32 --nodelist=talon32 \
    --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:1 --time=04:00:00 \
    --output=$LOG/v5_parallel_1gpu_%j.out --error=$LOG/v5_parallel_1gpu_%j.err \
    --wrap="
module purge
module load cuda11.8/toolkit/11.8.0 cudnn8.6-cuda11.8/8.6.0.163 pytorch-py39-cuda11.8-gcc11/1.13.0
pip install --user ray==2.3.1 pydantic==1.10.13 protobuf>=3.19.5,<5.0.0 -q
cd /home/emmanuel.keku
python3 $SCRIPT --gpus 1 --target temp
" | awk '{print $NF}')
echo "  1 GPU: job $jid1"

# 2 GPUs
jid2=$(sbatch --job-name=v5_par_2gpu \
    --account=hpcusers --partition=talon-gpu32 --nodelist=talon32 \
    --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:2 --time=04:00:00 \
    --output=$LOG/v5_parallel_2gpu_%j.out --error=$LOG/v5_parallel_2gpu_%j.err \
    --dependency=afterok:$jid1 \
    --wrap="
module purge
module load cuda11.8/toolkit/11.8.0 cudnn8.6-cuda11.8/8.6.0.163 pytorch-py39-cuda11.8-gcc11/1.13.0
pip install --user ray==2.3.1 pydantic==1.10.13 protobuf>=3.19.5,<5.0.0 -q
cd /home/emmanuel.keku
python3 $SCRIPT --gpus 2 --target temp
" | awk '{print $NF}')
echo "  2 GPU: job $jid2 (depends on $jid1)"

# 4 GPUs
jid4=$(sbatch --job-name=v5_par_4gpu \
    --account=hpcusers --partition=talon-gpu32 --nodelist=talon32 \
    --ntasks=1 --cpus-per-task=16 --mem=64GB --gres=gpu:4 --time=04:00:00 \
    --output=$LOG/v5_parallel_4gpu_%j.out --error=$LOG/v5_parallel_4gpu_%j.err \
    --dependency=afterok:$jid2 \
    --wrap="
module purge
module load cuda11.8/toolkit/11.8.0 cudnn8.6-cuda11.8/8.6.0.163 pytorch-py39-cuda11.8-gcc11/1.13.0
pip install --user ray==2.3.1 pydantic==1.10.13 protobuf>=3.19.5,<5.0.0 -q
cd /home/emmanuel.keku
python3 $SCRIPT --gpus 4 --target temp
" | awk '{print $NF}')
echo "  4 GPU: job $jid4 (depends on $jid2)"

# 8 GPUs
jid8=$(sbatch --job-name=v5_par_8gpu \
    --account=hpcusers --partition=talon-gpu32 --nodelist=talon32 \
    --ntasks=1 --cpus-per-task=32 --mem=128GB --gres=gpu:8 --time=06:00:00 \
    --output=$LOG/v5_parallel_8gpu_%j.out --error=$LOG/v5_parallel_8gpu_%j.err \
    --dependency=afterok:$jid4 \
    --wrap="
module purge
module load cuda11.8/toolkit/11.8.0 cudnn8.6-cuda11.8/8.6.0.163 pytorch-py39-cuda11.8-gcc11/1.13.0
pip install --user ray==2.3.1 pydantic==1.10.13 protobuf>=3.19.5,<5.0.0 -q
cd /home/emmanuel.keku
python3 $SCRIPT --gpus 8 --target temp
" | awk '{print $NF}')
echo "  8 GPU: job $jid8 (depends on $jid4)"

echo ""
echo "Jobs submitted in dependency chain: $jid1 → $jid2 → $jid4 → $jid8"
echo "Results will accumulate in: ~/results_v5/v5_scaling_results.csv"
echo "Figure auto-generates after ≥2 configs complete."
echo ""
echo "Monitor: squeue --me"
echo "Tail:    tail -f ~/logs/v5_parallel_1gpu_${jid1}.out"
