#!/bin/bash
#SBATCH --job-name=senior_experiments
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=/home/emmanuel.keku/logs/senior_experiments_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/senior_experiments_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  SENIOR EXPERIMENTS"
echo "  1. Stability   — 10 seeds × 11 models"
echo "  2. Uncertainty — variance heads"
echo "  3. Ray Tune    — actual Table 2"
echo "  4. Pruning     — L1 unstructured"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname)"
echo "  Start: $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
echo "Python: $PY"; $PY --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Dependencies
$PY -m pip install --user "numpy==1.24.4" -q --no-deps
$PY -m pip install --user "ray==2.3.1" -q
$PY -m pip install --user "pydantic==1.10.13" -q

# Verify
$PY -c "
import ray, torch
print('Ray    :', ray.__version__)
print('torch  :', torch.__version__)
print('GPUs   :', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('ALL OK')
" || { echo "FATAL: import check failed"; exit 1; }

mkdir -p /home/emmanuel.keku/logs
mkdir -p /home/emmanuel.keku/results_v4
mkdir -p /home/emmanuel.keku/figures_v4

cd /home/emmanuel.keku
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo ""
echo "======================================================"
echo "  EXPERIMENTS START: $(date)"
echo "======================================================"

$PY /home/emmanuel.keku/senior_experiments.py \
    2>&1 | tee /home/emmanuel.keku/logs/senior_experiments.log

EXIT=${PIPESTATUS[0]}

echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

echo "Output files:"
find /home/emmanuel.keku/results_v4 \
    -name "stability*" -o -name "uncertainty*" \
    -o -name "tuning*"  -o -name "pruning*" \
    | sort | xargs -I{} ls -lh {}

find /home/emmanuel.keku/figures_v4 \
    -name "SENIOR_*" | sort | xargs -I{} ls -lh {}

exit $EXIT
