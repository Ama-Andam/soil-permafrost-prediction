#!/bin/bash
#SBATCH --job-name=soil_scaling
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/scaling_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/scaling_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "======================================================"
echo "  GPU SCALING EXPERIMENT — Ray Remote"
echo "  Job: $SLURM_JOB_ID | Node: $(hostname)"
echo "  GPUs requested: 8x V100"
echo "  Start: $(date)"
echo "======================================================"

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

$PY -m pip install --user "numpy==1.24.4" -q --no-deps
# Pin Ray to version compatible with pydantic v1 on Python 3.9
$PY -m pip install --user "ray==2.3.1" -q
# Pin pydantic v1 — required by Ray 2.3.1
$PY -m pip install --user "pydantic==1.10.13" -q
# Fix protobuf conflict with tensorboard
$PY -m pip install --user "protobuf>=3.19.5,<5.0.0" -q

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
echo "  SCALING START: $(date)"
echo "======================================================"

$PY /home/emmanuel.keku/ray_scaling_experiment.py \
    2>&1 | tee /home/emmanuel.keku/logs/scaling_experiment.log

EXIT=${PIPESTATUS[0]}
echo "======================================================"
echo "  DONE: $(date) | Exit: $EXIT"
echo "======================================================"

find /home/emmanuel.keku/results_v4 -name "scaling*" -printf "  %p (%k KB)\n"
find /home/emmanuel.keku/figures_v4 -name "SCALE*"   -printf "  %p (%k KB)\n"
exit $EXIT
