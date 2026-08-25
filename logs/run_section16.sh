#!/bin/bash
#SBATCH --job-name=soil_dl_sec16
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/emmanuel.keku/logs/section16_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/section16_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "========================================"
echo "  SLURM Job : $SLURM_JOB_ID"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "========================================"

module purge
module load cuda/11.8
module load cudnn/8.6 || true

source activate soil_dl

nvidia-smi
python -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        n = torch.cuda.get_device_name(i)
        m = torch.cuda.get_device_properties(
              i).total_memory / 1e9
        print(f'GPU {i}  : {n} | {m:.1f} GB')
"

cd /home/emmanuel.keku

# Install CUDA PyTorch if needed
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
print('CUDA OK — skipping install')
" || pip install --user torch torchvision torchaudio \
       --index-url \
       https://download.pytorch.org/whl/cu118 -q

pip install --user mamba-ssm einops -q

export PYTHONPATH=/home/emmanuel.keku:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Starting training..."
python /home/emmanuel.keku/logs/section16_train.py \
    2>&1 | tee /home/emmanuel.keku/logs/section16_training.log

echo "========================================"
echo "  DONE : $(date)"
echo "========================================"
