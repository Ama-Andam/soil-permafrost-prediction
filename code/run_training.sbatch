#!/bin/bash
#SBATCH --job-name=soil_model_training
#SBATCH --partition=gpu-code-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/training_%j.out
#SBATCH --error=results/training_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

ray start --head --port=6379 --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=1

sleep 5

python code/train.py results/stl_prepared/dataset_prepared_for_training.csv \
    --outdir results/training \
    --n_lookback 24 \
    --epochs 15 \
    --batch_size 256

ray stop

echo "Job finished at $(date)"
