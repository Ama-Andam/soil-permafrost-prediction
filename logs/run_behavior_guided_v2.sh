#!/bin/bash

#SBATCH --job-name=behavior_v2

#SBATCH --account=hpcusers

#SBATCH --partition=talon-gpu32

#SBATCH --nodelist=talon32

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --mem=64GB

#SBATCH --gres=gpu:4

#SBATCH --time=06:00:00

#SBATCH --output=/home/emmanuel.keku/logs/behavior_v2_%j.out

#SBATCH --error=/home/emmanuel.keku/logs/behavior_v2_%j.err

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=emmanuel.keku@und.edu



module purge

module load cuda11.8/toolkit/11.8.0

module load cudnn8.6-cuda11.8/8.6.0.163

module load pytorch-py39-cuda11.8-gcc11/1.13.0



pip install ray --quiet 2>/dev/null

export PYTHONUNBUFFERED=1

export RAY_DISABLE_MEMORY_MONITOR=1



cd /home/emmanuel.keku

python3 /home/emmanuel.keku/experiment_behavior_guided_v2.py 2>&1 | tee /home/emmanuel.keku/logs/behavior_v2.log

