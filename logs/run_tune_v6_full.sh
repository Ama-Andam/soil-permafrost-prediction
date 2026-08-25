#!/bin/bash

#SBATCH --job-name=tune_v6_full

#SBATCH --account=hpcusers

#SBATCH --partition=talon-gpu32

#SBATCH --nodelist=talon32

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64GB

#SBATCH --gres=gpu:1

#SBATCH --time=72:00:00

#SBATCH --output=/home/emmanuel.keku/logs/tune_v6_full_%j.out

#SBATCH --error=/home/emmanuel.keku/logs/tune_v6_full_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --mail-user=emmanuel.keku@und.edu



module purge

module load cuda11.8/toolkit/11.8.0

module load cudnn8.6-cuda11.8/8.6.0.163

module load pytorch-py39-cuda11.8-gcc11/1.13.0



cd /home/emmanuel.keku

export PYTHONUNBUFFERED=1



for TARGET in temp smap moist; do

    echo Tuning TARGET $TARGET

    python3 /home/emmanuel.keku/train_soil_spatial_v6.py --mode tune --target $TARGET --tune_trials 50 2>&1 | tee -a /home/emmanuel.keku/logs/tune_v6_full.log

done

echo DONE

ls -lh /home/emmanuel.keku/results_v6/v6_tuning_*.csv

