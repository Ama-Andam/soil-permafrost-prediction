#!/bin/bash

#SBATCH --job-name=soil_spatial_v3

#SBATCH --account=hpcusers

#SBATCH --partition=talon-gpu32

#SBATCH --nodelist=talon32

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64GB

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --output=/home/emmanuel.keku/logs/soil_spatial_v3_%j.out

#SBATCH --error=/home/emmanuel.keku/logs/soil_spatial_v3_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --mail-user=emmanuel.keku@und.edu

module purge

module load cuda11.8/toolkit/11.8.0

module load cudnn8.6-cuda11.8/8.6.0.163

module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)

$PY -m pip install --user "numpy==1.24.4" -q --no-deps

$PY -m pip install --user "PyWavelets>=1.3" "scikit-learn>=1.0" "xgboost>=1.6" "lightgbm>=3.3" "seaborn>=0.12" -q

$PY -c "import numpy,scipy,torch,pandas,pywt,sklearn,xgboost,seaborn; print('ALL IMPORTS OK')" || exit 1

mkdir -p /home/emmanuel.keku/logs /home/emmanuel.keku/preprocessed_v3 /home/emmanuel.keku/results_v3 /home/emmanuel.keku/models_v3/dl /home/emmanuel.keku/figures_v3

cd /home/emmanuel.keku

export PYTHONUNBUFFERED=1

$PY /home/emmanuel.keku/train_soil_spatial.py 2>&1 | tee /home/emmanuel.keku/logs/soil_training_v3.log

exit ${PIPESTATUS[0]}

