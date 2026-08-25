#!/bin/bash
#SBATCH --job-name=tune_moist
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/emmanuel.keku/logs/tune_moist_%j.out
#SBATCH --error=/home/emmanuel.keku/logs/tune_moist_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0
export PYTHONUNBUFFERED=1
cd /home/emmanuel.keku

for MODEL in GCN_NoTemporal DeepESN SpatialESN GraphSAGE GAT STGCN SpatialTransformer SpatialInformer SpatialBiGRU SpatialMamba SpatialS4 SpatialFuseMoE; do
    python3 /home/emmanuel.keku/train_soil_spatial_v6.py --mode tune --target moist --arch $MODEL --tune_trials 50 2>&1 | tee -a /home/emmanuel.keku/logs/tune_moist.log
done

python3 /home/emmanuel.keku/train_soil_spatial_v6.py --mode tune --target smap --arch SpatialFuseMoE --tune_trials 50 2>&1 | tee -a /home/emmanuel.keku/logs/tune_moist.log
