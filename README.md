# Distributed AI-Based Multivariate Soil Temperature and Moisture Prediction
### Alaska Permafrost | 2022–2025 | DoD Project | University of North Dakota

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange)](https://pytorch.org)
[![Ray](https://img.shields.io/badge/Ray-Remote-green)](https://ray.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This project develops a **distributed spatial AI framework** for predicting soil temperature and moisture across Alaska permafrost sites. Unlike traditional single-point models, our framework predicts the full **2D spatial field simultaneously** across 256 measurement locations using Graph Convolutional Networks (GCN) combined with state-space models, reservoir computing, and mixture-of-experts architectures.

### Key Contributions

- **True spatial field prediction** — 256 locations × 24-hour lookback × 26 features → full spatial field output
- **Spatial holdout experiment** — Entire Wetland site (64 locations) withheld from training; predicted via GCN propagation from neighbouring sites
- **11 models across 4 architectural tiers** — Ablations, Reservoir (DeepESN), Graph (GraphSAGE, GAT, STGCN), SSM (Mamba, S4, BiGRU, FuseMoE)
- **GPU scaling via Ray Remote** — Model-level parallelism across 1→2→4→8 V100 GPUs
- **Recoverability curves** — Operational reliability metric per senior recommendation

### Results Summary

| Target | Best Model | Seen R² | Unseen R² | Freeze Acc |
|---|---|---|---|---|
| Weather Temp (°C) | GCN_NoTemporal | 0.9841 | **0.9878** | 98.5% |
| SMAP Temp L1 (K) | STGCN | 0.9924 | **0.9947** | 100.0% |
| Soil Moisture (m³/m³) | STGCN | 0.9445 | **0.8537** | — |

> **Unseen R²** = R² on Wetland site never seen during training, predicted only via GCN spatial graph propagation from Bedrock, Transition and Upland neighbours.

---

## Dataset

| Dataset | Period | Description |
|---|---|---|
| Dataset 1 | 2000–2025 | Alaska weather station observations (26 years) |
| Dataset 2 | 2022–2025 | Full spatio-temporal: weather + SMAP satellite + topography |

**Dataset 2 schema:**
```
Spatio_Temp_Ref : time_utc, Site, Latitude, Longitude, smap_node_x, smap_node_y
Topography      : elevation_m, elev_roughness_m, slope_deg
Weather         : temperature_2m, precipitation, snow_depth_weather,
                  soil_temperature_0_to_7cm, soil_moisture_0_to_7cm
SMAP Satellite  : Temp_K, SM_Surface, SM_Rootzone, Pressure, Greenness,
                  Snow_Depth_SMAP, Soil_Temp_L1, L2, L3, L4
```

**4 ecological sites:** Bedrock | Transition | Upland | Wetland (holdout)
**256 unique lat/lon locations** | **87 SMAP nodes** | **2,992,128 rows**

> Raw data not included (size). Contact authors for access.

---

## Model Architecture

### v3 — Spatial Field Prediction (Temporal Split)
```
Input  : (batch, 24 timesteps, 256 locations, 22 features)
Output : (batch, 256 locations, 1 target)
Split  : 2022–2023 train | 2024 val | 2025 test
```

### v4 — Spatial Holdout Experiment (Main Contribution)
```
Input  : (batch, 24 timesteps, 256 locations, 26 features)
Output : (batch, 256 locations, 1 target)
Split  : Temporal (2022–23/24/25) + Spatial holdout (Wetland withheld)
Loss   : Masked Huber (seen locations only) + Graph Laplacian regularisation
```

### The 11 Models

| Tier | Model | Description |
|---|---|---|
| ABLATION | BiGRU_NoGCN | Temporal only — no spatial graph |
| ABLATION | GCN_NoTemporal | Spatial only — no temporal encoder |
| RESERVOIR | DeepESN | Deep Echo State Network (Gallicchio & Micheli 2017) |
| RESERVOIR | SpatialESN | DeepESN + GCN (novel contribution) |
| GRAPH | GraphSAGE | Inductive node prediction (Hamilton et al. 2017) |
| GRAPH | GAT | Graph Attention Network (Velickovic et al. 2018) |
| GRAPH | STGCN | Spatio-Temporal GCN (Yu et al. 2018) |
| SSM | SpatialBiGRU | BiGRU + Attention + GCN |
| SSM | SpatialMamba | Mamba SSM + GCN (Gu & Dao 2023) |
| SSM | SpatialS4 | S4 HiPPO-LegS + GCN (Gu et al. 2022) |
| SSM | SpatialFuseMoE | Sparse MoE + Mamba + GCN |

---

## Repository Structure

```
soil-permafrost-prediction/
├── README.md
├── v3_baseline/
│   ├── train_soil_spatial.py       # v3 training script
│   ├── run_soil_spatial.sh         # SLURM batch script
│   └── postprocess_results.py      # Per-site evaluation & figures
├── v4_spatial_holdout/
│   ├── train_soil_spatial_v4.py    # v4 training (main contribution)
│   ├── run_soil_spatial_v4.sh      # SLURM — requests 8x V100
│   ├── LAUNCHER_v4.py              # deploy/submit/status/collect
│   └── postprocess_results_v4.py   # Full evaluation with holdout
├── analysis/
│   ├── extract_training_details.py # Loss curves, input vars, hyperparams
│   ├── baseline_comparison.py      # ML baselines vs DL with timing
│   ├── recoverability_curves.py    # Operational reliability metric
│   └── ray_scaling_experiment.py   # 1→8 GPU scaling via Ray Remote
└── results/
    └── README.md                   # Results summary (CSVs on HPC)
```

---

## Installation & Usage

### Requirements
```bash
# On UND Talon HPC
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0
pip install --user numpy==1.24.4 PyWavelets scikit-learn xgboost lightgbm seaborn ray
```

### v4 Quick Start (Recommended)
```bash
# 1. Deploy scripts to home directory
python3 LAUNCHER_v4.py
from LAUNCHER_v4 import deploy, submit, status, collect
deploy()

# 2. Submit SLURM job — log out immediately after
submit()

# 3. Check progress (from any new session)
status()

# 4. After job completes — collect results
collect()

# 5. Generate all figures
python3 ~/postprocess_results_v4.py
```

### GPU Scaling Experiment
```bash
cp ray_scaling_experiment.py ~/
cp run_scaling.sh ~/logs/
sbatch ~/logs/run_scaling.sh
```

---

## Training Configuration

All hyperparameters are **fixed literature standards** — no systematic tuning performed. Ray Tune optimisation is planned as future work.

| Parameter | Value | Description |
|---|---|---|
| Optimizer | AdamW | Weight decay 1e-4 |
| Learning Rate | 3e-4 | OneCycleLR cosine annealing |
| Max Epochs | 30 | Early stopping patience=7 |
| Lookback | 24 hours | Temporal history window |
| Dropout | 0.1 | All trainable layers |
| Loss | Huber δ=1.0 | + Graph Laplacian λ=0.05 |
| Spatial Holdout | Wetland | 64 locations withheld |
| GCN k-neighbours | 6 | Gaussian decay σ=9.99km |

---

## Compute Environment

- **HPC:** UND Talon — talon-gpu32 partition
- **GPU:** 8× NVIDIA V100-SXM2 32GB per node
- **CPU:** 36 cores, 1.4TB RAM
- **Parallelism:** Ray Remote model-level (NOT ray.train)

---

## References

```bibtex
@article{gallicchio2017deepesn,
  title={Deep Echo State Network (DeepESN): A Brief Survey},
  author={Gallicchio, Claudio and Micheli, Alessio},
  journal={arXiv:1712.04323}, year={2017}}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv:2312.00752}, year={2023}}

@article{gu2022s4,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert et al.},
  journal={arXiv:2111.00396}, year={2022}}

@inproceedings{hamilton2017graphsage,
  title={Inductive Representation Learning on Large Graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle={NeurIPS}, year={2017}}

@inproceedings{velickovic2018gat,
  title={Graph Attention Networks},
  author={Velickovic, Petar et al.},
  booktitle={ICLR}, year={2018}}

@inproceedings{yu2018stgcn,
  title={Spatio-Temporal Graph Convolutional Networks},
  author={Yu, Bing et al.},
  booktitle={IJCAI}, year={2018}}

@article{singh2025esnssm,
  title={Echo State Networks as State-Space Models: A Systems Perspective},
  author={Singh and Raman},
  journal={arXiv:2509.04422}, year={2025}}
```

---

## Authors

- **Emmanuel Keku** — University of North Dakota
- Supervisors: Shayeghmoradi, Meisam et al.

## Acknowledgements

This work used advanced cyberinfrastructure resources provided by the University of North Dakota Computational Research Center (ROR: https://ror.org/01sdmps70).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
