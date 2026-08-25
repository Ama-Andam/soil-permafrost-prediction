"""
================================================================================
LAUNCHER_v6.py — Control Panel for v6
================================================================================
DEPLOY & RUN (from DOD FOLDER2):
  cp "/home/emmanuel.keku/DOD FOLDER2/train_soil_spatial_v6.py" ~/train_soil_spatial_v6.py
  cp "/home/emmanuel.keku/DOD FOLDER2/figures_v6.py"            ~/figures_v6.py
  cp "/home/emmanuel.keku/DOD FOLDER2/run_soil_spatial_v6.sh"   ~/logs/run_soil_spatial_v6.sh
  cp "/home/emmanuel.keku/DOD FOLDER2/LAUNCHER_v6.py"           ~/LAUNCHER_v6.py
  chmod u+x ~/logs/run_soil_spatial_v6.sh

SUBMIT:
  module load slurm
  sbatch ~/logs/run_soil_spatial_v6.sh tune     # Step 1: 50-trial tuning (~8h)
  sbatch ~/logs/run_soil_spatial_v6.sh train    # Step 2: full training (~15h)
  sbatch ~/logs/run_soil_spatial_v6.sh ablation # Step 3: ablation study (~6h)
  sbatch ~/logs/run_soil_spatial_v6.sh figures  # Step 4: all figures (~1h)

v6 vs v5 CHANGES:
  + Cyclical features REMOVED (sin/cos time encodings)
  + Input uncertainty variance added per modality (PI doc)
  + Spatiotemporal quantization: impute missing with weighted neighbours
  + SpatialTransformer + SpatialInformer (new ATTENTION tier)
  + Heteroscedastic head: all models output μ AND σ²
  + Three test sets: unseen-space, unseen-time, unseen-both
  + Multi-stage random search: 35 broad + 15 narrow trials per model
  + Full ablation: all 5 components × all 13 models
  + Metrics: R², KGE, ubRMSE, CRPS, DTW, KL Div, NLL (removed plain RMSE/r)
  + Objectives: NLL + CRPS + Graph Laplacian + freeze/thaw physics
  + Figures: KDE, uncertainty dist, entropy initial→best, metric heatmap
================================================================================

MODELS (13 total):
  [ABLATION]  BiGRU_NoGCN, GCN_NoTemporal
  [RESERVOIR] DeepESN, SpatialESN
  [GRAPH]     GraphSAGE, GAT, STGCN
  [ATTENTION] SpatialTransformer, SpatialInformer  ← NEW
  [SSM]       SpatialBiGRU, SpatialMamba, SpatialS4, SpatialFuseMoE

METRICS (distinct, non-redundant):
  R²       — variance explained (keep)
  KGE      — bias + correlation + variability combined (keep)
  ubRMSE   — unbiased RMSE, removes mean bias (NEW)
  CRPS     — proper scoring rule for probabilistic forecasts (NEW)
  DTW      — temporal shape similarity (NEW)
  KL Div   — KL(predicted dist || observed dist) per site (NEW)
  NLL      — negative log likelihood, uncertainty calibration (NEW)
  Removed: plain RMSE (redundant with ubRMSE), plain r (redundant with R²)

OBJECTIVES (combined loss):
  NLL            — heteroscedastic, main loss
  CRPS           — proper scoring, encourages calibrated σ²
  Graph Laplacian — spatial smoothness (λ=0.05)
  Freeze/thaw    — physics-informed near 0°C boundary (temp only)
"""

import os, json, subprocess, shutil
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path("/home/emmanuel.keku")
LOG_DIR = PROJECT / "logs"
MODELS  = PROJECT / "models_v6" / "dl"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6"

ARCHES = [
    "BiGRU_NoGCN","GCN_NoTemporal",
    "DeepESN","SpatialESN",
    "GraphSAGE","GAT","STGCN",
    "SpatialTransformer","SpatialInformer",
    "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE",
]
TARGETS = ["temp","smap","moist"]
TIERS   = {
    "BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
    "SpatialTransformer":"ATTENTION","SpatialInformer":"ATTENTION",
    "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
    "SpatialS4":"SSM","SpatialFuseMoE":"SSM",
}

def status():
    try: import torch
    except ImportError:
        print("module load pytorch-py39-cuda11.8-gcc11/1.13.0"); return

    print("="*70)
    print(f"  v6 STATUS  {pd.Timestamp.now()}")
    print("="*70)

    r=subprocess.run(["squeue","--me","--format=%i %j %T %M %N","--noheader"],
                      capture_output=True,text=True)
    print("\n  SLURM queue:")
    for l in (r.stdout.strip().split("\n") if r.stdout.strip() else ["    (empty)"]):
        print(f"    {l}")

    print(f"\n  CHECKPOINTS  ({MODELS})")
    done=0; total=len(ARCHES)*len(TARGETS)
    print(f"  {'Tier':<12} {'Model':<22} {'temp':>8} {'smap':>8} {'moist':>8}")
    print("  "+"─"*62)
    for arch in ARCHES:
        row=f"  {TIERS.get(arch,'?'):<12} {arch:<22}"
        for tgt in TARGETS:
            ckpt=MODELS/f"{arch}_{tgt}_v6_best.pt"
            if ckpt.exists():
                try:
                    d=torch.load(ckpt,map_location="cpu")
                    row+=f" {d.get('val_r2',float('nan')):>8.4f}"; done+=1
                except Exception: row+=f"{'ERR':>8}"
            else: row+=f"{'missing':>8}"
        print(row)
    print(f"\n  Progress: {done}/{total} checkpoints")

    for log_name in ["soil_tune_v6.log","soil_train_v6.log","soil_ablation_v6.log"]:
        log=LOG_DIR/log_name
        if log.exists():
            print(f"\n  LOG TAIL ({log_name}):")
            with open(log) as f: lines=f.readlines()
            for l in lines[-8:]: print(f"    {l.rstrip()}")
    print("="*70)


def collect():
    try: import torch
    except ImportError:
        print("module load pytorch-py39-cuda11.8-gcc11/1.13.0"); return pd.DataFrame()

    print("Collecting v6 results...")
    records=[]
    for ckpt in sorted(MODELS.glob("*_v6_best.pt")):
        try:
            d=torch.load(ckpt,map_location="cpu")
            stem=ckpt.stem.replace("_v6_best","")
            for tgt in TARGETS:
                if stem.endswith(f"_{tgt}"):
                    arch=stem[:-len(f"_{tgt}")]; break
            else: continue
            tm=d.get("test_metrics",{})
            row=dict(Model=arch,Target=tgt,Tier=TIERS.get(arch,"?"),
                      Val_R2=round(d.get("val_r2",np.nan),4),
                      InitEntropy=round(d.get("initial_entropy",np.nan),4),
                      FinalEntropy=round(d.get("final_entropy",np.nan),4))
            for sp in ["unseen_space","unseen_time","unseen_both"]:
                m=tm.get(sp,{})
                for metric in ["unseen_R2","unseen_KGE","unseen_ubRMSE",
                                "unseen_CRPS","unseen_DTW","unseen_KL_Div",
                                "seen_R2","spatial_gap"]:
                    row[f"{sp}_{metric}"]=round(m.get(metric,np.nan),4)
            records.append(row)
            print(f"  ✓ {arch:<22} [{tgt}]  "
                  f"space={tm.get('unseen_space',{}).get('unseen_R2',np.nan):.4f}  "
                  f"time={tm.get('unseen_time',{}).get('unseen_R2',np.nan):.4f}  "
                  f"both={tm.get('unseen_both',{}).get('unseen_R2',np.nan):.4f}")
        except Exception as e:
            print(f"  ✗ {ckpt.name}: {e}")

    if not records:
        csv=RESULTS/"v6_results_all.csv"
        return pd.read_csv(csv) if csv.exists() else pd.DataFrame()

    df=(pd.DataFrame(records)
         .sort_values("unseen_space_unseen_R2",ascending=False)
         .reset_index(drop=True))
    df.to_csv(RESULTS/"v6_results_all.csv",index=False)

    print(f"\n  {'='*90}")
    print(f"  LEADERBOARD v6 — 3 test sets | Wetland holdout + Q4 2025 holdout")
    print(f"  {'='*90}")
    print(f"  {'Tier':<12} {'Model':<22} {'Tgt':<6} "
          f"{'Space R²':>9} {'Time R²':>9} {'Both R²':>9} "
          f"{'Gap':>6} {'CRPS':>7} {'KGE':>7}")
    print("  "+"─"*90)
    for _,r in df.iterrows():
        print(f"  {r['Tier']:<12} {r['Model']:<22} {r['Target']:<6} "
              f"{r.get('unseen_space_unseen_R2',np.nan):>9.4f} "
              f"{r.get('unseen_time_unseen_R2',np.nan):>9.4f} "
              f"{r.get('unseen_both_unseen_R2',np.nan):>9.4f} "
              f"{r.get('unseen_space_spatial_gap',np.nan):>6.4f} "
              f"{r.get('unseen_space_unseen_CRPS',np.nan):>7.4f} "
              f"{r.get('unseen_space_unseen_KGE',np.nan):>7.4f}")
    return df


QUICK_REF = """
╔══════════════════════════════════════════════════════════════════╗
║  v6 QUICK REFERENCE                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  STEP 1 — copy files (from DOD FOLDER2):                        ║
║    cp "...DOD FOLDER2/train_soil_spatial_v6.py" ~/              ║
║    cp "...DOD FOLDER2/figures_v6.py"            ~/              ║
║    cp "...DOD FOLDER2/run_soil_spatial_v6.sh"   ~/logs/         ║
║    cp "...DOD FOLDER2/LAUNCHER_v6.py"           ~/              ║
║    chmod u+x ~/logs/run_soil_spatial_v6.sh                      ║
╠══════════════════════════════════════════════════════════════════╣
║  STEP 2 — submit (from login1):                                  ║
║    module load slurm                                             ║
║    sbatch ~/logs/run_soil_spatial_v6.sh tune     # ~8h          ║
║    sbatch ~/logs/run_soil_spatial_v6.sh train    # ~15h         ║
║    sbatch ~/logs/run_soil_spatial_v6.sh ablation # ~6h          ║
║    sbatch ~/logs/run_soil_spatial_v6.sh figures  # ~1h          ║
╠══════════════════════════════════════════════════════════════════╣
║  STEP 3 — collect results:                                       ║
║    module load pytorch-py39-cuda11.8-gcc11/1.13.0               ║
║    python3 -c "from LAUNCHER_v6 import collect; collect()"      ║
╚══════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_REF)
