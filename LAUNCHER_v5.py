"""
================================================================================
LAUNCHER_v5.py — Control Panel for v5 Distributed Spatial AI
================================================================================
USAGE:
  from LAUNCHER_v5 import deploy, submit, status, collect

  deploy()        — copy v5 files to ~/
  submit()        — full training (11 models × 3 targets)
  submit_stab()   — stability benchmark (10 runs × selected models)
  submit_unc()    — uncertainty analysis on existing checkpoints
  submit_parallel()— Ray model-level parallelism scaling (1→8 GPUs)
  status()        — check job + checkpoints + log tail
  collect()       — load all results + uncertainty into DataFrame

v5 KEY ADDITIONS vs v4:
  DP=0.15, WD=5e-4, L1=1e-5         — regularisation
  Magnitude pruning (20%)            — compression
  MC-Dropout N=30                    — uncertainty quantification
  10-run stability benchmark         — per senior recommendation
  Ray remote model-level parallelism — no gradient distortion
================================================================================
"""

import os, json, subprocess, shutil
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT  = Path("/home/emmanuel.keku")
LOG_DIR  = PROJECT / "logs"
MODELS   = PROJECT / "models_v5" / "dl"
RESULTS  = PROJECT / "results_v5"
FIGS     = PROJECT / "figures_v5"
SCRIPTS  = Path(__file__).parent

ARCHES = [
    "BiGRU_NoGCN","GCN_NoTemporal",
    "DeepESN","SpatialESN",
    "GraphSAGE","GAT","STGCN",
    "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE",
]
TARGETS = ["temp","smap","moist"]
TIERS = {
    "BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
    "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
    "SpatialS4":"SSM","SpatialFuseMoE":"SSM",
}

# ── deploy ────────────────────────────────────────────────────────────────────
def deploy():
    """Copy all v5 scripts to Talon home directory."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        (SCRIPTS/"train_soil_spatial_v5.py",   PROJECT/"train_soil_spatial_v5.py"),
        (SCRIPTS/"uncertainty_analysis_v5.py",  PROJECT/"uncertainty_analysis_v5.py"),
        (SCRIPTS/"ray_model_parallel_v5.py",    PROJECT/"ray_model_parallel_v5.py"),
        (SCRIPTS/"run_soil_spatial_v5.sh",      LOG_DIR/"run_soil_spatial_v5.sh"),
        (SCRIPTS/"run_model_parallel_v5.sh",    LOG_DIR/"run_model_parallel_v5.sh"),
        (SCRIPTS/"LAUNCHER_v5.py",              PROJECT/"LAUNCHER_v5.py"),
    ]
    print("Deploying v5 files...")
    all_ok = True
    for src, dst in files:
        if src.exists():
            shutil.copy2(src, dst); os.chmod(dst, 0o755)
            print(f"  ✓ {dst.name:<45} {dst.stat().st_size:,} bytes")
        else:
            print(f"  ✗ NOT FOUND: {src}"); all_ok = False
    if all_ok:
        print("\n  ✓ All v5 files deployed.")
    return all_ok


# ── submit ────────────────────────────────────────────────────────────────────
def submit(mode="train"):
    """Submit v5 SLURM job."""
    slurm = LOG_DIR / "run_soil_spatial_v5.sh"
    if not slurm.exists():
        print("✗ SLURM script not found. Run deploy() first."); return None

    print(f"Submitting v5 job (mode={mode})...")
    r = subprocess.run(["sbatch", str(slurm), mode],
                        capture_output=True, text=True, cwd=str(PROJECT))
    if r.returncode != 0:
        print(f"✗ {r.stderr.strip()}"); return None

    job_id = r.stdout.strip().split()[-1]
    with open(LOG_DIR/f"last_v5_{mode}_job.json","w") as f:
        json.dump({"job_id":job_id, "mode":mode,
                   "submitted_at":pd.Timestamp.now().isoformat()}, f)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  v5 JOB SUBMITTED  [{mode.upper():<10}]                              ║
║  Job ID : {job_id:<51}║
╠══════════════════════════════════════════════════════════════╣
║  v5 ENHANCEMENTS vs v4:                                      ║
║  DP=0.15  WD=5e-4  L1=1e-5  Prune=20%                       ║
║  MC-Dropout uncertainty (N=30 samples)                       ║
╠══════════════════════════════════════════════════════════════╣
║  MONITOR:                                                    ║
║    squeue --me                                               ║
║    tail -f ~/logs/soil_training_v5.log                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    return job_id

def submit_stab():  return submit("stability")
def submit_unc():   return submit("uncertainty")


# ── submit_parallel ───────────────────────────────────────────────────────────
def submit_parallel():
    """Submit Ray model-level parallelism scaling experiment."""
    script = LOG_DIR / "run_model_parallel_v5.sh"
    if not script.exists():
        print("✗ Script not found. Run deploy() first."); return None
    r = subprocess.run(["bash", str(script)], capture_output=True,
                        text=True, cwd=str(PROJECT))
    print(r.stdout)
    if r.returncode != 0: print(f"✗ {r.stderr}")
    return r.returncode == 0


# ── status ────────────────────────────────────────────────────────────────────
def status():
    """Full v5 status check."""
    try:
        import torch
    except ImportError:
        print("Load modules: module load pytorch-py39-cuda11.8-gcc11/1.13.0"); return

    print("=" * 70)
    print(f"  v5 STATUS  {pd.Timestamp.now()}")
    print("=" * 70)

    r = subprocess.run(["squeue","--me","--format=%i %j %T %M %N","--noheader"],
                        capture_output=True, text=True)
    print(f"\n  SLURM queue:")
    for l in (r.stdout.strip().split("\n") if r.stdout.strip() else ["    (empty)"]):
        print(f"    {l}")

    print(f"\n  CHECKPOINTS  ({MODELS})")
    done = 0; total = len(ARCHES) * len(TARGETS)
    print(f"  {'Tier':<12} {'Model':<20} {'temp':>8} {'smap':>8} {'moist':>8}")
    print("  " + "─" * 58)
    for arch in ARCHES:
        row = f"  {TIERS.get(arch,'?'):<12} {arch:<20}"
        for tgt in TARGETS:
            ckpt = MODELS / f"{arch}_{tgt}_v5_best.pt"
            if ckpt.exists():
                try:
                    d  = torch.load(ckpt, map_location="cpu")
                    r2 = d.get("val_r2", float("nan"))
                    unc = d.get("uncertainty",{}).get("unc_ratio",float("nan"))
                    row += f" {r2:>8.4f}"
                    done += 1
                except Exception: row += f"{'ERR':>8}"
            else: row += f"{'missing':>8}"
        print(row)
    print(f"\n  Progress: {done}/{total} checkpoints")

    log = LOG_DIR / "soil_training_v5.log"
    if log.exists():
        print(f"\n  LOG TAIL:")
        with open(log) as f: lines = f.readlines()
        for l in lines[-12:]: print(f"    {l.rstrip()}")
    print("=" * 70)


# ── collect ───────────────────────────────────────────────────────────────────
def collect():
    """Load all v5 results including uncertainty metrics."""
    try:
        import torch
    except ImportError:
        print("Load modules first"); return pd.DataFrame()

    print("Collecting v5 results...")
    records = []
    for ckpt in sorted(MODELS.glob("*_v5_best.pt")):
        try:
            d    = torch.load(ckpt, map_location="cpu")
            stem = ckpt.stem.replace("_v5_best","")
            for tgt in TARGETS:
                if stem.endswith(f"_{tgt}"):
                    arch = stem[:-len(f"_{tgt}")]; break
            else: continue
            tm  = d.get("test_metrics", {})
            unc = d.get("uncertainty", {})
            records.append(dict(
                Model=arch, Target=tgt, Tier=TIERS.get(arch,"?"),
                Val_R2=d.get("val_r2", float("nan")),
                Seen_R2=tm.get("seen_R2", float("nan")),
                Unseen_R2=tm.get("unseen_R2", float("nan")),
                Spatial_Gap=tm.get("spatial_gap", float("nan")),
                Seen_KGE=tm.get("seen_KGE", float("nan")),
                Unseen_FreezeAcc=tm.get("unseen_FreezeAcc", float("nan")),
                # v5 additions
                Unc_Ratio=unc.get("unc_ratio", float("nan")),
                Calibration_Seen=unc.get("calibration_seen", float("nan")),
                Calibration_Unseen=unc.get("calibration_unseen", float("nan")),
                DP_Rate=d.get("dropout_rate", float("nan")),
                WD_Rate=d.get("weight_decay", float("nan")),
                Prune_Ratio=d.get("prune_ratio", float("nan")),
                Train_s=d.get("elapsed_s", float("nan"))))
            print(f"  ✓ {arch:<20} [{tgt}]  "
                  f"seen={tm.get('seen_R2',float('nan')):.4f}  "
                  f"unseen={tm.get('unseen_R2',float('nan')):.4f}  "
                  f"unc_ratio={unc.get('unc_ratio',float('nan')):.2f}")
        except Exception as e:
            print(f"  ✗ {ckpt.name}: {e}")

    if not records:
        csv = RESULTS/"v5_results_all.csv"
        if csv.exists():
            return pd.read_csv(csv)
        print("  No results yet."); return pd.DataFrame()

    df = (pd.DataFrame(records)
          .sort_values("Unseen_R2", ascending=False)
          .reset_index(drop=True))
    df.to_csv(RESULTS/"v5_results_all.csv", index=False)

    print(f"\n  {'='*80}")
    print(f"  LEADERBOARD v5 | Wetland holdout | DP=0.15 WD=5e-4 Prune=20%")
    print(f"  {'='*80}")
    print(f"  {'Tier':<12} {'Model':<20} {'Tgt':<6} "
          f"{'Seen R²':>8} {'Unseen R²':>10} {'Gap':>6} "
          f"{'UncRatio':>9} {'CalibUns':>9}")
    print("  " + "─"*85)
    for _, row in df.iterrows():
        gap = row.get("Spatial_Gap", float("nan"))
        flag = "✓" if (not np.isnan(gap) and gap < 0.05) else "~"
        print(f"  {flag} {row['Tier']:<11} {row['Model']:<20} {row['Target']:<6} "
              f"{row.get('Seen_R2',float('nan')):>8.4f} "
              f"{row.get('Unseen_R2',float('nan')):>10.4f} "
              f"{gap:>6.4f} "
              f"{row.get('Unc_Ratio',float('nan')):>9.2f} "
              f"{row.get('Calibration_Unseen',float('nan')):>9.3f}")
    print(f"\n  ✓ = gap < 0.05 | UncRatio >1.2 = well-calibrated uncertainty")
    return df


QUICK_REF = """
╔══════════════════════════════════════════════════════════════╗
║  v5 QUICK REFERENCE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  deploy()           copy scripts to ~/                       ║
║  submit()           full training (11×3, ~24h)               ║
║  submit_stab()      10-run stability benchmark (~8h)         ║
║  submit_unc()       MC-Dropout uncertainty analysis (~2h)    ║
║  submit_parallel()  Ray 1→2→4→8 GPU scaling (~4h each)      ║
║  status()           queue + checkpoints + log                ║
║  collect()          results + uncertainty leaderboard        ║
╠══════════════════════════════════════════════════════════════╣
║  v5 vs v4 CHANGES:                                           ║
║  + DP=0.15 (was 0.10)  — more regularisation                 ║
║  + WD=5e-4 (was 1e-4)  — stronger weight decay               ║
║  + L1=1e-5              — sparsity regularisation            ║
║  + 20% magnitude pruning after training                      ║
║  + MC-Dropout N=30: epistemic uncertainty per location       ║
║  + Uncertainty ratio: unseen/seen (target >1.2)              ║
║  + Calibration: uncertainty vs actual error (target >0.5)    ║
║  + Stability: 10 runs, reports mean±std±CV                   ║
║  + Ray Remote: model-level, no gradient distortion           ║
╚══════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_REF)
