"""
================================================================================
LAUNCHER_v4.py — Control Panel for v4 Distributed Spatial AI
================================================================================
USAGE (from any Talon terminal or JupyterLab):
  from LAUNCHER_v4 import deploy, submit, status, collect

  deploy()   — copy files to ~/
  submit()   — submit SLURM job, log out immediately after
  status()   — check job + checkpoints + log tail (safe after restart)
  collect()  — load all results after job finishes
================================================================================
"""

import os, json, subprocess, shutil
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT  = Path("/home/emmanuel.keku")
LOG_DIR  = PROJECT / "logs"
MODELS   = PROJECT / "models_v4" / "dl"
RESULTS  = PROJECT / "results_v4"
FIGS     = PROJECT / "figures_v4"
SCRIPTS  = Path(__file__).parent

ARCHES = [
    "BiGRU_NoGCN","GCN_NoTemporal",
    "DeepESN","SpatialESN",
    "GraphSAGE","GAT","STGCN",
    "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE",
]
TARGETS = ["temp","smap","moist"]
TIERS   = {
    "BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
    "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
    "SpatialS4":"SSM","SpatialFuseMoE":"SSM",
}


# ══════════════════════════════════════════════════════════════════════════════
def deploy():
    """Copy v4 scripts to Talon home directory."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        (SCRIPTS / "train_soil_spatial_v4.py",  PROJECT  / "train_soil_spatial_v4.py"),
        (SCRIPTS / "run_soil_spatial_v4.sh",     LOG_DIR  / "run_soil_spatial_v4.sh"),
        (SCRIPTS / "LAUNCHER_v4.py",             PROJECT  / "LAUNCHER_v4.py"),
        (SCRIPTS / "postprocess_results_v4.py",  PROJECT  / "postprocess_results_v4.py"),
    ]
    print("Deploying v4 files...")
    all_ok = True
    for src, dst in files:
        if src.exists():
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  ✓ {dst.name:<45} {dst.stat().st_size:,} bytes")
        else:
            print(f"  ✗ NOT FOUND: {src}")
            all_ok = False
    if all_ok:
        print("\n  All files deployed. Run submit() to start the job.")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
def submit():
    """Validate and submit the v4 SLURM job."""
    slurm = LOG_DIR / "run_soil_spatial_v4.sh"
    if not slurm.exists():
        print("✗ SLURM script not found. Run deploy() first.")
        return None

    print("Validating SLURM script...")
    val = subprocess.run(["sbatch","--test-only",str(slurm)],
                          capture_output=True, text=True)
    msg = val.stdout.strip() or val.stderr.strip()
    print(f"  {msg}")
    if val.returncode != 0:
        print("✗ Validation failed. Check account/partition.")
        return None

    print("Submitting v4 job...")
    r = subprocess.run(["sbatch", str(slurm)],
                        capture_output=True, text=True, cwd=str(PROJECT))
    if r.returncode != 0:
        print(f"✗ {r.stderr.strip()}")
        return None

    job_id = r.stdout.strip().split()[-1]
    with open(LOG_DIR/"last_v4_job.json","w") as f:
        json.dump({"job_id":job_id,
                   "submitted_at":pd.Timestamp.now().isoformat()}, f)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  v4 JOB SUBMITTED                                            ║
║  Job ID     : {job_id:<46}║
║  11 Models × 3 Targets = 33 checkpoints                     ║
║  Holdout    : Wetland (spatial generalisation test)          ║
╠══════════════════════════════════════════════════════════════╣
║  YOUR SESSION CAN END NOW                                    ║
╠══════════════════════════════════════════════════════════════╣
║  MONITOR:                                                    ║
║    squeue --me                                               ║
║    tail -f ~/logs/soil_training_v4.log                       ║
╠══════════════════════════════════════════════════════════════╣
║  AFTER JOB:                                                  ║
║    collect()          — load results                         ║
║    python3 ~/postprocess_results_v4.py  — full figures       ║
╚══════════════════════════════════════════════════════════════╝
""")
    return job_id


# ══════════════════════════════════════════════════════════════════════════════
def status():
    """Full status — safe after session restart."""
    try:
        import torch
    except ImportError:
        print("Load modules first: module load pytorch-py39-cuda11.8-gcc11/1.13.0")
        return

    print("=" * 65)
    print(f"  v4 STATUS  {pd.Timestamp.now()}")
    print("=" * 65)

    # Last job
    jf = LOG_DIR / "last_v4_job.json"
    job_id = None
    if jf.exists():
        rec = json.load(open(jf))
        job_id = rec.get("job_id")
        print(f"\n  Last job : {job_id} | {rec.get('submitted_at','')[:19]}")

    # Queue
    print("\n  SLURM queue:")
    r = subprocess.run(["squeue","--me","--format=%i %j %T %M %N","--noheader"],
                        capture_output=True, text=True)
    for l in (r.stdout.strip().split("\n") if r.stdout.strip() else ["    (empty)"]):
        print(f"    {l}")

    # sacct
    if job_id:
        r2 = subprocess.run(["sacct","-j",job_id,
                              "--format=JobID,State,ExitCode,Elapsed","--noheader"],
                             capture_output=True, text=True)
        print(f"\n  sacct [{job_id}]:")
        for l in r2.stdout.strip().split("\n"):
            if l.strip(): print(f"    {l}")

    # Checkpoint inventory
    print(f"\n  CHECKPOINTS  ({MODELS})")
    total = len(ARCHES) * len(TARGETS)
    done  = 0
    print(f"  {'Tier':<12} {'Model':<20} {'temp':>8} {'smap':>8} {'moist':>8}")
    print("  " + "─"*58)
    for arch in ARCHES:
        row = f"  {TIERS.get(arch,'?'):<12} {arch:<20}"
        for tgt in TARGETS:
            ckpt = MODELS / f"{arch}_{tgt}_v4_best.pt"
            if ckpt.exists():
                try:
                    d  = torch.load(ckpt, map_location="cpu")
                    r2 = d.get("val_r2", float("nan"))
                    row += f" {r2:>8.4f}"
                    done += 1
                except Exception:
                    row += f"{'ERR':>8}"
            else:
                row += f"{'missing':>8}"
        print(row)
    print(f"\n  Progress: {done}/{total} checkpoints complete")

    # Log tail
    log = LOG_DIR / "soil_training_v4.log"
    if log.exists():
        print(f"\n  LOG TAIL  ({log}):")
        with open(log) as f: lines = f.readlines()
        for l in lines[-12:]: print(f"    {l.rstrip()}")
    else:
        print("\n  Training log not created yet.")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
def collect():
    """Load all v4 results from checkpoints. Safe after restart."""
    try:
        import torch
    except ImportError:
        print("Load modules: module load pytorch-py39-cuda11.8-gcc11/1.13.0")
        return pd.DataFrame()

    print("Collecting v4 results...")
    records = []
    for ckpt in sorted(MODELS.glob("*_v4_best.pt")):
        try:
            d    = torch.load(ckpt, map_location="cpu")
            stem = ckpt.stem.replace("_v4_best","")
            # stem: {arch}_{tgt} — tgt is last token
            for tgt in TARGETS:
                if stem.endswith(f"_{tgt}"):
                    arch = stem[:-len(f"_{tgt}")]
                    break
            else:
                continue
            tm = d.get("test_metrics", {})
            records.append(dict(
                Model=arch, Target=tgt, Tier=TIERS.get(arch,"?"),
                Val_R2=d.get("val_r2", float("nan")),
                Seen_R2=tm.get("seen_R2",    float("nan")),
                Unseen_R2=tm.get("unseen_R2", float("nan")),
                All_R2=tm.get("all_R2",      float("nan")),
                Seen_RMSE=tm.get("seen_RMSE", float("nan")),
                Unseen_RMSE=tm.get("unseen_RMSE", float("nan")),
                Seen_KGE=tm.get("seen_KGE",  float("nan")),
                Unseen_KGE=tm.get("unseen_KGE", float("nan")),
                Seen_FreezeAcc=tm.get("seen_FreezeAcc",   float("nan")),
                Unseen_FreezeAcc=tm.get("unseen_FreezeAcc", float("nan")),
                Spatial_Gap=tm.get("spatial_gap", float("nan")),
                Train_s=d.get("elapsed_s", float("nan")),
                Job_ID=d.get("job_id","N/A"),
                Node=d.get("node","N/A")))
            print(f"  ✓ {arch:<20} [{tgt}]  "
                  f"seen={tm.get('seen_R2',float('nan')):.4f}  "
                  f"unseen={tm.get('unseen_R2',float('nan')):.4f}  "
                  f"gap={tm.get('spatial_gap',float('nan')):.4f}")
        except Exception as e:
            print(f"  ✗ {ckpt.name}: {e}")

    if not records:
        csv = RESULTS/"v4_results_all.csv"
        if csv.exists():
            print(f"  Loading from CSV: {csv}")
            return pd.read_csv(csv)
        print("  No results yet.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("Unseen_R2",ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS/"v4_results_all.csv", index=False)

    # Print leaderboard
    print(f"\n  {'='*75}")
    print(f"  LEADERBOARD — Spatial Generalisation | Holdout: Wetland")
    print(f"  {'='*75}")
    print(f"  {'Tier':<12} {'Model':<20} {'Tgt':<6} "
          f"{'Seen R²':>8} {'Unseen R²':>10} {'Gap':>6} "
          f"{'Unseen KGE':>11} {'Unseen Frz':>11}")
    print("  " + "─"*88)
    for _,row in df.iterrows():
        gap = row.get("Spatial_Gap", float("nan"))
        flag = "✓" if (not np.isnan(gap) and gap < 0.05) else "~"
        print(f"  {flag} {row['Tier']:<11} {row['Model']:<20} {row['Target']:<6} "
              f"{row.get('Seen_R2',float('nan')):>8.4f} "
              f"{row.get('Unseen_R2',float('nan')):>10.4f} "
              f"{gap:>6.4f} "
              f"{row.get('Unseen_KGE',float('nan')):>11.4f} "
              f"{row.get('Unseen_FreezeAcc',float('nan')):>10.2f}%")

    print(f"\n  ✓ = spatial gap < 0.05 (strong generalisation)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def resubmit_missing():
    """Resubmit — completed checkpoints are auto-skipped."""
    try:
        import torch
    except ImportError:
        print("Load modules first"); return

    missing = []
    for arch in ARCHES:
        for tgt in TARGETS:
            ckpt = MODELS / f"{arch}_{tgt}_v4_best.pt"
            if not ckpt.exists():
                missing.append(f"{arch}/{tgt}")
                continue
            try:
                d = torch.load(ckpt, map_location="cpu")
                if d.get("val_r2", -99) <= -10:
                    missing.append(f"{arch}/{tgt} (corrupt)")
            except Exception:
                missing.append(f"{arch}/{tgt} (unreadable)")

    total = len(ARCHES)*len(TARGETS)
    print(f"Missing: {len(missing)}/{total}")
    for m in missing: print(f"  ✗ {m}")
    if not missing:
        print("All checkpoints present."); return
    print(f"\nResubmitting ({total-len(missing)} will be skipped)...")
    return submit()


QUICK_REF = """
╔══════════════════════════════════════════════════════════════╗
║  v4 QUICK REFERENCE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  deploy()           copy scripts to ~/                       ║
║  submit()           submit SLURM job                         ║
║  status()           queue + checkpoints + log tail           ║
║  collect()          load results → leaderboard               ║
║  resubmit_missing() retry failed/missing models              ║
╠══════════════════════════════════════════════════════════════╣
║  11 MODELS:                                                  ║
║  [ABLATION]  BiGRU_NoGCN, GCN_NoTemporal                    ║
║  [RESERVOIR] DeepESN, SpatialESN                             ║
║  [GRAPH]     GraphSAGE, GAT, STGCN                           ║
║  [SSM]       SpatialBiGRU, SpatialMamba, SpatialS4,          ║
║              SpatialFuseMoE                                  ║
╠══════════════════════════════════════════════════════════════╣
║  KEY METRIC: Unseen_R² (Wetland holdout)                    ║
║  Small spatial_gap → GCN doing real spatial work             ║
╚══════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_REF)
