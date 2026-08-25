"""
================================================================================
LAUNCHER.py  —  The ONLY script you run interactively on Talon
================================================================================

PURPOSE:
  This file is your interactive control panel.
  It does NOT do any training itself.
  Training happens entirely under SLURM — session-independent.

WORKFLOW:
  Step 1:  Run deploy() once to copy files to Talon home directory
  Step 2:  Run submit() to submit the SLURM job and log out
  Step 3:  (Later) Run collect() to gather results after job finishes

  That's it. You do NOT need to stay connected.

RUN THIS FILE FROM:
  - JupyterLab on Talon  (run each function as a cell)
  - SSH terminal on Talon (python3 LAUNCHER.py)
================================================================================
"""

import os
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/home/emmanuel.keku")
LOG_DIR     = PROJECT_DIR / "logs"
SCRIPTS_DIR = Path(__file__).parent   # directory containing this file


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — deploy()
# Copies training script and SLURM script to your Talon home directory.
# Run once before submitting.
# ════════════════════════════════════════════════════════════════════════════════

def deploy():
    """
    Copy the two required files to /home/emmanuel.keku/.
    Run this once from your JupyterLab before submitting.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        (SCRIPTS_DIR / "train_soil_spatial.py",  PROJECT_DIR / "train_soil_spatial.py"),
        (SCRIPTS_DIR / "run_soil_spatial.sh",     LOG_DIR     / "run_soil_spatial.sh"),
    ]

    print("Deploying files to Talon home directory...")
    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            sz = dst.stat().st_size
            print(f"  ✓ {dst.name:<35} {sz:,} bytes")
        else:
            print(f"  ✗ Source not found: {src}")
            print(f"    Upload {src.name} to {SCRIPTS_DIR} first.")
            return False

    print("\n  Files deployed. Ready to submit.")
    print(f"  Training script : {PROJECT_DIR / 'train_soil_spatial.py'}")
    print(f"  SLURM script    : {LOG_DIR / 'run_soil_spatial.sh'}")
    return True


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — submit()
# Submits the SLURM job. Returns immediately.
# You can log out right after this.
# ════════════════════════════════════════════════════════════════════════════════

def submit():
    """
    Validate and submit the SLURM job.
    Returns immediately — your session is NOT needed after this.
    """
    slurm_script = LOG_DIR / "run_soil_spatial.sh"

    if not slurm_script.exists():
        print("✗ SLURM script not found. Run deploy() first.")
        return None

    # ── Validate before submitting ────────────────────────────────────────────
    print("Validating SLURM script...")
    val = subprocess.run(["sbatch", "--test-only", str(slurm_script)],
                          capture_output=True, text=True)
    msg = val.stdout.strip() or val.stderr.strip()
    print(f"  Validation: {msg}")

    if val.returncode != 0:
        print(f"\n✗ Validation failed.")
        print(f"  Common fixes:")
        print(f"    - Check account:   sacctmgr show user {os.environ.get('USER','')} withassoc format=account")
        print(f"    - Check partition: sinfo -p talon-gpu32")
        print(f"    - Check node:      scontrol show node talon32")
        return None

    # ── Submit ────────────────────────────────────────────────────────────────
    print("\nSubmitting job...")
    result = subprocess.run(["sbatch", str(slurm_script)],
                             capture_output=True, text=True,
                             cwd=str(PROJECT_DIR))

    if result.returncode != 0:
        print(f"✗ Submission failed: {result.stderr.strip()}")
        print(f"  Manual fallback:  sbatch {slurm_script}")
        return None

    job_id = result.stdout.strip().split()[-1]

    # ── Save job record ───────────────────────────────────────────────────────
    record = dict(job_id=job_id, submitted_at=pd.Timestamp.now().isoformat(),
                  slurm_script=str(slurm_script),
                  train_script=str(PROJECT_DIR/"train_soil_spatial.py"))
    with open(LOG_DIR / "last_job.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  JOB SUBMITTED SUCCESSFULLY                                  ║
║                                                              ║
║  Job ID     : {job_id:<46}║
║  Partition  : talon-gpu32                                    ║
║  Wall time  : 24 hours                                       ║
╠══════════════════════════════════════════════════════════════╣
║  YOUR SESSION CAN END NOW.                                   ║
║  The job runs under SLURM — completely independent.          ║
╠══════════════════════════════════════════════════════════════╣
║  MONITOR (from any new session):                             ║
║    squeue -j {job_id:<48}║
║    squeue --me                                               ║
║    tail -f ~/logs/soil_training.log                          ║
║    tail -f ~/logs/soil_spatial_{job_id}.out                 ║
╠══════════════════════════════════════════════════════════════╣
║  COLLECT RESULTS (after job finishes):                       ║
║    python3 LAUNCHER.py  → then call collect()                ║
╚══════════════════════════════════════════════════════════════╝
""")
    return job_id


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2b — status()
# Check job and checkpoint state from any new session.
# ════════════════════════════════════════════════════════════════════════════════

def status():
    """
    Full status snapshot — safe to run any time, even after session restart.
    Shows: SLURM queue, job history, checkpoint inventory, log tail.
    """
    import torch
    import numpy as np

    print("=" * 60)
    print(f"  STATUS  {pd.Timestamp.now()}")
    print("=" * 60)

    # ── Last submitted job ────────────────────────────────────────────────────
    job_file = LOG_DIR / "last_job.json"
    job_id   = None
    if job_file.exists():
        with open(job_file) as f: rec = json.load(f)
        job_id = rec.get("job_id")
        print(f"\n  Last job: {job_id} | submitted {rec.get('submitted_at','N/A')[:19]}")

    # ── SLURM queue ───────────────────────────────────────────────────────────
    print("\n  SLURM queue (squeue --me):")
    r = subprocess.run(["squeue","--me","--format=%i %j %T %M %N","--noheader"],
                        capture_output=True, text=True)
    for line in (r.stdout.strip().split("\n") if r.stdout.strip() else ["    (empty)"]):
        print(f"    {line}")

    # ── Job history via sacct ─────────────────────────────────────────────────
    if job_id:
        print(f"\n  Job {job_id} history (sacct):")
        r2 = subprocess.run(
            ["sacct","-j",job_id,"--format=JobID,State,ExitCode,Elapsed","--noheader"],
            capture_output=True, text=True)
        for line in r2.stdout.strip().split("\n"):
            if line.strip(): print(f"    {line}")

    # ── Checkpoint inventory ──────────────────────────────────────────────────
    mdir = PROJECT_DIR / "models" / "dl"
    print(f"\n  CHECKPOINTS  ({mdir}):")
    print(f"  {'Model':<22} {'temp':>8} {'smap':>8} {'moist':>8}")
    print("  " + "─" * 50)
    for arch in ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]:
        row = f"  {arch:<22}"
        for tgt in ["temp","smap","moist"]:
            ckpt = mdir / f"{arch}_{tgt}_spatial_best.pt"
            if ckpt.exists():
                try:
                    d  = torch.load(ckpt, map_location="cpu")
                    r2 = d.get("val_r2", float("nan"))
                    row += f" {r2:>8.4f}"
                except Exception:
                    row += f"{'ERR':>8}"
            else:
                row += f"{'missing':>8}"
        print(row)

    # ── Log tail ──────────────────────────────────────────────────────────────
    log = LOG_DIR / "soil_training.log"
    if log.exists():
        print(f"\n  LOG TAIL  ({log}):")
        with open(log) as f: lines = f.readlines()
        for l in lines[-15:]: print(f"    {l.rstrip()}")
    else:
        print("\n  Log not created yet (job hasn't started or is queued).")

    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — collect()
# Load results after job completes. Safe to run any time.
# ════════════════════════════════════════════════════════════════════════════════

def collect():
    """
    Load all results from checkpoints and CSVs.
    Safe to call any time — works even after session restart.
    Returns a DataFrame of all model results.
    """
    import torch
    import numpy as np

    print("Collecting results...")
    RESULTS_DIR = PROJECT_DIR / "results"
    MODELS_DIR  = PROJECT_DIR / "models" / "dl"
    records = []

    for ckpt in sorted(MODELS_DIR.glob("*_spatial_best.pt")):
        try:
            d    = torch.load(ckpt, map_location="cpu")
            stem = ckpt.stem.replace("_spatial_best","")
            # stem: {arch}_{tgt}  — split on LAST underscore
            parts = stem.rsplit("_",1)
            if len(parts) != 2: continue
            arch, tgt = parts
            tm = d.get("test_metrics",{})
            es = d.get("entropy_summary",{})
            records.append(dict(
                Model=arch, Target=tgt,
                Val_R2=d.get("val_r2",float("nan")),
                Test_R2=tm.get("R2",float("nan")),
                Test_RMSE=tm.get("RMSE",float("nan")),
                Test_Skill=tm.get("Skill",float("nan")),
                Test_KGE=tm.get("KGE",float("nan")),
                Test_FreezeAcc=tm.get("Freeze_Acc",float("nan")),
                NodeR2_mean=tm.get("node_r2_mean",float("nan")),
                SpatialVarRatio=tm.get("spatial_var_ratio",float("nan")),
                Entropy_H=es.get("final_H",float("nan")),
                Diagnosis=es.get("diagnosis","N/A"),
                Train_time_s=d.get("elapsed_s",float("nan")),
                Job_ID=d.get("job_id","N/A"),
                Node=d.get("node","N/A")))
            print(f"  ✓ {arch:<20} [{tgt}]  val_r2={d.get('val_r2',float('nan')):.4f}")
        except Exception as e:
            print(f"  ✗ {ckpt.name}: {e}")

    if not records:
        csv = RESULTS_DIR / "spatial_results_all.csv"
        if csv.exists():
            print(f"  Loading from CSV: {csv}")
            return pd.read_csv(csv)
        print("  No results yet — job may still be running.")
        return pd.DataFrame()

    df = (pd.DataFrame(records)
          .sort_values("Test_R2", ascending=False)
          .reset_index(drop=True))

    df.to_csv(RESULTS_DIR / "spatial_results_all.csv", index=False)

    # ── Print leaderboard ─────────────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  LEADERBOARD  ({len(df)} records)")
    print(f"  {'='*70}")
    print(f"  {'Model':<20} {'Target':<8} {'Val R²':>8} {'Test R²':>8} "
          f"{'Skill':>8} {'FreezeAcc':>10} {'Diagnosis':<22}")
    print("  " + "─"*80)
    for _, row in df.iterrows():
        beat = "✓" if (not pd.isna(row.get("Test_Skill")) and row.get("Test_Skill",-1)>0) else "✗"
        diag = str(row.get("Diagnosis","N/A"))
        ds   = "LEARNING" if "LEARNING" in diag else "SEASONAL" if "SEASONAL" in diag else "N/A"
        print(f"  {beat} {row['Model']:<19} {row['Target']:<8} "
              f"{row.get('Val_R2',float('nan')):>8.4f} "
              f"{row.get('Test_R2',float('nan')):>8.4f} "
              f"{row.get('Test_Skill',float('nan')):>8.4f} "
              f"{row.get('Test_FreezeAcc',float('nan')):>9.2f}% "
              f"{ds:<22}")

    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2c — resubmit_missing()
# If some models failed or timed out, resubmit.
# Already-completed checkpoints will be skipped automatically.
# ════════════════════════════════════════════════════════════════════════════════

def resubmit_missing():
    """
    Check which models still need to run and resubmit the SLURM job.
    Already-completed checkpoints are SKIPPED automatically by the training script.
    Just re-run the same sbatch — no changes needed.
    """
    import torch
    import numpy as np

    MODELS_DIR = PROJECT_DIR / "models" / "dl"
    ARCHES     = ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]
    TGTS       = ["temp","smap","moist"]

    missing = []
    for arch in ARCHES:
        for tgt in TGTS:
            ckpt = MODELS_DIR / f"{arch}_{tgt}_spatial_best.pt"
            if not ckpt.exists():
                missing.append(f"{arch}/{tgt}")
                continue
            try:
                d = torch.load(ckpt, map_location="cpu")
                if d.get("val_r2",-99) <= -10:
                    missing.append(f"{arch}/{tgt} (corrupt)")
            except Exception:
                missing.append(f"{arch}/{tgt} (unreadable)")

    print(f"Missing or failed: {len(missing)} / {len(ARCHES)*len(TGTS)}")
    for m in missing: print(f"  ✗ {m}")

    if not missing:
        print("\n  All checkpoints present — nothing to resubmit.")
        return

    print(f"\n  Resubmitting (completed models will be skipped)...")
    return submit()


# ════════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ════════════════════════════════════════════════════════════════════════════════

QUICK_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════╗
║  QUICK REFERENCE — UND Talon SLURM Workflow                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  FIRST TIME:                                                         ║
║    from LAUNCHER import deploy, submit, status, collect              ║
║    deploy()        # copy scripts to ~/                              ║
║    submit()        # submit job — you can log out immediately        ║
║                                                                      ║
║  MONITORING (from any new session):                                  ║
║    status()        # SLURM queue + checkpoint inventory + log tail   ║
║    squeue --me     # quick queue check from terminal                 ║
║                                                                      ║
║  AFTER JOB FINISHES:                                                 ║
║    df = collect()  # load results from checkpoints                   ║
║                                                                      ║
║  IF JOB TIMED OUT OR FAILED:                                         ║
║    resubmit_missing()   # skip completed, redo the rest              ║
║                                                                      ║
║  FILE LOCATIONS:                                                      ║
║    Training script : ~/train_soil_spatial.py                         ║
║    SLURM script    : ~/logs/run_soil_spatial.sh                      ║
║    Training log    : ~/logs/soil_training.log                        ║
║    SLURM stdout    : ~/logs/soil_spatial_<JOB_ID>.out                ║
║    Checkpoints     : ~/models/dl/*.pt                                ║
║    Results CSV     : ~/results/spatial_results_all.csv               ║
║    Figures         : ~/figures/SP0*.png                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)
    print("Available functions: deploy() | submit() | status() | collect() | resubmit_missing()")
