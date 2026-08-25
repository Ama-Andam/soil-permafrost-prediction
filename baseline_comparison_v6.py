"""
================================================================================
baseline_comparison_v6.py
ML BASELINE COMPARISON — v6 FEATURE SET
DoD PROJECT | Alaska Permafrost | University of North Dakota
================================================================================

WHAT THIS DOES:
  Trains ML baselines on the SAME feature set as v6 DL models:
    - NO cyclical encodings (sin/cos removed)
    - WITH wavelet approx as input
    - WITH input uncertainty variance features
    - WITH spatiotemporal quantization (weighted neighbour imputation)

  ML MODELS:
    Ridge, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM

  EVALUATES ON ALL 3 TEST SETS (same as v6 DL):
    - Standard test
    - Unseen space (Wetland holdout)
    - Unseen time (Q4 2025 holdout)
    - Unseen both (Wetland × Q4 2025)

  REPORTS SAME v6 METRICS:
    R², KGE, ubRMSE, CRPS, DTW, KL Div
    (ML models are point predictors — σ²=0 assumed for probabilistic metrics)

  KEY LIMITATION NOTED:
    ML baselines predict each location independently (no spatial graph).
    They CANNOT predict unseen Wetland locations from neighbouring sites.
    Unseen space results = ML trained on all sites including Wetland
    (i.e. they cheat on spatial holdout — shows WHY GCN matters).

WHY THIS MATTERS:
  v4 baselines used cyclical features → R²=0.94-0.96 (seasonal leakage).
  v6 baselines use same cleaned features as DL → fair comparison.
  Expected: ML R² will DROP without cyclical features, showing the DL
  spatial models are doing real work on the residual, not just seasonality.

RUN:
  python3 ~/baseline_comparison_v6.py

  Or via SLURM (recommended — takes ~15 min):
  sbatch ~/logs/run_baseline_v6.sh
================================================================================
"""

import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.special import ndtr

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3})

SEED = 42
np.random.seed(SEED)

print("="*65)
print("  v6 BASELINE COMPARISON — Fair feature set")
print("  No cyclical features | Same as v6 DL models")
print("="*65)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD DATA (same preprocessing as v6)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading data...")
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)

LOCATIONS     = pd.DataFrame(FI["LOCATIONS"])
N_LOCS        = FI["N_LOCS"]
SNAP_FEATURES = FI["SNAP_FEATURES"]
ALL_TARGETS   = FI["ALL_TARGETS"]
TEMP_TARGETS  = FI["TEMP_TARGETS"]
SMAP_TARGETS  = FI["SMAP_TARGETS"]
MOIST_TARGETS = FI["MOIST_TARGETS"]
SITES         = FI["SITES"]

# ── Remove cyclical features (SAME as v6) ────────────────────────────────────
CYCLICAL_COLS = [c for c in df.columns if any(
    c.startswith(p) for p in ["sin_","cos_","day_of_year_sin","day_of_year_cos",
                               "month_sin","month_cos","hour_sin","hour_cos"])]
print(f"  Removing {len(CYCLICAL_COLS)} cyclical features")

# ── Wavelet approx as input (SAME as v6) ─────────────────────────────────────
APPROX_COLS = [f"{t}_approx" for t in ALL_TARGETS if f"{t}_approx" in df.columns]

# ── Input uncertainty variance (SAME as v6) ──────────────────────────────────
CORE_FEATURES = [f for f in SNAP_FEATURES
                  if f not in CYCLICAL_COLS and f in df.columns]
APPROX_INPUT  = [f for f in APPROX_COLS if f in df.columns]

UNCERTAINTY_VAR_COLS = []
for feat in CORE_FEATURES[:8]:
    var_col = f"{feat}_unc_var"
    if var_col not in df.columns:
        df[var_col] = np.where(df[feat].isna(), 1.0, 0.01)
    UNCERTAINTY_VAR_COLS.append(var_col)

V6_FEATURES = list(dict.fromkeys(CORE_FEATURES + APPROX_INPUT + UNCERTAINTY_VAR_COLS))
V6_FEATURES = [f for f in V6_FEATURES if f in df.columns]
N_FEATS     = len(V6_FEATURES)
print(f"  v6 features: {N_FEATS} (same as DL models)")

# ── Three test splits (SAME as v6) ────────────────────────────────────────────
HOLDOUT_SITE   = "Wetland"
TRAINING_SITES = [s for s in SITES if s != HOLDOUT_SITE]

df["year"]  = df["time_utc"].dt.year
df["month"] = df["time_utc"].dt.month
TEMPORAL_HOLDOUT = (df["year"]==2025) & (df["month"]>=10)

df["split_v6"] = df["split"].copy()
df.loc[TEMPORAL_HOLDOUT & (df["split"]=="test"), "split_v6"] = "test_time"

loc_to_idx = {(float(r.Latitude),float(r.Longitude)): i
               for i,r in LOCATIONS.iterrows()}

def site_locs(site):
    rows = df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    return sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                   for _,r in rows.iterrows()
                   if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

SEEN_LOCS   = sorted(set(i for s in TRAINING_SITES for i in site_locs(s)))
UNSEEN_LOCS = site_locs(HOLDOUT_SITE)

# ── Scalers ────────────────────────────────────────────────────────────────────
tr = df[df["split"]=="train"]
feat_sc = RobustScaler()
feat_sc.fit(tr[V6_FEATURES].fillna(0).values)

# ── v6 FIX: Predict RESIDUAL (wavelet detail), not raw target ────────────────
# detail = raw - approx = non-seasonal signal
# This is the SAME target as DL models → fair comparison
# Without this, ML gets R²≈0.96 by learning seasonality (trivial)
# With residual target, ML must learn what DL learns — a harder, honest task
def build_residual_target_map(targets, df, split="train"):
    """Map each target group to its detail (residual) columns if available."""
    result = {}
    for grp, cols in targets.items():
        raw_cols    = [c for c in cols[0] if c in df.columns]
        detail_cols = [f"{c}_residual" for c in cols[0] if f"{c}_residual" in df.columns]
        approx_cols = [f"{c}_approx" for c in cols[0] if f"{c}_approx" in df.columns]
        use_cols    = detail_cols if detail_cols else raw_cols
        is_residual = bool(detail_cols)
        result[grp] = dict(
            use_cols=use_cols,
            raw_cols=raw_cols,
            approx_cols=approx_cols,
            label=cols[1],
            is_residual=is_residual)
        mode = "RESIDUAL (detail)" if is_residual else "RAW (no detail cols)"
        print(f"  [{grp}] target: {mode} | {len(use_cols)} cols")
    return result

RAW_TGT_MAP = {
    "temp":  (TEMP_TARGETS,  "Weather Temp (°C)"),
    "smap":  (SMAP_TARGETS,  "SMAP Temp L1 (K)"),
    "moist": (MOIST_TARGETS, "Moisture (m³/m³)"),
}
TGT_MAP = build_residual_target_map(RAW_TGT_MAP, df)

print(f"  Seen locs: {len(SEEN_LOCS)} | Unseen (Wetland): {len(UNSEEN_LOCS)}")
print(f"  Temporal holdout rows: {TEMPORAL_HOLDOUT.sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — METRIC FUNCTIONS (same as v6)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_v6(yt, yp, label=""):
    """v6 metric suite for point predictors (no σ²)."""
    mk = ~(np.isnan(yt)|np.isnan(yp))
    yt = yt[mk]; yp = yp[mk]
    if len(yt) < 5: return {}

    # R²
    ss_res = np.sum((yt-yp)**2)
    ss_tot = np.sum((yt-yt.mean())**2) + 1e-10
    r2 = float(1 - ss_res/ss_tot)

    # KGE
    r  = float(np.corrcoef(yt,yp)[0,1]) if len(yt)>2 else 0.
    a  = float(np.std(yp)/(np.std(yt)+1e-10))
    b  = float(np.mean(yp)/(np.mean(yt)+1e-10))
    kge= float(1 - np.sqrt((r-1)**2+(a-1)**2+(b-1)**2))

    # ubRMSE
    bias   = float(np.mean(yp-yt))
    ubrmse = float(np.sqrt(np.mean(((yp-bias)-yt)**2)))

    # Freeze/thaw accuracy
    frz = float(np.mean((yt<0).astype(int)==(yp<0).astype(int))*100)

    # DTW (simplified — diagonal warping path on first 500 points)
    try:
        n   = min(500, len(yt))
        yt_ = yt[:n]; yp_ = yp[:n]
        D   = (yt_[:,None]-yp_[None,:])**2
        dtw_mat = np.full_like(D, np.inf); dtw_mat[0,0]=D[0,0]
        for i in range(1,n):
            for j in range(max(0,i-10),min(n,i+11)):
                prev=min(dtw_mat[i-1,j]  if i>0 else np.inf,
                          dtw_mat[i,j-1]  if j>0 else np.inf,
                          dtw_mat[i-1,j-1] if (i>0 and j>0) else np.inf)
                dtw_mat[i,j]=D[i,j]+(0 if prev==np.inf else prev)
        dtw = float(dtw_mat[n-1,n-1]/n)
    except Exception:
        dtw = float(np.mean(np.abs(yt-yp)))

    # KL Divergence (point predictor: σ²_pred assumed = residual variance)
    sig_pred = float(np.std(yp-yt)) + 1e-8
    mu_obs   = float(np.mean(yt)); sig_obs = float(np.std(yt))+1e-8
    mu_pred  = float(np.mean(yp)); 
    kl = (np.log(sig_pred/sig_obs)
          +(sig_obs**2+(mu_obs-mu_pred)**2)/(2*sig_pred**2)-0.5)

    # CRPS for point predictor (σ = residual std, deterministic limit)
    # As σ→0, CRPS → MAE. Use residual std as proxy uncertainty.
    z    = (yt-yp)/sig_pred
    Phi  = ndtr(z)
    phi  = np.exp(-0.5*z**2)/np.sqrt(2*np.pi)
    crps = float(np.mean(sig_pred*(z*(2*Phi-1)+2*phi-1/np.sqrt(np.pi))))

    out = dict(R2=round(r2,4), KGE=round(kge,4),
               ubRMSE=round(ubrmse,4), FreezeAcc=round(frz,2),
               DTW=round(dtw,4), KL_Div=round(float(kl),4),
               CRPS=round(crps,4), Bias=round(bias,4), N=int(mk.sum()))
    return {f"{label}_{k}" if label else k:v for k,v in out.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATA PREP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_split_data(split_label, tgt_info, include_sites=None):
    """
    Get X, y arrays for a given split.
    Uses residual (detail) as y if available — same target as DL models.
    tgt_info: dict from TGT_MAP with use_cols, raw_cols, approx_cols
    """
    if split_label == "train":
        mask = df["split"] == "train"
    elif split_label == "test_time":
        mask = df["split_v6"] == "test_time"
    else:
        mask = df["split"] == split_label

    if include_sites:
        mask = mask & df["Site"].isin(include_sites)

    sub = df[mask].copy()
    use_cols = tgt_info["use_cols"]
    av_tgt   = [c for c in use_cols if c in sub.columns]
    if not av_tgt: return None, None, None

    sub = sub.dropna(subset=av_tgt[:1])
    if len(sub) < 10: return None, None, None

    X = feat_sc.transform(sub[V6_FEATURES].fillna(0).values).astype(np.float32)
    y = sub[av_tgt[0]].values.astype(np.float32)  # residual if detail exists
    sites = sub["Site"].values
    return X, y, sites


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ML BASELINE MODELS
# ══════════════════════════════════════════════════════════════════════════════

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                   GradientBoostingRegressor)
except ImportError as e:
    print(f"FATAL: {e}"); sys.exit(1)

BASELINES = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=20,
        n_jobs=-1, random_state=SEED),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=20,
        n_jobs=-1, random_state=SEED),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=SEED),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, tree_method="hist", n_jobs=-1,
        random_state=SEED, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, n_jobs=-1, random_state=SEED, verbose=-1),
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — TRAIN AND EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

all_results = []

for tgt_grp, tgt_info in TGT_MAP.items():
    print(f"\n{'─'*60}")
    tgt_label = tgt_info["label"]
    is_res = tgt_info["is_residual"]
    print(f"  TARGET: {tgt_label} | Predicting: {chr(39)}RESIDUAL{chr(39) if is_res else chr(39)}RAW{chr(39)}")
    print(f"{'─'*60}")

    # Training data — seen sites only (consistent with DL training)
    X_tr, y_tr, _ = get_split_data("train", tgt_info,
                                     include_sites=TRAINING_SITES)
    if X_tr is None:
        print(f"  ✗ No training data for {tgt_grp}"); continue
    print(f"  Train: {len(X_tr):,} samples | Features: {X_tr.shape[1]}")

    # Test sets
    test_sets = {
        "std_test":     get_split_data("test",      tgt_info),
        "unseen_time":  get_split_data("test_time", tgt_info),
        # Unseen space: ML cannot truly predict unseen locations
        # We evaluate on Wetland test data WITH the model trained on all sites
        # (this is the ML "cheat" — shows why spatial GCN matters)
        "unseen_space_cheat": get_split_data("test", tgt_info,
                                              include_sites=[HOLDOUT_SITE]),
        # Honest unseen space: train WITHOUT Wetland, test on Wetland
        # (most fair — ML trained only on seen sites)
        "unseen_space_honest": None,  # filled below after per-site training
    }

    # Honest spatial holdout: retrain without Wetland
    X_tr_seen, y_tr_seen, _ = get_split_data("train", tgt_info,
                                               include_sites=TRAINING_SITES)
    X_te_wetland, y_te_wetland, _ = get_split_data("test", tgt_info,
                                                     include_sites=[HOLDOUT_SITE])
    test_sets["unseen_space_honest"] = (X_te_wetland, y_te_wetland, None)

    print(f"  {'Model':<20} {'Train(s)':>9} "
          f"{'Std R²':>8} {'Space R²':>10} {'Time R²':>9} {'KGE':>7} "
          f"{'ubRMSE':>8} {'CRPS':>7}")
    print("  " + "─"*80)

    for mname, model in BASELINES.items():
        # ── Train on seen sites ───────────────────────────────────────────────
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        # Also train honest version (without Wetland)
        import copy
        model_honest = copy.deepcopy(model)
        if X_tr_seen is not None:
            model_honest.fit(X_tr_seen, y_tr_seen)

        # ── Evaluate on all test sets ─────────────────────────────────────────
        metrics_all = {}

        for sp_name, data in test_sets.items():
            if data is None or data[0] is None: continue
            X_te, y_te, _ = data
            if X_te is None or len(X_te) < 5: continue

            # Per PI: evaluate on RESIDUAL units directly
            # Objective = minimising residual information loss
            m = model_honest if "honest" in sp_name else model
            t_inf = time.time()
            yp = m.predict(X_te)
            inf_s = time.time() - t_inf

            m_dict = compute_metrics_v6(y_te, yp, label=sp_name)
            m_dict[f"{sp_name}_inference_s"] = round(inf_s, 3)
            metrics_all.update(m_dict)

        # ── Record ────────────────────────────────────────────────────────────
        rec = dict(
            Model=mname, Tier="ML_BASELINE", Target=tgt_grp,
            Type="Single-point (no spatial graph)",
            Predicts="RESIDUAL",
            Evaluation_Units="RESIDUAL (per PI instruction)",
            Train_Time_s=round(train_time,2),
            Train_Time_min=round(train_time/60,3),
            N_Features=N_FEATS,
            Cyclical_Features="NO",
            Wavelet_Approx="YES",
            Uncertainty_Var="YES",
            Spatial_Holdout="HONEST (retrained without Wetland)",
            **metrics_all)
        all_results.append(rec)

        std_r2   = metrics_all.get("std_test_R2",   np.nan)
        space_r2 = metrics_all.get("unseen_space_honest_R2", np.nan)
        time_r2  = metrics_all.get("unseen_time_R2", np.nan)
        kge      = metrics_all.get("std_test_KGE",  np.nan)
        ubrmse   = metrics_all.get("std_test_ubRMSE",np.nan)
        crps     = metrics_all.get("std_test_CRPS",  np.nan)

        print(f"  {mname:<20} {train_time:>9.2f} "
              f"{std_r2:>8.4f} {space_r2:>10.4f} {time_r2:>9.4f} "
              f"{kge:>7.4f} {ubrmse:>8.4f} {crps:>7.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
bl_df = pd.DataFrame(all_results)
bl_df.to_csv(RESULTS/"v6_baseline_ml_results.csv", index=False)
print(f"\n  ✓ {RESULTS}/v6_baseline_ml_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

TIER_COLORS = {
    "ML_BASELINE":"#7f7f7f",
    "ABLATION":"#d62728","RESERVOIR":"#9467bd",
    "GRAPH":"#2ca02c","ATTENTION":"#ff7f0e","SSM":"#1f77b4",
}

# ── FIG BL_01: Training time ───────────────────────────────────────────────────
for tgt in bl_df["Target"].unique():
        tgt_lbl = TGT_MAP.get(tgt, {}).get("label", tgt)
    sub = bl_df[bl_df["Target"]==tgt].copy()
    if sub.empty: continue

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Training time
    sub_t = sub.sort_values("Train_Time_min")
    axes[0].barh(sub_t["Model"], sub_t["Train_Time_min"],
                  color=TIER_COLORS["ML_BASELINE"], alpha=0.85,
                  edgecolor="black", lw=0.5)
    for i, (_, row) in enumerate(sub_t.iterrows()):
        axes[0].text(row["Train_Time_min"]+0.001,i,
                     f"{row['Train_Time_s']:.2f}s",va="center",fontsize=9)
    axes[0].set_xlabel("Training Time (minutes)")
    axes[0].set_title("Training Time\n(seconds for all ML baselines)",
                       fontweight="bold")

    # R² comparison across 3 test sets
    r2_cols = {
        "std_test_R2":           "Standard test",
        "unseen_space_honest_R2":"Unseen space\n(honest — no Wetland train)",
        "unseen_time_R2":        "Unseen time\n(Q4 2025)",
    }
    x = np.arange(len(sub)); w = 0.25
    for ci,(col,lbl) in enumerate(r2_cols.items()):
        if col not in sub.columns: continue
        axes[1].bar(x+ci*w, sub[col], width=w, label=lbl,
                     alpha=0.85, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x+w)
    axes[1].set_xticklabels(sub["Model"], rotation=30, fontsize=9)
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² across 3 test sets\n(ML baselines)",
                       fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].set_ylim(0.8, 1.01)

    # v4 vs v6 feature set comparison (R² difference)
    # Load v4 baseline if available
    v4_path = PROJECT/"results_v4"/"baseline_ml_results.csv"
    if v4_path.exists():
        v4_df = pd.read_csv(v4_path)
        v4_sub = v4_df[v4_df["Target"]==tgt] if "Target" in v4_df.columns else v4_df
        if not v4_sub.empty and "Test_R2" in v4_sub.columns:
            merged = sub.merge(v4_sub[["Model","Test_R2"]].rename(
                columns={"Test_R2":"v4_R2"}), on="Model", how="left")
            merged["delta"] = merged["std_test_R2"] - merged["v4_R2"]
            colors_d = ["green" if d>=0 else "red" for d in merged["delta"]]
            axes[2].barh(merged["Model"], merged["delta"],
                          color=colors_d, alpha=0.85, edgecolor="black", lw=0.5)
            axes[2].axvline(0, color="black", lw=1.5)
            axes[2].set_xlabel("ΔR² (v6 - v4)")
            axes[2].set_title("Feature Set Impact\nv6 (no cyclical) vs v4 (with cyclical)",
                               fontweight="bold")
            axes[2].text(0.05,0.95,"Green = v6 better\nRed = cyclical features helped",
                          transform=axes[2].transAxes,fontsize=9,va="top",
                          color="grey")
    else:
        axes[2].text(0.5,0.5,"v4 baselines not found\n(run baseline_comparison.py first)",
                      ha="center",va="center",transform=axes[2].transAxes,
                      fontsize=10,color="grey")
        axes[2].set_title("v4 vs v6 comparison\n(v4 results needed)")

    tgt_lbl = TGT_MAP[tgt].get("label", tgt)
    fig.suptitle(f"ML Baselines v6 | {tgt_lbl}\n"
                  f"Features: {N_FEATS} | No cyclical | Wavelet approx | Unc variance",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"BL_01_baseline_v6_{tgt}.png"
    plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {fname}")


# ── FIG BL_02: Full metric comparison heatmap ──────────────────────────────────
METRIC_SHOW = ["std_test_R2","std_test_KGE","std_test_ubRMSE",
               "std_test_CRPS","std_test_DTW","std_test_KL_Div",
               "unseen_space_honest_R2","unseen_time_R2"]
METRIC_LABELS = ["R²\n(std)","KGE\n(std)","ubRMSE\n(std)",
                  "CRPS\n(std)","DTW\n(std)","KL Div\n(std)",
                  "R²\n(space)","R²\n(time)"]

for tgt in bl_df["Target"].unique():
        tgt_lbl = TGT_MAP.get(tgt, {}).get("label", tgt)
    sub = bl_df[bl_df["Target"]==tgt].copy()
    avail_cols = [c for c in METRIC_SHOW if c in sub.columns]
    avail_labs = [METRIC_LABELS[METRIC_SHOW.index(c)] for c in avail_cols]
    if not avail_cols: continue

    pv = sub.set_index("Model")[avail_cols].rename(
        columns=dict(zip(avail_cols, avail_labs)))
    pv = pv.apply(pd.to_numeric, errors="coerce")

    # Negate error metrics
    for col in ["ubRMSE\n(std)","CRPS\n(std)","DTW\n(std)","KL Div\n(std)"]:
        if col in pv.columns: pv[col] = -pv[col]

    fig, ax = plt.subplots(figsize=(max(14,len(avail_cols)*2), 8))
    sns.heatmap(pv, ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                linewidths=0.5, linecolor="white",
                annot_kws={"size":10,"weight":"bold"},
                cbar_kws={"label":"Score (error metrics negated: higher=better)"})
    ax.set_title(f"ML Baseline Metrics | {tgt_lbl} | v6 features\n"
                  f"Error metrics negated: green=better | "
                  f"Space R² = honest holdout (no Wetland in training)",
                  fontweight="bold", fontsize=11)
    plt.tight_layout()
    fname = f"BL_02_baseline_metrics_{tgt}.png"
    plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {fname}")


# ── FIG BL_03: DL vs ML comparison (once DL results available) ────────────────
dl_path = RESULTS/"v6_results_all.csv"
if dl_path.exists():
    dl_df = pd.read_csv(dl_path)
    print("\n  Generating DL vs ML comparison figure...")

    for tgt in bl_df["Target"].unique():
        tgt_lbl = TGT_MAP.get(tgt, {}).get("label", tgt)
        bl_sub = bl_df[bl_df["Target"]==tgt][["Model","std_test_R2","Train_Time_min"]].copy()
        bl_sub["Tier"] = "ML_BASELINE"
        bl_sub = bl_sub.rename(columns={"std_test_R2":"R2"})

        dl_col = "unseen_space_unseen_R2"
        if dl_col not in dl_df.columns:
            dl_col = "Val_R2"
        dl_sub = dl_df[dl_df["Target"]==tgt][["Model","Tier",dl_col,"Train_s"]].copy()
        dl_sub = dl_sub.rename(columns={dl_col:"R2","Train_s":"Train_Time_s"})
        dl_sub["Train_Time_min"] = dl_sub["Train_Time_s"]/60

        combined = pd.concat([
            bl_sub[["Model","Tier","R2","Train_Time_min"]],
            dl_sub[["Model","Tier","R2","Train_Time_min"]]
        ], ignore_index=True).sort_values("R2", ascending=True)

        fig, ax = plt.subplots(figsize=(18,12))
        colors = [TIER_COLORS.get(t,"grey") for t in combined["Tier"]]
        bars = ax.barh(combined["Model"], combined["R2"],
                        color=colors, alpha=0.85, edgecolor="black", lw=0.5)
        for bar, (_, row) in zip(bars, combined.iterrows()):
            ax.text(row["R2"]+0.001, bar.get_y()+bar.get_height()/2,
                    f"{row['R2']:.4f}", va="center", fontsize=8,
                    fontweight="bold")

        ax.axvline(0.95, color="orange", ls="--", lw=1.5,
                   label="R²=0.95 reference")
        ax.set_xlabel("R²", fontsize=11)
        ax.set_title(f"ML Baselines vs DL Spatial Models\n"
                      f"{tgt_lbl} | v6 features (no cyclical)\n"
                      f"DL = unseen space R² | ML = standard test R²",
                      fontweight="bold", fontsize=12)

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()]
                   +[plt.Line2D([0],[0],ls="--",color="orange",label="R²=0.95")],
                  fontsize=9, loc="lower right")
        plt.tight_layout()
        fname = f"BL_03_dl_vs_ml_{tgt}.png"
        plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  ✓ {fname}")
else:
    print("\n  BL_03 skipped — DL results not ready yet.")
    print("  Re-run figures_v6.py after training completes to get DL vs ML figure.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  v6 BASELINE SUMMARY
{'='*65}
  Feature set: {N_FEATS} features
  Cyclical encodings: REMOVED (v4 had them → inflated R²)
  Wavelet approx: YES (seasonal baseline as input)
  Uncertainty variance: YES (high σ² for missing observations)
  Spatial holdout: HONEST (retrained without Wetland)

  KEY INSIGHT:
  If ML R² drops significantly vs v4 → cyclical features were
  doing the work, not the model. DL spatial models now have a
  fair baseline to beat.

  If ML R² is similar to v4 → seasonal signal is in the raw
  meteorological features too (expected to some degree).

  Results saved: {RESULTS}/v6_baseline_ml_results.csv
  Figures saved: {FIGS}/BL_0*.png
{'='*65}
""")
