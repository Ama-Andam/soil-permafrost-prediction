"""
================================================================================
uncertainty_analysis_v5.py
MC-DROPOUT UNCERTAINTY ANALYSIS & RECOVERABILITY WITH UNCERTAINTY BANDS
================================================================================

WHAT THIS DOES:
  1. Loads v5_uncertainty_mc.csv (from train_soil_spatial_v5.py --mode uncertainty)
  2. Plots per-model uncertainty: seen vs unseen (Wetland)
  3. Uncertainty ratio chart: unseen/seen > 1 means model knows it's uncertain
  4. Calibration plot: uncertainty vs actual error (good model: r > 0.5)
  5. ENHANCED RECOVERABILITY: uncertainty band around recoverability curve
     - Shows credible interval (5th–95th percentile from MC samples)
     - Models with tight bands → confident predictions
     - Models with wide bands → uncertain, more caution needed

KEY INTERPRETATION:
  uncertainty_ratio > 1.2 → model correctly more uncertain for unseen locations
  calibration_r > 0.5     → model uncertainty tracks actual error (well calibrated)
  calibration_r < 0.2     → overconfident (uncertainty doesn't predict errors)

RUN:
  python3 ~/uncertainty_analysis_v5.py
================================================================================
"""

import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
RESULTS = PROJECT / "results_v5"
FIGS    = PROJECT / "figures_v5"
FIGS.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi":150, "font.size":11,
                              "axes.grid":True, "grid.alpha":0.3})

TIER_COLORS = {"ABLATION":"#d62728","RESERVOIR":"#9467bd",
               "GRAPH":"#2ca02c","SSM":"#1f77b4"}
ARCH_TIERS  = {
    "BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
    "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
    "SpatialS4":"SSM","SpatialFuseMoE":"SSM"}

print("=" * 65)
print("  v5 UNCERTAINTY ANALYSIS — MC-Dropout")
print("=" * 65)

# ── Load results ──────────────────────────────────────────────────────────────
unc_path = RESULTS / "v5_uncertainty_mc.csv"
if not unc_path.exists():
    print(f"  ✗ {unc_path} not found.")
    print("  Run: python3 ~/train_soil_spatial_v5.py --mode uncertainty")
    exit(1)

unc_df = pd.read_csv(unc_path)
print(f"  Loaded: {len(unc_df)} records")
print(unc_df[["arch","target","unc_ratio","calibration_seen","calibration_unseen"]].to_string())

TGT_LABELS = {"temp":"Weather Temp (°C)",
              "smap":"SMAP Temp L1 (K)",
              "moist":"Moisture (m³/m³)"}
ARCHES = list(ARCH_TIERS.keys())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Uncertainty ratio — unseen/seen (per model)
# ══════════════════════════════════════════════════════════════════════════════
for tgt in unc_df["target"].unique():
    sub = unc_df[unc_df["target"]==tgt].copy()
    if sub.empty: continue
    sub = sub.sort_values("unc_ratio", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Uncertainty ratio
    ax = axes[0]
    colors = [TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey") for a in sub["arch"]]
    bars = ax.barh(sub["arch"], sub["unc_ratio"], color=colors,
                   alpha=0.85, edgecolor="black", lw=0.5)
    for bar, v in zip(bars, sub["unc_ratio"]):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}×", va="center", fontsize=9, fontweight="bold")
    ax.axvline(1.0, color="black", lw=2, label="1.0 = same uncertainty")
    ax.axvline(1.2, color="green", ls="--", lw=1.5, alpha=0.8,
               label="1.2× threshold (well calibrated)")
    ax.set_xlabel("Uncertainty Ratio (Unseen / Seen)", fontsize=11)
    ax.set_title("MC-Dropout Uncertainty Ratio\n"
                 ">1.0 = model correctly more uncertain for Wetland (unseen)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(sub["unc_ratio"].max() + 0.3, 2.0))

    # Calibration (uncertainty-error correlation)
    ax = axes[1]
    x = np.arange(len(sub)); w = 0.35
    b1 = ax.bar(x-w/2, sub["calibration_seen"],   width=w,
                label="Seen locs",   color="#1f77b4", alpha=0.85,
                edgecolor="black", lw=0.5)
    b2 = ax.bar(x+w/2, sub["calibration_unseen"], width=w,
                label="Unseen (Wetland)", color="#d62728", alpha=0.85,
                edgecolor="black", lw=0.5)
    for bar, v in zip(list(b1)+list(b2), list(sub["calibration_seen"])+list(sub["calibration_unseen"])):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="green", ls="--", lw=1.5, alpha=0.8,
               label="r=0.5 (good calibration threshold)")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{ARCH_TIERS.get(a,'?')}]\n{a}" for a in sub["arch"]],
                        rotation=30, fontsize=8)
    ax.set_ylabel("Calibration (r: uncertainty vs error)", fontsize=11)
    ax.set_title("MC-Dropout Calibration\n"
                 "r = correlation between predicted uncertainty and actual error",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9); ax.set_ylim(-0.1, 1.1)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
               loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(f"MC-Dropout Uncertainty Analysis | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"MC samples=30 | Spatial holdout: Wetland | v5",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0.05,1,1])
    fname = f"UNC_01_uncertainty_ratio_{tgt}.png"
    plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Seen vs Unseen uncertainty absolute values
# ══════════════════════════════════════════════════════════════════════════════
for tgt in unc_df["target"].unique():
    sub = unc_df[unc_df["target"]==tgt].copy()
    if sub.empty: continue
    sub = sub.sort_values("unc_seen_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(sub)); w = 0.35
    b1 = ax.barh(x-w/2, sub["unc_seen_mean"],   height=w, xerr=sub["unc_seen_std"],
                 label="Seen locs (mean±std)", color="#1f77b4",
                 alpha=0.85, edgecolor="black", lw=0.5,
                 capsize=3, error_kw={"elinewidth":1,"capthick":1})
    b2 = ax.barh(x+w/2, sub["unc_unseen_mean"], height=w, xerr=sub["unc_unseen_std"],
                 label="Unseen Wetland (mean±std)", color="#d62728",
                 alpha=0.85, edgecolor="black", lw=0.5,
                 capsize=3, error_kw={"elinewidth":1,"capthick":1})
    ax.set_yticks(x)
    ax.set_yticklabels([f"[{ARCH_TIERS.get(a,'?')}] {a}" for a in sub["arch"]],
                        fontsize=9)
    ax.set_xlabel("MC-Dropout Uncertainty (std of N=30 predictions, scaled units)",
                   fontsize=11)
    ax.set_title(f"Epistemic Uncertainty: Seen vs Unseen | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"Higher bars = more uncertain predictions | "
                 f"Unseen should have higher uncertainty",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
               loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5,-0.04))
    plt.tight_layout(rect=[0,0.05,1,1])
    fname = f"UNC_02_absolute_uncertainty_{tgt}.png"
    plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Summary heatmap — calibration by tier and target
# ══════════════════════════════════════════════════════════════════════════════
if len(unc_df) > 0 and "calibration_unseen" in unc_df.columns:
    sub = unc_df.copy()
    sub["tier"] = sub["arch"].map(ARCH_TIERS)
    pv = sub.pivot_table(index="arch", columns="target",
                          values="calibration_unseen", aggfunc="mean").round(3)
    if not pv.empty:
        fig, ax = plt.subplots(figsize=(14, max(8, len(pv)*0.7+2)))
        sns.heatmap(pv, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
                    annot_kws={"size":11,"weight":"bold"},
                    cbar_kws={"label":"Calibration r (uncertainty vs error)",
                              "shrink":0.85})
        ax.set_yticklabels([f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in pv.index],
                            rotation=0, fontsize=9)
        ax.set_xticklabels([TGT_LABELS.get(c,c) for c in pv.columns],
                            rotation=15, fontsize=10)
        ax.set_title("Calibration Heatmap — Unseen (Wetland) Locations\n"
                     "r=0.5+ = uncertainty reliably predicts actual error",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGS/"UNC_03_calibration_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  ✓ UNC_03_calibration_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Print interpretation guide
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  UNCERTAINTY INTERPRETATION GUIDE
{'='*65}

  UNCERTAINTY RATIO (Unseen/Seen):
    < 1.0   → Model is LESS uncertain for unseen Wetland (bad — overconfident)
    1.0–1.2 → Marginal awareness of spatial holdout
    > 1.2   → Model correctly identifies Wetland as uncertain (well-calibrated)
    > 2.0   → Model is highly uncertain for unseen (may need more regularisation)

  CALIBRATION (uncertainty vs error correlation):
    < 0.2   → Overconfident: uncertainty doesn't track actual errors
    0.2–0.5 → Moderate: some correlation between uncertainty and errors
    > 0.5   → Well calibrated: uncertainty reliably indicates problem regions
    > 0.7   → Excellent calibration (rare for spatial models)

  EXPECTED PATTERN:
    Models with GCN (spatial graph): should show higher unc_ratio
    because they know which locations were excluded from training.
    Models without GCN (BiGRU_NoGCN): may show unc_ratio ≈ 1.0
    because they predict each location independently.

  Figures saved to: {FIGS}
{'='*65}
""")
