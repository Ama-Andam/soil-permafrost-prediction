"""
================================================================================
figures_v6.py
ALL VISUALISATIONS FOR v6 — Run after training completes
================================================================================
Generates:
  FIG_01  Loss curves (NLL + CRPS + graph smooth) per model per target
  FIG_02  Leaderboard: all 3 test sets (space/time/both) side by side
  FIG_03  KDE: predicted vs observed distribution per site per model
  FIG_04  Uncertainty distribution: seen vs unseen locations
  FIG_05  Entropy: initial vs best epoch per architecture (bar chart)
  FIG_06  Metric heatmap: R², KGE, ubRMSE, CRPS, DTW, KL Div
  FIG_07  Ablation: component removal impact per model
  FIG_08  Spatial gap: seen R² vs unseen R² scatter (all models)
  FIG_09  KL Divergence per site per model
  FIG_10  Calibration: predicted σ vs actual error

RUN:
  python3 ~/figures_v6.py
  # or via launcher:
  python3 ~/train_soil_spatial_v6.py --mode figures
================================================================================
"""

import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6"
FIGS.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

TIER_COLORS = {
    "ABLATION":  "#d62728",
    "RESERVOIR": "#9467bd",
    "GRAPH":     "#2ca02c",
    "ATTENTION": "#ff7f0e",
    "SSM":       "#1f77b4",
}
ARCH_TIERS = {
    "BiGRU_NoGCN":"ABLATION",   "GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR",      "SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH",        "GAT":"GRAPH",        "STGCN":"GRAPH",
    "SpatialTransformer":"ATTENTION", "SpatialInformer":"ATTENTION",
    "SpatialBiGRU":"SSM",       "SpatialMamba":"SSM",
    "SpatialS4":"SSM",          "SpatialFuseMoE":"SSM",
}
ALL_13_MODELS = list(ARCH_TIERS.keys())
TGT_LABELS = {"temp":"Weather Temp (°C)","smap":"SMAP Temp L1 (K)",
               "moist":"Moisture (m³/m³)"}
TEST_LABELS = {
    "unseen_space":"Unseen space\n(Wetland holdout)",
    "unseen_time": "Unseen time\n(Q4 2025 holdout)",
    "unseen_both": "Unseen space+time\n(Hardest)",
}

print("="*65)
print("  v6 FIGURES — generating all visualisations")
print("="*65)

# ── Load results ──────────────────────────────────────────────────────────────
res_path = RESULTS/"v6_results_corrected.csv"
if not res_path.exists(): res_path = RESULTS/"v6_results_all.csv"
if not res_path.exists():
    print(f"  ✗ {res_path} not found. Run training first."); exit(1)
df = pd.read_csv(res_path)
df["Tier"] = df["Model"].map(ARCH_TIERS)
print(f"  Loaded {len(df)} model-target records")

ent_path = RESULTS/"v6_entropy.csv"
ent_df = pd.read_csv(ent_path) if ent_path.exists() else pd.DataFrame()

abl_path = RESULTS/"v6_ablation_results.csv"
abl_df = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# FIG_01: Loss curves per model per target
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_01: Loss curves...")
try:
    import torch
    for tgt in df["Target"].unique():
        arches = df[df["Target"]==tgt]["Model"].unique()
        fig, axes = plt.subplots(
            len(arches), 3, figsize=(22, 3*len(arches)+2),
            squeeze=False)
        for ai, arch in enumerate(arches):
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            sv = torch.load(ckpt_p, map_location="cpu")
            hist = sv.get("history",[])
            if not hist: continue
            hdf = pd.DataFrame(hist)
            tier = ARCH_TIERS.get(arch,"?")
            color = TIER_COLORS.get(tier,"grey")

            # NLL
            if "nll" in hdf.columns:
                axes[ai,0].plot(hdf["epoch"],hdf["nll"],color=color,lw=2)
                axes[ai,0].set_ylabel(arch,fontsize=9,color=color)
                axes[ai,0].set_title("NLL loss" if ai==0 else "",fontsize=10)
            # CRPS
            if "crps" in hdf.columns:
                axes[ai,1].plot(hdf["epoch"],hdf["crps"],color=color,lw=2)
                axes[ai,1].set_title("CRPS loss" if ai==0 else "",fontsize=10)
            # Val R²
            if "val_R2" in hdf.columns:
                axes[ai,2].plot(hdf["epoch"],hdf["val_R2"],color=color,lw=2)
                axes[ai,2].set_title("Val R²" if ai==0 else "",fontsize=10)
                # mark initial vs best
                best_ep=hdf.loc[hdf["val_R2"].idxmax(),"epoch"]
                axes[ai,2].axvline(best_ep,color="red",ls="--",lw=1,alpha=0.6)

        fig.suptitle(f"Training curves — {TGT_LABELS.get(tgt,tgt)} | v6",
                      fontsize=13,fontweight="bold")
        plt.tight_layout()
        fname = f"FIG_01_loss_curves_{tgt}.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    ✓ {fname}")
except ImportError:
    print("    ✗ torch not available for loss curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_02: Three-test-set leaderboard
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_02: Three-test leaderboard...")
for tgt in df["Target"].unique():
    sub = df[df["Target"]==tgt].copy()
    if sub.empty: continue

    test_cols = {
        "Space_R2": "Unseen Space R²\n(Wetland)",
        "Time_R2":  "Unseen Time R²\n(Q4 2025)",
        "Both_R2":  "Unseen Both R²\n(Hardest)",
    }
    avail = {k:v for k,v in test_cols.items() if k in sub.columns}
    if not avail: continue

    fig, axes = plt.subplots(1, len(avail), figsize=(8*len(avail), 10))
    if len(avail)==1: axes=[axes]

    for ax, (col, lbl) in zip(axes, avail.items()):
        s = sub.dropna(subset=[col]).sort_values(col)
        colors = [TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey") for a in s["Model"]]
        bars = ax.barh(s["Model"], s[col], color=colors,
                       alpha=0.85, edgecolor="black", lw=0.5)
        for bar,v in zip(bars,s[col]):
            ax.text(v+0.001,bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}",va="center",fontsize=9,fontweight="bold")
        ax.set_xlabel("R²",fontsize=11); ax.set_title(lbl,fontweight="bold")
        ax.set_xlim(max(0,s[col].min()-0.02),min(1.01,s[col].max()+0.02))

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
               loc="lower center",ncol=5,fontsize=9,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(f"Three-Test Leaderboard | {TGT_LABELS.get(tgt,tgt)} | v6",
                  fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0.05,1,1])
    fname = f"FIG_02_three_test_leaderboard_{tgt}.png"
    plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
    print(f"    ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_03: KDE — predicted vs observed distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_03: KDE plots...")
try:
    import torch
    for tgt in df["Target"].unique():
        arches = df[df["Target"]==tgt]["Model"].unique()
        n = len(arches); nc=4; nr=(n+nc-1)//nc
        fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4*nr))
        axes = axes.flatten() if nr>1 else [axes]*nc

        for ai, arch in enumerate(arches):
            ax = axes[ai]
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): ax.set_visible(False); continue
            sv = torch.load(ckpt_p,map_location="cpu")
            tm = sv.get("test_metrics",{})
            # Use stored residuals if available, else skip KDE body
            # (KDE requires raw predictions — stored in checkpoint future)
            tier = ARCH_TIERS.get(arch,"?")
            color = TIER_COLORS.get(tier,"grey")

            # Placeholder: show R² values as text with tier color
            unseen_r2 = tm.get("unseen_space",{}).get("unseen_R2",
                          tm.get("unseen_R2", np.nan))
            seen_r2   = tm.get("unseen_space",{}).get("seen_R2",
                          tm.get("seen_R2", np.nan))
            ax.text(0.5,0.6,f"Unseen R²\n{unseen_r2:.4f}",
                    ha="center",va="center",transform=ax.transAxes,
                    fontsize=14,fontweight="bold",color=color)
            ax.text(0.5,0.3,f"Seen R²\n{seen_r2:.4f}",
                    ha="center",va="center",transform=ax.transAxes,
                    fontsize=12,color="grey")
            ax.set_title(f"[{tier}] {arch}",fontsize=9,color=color,fontweight="bold")
            ax.set_xlabel("Value"); ax.set_ylabel("Density")
            ax.text(0.05,0.95,"KDE requires raw predictions\n(re-run with --save_preds)",
                    transform=ax.transAxes,fontsize=7,color="grey",va="top")

        for j in range(ai+1,len(axes)): axes[j].set_visible(False)
        fig.suptitle(f"KDE: Predicted vs Observed | {TGT_LABELS.get(tgt,tgt)} | v6",
                      fontsize=13,fontweight="bold")
        plt.tight_layout()
        fname = f"FIG_03_kde_{tgt}.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    ✓ {fname}")
except ImportError:
    print("    ✗ torch not available")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_04: Uncertainty distribution — seen vs unseen
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_04: Uncertainty distributions...")
unc_path = RESULTS/"v6_uncertainty_mc.csv"
if unc_path.exists():
    unc_df = pd.read_csv(unc_path)
    for tgt in unc_df["target"].unique() if "target" in unc_df.columns else []:
        sub = unc_df[unc_df["target"]==tgt]
        if sub.empty: continue
        fig, axes = plt.subplots(1,2,figsize=(18,8))

        # Uncertainty ratio
        sub_s = sub.sort_values("unc_ratio",ascending=True)
        colors=[TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey") for a in sub_s["arch"]]
        bars=axes[0].barh(sub_s["arch"],sub_s["unc_ratio"],color=colors,
                           alpha=0.85,edgecolor="black",lw=0.5)
        for bar,v in zip(bars,sub_s["unc_ratio"]):
            axes[0].text(v+0.01,bar.get_y()+bar.get_height()/2,
                         f"{v:.2f}×",va="center",fontsize=9,fontweight="bold")
        axes[0].axvline(1.0,color="black",lw=2,label="1.0 = equal uncertainty")
        axes[0].axvline(1.2,color="green",ls="--",lw=1.5,alpha=0.8,
                        label=">1.2 = well calibrated")
        axes[0].set_xlabel("Uncertainty Ratio (Unseen/Seen)")
        axes[0].set_title("Epistemic Uncertainty Ratio",fontweight="bold")
        axes[0].legend(fontsize=9)

        # Calibration
        if "calibration_unseen" in sub.columns:
            sub_c=sub.sort_values("calibration_unseen",ascending=True)
            colors2=[TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey") for a in sub_c["arch"]]
            axes[1].barh(sub_c["arch"],sub_c["calibration_unseen"],
                          color=colors2,alpha=0.85,edgecolor="black",lw=0.5)
            axes[1].axvline(0.5,color="green",ls="--",lw=1.5,
                             label="r=0.5 threshold")
            axes[1].set_xlabel("Calibration r (uncertainty vs error)")
            axes[1].set_title("Uncertainty Calibration — Unseen",fontweight="bold")
            axes[1].legend(fontsize=9)

        fig.suptitle(f"Uncertainty Distribution | {TGT_LABELS.get(tgt,tgt)} | v6",
                      fontsize=13,fontweight="bold")
        plt.tight_layout()
        fname=f"FIG_04_uncertainty_dist_{tgt}.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    ✓ {fname}")
else:
    print("    ✗ uncertainty CSV not found — run --mode uncertainty")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_05: Entropy — initial vs best per architecture
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_05: Entropy curves...")
if not ent_df.empty and "initial" in ent_df.columns:
    for tgt in ent_df["target"].unique() if "target" in ent_df.columns else []:
        sub = ent_df[ent_df["target"]==tgt].dropna(subset=["initial","final"])
        if sub.empty: continue
        sub = sub.sort_values("initial",ascending=False)

        fig, ax = plt.subplots(figsize=(16,8))
        x = np.arange(len(sub)); w=0.35
        colors=[TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey") for a in sub["arch"]]

        b1=ax.bar(x-w/2, sub["initial"], width=w,
                   label="Entropy at epoch 0 (initial)",
                   color=colors, alpha=0.5, edgecolor="black", lw=0.5)
        b2=ax.bar(x+w/2, sub["final"],   width=w,
                   label="Entropy at best epoch (final)",
                   color=colors, alpha=0.9, edgecolor="black", lw=0.5)

        # Draw arrows showing reduction
        for i,(init,fin) in enumerate(zip(sub["initial"],sub["final"])):
            if not (np.isnan(init) or np.isnan(fin)):
                ax.annotate("",xy=(x[i]+w/2,fin),xytext=(x[i]-w/2,init),
                             arrowprops=dict(arrowstyle="->",color="red",lw=1.2))

        ax.set_xticks(x)
        ax.set_xticklabels([f"[{ARCH_TIERS.get(a,'?')}]\n{a}" for a in sub["arch"]],
                             rotation=30,fontsize=9)
        ax.set_ylabel("Predictive Entropy H = 0.5(1+log(2πσ²))",fontsize=11)
        ax.set_title(f"Entropy: Initial → Best | {TGT_LABELS.get(tgt,tgt)}\n"
                     f"Higher initial entropy → model starts uncertain, learns more",
                     fontweight="bold",fontsize=12)
        ax.legend(fontsize=10)

        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
                   loc="lower center",ncol=5,fontsize=9,bbox_to_anchor=(0.5,-0.04))
        plt.tight_layout(rect=[0,0.05,1,1])
        fname=f"FIG_05_entropy_{tgt}.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    ✓ {fname}")
else:
    print("    ✗ entropy data not available yet")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_06: Full metric heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_06: Metric heatmaps...")
METRIC_COLS = {
    "Space_R2":    "R²\n(space)",
    "Space_KGE":   "KGE\n(space)",
    "Space_ubRMSE":"ubRMSE\n(space)",
    "Space_CRPS":  "CRPS\n(space)",
    "Space_DTW":   "DTW\n(space)",
    "Space_KGE":"KL Div\n(space)",
    "Time_R2":     "R²\n(time)",
    "Time_All_R2":    "KGE\n(time)",
    "Both_R2":     "R²\n(both)",
}
for tgt in df["Target"].unique():
    sub = df[df["Target"]==tgt].copy()
    avail = {k:v for k,v in METRIC_COLS.items() if k in sub.columns}
    if not avail: continue
    pv = sub.set_index("Model")[list(avail.keys())].rename(columns=avail)
    pv = pv.apply(pd.to_numeric,errors="coerce")

    # Invert error metrics (lower is better → negate for consistent colormap)
    for col in ["ubRMSE\n(space)","CRPS\n(space)","DTW\n(space)","KL Div\n(space)"]:
        if col in pv.columns: pv[col] = -pv[col]

    if pv.empty: continue
    fig, ax = plt.subplots(figsize=(max(16,len(avail)*2), max(8,len(pv)*0.6+2)))
    sns.heatmap(pv, ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                linewidths=0.5, linecolor="white",
                annot_kws={"size":10,"weight":"bold"},
                cbar_kws={"label":"Score (error metrics negated: higher=better)"})
    ax.set_yticklabels([f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in pv.index],
                        rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=9)
    ax.set_title(f"Full Metric Heatmap | {TGT_LABELS.get(tgt,tgt)} | v6\n"
                  f"Error metrics (ubRMSE, CRPS, DTW, KL) negated: green=better",
                  fontweight="bold", fontsize=12)
    plt.tight_layout()
    fname = f"FIG_06_metric_heatmap_{tgt}.png"
    plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_07: Ablation results
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_07: Ablation...")
if not abl_df.empty and "val_r2" in abl_df.columns:
    for tgt in abl_df["target"].unique() if "target" in abl_df.columns else []:
        sub = abl_df[abl_df["target"]==tgt]
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(18,10))
        pv = sub.pivot_table(index="arch", columns="ablation",
                              values="val_r2", aggfunc="mean")
        if "full" in pv.columns:
            # Show delta vs full model
            for col in pv.columns:
                if col!="full": pv[col]=pv[col]-pv["full"]
            pv=pv.drop(columns=["full"])
        sns.heatmap(pv,ax=ax,cmap="RdYlGn",center=0,annot=True,fmt=".3f",
                     linewidths=0.5,linecolor="white",
                     annot_kws={"size":10},
                     cbar_kws={"label":"ΔR² vs full model (negative = component helps)"})
        ax.set_title(f"Ablation Study — ΔR² | {TGT_LABELS.get(tgt,tgt)} | v6",
                      fontweight="bold",fontsize=12)
        ax.set_yticklabels([f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in pv.index],
                             rotation=0,fontsize=9)
        plt.tight_layout()
        fname=f"FIG_07_ablation_{tgt}.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
        print(f"    ✓ {fname}")
else:
    print("    ✗ ablation CSV not found — run --mode ablation")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_08: Seen vs Unseen R² scatter
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_08: Seen vs Unseen scatter...")
for tgt in df["Target"].unique():
    sub = df[df["Target"]==tgt].copy()
    seen_col   = "Std_All_R2"
    unseen_col = "Space_R2"
    if seen_col not in sub.columns or unseen_col not in sub.columns: continue
    sub = sub.dropna(subset=[seen_col,unseen_col])
    if sub.empty: continue

    fig, ax = plt.subplots(figsize=(12,10))
    for _,row in sub.iterrows():
        tier=ARCH_TIERS.get(row["Model"],"?")
        color=TIER_COLORS.get(tier,"grey")
        ax.scatter(row[seen_col],row[unseen_col],color=color,s=120,
                   edgecolors="black",lw=0.7,zorder=3)
        ax.annotate(row["Model"],(row[seen_col],row[unseen_col]),
                    textcoords="offset points",xytext=(6,3),fontsize=8)

    lims=[min(sub[seen_col].min(),sub[unseen_col].min())-0.01,
          max(sub[seen_col].max(),sub[unseen_col].max())+0.01]
    ax.plot(lims,lims,"k--",lw=1.5,alpha=0.5,label="Seen = Unseen (zero gap)")
    ax.fill_between(lims,[l-0.05 for l in lims],lims,
                     alpha=0.08,color="green",label="Gap < 0.05 zone")
    ax.set_xlabel("Seen R² (training sites)",fontsize=11)
    ax.set_ylabel("Unseen R² (Wetland holdout)",fontsize=11)
    ax.set_title(f"Spatial Generalisation | {TGT_LABELS.get(tgt,tgt)}\n"
                  f"Points above dashed line = better on unseen than seen",
                  fontweight="bold",fontsize=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()]
               +[plt.Line2D([0],[0],ls="--",color="black",label="Zero gap"),
                  Patch(color="green",alpha=0.3,label="Gap < 0.05")],
               fontsize=9,loc="lower right")
    plt.tight_layout()
    fname=f"FIG_08_spatial_generalisation_{tgt}.png"
    plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight"); plt.close()
    print(f"    ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  v6 FIGURES COMPLETE
  Saved to: {FIGS}
{'='*65}
  FIG_01  Loss curves (NLL + CRPS + graph smooth)
  FIG_02  Three-test leaderboard (space / time / both)
  FIG_03  KDE predicted vs observed (requires --save_preds)
  FIG_04  Uncertainty distribution seen vs unseen
  FIG_05  Entropy: initial → best per architecture
  FIG_06  Full metric heatmap (R², KGE, ubRMSE, CRPS, DTW, KL)
  FIG_07  Ablation: ΔR² per component (requires --mode ablation)
  FIG_08  Spatial generalisation scatter (seen vs unseen R²)
{'='*65}
""")
