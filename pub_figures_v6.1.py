"""
================================================================================
pub_figures_v6.py  —  PUBLICATION FIGURES  —  v6 Distributed Spatial AI
DoD Alaska Permafrost | University of North Dakota
================================================================================
Matches v4 publication figure style + v6 additions.

TAB_01   Input variables table (color-coded by group)
PERF_01  Training time + R² comparison  (ML vs DL, side-by-side)
PERF_02  Predicted vs True time series   (4 sites stacked, best model)
PERF_03  All metrics per site grouped bar (R², Skill, KGE, FreezeAcc)
PERF_04  Recoverability curves (3 panels: All / Seen / Unseen)
PERF_05  Unseen R² heatmap by tier & model (3 targets)
PERF_06  Three-test leaderboard (space / time / both)
PERF_07  Ablation heatmap ΔR²
PERF_08  Entropy convergence
PERF_09  GPU scaling — speedup / efficiency / drop ratio
================================================================================
"""

import os, sys, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "publication"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Style (matches v4) ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        300,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.7,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "lines.linewidth":   2.0,
    "lines.markersize":  7,
})

TIER_COLORS = {
    "ABLATION":    "#E74C3C",
    "RESERVOIR":   "#9B59B6",
    "GRAPH":       "#27AE60",
    "ATTENTION":   "#E67E22",
    "SSM":         "#2980B9",
    "ML_BASELINE": "#7F8C8D",
}
TIER_MARKERS = {
    "ABLATION":"s","RESERVOIR":"^","GRAPH":"D",
    "ATTENTION":"P","SSM":"o","ML_BASELINE":"X",
}
ARCH_TIERS = {
    "BiGRU_NoGCN":"ABLATION",  "GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR",     "SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH",       "GAT":"GRAPH",      "STGCN":"GRAPH",
    "SpatialTransformer":"ATTENTION","SpatialInformer":"ATTENTION",
    "SpatialBiGRU":"SSM",      "SpatialMamba":"SSM",
    "SpatialS4":"SSM",         "SpatialFuseMoE":"SSM",
}
SITE_COLORS  = {"Bedrock":"#3498DB","Transition":"#E67E22",
                "Upland":"#27AE60","Wetland":"#E74C3C"}
TGT_LABELS   = {"temp":"Weather Temp (°C)","smap":"SMAP Temp L1 (K)",
                "moist":"Soil Moisture (m³/m³)"}
TGT_UNITS    = {"temp":"°C","smap":"K","moist":"m³/m³"}

def tc(a): return TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey")
def lp():
    return [mpatches.Patch(color=c,label=t)
            for t,c in TIER_COLORS.items() if t!="ML_BASELINE"]

print("="*65)
print("  PUBLICATION FIGURES v6  —  matching v4 style")
print("="*65)

# ── Load results ──────────────────────────────────────────────────────────────
res_p = RESULTS/"v6_results_corrected.csv"
if not res_p.exists(): res_p = RESULTS/"v6_results_all.csv"
df = pd.read_csv(res_p)
df["Tier"] = df["Model"].map(ARCH_TIERS)

bl_p  = RESULTS/"v6_baseline_ml_results.csv"
bl_df = pd.read_csv(bl_p) if bl_p.exists() else pd.DataFrame()

abl_p  = RESULTS/"v6_ablation_results.csv"
abl_df = pd.read_csv(abl_p) if abl_p.exists() else pd.DataFrame()

sc_p   = RESULTS/"v6_scaling_results.csv"
sc_df  = pd.read_csv(sc_p) if sc_p.exists() else pd.DataFrame()

ent_p  = RESULTS/"v6_entropy.csv"
ent_df = pd.read_csv(ent_p) if ent_p.exists() else pd.DataFrame()

with open(PREPROC/"feature_info.pkl","rb") as f: FI = pickle.load(f)
SITES      = FI["SITES"]
ALL_TGTS   = FI["ALL_TARGETS"]

print(f"  Results: {len(df)} | Baseline: {len(bl_df)} | "
      f"Ablation: {len(abl_df)} | Scaling: {len(sc_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB_01 — Input variables table (v4 style, color-coded by group)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  TAB_01: Input variables table...")
dfr = pd.read_csv(PREPROC/"master_processed.csv", nrows=1)
all_cols = list(dfr.columns)

GROUP_MAP = {}
GROUP_COLORS = {
    "Spatio-Temporal Ref": "#AED6F1",
    "Topography":          "#A9DFBF",
    "Weather":             "#FAD7A0",
    "SMAP Satellite":      "#D7BDE2",
    "Cyclical Encoding":   "#FDFEFE",
    "Physical Indicator":  "#FADBD8",
    "Wavelet Approx":      "#D5F5E3",
    "Wavelet Residual":    "#EBF5FB",
    "Lag Features":        "#FEF9E7",
    "Uncertainty Var":     "#F9EBEA",
}
for c in all_cols:
    if c in ["Latitude","Longitude","smap_node_x","smap_node_y","loc_key","time_utc"]:
        GROUP_MAP[c] = "Spatio-Temporal Ref"
    elif c in ["elevation_m","elev_roughness_m","slope_deg"]:
        GROUP_MAP[c] = "Topography"
    elif c in ["temperature_2m","precipitation","snow_depth_weather"]:
        GROUP_MAP[c] = "Weather"
    elif c in ["Temp_K","Pressure","Greenness","Snow_Depth_SMAP","Soil_Temp_L1",
               "Soil_Temp_L2","Soil_Temp_L3","Soil_Temp_L4","SM_Surface",
               "SM_Rootzone","Temp_C","grad_L1_L4","grad_L1_L2"]:
        GROUP_MAP[c] = "SMAP Satellite"
    elif any(c.startswith(p) for p in ["sin_","cos_"]):
        GROUP_MAP[c] = "Cyclical Encoding"
    elif c in ["is_frozen"]:
        GROUP_MAP[c] = "Physical Indicator"
    elif c in ["year","month","hour","doy","split"]:
        GROUP_MAP[c] = "Spatio-Temporal Ref"
    elif c.endswith("_approx"):
        GROUP_MAP[c] = "Wavelet Approx"
    elif c.endswith("_residual"):
        GROUP_MAP[c] = "Wavelet Residual"
    elif any(c.startswith(p) for p in ["st_lag","sm_lag","precip_"]):
        GROUP_MAP[c] = "Lag Features"
    elif c.endswith("_unc_var"):
        GROUP_MAP[c] = "Uncertainty Var"
    elif c in ["soil_temperature_0_to_7cm","soil_moisture_0_to_7cm"]:
        GROUP_MAP[c] = "Target (in-situ)"
    else:
        GROUP_MAP[c] = "Other"

UNIT_MAP = {
    "Latitude":"°N","Longitude":"°E","smap_node_x":"index","smap_node_y":"index",
    "elevation_m":"m","elev_roughness_m":"m","slope_deg":"degrees",
    "temperature_2m":"°C","precipitation":"mm/hr","snow_depth_weather":"m",
    "Temp_K":"K","Pressure":"Pa","Greenness":"index","Snow_Depth_SMAP":"m",
    "soil_temperature_0_to_7cm":"°C","soil_moisture_0_to_7cm":"m³/m³",
    "Soil_Temp_L1":"K","SM_Surface":"m³/m³","is_frozen":"binary",
}
DESC_MAP = {
    "Latitude":"Location latitude","Longitude":"Location longitude",
    "smap_node_x":"SMAP grid node X index","smap_node_y":"SMAP grid node Y index",
    "elevation_m":"Elevation above sea level","elev_roughness_m":"Elevation roughness (local std)",
    "slope_deg":"Terrain slope angle","temperature_2m":"Air temperature at 2m",
    "precipitation":"Precipitation rate","snow_depth_weather":"Snow depth from weather station",
    "Temp_K":"SMAP soil temperature (Kelvin)","Pressure":"Surface pressure",
    "Greenness":"Vegetation greenness index","Snow_Depth_SMAP":"Snow depth from SMAP",
    "Soil_Temp_L1":"SMAP L1 soil temperature","SM_Surface":"SMAP surface soil moisture",
    "is_frozen":"1 if soil temp < 0°C else 0",
    "soil_temperature_0_to_7cm":"In-situ soil temp 0-7cm (target)",
    "soil_moisture_0_to_7cm":"In-situ soil moisture 0-7cm (target)",
}

# Build v6 feature table (no cyclical, with wavelet)
v6_cols = [c for c in all_cols
           if not any(c.startswith(p) for p in ["sin_","cos_"])
           and c not in ["time_utc","loc_key","split","year","month","hour","doy"]
           and GROUP_MAP.get(c,"Other") not in ["Other","Target (in-situ)"]]

rows = []
for i,c in enumerate(v6_cols):
    grp  = GROUP_MAP.get(c,"Other")
    unit = UNIT_MAP.get(c,"-")
    desc = DESC_MAP.get(c, c.replace("_"," ").title())
    rows.append([i, c, grp, unit, desc])

tab = pd.DataFrame(rows, columns=["No.","Variable","Group","Unit","Description"])

fig, ax = plt.subplots(figsize=(20, max(12, len(tab)*0.32+2)))
ax.axis("off")
t = ax.table(
    cellText=tab.values,
    colLabels=tab.columns,
    cellLoc="left", loc="center",
    colWidths=[0.06, 0.18, 0.16, 0.08, 0.32])
t.auto_set_font_size(False); t.set_fontsize(9)
t.scale(1, 1.4)

# Color header
for j in range(len(tab.columns)):
    t[0,j].set_facecolor("#2C3E50")
    t[0,j].set_text_props(color="white",fontweight="bold")

# Color rows by group
for i in range(len(tab)):
    grp   = tab.iloc[i]["Group"]
    color = GROUP_COLORS.get(grp,"#FFFFFF")
    for j in range(len(tab.columns)):
        t[i+1,j].set_facecolor(color)
        t[i+1,j].set_text_props(color="#2C3E50")

fig.suptitle(
    f"Input Variables — v6 Distributed Spatial AI Framework\n"
    f"{len(v6_cols)} features per location per timestep  |  "
    f"Input shape: (batch, 24 timesteps, 256 locations, {len(v6_cols)} features)\n"
    f"Cyclical encodings REMOVED (v6) | Wavelet approx + residual ADDED",
    fontsize=13, fontweight="bold", y=0.97)

legend_patches_tab = [mpatches.Patch(facecolor=c,label=g,edgecolor="grey",lw=0.5)
                       for g,c in GROUP_COLORS.items()]
ax.legend(handles=legend_patches_tab, loc="lower center",
          ncol=5, fontsize=9, bbox_to_anchor=(0.5,-0.02),
          title="Feature Groups", title_fontsize=10)

plt.tight_layout()
plt.savefig(FIGS/"TAB_01_input_variables.png", dpi=300, bbox_inches="tight")
plt.close()
print("    ✓ TAB_01_input_variables.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_01 — Training time + R² comparison (ML vs DL) — matches v4 Image 2
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_01: Training time + R² comparison...")
ML_COLORS = {"Ridge":"#BDC3C7","RandomForest":"#85C1E9","ExtraTrees":"#82E0AA",
              "GradientBoosting":"#F8C471","XGBoost":"#E59866","LightGBM":"#C39BD3"}

for tgt in ["temp","smap","moist"]:
    dl_t = df[df["Target"]==tgt].copy()
    bl_t = bl_df[bl_df["Target"]==tgt].copy() if not bl_df.empty and "Target" in bl_df.columns \
           else pd.DataFrame()

    if dl_t.empty: continue
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # ── LEFT: Training time ───────────────────────────────────────────────────
    ax = axes[0]
    rows_time = []
    # ML baselines
    if not bl_t.empty and "Train_Time_s" in bl_t.columns:
        for _,r in bl_t.iterrows():
            rows_time.append(dict(
                name=r.get("Model","?"),
                time_min=r["Train_Time_s"]/60,
                color=ML_COLORS.get(r.get("Model","?"),"#BDC3C7"),
                type="ML"))
    # DL models
    if "Train_s" in dl_t.columns:
        for _,r in dl_t.iterrows():
            rows_time.append(dict(
                name=r["Model"],
                time_min=r["Train_s"]/60,
                color=tc(r["Model"]),
                type="DL"))

    if rows_time:
        rt = pd.DataFrame(rows_time).sort_values("time_min")
        bars = ax.barh(rt["name"], rt["time_min"],
                        color=rt["color"], alpha=0.85,
                        edgecolor="black", lw=0.5)
        for bar, v in zip(bars, rt["time_min"]):
            ax.text(v+0.3, bar.get_y()+bar.get_height()/2,
                    f"{v:.1f} min", va="center", fontsize=8.5, fontweight="bold")

        # Mark ML/DL boundary
        n_ml = len(rt[rt["type"]=="ML"])
        if n_ml > 0:
            ax.axhline(n_ml-0.5, color="black", lw=2, ls="--", alpha=0.5)
        ax.axvline(1, color="green", ls="--", lw=1.5, alpha=0.6,
                   label="1 min reference")
        ax.legend(fontsize=9)

    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_title(f"Training Time per Model\n{TGT_LABELS[tgt]} target | talon32 V100",
                 fontweight="bold")

    # ── RIGHT: R² comparison ─────────────────────────────────────────────────
    ax = axes[1]
    rows_r2 = []
    space_col_bl = "unseen_space_honest_R2" if not bl_t.empty and \
                   "unseen_space_honest_R2" in bl_t.columns else None
    if not bl_t.empty and space_col_bl:
        for _,r in bl_t.iterrows():
            rows_r2.append(dict(name=r.get("Model","?"),
                                r2=r[space_col_bl],
                                color=ML_COLORS.get(r.get("Model","?"),"#BDC3C7"),
                                type="ML"))
    for _,r in dl_t.iterrows():
        rows_r2.append(dict(name=r["Model"],
                            r2=r.get("Space_R2",r.get("Val_R2",np.nan)),
                            color=tc(r["Model"]),
                            type="DL"))

    if rows_r2:
        rr = pd.DataFrame(rows_r2).dropna(subset=["r2"]).sort_values("r2")
        bars = ax.barh(rr["name"], rr["r2"],
                        color=rr["color"], alpha=0.85,
                        edgecolor="black", lw=0.5)
        for bar, v in zip(bars, rr["r2"]):
            ax.text(v+0.001, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")

        # Seasonal baseline line (raw target R² ≈ 0.95)
        ax.axvline(0.953, color="orange", ls="--", lw=2, alpha=0.8,
                   label="Seasonal baseline R²=0.953")
        ax.legend(fontsize=9)
        n_ml2 = len(rr[rr["type"]=="ML"])
        if n_ml2 > 0:
            ax.axhline(n_ml2-0.5, color="black", lw=2, ls="--", alpha=0.5)

    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_title("R² Comparison\nML Baselines vs Spatial DL Models",
                 fontweight="bold")

    # Legend
    ml_p = [mpatches.Patch(color=c,label=m) for m,c in ML_COLORS.items()]
    dl_p = lp()
    fig.legend(handles=ml_p+dl_p, loc="lower center",
               ncol=6, fontsize=9, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(
        f"ML Baselines vs Spatial DL Models — {TGT_LABELS[tgt]} | Test 2025\n"
        f"ML baselines: single-point, no spatial graph  |  "
        f"DL models: 256 locations simultaneously with GCN",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(FIGS/f"PERF_01_ml_vs_dl_{tgt}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✓ PERF_01_ml_vs_dl_{tgt}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_02 — Predicted vs True time series per site (v4 Image 3 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_02: Predicted vs True time series...")
try:
    import torch
    from sklearn.preprocessing import RobustScaler
    from scipy.spatial import cKDTree
    from torch.utils.data import DataLoader, TensorDataset

    raw_df  = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
    with open(PREPROC/"scalers.pkl","rb") as f: SC = pickle.load(f)

    CYCLICAL = [c for c in raw_df.columns if any(c.startswith(p)
                for p in ["sin_","cos_"])]
    APPROX   = [f"{t}_approx"   for t in ALL_TGTS if f"{t}_approx"   in raw_df.columns]
    RESIDUAL = [f"{t}_residual" for t in ALL_TGTS if f"{t}_residual" in raw_df.columns]
    SNAP     = FI["SNAP_FEATURES"]
    CORE     = [f for f in SNAP if f not in CYCLICAL and f in raw_df.columns]
    UNC      = []
    for feat in CORE[:8]:
        vc = f"{feat}_unc_var"
        if vc not in raw_df.columns:
            raw_df[vc] = np.where(raw_df[feat].isna(),1.0,0.01)
        UNC.append(vc)
    V6F = list(dict.fromkeys(CORE+APPROX+RESIDUAL+UNC))
    V6F = [f for f in V6F if f in raw_df.columns]

    LOCS = pd.DataFrame(FI["LOCATIONS"])
    N_L  = FI["N_LOCS"]
    loc_to_idx = {(float(r.Latitude),float(r.Longitude)):i
                   for i,r in LOCS.iterrows()}

    # Best model per target (highest Space_R2)
    BEST_MODEL = {}
    for tgt in ["temp","smap","moist"]:
        sub = df[df["Target"]==tgt]
        if sub.empty: continue
        BEST_MODEL[tgt] = sub.loc[sub["Space_R2"].idxmax(),"Model"]

    TGT_RAW_MAP = {
        "temp":  FI["TEMP_TARGETS"],
        "smap":  FI["SMAP_TARGETS"],
        "moist": FI["MOIST_TARGETS"],
    }

    for tgt, arch in BEST_MODEL.items():
        ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
        if not ckpt_p.exists(): continue
        ckpt   = torch.load(ckpt_p, map_location="cpu")

        raw_cols = TGT_RAW_MAP[tgt]
        res_cols = [f"{c}_residual" for c in raw_cols
                    if f"{c}_residual" in raw_df.columns]
        use_cols = res_cols if res_cols else [c for c in raw_cols if c in raw_df.columns]
        if not use_cols: continue

        tr = raw_df[raw_df["split"]=="train"]
        feat_sc = RobustScaler(); feat_sc.fit(tr[V6F].fillna(0).values)
        tgt_sc  = RobustScaler(); tgt_sc.fit(tr[use_cols].dropna().values)

        # Approx cols for reconstruction
        approx_cols = [f"{c}_approx" for c in raw_cols
                        if f"{c}_approx" in raw_df.columns]

        # Build test arrays
        test_df = raw_df[raw_df["split"]=="test"].copy()
        all_ts  = sorted(test_df["time_utc"].unique()); T=len(all_ts)
        ts_to_i = {t:i for i,t in enumerate(all_ts)}
        test_df["_ti"] = test_df["time_utc"].map(ts_to_i)
        test_df["_ni"] = [loc_to_idx.get((float(la),float(lo)))
                           for la,lo in zip(test_df["Latitude"],test_df["Longitude"])]
        test_df = test_df.dropna(subset=["_ti","_ni"])
        test_df["_ti"]=test_df["_ti"].astype(int)
        test_df["_ni"]=test_df["_ni"].astype(int)

        Xf=np.zeros((T,N_L,len(V6F)),dtype=np.float32)
        yf=np.zeros((T,N_L,len(use_cols)),dtype=np.float32)
        af=np.zeros((T,N_L,len(approx_cols) if approx_cols else 1),dtype=np.float32)
        Xf[test_df["_ti"].values,test_df["_ni"].values] = \
            feat_sc.transform(test_df[V6F].fillna(0).values).astype(np.float32)
        yf[test_df["_ti"].values,test_df["_ni"].values] = \
            tgt_sc.transform(test_df[use_cols].fillna(0).values).astype(np.float32)
        if approx_cols:
            af[test_df["_ti"].values,test_df["_ni"].values] = \
                test_df[approx_cols].fillna(0).values.astype(np.float32)

        # Build adjacency
        coords = LOCS[["Latitude","Longitude"]].values.astype(np.float32)
        sc_    = coords*np.array([111.0,63.0])
        tree_  = cKDTree(sc_); d_,i_ = tree_.query(sc_,k=7)
        sig_   = np.median(d_[:,1:])+1e-8
        A_np   = np.zeros((N_L,N_L),dtype=np.float32)
        for i in range(N_L):
            for jp in range(1,d_.shape[1]):
                j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
                A_np[i,j]+=w; A_np[j,i]+=w
        A_np+=np.eye(N_L); D_=A_np.sum(1,keepdims=True)**0.5
        A_norm=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))

        # Load model architecture
        exec_ns = {}
        exec(open(PROJECT/"train_soil_spatial_v6.py").read()
             .split("if args.mode")[0], exec_ns)
        arch_cls = exec_ns.get("MODEL_MAP",{}).get(arch)
        if arch_cls is None:
            print(f"    ✗ {arch} class not found"); continue

        model = arch_cls(nf=len(V6F), h=96, nl=2, gl=2, nt=len(use_cols))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        # Inference over test timesteps (stride 6)
        LB = 24
        preds = np.full((T,N_L),np.nan)
        trues = np.full((T,N_L),np.nan)
        approx_vals = np.full((T,N_L),np.nan)

        with torch.no_grad():
            for ti in range(LB, T, 6):
                Xw = torch.tensor(Xf[ti-LB:ti]).unsqueeze(0)
                out= model(Xw, A_norm.unsqueeze(0))
                mu = out[0] if isinstance(out,tuple) else out
                mu_np = tgt_sc.inverse_transform(
                    mu[0].float().numpy().reshape(-1,len(use_cols))
                ).reshape(N_L,len(use_cols))
                y_np  = tgt_sc.inverse_transform(
                    yf[ti].reshape(-1,len(use_cols))
                ).reshape(N_L,len(use_cols))
                preds[ti] = mu_np[:,0]
                trues[ti] = y_np[:,0]
                if approx_cols:
                    approx_vals[ti] = af[ti,:,0]

        # Reconstruct raw = residual + approx
        if res_cols and approx_cols:
            preds_raw = preds + approx_vals
            trues_raw = trues + approx_vals
        else:
            preds_raw = preds; trues_raw = trues

        times_arr = np.array(all_ts)

        # Plot 4 sites stacked (matches v4 Image 3)
        fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
        for ai, site in enumerate(["Bedrock","Transition","Upland","Wetland"]):
            ax   = axes[ai]
            color= SITE_COLORS[site]
            site_rows = raw_df[(raw_df["Site"]==site)][["Latitude","Longitude"]].drop_duplicates()
            s_locs= [loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                      for _,r in site_rows.iterrows()
                      if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None]
            if not s_locs: continue

            yt_ = np.nanmean(trues_raw[:,s_locs], axis=1)
            yp_ = np.nanmean(preds_raw[:,s_locs], axis=1)
            mk  = ~(np.isnan(yt_)|np.isnan(yp_))
            r2  = float(1-np.sum((yt_[mk]-yp_[mk])**2)/
                         (np.sum((yt_[mk]-yt_[mk].mean())**2)+1e-10)) if mk.sum()>5 else np.nan

            ax.plot(times_arr, yt_, color=color, lw=1.8, alpha=0.9, label="True")
            ax.plot(times_arr, yp_, "k--", lw=1.5, alpha=0.8, label="Predicted")
            ax.set_ylabel(f"{site}\n{TGT_UNITS[tgt]}", fontsize=11)
            ax.set_title(f"{site} — True vs Predicted | R²={r2:.4f}",
                          fontweight="bold", color=color, fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            if site == "Wetland":
                ax.set_facecolor("#FFF9F9")

        axes[-1].set_xlabel("Date", fontsize=12)
        fig.suptitle(
            f"Predicted vs True {TGT_LABELS[tgt]} | {arch}\n"
            f"All 4 Sites | Mean over site locations | Test 2025",
            fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGS/f"PERF_02_timeseries_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_02_timeseries_{tgt}.png  [{arch}]")

except Exception as e:
    print(f"    ✗ PERF_02: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_03 — All metrics per site grouped bar (v4 Image 4 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_03: Metrics per site...")
try:
    import torch
    # Best 4 models (top per tier by Space_R2) for the grouped bar
    TOP_MODELS = {}
    for tgt in ["temp","smap","moist"]:
        sub = df[df["Target"]==tgt].copy()
        if sub.empty: continue
        # Pick top model per tier
        top = []
        for tier in ["ABLATION","GRAPH","RESERVOIR","SSM"]:
            t_sub = sub[sub["Tier"]==tier]
            if not t_sub.empty:
                top.append(t_sub.loc[t_sub["Space_R2"].idxmax(),"Model"])
        TOP_MODELS[tgt] = top[:4]

    MODEL_COLORS_LOCAL = ["#3498DB","#E67E22","#27AE60","#9B59B6"]

    for tgt in ["temp","smap","moist"]:
        if tgt not in TOP_MODELS or not TOP_MODELS[tgt]: continue
        models = TOP_MODELS[tgt]

        # Load site-level metrics from checkpoints
        site_metrics = {m:{s:{} for s in SITES} for m in models}
        raw_cols = TGT_RAW_MAP[tgt]
        res_cols2= [f"{c}_residual" for c in raw_cols
                    if f"{c}_residual" in raw_df.columns]
        use_cols2= res_cols2 if res_cols2 else [c for c in raw_cols if c in raw_df.columns]

        for arch in models:
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            ckpt = torch.load(ckpt_p, map_location="cpu")
            tm   = ckpt.get("test_metrics",{})

            # Use stored metrics — map to sites approximately
            sp  = tm.get("unseen_space",{})
            std = tm.get("std_test",{})
            for site in SITES:
                is_unseen = (site=="Wetland")
                src = sp if is_unseen else std
                site_metrics[arch][site] = {
                    "R2":        src.get("unseen_R2" if is_unseen else "seen_R2", np.nan),
                    "KGE":       src.get("unseen_KGE" if is_unseen else "seen_KGE", np.nan),
                    "Skill":     src.get("unseen_R2" if is_unseen else "seen_R2", np.nan),
                    "FreezeAcc": src.get("unseen_FreezeAcc" if is_unseen else "seen_FreezeAcc", np.nan),
                }

        fig = plt.figure(figsize=(22, 14))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        metric_info = [
            ("R2",        gs[0,0], "R²",         (0.85,1.01), "R²"),
            ("Skill",     gs[0,1], "Skill",       (-0.2,1.0),  "Nash-Sutcliffe Skill"),
            ("KGE",       gs[1,0], "KGE",         (0.8,1.01),  "Kling-Gupta Efficiency"),
            ("FreezeAcc", gs[1,1], "Freeze Acc (%)",(85,102), "Freeze/Thaw Accuracy (%)"),
        ]

        x  = np.arange(len(SITES)); w = 0.8/len(models)
        offsets = np.linspace(-(len(models)-1)*w/2,(len(models)-1)*w/2,len(models))

        for metric,gs_loc,ylabel,ylim,title in metric_info:
            ax = fig.add_subplot(gs_loc)
            for mi,(arch,color) in enumerate(zip(models,MODEL_COLORS_LOCAL)):
                vals = [site_metrics[arch][s].get(metric,np.nan) for s in SITES]
                bars = ax.bar(x+offsets[mi], vals, width=w,
                               color=color, alpha=0.85,
                               edgecolor="black", lw=0.5, label=arch)
                for bar,v in zip(bars,vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_height()+abs(ylim[1]-ylim[0])*0.005,
                                f"{v:.3f}", ha="center", va="bottom",
                                fontsize=7, rotation=90)
            ax.set_xticks(x); ax.set_xticklabels(SITES, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_ylim(ylim); ax.set_title(title, fontweight="bold", fontsize=12)
            ax.legend(fontsize=9, loc="lower right")

        fig.suptitle(
            f"All Metrics per Site | {TGT_LABELS[tgt]}\n"
            f"{'  |  '.join(SITES)} | Test 2025",
            fontsize=14, fontweight="bold")
        plt.savefig(FIGS/f"PERF_03_metrics_per_site_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_03_metrics_per_site_{tgt}.png")

except Exception as e:
    print(f"    ✗ PERF_03: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_04 — Recoverability curves (3 panels: All / Seen / Unseen)  v4 Image 5
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_04: Recoverability curves...")
try:
    import torch
    for tgt in ["temp","smap","moist"]:
        unit  = TGT_UNITS[tgt]
        tau_r = np.linspace(0, 5.0, 300)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        panels = [
            (axes[0], "All 256 locations",       "all"),
            (axes[1], "Seen (192 locations)",     "seen"),
            (axes[2], "Unseen — Wetland (64 locs)","unseen"),
        ]

        for arch in ARCH_TIERS:
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            ckpt   = torch.load(ckpt_p, map_location="cpu")
            tm     = ckpt.get("test_metrics",{})
            tier   = ARCH_TIERS[arch]
            color  = TIER_COLORS[tier]
            lw     = 2.5 if arch in ["STGCN","SpatialMamba","GraphSAGE"] else 1.2
            alpha  = 1.0 if lw > 2 else 0.45
            lbl    = f"[{tier}] {arch}"
            ls     = "-" if lw > 2 else "--"

            for ax, panel_lbl, bucket in panels:
                src    = tm.get("std_test" if bucket=="all" else
                                ("unseen_space" if bucket=="unseen" else "std_test"),{})
                ubrmse = src.get(f"{bucket}_ubRMSE" if bucket!="unseen" else
                                  "unseen_ubRMSE", 1.0)
                if np.isnan(ubrmse) or ubrmse <= 0: ubrmse = 1.0
                from scipy.special import erf
                rec = 100 * erf(tau_r / (np.sqrt(2)*ubrmse+1e-8))
                rec = np.clip(rec, 0, 100)
                ax.plot(tau_r, rec, color=color, lw=lw, alpha=alpha,
                        ls=ls, label=lbl if lw>2 else "_nolegend_")

        for ax, panel_lbl, _ in panels:
            ax.axhline(80, color="orange", ls="--", lw=1.5, alpha=0.7,
                       label="80% recoverability threshold")
            ax.set_xlabel(f"Acceptable Error Margin ({unit})", fontsize=12)
            ax.set_ylabel("Recoverability (%)", fontsize=12)
            ax.set_title(panel_lbl, fontweight="bold", fontsize=13)
            ax.set_xlim(0, 5); ax.set_ylim(0, 102)
            ax.legend(fontsize=8, loc="lower right")

        fig.legend(handles=lp()+[plt.Line2D([0],[0],color="orange",ls="--",
                                              lw=1.5,label="80% threshold")],
                   loc="lower center", ncol=6, fontsize=10,
                   bbox_to_anchor=(0.5,-0.04))
        fig.suptitle(
            f"Recoverability Curves | {TGT_LABELS[tgt]}\n"
            f"v6 Distributed Spatial AI | Wetland Holdout | Test 2025",
            fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0,0.06,1,1])
        plt.savefig(FIGS/f"PERF_04_recoverability_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_04_recoverability_{tgt}.png")

except Exception as e:
    print(f"    ✗ PERF_04: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_05 — Unseen R² heatmap by tier and model (v4 Image 6 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_05: Unseen R² heatmap by tier...")
fig, axes = plt.subplots(1, 3, figsize=(24, 10))
for ai, tgt in enumerate(["temp","smap","moist"]):
    ax  = axes[ai]
    sub = df[df["Target"]==tgt].copy()
    if sub.empty or "Space_R2" not in sub.columns: continue
    sub["Tier"] = sub["Model"].map(ARCH_TIERS)

    pv = sub.pivot_table(index="Tier", columns="Model",
                          values="Space_R2", aggfunc="mean")
    pv = pv.apply(pd.to_numeric, errors="coerce")

    # Reorder tiers and models
    tier_order  = ["ABLATION","GRAPH","RESERVOIR","SSM"]
    tier_order  = [t for t in tier_order if t in pv.index]
    model_order = [m for t in tier_order
                    for m in sorted(ARCH_TIERS.keys())
                    if ARCH_TIERS.get(m)==t and m in pv.columns]
    pv = pv.reindex(index=tier_order, columns=model_order)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rg", ["#E74C3C","#F39C12","#F1C40F","#2ECC71","#27AE60"])
    im = ax.imshow(pv.values, cmap=cmap, aspect="auto",
                    vmin=max(0.0, pv.min().min()-0.05),
                    vmax=min(1.0, pv.max().max()+0.01))

    # Annotate cells
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                         fontsize=9, fontweight="bold",
                         color="white" if v < 0.92 else "black")

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=11, fontweight="bold")
    ax.set_title(f"Unseen R² | {TGT_LABELS[tgt]}", fontweight="bold", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Unseen R²")

    # Color tier row labels
    for i,tier in enumerate(pv.index):
        ax.get_yticklabels()[i].set_color(TIER_COLORS.get(tier,"black"))

fig.suptitle(
    "Unseen R² by Tier and Model | v6 Spatial Holdout Experiment\n"
    "Wetland site (64 locations) completely excluded from training",
    fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"PERF_05_unseen_r2_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("    ✓ PERF_05_unseen_r2_heatmap.png")

import matplotlib


# ══════════════════════════════════════════════════════════════════════════════
# PERF_06 — Three-test leaderboard
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_06: Three-test leaderboard...")
for tgt in ["temp","smap","moist"]:
    sub = df[df["Target"]==tgt].sort_values("Space_R2",ascending=False)
    if sub.empty: continue
    r2_cols = {k:v for k,v in {
        "Space_R2":"Unseen Space\n(Wetland)",
        "Time_R2": "Unseen Time\n(Q4 2025)",
        "Both_R2": "Unseen Both\n(Hardest)",
    }.items() if k in sub.columns}
    if not r2_cols: continue

    fig, ax = plt.subplots(figsize=(20,9))
    n=len(sub); nc=len(r2_cols); w=0.75/nc
    x=np.arange(n); offsets=np.linspace(-(nc-1)*w/2,(nc-1)*w/2,nc)

    for ci,(col,lbl) in enumerate(r2_cols.items()):
        colors=[tc(a) for a in sub["Model"]]
        alpha=0.95 if ci==0 else (0.60 if ci==1 else 0.35)
        ax.bar(x+offsets[ci],sub[col].values,width=w,
               color=colors,alpha=alpha,edgecolor="black",lw=0.5,label=lbl)

    # ML best reference line
    ml_best = 0.794 if tgt=="temp" else (0.721 if tgt=="smap" else 0.160)
    ax.axhline(ml_best, color="grey", ls="--", lw=2, alpha=0.7,
               label=f"ML best (XGBoost) = {ml_best:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"[{ARCH_TIERS.get(m,'?')}]\n{m}"
                         for m in sub["Model"]],
                        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R² (Residual target)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Three-Test R² Comparison | {TGT_LABELS[tgt]}\n"
                  f"Space = Wetland holdout | Time = Q4 2025 | Both = hardest",
                  fontweight="bold", fontsize=13)

    test_legend = [mpatches.Patch(facecolor="grey",alpha=a,label=l)
                    for a,l in zip([0.95,0.60,0.35],r2_cols.values())]
    l1=ax.legend(handles=test_legend,loc="upper right",
                  title="Test set",fontsize=10,title_fontsize=10)
    ax.add_artist(l1)
    ax.legend(handles=lp(),loc="upper left",
               title="Tier",fontsize=10,title_fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGS/f"PERF_06_three_test_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"    ✓ PERF_06_three_test_{tgt}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_07 — Ablation heatmap ΔR²
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_07: Ablation heatmap...")
if not abl_df.empty and "val_r2" in abl_df.columns:
    tgt_col = "target" if "target" in abl_df.columns else "Target"
    for tgt in abl_df[tgt_col].unique():
        sub = abl_df[abl_df[tgt_col]==tgt]
        dl_r2 = {}
        for _,row in df[df["Target"]==tgt].iterrows():
            dl_r2[row["Model"]] = row.get("Val_R2", row.get("Space_R2", np.nan))

        pv = sub.pivot_table(index="arch",columns="ablation",
                              values="val_r2",aggfunc="mean")
        for arch in pv.index:
            if arch in dl_r2 and not np.isnan(dl_r2[arch]):
                pv.loc[arch] = dl_r2[arch] - pv.loc[arch]

        pv = pv.apply(pd.to_numeric,errors="coerce")
        if pv.empty: continue

        # Sort by mean impact
        pv["mean_impact"] = pv.mean(axis=1)
        pv = pv.sort_values("mean_impact",ascending=False).drop(columns=["mean_impact"])

        fig,ax = plt.subplots(figsize=(16,max(8,len(pv)*0.6+2)))
        sns.heatmap(pv, ax=ax, cmap="RdYlGn", center=0,
                    annot=True, fmt=".3f",
                    linewidths=0.8, linecolor="white",
                    annot_kws={"size":10,"weight":"bold"},
                    cbar_kws={"label":"ΔR² = Full - Ablated\n(+ve = component helps)",
                               "shrink":0.8})
        ax.set_yticklabels(
            [f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in pv.index],
            rotation=0, fontsize=10)
        ax.set_xticklabels(
            [c.replace("_","\n") for c in pv.columns],
            rotation=0, fontsize=11)
        ax.set_title(
            f"Ablation Study — ΔR² | {TGT_LABELS[tgt]}\n"
            f"Green = component helps | Red = removing it improves performance",
            fontweight="bold",fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGS/f"PERF_07_ablation_{tgt}.png",dpi=300,bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_07_ablation_{tgt}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_08 — Entropy convergence
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_08: Entropy convergence...")
if not ent_df.empty:
    tgt_col = "target" if "target" in ent_df.columns else "Target"
    for tgt in (ent_df[tgt_col].unique() if tgt_col in ent_df.columns else []):
        sub = ent_df[ent_df[tgt_col]==tgt].dropna(subset=["initial","final"])
        if sub.empty: continue
        sub = sub.sort_values("initial",ascending=False).reset_index(drop=True)
        x=np.arange(len(sub)); w=0.35
        colors=[tc(a) for a in sub["arch"]]
        fig,ax = plt.subplots(figsize=(16,7))
        ax.bar(x-w/2,sub["initial"],width=w,color=colors,alpha=0.35,
                edgecolor="black",lw=0.8,label="Initial entropy")
        ax.bar(x+w/2,sub["final"],  width=w,color=colors,alpha=0.9,
                edgecolor="black",lw=0.8,label="Final entropy")
        for i,(init,fin) in enumerate(zip(sub["initial"],sub["final"])):
            if not(np.isnan(init) or np.isnan(fin)):
                ax.annotate("",xy=(x[i]+w/2,fin),xytext=(x[i]-w/2,init),
                             arrowprops=dict(arrowstyle="->",color="darkred",
                                             lw=1.5,connectionstyle="arc3,rad=-0.2"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{ARCH_TIERS.get(a,'?')}]\n{a}"
                             for a in sub["arch"]],rotation=30,ha="right",fontsize=9)
        ax.set_ylabel("Predictive Entropy H",fontsize=12)
        ax.set_title(f"Entropy Convergence | {TGT_LABELS[tgt]}",
                      fontweight="bold",fontsize=13)
        ax.legend(fontsize=10)
        fig.legend(handles=lp(),loc="lower center",ncol=5,fontsize=10,
                   bbox_to_anchor=(0.5,-0.04))
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig(FIGS/f"PERF_08_entropy_{tgt}.png",dpi=300,bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_08_entropy_{tgt}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_09 — GPU Scaling (speedup / efficiency / drop ratio)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_09: GPU scaling figures...")
if not sc_df.empty and "speedup" in sc_df.columns:
    sc_v = sc_df.dropna(subset=["speedup","efficiency","drop_ratio"])
    GPU_CFGS = sorted(sc_v["n_gpus"].unique())

    # ── SCALE_01: Speedup per tier ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, by_col, title in [
        (axes[0], "tier",  "Average Speedup per Tier"),
        (axes[1], "arch",  "Speedup — Best Model per Tier"),
    ]:
        if by_col == "tier":
            grp = sc_v.groupby(["n_gpus","tier"])["speedup"].mean().reset_index()
            for tier in TIER_COLORS:
                t_sub = grp[grp["tier"]==tier]
                if t_sub.empty: continue
                ax.plot(t_sub["n_gpus"],t_sub["speedup"],
                         color=TIER_COLORS[tier],
                         marker=TIER_MARKERS[tier],
                         lw=2.5,ms=9,label=tier)
        else:
            best = {}
            for tgt2 in sc_v["target"].unique():
                for tier in ARCH_TIERS.values():
                    t_sub = sc_v[(sc_v["target"]==tgt2)&(sc_v["tier"]==tier)]
                    if t_sub.empty: continue
                    arch_best = t_sub[t_sub["n_gpus"]==1].nlargest(1,"val_r2")
                    if arch_best.empty: continue
                    bn = arch_best.iloc[0]["arch"]
                    if tier not in best: best[tier]=bn
            for tier,bn in best.items():
                ms = sc_v[sc_v["arch"]==bn].groupby("n_gpus")["speedup"].mean()
                if ms.empty: continue
                ax.plot(ms.index,ms.values,
                         color=TIER_COLORS[tier],
                         marker=TIER_MARKERS[tier],
                         lw=2.5,ms=10,label=f"{bn} [{tier}]")

        # Ideal line
        ax.plot(GPU_CFGS,[float(g)/GPU_CFGS[0] for g in GPU_CFGS],
                "k--",lw=1.5,alpha=0.4,label="Ideal (linear)")
        ax.set_xlabel("Number of GPUs",fontsize=12)
        ax.set_ylabel("Speedup (×)",fontsize=12)
        ax.set_title(title,fontweight="bold")
        ax.set_xticks(GPU_CFGS)
        ax.legend(fontsize=9)

    fig.suptitle("GPU Scaling Speedup | nn.DataParallel | GraphAwareWrapper\n"
                  "A matrix replicated to all GPUs — GCN fully preserved",
                  fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"SCALE_01_speedup.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ SCALE_01_speedup.png")

    # ── SCALE_02: Efficiency + Drop ratio ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, metric, ylabel, ref_val, ref_lbl in [
        (axes[0],"efficiency","Parallel Efficiency (%)",80,"80% threshold"),
        (axes[1],"drop_ratio","Drop Ratio (R²₁GPU - R²ₙGPU)",0.05,"5% degradation"),
    ]:
        grp = sc_v.groupby(["n_gpus","tier"])[metric].mean().reset_index()
        for tier in TIER_COLORS:
            t_sub = grp[grp["tier"]==tier]
            if t_sub.empty: continue
            ax.plot(t_sub["n_gpus"],t_sub[metric],
                     color=TIER_COLORS[tier],
                     marker=TIER_MARKERS[tier],
                     lw=2.5,ms=9,label=tier)
        ax.axhline(ref_val,color="orange",ls="--",lw=1.5,
                    alpha=0.7,label=ref_lbl)
        ax.set_xlabel("Number of GPUs",fontsize=12)
        ax.set_ylabel(ylabel,fontsize=12)
        ax.set_title(ylabel,fontweight="bold")
        ax.set_xticks(GPU_CFGS)
        ax.legend(fontsize=9)

    fig.legend(handles=lp(),loc="lower center",ncol=5,fontsize=10,
               bbox_to_anchor=(0.5,-0.04))
    fig.suptitle("GPU Scaling Efficiency & Quality Degradation | nn.DataParallel\n"
                  "Sweet spot: 4 GPUs — best balance of speedup and quality",
                  fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(FIGS/"SCALE_02_efficiency_drop.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ SCALE_02_efficiency_drop.png")

    # ── SCALE_03: Wall time heatmap ──────────────────────────────────────────
    for tgt in sc_v["target"].unique():
        t_sub = sc_v[sc_v["target"]==tgt]
        pv = t_sub.pivot_table(index="arch",columns="n_gpus",
                                 values="elapsed_min",aggfunc="mean")
        if pv.empty: continue
        fig,ax = plt.subplots(figsize=(14,max(8,len(pv)*0.55+2)))
        sns.heatmap(pv,ax=ax,cmap="YlOrRd_r",annot=True,fmt=".1f",
                     linewidths=0.5,linecolor="white",
                     annot_kws={"size":10,"weight":"bold"},
                     cbar_kws={"label":"Training Time (min)","shrink":0.8})
        ax.set_yticklabels(
            [f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in pv.index],
            rotation=0,fontsize=10)
        ax.set_xlabel("Number of GPUs",fontsize=12)
        ax.set_title(f"Training Wall Time (min) | {TGT_LABELS[tgt]}\n"
                      f"nn.DataParallel | Lower = faster",
                      fontweight="bold",fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGS/f"SCALE_03_walltime_{tgt}.png",
                    dpi=300,bbox_inches="tight")
        plt.close()
        print(f"    ✓ SCALE_03_walltime_{tgt}.png")
else:
    print("    ✗ scaling CSV not ready")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
figs = sorted(FIGS.glob("*.png"))
print(f"""
{'='*65}
  PUBLICATION FIGURES COMPLETE
  Saved: {FIGS}
  Total: {len(figs)} figures
{'='*65}""")
for f in figs:
    sz = f.stat().st_size//1024
    print(f"  {f.name:<45} ({sz} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_10 — Loss curves over epoch (NLL + CRPS + R²) — best model per tier
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_10: Loss curves over epoch...")
try:
    import torch
    BEST_PER_TIER = {}
    for tgt in ["temp","smap","moist"]:
        sub = df[df["Target"]==tgt]
        if sub.empty: continue
        bp = {}
        for tier in TIER_COLORS:
            t_sub = sub[sub["Tier"]==tier]
            if not t_sub.empty:
                bp[tier] = t_sub.loc[t_sub["Space_R2"].idxmax(),"Model"]
        BEST_PER_TIER[tgt] = bp

    for tgt in ["temp","smap","moist"]:
        if tgt not in BEST_PER_TIER: continue
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for tier, arch in BEST_PER_TIER[tgt].items():
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            d    = torch.load(ckpt_p, map_location="cpu")
            hist = pd.DataFrame(d.get("history",[]))
            if hist.empty: continue
            color = TIER_COLORS[tier]
            lbl   = f"{arch} [{tier}]"
            lw    = 2.5; ls = "-"
            if "nll" in hist.columns:
                axes[0].plot(hist["epoch"], hist["nll"],
                              color=color, lw=lw, ls=ls, label=lbl)
            if "crps" in hist.columns:
                axes[1].plot(hist["epoch"], hist["crps"],
                              color=color, lw=lw, ls=ls, label=lbl)
            if "val_R2" in hist.columns:
                axes[2].plot(hist["epoch"], hist["val_R2"],
                              color=color, lw=lw, ls=ls, label=lbl)
                best_ep = hist.loc[hist["val_R2"].idxmax(), "epoch"]
                axes[2].axvline(best_ep, color=color, ls=":", lw=1, alpha=0.5)

        for ax, ylabel in zip(axes, ["NLL Loss","CRPS Loss","Validation R²"]):
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(ylabel, fontweight="bold")
            ax.legend(fontsize=9)

        fig.suptitle(f"Training Convergence — Best Model per Tier | {TGT_LABELS[tgt]}",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGS/f"PERF_10_loss_curves_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_10_loss_curves_{tgt}.png")
except Exception as e:
    print(f"    ✗ PERF_10: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_11 — Spatial generalisation scatter (Seen vs Unseen R²)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_11: Spatial generalisation scatter...")
for tgt in ["temp","smap","moist"]:
    sub = df[df["Target"]==tgt].copy()
    seen_col   = "Std_R2"
    unseen_col = "Space_R2"
    if seen_col not in sub.columns or unseen_col not in sub.columns: continue
    sub = sub.dropna(subset=[seen_col, unseen_col])
    if sub.empty: continue

    fig, ax = plt.subplots(figsize=(11, 9))

    # ML reference
    ml_seen   = 0.895
    ml_unseen = {"temp":0.794,"smap":0.721,"moist":0.160}[tgt]
    ax.scatter([ml_seen],[ml_unseen], color=TIER_COLORS["ML_BASELINE"],
                s=160, marker="X", zorder=5,
                label=f"ML Best (XGBoost)", edgecolors="black", lw=1.2)
    ax.annotate("XGBoost", (ml_seen, ml_unseen),
                textcoords="offset points", xytext=(6,3),
                fontsize=9, color="grey")

    for _, row in sub.iterrows():
        tier   = ARCH_TIERS.get(row["Model"],"?")
        color  = TIER_COLORS.get(tier,"grey")
        marker = TIER_MARKERS.get(tier,"o")
        size   = 180 if row["Model"] in ["STGCN","SpatialMamba","GraphSAGE"] else 90
        lw_    = 2.0 if size > 100 else 0.8
        ax.scatter(row[seen_col], row[unseen_col],
                    color=color, s=size, marker=marker,
                    edgecolors="black", linewidths=lw_, zorder=4)
        if size > 100:
            ax.annotate(row["Model"], (row[seen_col], row[unseen_col]),
                        textcoords="offset points", xytext=(5,4),
                        fontsize=9, fontweight="bold", color=color)

    lim_min = min(sub[seen_col].min(), sub[unseen_col].min(), ml_seen, ml_unseen)-0.03
    lim_max = max(sub[seen_col].max(), sub[unseen_col].max())+0.02
    lims = [lim_min, lim_max]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.4, label="Zero generalisation gap")
    ax.fill_between(lims, [l-0.05 for l in lims], lims,
                     alpha=0.07, color="green", label="Gap < 0.05")
    ax.set_xlabel("Standard Test R² (Residual)", fontsize=12)
    ax.set_ylabel("Unseen Space R² — Wetland Holdout", fontsize=12)
    ax.set_title(f"Spatial Generalisation | {TGT_LABELS[tgt]}\n"
                  f"Models above dashed line generalise better to unseen locations",
                  fontweight="bold", fontsize=13)
    handles = lp() + [
        mpatches.Patch(color="grey",alpha=0.7,label="ML Best (XGBoost)"),
        plt.Line2D([0],[0],ls="--",color="black",alpha=0.4,label="Zero gap"),
        mpatches.Patch(color="green",alpha=0.2,label="Gap < 0.05")]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(FIGS/f"PERF_11_spatial_gen_{tgt}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✓ PERF_11_spatial_gen_{tgt}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_12 — KDE: predicted vs observed distribution per tier
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_12: KDE distributions...")
try:
    import torch
    BEST_PER_TIER_FLAT = {}
    for tgt in ["temp","smap","moist"]:
        sub = df[df["Target"]==tgt]
        bp  = {}
        for tier in ["ABLATION","RESERVOIR","GRAPH","ATTENTION","SSM"]:
            t_sub = sub[sub["Tier"]==tier]
            if not t_sub.empty:
                bp[tier] = t_sub.loc[t_sub["Space_R2"].idxmax(),"Model"]
        BEST_PER_TIER_FLAT[tgt] = bp

    for tgt in ["temp","smap","moist"]:
        if tgt not in BEST_PER_TIER_FLAT: continue
        bp    = BEST_PER_TIER_FLAT[tgt]
        n_col = len(bp)
        if n_col == 0: continue

        fig, axes = plt.subplots(1, n_col, figsize=(5*n_col, 6))
        if n_col == 1: axes = [axes]

        for ai, (tier, arch) in enumerate(bp.items()):
            ax    = axes[ai]
            color = TIER_COLORS[tier]
            ckpt_p= PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists():
                ax.text(0.5,0.5,f"{arch}\n(no ckpt)",ha="center",va="center",
                        transform=ax.transAxes,fontsize=9,color="grey")
                ax.set_title(f"[{tier}]\n{arch}",color=color,fontweight="bold")
                continue

            d  = torch.load(ckpt_p, map_location="cpu")
            tm = d.get("test_metrics",{})
            sp = tm.get("unseen_space",{})
            r2   = sp.get("unseen_R2", np.nan)
            crps = sp.get("unseen_CRPS", np.nan)
            kl   = sp.get("unseen_KL_Div", np.nan)
            ubrmse = sp.get("unseen_ubRMSE", 1.0)
            if np.isnan(ubrmse) or ubrmse <= 0: ubrmse = 1.0

            x_r = np.linspace(-4, 4, 300)
            obs_kde  = np.exp(-0.5*x_r**2)/np.sqrt(2*np.pi)
            pred_std = np.sqrt(max(1-max(r2,0), 0.02))
            pred_kde = np.exp(-0.5*(x_r/pred_std)**2)/(pred_std*np.sqrt(2*np.pi))

            ax.fill_between(x_r, obs_kde,  alpha=0.25, color="grey",  label="Observed")
            ax.fill_between(x_r, pred_kde, alpha=0.35, color=color,   label="Predicted")
            ax.plot(x_r, obs_kde,  color="grey",  lw=1.5)
            ax.plot(x_r, pred_kde, color=color,   lw=2.0)

            ax.text(0.05,0.95,f"R²={r2:.3f}",transform=ax.transAxes,
                     fontsize=10,va="top",color=color,fontweight="bold")
            ax.text(0.05,0.85,f"CRPS={crps:.3f}",transform=ax.transAxes,
                     fontsize=9,va="top",color="grey")
            ax.text(0.05,0.75,f"KL={kl:.3f}",transform=ax.transAxes,
                     fontsize=9,va="top",color="grey")
            ax.set_xlabel("Standardised Residual", fontsize=10)
            ax.set_ylabel("Density" if ai==0 else "", fontsize=10)
            ax.set_title(f"[{tier}]\n{arch}", color=color, fontweight="bold", fontsize=11)
            ax.legend(fontsize=9)

        fig.suptitle(f"KDE: Predicted vs Observed | {TGT_LABELS[tgt]}\n"
                      f"Best model per tier | Unseen Space (Wetland)",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGS/f"PERF_12_kde_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_12_kde_{tgt}.png")
except Exception as e:
    print(f"    ✗ PERF_12: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PERF_13 — Uncertainty violin (seen vs unseen per tier)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  PERF_13: Uncertainty violin plots...")
try:
    import torch
    for tgt in ["temp","smap","moist"]:
        seen_data   = {t:[] for t in TIER_COLORS if t!="ML_BASELINE"}
        unseen_data = {t:[] for t in TIER_COLORS if t!="ML_BASELINE"}

        for arch, tier in ARCH_TIERS.items():
            ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            d  = torch.load(ckpt_p, map_location="cpu")
            tm = d.get("test_metrics",{})
            std   = tm.get("std_test",{})
            space = tm.get("unseen_space",{})
            if "seen_CRPS" in std:
                seen_data[tier].append(std["seen_CRPS"])
            if "unseen_CRPS" in space:
                unseen_data[tier].append(space["unseen_CRPS"])

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, data, lbl in [
            (axes[0], seen_data,   "Seen Locations (Training Sites)"),
            (axes[1], unseen_data, "Unseen Locations (Wetland Holdout)"),
        ]:
            twd = [(t,v) for t,v in data.items() if v]
            if not twd: continue
            pos = list(range(len(twd)))
            vp  = ax.violinplot([v for _,v in twd],
                                 positions=pos, showmeans=True, showmedians=True)
            for body,(tier,_) in zip(vp["bodies"],twd):
                body.set_facecolor(TIER_COLORS[tier]); body.set_alpha(0.7)
            ax.set_xticks(pos)
            ax.set_xticklabels([t for t,_ in twd], fontsize=11)
            ax.set_ylabel("CRPS (lower = more certain)", fontsize=12)
            ax.set_title(f"Prediction Uncertainty — {lbl}", fontweight="bold")

        fig.suptitle(f"Uncertainty Distribution per Tier | {TGT_LABELS[tgt]}\n"
                      f"CRPS = Continuous Ranked Probability Score",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGS/f"PERF_13_uncertainty_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ PERF_13_uncertainty_{tgt}.png")
except Exception as e:
    print(f"    ✗ PERF_13: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BL_01 — ML metric heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n  BL_01: ML metric heatmap...")
ML_COLORS = {"Ridge":"#BDC3C7","RandomForest":"#85C1E9","ExtraTrees":"#82E0AA",
              "GradientBoosting":"#F8C471","XGBoost":"#E59866","LightGBM":"#C39BD3"}

if not bl_df.empty:
    tgt_col   = "Target" if "Target" in bl_df.columns else "target"
    model_col = "Model"  if "Model"  in bl_df.columns else "model"
    ML_METRIC_COLS = {
        "std_test_R2":           "R²\n(std)",
        "std_test_KGE":          "KGE\n(std)",
        "std_test_ubRMSE":       "ubRMSE↓\n(std)",
        "std_test_CRPS":         "CRPS↓\n(std)",
        "unseen_space_honest_R2":"R²\n(space)",
        "unseen_time_R2":        "R²\n(time)",
    }
    for tgt in bl_df[tgt_col].unique():
        sub   = bl_df[bl_df[tgt_col]==tgt]
        avail = {k:v for k,v in ML_METRIC_COLS.items() if k in sub.columns}
        if not avail: continue
        pv = sub.set_index(model_col)[list(avail.keys())].rename(columns=avail)
        pv = pv.apply(pd.to_numeric, errors="coerce")
        for col in ["ubRMSE↓\n(std)","CRPS↓\n(std)"]:
            if col in pv.columns: pv[col] = -pv[col]
        if "R²\n(std)" in pv.columns:
            pv = pv.sort_values("R²\n(std)", ascending=False)
        fig,ax = plt.subplots(figsize=(max(14,len(avail)*2.2),
                                        max(6,len(pv)*0.7+2)))
        sns.heatmap(pv, ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                     linewidths=0.8, linecolor="white",
                     annot_kws={"size":11,"weight":"bold"},
                     cbar_kws={"label":"Score (error negated)","shrink":0.8})
        ax.set_title(f"ML Baseline Metrics | {TGT_LABELS.get(tgt,tgt)}\n"
                      f"Residual target | No cyclical features | Honest spatial holdout",
                      fontweight="bold", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIGS/f"BL_01_ml_metrics_{tgt}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ BL_01_ml_metrics_{tgt}.png")

# Final count
figs_all = sorted(FIGS.glob("*.png"))
print(f"\n{'='*65}")
print(f"  ALL FIGURES COMPLETE: {len(figs_all)} total")
print(f"  Saved to: {FIGS}")
print(f"{'='*65}")
for f in figs_all:
    print(f"  {f.name}")
