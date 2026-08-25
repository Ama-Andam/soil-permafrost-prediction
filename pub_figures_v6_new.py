"""
================================================================================
pub_figures_v6_new.py  —  MANUSCRIPT FIGURES  —  v6 Distributed Spatial AI
DoD Alaska Permafrost | University of North Dakota
================================================================================
MANUSCRIPT QUALITY — Comparative, all sites, all 256 locations, continuous time

FIG_01  Input variables table (color-coded by group)
FIG_02  Comparative scatter: Predicted vs Observed — all 4 sites in ONE figure
         Best model per tier, colored by site, 2×3 grid
FIG_03  Comparative temporal: Actual vs Predicted — all 4 sites stacked
         Multiple models overlaid, full test period
FIG_04  Spatial field: 256 locations predicted vs observed (map-style)
         Best model, snapshot at key timesteps
FIG_05  ML vs DL: Training time + R² side-by-side (all 3 targets)
FIG_06  Three-test leaderboard: Space/Time/Both R² grouped bar
FIG_07  Recoverability curves: All/Seen/Unseen 3 panels
FIG_08  Metric heatmap: R², KGE, ubRMSE, CRPS per model
FIG_09  Unseen R² heatmap by tier and model (3 targets in one figure)
FIG_10  Ablation: ΔR² per component per model
FIG_11  GPU scaling: Speedup/Efficiency/Drop ratio
FIG_12  Entropy convergence: initial → best
================================================================================
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "manuscript"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
    "legend.fontsize": 10, "legend.framealpha": 0.9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})

# PI instruction: remove ablation models from all figures

ABLATION_MODELS = ["BiGRU_NoGCN", "GCN_NoTemporal"]



TIER_COLORS = {
    "STANDALONE": "#E74C3C",  # BiGRU_NoGCN, GCN_NoTemporal
    "RESERVOIR":  "#9B59B6",
    "GRAPH":      "#27AE60",
    "ATTENTION":  "#E67E22",
    "SSM":        "#2980B9",
    "ML":         "#7F8C8D",
}
SITE_COLORS  = {"Bedrock":"#3498DB","Transition":"#E67E22",
                "Upland":"#27AE60","Wetland":"#E74C3C"}
SITE_MARKERS = {"Bedrock":"o","Transition":"s","Upland":"^","Wetland":"D"}
ARCH_TIERS   = {
    "BiGRU_NoGCN":"STANDALONE", "GCN_NoTemporal":"STANDALONE",
    "DeepESN":"RESERVOIR",      "SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH",        "GAT":"GRAPH",      "STGCN":"GRAPH",
    "SpatialTransformer":"ATTENTION","SpatialInformer":"ATTENTION",
    "SpatialBiGRU":"SSM",       "SpatialMamba":"SSM",
    "SpatialS4":"SSM",          "SpatialFuseMoE":"SSM",
}
TGT_LABELS = {"temp":"Weather Temp (°C)","smap":"SMAP Temp L1 (K)",
               "moist":"Soil Moisture (m³/m³)"}
TGT_UNITS  = {"temp":"°C","smap":"K","moist":"m³/m³"}
SITES      = ["Bedrock","Transition","Upland","Wetland"]

def tc(a): return TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey")
def lp():
    return [mpatches.Patch(color=c,label=t) for t,c in TIER_COLORS.items()]

print("="*65)
print("  MANUSCRIPT FIGURES v6  —  All sites | All 256 locations")
print("="*65)

# ── Load all data ─────────────────────────────────────────────────────────────
with open(PREPROC/"feature_info.pkl","rb") as f: FI=pickle.load(f)
LOCS      = pd.DataFrame(FI["LOCATIONS"])
N_LOCS    = FI["N_LOCS"]
ALL_TGTS  = FI["ALL_TARGETS"]
TGT_RAW   = {"temp":FI["TEMP_TARGETS"],"smap":FI["SMAP_TARGETS"],
              "moist":FI["MOIST_TARGETS"]}

raw_df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
loc_to_idx = {(float(r.Latitude),float(r.Longitude)):i
               for i,r in LOCS.iterrows()}

res_df  = pd.read_csv(RESULTS/"v6_results_corrected.csv")
res_df  = res_df[~res_df["Model"].isin(ABLATION_MODELS)].copy()  # PI: exclude ablation
bl_df   = pd.read_csv(RESULTS/"v6_baseline_ml_results.csv") if (RESULTS/"v6_baseline_ml_results.csv").exists() else pd.DataFrame()
abl_df  = pd.read_csv(RESULTS/"v6_ablation_results.csv")   if (RESULTS/"v6_ablation_results.csv").exists()   else pd.DataFrame()
sc_df   = pd.read_csv(RESULTS/"v6_scaling_results.csv")     if (RESULTS/"v6_scaling_results.csv").exists()     else pd.DataFrame()
ent_df  = pd.read_csv(RESULTS/"v6_entropy.csv")             if (RESULTS/"v6_entropy.csv").exists()             else pd.DataFrame()
res_df["Tier"] = res_df["Model"].map(ARCH_TIERS)

print(f"  Results: {len(res_df)} | Baseline: {len(bl_df)} | Ablation: {len(abl_df)}")
print(f"  Scaling: {len(sc_df)} | Entropy: {len(ent_df)}")

# ── Build features ────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
CYCLICAL  = [c for c in raw_df.columns if any(c.startswith(p) for p in ["sin_","cos_"])]
SNAP      = FI["SNAP_FEATURES"]
CORE      = [f for f in SNAP if f not in CYCLICAL and f in raw_df.columns]
APPROX    = [f"{t}_approx"   for t in ALL_TGTS if f"{t}_approx"   in raw_df.columns]
RESIDUAL  = [f"{t}_residual" for t in ALL_TGTS if f"{t}_residual" in raw_df.columns]
UNC       = []
for feat in CORE[:8]:
    vc = f"{feat}_unc_var"
    if vc not in raw_df.columns: raw_df[vc]=np.where(raw_df[feat].isna(),1.0,0.01)
    UNC.append(vc)
V6F = list(dict.fromkeys(CORE+APPROX+RESIDUAL+UNC))
V6F = [f for f in V6F if f in raw_df.columns]

tr = raw_df[raw_df["split"]=="train"]
feat_sc = RobustScaler(); feat_sc.fit(tr[V6F].fillna(0).values)

# ── Build graph ───────────────────────────────────────────────────────────────
coords = LOCS[["Latitude","Longitude"]].values.astype(np.float32)
sc_ = coords*np.array([111.0,63.0])
tree_ = cKDTree(sc_); d_,i_ = tree_.query(sc_,k=7)
sig_  = np.median(d_[:,1:])+1e-8
A_np  = np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1,d_.shape[1]):
        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
        A_np[i,j]+=w; A_np[j,i]+=w
A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5
A_norm_np=(A_np/(D_*D_.T+1e-8)).astype(np.float32)

import torch
A_norm = torch.tensor(A_norm_np)
print(f"  Features: {len(V6F)} | Graph: {N_LOCS} nodes")

# ── Helper: load model and run inference across full test period ──────────────
def run_inference(arch, tgt, stride=4, max_t=None):
    """
    Run inference for a model on full test set.
    Returns: times, preds (T x N_LOCS), trues (T x N_LOCS), site_locs dict
    All 256 locations, continuous time.
    """
    raw_cols = TGT_RAW[tgt]
    res_cols = [f"{c}_residual" for c in raw_cols if f"{c}_residual" in raw_df.columns]
    use_cols = res_cols if res_cols else [c for c in raw_cols if c in raw_df.columns]
    approx_c = [f"{c}_approx" for c in raw_cols if f"{c}_approx" in raw_df.columns]
    if not use_cols: return None,None,None,None

    tgt_sc = RobustScaler()
    tgt_sc.fit(tr[use_cols].dropna().values)

    test_df = raw_df[raw_df["split"]=="test"].copy()
    all_ts  = sorted(test_df["time_utc"].unique())
    if max_t: all_ts = all_ts[:max_t]
    T = len(all_ts)
    ts_to_i = {t:i for i,t in enumerate(all_ts)}
    test_df["_ti"] = test_df["time_utc"].map(ts_to_i)
    test_df["_ni"] = [loc_to_idx.get((float(la),float(lo)))
                       for la,lo in zip(test_df["Latitude"],test_df["Longitude"])]
    test_df = test_df.dropna(subset=["_ti","_ni"])
    test_df["_ti"]=test_df["_ti"].astype(int)
    test_df["_ni"]=test_df["_ni"].astype(int)
    # Filter to valid timesteps
    test_df = test_df[test_df["_ti"]<T]

    Xf=np.zeros((T,N_LOCS,len(V6F)),dtype=np.float32)
    yf=np.zeros((T,N_LOCS,len(use_cols)),dtype=np.float32)
    af=np.zeros((T,N_LOCS,max(len(approx_c),1)),dtype=np.float32)
    Xf[test_df["_ti"].values,test_df["_ni"].values]=\
        feat_sc.transform(test_df[V6F].fillna(0).values).astype(np.float32)
    yf[test_df["_ti"].values,test_df["_ni"].values]=\
        tgt_sc.transform(test_df[use_cols].fillna(0).values).astype(np.float32)
    if approx_c:
        af[test_df["_ti"].values,test_df["_ni"].values]=\
            test_df[approx_c].fillna(0).values.astype(np.float32)

    ckpt_p = PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
    if not ckpt_p.exists(): return None,None,None,None
    ckpt = torch.load(ckpt_p, map_location="cpu")

    # Load model class
    exec_ns = {}
    try:
        exec(open(PROJECT/"train_soil_spatial_v6.py").read()
             .split("if args.mode")[0], exec_ns)
        arch_cls = exec_ns.get("MODEL_MAP",{}).get(arch)
    except: arch_cls=None
    if arch_cls is None: return None,None,None,None

    # Infer hidden dim from checkpoint
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict",{}))
    _h = 96
    for k,v in sd.items():
        if "hd.mu.0.weight" in k: _h=int(v.shape[1]); break
    _hcfg = ckpt.get("hcfg",{})
    _nl = int(_hcfg.get("n_layers",2))
    _gl = int(_hcfg.get("gcn_layers",2))
    _nf = ckpt.get("n_feats",len(V6F))

    try:
        model = arch_cls(nf=_nf, h=_h, nl=_nl, gl=_gl, nt=len(use_cols))
        model.load_state_dict(sd, strict=True)
    except Exception:
        try:
            model = arch_cls(nf=_nf, h=_h, nl=_nl, gl=_gl, nt=len(use_cols))
            model.load_state_dict(sd, strict=False)
        except: return None,None,None,None
    model.eval()

    preds = np.full((T,N_LOCS),np.nan)
    trues = np.full((T,N_LOCS),np.nan)
    LB = 24

    with torch.no_grad():
        for ti in range(LB, T, stride):
            Xw = torch.tensor(Xf[ti-LB:ti]).unsqueeze(0)
            out = model(Xw, A_norm.unsqueeze(0))
            mu  = out[0] if isinstance(out,tuple) else out
            mu_np = tgt_sc.inverse_transform(
                mu[0].float().numpy().reshape(-1,len(use_cols))).reshape(N_LOCS,len(use_cols))
            y_np  = tgt_sc.inverse_transform(
                yf[ti].reshape(-1,len(use_cols))).reshape(N_LOCS,len(use_cols))
            av = af[ti,:,0] if approx_c else np.zeros(N_LOCS)
            preds[ti] = mu_np[:,0]  # PI: residual only — no approx reconstruction
            trues[ti] = y_np[:,0]  # PI: residual only — no approx reconstruction

    # Site location indices
    site_locs = {}
    for site in SITES:
        sr = raw_df[raw_df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
        site_locs[site] = [loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                            for _,r in sr.iterrows()
                            if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None]

    return np.array(all_ts), preds, trues, site_locs


# ══════════════════════════════════════════════════════════════════════════════
# FIG_01 — Input variables table
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_01: Input variables table...")
GROUP_COLORS = {
    "Spatio-Temporal":  "#AED6F1",
    "Topography":       "#A9DFBF",
    "Weather":          "#FAD7A0",
    "SMAP Satellite":   "#D7BDE2",
    "Physical":         "#FADBD8",
    "Wavelet Approx":   "#D5F5E3",
    "Wavelet Residual": "#EBF5FB",
    "Lag Features":     "#FEF9E7",
    "Uncertainty":      "#F9EBEA",
}
GROUP_MAP = {}
for c in raw_df.columns:
    if c in ["Latitude","Longitude","smap_node_x","smap_node_y","year","month","hour","doy"]:
        GROUP_MAP[c]="Spatio-Temporal"
    elif c in ["elevation_m","elev_roughness_m","slope_deg"]:
        GROUP_MAP[c]="Topography"
    elif c in ["temperature_2m","precipitation","snow_depth_weather"]:
        GROUP_MAP[c]="Weather"
    elif c in ["Temp_K","Temp_C","Pressure","Greenness","Snow_Depth_SMAP",
               "Soil_Temp_L1","Soil_Temp_L2","Soil_Temp_L3","Soil_Temp_L4",
               "SM_Surface","SM_Rootzone","grad_L1_L4","grad_L1_L2","is_frozen"]:
        GROUP_MAP[c]="SMAP Satellite"
    elif c in ["is_frozen"]: GROUP_MAP[c]="Physical"
    elif c.endswith("_approx"):   GROUP_MAP[c]="Wavelet Approx"
    elif c.endswith("_residual"): GROUP_MAP[c]="Wavelet Residual"
    elif any(c.startswith(p) for p in ["st_lag","sm_lag","precip_"]):
        GROUP_MAP[c]="Lag Features"
    elif c.endswith("_unc_var"):  GROUP_MAP[c]="Uncertainty"

UNIT_MAP = {"Latitude":"°N","Longitude":"°E","smap_node_x":"idx","smap_node_y":"idx",
            "elevation_m":"m","elev_roughness_m":"m","slope_deg":"°",
            "temperature_2m":"°C","precipitation":"mm/hr","snow_depth_weather":"m",
            "Temp_K":"K","Temp_C":"°C","Pressure":"Pa","Greenness":"idx",
            "Snow_Depth_SMAP":"m","Soil_Temp_L1":"K","SM_Surface":"m³/m³","is_frozen":"0/1"}

v6_cols=[c for c in V6F if c not in ["loc_key","split","time_utc"]]
rows=[]
for i,c in enumerate(v6_cols):
    rows.append([i,c,GROUP_MAP.get(c,"Other"),UNIT_MAP.get(c,"-"),
                  c.replace("_"," ").title()])
tab=pd.DataFrame(rows,columns=["No.","Variable","Group","Unit","Description"])

fig,ax=plt.subplots(figsize=(20,max(12,len(tab)*0.30+2)))
ax.axis("off")
t=ax.table(cellText=tab.values,colLabels=tab.columns,
            cellLoc="left",loc="center",
            colWidths=[0.05,0.18,0.14,0.07,0.36])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,1.35)
for j in range(len(tab.columns)):
    t[0,j].set_facecolor("#2C3E50"); t[0,j].set_text_props(color="white",fontweight="bold")
for i in range(len(tab)):
    clr=GROUP_COLORS.get(tab.iloc[i]["Group"],"#FFFFFF")
    for j in range(len(tab.columns)):
        t[i+1,j].set_facecolor(clr)
fig.suptitle(f"Input Variables — v6 Distributed Spatial AI Framework\n"
              f"{len(v6_cols)} features | Shape: (batch, 24 timesteps, 256 locations, {len(v6_cols)} features)\n"
              f"Cyclical encodings REMOVED | Wavelet approx + residual ADDED",
              fontsize=13,fontweight="bold",y=0.97)
lp_tab=[mpatches.Patch(facecolor=c,label=g,edgecolor="grey",lw=0.5)
         for g,c in GROUP_COLORS.items()]
ax.legend(handles=lp_tab,loc="lower center",ncol=5,fontsize=9,
           bbox_to_anchor=(0.5,-0.02),title="Feature Groups",title_fontsize=10)
plt.tight_layout()
plt.savefig(FIGS/"FIG_01_input_variables.png",dpi=300,bbox_inches="tight")
plt.close()
print("    ✓ FIG_01_input_variables.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_02 — Comparative scatter: Predicted vs Observed — ALL 4 sites in ONE
# Best model per tier, colored by site, 2×3 grid (3 targets × best model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_02: Comparative scatter (all sites, all targets)...")
try:
    # Pick best overall model (highest mean Space_R2 across targets)
    mean_r2 = res_df.groupby("Model")["Space_R2"].mean().sort_values(ascending=False)
    BEST_OVERALL = mean_r2.index[0]
    print(f"    Best overall model: {BEST_OVERALL}")

    fig = plt.figure(figsize=(24, 18))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    for ti, tgt in enumerate(["temp","smap","moist"]):
        # Top row: best overall model
        ax = fig.add_subplot(gs[0,ti])
        times, preds, trues, site_locs = run_inference(BEST_OVERALL, tgt, stride=4)
        if preds is None:
            ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
            continue

        all_yt=[]; all_yp=[]; all_colors=[]; all_markers=[]
        r2_per_site = {}
        for site in SITES:
            sl = site_locs.get(site,[])
            if not sl: continue
            # All 256 location points — not just means
            for loc in sl:
                yt_loc = trues[:,loc]; yp_loc = preds[:,loc]
                mk = ~(np.isnan(yt_loc)|np.isnan(yp_loc))
                all_yt.extend(yt_loc[mk].tolist())
                all_yp.extend(yp_loc[mk].tolist())
                all_colors.extend([SITE_COLORS[site]]*mk.sum())

            # Site-level R²
            yt_s = np.nanmean(trues[:,sl],axis=1)
            yp_s = np.nanmean(preds[:,sl],axis=1)
            mk_s = ~(np.isnan(yt_s)|np.isnan(yp_s))
            if mk_s.sum()>5:
                r2_per_site[site] = float(1-np.sum((yt_s[mk_s]-yp_s[mk_s])**2)/
                                           (np.sum((yt_s[mk_s]-yt_s[mk_s].mean())**2)+1e-10))

        if all_yt:
            ax.scatter(all_yt,all_yp,c=all_colors,alpha=0.15,s=4,
                        edgecolors="none",rasterized=True)
            lim=[min(all_yt)-0.5,max(all_yt)+0.5]
            ax.plot(lim,lim,"k-",lw=2,alpha=0.8,label="1:1")
            ax.set_xlim(lim); ax.set_ylim(lim)

        # R² annotation per site
        y_txt=0.97
        for site in SITES:
            if site in r2_per_site:
                ax.text(0.03,y_txt,f"{site}: R²={r2_per_site[site]:.3f}",
                         transform=ax.transAxes,fontsize=8,va="top",
                         color=SITE_COLORS[site],fontweight="bold")
                y_txt-=0.07

        ax.set_xlabel(f"Observed ({TGT_UNITS[tgt]})",fontsize=11)
        ax.set_ylabel(f"Predicted ({TGT_UNITS[tgt]})",fontsize=11)
        ax.set_title(f"{TGT_LABELS[tgt]}\n[{ARCH_TIERS.get(BEST_OVERALL,'?')}] {BEST_OVERALL}",
                      fontweight="bold",fontsize=12)

        # Bottom row: STGCN (best GRAPH model)
        ax2 = fig.add_subplot(gs[1,ti])
        arch2 = "STGCN"
        times2,preds2,trues2,sl2 = run_inference(arch2,tgt,stride=4)
        if preds2 is None: continue

        all_yt2=[]; all_yp2=[]; all_colors2=[]
        r2_per_site2={}
        for site in SITES:
            sl_s=sl2.get(site,[])
            if not sl_s: continue
            for loc in sl_s:
                yt_=trues2[:,loc]; yp_=preds2[:,loc]
                mk=~(np.isnan(yt_)|np.isnan(yp_))
                all_yt2.extend(yt_[mk].tolist())
                all_yp2.extend(yp_[mk].tolist())
                all_colors2.extend([SITE_COLORS[site]]*mk.sum())
            yt_s2=np.nanmean(trues2[:,sl_s],axis=1); yp_s2=np.nanmean(preds2[:,sl_s],axis=1)
            mk_s2=~(np.isnan(yt_s2)|np.isnan(yp_s2))
            if mk_s2.sum()>5:
                r2_per_site2[site]=float(1-np.sum((yt_s2[mk_s2]-yp_s2[mk_s2])**2)/
                                          (np.sum((yt_s2[mk_s2]-yt_s2[mk_s2].mean())**2)+1e-10))

        if all_yt2:
            ax2.scatter(all_yt2,all_yp2,c=all_colors2,alpha=0.15,s=4,
                         edgecolors="none",rasterized=True)
            lim2=[min(all_yt2)-0.5,max(all_yt2)+0.5]
            ax2.plot(lim2,lim2,"k-",lw=2,alpha=0.8)
            ax2.set_xlim(lim2); ax2.set_ylim(lim2)

        y_txt2=0.97
        for site in SITES:
            if site in r2_per_site2:
                ax2.text(0.03,y_txt2,f"{site}: R²={r2_per_site2[site]:.3f}",
                          transform=ax2.transAxes,fontsize=8,va="top",
                          color=SITE_COLORS[site],fontweight="bold")
                y_txt2-=0.07

        ax2.set_xlabel(f"Observed ({TGT_UNITS[tgt]})",fontsize=11)
        ax2.set_ylabel(f"Predicted ({TGT_UNITS[tgt]})",fontsize=11)
        ax2.set_title(f"{TGT_LABELS[tgt]}\n[GRAPH] {arch2}",
                       fontweight="bold",fontsize=12)

    # Site legend
    site_handles=[mpatches.Patch(color=SITE_COLORS[s],label=f"{s} ({'Unseen' if s=='Wetland' else 'Seen'})")
                   for s in SITES]
    fig.legend(handles=site_handles,loc="lower center",ncol=4,fontsize=11,
               bbox_to_anchor=(0.5,-0.02),
               title="Ecological Sites (each point = one location × one timestep)",
               title_fontsize=11)
    fig.suptitle("Predicted vs Observed — Comparative Scatter | All 4 Sites | All 256 Locations\n"
                  "v6 Distributed Spatial AI | Test 2025 | Wetland = unseen spatial holdout",
                  fontsize=14,fontweight="bold")
    plt.savefig(FIGS/"FIG_02_comparative_scatter.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ FIG_02_comparative_scatter.png")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"    ✗ FIG_02: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_03 — Comparative temporal: All 4 sites stacked, multiple models
# Full test period, site means, uncertainty bands
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_03: Comparative temporal (all sites stacked)...")
try:
    # Models to compare: best per tier
    COMPARE_MODELS = {}
    for tgt in ["temp","smap","moist"]:
        sub = res_df[res_df["Target"]==tgt]
        cm  = []
        seen_tiers = set()
        for _,row in sub.sort_values("Space_R2",ascending=False).iterrows():
            tier = ARCH_TIERS.get(row["Model"],"?")
            if tier not in seen_tiers:
                cm.append(row["Model"]); seen_tiers.add(tier)
            if len(cm)>=4: break
        COMPARE_MODELS[tgt] = cm

    for tgt in ["temp","smap","moist"]:
        models_to_plot = COMPARE_MODELS.get(tgt,[])
        if not models_to_plot: continue

        fig,axes = plt.subplots(4,1,figsize=(22,20),sharex=True)
        fig.subplots_adjust(hspace=0.06)

        # Get first model to establish time axis and true values
        times0,preds0,trues0,site_locs0 = run_inference(models_to_plot[0],tgt,stride=3)
        if times0 is None: continue

        for ai,site in enumerate(SITES):
            ax   = axes[ai]
            sl   = site_locs0.get(site,[])
            if not sl: continue

            # True values (same for all models)
            yt   = np.nanmean(trues0[:,sl],axis=1)

            # Shaded range across all 256 locs
            yt_lo= np.nanpercentile(trues0[:,sl],10,axis=1)
            yt_hi= np.nanpercentile(trues0[:,sl],90,axis=1)
            ax.fill_between(times0,yt_lo,yt_hi,
                             alpha=0.12,color=SITE_COLORS[site],
                             label="Obs P10-P90 range")
            ax.plot(times0,yt,color=SITE_COLORS[site],lw=2.5,
                     alpha=0.95,label="Observed (mean)",zorder=5)

            # Predicted — multiple models
            r2_txt = []
            for mi,arch in enumerate(models_to_plot):
                if mi==0:
                    p_times,ppreds,_,psl = times0,preds0,trues0,site_locs0
                else:
                    p_times,ppreds,_,psl = run_inference(arch,tgt,stride=3)
                if ppreds is None: continue
                psl2 = psl.get(site,[])
                if not psl2: continue
                yp = np.nanmean(ppreds[:,psl2],axis=1)
                mk = ~(np.isnan(yt)|np.isnan(yp))
                r2 = float(1-np.sum((yt[mk]-yp[mk])**2)/
                            (np.sum((yt[mk]-yt[mk].mean())**2)+1e-10)) if mk.sum()>5 else np.nan
                tier = ARCH_TIERS.get(arch,"?")
                ax.plot(p_times,yp,
                         color=TIER_COLORS.get(tier,"grey"),
                         lw=1.8,ls="--",alpha=0.85,
                         label=f"[{tier}] {arch} R²={r2:.3f}")
                r2_txt.append(f"{arch}={r2:.3f}")

            # Freeze line
            ax.axhline(0,color="grey",lw=0.8,ls=":",alpha=0.5)

            unseen=" ◄ UNSEEN" if site=="Wetland" else ""
            ax.set_ylabel(TGT_UNITS[tgt],fontsize=11)
            ax.set_title(f"{site}{unseen}",fontweight="bold",
                          color=SITE_COLORS[site],fontsize=13,loc="left")
            ax.legend(loc="upper right",fontsize=8,framealpha=0.9,ncol=2)
            if site=="Wetland":
                ax.set_facecolor("#FFFAFA")
                ax.spines["left"].set_color(SITE_COLORS[site])
                ax.spines["left"].set_linewidth(2.5)

        import matplotlib.dates as mdates
        axes[-1].set_xlabel("Date",fontsize=12)
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)

        fig.suptitle(f"Actual vs Predicted {TGT_LABELS[tgt]} | All 4 Ecological Sites\n"
                      f"v6 Distributed Spatial AI | 256 locations | Test 2025 | "
                      f"Shaded band = P10-P90 across all locations",
                      fontsize=14,fontweight="bold",y=1.005)
        plt.savefig(FIGS/f"FIG_03_temporal_{tgt}.png",dpi=300,bbox_inches="tight")
        plt.close()
        print(f"    ✓ FIG_03_temporal_{tgt}.png")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"    ✗ FIG_03: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_04 — ML vs DL comparison (v4 Image 2 style, all 3 targets in one figure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_04: ML vs DL comparison...")
ML_COLORS={"Ridge":"#BDC3C7","RandomForest":"#85C1E9","ExtraTrees":"#82E0AA",
             "GradientBoosting":"#F8C471","XGBoost":"#E59866","LightGBM":"#C39BD3"}

fig,axes=plt.subplots(2,3,figsize=(26,14))
for ti,tgt in enumerate(["temp","smap","moist"]):
    dl_t = res_df[res_df["Target"]==tgt]
    bl_t = bl_df[bl_df["Target"]==tgt] if not bl_df.empty and "Target" in bl_df.columns else pd.DataFrame()

    # Top: Training time
    ax=axes[0,ti]
    rows_t=[]
    if not bl_t.empty and "Train_Time_s" in bl_t.columns:
        for _,r in bl_t.iterrows():
            rows_t.append(dict(name=r.get("Model","?"),
                                t=r["Train_Time_s"]/60,
                                c=ML_COLORS.get(r.get("Model","?"),"grey"),tp="ML"))
    if "Train_s" in dl_t.columns:
        for _,r in dl_t.iterrows():
            rows_t.append(dict(name=r["Model"],t=r["Train_s"]/60,c=tc(r["Model"]),tp="DL"))
    if rows_t:
        rt=pd.DataFrame(rows_t).sort_values("t")
        colors_t=rt["c"].tolist()
        bars=ax.barh(rt["name"],rt["t"],color=colors_t,alpha=0.85,
                      edgecolor="black",lw=0.5)
        for bar,v in zip(bars,rt["t"]):
            ax.text(v+0.1,bar.get_y()+bar.get_height()/2,
                    f"{v:.1f}m",va="center",fontsize=7.5,fontweight="bold")
        ax.axvline(1,color="green",ls="--",lw=1.5,alpha=0.6,label="1 min")
        n_ml=len(rt[rt["tp"]=="ML"])
        if n_ml>0: ax.axhline(n_ml-0.5,color="black",lw=1.5,ls="--",alpha=0.5)
        ax.legend(fontsize=8)
    ax.set_xlabel("Training Time (min)",fontsize=10)
    ax.set_title(f"{TGT_LABELS[tgt]}\nTraining Time",fontweight="bold",fontsize=11)

    # Bottom: R²
    ax2=axes[1,ti]
    rows_r=[]
    sp_bl="unseen_space_honest_R2"
    if not bl_t.empty and sp_bl in bl_t.columns:
        for _,r in bl_t.iterrows():
            rows_r.append(dict(name=r.get("Model","?"),
                                r2=r[sp_bl],
                                c=ML_COLORS.get(r.get("Model","?"),"grey"),tp="ML"))
    for _,r in dl_t.iterrows():
        rows_r.append(dict(name=r["Model"],
                            r2=r.get("Space_R2",np.nan),
                            c=tc(r["Model"]),tp="DL"))
    if rows_r:
        rr=pd.DataFrame(rows_r).dropna(subset=["r2"]).sort_values("r2")
        bars2=ax2.barh(rr["name"],rr["r2"],color=rr["c"].tolist(),
                        alpha=0.85,edgecolor="black",lw=0.5)
        for bar,v in zip(bars2,rr["r2"]):
            ax2.text(v+0.002,bar.get_y()+bar.get_height()/2,
                     f"{v:.4f}",va="center",fontsize=7.5,fontweight="bold")
        n_ml2=len(rr[rr["tp"]=="ML"])
        if n_ml2>0: ax2.axhline(n_ml2-0.5,color="black",lw=1.5,ls="--",alpha=0.5)
    ax2.set_xlabel("Unseen Space R²",fontsize=10)
    ax2.set_title(f"{TGT_LABELS[tgt]}\nR² Comparison",fontweight="bold",fontsize=11)

ml_p=[mpatches.Patch(color=c,label=m) for m,c in ML_COLORS.items()]
dl_p=[mpatches.Patch(color=c,label=t) for t,c in TIER_COLORS.items() if t!="ML"]
fig.legend(handles=ml_p+dl_p,loc="lower center",ncol=6,fontsize=9,
           bbox_to_anchor=(0.5,-0.02),title="ML Baselines | DL Tiers",title_fontsize=10)
fig.suptitle("ML Baselines vs DL Spatial Models | All 3 Targets | Test 2025\n"
              "ML: single-point, no spatial graph | DL: 256 locations, GCN, heteroscedastic",
              fontsize=14,fontweight="bold")
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig(FIGS/"FIG_04_ml_vs_dl.png",dpi=300,bbox_inches="tight")
plt.close()
print("    ✓ FIG_04_ml_vs_dl.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_05 — Three-test leaderboard (space/time/both) — all 3 targets
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_05: Three-test leaderboard...")
fig,axes=plt.subplots(1,3,figsize=(26,10))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax  = axes[ti]
    sub = res_df[res_df["Target"]==tgt].sort_values("Space_R2",ascending=False)
    if sub.empty: continue
    r2_cols={"Space_R2":"Space\n(Wetland)","Time_R2":"Time\n(Q4 2025)","Both_R2":"Both\n(Hardest)"}
    avail={k:v for k,v in r2_cols.items() if k in sub.columns}
    if not avail: continue

    n=len(sub); nc=len(avail); w=0.75/nc
    x=np.arange(n); offsets=np.linspace(-(nc-1)*w/2,(nc-1)*w/2,nc)
    colors=[tc(a) for a in sub["Model"]]
    for ci,(col,lbl) in enumerate(avail.items()):
        alpha=0.95 if ci==0 else (0.60 if ci==1 else 0.35)
        ax.bar(x+offsets[ci],sub[col].values,width=w,
                color=colors,alpha=alpha,edgecolor="black",lw=0.4,label=lbl)

    ml_best={"temp":0.794,"smap":0.721,"moist":0.160}[tgt]
    ax.axhline(ml_best,color="grey",ls="--",lw=2,alpha=0.7,
                label=f"ML best={ml_best:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{ARCH_TIERS.get(m,'?')[:3]}]\n{m}"
                         for m in sub["Model"]],rotation=35,ha="right",fontsize=8)
    ax.set_ylabel("R² (Residual)",fontsize=11)
    ax.set_ylim(0,1.05)
    ax.set_title(TGT_LABELS[tgt],fontweight="bold",fontsize=12)

    test_h=[mpatches.Patch(facecolor="grey",alpha=a,label=l)
             for a,l in zip([0.95,0.60,0.35],avail.values())]
    l1=ax.legend(handles=test_h,loc="lower right",fontsize=9,title="Test set")
    ax.add_artist(l1)

fig.legend(handles=lp(),loc="lower center",ncol=6,fontsize=10,
           bbox_to_anchor=(0.5,-0.04),title="Model Tier",title_fontsize=10)
fig.suptitle("Three-Test R² | Space (Wetland holdout) | Time (Q4 2025) | Both\n"
              "v6 Distributed Spatial AI | Residual target",
              fontsize=14,fontweight="bold")
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig(FIGS/"FIG_05_three_test.png",dpi=300,bbox_inches="tight")
plt.close()
print("    ✓ FIG_05_three_test.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_06 — Recoverability curves — all 3 targets side by side
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_06: Recoverability curves...")
try:
    import torch
    fig,axes=plt.subplots(1,3,figsize=(24,8))
    for ti,tgt in enumerate(["temp","smap","moist"]):
        ax=axes[ti]; unit=TGT_UNITS[tgt]
        tau_r=np.linspace(0,5.0,300)
        from scipy.special import erf

        for arch in ARCH_TIERS:
            ckpt_p=PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
            if not ckpt_p.exists(): continue
            ckpt=torch.load(ckpt_p,map_location="cpu")
            tm=ckpt.get("test_metrics",{})
            tier=ARCH_TIERS[arch]; color=TIER_COLORS[tier]
            is_best=(arch in res_df[res_df["Target"]==tgt].nlargest(3,"Space_R2")["Model"].values)
            lw=2.5 if is_best else 1.2; alpha=1.0 if is_best else 0.35

            sp=tm.get("unseen_space",{})
            ubr=sp.get("unseen_ubRMSE",1.0)
            if np.isnan(ubr) or ubr<=0: ubr=1.0
            rec=100*erf(tau_r/(np.sqrt(2)*ubr+1e-8))
            rec=np.clip(rec,0,100)
            lbl=f"[{tier}] {arch}" if is_best else "_nolegend_"
            ax.plot(tau_r,rec,color=color,lw=lw,alpha=alpha,label=lbl)

        ax.axhline(80,color="orange",ls="--",lw=1.5,alpha=0.8,label="80% threshold")
        ax.set_xlabel(f"Error Tolerance ({unit})",fontsize=11)
        ax.set_ylabel("Recoverability (%)",fontsize=11)
        ax.set_title(TGT_LABELS[tgt],fontweight="bold",fontsize=12)
        ax.set_xlim(0,5); ax.set_ylim(0,102)
        ax.legend(fontsize=8,loc="lower right")

    fig.legend(handles=lp()+[plt.Line2D([0],[0],color="orange",ls="--",
                                          lw=1.5,label="80% threshold")],
               loc="lower center",ncol=6,fontsize=10,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle("Recoverability Curves | Unseen Space (Wetland) | All 3 Targets\n"
                  "v6 Distributed Spatial AI | Bold = top-3 models per target",
                  fontsize=14,fontweight="bold")
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(FIGS/"FIG_06_recoverability.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ FIG_06_recoverability.png")
except Exception as e:
    print(f"    ✗ FIG_06: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_07 — Metric heatmap: all metrics, all models (3 targets, 3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_07: Metric heatmap...")
METRIC_COLS={
    "Space_R2":"R²\n(space)","Space_KGE":"KGE\n(space)",
    "Space_ubRMSE":"ubRMSE↓\n(space)","Space_CRPS":"CRPS↓\n(space)",
    "Time_R2":"R²\n(time)","Both_R2":"R²\n(both)",
}
fig,axes=plt.subplots(1,3,figsize=(28,12))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=res_df[res_df["Target"]==tgt].copy()
    avail={k:v for k,v in METRIC_COLS.items() if k in sub.columns}
    if not avail: continue
    pv=sub.set_index("Model")[list(avail.keys())].rename(columns=avail)
    pv=pv.apply(pd.to_numeric,errors="coerce")
    for col in ["ubRMSE↓\n(space)","CRPS↓\n(space)"]:
        if col in pv.columns: pv[col]=-pv[col]
    if "R²\n(space)" in pv.columns:
        pv=pv.sort_values("R²\n(space)",ascending=False)
    # Add tier label
    pv.index=[f"[{ARCH_TIERS.get(m,'?')[:3]}] {m}" for m in pv.index]
    sns.heatmap(pv,ax=ax,cmap="RdYlGn",annot=True,fmt=".3f",
                 linewidths=0.6,linecolor="white",
                 annot_kws={"size":9,"weight":"bold"},
                 cbar_kws={"label":"Score","shrink":0.7})
    ax.set_title(TGT_LABELS[tgt],fontweight="bold",fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20,ha="right",fontsize=10)

fig.suptitle("Performance Metrics | All Models | v6 Distributed Spatial AI\n"
              "Error metrics negated: green=better | Space=Wetland holdout | Time=Q4 2025",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"FIG_07_metric_heatmap.png",dpi=300,bbox_inches="tight")
plt.close()
print("    ✓ FIG_07_metric_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_08 — Unseen R² heatmap by tier (v4 Image 6 style, 3 targets in ONE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_08: Unseen R² heatmap by tier...")
fig,axes=plt.subplots(1,3,figsize=(26,10))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=res_df[res_df["Target"]==tgt].copy()
    if sub.empty or "Space_R2" not in sub.columns: continue
    sub["Tier"]=sub["Model"].map(ARCH_TIERS)
    pv=sub.pivot_table(index="Tier",columns="Model",values="Space_R2",aggfunc="mean")
    tier_order=["STANDALONE","GRAPH","RESERVOIR","SSM","ATTENTION"]
    tier_order=[t for t in tier_order if t in pv.index]
    model_order=[m for t in tier_order
                  for m in sorted(ARCH_TIERS.keys())
                  if ARCH_TIERS.get(m)==t and m in pv.columns]
    pv=pv.reindex(index=tier_order,columns=model_order)
    pv=pv.apply(pd.to_numeric,errors="coerce")

    cmap=mcolors.LinearSegmentedColormap.from_list("rg",
         ["#E74C3C","#F39C12","#F1C40F","#2ECC71","#1A8A3C"])
    vmin=max(0.0,float(pv.min().min())-0.02) if not pv.empty else 0.0
    vmax=min(1.0,float(pv.max().max())+0.01) if not pv.empty else 1.0
    im=ax.imshow(pv.values,cmap=cmap,aspect="auto",vmin=vmin,vmax=vmax)

    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v=pv.values[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:.4f}",ha="center",va="center",
                         fontsize=8.5,fontweight="bold",
                         color="white" if v<(vmin+vmax)/2+0.05 else "black")

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns,rotation=35,ha="right",fontsize=9)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index,fontsize=11,fontweight="bold")
    ax.set_title(TGT_LABELS[tgt],fontweight="bold",fontsize=12)
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)

    for i,tier in enumerate(pv.index):
        if i<len(ax.get_yticklabels()):
            ax.get_yticklabels()[i].set_color(TIER_COLORS.get(tier,"black"))

fig.suptitle("Unseen R² by Tier and Model | v6 Spatial Holdout\n"
              "Wetland site (64 locations) completely excluded from training",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"FIG_08_unseen_r2_heatmap.png",dpi=300,bbox_inches="tight")
plt.close()
print("    ✓ FIG_08_unseen_r2_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_09 — Ablation ΔR² (all 3 targets, 3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_09: Ablation heatmap...")
if not abl_df.empty:
    tgt_col="target" if "target" in abl_df.columns else "Target"
    fig,axes=plt.subplots(1,3,figsize=(26,11))
    for ti,tgt in enumerate(abl_df[tgt_col].unique()):
        ax=axes[ti] if ti<3 else None
        if ax is None: continue
        sub=abl_df[abl_df[tgt_col]==tgt]
        dl_r2={}
        for _,row in res_df[res_df["Target"]==tgt].iterrows():
            dl_r2[row["Model"]]=row.get("Val_R2",row.get("Space_R2",np.nan))
        pv=sub.pivot_table(index="arch",columns="ablation",values="val_r2",aggfunc="mean")
        for arch in pv.index:
            if arch in dl_r2 and not np.isnan(dl_r2[arch]):
                pv.loc[arch]=dl_r2[arch]-pv.loc[arch]
        pv=pv.apply(pd.to_numeric,errors="coerce")
        if pv.empty: continue
        pv["mean"]=pv.mean(axis=1); pv=pv.sort_values("mean",ascending=False).drop(columns=["mean"])
        pv.index=[f"[{ARCH_TIERS.get(m,'?')[:3]}] {m}" for m in pv.index]
        sns.heatmap(pv,ax=ax,cmap="RdYlGn",center=0,annot=True,fmt=".3f",
                     linewidths=0.6,linecolor="white",
                     annot_kws={"size":9,"weight":"bold"},
                     cbar_kws={"label":"ΔR²","shrink":0.7})
        ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=9)
        ax.set_xticklabels([c.replace("_","\n") for c in pv.columns],
                             rotation=0,fontsize=10)
        ax.set_title(TGT_LABELS.get(tgt,tgt),fontweight="bold",fontsize=12)
    fig.suptitle("Ablation Study — ΔR² (Full - Ablated) | All 3 Targets\n"
                  "Green = component helps | Red = removal helps | Rows sorted by mean impact",
                  fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"FIG_09_ablation.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ FIG_09_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_10 — GPU Scaling (speedup/efficiency/drop in ONE figure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_10: GPU scaling...")
if not sc_df.empty and "speedup" in sc_df.columns:
    sc_v=sc_df.dropna(subset=["speedup","efficiency","drop_ratio"])
    GPU_CFGS=sorted(sc_v["n_gpus"].unique())

    fig,axes=plt.subplots(1,3,figsize=(24,8))

    # Speedup
    ax=axes[0]
    grp=sc_v.groupby(["n_gpus","tier"])["speedup"].mean().reset_index()
    for tier in TIER_COLORS:
        t_sub=grp[grp["tier"]==tier]
        if t_sub.empty: continue
        ax.plot(t_sub["n_gpus"],t_sub["speedup"],
                 color=TIER_COLORS[tier],marker="o",lw=2.5,ms=9,label=tier)
    ax.plot(GPU_CFGS,[float(g)/GPU_CFGS[0] for g in GPU_CFGS],
             "k--",lw=1.5,alpha=0.4,label="Ideal")
    ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Speedup (×)")
    ax.set_title("Speedup",fontweight="bold"); ax.set_xticks(GPU_CFGS)
    ax.legend(fontsize=9)

    # Efficiency
    ax=axes[1]
    grp2=sc_v.groupby(["n_gpus","tier"])["efficiency"].mean().reset_index()
    for tier in TIER_COLORS:
        t_sub=grp2[grp2["tier"]==tier]
        if t_sub.empty: continue
        ax.plot(t_sub["n_gpus"],t_sub["efficiency"],
                 color=TIER_COLORS[tier],marker="o",lw=2.5,ms=9,label=tier)
    ax.axhline(80,color="orange",ls="--",lw=1.5,label="80% threshold")
    ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Efficiency (%)")
    ax.set_title("Parallel Efficiency",fontweight="bold"); ax.set_xticks(GPU_CFGS)
    ax.legend(fontsize=9)

    # Drop ratio
    ax=axes[2]
    grp3=sc_v.groupby(["n_gpus","tier"])["drop_ratio"].mean().reset_index()
    for tier in TIER_COLORS:
        t_sub=grp3[grp3["tier"]==tier]
        if t_sub.empty: continue
        ax.plot(t_sub["n_gpus"],t_sub["drop_ratio"],
                 color=TIER_COLORS[tier],marker="o",lw=2.5,ms=9,label=tier)
    ax.axhline(0.05,color="orange",ls="--",lw=1.5,label="5% threshold")
    ax.axhline(0,color="black",lw=1,alpha=0.4)
    ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Drop Ratio")
    ax.set_title("Quality Degradation",fontweight="bold"); ax.set_xticks(GPU_CFGS)
    ax.legend(fontsize=9)

    fig.legend(handles=lp(),loc="lower center",ncol=5,fontsize=10,
               bbox_to_anchor=(0.5,-0.04))
    fig.suptitle("GPU Scaling | nn.DataParallel + GraphAwareWrapper | 1→2→4→8 GPUs\n"
                  "Sweet spot: 4 GPUs — 1.86× speedup, 9.8% R² drop, 46% efficiency",
                  fontsize=14,fontweight="bold")
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(FIGS/"FIG_10_gpu_scaling.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("    ✓ FIG_10_gpu_scaling.png")
else:
    print("    ✗ FIG_10: scaling CSV not ready")


# ══════════════════════════════════════════════════════════════════════════════
# FIG_11 — Entropy convergence (all targets, all models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  FIG_11: Entropy convergence...")
if not ent_df.empty and "initial" in ent_df.columns:
    tgt_col="target" if "target" in ent_df.columns else "Target"
    targets_with_data=[t for t in ent_df[tgt_col].unique()
                        if not ent_df[ent_df[tgt_col]==t].dropna(subset=["initial","final"]).empty]
    if targets_with_data:
        fig,axes=plt.subplots(1,len(targets_with_data),
                               figsize=(8*len(targets_with_data),8))
        if len(targets_with_data)==1: axes=[axes]
        for ai,tgt in enumerate(targets_with_data):
            ax=axes[ai]
            sub=ent_df[ent_df[tgt_col]==tgt].dropna(subset=["initial","final"])
            sub=sub.sort_values("initial",ascending=False).reset_index(drop=True)
            x=np.arange(len(sub)); w=0.35
            colors=[tc(a) for a in sub["arch"]]
            ax.bar(x-w/2,sub["initial"],width=w,color=colors,alpha=0.3,
                    edgecolor="black",lw=0.8,label="Initial")
            ax.bar(x+w/2,sub["final"],  width=w,color=colors,alpha=0.9,
                    edgecolor="black",lw=0.8,label="Final")
            for i,(init,fin) in enumerate(zip(sub["initial"],sub["final"])):
                if not(np.isnan(init) or np.isnan(fin)):
                    delta=init-fin
                    ax.text(x[i],max(init,abs(fin))+0.02,f"Δ{delta:.2f}",
                             ha="center",fontsize=7.5,color="darkred")
            ax.set_xticks(x)
            ax.set_xticklabels([f"[{ARCH_TIERS.get(a,'?')[:3]}]\n{a}"
                                  for a in sub["arch"]],rotation=30,ha="right",fontsize=8)
            ax.set_ylabel("Predictive Entropy H",fontsize=11)
            ax.set_title(TGT_LABELS.get(tgt,tgt),fontweight="bold",fontsize=12)
            ax.legend(fontsize=9)

        fig.suptitle("Entropy Convergence: Initial → Best Epoch | v6 Heteroscedastic Models\n"
                      "H = 0.5(1+log(2πσ²)) | Larger Δ = more learning",
                      fontsize=14,fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGS/"FIG_11_entropy.png",dpi=300,bbox_inches="tight")
        plt.close()
        print("    ✓ FIG_11_entropy.png")
else:
    print("    ✗ FIG_11: entropy data not available")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
figs_all=sorted(FIGS.glob("*.png"))
print(f"\n{'='*65}")
print(f"  MANUSCRIPT FIGURES COMPLETE")
print(f"  Location: {FIGS}")
print(f"  Total: {len(figs_all)} figures")
print(f"{'='*65}")
for f in figs_all:
    sz=f.stat().st_size//1024
    print(f"  {f.name:<45} ({sz} KB)")
