"""
================================================================================
postprocess_results_v4.py — Per-Site + Full Spatial Evaluation
================================================================================
RUN AFTER job finishes (login node, no GPU needed):
  module load pytorch-py39-cuda11.8-gcc11/1.13.0
  python3 ~/postprocess_results_v4.py

FIGURES GENERATED:
  PP01  Per-site R² heatmap (Model × Site) — seen locations
  PP02  Per-site R² heatmap — UNSEEN Wetland (key spatial result)
  PP03  Seen vs Unseen comparison per site per model
  PP04  Spatial gap bar chart (seen - unseen) — ablation proof
  PP05  Time series: True vs Predicted at all 4 sites
  PP06  Spatial field snapshot — True vs Predicted vs Residual
  PP07  DeepESN vs SSM comparison per site
  PP08  Freeze-thaw accuracy per site (seen + unseen)
  PP09  Training curves (loss + seen_R² + unseen_R²)
  PP10  Publication summary table
================================================================================
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT  = Path("/home/emmanuel.keku")
PREPROC  = PROJECT / "preprocessed_v3"
RESULTS  = PROJECT / "results_v4"
MODELS   = PROJECT / "models_v4" / "dl"
FIGS     = PROJECT / "figures_v4"
LOG_DIR  = PROJECT / "logs"
for d in [RESULTS, FIGS]: d.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3,
                              "axes.spines.top":False,"axes.spines.right":False})

print("=" * 70)
print("  POST-PROCESSING v4 — Per-Site Spatial Generalisation")
print(f"  Start: {pd.Timestamp.now()}")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = torch.device("cpu")
    print(f"PyTorch: {torch.__version__} | Device: cpu (login node)")
except ImportError:
    print("FATAL: Load modules first"); sys.exit(1)

# ── Load data and scalers ─────────────────────────────────────────────────────
if not (PREPROC/"master_processed.csv").exists():
    print("FATAL: Run train scripts first"); sys.exit(1)

print("\nLoading data...")
df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)

LOCATIONS        = pd.DataFrame(FI["LOCATIONS"])
N_LOCS           = FI["N_LOCS"]
SNAP_FEATURES    = FI["SNAP_FEATURES"]
ALL_TARGETS      = FI["ALL_TARGETS"]
TEMP_TARGETS     = FI["TEMP_TARGETS"]
SMAP_TARGETS     = FI["SMAP_TARGETS"]
MOIST_TARGETS    = FI["MOIST_TARGETS"]
SITES            = FI["SITES"]
snap_feat_scaler = SC["snap_feat_scaler"]

loc_coords = LOCATIONS[["Latitude","Longitude"]].values

# Rebuild v4 features (approx as input)
APPROX_FEATS = [f"{t}_approx" for t in ALL_TARGETS if f"{t}_approx" in df.columns]
V4_FEATURES  = list(dict.fromkeys(SNAP_FEATURES + APPROX_FEATS))
V4_FEATURES  = [f for f in V4_FEATURES if f in df.columns]
N_V4_FEATURES= len(V4_FEATURES)

from sklearn.preprocessing import RobustScaler
tr_all = df[df["split"]=="train"]
v4_feat_scaler = RobustScaler()
v4_feat_scaler.fit(tr_all[V4_FEATURES].fillna(0).values)

v4_tgt_scalers = {}
for grp, tgts in [("temp",TEMP_TARGETS),("smap",SMAP_TARGETS),("moist",MOIST_TARGETS)]:
    av = [c for c in tgts if c in tr_all.columns]
    if not av: continue
    ts = RobustScaler(); ts.fit(tr_all[av].dropna().values)
    v4_tgt_scalers[grp] = ts

# ── Spatial graph ─────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree

def build_graph(locs_df, k=6):
    coords = locs_df[["Latitude","Longitude"]].values.astype(np.float32)
    N      = len(coords)
    scaled = coords * np.array([111.0, 63.0], dtype=np.float32)
    tree   = cKDTree(scaled)
    dists, idxs = tree.query(scaled, k=min(k+1,N))
    sigma  = np.median(dists[:,1:])+1e-8
    A      = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for jp in range(1,dists.shape[1]):
            j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
            A[i,j]+=w; A[j,i]+=w
    A += np.eye(N); D=A.sum(1,keepdims=True)**0.5
    return (A/(D*D.T+1e-8)).astype(np.float32)

A_norm = torch.tensor(build_graph(LOCATIONS, k=6))

# ── Site index mapping ────────────────────────────────────────────────────────
loc_to_idx = {(float(r.Latitude),float(r.Longitude)):i
              for i,r in LOCATIONS.iterrows()}

HOLDOUT_SITE   = "Wetland"
TRAINING_SITES = [s for s in SITES if s!=HOLDOUT_SITE]

def get_site_idxs(site):
    sd = df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    return sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                   for _,r in sd.iterrows()
                   if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

site_idxs = {s: get_site_idxs(s) for s in SITES}
SEEN_LOCS   = sorted(set(i for s in TRAINING_SITES for i in site_idxs[s]))
UNSEEN_LOCS = site_idxs[HOLDOUT_SITE]

print(f"  Seen locs  : {len(SEEN_LOCS)}")
print(f"  Unseen locs: {len(UNSEEN_LOCS)} ({HOLDOUT_SITE})")

# ── Model definitions (lightweight copies) ─────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self,id,od,dp=0.1):
        super().__init__()
        self.W=nn.Linear(id,od,bias=False); self.n=nn.LayerNorm(od)
        self.d=nn.Dropout(dp); self.a=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),self.W(self.d(H)))))

def make_model(arch, nt):
    """Reconstruct model from arch name for checkpoint loading."""
    nf = N_V4_FEATURES; h=96; d=96
    if arch == "BiGRU_NoGCN":
        m = nn.Sequential()
        # Minimal reconstruction — load from checkpoint handles this
    # We use exec to avoid duplicating all model code
    # Instead load the full model by importing from train script
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "v4mod", str(PROJECT/"train_soil_spatial_v4.py"))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ARCH_MAP[arch](nt)
    except Exception as e:
        print(f"  ✗ Cannot load model class {arch}: {e}")
        return None

# ── Dataset for evaluation ────────────────────────────────────────────────────
from torch.utils.data import Dataset, DataLoader

class EvalDataset(Dataset):
    def __init__(self, df, tgt_cols, tgt_scaler, split="test",
                 lookback=24, stride=24):
        self.A = A_norm
        N=N_LOCS; nf=N_V4_FEATURES; nt=len(tgt_cols)
        sub = df[df["split"]==split].copy()
        all_ts = sorted(sub["time_utc"].unique())
        T = len(all_ts)
        self.timestamps = all_ts
        if T < lookback+2: self.X=self.y=self.mask=torch.zeros(0); return

        ts_to_i={ts:i for i,ts in enumerate(all_ts)}
        sub2=sub.copy()
        sub2["_ti"]=sub2["time_utc"].map(ts_to_i)
        sub2["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                     for la,lo in zip(sub2["Latitude"].astype(float),
                                      sub2["Longitude"].astype(float))]
        sub2=sub2.dropna(subset=["_ti","_ni"])
        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
        ti_arr=sub2["_ti"].values; ni_arr=sub2["_ni"].values

        X_full=np.full((T,N,nf),np.nan,dtype=np.float32)
        y_full=np.full((T,N,nt),np.nan,dtype=np.float32)
        X_full[ti_arr,ni_arr,:]=v4_feat_scaler.transform(
            sub2[V4_FEATURES].fillna(0).values).astype(np.float32)
        y_full[ti_arr,ni_arr,:]=tgt_scaler.transform(
            sub2[tgt_cols].fillna(0).values).astype(np.float32)

        tidxs=list(range(lookback,T,stride))
        Xl=[]; yl=[]; tsl=[]
        for ti in tidxs:
            Xw=X_full[ti-lookback:ti]; yi=y_full[ti]
            if np.isnan(Xw).mean()>0.25: continue
            Xl.append(np.nan_to_num(Xw,nan=0.0))
            yl.append(np.nan_to_num(yi,nan=0.0))
            tsl.append(all_ts[ti])
        if not Xl: self.X=self.y=torch.zeros(0); return
        self.X=torch.tensor(np.array(Xl),dtype=torch.float32)
        self.y=torch.tensor(np.array(yl),dtype=torch.float32)
        self.timestamps=tsl
        print(f"    [eval {split}] {len(self.X)} samples")

    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i],self.A


def site_metrics(yt, yp, idxs, label):
    if not idxs: return {}
    a=yt[:,idxs,0].flatten(); b=yp[:,idxs,0].flatten()
    mk=~(np.isnan(a)|np.isnan(b)); a=a[mk]; b=b[mk]
    if len(a)<5: return {}
    r2  = float(1-np.sum((a-b)**2)/(np.sum((a-a.mean())**2)+1e-10))
    rms = float(np.sqrt(np.mean((a-b)**2)))
    r   = float(np.corrcoef(a,b)[0,1])
    kge = float(1-np.sqrt((r-1)**2+(np.std(b)/(np.std(a)+1e-10)-1)**2+
                           (np.mean(b)/(np.mean(a)+1e-10)-1)**2))
    frz = float(np.mean((a<0).astype(int)==(b<0).astype(int))*100)
    sk  = float(1-np.mean((a-b)**2)/(np.var(a)+1e-10))
    return {f"{label}_R2":round(r2,4), f"{label}_RMSE":round(rms,4),
            f"{label}_KGE":round(kge,4), f"{label}_Freeze":round(frz,2),
            f"{label}_Skill":round(sk,4)}


# ── Main evaluation loop ───────────────────────────────────────────────────────
ARCHES = ["BiGRU_NoGCN","GCN_NoTemporal","DeepESN","SpatialESN",
          "GraphSAGE","GAT","STGCN",
          "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]
TIERS  = {"BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
           "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
           "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
           "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
           "SpatialS4":"SSM","SpatialFuseMoE":"SSM"}
TGT_MAP= {"temp":TEMP_TARGETS,"smap":SMAP_TARGETS,"moist":MOIST_TARGETS}
TGT_LABELS={"temp":"Weather Temp","smap":"SMAP Temp L1","moist":"Soil Moisture"}
MODEL_COLORS={"BiGRU_NoGCN":"#d62728","GCN_NoTemporal":"#ff9896",
              "DeepESN":"#9467bd","SpatialESN":"#c5b0d5",
              "GraphSAGE":"#2ca02c","GAT":"#98df8a","STGCN":"#17becf",
              "SpatialBiGRU":"#1f77b4","SpatialMamba":"#aec7e8",
              "SpatialS4":"#ff7f0e","SpatialFuseMoE":"#ffbb78"}
TIER_COLORS={"ABLATION":"#d62728","RESERVOIR":"#9467bd",
             "GRAPH":"#2ca02c","SSM":"#1f77b4"}

all_records  = []
ts_store     = {}   # for time series plots

print("\n" + "="*60)
print("  EVALUATING ALL MODELS PER SITE")
print("="*60)

for tgt_name, tgt_cols in TGT_MAP.items():
    tgt_sc = v4_tgt_scalers.get(tgt_name)
    if tgt_sc is None: continue
    av_tgt = [c for c in tgt_cols if c in df.columns]
    if not av_tgt: continue

    ds = EvalDataset(df, av_tgt, tgt_sc, split="test",
                     lookback=24, stride=24)
    if len(ds)==0: continue
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    for arch in ARCHES:
        ckpt = MODELS / f"{arch}_{tgt_name}_v4_best.pt"
        if not ckpt.exists():
            print(f"  ✗ {arch} [{tgt_name}] — missing"); continue

        try:
            sv    = torch.load(ckpt, map_location=DEVICE)
            model = make_model(arch, len(av_tgt))
            if model is None: continue
            model.load_state_dict(sv["state_dict"])
            model.eval()
            is_moe = (arch=="SpatialFuseMoE")

            yt_all=[]; yp_all=[]
            with torch.no_grad():
                for X,y,A_ in loader:
                    X,y,A_=[b.to(DEVICE) for b in [X,y,A_]]
                    out=model(X,A_); pred=out[0] if is_moe else out
                    B_,N_,T_=pred.shape
                    pr=pred.cpu().float().numpy()
                    pr_r=tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
                    y_r=tgt_sc.inverse_transform(
                        y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
                    yt_all.append(y_r); yp_all.append(pr_r)

            yt=np.concatenate(yt_all,0); yp=np.concatenate(yp_all,0)
            ts_store[f"{arch}_{tgt_name}"] = dict(
                yt=yt, yp=yp, ts=ds.timestamps)

            rec = dict(Model=arch,Target=tgt_name,Tier=TIERS.get(arch,"?"))
            # Per-site metrics
            for site in SITES:
                idxs = site_idxs.get(site,[])
                is_unseen = (site==HOLDOUT_SITE)
                label = f"{'UNSEEN_' if is_unseen else ''}{site}"
                m = site_metrics(yt, yp, idxs, site)
                rec.update(m)
                flag = "UNSEEN" if is_unseen else "seen"
                print(f"  ✓ {arch:<20} [{tgt_name}] {site:<12} [{flag}] "
                      f"R²={m.get(f'{site}_R2',float('nan')):.4f} "
                      f"Skill={m.get(f'{site}_Skill',float('nan')):.4f} "
                      f"Frz={m.get(f'{site}_Freeze',float('nan')):.1f}%")
            # Seen/unseen aggregate
            m_seen  = site_metrics(yt,yp,SEEN_LOCS,"seen")
            m_unseen= site_metrics(yt,yp,UNSEEN_LOCS,"unseen")
            rec.update(m_seen); rec.update(m_unseen)
            rec["spatial_gap"] = round(
                m_seen.get("seen_R2",float("nan")) -
                m_unseen.get("unseen_R2",float("nan")), 4)
            all_records.append(rec)

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {arch} [{tgt_name}]: {e}")

# ── Save results ───────────────────────────────────────────────────────────────
site_df = pd.DataFrame(all_records)
site_df.to_csv(RESULTS/"postprocess_v4_results.csv", index=False)
print(f"\n  Saved: {len(site_df)} records")

if len(site_df)==0:
    print("No results — check checkpoints."); sys.exit(0)

tgts   = sorted(site_df["Target"].unique())
models = [m for m in ARCHES if m in site_df["Model"].unique()]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  GENERATING FIGURES")
print("="*55)
from matplotlib.patches import Patch

# ── PP00: Full Model × Site heatmap (all 4 sites) — matches v3 style ─────────
# This is the direct equivalent of v3 Image 7
# Wetland column is UNSEEN — highlighted with red border
for tgt in tgts:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty: continue

    for metric, lbl, vmin in [
        ("R2",    "R²",              0.90),
        ("Skill", "Skill vs Seasonal",-0.1),
        ("KGE",   "KGE",              0.60),
        ("Freeze","Freeze Acc (%)",   90.0),
    ]:
        # Build pivot: rows=models, cols=all 4 sites
        pivot_data = {}
        for site in SITES:
            col = f"{site}_{metric}"
            if col in sub.columns:
                pivot_data[site] = sub.set_index("Model")[col]

        if not pivot_data: continue
        pv = pd.DataFrame(pivot_data)
        pv = pv.reindex(index=[m for m in ARCHES if m in pv.index])
        if pv.empty: continue

        fig, ax = plt.subplots(figsize=(16, max(9, len(pv)*0.7+2)))

        # Color scale
        vmax = 1.0 if metric != "Freeze" else 100.0
        cmap = "RdYlGn"

        im = sns.heatmap(pv.astype(float), ax=ax, cmap=cmap,
                         vmin=vmin, vmax=vmax,
                         annot=True, fmt=".3f" if metric!="Freeze" else ".1f",
                         linewidths=0.5, linecolor="white",
                         annot_kws={"size":11,"weight":"bold"},
                         cbar_kws={"label":lbl,"shrink":0.85})

        # Highlight Wetland column (UNSEEN) with red border
        unseen_col_idx = list(pv.columns).index(HOLDOUT_SITE) if HOLDOUT_SITE in pv.columns else None
        if unseen_col_idx is not None:
            for row_idx in range(len(pv)):
                ax.add_patch(plt.Rectangle(
                    (unseen_col_idx, row_idx), 1, 1,
                    fill=False, edgecolor="red", lw=3, zorder=5))

        # Column labels with UNSEEN marker
        new_labels = [f"{s}\n[UNSEEN]" if s==HOLDOUT_SITE else s
                      for s in pv.columns]
        ax.set_xticklabels(new_labels, rotation=0, fontsize=11)
        ax.set_yticklabels([f"[{TIERS.get(m,'?')}] {m}" for m in pv.index],
                           rotation=0, fontsize=9)

        # Add tier separator lines
        tier_groups = {"ABLATION":[],"RESERVOIR":[],"GRAPH":[],"SSM":[]}
        for i,m in enumerate(pv.index):
            t = TIERS.get(m,"?")
            if t in tier_groups: tier_groups[t].append(i)
        prev_tier = None
        for i,m in enumerate(pv.index):
            t = TIERS.get(m,"?")
            if t != prev_tier and i > 0:
                ax.axhline(i, color="white", lw=3)
            prev_tier = t

        ax.set_title(
            f"{lbl} per Model × Site | {TGT_LABELS.get(tgt,tgt)}\n"
            f"Red border = UNSEEN ({HOLDOUT_SITE}) — predicted via GCN only | Test 2025",
            fontweight="bold", fontsize=12)
        ax.set_xlabel("Site", fontsize=11)
        ax.set_ylabel("Model", fontsize=11)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=c, label=t) for t,c in TIER_COLORS.items()
        ] + [
            plt.Rectangle((0,0),1,1, fill=False, edgecolor="red",
                          lw=3, label=f"UNSEEN ({HOLDOUT_SITE})")
        ]
        ax.legend(handles=legend_handles, fontsize=9, ncol=3,
                  loc="upper right", bbox_to_anchor=(1.0, -0.08))

        plt.tight_layout()
        fname = f"PP00_{metric.lower()}_heatmap_all_sites_{tgt}.png"
        plt.savefig(FIGS/fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ PP00 {metric} heatmap all sites [{tgt}]")

# ── PP01: Per-site R² heatmap — seen sites ────────────────────────────────────
for tgt in tgts:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty: continue
    seen_sites = [s for s in SITES if s!=HOLDOUT_SITE]
    r2_cols = [f"{s}_R2" for s in seen_sites if f"{s}_R2" in sub.columns]
    if not r2_cols: continue
    pv = sub.set_index("Model")[r2_cols]
    pv.columns = [c.replace("_R2","") for c in pv.columns]
    pv = pv.reindex([m for m in ARCHES if m in pv.index])
    fig,ax=plt.subplots(figsize=(14,9))
    sns.heatmap(pv.astype(float),ax=ax,cmap="RdYlGn",vmin=0.85,vmax=1.0,
                annot=True,fmt=".4f",linewidths=0.5,linecolor="white",
                annot_kws={"size":11,"weight":"bold"},
                cbar_kws={"label":"R²","shrink":0.85})
    ax.set_title(f"R² per Model × Site (SEEN locations)\n"
                 f"{TGT_LABELS.get(tgt,tgt)} | Test 2025 | "
                 f"Training sites: {seen_sites}",
                 fontweight="bold",fontsize=12)
    ax.tick_params(axis="x",rotation=20,labelsize=10)
    ax.tick_params(axis="y",rotation=0, labelsize=10)
    plt.tight_layout()
    plt.savefig(FIGS/f"PP01_seen_r2_heatmap_{tgt}.png",dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP01 [{tgt}]")

# ── PP02: UNSEEN site R² — the key spatial generalisation figure ──────────────
unseen_col = f"{HOLDOUT_SITE}_R2"
for tgt in tgts:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty or unseen_col not in sub.columns: continue
    sub2 = sub[["Model","Tier",unseen_col]].dropna().sort_values(unseen_col,ascending=True)
    fig,ax=plt.subplots(figsize=(14,9))
    colors=[TIER_COLORS.get(TIERS.get(m,"?"),"grey") for m in sub2["Model"]]
    bars=ax.barh(sub2["Model"],sub2[unseen_col],color=colors,
                  alpha=0.85,edgecolor="black",lw=0.6,height=0.6)
    for bar,v in zip(bars,sub2[unseen_col]):
        ax.text(v+0.001,bar.get_y()+bar.get_height()/2,
                f"{v:.4f}",va="center",fontsize=10,fontweight="bold")
    ax.axvline(0,color="black",lw=2)
    ax.axvline(0.85,color="green",ls="--",lw=1.5,alpha=0.7,label="R²=0.85 threshold")
    ax.set_xlabel(f"R² on {HOLDOUT_SITE} (UNSEEN — never seen during training)",
                  fontsize=11)
    ax.set_title(f"Spatial Generalisation to UNSEEN {HOLDOUT_SITE}\n"
                 f"{TGT_LABELS.get(tgt,tgt)} | Predicted ONLY via GCN from neighbours",
                 fontweight="bold",fontsize=12)
    ax.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()]+
             [plt.Line2D([],[],color="green",ls="--",label="R²=0.85")],fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS/f"PP02_unseen_r2_{tgt}.png",dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP02 unseen [{tgt}]")

# ── PP03: Seen vs Unseen side-by-side ─────────────────────────────────────────
for tgt in tgts:
    sub=site_df[site_df["Target"]==tgt].dropna(subset=["seen_R2","unseen_R2"])
    if sub.empty: continue
    sub=sub.sort_values("unseen_R2",ascending=True)
    fig,ax=plt.subplots(figsize=(16,10))
    x=np.arange(len(sub)); w=0.38
    b1=ax.barh(x-w/2,sub["seen_R2"],  height=w,label=f"Seen ({len(SEEN_LOCS)} locs)",
               color="#1f77b4",alpha=0.85,edgecolor="black",lw=0.5)
    b2=ax.barh(x+w/2,sub["unseen_R2"],height=w,
               label=f"Unseen — {HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locs)",
               color="#d62728",alpha=0.85,edgecolor="black",lw=0.5)
    for bar,v in zip(b1,sub["seen_R2"]):
        ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                va="center",fontsize=8.5,fontweight="bold",color="#1f77b4")
    for bar,v in zip(b2,sub["unseen_R2"]):
        ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                va="center",fontsize=8.5,fontweight="bold",color="#d62728")
    ax.set_yticks(x)
    ax.set_yticklabels([f"[{TIERS.get(m,'?')}] {m}" for m in sub["Model"]],fontsize=9)
    ax.set_xlabel("R²",fontsize=11); ax.set_xlim(0,1.05)
    ax.axvline(0.85,color="grey",ls="--",lw=1,alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title(f"Seen vs Unseen R² | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"Small gap = strong spatial generalisation via GCN",
                 fontweight="bold",fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS/f"PP03_seen_vs_unseen_{tgt}.png",dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP03 [{tgt}]")

# ── PP04: Spatial gap — ablation proof ───────────────────────────────────────
for tgt in tgts:
    sub=site_df[site_df["Target"]==tgt].dropna(subset=["spatial_gap"])
    if sub.empty: continue
    sub=sub.sort_values("spatial_gap",ascending=False)
    fig,ax=plt.subplots(figsize=(14,8))
    colors=[TIER_COLORS.get(TIERS.get(m,"?"),"grey") for m in sub["Model"]]
    bars=ax.bar(sub["Model"],sub["spatial_gap"],color=colors,
                alpha=0.85,edgecolor="black",lw=0.5)
    for bar,v in zip(bars,sub["spatial_gap"]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
                f"{v:.4f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.axhline(0,color="black",lw=2)
    ax.axhline(0.05,color="orange",ls="--",lw=1.5,alpha=0.8,label="5% threshold")
    ax.set_ylabel("Spatial Gap (seen_R² − unseen_R²)",fontsize=11)
    ax.set_title(f"Spatial Generalisation Gap | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"BiGRU_NoGCN should have LARGE gap (no spatial propagation)\n"
                 f"GraphSAGE/SpatialESN should have SMALL gap (GCN propagates to unseen)",
                 fontweight="bold",fontsize=12)
    ax.tick_params(axis="x",rotation=30,labelsize=9)
    ax.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()]+
             [plt.Line2D([],[],color="orange",ls="--",label="5% threshold")],fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS/f"PP04_spatial_gap_{tgt}.png",dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP04 [{tgt}]")

# ── PP05: Time series per site (best model) ───────────────────────────────────
best_model = (site_df[site_df["Target"]=="temp"]
              .sort_values("unseen_R2",ascending=False)["Model"].iloc[0]
              if "unseen_R2" in site_df.columns and
                 not site_df[site_df["Target"]=="temp"]["unseen_R2"].isna().all()
              else ARCHES[0])
key = f"{best_model}_temp"
if key in ts_store:
    store=ts_store[key]; yt_all=store["yt"]; yp_all=store["yp"]; tss=store["ts"]
    fig,axes=plt.subplots(len(SITES),1,figsize=(20,5*len(SITES)),sharex=True)
    for ax,site in zip(axes,SITES):
        idxs=site_idxs.get(site,[])
        if not idxs: continue
        yt_s=yt_all[:,idxs,0].mean(1); yp_s=yp_all[:,idxs,0].mean(1)
        ts_pd=pd.to_datetime(tss)
        col="#d62728" if site==HOLDOUT_SITE else "#1f77b4"
        ax.plot(ts_pd,yt_s,lw=1.5,color=col,label="True",alpha=0.9)
        ax.plot(ts_pd,yp_s,lw=1.5,color="black",ls="--",label="Predicted",alpha=0.8)
        ax.axhline(0,color="grey",ls=":",lw=1,alpha=0.5)
        r2_v=site_df[(site_df["Model"]==best_model)&(site_df["Target"]=="temp")]\
              .get(f"{site}_R2",pd.Series([float("nan")])).values
        tag = " [UNSEEN — spatial holdout]" if site==HOLDOUT_SITE else " [SEEN]"
        ax.set_ylabel(f"{site}\nTemp (°C)",fontsize=10)
        ax.set_title(f"{site}{tag} | R²={r2_v[0]:.4f}" if len(r2_v)>0 else site,
                     fontweight="bold",fontsize=11,
                     color="#d62728" if site==HOLDOUT_SITE else "black")
        ax.legend(fontsize=9,loc="upper right")
    axes[-1].set_xlabel("Date",fontsize=11)
    fig.suptitle(f"True vs Predicted | {best_model} | Test 2025\n"
                 f"Red site = UNSEEN during training — predicted via GCN only",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"PP05_timeseries_all_sites.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP05 time series")

# ── PP06: Spatial field snapshot ─────────────────────────────────────────────
if key in ts_store:
    store=ts_store[key]; yt_all=store["yt"]; yp_all=store["yp"]
    mid=len(yt_all)//2
    yt_s=yt_all[mid,:,0]; yp_s=yp_all[mid,:,0]; res=yp_s-yt_s
    ts_str=str(store["ts"][mid])[:10] if store["ts"] else "Test"
    fig,axes=plt.subplots(1,3,figsize=(24,9))
    for ax,data,title,cmap in [
        (axes[0],yt_s,"True Spatial Field","RdBu_r"),
        (axes[1],yp_s,"Predicted Spatial Field","RdBu_r"),
        (axes[2],res,"Residual (Pred - True)","RdBu_r")]:
        vm=max(abs(np.nanmin(data)),abs(np.nanmax(data)))
        sc=ax.scatter(loc_coords[:,1],loc_coords[:,0],c=data,cmap=cmap,
                      vmin=-vm,vmax=vm,s=55,edgecolors="black",lw=0.3,zorder=5)
        plt.colorbar(sc,ax=ax,label="Soil Temp (°C)",shrink=0.85)
        ax.set_xlabel("Longitude",fontsize=10); ax.set_ylabel("Latitude",fontsize=10)
        ax.set_title(title,fontweight="bold",fontsize=12)
        for site,(lat,lon) in [("Bedrock",(66.25,-150.7)),
                                ("Transition",(67.5,-150.5)),
                                ("Upland",(68.5,-150.5)),
                                ("Wetland",(67.2,-151.0))]:
            col="red" if site==HOLDOUT_SITE else "navy"
            ax.annotate(site,xy=(lon,lat),fontsize=9,fontweight="bold",
                        color=col,ha="center",
                        bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.7))
    fig.suptitle(f"Spatial Field Snapshot | {best_model} | {ts_str}\n"
                 f"Red = {HOLDOUT_SITE} (UNSEEN — predicted via GCN neighbours)",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"PP06_spatial_field_snapshot.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP06 spatial snapshot")

# ── PP07: DeepESN vs SSM ─────────────────────────────────────────────────────
res_ssm=site_df[site_df["Tier"].isin(["RESERVOIR","SSM"])]
if not res_ssm.empty and "unseen_R2" in res_ssm.columns:
    fig,axes=plt.subplots(1,2,figsize=(20,9))
    for ax,metric,lbl in [(axes[0],"seen_R2","Seen R²"),
                           (axes[1],"unseen_R2",f"Unseen R² ({HOLDOUT_SITE})")]:
        if metric not in res_ssm.columns: continue
        sub=res_ssm.dropna(subset=[metric]).sort_values(metric,ascending=True)
        colors=[TIER_COLORS.get(TIERS.get(m,"?"),"grey") for m in sub["Model"]]
        bars=ax.barh(sub["Model"],sub[metric],color=colors,
                      alpha=0.85,edgecolor="black",lw=0.5)
        for bar,v in zip(bars,sub[metric]):
            ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                    va="center",fontsize=9,fontweight="bold")
        ax.set_xlabel(lbl,fontsize=11)
        ax.set_title(lbl,fontweight="bold",fontsize=12)
    fig.suptitle("DeepESN vs SSM Models\n"
                 "Per senior recommendation — arXiv:1712.04323 & arXiv:2509.04422",
                 fontsize=14,fontweight="bold")
    fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
               loc="lower center",ncol=4,fontsize=10,bbox_to_anchor=(0.5,-0.04))
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(FIGS/"PP07_deepesn_vs_ssm.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP07 DeepESN vs SSM")

# ── PP08: Freeze-thaw per site ────────────────────────────────────────────────
frz_cols = [f"{s}_Freeze" for s in SITES if f"{s}_Freeze" in site_df.columns]
if frz_cols:
    sub_temp=site_df[site_df["Target"]=="temp"].dropna(subset=frz_cols[:1])
    if not sub_temp.empty:
        fig,ax=plt.subplots(figsize=(18,9))
        x=np.arange(len(SITES)); w=0.8/max(len(models),1)
        for mi,model in enumerate([m for m in ARCHES if m in sub_temp["Model"].unique()]):
            row=sub_temp[sub_temp["Model"]==model]
            if row.empty: continue
            vals=[row[f"{s}_Freeze"].values[0] if f"{s}_Freeze" in row.columns else 0
                  for s in SITES]
            color=MODEL_COLORS.get(model,"grey")
            bars=ax.bar(x+mi*w-0.4+w/2,vals,width=w*0.9,label=model,
                        color=color,alpha=0.85,edgecolor="black",lw=0.4)
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_height()+0.1,f"{v:.0f}%",
                                ha="center",va="bottom",fontsize=6.5,fontweight="bold")
        ax.set_xticks(x)
        xlabels=[f"{s}\n[UNSEEN]" if s==HOLDOUT_SITE else s for s in SITES]
        ax.set_xticklabels(xlabels,fontsize=11)
        ax.set_ylabel("Freeze-Thaw Accuracy (%)",fontsize=11); ax.set_ylim(80,103)
        ax.axhline(95,color="orange",ls="--",lw=2,alpha=0.8,label="95% threshold")
        ax.set_title("Freeze-Thaw Accuracy per Site\n"
                     "Seen and Unseen (Wetland) locations | Test 2025",
                     fontweight="bold",fontsize=12)
        ax.legend(fontsize=7,ncol=4,loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGS/"PP08_freeze_thaw_per_site.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  ✓ PP08 freeze-thaw")

# ── PP09: Training curves ─────────────────────────────────────────────────────
ckpt_files=sorted(MODELS.glob("*_v4_best.pt"))
if ckpt_files:
    fig,axes=plt.subplots(3,1,figsize=(18,16),sharex=False)
    for ckpt in ckpt_files[:12]:
        try:
            sv=torch.load(ckpt,map_location="cpu")
            hist=sv.get("history",[]); arch=sv.get("arch","?")
            if not hist: continue
            tgt_=ckpt.stem.replace("_v4_best","").replace(f"{arch}_","")
            col=MODEL_COLORS.get(arch,"grey"); lbl=f"{arch}[{tgt_}]"
            eps=[h["epoch"] for h in hist]
            axes[0].plot(eps,[h["train_loss"] for h in hist],lw=1.2,alpha=0.7,color=col,label=lbl)
            axes[1].plot(eps,[h.get("val_R2_seen",0) for h in hist],lw=1.2,alpha=0.7,color=col)
            axes[2].plot(eps,[h.get("val_R2_unseen",0) for h in hist],lw=1.2,alpha=0.7,color=col,ls="--")
        except Exception: continue
    for ax,yl,tl in [(axes[0],"Loss","Training Loss"),
                      (axes[1],"Seen R²","Val R² — Seen Locations"),
                      (axes[2],"Unseen R²",f"Val R² — {HOLDOUT_SITE} (Unseen — KEY)")]:
        ax.set_ylabel(yl,fontsize=10); ax.set_title(tl,fontweight="bold",fontsize=11)
        ax.set_xlabel("Epoch",fontsize=10)
    axes[0].legend(fontsize=6,ncol=4,loc="best")
    fig.suptitle("Training Curves — v4 All Models\n"
                 "Dashed = unseen Wetland R² (spatial generalisation)",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"PP09_training_curves.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP09 training curves")

# ── PP10: Publication summary table ──────────────────────────────────────────
if "unseen_R2" in site_df.columns:
    summary=site_df.groupby(["Tier","Model","Target"])[
        ["seen_R2","unseen_R2","spatial_gap",
         f"{HOLDOUT_SITE}_Freeze","seen_KGE"]].mean().round(4).reset_index()
    summary=summary.sort_values("unseen_R2",ascending=False)

    fig,ax=plt.subplots(figsize=(22,max(8,len(summary)*0.55+2)))
    ax.axis("off")
    cols=["Tier","Model","Target","Seen R²","Unseen R²","Gap","KGE","Freeze(Unseen)"]
    rows=[]
    for _,r in summary.iterrows():
        rows.append([r["Tier"],r["Model"],r["Target"],
                     f"{r['seen_R2']:.4f}",f"{r['unseen_R2']:.4f}",
                     f"{r['spatial_gap']:.4f}",f"{r['seen_KGE']:.4f}",
                     f"{r[f'{HOLDOUT_SITE}_Freeze']:.1f}%"])
    tbl=ax.table(cellText=rows,colLabels=cols,cellLoc="center",
                  loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        if r==0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white",fontweight="bold")
        elif r%2==0: cell.set_facecolor("#ecf0f1")
        if r>0 and c==4:  # unseen R² column
            try:
                v=float(rows[r-1][4])
                if v>=0.90: cell.set_facecolor("#c8e6c9")
                elif v>=0.80: cell.set_facecolor("#fff9c4")
                else: cell.set_facecolor("#ffcdd2")
            except Exception: pass
        cell.set_edgecolor("white")
    ax.set_title(f"Publication Summary Table | v4 Distributed Spatial AI\n"
                 f"Spatial Holdout: {HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locations never seen in training)\n"
                 f"Green=Unseen R²≥0.90 | Yellow=0.80-0.90 | Red<0.80",
                 fontweight="bold",fontsize=12,pad=20)
    plt.tight_layout()
    plt.savefig(FIGS/"PP10_publication_summary_table.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP10 publication table")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  POST-PROCESSING v4 COMPLETE")
print("="*70)
if "unseen_R2" in site_df.columns:
    best=site_df.sort_values("unseen_R2",ascending=False).iloc[0]
    print(f"\n  Best on UNSEEN {HOLDOUT_SITE}:")
    print(f"    Model   : {best['Model']} [{best['Tier']}]")
    print(f"    Target  : {best['Target']}")
    print(f"    Seen R² : {best.get('seen_R2',float('nan')):.4f}")
    print(f"    Unseen R²: {best.get('unseen_R2',float('nan')):.4f}")
    print(f"    Gap     : {best.get('spatial_gap',float('nan')):.4f}")

figs=sorted(FIGS.glob("PP*.png"))
print(f"\n  {len(figs)} figures saved to: {FIGS}")
for f in figs: print(f"    {f.name}")
print(f"\n  Results: {RESULTS}/postprocess_v4_results.csv")
print(f"  Done   : {pd.Timestamp.now()}")
print("="*70)
