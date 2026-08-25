"""
recoverability_curves.py
Recoverability Curves per senior's suggestion.

WHAT IT DOES:
  For each model and each error margin epsilon:
    Recoverability(epsilon) = % of predictions where |pred - true| <= epsilon

  Shows: at what error tolerance does each model become operationally reliable?
  Key comparison: SEEN vs UNSEEN (Wetland) locations

  Also computes:
    - Recoverability at standard thresholds (0.5°C, 1°C, 2°C for temp)
    - Area Under Recoverability Curve (AURC) — single number summary
    - Crossover point: where model reaches 80% recoverability

RUN ON TALON:
  module load pytorch-py39-cuda11.8-gcc11/1.13.0
  python3 ~/recoverability_curves.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v4"
MODELS  = PROJECT / "models_v4" / "dl"
FIGS    = PROJECT / "figures_v4"

matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3})

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = torch.device("cpu")
except ImportError:
    print("Load modules first"); sys.exit(1)

print("="*65)
print("  RECOVERABILITY CURVES — per senior suggestion")
print("="*65)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl","rb") as f: SC=pickle.load(f)
with open(PREPROC/"feature_info.pkl","rb") as f: FI=pickle.load(f)

LOCATIONS        = pd.DataFrame(FI["LOCATIONS"])
N_LOCS           = FI["N_LOCS"]
SNAP_FEATURES    = FI["SNAP_FEATURES"]
TEMP_TARGETS     = FI["TEMP_TARGETS"]
SMAP_TARGETS     = FI["SMAP_TARGETS"]
MOIST_TARGETS    = FI["MOIST_TARGETS"]
SITES            = FI["SITES"]
snap_feat_scaler = SC["snap_feat_scaler"]

ALL_TARGETS  = TEMP_TARGETS+SMAP_TARGETS+MOIST_TARGETS
APPROX_FEATS = [f"{t}_approx" for t in ALL_TARGETS if f"{t}_approx" in df.columns]
V4_FEATURES  = list(dict.fromkeys(SNAP_FEATURES+APPROX_FEATS))
V4_FEATURES  = [f for f in V4_FEATURES if f in df.columns]
N_V4_FEATURES= len(V4_FEATURES)

from sklearn.preprocessing import RobustScaler
tr_all = df[df["split"]=="train"]
v4_feat_scaler = RobustScaler()
v4_feat_scaler.fit(tr_all[V4_FEATURES].fillna(0).values)

v4_tgt_scalers = {}
for grp,tgts in [("temp",TEMP_TARGETS),("smap",SMAP_TARGETS),("moist",MOIST_TARGETS)]:
    av=[c for c in tgts if c in tr_all.columns]
    if not av: continue
    ts=RobustScaler(); ts.fit(tr_all[av].dropna().values)
    v4_tgt_scalers[grp]=ts

# ── Spatial setup ─────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree
HOLDOUT_SITE   = "Wetland"
TRAINING_SITES = [s for s in SITES if s!=HOLDOUT_SITE]
loc_to_idx = {(float(r.Latitude),float(r.Longitude)):i
              for i,r in LOCATIONS.iterrows()}

def get_site_idxs(site):
    sd=df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    return sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                   for _,r in sd.iterrows()
                   if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

SEEN_LOCS   = sorted(set(i for s in TRAINING_SITES for i in get_site_idxs(s)))
UNSEEN_LOCS = get_site_idxs(HOLDOUT_SITE)
site_idxs   = {s: get_site_idxs(s) for s in SITES}

coords = LOCATIONS[["Latitude","Longitude"]].values.astype(np.float32)
scaled = coords*np.array([111.0,63.0],dtype=np.float32)
tree   = cKDTree(scaled)
dists,idxs = tree.query(scaled,k=min(7,N_LOCS))
sigma  = np.median(dists[:,1:])+1e-8
A      = np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1,dists.shape[1]):
        j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
        A[i,j]+=w; A[j,i]+=w
A+=np.eye(N_LOCS); D=A.sum(1,keepdims=True)**0.5
A_norm=torch.tensor((A/(D*D.T+1e-8)).astype(np.float32))

# ── Model classes (minimal for loading checkpoints) ───────────────────────────
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
    nf=N_V4_FEATURES; h=96; dp=0.1
    try:
        import importlib.util
        spec=importlib.util.spec_from_file_location("v4m",str(PROJECT/"train_soil_spatial_v4.py"))
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.ARCH_MAP[arch](nt)
    except Exception as e:
        print(f"  ✗ Cannot load {arch}: {e} — using generic")
        class Generic(nn.Module):
            def __init__(self):
                super().__init__()
                self.p=nn.Linear(nf,h); self.g=nn.GRU(h,h,2,batch_first=True,bidirectional=True,dropout=dp)
                self.r=nn.Linear(h*2,h); self.gcn=nn.ModuleList([GraphConv(h,h,dp),GraphConv(h,h,dp)])
                self.hd=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Dropout(dp),nn.Linear(h,nt))
            def forward(self,x,A):
                B,L,N,F=x.shape; h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
                h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); h0=h
                for g in self.gcn: h=g(h,A)
                return self.hd(torch.cat([h0,h],dim=-1))
        return Generic()

# ── Evaluation dataset ────────────────────────────────────────────────────────
from torch.utils.data import Dataset, DataLoader

class EvalDS(Dataset):
    def __init__(self, tgt_cols, tgt_scaler, lookback=24, stride=24):
        self.A=A_norm; N=N_LOCS; nf=N_V4_FEATURES; nt=len(tgt_cols)
        sub=df[df["split"]=="test"].copy()
        all_ts=sorted(sub["time_utc"].unique()); T=len(all_ts)
        self.ts=all_ts
        if T<lookback+2: self.X=self.y=torch.zeros(0); return
        ts_to_i={ts:i for i,ts in enumerate(all_ts)}
        sub2=sub.copy()
        sub2["_ti"]=sub2["time_utc"].map(ts_to_i)
        sub2["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                     for la,lo in zip(sub2["Latitude"].astype(float),
                                      sub2["Longitude"].astype(float))]
        sub2=sub2.dropna(subset=["_ti","_ni"])
        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
        ti=sub2["_ti"].values; ni=sub2["_ni"].values
        Xf=np.full((T,N,nf),np.nan,dtype=np.float32)
        yf=np.full((T,N,nt),np.nan,dtype=np.float32)
        Xf[ti,ni,:]=v4_feat_scaler.transform(sub2[V4_FEATURES].fillna(0).values).astype(np.float32)
        yf[ti,ni,:]=tgt_scaler.transform(sub2[tgt_cols].fillna(0).values).astype(np.float32)
        tidxs=list(range(lookback,T,stride))
        Xl=[]; yl=[]
        for ti2 in tidxs:
            Xw=Xf[ti2-lookback:ti2]; yi=yf[ti2]
            if np.isnan(Xw).mean()>0.25: continue
            Xl.append(np.nan_to_num(Xw,nan=0.0)); yl.append(np.nan_to_num(yi,nan=0.0))
        if not Xl: self.X=self.y=torch.zeros(0); return
        self.X=torch.tensor(np.array(Xl)); self.y=torch.tensor(np.array(yl))
        self.tgt_scaler=tgt_scaler
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i],self.A


def compute_recoverability(errors, epsilons):
    """
    For each epsilon, compute % of predictions within epsilon.
    errors: 1D array of |pred - true| values
    """
    errors = np.abs(errors[~np.isnan(errors)])
    return np.array([100.0*(errors<=e).mean() for e in epsilons])


def area_under_curve(epsilons, recov):
    """Trapezoidal AURC — higher is better."""
    return float(np.trapz(recov, epsilons) / (epsilons[-1]-epsilons[0]))


# ── Main loop ─────────────────────────────────────────────────────────────────
ARCHES = ["BiGRU_NoGCN","GCN_NoTemporal","DeepESN","SpatialESN",
          "GraphSAGE","GAT","STGCN",
          "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]
TIERS  = {"BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
          "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
          "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
          "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
          "SpatialS4":"SSM","SpatialFuseMoE":"SSM"}
TIER_COLORS={"ABLATION":"#d62728","RESERVOIR":"#9467bd",
             "GRAPH":"#2ca02c","SSM":"#1f77b4"}
TGT_MAP={"temp":TEMP_TARGETS,"smap":SMAP_TARGETS,"moist":MOIST_TARGETS}
TGT_LABELS={"temp":"Weather Temp (°C)","smap":"SMAP Temp L1 (K)","moist":"Moisture (m³/m³)"}

# Error margins — physical units after inverse scaling
# For temp: 0.5°C to 5°C
# For smap: 0.5K to 5K
# For moist: 0.01 to 0.20 m³/m³
EPSILONS = {
    "temp" : np.linspace(0.1, 5.0, 50),
    "smap" : np.linspace(0.1, 5.0, 50),
    "moist": np.linspace(0.005, 0.20, 50),
}

all_recov_records = []
recov_curves      = {}   # (arch, tgt, split) → recov array

print("\n" + "="*60)
print("  COMPUTING RECOVERABILITY CURVES")
print("="*60)

for tgt_name, tgt_cols in TGT_MAP.items():
    tgt_sc = v4_tgt_scalers.get(tgt_name)
    if tgt_sc is None: continue
    av_tgt = [c for c in tgt_cols if c in df.columns]
    if not av_tgt: continue
    eps = EPSILONS[tgt_name]

    ds = EvalDS(av_tgt, tgt_sc)
    if len(ds)==0: continue
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    print(f"\n  Target: {TGT_LABELS[tgt_name]}")

    for arch in ARCHES:
        ckpt = MODELS/f"{arch}_{tgt_name}_v4_best.pt"
        if not ckpt.exists():
            print(f"  ✗ {arch} — missing"); continue
        try:
            sv    = torch.load(ckpt, map_location=DEVICE)
            model = make_model(arch, len(av_tgt))
            model.load_state_dict(sv["state_dict"]); model.eval()
            is_moe = (arch=="SpatialFuseMoE")

            yt_all=[]; yp_all=[]
            with torch.no_grad():
                for X,y,A_ in loader:
                    out=model(X,A_); pred=out[0] if is_moe else out
                    B_,N_,T_=pred.shape
                    pr=tgt_sc.inverse_transform(pred.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
                    yr=tgt_sc.inverse_transform(y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
                    yt_all.append(yr); yp_all.append(pr)

            yt=np.concatenate(yt_all,0); yp=np.concatenate(yp_all,0)
            errors_all    = (yp-yt)[:,:,0].flatten()
            errors_seen   = (yp-yt)[:,SEEN_LOCS,0].flatten()
            errors_unseen = (yp-yt)[:,UNSEEN_LOCS,0].flatten()

            # Compute recoverability curves
            rc_all    = compute_recoverability(errors_all,    eps)
            rc_seen   = compute_recoverability(errors_seen,   eps)
            rc_unseen = compute_recoverability(errors_unseen, eps)

            recov_curves[(arch,tgt_name,"all")]    = rc_all
            recov_curves[(arch,tgt_name,"seen")]   = rc_seen
            recov_curves[(arch,tgt_name,"unseen")] = rc_unseen

            # AURC and crossover point (where recov >= 80%)
            aurc_all    = area_under_curve(eps, rc_all)
            aurc_seen   = area_under_curve(eps, rc_seen)
            aurc_unseen = area_under_curve(eps, rc_unseen)

            # Crossover: min epsilon where recov >= 80%
            def crossover(rc, eps, threshold=80.0):
                idx = np.where(rc >= threshold)[0]
                return float(eps[idx[0]]) if len(idx)>0 else float("nan")

            co_all    = crossover(rc_all,    eps)
            co_seen   = crossover(rc_seen,   eps)
            co_unseen = crossover(rc_unseen, eps)

            all_recov_records.append(dict(
                Model=arch, Target=tgt_name, Tier=TIERS.get(arch,"?"),
                AURC_all=round(aurc_all,2), AURC_seen=round(aurc_seen,2),
                AURC_unseen=round(aurc_unseen,2),
                Crossover80_all=round(co_all,4) if not np.isnan(co_all) else None,
                Crossover80_seen=round(co_seen,4) if not np.isnan(co_seen) else None,
                Crossover80_unseen=round(co_unseen,4) if not np.isnan(co_unseen) else None,
                Units={"temp":"°C","smap":"K","moist":"m³/m³"}[tgt_name]))

            print(f"  ✓ {arch:<20} AURC_seen={aurc_seen:.1f} "
                  f"AURC_unseen={aurc_unseen:.1f} "
                  f"CO80_seen={co_seen:.3f} CO80_unseen={co_unseen:.3f}")

        except Exception as e:
            print(f"  ✗ {arch}: {e}")
            import traceback; traceback.print_exc()

# ── Save CSV ──────────────────────────────────────────────────────────────────
recov_df = pd.DataFrame(all_recov_records)
recov_df.to_csv(RESULTS/"recoverability_results.csv", index=False)
print(f"\n  ✓ recoverability_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating recoverability figures...")

for tgt_name in ["temp","smap","moist"]:
    eps = EPSILONS[tgt_name]
    unit = {"temp":"°C","smap":"K","moist":"m³/m³"}[tgt_name]

    # ── RC01: Recoverability curves — ALL locations ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    split_labels = [("all","All 256 locations","#333333"),
                    ("seen",f"Seen ({len(SEEN_LOCS)} locations)","#1f77b4"),
                    ("unseen",f"Unseen — {HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locs)","#d62728")]

    for ax, (split, split_lbl, _) in zip(axes, split_labels):
        plotted = 0
        for arch in ARCHES:
            key = (arch, tgt_name, split)
            if key not in recov_curves: continue
            rc  = recov_curves[key]
            col = TIER_COLORS.get(TIERS.get(arch,"?"), "grey")
            ls  = "--" if TIERS.get(arch)=="ABLATION" else "-"
            ax.plot(eps, rc, lw=2, alpha=0.85, color=col, ls=ls,
                    label=f"[{TIERS.get(arch,'?')}] {arch}")
            plotted += 1

        ax.axhline(80, color="orange", ls="--", lw=1.5, alpha=0.8,
                   label="80% recoverability threshold")
        ax.set_xlabel(f"Acceptable Error Margin ({unit})", fontsize=11)
        ax.set_ylabel("Recoverability (%)", fontsize=11)
        ax.set_title(f"{split_lbl}", fontweight="bold", fontsize=12)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7, ncol=1, loc="lower right")
        if plotted == 0:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,
                    ha="center",va="center",fontsize=14)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
        loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5,-0.03))
    fig.suptitle(
        f"Recoverability Curves | {TGT_LABELS[tgt_name]}\n"
        f"v4 Distributed Spatial AI | Wetland Holdout | Test 2025",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(FIGS/f"RECOV_01_curves_{tgt_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ RECOV_01_curves_{tgt_name}.png")

    # ── RC02: AURC heatmap ────────────────────────────────────────────────────
    sub = recov_df[recov_df["Target"]==tgt_name]
    if sub.empty: continue
    pv = sub.set_index("Model")[["AURC_seen","AURC_unseen"]].round(1)
    pv.columns = [f"Seen ({len(SEEN_LOCS)} locs)",
                  f"Unseen-{HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locs)"]
    pv = pv.reindex([m for m in ARCHES if m in pv.index])
    fig, ax = plt.subplots(figsize=(12, max(8, len(pv)*0.7+2)))
    sns_data = pv.astype(float)
    import seaborn as sns
    sns.heatmap(sns_data, ax=ax, cmap="RdYlGn",
                vmin=0, vmax=100,
                annot=True, fmt=".1f", linewidths=0.5,
                linecolor="white",
                annot_kws={"size":12,"weight":"bold"},
                cbar_kws={"label":"AURC (Area Under Recoverability Curve)","shrink":0.85})
    ax.set_yticklabels([f"[{TIERS.get(m,'?')}] {m}" for m in pv.index],
                       rotation=0, fontsize=9)
    ax.set_xticklabels(pv.columns, rotation=15, fontsize=10)
    ax.set_title(
        f"Area Under Recoverability Curve (AURC) | {TGT_LABELS[tgt_name]}\n"
        f"Higher = more recoverable predictions | "
        f"Unseen = {HOLDOUT_SITE} spatial holdout",
        fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS/f"RECOV_02_aurc_heatmap_{tgt_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ RECOV_02_aurc_heatmap_{tgt_name}.png")

    # ── RC03: Crossover point bar (when does model reach 80% recov?) ──────────
    sub2 = recov_df[recov_df["Target"]==tgt_name].dropna(subset=["Crossover80_seen"])
    if sub2.empty: continue
    sub2 = sub2.sort_values("Crossover80_seen", ascending=True)
    fig, axes2 = plt.subplots(1,2,figsize=(20,9))
    for ax, col, lbl in [
        (axes2[0],"Crossover80_seen",  f"Seen ({len(SEEN_LOCS)} locs)"),
        (axes2[1],"Crossover80_unseen",f"Unseen — {HOLDOUT_SITE}")]:
        if col not in sub2.columns: continue
        sub3 = sub2.dropna(subset=[col]).sort_values(col,ascending=True)
        colors=[TIER_COLORS.get(TIERS.get(m,"?"),"grey") for m in sub3["Model"]]
        bars=ax.barh(sub3["Model"],sub3[col],color=colors,
                     alpha=0.85,edgecolor="black",lw=0.5)
        for bar,v in zip(bars,sub3[col]):
            ax.text(v+0.01,bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}{unit}",va="center",fontsize=9,fontweight="bold")
        ax.set_xlabel(f"Error margin for 80% recoverability ({unit})",fontsize=11)
        ax.set_title(f"80% Recoverability Crossover\n{lbl}",
                     fontweight="bold",fontsize=12)
        ax.axvline(1.0,color="green",ls="--",lw=1.5,alpha=0.7,
                   label=f"1{unit} operational threshold")
        ax.legend(fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
               loc="lower center",ncol=4,fontsize=10,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(
        f"When Does Each Model Reach 80% Recoverability? | {TGT_LABELS[tgt_name]}\n"
        f"Smaller = more precise | Seen vs Unseen (Wetland) comparison",
        fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(FIGS/f"RECOV_03_crossover_{tgt_name}.png",
                dpi=150,bbox_inches="tight")
    plt.close()
    print(f"  ✓ RECOV_03_crossover_{tgt_name}.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RECOVERABILITY SUMMARY")
print("="*65)
if len(recov_df)>0:
    print(f"\n  {'Model':<20} {'Target':<6} {'AURC_seen':>10} "
          f"{'AURC_unseen':>12} {'CO80_seen':>10} {'CO80_unseen':>12}")
    print("  " + "─"*72)
    for _,r in recov_df.sort_values(["Target","AURC_unseen"],ascending=[True,False]).iterrows():
        co_s = f"{r['Crossover80_seen']:.3f}"   if r['Crossover80_seen']  else "N/A"
        co_u = f"{r['Crossover80_unseen']:.3f}" if r['Crossover80_unseen'] else "N/A"
        print(f"  {r['Model']:<20} {r['Target']:<6} {r['AURC_seen']:>10.1f} "
              f"{r['AURC_unseen']:>12.1f} {co_s:>10} {co_u:>12} {r['Units']}")
print(f"\n  Saved: {RESULTS}/recoverability_results.csv")
print(f"  Figures: RECOV_01/02/03 in {FIGS}")
print(f"  Done: {pd.Timestamp.now()}")
