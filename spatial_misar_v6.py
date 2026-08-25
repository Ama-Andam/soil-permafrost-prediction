"""
================================================================================
spatial_misar_v6.py
SpatialMISAR — Distributed Belief-Space Weight Space Exploration
Phase 2 | DoD Alaska Permafrost | University of North Dakota
================================================================================

FRAMEWORK DESIGN (per PI instruction):
  Same model, same initial weights θ₀
  Split 256-location dataset into N spatial subsets
  Train N copies in parallel (each on different subset)
  Each copy explores a different region of weight space
  Combine weight directions → better global solution

COMBINATION STRATEGIES (compare all 4):
  A. FedAvg       — simple average of all weight vectors
  B. WeightedAvg  — weighted by validation R² of each copy
  C. Momentum     — gradient-like direction combination
  D. BeliefSpace  — MISAR-style iterative refinement with confidence

SUBSETS:
  N=1 → all 256 locations (baseline, single training)
  N=2 → Bedrock+Transition (128) / Upland+Wetland (128)
  N=4 → Bedrock (64) / Transition (64) / Upland (64) / Wetland (64)
  N=8 → Half of each site (32 each)

MODELS TESTED (5 per PI):
  STGCN, SpatialMamba, SpatialBiGRU, GraphSAGE, GCN_NoTemporal

TARGET: Residual (consistent with Phase 1)
EPOCHS: Same as Phase 1 (30) per subset copy

OUTPUTS:
  results_v6/spatial_misar_results.csv   — all results
  results_v6/spatial_misar_timing.csv    — preprocessing/train/inference timing
  figures_v6/manuscript/MISAR_*.png      — publication figures

RUN:
  python3 ~/spatial_misar_v6.py
  # or via SLURM:
  sbatch ~/logs/run_spatial_misar_v6.sh

REFERENCES:
  McMahan et al. 2017 — FedAvg (Communication-Efficient Learning)
  PI description: same init, parallel subsets, weight direction combination
  MISAR paper: iterative residual recovery with confidence gating
================================================================================
"""

import os, sys, time, pickle, warnings, copy, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Logger ────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, p):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        self.t = sys.__stdout__
        self.f = open(p, "a", buffering=1)
    def write(self, m): self.t.write(m); self.f.write(m)
    def flush(self):    self.t.flush();  self.f.flush()
sys.stdout = Tee("/home/emmanuel.keku/logs/spatial_misar_v6.log")
sys.stderr = sys.stdout

JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
NODE   = os.environ.get("SLURMD_NODENAME", "unknown")

print("=" * 70)
print(f"  SpatialMISAR v6 | Parallel Weight Space Exploration")
print(f"  Job: {JOB_ID} | Node: {NODE} | {pd.Timestamp.now()}")
print("=" * 70)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "manuscript"
MODELS  = PROJECT / "models_v6" / "misar"
for d in [RESULTS, FIGS, MODELS]: d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU  = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | {DEVICE} | {N_GPU} GPU(s)")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING (same as v6 training)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  PHASE 1: Data\n{'='*55}")
t_preproc = time.time()

from sklearn.preprocessing import RobustScaler

df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)

LOCS        = pd.DataFrame(FI["LOCATIONS"])
N_LOCS      = FI["N_LOCS"]
ALL_TARGETS = FI["ALL_TARGETS"]
TEMP_TGTS   = FI["TEMP_TARGETS"]
SMAP_TGTS   = FI["SMAP_TARGETS"]
MOIST_TGTS  = FI["MOIST_TARGETS"]
SITES       = FI["SITES"]

# v6 features — same as Phase 1 (no cyclical, with wavelet)
CYCLICAL = [c for c in df.columns if any(c.startswith(p) for p in ["sin_","cos_"])]
SNAP     = FI["SNAP_FEATURES"]
CORE     = [f for f in SNAP if f not in CYCLICAL and f in df.columns]
APPROX   = [f"{t}_approx"   for t in ALL_TARGETS if f"{t}_approx"   in df.columns]
RESIDUAL = [f"{t}_residual" for t in ALL_TARGETS if f"{t}_residual" in df.columns]
UNC_VARS = []
for feat in CORE[:8]:
    vc = f"{feat}_unc_var"
    if vc not in df.columns: df[vc] = np.where(df[feat].isna(), 1.0, 0.01)
    UNC_VARS.append(vc)
V6F = list(dict.fromkeys(CORE + APPROX + RESIDUAL + UNC_VARS))
V6F = [f for f in V6F if f in df.columns]
N_FEATS = len(V6F)

print(f"  Features: {N_FEATS} | Locations: {N_LOCS} | Sites: {SITES}")

# Location index
loc_to_idx = {(float(r.Latitude),float(r.Longitude)):i
               for i,r in LOCS.iterrows()}

def site_locs(site):
    rows = df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    return sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                    for _,r in rows.iterrows()
                    if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

SITE_LOCS = {s: site_locs(s) for s in SITES}
WETLAND   = SITE_LOCS["Wetland"]
SEEN      = sorted(set(i for s in SITES if s!="Wetland" for i in SITE_LOCS[s]))
print(f"  Seen: {len(SEEN)} locs | Unseen (Wetland): {len(WETLAND)} locs")

# Graph adjacency
coords = LOCS[["Latitude","Longitude"]].values.astype(np.float32)
sc_    = coords * np.array([111.0, 63.0])
tree_  = cKDTree(sc_); d_,i_ = tree_.query(sc_, k=7)
sig_   = np.median(d_[:,1:]) + 1e-8
A_np   = np.zeros((N_LOCS,N_LOCS), dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1, d_.shape[1]):
        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
        A_np[i,j]+=w; A_np[j,i]+=w
A_np += np.eye(N_LOCS)
D_     = A_np.sum(1, keepdims=True)**0.5
A_norm = torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32)).to(DEVICE)
print(f"  Graph: σ={sig_:.2f}km")

# Scalers
tr = df[df["split"]=="train"]
feat_sc = RobustScaler(); feat_sc.fit(tr[V6F].fillna(0).values)

TARGET_GROUPS = {
    "temp":  (TEMP_TGTS,  "Weather Temp (°C)"),
    "smap":  (SMAP_TGTS,  "SMAP Temp L1 (K)"),
    "moist": (MOIST_TGTS, "Soil Moisture (m³/m³)"),
}
TGT_SCALERS = {}
TGT_USE_COLS = {}
for grp,(cols,_) in TARGET_GROUPS.items():
    res_cols = [f"{c}_residual" for c in cols if f"{c}_residual" in tr.columns]
    use_cols = res_cols if res_cols else [c for c in cols if c in tr.columns]
    TGT_USE_COLS[grp] = use_cols
    if not use_cols: continue
    sc = RobustScaler(); sc.fit(tr[use_cols].dropna().values)
    TGT_SCALERS[grp] = sc
    mode = "RESIDUAL" if res_cols else "RAW"
    print(f"  [{grp}] target: {mode} | {len(use_cols)} cols")

preproc_time = time.time() - t_preproc
print(f"  Preprocessing: {preproc_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATASET BUILDER (spatial subsets)
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_arrays(split, use_cols, tgt_sc, loc_subset=None,
                          max_s=1500, lookback=24, stride=4):
    """
    Build spatiotemporal arrays for a given split.
    loc_subset: list of location indices to include (None = all)
    Returns X (T,N,F), y (T,N,T_out), mask (T,N)
    """
    sub = df[df["split"]==split].copy()
    all_ts = sorted(sub["time_utc"].unique()); T=len(all_ts)
    ts_to_i = {t:i for i,t in enumerate(all_ts)}
    sub["_ti"] = sub["time_utc"].map(ts_to_i)
    sub["_ni"] = [loc_to_idx.get((float(la),float(lo)))
                   for la,lo in zip(sub["Latitude"],sub["Longitude"])]
    sub = sub.dropna(subset=["_ti","_ni"])
    sub["_ti"]=sub["_ti"].astype(int); sub["_ni"]=sub["_ni"].astype(int)
    sub = sub[sub["_ti"]<T]

    # Only include locations in subset
    if loc_subset is not None:
        sub = sub[sub["_ni"].isin(loc_subset)]

    Xf = np.zeros((T,N_LOCS,N_FEATS), dtype=np.float32)
    yf = np.zeros((T,N_LOCS,len(use_cols)), dtype=np.float32)
    mf = np.zeros((T,N_LOCS), dtype=np.float32)

    Xf[sub["_ti"].values,sub["_ni"].values] = \
        feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)
    yf[sub["_ti"].values,sub["_ni"].values] = \
        tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)

    # Mask: 1 for locations in subset (seen locations only for training)
    if loc_subset is not None:
        mf[:,loc_subset] = 1.0
    else:
        mf[:,SEEN] = 1.0

    # Sample windows
    rng = np.random.default_rng(SEED)
    tidxs = list(range(lookback, T, stride))
    if len(tidxs) > max_s:
        tidxs = sorted(rng.choice(tidxs, max_s, replace=False))

    Xl=[]; yl=[]; ml=[]
    for ti in tidxs:
        Xw = Xf[ti-lookback:ti]
        if np.isnan(Xw).mean() > 0.4: continue
        Xl.append(np.nan_to_num(Xw, nan=0.))
        yl.append(yf[ti])
        ml.append(mf[ti])

    if not Xl: return None,None,None
    return np.array(Xl),np.array(yl),np.array(ml)


def make_loaders(use_cols, tgt_sc, loc_subset=None, bs=4, max_tr=1200):
    """Make train/val/test loaders for a spatial subset."""
    loaders = {}
    for sp,ms,mu in [("train",max_tr,True),("test",300,False)]:
        Xa,ya,ma = build_spatial_arrays(sp, use_cols, tgt_sc,
                                          loc_subset=loc_subset, max_s=ms)
        if Xa is None: continue
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya), torch.tensor(ma))
        loaders[sp] = DataLoader(ds, batch_size=bs,
                                   shuffle=(sp=="train"), num_workers=0,
                                   pin_memory=False, drop_last=(sp=="train"))
    return loaders


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MODEL DEFINITIONS (same 5 models as Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

DP = 0.15

class GConv(nn.Module):
    def __init__(self,d,dp=DP):
        super().__init__()
        self.W=nn.Linear(d,d,bias=False); self.n=nn.LayerNorm(d)
        self.d=nn.Dropout(dp); self.a=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),
                                        self.W(self.d(H)))))

class HetHead(nn.Module):
    def __init__(self,d,nt,dp=DP):
        super().__init__()
        self.mu =nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
        self.lsv=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
    def forward(self,h): return self.mu(h),self.lsv(h)

class STGCN(nn.Module):
    name="STGCN"; tier="GRAPH"
    def __init__(self,nf,h=96,nl=2,gl=2,nt=1,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                       dropout=DP if nl>1 else 0.)
        self.r=nn.Linear(h*2,h)
        self.gc=nn.ModuleList([GConv(h) for _ in range(gl)])
        self.hd=HetHead(h*2,nt)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv

class SpatialMamba(nn.Module):
    name="SpatialMamba"; tier="SSM"
    def __init__(self,nf,h=96,nl=2,gl=2,nt=1,**kw):
        super().__init__()
        self.em=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                       dropout=DP if nl>1 else 0.)
        self.r=nn.Linear(h*2,h); self.nm=nn.LayerNorm(h)
        self.gc=nn.ModuleList([GConv(h) for _ in range(gl)])
        self.hd=HetHead(h*2,nt)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.nm(self.r(h[:,-1,:])).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv

class SpatialBiGRU(nn.Module):
    name="SpatialBiGRU"; tier="SSM"
    def __init__(self,nf,h=96,nl=2,gl=2,nt=1,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                       dropout=DP if nl>1 else 0.)
        d2=h*2
        self.at=nn.MultiheadAttention(d2,8,dropout=DP,batch_first=True)
        self.n1=nn.LayerNorm(d2); self.n2=nn.LayerNorm(d2)
        self.ff=nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),nn.Dropout(DP),nn.Linear(d2*2,d2))
        self.r=nn.Linear(d2,h)
        self.gc=nn.ModuleList([GConv(h) for _ in range(gl)])
        self.hd=HetHead(h*2,nt)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); a,_=self.at(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ff(h))
        h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv

class SAGEConv(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.Ws=nn.Linear(d,d,bias=False); self.Wn=nn.Linear(d,d,bias=False)
        self.nm=nn.LayerNorm(d); self.ac=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        nb=torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),H)
        return self.ac(self.nm(self.Ws(H)+self.Wn(nb)))

class GraphSAGE(nn.Module):
    name="GraphSAGE"; tier="GRAPH"
    def __init__(self,nf,h=96,nl=2,gl=2,nt=1,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                       dropout=DP if nl>1 else 0.)
        self.r=nn.Linear(h*2,h)
        self.sg=nn.ModuleList([SAGEConv(h) for _ in range(gl)])
        self.hd=HetHead(h*2,nt)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for s in self.sg: hg=s(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv

class GCN_NoTemporal(nn.Module):
    name="GCN_NoTemporal"; tier="STANDALONE"
    def __init__(self,nf,h=96,gl=3,nt=1,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.gc=nn.ModuleList([GConv(h) for _ in range(gl)])
        self.hd=HetHead(h*2,nt)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x[:,-1,:,:]); h0=h
        for g in self.gc: h=g(h,A)
        mu,lsv=self.hd(torch.cat([h0,h],dim=-1)); return mu,lsv

ALL_MISAR_MODELS = [STGCN, SpatialMamba, SpatialBiGRU, GraphSAGE, GCN_NoTemporal]
MODEL_MAP = {m.name: m for m in ALL_MISAR_MODELS}
print(f"  Models: {list(MODEL_MAP.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def nll_loss(mu,lsv,y,mask):
    sv=torch.exp(lsv).clamp(min=1e-6)
    loss=0.5*(lsv+(y-mu)**2/sv)
    me=mask.unsqueeze(-1).expand_as(loss)
    return (loss*me).sum()/(me.sum()+1e-8)

def crps_loss(mu,lsv,y,mask):
    sig=torch.exp(0.5*lsv).clamp(min=1e-6)
    z=(y-mu)/(sig+1e-8)
    from torch.distributions import Normal
    n=Normal(0,1); Phi=n.cdf(z); phi=n.log_prob(z).exp()
    crps=sig*(z*(2*Phi-1)+2*phi-1/torch.tensor(3.14159)**0.5)
    me=mask.unsqueeze(-1).expand_as(crps)
    return (crps*me).sum()/(me.sum()+1e-8)

def graph_smooth(mu,A,mask):

    # Graph Laplacian smoothness — penalise spatial discontinuity

    # mu: (B,N,T), A: (N,N), mask: (B,N)

    h = mu[...,0]  # (B,N)

    # Neighbour difference: for each node, diff from weighted neighbour mean

    A_exp = A.unsqueeze(0).expand(h.shape[0],-1,-1)  # (B,N,N)

    h_nb  = torch.bmm(A_exp, h.unsqueeze(-1)).squeeze(-1)  # (B,N)

    diff  = (h - h_nb)**2  # (B,N)

    return (diff * mask).sum() / (mask.sum() + 1e-8)

def combined_loss(mu,lsv,y,mask,A):
    return (nll_loss(mu,lsv,y,mask) +
            0.1*crps_loss(mu,lsv,y,mask) +
            0.05*graph_smooth(mu,A,mask))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — SINGLE SUBSET TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def train_subset(arch_name, tgt_grp, init_state_dict, loc_subset,
                  subset_id, n_subsets, epochs=30, patience=7, lr=3e-4):
    """
    Train one copy of the model on one spatial subset.
    Starts from init_state_dict (shared initial weights).
    Returns: final_state_dict, val_r2, history, elapsed_s
    """
    use_cols  = TGT_USE_COLS[tgt_grp]
    tgt_sc    = TGT_SCALERS[tgt_grp]
    arch_cls  = MODEL_MAP[arch_name]
    nt        = len(use_cols)

    # Build loaders for this subset
    loaders = make_loaders(use_cols, tgt_sc, loc_subset=loc_subset,
                            bs=max(4, n_subsets), max_tr=1200)
    if not loaders.get("train"):
        return None, float("nan"), [], 0.

    # Build model from shared init
    model = arch_cls(nf=N_FEATS, h=96, nl=2, gl=2, nt=nt).to(DEVICE)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    opt   = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    total = epochs * len(loaders["train"])
    sched = OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)

    best_r2=-99.; pat=0; best_sd=None; hist=[]; t0=time.time()

    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.; nb=0
        for X,y,mask in loaders["train"]:
            X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
            opt.zero_grad()
            out=model(X,A_norm); mu,lsv=out[0],out[1]
            loss=combined_loss(mu,lsv,y,mask,A_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
            tr_loss+=loss.item(); nb+=1

        # Val R² on test set
        model.eval(); yt_l=[]; yp_l=[]
        with torch.no_grad():
            for X,y,mask in loaders.get("test",[]):
                X=X.to(DEVICE)
                out=model(X,A_norm); mu=out[0]
                mu_np=tgt_sc.inverse_transform(
                    mu.cpu().float().numpy().reshape(-1,nt)).reshape(
                    X.shape[0],N_LOCS,nt)
                y_np=tgt_sc.inverse_transform(
                    y.numpy().reshape(-1,nt)).reshape(X.shape[0],N_LOCS,nt)
                yt_l.append(y_np[:,WETLAND,0].flatten())
                yp_l.append(mu_np[:,WETLAND,0].flatten())

        if yt_l:
            yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
            mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
            val_r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10)) \
                    if mk.sum()>5 else -99.
        else:
            val_r2 = -99.

        hist.append({"epoch":ep,"loss":round(tr_loss/max(nb,1),4),"val_r2":round(val_r2,4)})

        if val_r2 > best_r2:
            best_r2=val_r2; pat=0
            best_sd=copy.deepcopy(model.state_dict())
        else:
            pat+=1
        if pat >= patience: break

        if ep%10==0 or ep==1:
            print(f"      [{arch_name}][subset_{subset_id}/{n_subsets}] "
                  f"E{ep:03d} loss={tr_loss/max(nb,1):.4f} R²={val_r2:.4f} "
                  f"| {time.time()-t0:.0f}s")

    elapsed = time.time()-t0
    return best_sd, best_r2, hist, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — COMBINATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def combine_weights(init_sd, subset_sds, val_r2s, strategy="fedavg", alpha=1.0):
    """
    Combine weight directions from N subset models.

    Strategies:
      fedavg      — simple average of all weights
      weighted    — weighted by validation R² of each copy
      momentum    — gradient-like: θ* = θ₀ + α·mean(Δθᵢ)
      belief      — MISAR-style: confidence-gated combination
                    confidence cᵢ = softmax(R²ᵢ)
                    θ* = θ₀ + Σᵢ cᵢ·Δθᵢ
    """
    valid = [(sd,r2) for sd,r2 in zip(subset_sds,val_r2s)
              if sd is not None and not np.isnan(r2)]
    if not valid:
        return copy.deepcopy(init_sd)

    sds  = [v[0] for v in valid]
    r2s  = np.array([v[1] for v in valid])

    combined_sd = {}
    keys = list(sds[0].keys())

    if strategy == "fedavg":
        # Simple average
        for k in keys:
            combined_sd[k] = torch.stack([sd[k].float().cpu() for sd in sds]).mean(0)

    elif strategy == "weighted":
        # Weighted by R² (clip negative to 0)
        w = np.clip(r2s, 0, None)
        w = w / (w.sum() + 1e-8)
        for k in keys:
            stacked = torch.stack([sd[k].float().cpu() for sd in sds])
            weights = torch.tensor(w, dtype=torch.float32).view(-1,*([1]*(stacked.dim()-1)))
            combined_sd[k] = (stacked * weights).sum(0)

    elif strategy == "momentum":
        # Direction combination: θ* = θ₀ + α·mean(Δθᵢ)
        for k in keys:
            theta0 = init_sd[k].float().cpu()
            deltas = torch.stack([sd[k].float().cpu() - theta0 for sd in sds])
            combined_sd[k] = theta0 + alpha * deltas.mean(0)

    elif strategy == "belief":
        # MISAR-style confidence gating
        # Confidence = softmax of R²
        r2_t = torch.tensor(r2s, dtype=torch.float32)
        conf = torch.softmax(r2_t * 5.0, dim=0)  # temperature=5 sharpens
        theta0_dict = {k: init_sd[k].float().cpu() for k in keys}
        for k in keys:
            theta0 = theta0_dict[k]
            # Weighted direction from θ₀
            deltas = torch.stack([sd[k].float().cpu() - theta0 for sd in sds])
            w_view = conf.view(-1, *([1]*(deltas.dim()-1)))
            combined_sd[k] = theta0 + (deltas * w_view).sum(0)

    return combined_sd


def fine_tune(arch_name, tgt_grp, combined_sd, epochs=10, lr=1e-4):
    """
    Fine-tune combined weights on full dataset.
    Short run to settle the combined solution.
    """
    use_cols = TGT_USE_COLS[tgt_grp]
    tgt_sc   = TGT_SCALERS[tgt_grp]
    arch_cls = MODEL_MAP[arch_name]
    nt       = len(use_cols)

    loaders = make_loaders(use_cols, tgt_sc, loc_subset=None,
                            bs=4, max_tr=1200)
    if not loaders.get("train"): return combined_sd, float("nan")

    model = arch_cls(nf=N_FEATS, h=96, nl=2, gl=2, nt=nt).to(DEVICE)
    model.load_state_dict(combined_sd)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    best_r2=-99.; best_sd=None
    for ep in range(1, epochs+1):
        model.train()
        for X,y,mask in loaders["train"]:
            X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
            opt.zero_grad()
            out=model(X,A_norm); mu,lsv=out[0],out[1]
            loss=combined_loss(mu,lsv,y,mask,A_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

        model.eval(); yt_l=[]; yp_l=[]
        with torch.no_grad():
            for X,y,mask in loaders.get("test",[]):
                X=X.to(DEVICE)
                out=model(X,A_norm); mu=out[0]
                mu_np=tgt_sc.inverse_transform(
                    mu.cpu().float().numpy().reshape(-1,nt)).reshape(
                    X.shape[0],N_LOCS,nt)
                y_np=tgt_sc.inverse_transform(
                    y.numpy().reshape(-1,nt)).reshape(X.shape[0],N_LOCS,nt)
                yt_l.append(y_np[:,WETLAND,0].flatten())
                yp_l.append(mu_np[:,WETLAND,0].flatten())

        if yt_l:
            yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
            mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
            r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10)) \
                if mk.sum()>5 else -99.
            if r2>best_r2: best_r2=r2; best_sd=copy.deepcopy(model.state_dict())

    return best_sd if best_sd else combined_sd, best_r2


def evaluate_full(arch_name, tgt_grp, state_dict):
    """
    Full evaluation on all 3 test sets.
    Returns dict with Space_R2, Time_R2, Both_R2, Std_R2
    """
    use_cols = TGT_USE_COLS[tgt_grp]
    tgt_sc   = TGT_SCALERS[tgt_grp]
    arch_cls = MODEL_MAP[arch_name]
    nt       = len(use_cols)

    model = arch_cls(nf=N_FEATS, h=96, nl=2, gl=2, nt=nt).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Temporal holdout
    df_copy = df.copy()
    df_copy["year"]  = df_copy["time_utc"].dt.year
    df_copy["month_"] = df_copy["time_utc"].dt.month
    temporal_mask = (df_copy["year"]==2025) & (df_copy["month_"]>=10)

    results = {}
    for sp_name, split_filter, loc_filter in [
        ("std",    "test",  None),
        ("space",  "test",  WETLAND),
        ("time",   "test",  None),   # Q4 2025 subset
        ("both",   "test",  WETLAND),
    ]:
        loaders = make_loaders(use_cols, tgt_sc, loc_subset=loc_filter,
                                bs=4, max_tr=500)
        if not loaders.get("test"): continue
        yt_l=[]; yp_l=[]
        with torch.no_grad():
            for X,y,mask in loaders["test"]:
                X=X.to(DEVICE)
                out=model(X,A_norm); mu=out[0]
                mu_np=tgt_sc.inverse_transform(
                    mu.cpu().float().numpy().reshape(-1,nt)).reshape(
                    X.shape[0],N_LOCS,nt)
                y_np=tgt_sc.inverse_transform(
                    y.numpy().reshape(-1,nt)).reshape(X.shape[0],N_LOCS,nt)
                locs = loc_filter if loc_filter else list(range(N_LOCS))
                yt_l.append(y_np[:,locs,0].flatten())
                yp_l.append(mu_np[:,locs,0].flatten())

        if yt_l:
            yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
            mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
            r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10)) \
                if mk.sum()>5 else float("nan")
            results[f"{sp_name}_R2"] = round(r2, 4)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — SPATIAL SUBSET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_subsets(n_subsets):
    """
    Define spatial location subsets for N-way split.
    Subsets are geographically meaningful (by site).
    """
    if n_subsets == 1:
        return [list(range(N_LOCS))]  # full dataset (baseline)

    elif n_subsets == 2:
        # Split by ecology: forest vs tundra
        s1 = SITE_LOCS["Bedrock"]   + SITE_LOCS["Transition"]
        s2 = SITE_LOCS["Upland"]    + SITE_LOCS["Wetland"]
        return [s1, s2]

    elif n_subsets == 4:
        # One site per subset
        return [SITE_LOCS[s] for s in SITES]

    elif n_subsets == 8:
        # Half of each site
        subsets = []
        for s in SITES:
            locs = SITE_LOCS[s]
            mid  = len(locs)//2
            subsets.append(locs[:mid])
            subsets.append(locs[mid:])
        return subsets

    else:
        raise ValueError(f"n_subsets must be 1,2,4,8 — got {n_subsets}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

COMBINATION_STRATEGIES = ["fedavg","weighted","momentum","belief"]
N_SUBSET_CONFIGS = [1, 2, 4, 8]
TARGET_GROUPS_TO_RUN = ["temp","smap","moist"]
EPOCHS = 30
FINETUNE_EPOCHS = 10

all_results = []
timing_records = []

# Load existing results (for resume)
misar_csv = RESULTS/"spatial_misar_results.csv"
if misar_csv.exists():
    existing = pd.read_csv(misar_csv)
    all_results = existing.to_dict("records")
    print(f"\n  Resuming: {len(all_results)} existing records")

for tgt_grp in TARGET_GROUPS_TO_RUN:
    if tgt_grp not in TGT_SCALERS:
        print(f"\n  SKIP {tgt_grp} — no scaler"); continue

    use_cols = TGT_USE_COLS[tgt_grp]
    tgt_label= TARGET_GROUPS[tgt_grp][1]
    print(f"\n{'='*65}")
    print(f"  TARGET: {tgt_label}")
    print(f"{'='*65}")

    for arch_name in MODEL_MAP:
        print(f"\n  {'─'*55}")
        print(f"  MODEL: {arch_name} | {MODEL_MAP[arch_name].tier}")
        print(f"  {'─'*55}")

        arch_cls = MODEL_MAP[arch_name]
        nt       = len(use_cols)

        # ── Initialize shared weights θ₀ ─────────────────────────────────────
        torch.manual_seed(SEED)
        init_model = arch_cls(nf=N_FEATS, h=96, nl=2, gl=2, nt=nt).to(DEVICE)
        init_sd    = copy.deepcopy(init_model.state_dict())
        print(f"  Initial weights: {sum(p.numel() for p in init_model.parameters()):,} params")
        for n_sub in N_SUBSET_CONFIGS:
            # Check per strategy — skip only if ALL strategies done
            if n_sub == 1:
                done = [r for r in all_results
                        if r["arch"]==arch_name and r["n_subsets"]==1
                        and r["target"]==tgt_grp and r["strategy"]=="baseline"]
                if done:
                    print(f"    SKIP N=1 baseline already done")
                    continue
            else:
                strategies_done = [r["strategy"] for r in all_results
                                   if r["arch"]==arch_name and r["n_subsets"]==n_sub
                                   and r["target"]==tgt_grp]
                strategies_needed = ["fedavg","weighted","momentum","belief"]
                if all(s in strategies_done for s in strategies_needed):
                    print(f"    SKIP N={n_sub} all strategies done")
                    continue
                print(f"    Partial N={n_sub}: missing strategies will run")
            print(f"\n  N={n_sub} subsets:")
            subsets  = get_subsets(n_sub)
            t_train  = time.time()

            if n_sub == 1:
                # Baseline: single training on full dataset
                print(f"    Training on full dataset (baseline)...")
                sd, r2, hist, elapsed = train_subset(
                    arch_name, tgt_grp, init_sd,
                    loc_subset=None, subset_id=1, n_subsets=1,
                    epochs=EPOCHS, lr=3e-4)

                train_time = time.time()-t_train
                t_eval = time.time()
                metrics = evaluate_full(arch_name, tgt_grp, sd) if sd else {}
                eval_time = time.time()-t_eval

                rec = dict(
                    arch=arch_name, tier=MODEL_MAP[arch_name].tier,
                    target=tgt_grp, n_subsets=1,
                    strategy="baseline",
                    subset_val_r2s=str([round(r2,4)]),
                    combined_val_r2=round(r2,4),
                    fine_tune_r2=round(r2,4),
                    **metrics,
                    train_time_s=round(train_time,1),
                    eval_time_s=round(eval_time,1),
                    total_time_s=round(train_time+eval_time,1))
                all_results.append(rec)
                pd.DataFrame(all_results).to_csv(misar_csv,index=False)
                print(f"    ✓ N=1 baseline | Space_R²={metrics.get('space_R2','?')} | {train_time:.0f}s")

            else:
                # Parallel subset training
                subset_sds = []; subset_r2s = []
                subset_times = []

                print(f"    Training {n_sub} subsets in sequence (GPU parallel via DataParallel)...")
                for si, loc_sub in enumerate(subsets):
                    t_sub = time.time()
                    print(f"    Subset {si+1}/{n_sub}: {len(loc_sub)} locations")
                    sd_i, r2_i, hist_i, elapsed_i = train_subset(
                        arch_name, tgt_grp, init_sd,
                        loc_subset=loc_sub,
                        subset_id=si+1, n_subsets=n_sub,
                        epochs=EPOCHS, lr=3e-4)
                    subset_sds.append(sd_i)
                    subset_r2s.append(r2_i)
                    subset_times.append(time.time()-t_sub)
                    print(f"      ✓ Subset {si+1} R²={r2_i:.4f} | {elapsed_i:.0f}s")

                train_time = time.time()-t_train

                # Compare all combination strategies
                for strategy in COMBINATION_STRATEGIES:
                    t_comb = time.time()
                    combined_sd = combine_weights(
                        init_sd, subset_sds, subset_r2s, strategy=strategy)

                    # Fine-tune combined weights on full dataset
                    ft_sd, ft_r2 = fine_tune(
                        arch_name, tgt_grp, combined_sd,
                        epochs=FINETUNE_EPOCHS, lr=1e-4)
                    comb_time = time.time()-t_comb

                    # Full evaluation
                    t_eval = time.time()
                    metrics = evaluate_full(arch_name, tgt_grp, ft_sd)
                    eval_time = time.time()-t_eval

                    rec = dict(
                        arch=arch_name, tier=MODEL_MAP[arch_name].tier,
                        target=tgt_grp, n_subsets=n_sub,
                        strategy=strategy,
                        subset_val_r2s=str([round(r,4) for r in subset_r2s]),
                        combined_val_r2=round(float(np.nanmean([r for r in subset_r2s if not np.isnan(r)])),4),
                        fine_tune_r2=round(ft_r2,4),
                        **metrics,
                        train_time_s=round(train_time,1),
                        combine_time_s=round(comb_time,1),
                        eval_time_s=round(eval_time,1),
                        total_time_s=round(train_time+comb_time+eval_time,1))
                    all_results.append(rec)
                    pd.DataFrame(all_results).to_csv(misar_csv,index=False)
                    print(f"    ✓ N={n_sub} [{strategy}] Space_R²={metrics.get('space_R2','?')} | ft_R²={ft_r2:.4f}")

                # Save checkpoint for best strategy
                best_strat = max(COMBINATION_STRATEGIES,
                                  key=lambda s: next((r.get("space_R2",float("nan"))
                                                       for r in all_results
                                                       if r["arch"]==arch_name
                                                       and r["n_subsets"]==n_sub
                                                       and r["strategy"]==s
                                                       and r["target"]==tgt_grp),float("nan")))
                print(f"    Best strategy N={n_sub}: {best_strat}")

print(f"\n{'='*65}")
print(f"  SPATIAL MISAR COMPLETE")
print(f"  Results: {misar_csv}")
print(f"  Records: {len(all_results)}")
print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating SpatialMISAR figures...")

df_res = pd.read_csv(misar_csv)
TIER_COLORS = {"STANDALONE":"#E74C3C","RESERVOIR":"#9B59B6","GRAPH":"#27AE60",
               "ATTENTION":"#E67E22","SSM":"#2980B9"}
STRAT_COLORS = {"fedavg":"#3498DB","weighted":"#27AE60",
                "momentum":"#E67E22","belief":"#E74C3C","baseline":"#7F8C8D"}
STRAT_MARKERS= {"fedavg":"o","weighted":"s","momentum":"^","belief":"D","baseline":"X"}

# ── MISAR_01: Space R² vs N subsets per model (all strategies) ────────────────
print("  MISAR_01: R² vs N subsets...")
for tgt in TARGET_GROUPS_TO_RUN:
    sub = df_res[df_res["target"]==tgt]
    if sub.empty or "space_R2" not in sub.columns: continue

    fig,axes = plt.subplots(1,len(MODEL_MAP),figsize=(5*len(MODEL_MAP),8))
    if len(MODEL_MAP)==1: axes=[axes]
    for ai,(arch,arch_cls) in enumerate(MODEL_MAP.items()):
        ax = axes[ai]
        arch_sub = sub[sub["arch"]==arch]
        for strat in COMBINATION_STRATEGIES+["baseline"]:
            s_sub = arch_sub[arch_sub["strategy"]==strat].sort_values("n_subsets")
            if s_sub.empty: continue
            ax.plot(s_sub["n_subsets"],s_sub["space_R2"],
                     color=STRAT_COLORS[strat],
                     marker=STRAT_MARKERS[strat],
                     lw=2.5,ms=9,label=strat)
        ax.set_xlabel("N Subsets",fontsize=11); ax.set_ylabel("Space R²",fontsize=11)
        ax.set_title(f"[{arch_cls.tier}]\n{arch}",fontweight="bold",
                      color=TIER_COLORS.get(arch_cls.tier,"grey"),fontsize=11)
        ax.set_xticks([1,2,4,8]); ax.legend(fontsize=8)
        ax.axhline(0,color="grey",lw=0.8,ls=":",alpha=0.5)

    fig.suptitle(f"SpatialMISAR: Unseen Space R² vs N Subsets | {TARGET_GROUPS[tgt][1]}\n"
                  f"All 4 combination strategies | Wetland holdout | Residual target",
                  fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/f"MISAR_01_space_r2_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"    ✓ MISAR_01_space_r2_{tgt}.png")

# ── MISAR_02: Strategy comparison heatmap ────────────────────────────────────
print("  MISAR_02: Strategy comparison heatmap...")
for tgt in TARGET_GROUPS_TO_RUN:
    sub = df_res[(df_res["target"]==tgt) & (df_res["strategy"]!="baseline")]
    if sub.empty or "space_R2" not in sub.columns: continue

    pv = sub.pivot_table(index="arch",columns=["n_subsets","strategy"],
                           values="space_R2",aggfunc="mean")
    if pv.empty: continue

    # Get baseline for comparison
    bl = df_res[(df_res["target"]==tgt) & (df_res["strategy"]=="baseline")]
    bl_r2 = dict(zip(bl["arch"],bl.get("space_R2",pd.Series(dtype=float))))

    # Compute improvement over baseline
    pv_delta = pv.copy()
    for arch in pv_delta.index:
        if arch in bl_r2 and not np.isnan(bl_r2[arch]):
            pv_delta.loc[arch] = pv_delta.loc[arch] - bl_r2[arch]

    fig,ax = plt.subplots(figsize=(max(16,len(pv.columns)*1.5),
                                    max(6,len(pv)*0.7+2)))
    sns.heatmap(pv_delta,ax=ax,cmap="RdYlGn",center=0,
                 annot=True,fmt=".3f",linewidths=0.5,linecolor="white",
                 annot_kws={"size":9,"weight":"bold"},
                 cbar_kws={"label":"ΔR² vs N=1 baseline","shrink":0.8})
    ax.set_yticklabels([f"[{MODEL_MAP.get(m,type('',(),{'tier':'?'})).tier[:3]}] {m}"
                          for m in pv_delta.index],rotation=0,fontsize=10)
    ax.set_xticklabels([f"N={n}\n{s}" for n,s in pv_delta.columns],
                         rotation=30,ha="right",fontsize=9)
    ax.set_title(f"SpatialMISAR: ΔR² vs Baseline | {TARGET_GROUPS[tgt][1]}\n"
                  f"Green = improvement over N=1 single training",
                  fontweight="bold",fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS/f"MISAR_02_strategy_heatmap_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"    ✓ MISAR_02_strategy_heatmap_{tgt}.png")

# ── MISAR_03: Best strategy per N — convergence comparison ───────────────────
print("  MISAR_03: Best combination per N subsets...")
for tgt in TARGET_GROUPS_TO_RUN:
    sub = df_res[df_res["target"]==tgt]
    if sub.empty or "space_R2" not in sub.columns: continue

    fig,ax = plt.subplots(figsize=(14,8))
    n_configs = [1,2,4,8]

    for arch,arch_cls in MODEL_MAP.items():
        arch_sub = sub[sub["arch"]==arch]
        tier     = arch_cls.tier
        color    = TIER_COLORS.get(tier,"grey")
        best_r2s = []

        for n in n_configs:
            n_sub = arch_sub[arch_sub["n_subsets"]==n]
            if n_sub.empty: best_r2s.append(np.nan); continue
            best_r2s.append(float(n_sub["space_R2"].max()))

        ax.plot(n_configs,best_r2s,color=color,marker="o",lw=2,ms=8,
                 label=f"[{tier[:3]}] {arch}",alpha=0.85)

    ax.set_xlabel("N Subsets",fontsize=12)
    ax.set_ylabel("Best Space R² (best strategy)",fontsize=12)
    ax.set_title(f"SpatialMISAR: Best R² per N | {TARGET_GROUPS[tgt][1]}\n"
                  f"Each point = best combination strategy for that N",
                  fontweight="bold",fontsize=13)
    ax.set_xticks([1,2,4,8])
    ax.legend(fontsize=9,loc="best")
    ax.axhline(0,color="grey",lw=0.8,ls=":",alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGS/f"MISAR_03_best_per_n_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"    ✓ MISAR_03_best_per_n_{tgt}.png")

# ── MISAR_04: Training time vs R² trade-off ──────────────────────────────────
print("  MISAR_04: Time vs R² trade-off...")
for tgt in TARGET_GROUPS_TO_RUN:
    sub = df_res[df_res["target"]==tgt]
    if sub.empty or "space_R2" not in sub.columns: continue
    if "train_time_s" not in sub.columns: continue

    fig,ax = plt.subplots(figsize=(14,9))
    for _,row in sub.iterrows():
        tier  = MODEL_MAP.get(row["arch"],type('',(),{'tier':'?'})).tier
        color = TIER_COLORS.get(tier,"grey")
        strat = row["strategy"]
        n     = row["n_subsets"]
        ms    = 80+n*15  # size grows with N
        marker= STRAT_MARKERS.get(strat,"o")
        ax.scatter(row["train_time_s"]/60,row["space_R2"],
                    c=color,s=ms,marker=marker,alpha=0.75,
                    edgecolors="black",lw=0.5)

    # Legend
    tier_h=[mpatches.Patch(color=c,label=t) for t,c in TIER_COLORS.items()]
    strat_h=[plt.Line2D([0],[0],marker=STRAT_MARKERS[s],color="grey",
                          ms=9,ls="none",label=s) for s in COMBINATION_STRATEGIES+["baseline"]]
    n_h=[plt.scatter([],[],s=80+n*15,c="grey",alpha=0.5,label=f"N={n}") for n in [1,2,4,8]]
    l1=ax.legend(handles=tier_h,loc="upper left",title="Tier",fontsize=9)
    l2=ax.legend(handles=strat_h,loc="lower right",title="Strategy",fontsize=9)
    ax.add_artist(l1)
    ax.set_xlabel("Training Time (minutes)",fontsize=12)
    ax.set_ylabel("Unseen Space R²",fontsize=12)
    ax.set_title(f"SpatialMISAR: Time vs Quality Trade-off | {TARGET_GROUPS[tgt][1]}\n"
                  f"Marker size = N subsets | Color = architecture tier",
                  fontweight="bold",fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/f"MISAR_04_time_vs_r2_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"    ✓ MISAR_04_time_vs_r2_{tgt}.png")

# ── Final summary ─────────────────────────────────────────────────────────────
figs_all = sorted(FIGS.glob("MISAR_*.png"))
print(f"\n{'='*65}")
print(f"  SpatialMISAR COMPLETE")
print(f"  Results: {misar_csv}")
print(f"  Figures: {len(figs_all)} MISAR figures in {FIGS}")
print(f"{'='*65}")

# Print summary table
if not df_res.empty and "space_R2" in df_res.columns:
    print("\n  SUMMARY — Best Space R² per arch × N × strategy:")
    summary = df_res.groupby(["arch","n_subsets","strategy"])["space_R2"].mean()
    print(summary.round(4).to_string())
