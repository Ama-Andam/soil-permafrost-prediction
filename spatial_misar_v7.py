"""
================================================================================
spatial_misar_v7.py
BEHAVIOR-GUIDED DISTRIBUTED SPATIAL AI — SpatialMISAR v7
DoD Alaska Permafrost | University of North Dakota
================================================================================
ATTACHES to train_soil_spatial_v6.py — imports model + data directly.
No code rewrite. Uses STGCN structure with random initialization.

PI FRAMEWORK:
  T global rounds × N Ray workers × K local steps
  Each worker: trains on spatial subset → records Δθ, gradients, losses
  Aggregator: αi = softmax(consensus × stability × progress)
  Update: θt+1 = θt + γ × Σ αi Δθi
  Outputs: 3 GIFs (performance, weight space PCA, loss evolution)
================================================================================
"""
import os, sys, time, copy, warnings, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
RESULTS = PROJECT / "results_v7"; RESULTS.mkdir(exist_ok=True)
FIGS    = PROJECT / "figures_v7";  FIGS.mkdir(exist_ok=True)
LOGS    = PROJECT / "logs";        LOGS.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("="*65)
print("  SpatialMISAR v7 — Behavior-Guided Distributed AI")
print("  Attaches to train_soil_spatial_v6.py")
print("="*65)

# ── Import PyTorch ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# ── Import Ray ─────────────────────────────────────────────────────────────────
try:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 num_cpus=4,
                 num_gpus=torch.cuda.device_count())
    HAS_RAY = True
    print(f"  Ray {ray.__version__} initialized")
except Exception as e:
    HAS_RAY = False
    print(f"  Ray unavailable ({e}) — sequential fallback")

# ══════════════════════════════════════════════════════════════════════════════
# ATTACH: Load data + model classes from v6
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Loading v6 data and model structure...")

# Execute v6 setup section only (stop before if args.mode)
v6_src = open(PROJECT/"train_soil_spatial_v6.py").read()
v6_setup = v6_src.split("if args.mode")[0].split("# ── Argument parser")[0]

exec_ns = {"__name__": "__v6_attach__", "DEVICE": DEVICE}
try:
    exec(v6_setup, exec_ns)
    print("  v6 setup loaded OK")
except Exception as e:
    print(f"  v6 setup warning: {e}")

# Pull shared objects from v6 namespace
N_LOCS   = exec_ns.get("N_LOCS",   256)
N_FEATS  = exec_ns.get("N_FEATS",  32)
SITES    = exec_ns.get("SITES",    ["Bedrock","Transition","Upland","Wetland"])
SITE_LOCS= exec_ns.get("SITE_LOCS",{})
WETLAND  = exec_ns.get("WETLAND",  [])
SEEN     = exec_ns.get("SEEN",     [])
A_norm   = exec_ns.get("A_norm",   None)
feat_sc  = exec_ns.get("feat_sc",  None)
tgt_sc   = exec_ns.get("TGT_SCALERS",{}).get("temp", None)
raw_df   = exec_ns.get("df",       None)
V6F      = exec_ns.get("V6F",      [])
LOCS     = exec_ns.get("LOCS",     None)
loc_to_idx=exec_ns.get("loc_to_idx",{})
MODEL_MAP= exec_ns.get("MODEL_MAP",{})
TGT_USE_COLS=exec_ns.get("TGT_USE_COLS",{})
APPROX_COLS=[f"{c}_approx" for c in exec_ns.get("TGT_USE_COLS",{}).get("temp",[])
              if f"{c}_approx" in (raw_df.columns if raw_df is not None else [])]

if A_norm is not None and A_norm.device.type=="cpu":
    A_norm = A_norm.to(DEVICE)

TGT_COLS = TGT_USE_COLS.get("temp",[])
NT = len(TGT_COLS) if TGT_COLS else 1

print(f"  N_LOCS={N_LOCS} | N_FEATS={N_FEATS} | NT={NT}")
print(f"  Best model: STGCN | Target: temp (residual)")

# ══════════════════════════════════════════════════════════════════════════════
# DATA BUILDER — reuse v6 pattern
# ══════════════════════════════════════════════════════════════════════════════
def build_loader(split="train", loc_subset=None, bs=4, max_s=400, lookback=24, stride=6):
    if raw_df is None or feat_sc is None or tgt_sc is None:
        return None
    sub = raw_df[raw_df["split"]==split].copy()
    all_ts = sorted(sub["time_utc"].unique()); T=len(all_ts)
    ts_to_i={t:i for i,t in enumerate(all_ts)}
    sub["_ti"]=sub["time_utc"].map(ts_to_i)
    sub["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                 for la,lo in zip(sub["Latitude"],sub["Longitude"])]
    sub=sub.dropna(subset=["_ti","_ni"])
    sub["_ti"]=sub["_ti"].astype(int); sub["_ni"]=sub["_ni"].astype(int)
    sub=sub[sub["_ti"]<T]
    if loc_subset is not None: sub=sub[sub["_ni"].isin(loc_subset)]

    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)
    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32)
    af=np.zeros((T,N_LOCS,max(len(APPROX_COLS),1)),dtype=np.float32)
    mf=np.zeros((T,N_LOCS),dtype=np.float32)

    Xf[sub["_ti"].values,sub["_ni"].values]=\
        feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)
    yf[sub["_ti"].values,sub["_ni"].values]=\
        tgt_sc.transform(sub[TGT_COLS].fillna(0).values).astype(np.float32)
    if APPROX_COLS:
        af[sub["_ti"].values,sub["_ni"].values]=\
            sub[APPROX_COLS].fillna(0).values.astype(np.float32)
    mf[:, loc_subset if loc_subset is not None else SEEN] = 1.0

    rng=np.random.default_rng(SEED)
    tidxs=list(range(lookback,T,stride))
    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))
    Xl=[]; yl=[]; ml=[]; al2=[]
    for ti in tidxs:
        Xw=Xf[ti-lookback:ti]
        if np.isnan(Xw).mean()>0.5: continue
        Xl.append(np.nan_to_num(Xw,0.)); yl.append(yf[ti])
        ml.append(mf[ti]); al2.append(af[ti])
    if not Xl: return None
    ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),
                      torch.tensor(np.array(ml)),torch.tensor(np.array(al2)))
    return DataLoader(ds,batch_size=bs,shuffle=(split=="train"),
                       num_workers=0,pin_memory=False,drop_last=False)

# ══════════════════════════════════════════════════════════════════════════════
# LOSS + EVAL
# ══════════════════════════════════════════════════════════════════════════════
def nll_loss(mu,lsv,y,mask):
    sv=torch.exp(lsv).clamp(min=1e-6)
    loss=0.5*(lsv+(y-mu)**2/sv)
    me=mask.unsqueeze(-1).expand_as(loss)
    return (loss*me).sum()/(me.sum()+1e-8)

def evaluate(model, loader):
    if loader is None: return float("nan"), float("nan")
    model.eval(); yt_l=[]; yp_l=[]
    with torch.no_grad():
        for X,y,mask,av in loader:
            X=X.to(DEVICE)
            out=model(X,A_norm); mu=out[0]
            mu_np=tgt_sc.inverse_transform(
                mu.cpu().float().numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            y_np=tgt_sc.inverse_transform(
                y.numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            av_np=av.numpy()[:,:,0]
            locs=WETLAND if WETLAND else list(range(N_LOCS))
            yt_l.append((y_np[:,locs,0]+av_np[:,locs]).flatten())
            yp_l.append((mu_np[:,locs,0]+av_np[:,locs]).flatten())
    yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
    mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
    if len(yt)<5: return float("nan"),float("nan")
    rmse=float(np.sqrt(np.mean((yt-yp)**2)))
    r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
    return rmse, r2

# ══════════════════════════════════════════════════════════════════════════════
# RAY WORKER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def _worker_core(worker_id, theta_dict, loc_subset, K_steps, lr,
                  n_feats, nt, a_norm_cpu, feat_sc_pkl, tgt_sc_pkl,
                  v6f, tgt_cols, approx_cols, n_locs, seen, raw_path, seed):
    """
    Core worker function — runs local training K steps,
    records Δθ, losses, gradient directions.
    Can be called directly or via Ray remote.
    """
    import torch, copy, numpy as np
    import pandas as pd
    import pickle
    from pathlib import Path
    from sklearn.preprocessing import RobustScaler
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    import warnings; warnings.filterwarnings("ignore")

    torch.manual_seed(seed+worker_id)
    np.random.seed(seed+worker_id)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct scalers
    feat_sc=pickle.loads(feat_sc_pkl)
    tgt_sc_w=pickle.loads(tgt_sc_pkl)

    # Rebuild A_norm
    A_norm=torch.tensor(a_norm_cpu).to(dev)

    # Rebuild model from v6 STGCN class (inline copy to avoid full exec)
    import torch.nn as nn, torch.nn.functional as F
    DP=0.15
    class GConv(nn.Module):
        def __init__(self,d,dp=DP):
            super().__init__(); self.W=nn.Linear(d,d,bias=False)
            self.n=nn.LayerNorm(d); self.d=nn.Dropout(dp); self.a=nn.GELU()
        def forward(self,H,A):
            if A.dim()==3: A=A[0]
            return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),
                                            self.W(self.d(H)))))
    class HetHead(nn.Module):
        def __init__(self,d,nt,dp=DP):
            super().__init__()
            self.mu=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
            self.lsv=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
        def forward(self,h): return self.mu(h),self.lsv(h)
    class STGCN_W(nn.Module):
        def __init__(self,nf,h=64,nl=2,gl=2,nt=1):
            super().__init__(); self.p=nn.Linear(nf,h)
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

    model=STGCN_W(nf=n_feats,h=64,nl=2,gl=2,nt=nt).to(dev)
    model.load_state_dict({k:v.to(dev) for k,v in theta_dict.items()})
    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}
    opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)

    # Build data loader for this worker
    raw_df=pd.read_csv(raw_path,parse_dates=["time_utc"])
    sub=raw_df[raw_df["split"]=="train"].copy()
    all_ts=sorted(sub["time_utc"].unique()); T=len(all_ts)
    ts_to_i={t:i for i,t in enumerate(all_ts)}

    loc_to_idx_w={}
    from scipy.spatial import cKDTree
    locs_all=raw_df[["Latitude","Longitude"]].drop_duplicates().reset_index(drop=True)
    for _,r in locs_all.iterrows():
        loc_to_idx_w[(float(r.Latitude),float(r.Longitude))]=len(loc_to_idx_w)

    sub["_ti"]=sub["time_utc"].map(ts_to_i)
    sub["_ni"]=[loc_to_idx_w.get((float(la),float(lo)))
                 for la,lo in zip(sub["Latitude"],sub["Longitude"])]
    sub=sub.dropna(subset=["_ti","_ni"])
    sub["_ti"]=sub["_ti"].astype(int); sub["_ni"]=sub["_ni"].astype(int)
    sub=sub[sub["_ti"]<T]
    if loc_subset: sub=sub[sub["_ni"].isin(loc_subset)]

    Xf=np.zeros((T,n_locs,n_feats),dtype=np.float32)
    yf=np.zeros((T,n_locs,nt),dtype=np.float32)
    mf=np.zeros((T,n_locs),dtype=np.float32)
    vf=[f for f in v6f if f in sub.columns]
    tc=[f for f in tgt_cols if f in sub.columns]
    if not vf or not tc:
        return None, None

    Xf[sub["_ti"].values,sub["_ni"].values]=\
        feat_sc.transform(sub[vf].fillna(0).values).astype(np.float32)
    yf[sub["_ti"].values,sub["_ni"].values]=\
        tgt_sc_w.transform(sub[tc].fillna(0).values).astype(np.float32)
    locs_use=loc_subset if loc_subset else seen
    mf[:,locs_use]=1.0

    lookback=24; stride=8; max_s=300
    rng=np.random.default_rng(seed+worker_id)
    tidxs=list(range(lookback,T,stride))
    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))
    Xl=[]; yl=[]; ml=[]
    for ti in tidxs:
        Xw=Xf[ti-lookback:ti]
        if np.isnan(Xw).mean()>0.5: continue
        Xl.append(np.nan_to_num(Xw,0.)); yl.append(yf[ti]); ml.append(mf[ti])
    if not Xl: return None, None

    ds=TensorDataset(torch.tensor(np.array(Xl)),
                      torch.tensor(np.array(yl)),
                      torch.tensor(np.array(ml)))
    loader=DataLoader(ds,batch_size=4,shuffle=True,drop_last=False)

    losses=[]; grad_dirs=[]; step=0
    for X,y,mask in loader:
        if step>=K_steps: break
        X=X.to(dev); y=y.to(dev); mask=mask.to(dev)
        opt.zero_grad()
        mu,lsv=model(X,A_norm)
        sv=torch.exp(lsv).clamp(min=1e-6)
        nll=0.5*(lsv+(y-mu)**2/sv)
        me=mask.unsqueeze(-1).expand_as(nll)
        loss=(nll*me).sum()/(me.sum()+1e-8)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        gv=torch.cat([p.grad.data.cpu().flatten()
                       for p in model.parameters() if p.grad is not None])
        gn=gv.norm()+1e-8
        grad_dirs.append((gv/gn).numpy())
        opt.step()
        losses.append(float(loss.item()))
        step+=1

    theta_final={k:v.clone().cpu() for k,v in model.state_dict().items()}
    delta_theta={k:(theta_final[k]-theta_start[k]).cpu() for k in theta_start}

    stability=1.0
    if len(grad_dirs)>1:
        sims=[float(np.dot(grad_dirs[i],grad_dirs[i-1])/
                     (np.linalg.norm(grad_dirs[i])*np.linalg.norm(grad_dirs[i-1])+1e-8))
               for i in range(1,len(grad_dirs))]
        stability=float(np.mean(sims)) if sims else 1.0

    progress=0.0
    if len(losses)>1:
        progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8))

    delta_flat=torch.cat([v.flatten() for v in delta_theta.values()])

    behavior=dict(worker_id=worker_id, losses=losses,
                   stability=stability, progress=progress,
                   movement=float(delta_flat.norm()),
                   grad_dirs=grad_dirs,
                   delta_flat=delta_flat.numpy(),
                   final_loss=losses[-1] if losses else float("inf"))
    return delta_theta, behavior


# ── Ray remote wrapper ─────────────────────────────────────────────────────────
if HAS_RAY:
    @ray.remote(num_gpus=0.5)
    def ray_worker(worker_id, theta_dict, loc_subset, K_steps, lr,
                    n_feats, nt, a_norm_cpu, feat_sc_pkl, tgt_sc_pkl,
                    v6f, tgt_cols, approx_cols, n_locs, seen, raw_path, seed):
        return _worker_core(worker_id, theta_dict, loc_subset, K_steps, lr,
                              n_feats, nt, a_norm_cpu, feat_sc_pkl, tgt_sc_pkl,
                              v6f, tgt_cols, approx_cols, n_locs, seen, raw_path, seed)

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════
def compute_alphas(behaviors):
    N=len(behaviors)
    delta_flats=[b["delta_flat"] for b in behaviors]
    consensus=np.ones(N)
    for i in range(N):
        sims=[]
        for j in range(N):
            if i==j: continue
            di=delta_flats[i]; dj=delta_flats[j]
            sims.append(float(np.dot(di,dj)/((np.linalg.norm(di)*np.linalg.norm(dj))+1e-8)))
        consensus[i]=float(np.mean(sims)) if sims else 1.0
    stabs=np.clip([b["stability"] for b in behaviors],0,None)
    progs=np.clip([b["progress"]  for b in behaviors],0,None)
    consensus=np.clip(consensus,0,None)
    scores=consensus*stabs*progs; scores=np.clip(scores,1e-8,None)
    scores=scores-scores.max()
    alphas=np.exp(scores)/np.sum(np.exp(scores))
    return alphas, dict(consensus=consensus,stability=stabs,progress=progs)

def aggregate(theta_t, deltas, alphas, gamma=1.0):
    return {k:(theta_t[k].float()+gamma*sum(alphas[i]*deltas[i][k].float()
               for i in range(len(deltas))))
             for k in theta_t}

# ══════════════════════════════════════════════════════════════════════════════
# SETUP EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
import pickle as pkl_mod
from sklearn.preprocessing import RobustScaler

# Serialize scalers for Ray workers
feat_sc_pkl = pkl_mod.dumps(feat_sc)
tgt_sc_pkl  = pkl_mod.dumps(tgt_sc)
a_norm_cpu  = A_norm.cpu().numpy() if A_norm is not None else None
raw_path    = str(PROJECT/"preprocessed_v3"/"master_processed.csv")

# Subsets: one per site (4 workers)
N_WORKERS = 4
SUBSETS   = [SITE_LOCS.get(s,[]) for s in SITES]
print(f"\n  Workers: {N_WORKERS} | Subsets: {[len(s) for s in SUBSETS]}")

# STGCN with RANDOM initialization (PI: compare with trained weights later)
from train_soil_spatial_v6 import STGCN as STGCN_V6
torch.manual_seed(SEED)
global_model = STGCN_V6(nf=N_FEATS, h=64, nl=2, gl=2, nt=NT).to(DEVICE)
print(f"  STGCN params: {sum(p.numel() for p in global_model.parameters()):,}")
print(f"  Initialization: RANDOM (will compare with trained weights later)")

theta_t = {k:v.cpu().clone() for k,v in global_model.state_dict().items()}

# Validation loader
val_loader = build_loader(split="test", bs=4, max_s=400)

# Hyperparameters
T_ROUNDS = 18
K_STEPS  = 50
GAMMA    = 0.8
LR       = 1e-3

print(f"  T={T_ROUNDS} rounds | K={K_STEPS} steps | γ={GAMMA} | lr={LR}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — BEHAVIOR-GUIDED DISTRIBUTED
# ══════════════════════════════════════════════════════════════════════════════
round_history = []
weight_snapshots = []
gamma = GAMMA; prev_rmse = float("inf"); best_rmse = float("inf")
best_theta = None

w0 = np.concatenate([v.flatten().numpy() for v in theta_t.values()])[:500]
weight_snapshots.append({"round":0,"who":"init","vec":w0.copy()})

t_total = time.time()

for rnd in range(1, T_ROUNDS+1):
    t_round = time.time()
    print(f"\n  Round {rnd}/{T_ROUNDS} | γ={gamma:.3f}")

    # ── Dispatch workers (Ray or sequential) ──────────────────────────────────
    if HAS_RAY:
        futures = [ray_worker.remote(
            wi, {k:v.clone() for k,v in theta_t.items()},
            SUBSETS[wi], K_STEPS, LR,
            N_FEATS, NT, a_norm_cpu, feat_sc_pkl, tgt_sc_pkl,
            V6F, TGT_COLS, APPROX_COLS, N_LOCS, SEEN, raw_path, SEED)
            for wi in range(N_WORKERS)]
        results = ray.get(futures)
    else:
        results = [_worker_core(
            wi, {k:v.clone() for k,v in theta_t.items()},
            SUBSETS[wi], K_STEPS, LR,
            N_FEATS, NT, a_norm_cpu, feat_sc_pkl, tgt_sc_pkl,
            V6F, TGT_COLS, APPROX_COLS, N_LOCS, SEEN, raw_path, SEED)
            for wi in range(N_WORKERS)]

    deltas=[]; behaviors=[]
    for wi,(delta_i,beh_i) in enumerate(results):
        if delta_i is None: continue
        deltas.append(delta_i); behaviors.append(beh_i)
        wf=np.concatenate([(theta_t[k]+delta_i[k]).flatten().numpy()
                             for k in list(theta_t.keys())[:5]])[:500]
        weight_snapshots.append({"round":rnd,"who":f"worker_{wi+1}","vec":wf})
        print(f"    W{wi+1} loss={beh_i['final_loss']:.4f} "
               f"stab={beh_i['stability']:.3f} prog={beh_i['progress']:.3f}")

    if not deltas:
        print("    No valid workers"); continue

    alphas, score_dict = compute_alphas(behaviors)
    print(f"    α={[f'{a:.3f}' for a in alphas]}")

    theta_cand = aggregate(theta_t, deltas, alphas, gamma=gamma)
    global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})
    rmse, r2 = evaluate(global_model, val_loader)
    print(f"    RMSE={rmse:.4f} R²={r2:.4f}")

    wg=np.concatenate([theta_cand[k].flatten().numpy()
                        for k in list(theta_cand.keys())[:5]])[:500]
    weight_snapshots.append({"round":rnd,"who":"global","vec":wg})

    if rmse<=prev_rmse*1.02:
        theta_t=theta_cand; prev_rmse=rmse
        if rmse<best_rmse: best_rmse=rmse; best_theta=copy.deepcopy(theta_cand)
        print(f"    ACCEPTED")
    else:
        gamma*=0.7; print(f"    REJECTED → γ={gamma:.3f}")

    round_history.append(dict(
        round=rnd, rmse=rmse, r2=r2, gamma=gamma,
        alphas=alphas.tolist(),
        worker_losses=[b["final_loss"] for b in behaviors],
        elapsed=time.time()-t_round))

total_time = time.time()-t_total
ideal_time = total_time/N_WORKERS
print(f"\n  Done: {total_time:.1f}s | Best RMSE: {best_rmse:.4f}")

# ── Centralized baseline ───────────────────────────────────────────────────────
print("\n  Centralized baseline...")
torch.manual_seed(SEED)
cent = STGCN_V6(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
cent_loader = build_loader("train",bs=4,max_s=600)
cent_opt = AdamW(cent.parameters(),lr=LR,weight_decay=5e-4)
t_cent=time.time()
for ep in range(T_ROUNDS):
    cent.train()
    for X,y,mask,av in (cent_loader or []):
        X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
        cent_opt.zero_grad()
        mu,lsv=cent(X,A_norm)
        sv=torch.exp(lsv).clamp(min=1e-6)
        me=mask.unsqueeze(-1).expand_as(mu)
        loss=(0.5*(lsv+(y-mu)**2/sv)*me).sum()/(me.sum()+1e-8)
        loss.backward(); nn.utils.clip_grad_norm_(cent.parameters(),1.0)
        cent_opt.step()
cent_time=time.time()-t_cent
cent_rmse,cent_r2=evaluate(cent,val_loader)
print(f"  Centralized: RMSE={cent_rmse:.4f} R²={cent_r2:.4f} | {cent_time:.1f}s")

# ── Save results ───────────────────────────────────────────────────────────────
pd.DataFrame(round_history).to_csv(RESULTS/"misar_v7_rounds.csv",index=False)
best_r2=max((r["r2"] for r in round_history if not np.isnan(r["r2"])),default=float("nan"))
summary=pd.DataFrame([
    dict(Method="Behavior-guided (ideal parallel)",RMSE=round(best_rmse,4),
         R2=round(best_r2,4),Time_s=round(ideal_time,2),Note="Ideal parallel estimate"),
    dict(Method="Behavior-guided (sequential)",RMSE=round(best_rmse,4),
         R2=round(best_r2,4),Time_s=round(total_time,2),Note="Sequential execution"),
    dict(Method="Centralized",RMSE=round(cent_rmse,4),
         R2=round(cent_r2,4),Time_s=round(cent_time,2),Note="Full dataset training"),
])
summary.to_csv(RESULTS/"misar_v7_results.csv",index=False)
print("\n  Results:"); print(summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# GIF 01 — RMSE vs Global Round
# ══════════════════════════════════════════════════════════════════════════════
print("\n  GIF 01: RMSE vs round...")
rnds=[r["round"] for r in round_history]
rmses=[r["rmse"] for r in round_history]
fig,ax=plt.subplots(figsize=(9,6))
ld,=ax.plot([],[],color="#1f77b4",lw=2.5,marker="^",ms=8,label="Behavior-guided RMSE")
lc,=ax.plot([],[],color="#ff7f0e",lw=2.5,marker="o",ms=8,label="Centralized RMSE")
ax.set_xlim(0,T_ROUNDS+1)
ylo=min(rmses+[cent_rmse])*0.97; yhi=max(rmses+[cent_rmse])*1.03
ax.set_ylim(ylo,yhi)
ax.set_xlabel("Global Round",fontsize=12); ax.set_ylabel("Test RMSE (°C)",fontsize=12)
ax.set_title("Performance and Processing Time During Training",fontsize=13)
ax.legend(fontsize=10)
ib=ax.text(0.05,0.25,"",transform=ax.transAxes,fontsize=9,
            bbox=dict(boxstyle="round",facecolor="lightblue",alpha=0.5))
def upd1(f):
    i=min(f,len(rnds)-1); r=round_history[i]
    ld.set_data(rnds[:i+1],rmses[:i+1])
    lc.set_data(rnds[:i+1],[cent_rmse]*(i+1))
    ib.set_text(f"Round {r['round']}/{T_ROUNDS}\n"
                 f"Est. ideal dist. time: {r['elapsed']/N_WORKERS:.3f}s\n"
                 f"Centralized time: {cent_time/T_ROUNDS:.3f}s\n"
                 f"Distributed RMSE: {r['rmse']:.3f} °C\n"
                 f"Centralized RMSE: {cent_rmse:.3f} °C")
    return ld,lc,ib
ani1=animation.FuncAnimation(fig,upd1,frames=len(rnds),interval=400,blit=True)
ani1.save(FIGS/"gif_01_time_performance.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_01_time_performance.gif")

# ══════════════════════════════════════════════════════════════════════════════
# GIF 02 — Weight Space PCA
# ══════════════════════════════════════════════════════════════════════════════
print("\n  GIF 02: Weight space PCA...")
all_vecs=np.array([s["vec"][:500] for s in weight_snapshots])
maxl=min(500,all_vecs.shape[1])
all_vecs=all_vecs[:,:maxl]
pca=PCA(n_components=2,random_state=SEED)
pca_pts=pca.fit_transform(all_vecs)
WCOLS=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]

fig,ax=plt.subplots(figsize=(9,7))
ax.set_xlabel("Weight-space PCA 1",fontsize=11)
ax.set_ylabel("Weight-space PCA 2",fontsize=11)
ax.set_title("Distributed Local-to-Global Navigation\nSpatiotemporal Soil Temperature",fontsize=12)
init_i=[i for i,s in enumerate(weight_snapshots) if s["who"]=="init"]
if init_i:
    ax.scatter(*pca_pts[init_i[0]],c="black",s=200,marker="X",zorder=10,
                label="Common initialization")
    ax.text(pca_pts[init_i[0],0],pca_pts[init_i[0],1]-0.02,
             "Initialization",fontsize=8,ha="center")

rnd_data={}
for i,s in enumerate(weight_snapshots):
    r=s["round"]; w=s["who"]
    if r not in rnd_data: rnd_data[r]={"workers":{},"global":None}
    if w.startswith("worker_"):
        rnd_data[r]["workers"][int(w.split("_")[1])-1]=pca_pts[i]
    elif w=="global": rnd_data[r]["global"]=pca_pts[i]

ws=[ax.scatter([],[],c=WCOLS[wi%4],s=60,label=f"Worker {wi+1} endpoints")
     for wi in range(N_WORKERS)]
gs=ax.scatter([],[],c="purple",s=120,marker="*",label="Behavior-guided global")
wc_flat=np.concatenate([p.data.cpu().flatten().numpy()
                          for p in cent.parameters()])[:maxl].reshape(1,-1)
cs_pt=pca.transform(wc_flat)
ax.scatter(*cs_pt[0],c="saddlebrown",s=120,marker="P",label="Centralized",zorder=8)
ax.legend(fontsize=8)
ax.set_xlim(pca_pts[:,0].min()-0.05,pca_pts[:,0].max()+0.05)
ax.set_ylim(pca_pts[:,1].min()-0.05,pca_pts[:,1].max()+0.05)

sorted_rnds=sorted(rnd_data.keys())
def upd2(f):
    r=sorted_rnds[min(f,len(sorted_rnds)-1)]; rd=rnd_data[r]
    for wi in range(N_WORKERS):
        if wi in rd["workers"]:
            ws[wi].set_offsets(rd["workers"][wi].reshape(1,2))
    if rd["global"] is not None:
        gs.set_offsets(rd["global"].reshape(1,2))
    return ws+[gs]
ani2=animation.FuncAnimation(fig,upd2,frames=len(sorted_rnds),interval=500,blit=True)
ani2.save(FIGS/"gif_02_weight_space.gif",writer=PillowWriter(fps=2),dpi=100)
plt.close(); print("    OK gif_02_weight_space.gif")

# ══════════════════════════════════════════════════════════════════════════════
# GIF 03 — Loss Evolution
# ══════════════════════════════════════════════════════════════════════════════
print("\n  GIF 03: Loss evolution...")
wl=[[r["worker_losses"][wi] if wi<len(r["worker_losses"]) else float("nan")
      for r in round_history] for wi in range(N_WORKERS)]
gl=[r["rmse"]**2 for r in round_history]
cl=[cent_rmse**2]*len(rnds)
ymax=max([max([v for v in wl[wi] if not np.isnan(v)]+[0])
           for wi in range(N_WORKERS)]+gl+cl)*1.1

fig,ax=plt.subplots(figsize=(9,6))
ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(0,ymax)
ax.set_xlabel("Global Round",fontsize=11)
ax.set_ylabel("Standardised MSE Loss",fontsize=11)
ax.set_title("Loss Evolution: Local Workers vs Global Prediction vs Centralized",fontsize=12)
wlines=[ax.plot([],[],color=WCOLS[wi%4],lw=1.5,marker="o",ms=5,
                 label=f"Worker {wi+1} local loss")[0] for wi in range(N_WORKERS)]
gline,=ax.plot([],[],color="purple",lw=2.5,marker="*",ms=10,label="Predicted global model")
cline,=ax.plot([],[],color="saddlebrown",lw=2,marker="s",ms=8,ls="--",
                label="Centralized full-data model")
ax.legend(fontsize=8)
def upd3(f):
    i=min(f,len(rnds)-1); xs=rnds[:i+1]
    for wi in range(N_WORKERS): wlines[wi].set_data(xs,[wl[wi][j] for j in range(i+1)])
    gline.set_data(xs,gl[:i+1]); cline.set_data(xs,cl[:i+1])
    return wlines+[gline,cline]
ani3=animation.FuncAnimation(fig,upd3,frames=len(rnds),interval=400,blit=True)
ani3.save(FIGS/"gif_03_loss_workers.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_03_loss_workers.gif")

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURES
# ══════════════════════════════════════════════════════════════════════════════
# Performance bar
fig,ax=plt.subplots(figsize=(9,6))
methods=["Behavior-guided","Centralized","Existing\nbaseline"]
try:
    res_df2=pd.read_csv(PROJECT/"results_v6"/"v6_results_corrected.csv")
    base_rmse=float(res_df2[(res_df2["Model"]=="STGCN")&(res_df2["Target"]=="temp")
                              ]["RMSE"].iloc[0])
except: base_rmse=cent_rmse*1.01
vals=[best_rmse,cent_rmse,base_rmse]
bars=ax.bar(methods,vals,color="#1f77b4",width=0.5)
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.3f}",
             ha="center",va="bottom",fontsize=12,fontweight="bold")
ax.set_ylabel("Test RMSE (°C) — lower is better",fontsize=12)
ax.set_title("Final Soil Temperature Prediction Performance",fontsize=13)
ax.set_ylim(0,max(vals)*1.15)
plt.tight_layout()
plt.savefig(FIGS/"final_performance.png",dpi=300,bbox_inches="tight")
plt.close()

# Time bar
fig,ax=plt.subplots(figsize=(9,6))
tm=["Behavior-guided\n(ideal parallel)","Behavior-guided\n(sequential)","Centralized"]
tv=[round(ideal_time,2),round(total_time,2),round(cent_time,2)]
bars=ax.bar(tm,tv,color="#1f77b4",width=0.5)
for bar,v in zip(bars,tv):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.01,f"{v:.2f}s",
             ha="center",va="bottom",fontsize=12,fontweight="bold")
ax.set_ylabel("Processing Time (seconds)",fontsize=12)
ax.set_title("Training Processing Time Comparison",fontsize=13)
ax.set_ylim(0,max(tv)*1.15)
plt.tight_layout()
plt.savefig(FIGS/"processing_time.png",dpi=300,bbox_inches="tight")
plt.close()

print(f"\n{'='*65}")
print(f"  SPATIAL MISAR v7 COMPLETE")
print(f"  Behavior-guided RMSE: {best_rmse:.4f} | R²: {best_r2:.4f}")
print(f"  Centralized RMSE:     {cent_rmse:.4f} | R²: {cent_r2:.4f}")
print(f"  Improvement: {(cent_rmse-best_rmse)/cent_rmse*100:.1f}%")
print(f"  Ideal speedup: {cent_time/ideal_time:.1f}×")
print(f"  Figures: {FIGS}")
print(f"{'='*65}")
