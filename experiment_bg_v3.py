"""
experiment_bg_v3.py
BEHAVIOR-GUIDED DISTRIBUTED EXPERIMENT v3
PI-corrected framework:
  - Train: each worker on its spatial subregion (SEEN only, NO Wetland)
  - Val: Wetland holdout (unseen space — out of distribution)
  - Objective: minimize validation loss on unseen locations
  - Two models: STGCN + SpatialMamba
  - Subgraph masking: worker i only sees edges within its subset
  - No data leakage: Wetland never in any worker training data
"""
import os,sys,time,copy,pickle,warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT=Path("/home/emmanuel.keku")
PREPROC=PROJECT/"preprocessed_v3"
RESULTS=PROJECT/"results_v7"; RESULTS.mkdir(parents=True,exist_ok=True)
FIGS   =PROJECT/"figures_v7";  FIGS.mkdir(parents=True,exist_ok=True)
SEED=42; np.random.seed(SEED)

print("="*65)
print("  BEHAVIOR-GUIDED DISTRIBUTED EXPERIMENT v3")
print("  PI framework: train on subregion, val on unseen Wetland")
print("  Models: STGCN + SpatialMamba | No Wetland leakage")
print("="*65)

import torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import AdamW
from sklearn.preprocessing import RobustScaler
from scipy.spatial import cKDTree
torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

try:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,num_cpus=8,num_gpus=0)
    HAS_RAY=True; print(f"  Ray: {ray.__version__}")
except Exception as e:
    HAS_RAY=False; print(f"  Ray: sequential mode")

# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE DATA SETUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Setting up data pipeline...")
with open(PREPROC/"feature_info.pkl","rb") as f: FI=pickle.load(f)
raw_df=pd.read_csv(PREPROC/"master_processed.csv",parse_dates=["time_utc"])

SITES  =FI["SITES"]
N_LOCS =FI["N_LOCS"]
ALL_TGTS=FI["ALL_TARGETS"]
LOCS   =pd.DataFrame(FI["LOCATIONS"])

# Features
CYCLICAL=[c for c in raw_df.columns if any(c.startswith(p) for p in ["sin_","cos_"])]
SNAP=FI["SNAP_FEATURES"]
CORE=[f for f in SNAP if f not in CYCLICAL and f in raw_df.columns]
APPROX=[f"{t}_approx"   for t in ALL_TGTS if f"{t}_approx"   in raw_df.columns]
RESID =[f"{t}_residual" for t in ALL_TGTS if f"{t}_residual" in raw_df.columns]
UNC=[]
for feat in CORE[:8]:
    vc=f"{feat}_unc_var"
    if vc not in raw_df.columns: raw_df[vc]=np.where(raw_df[feat].isna(),1.0,0.01)
    UNC.append(vc)
V6F=list(dict.fromkeys(CORE+APPROX+RESID+UNC))
V6F=[f for f in V6F if f in raw_df.columns]
N_FEATS=len(V6F)

# Target: RESIDUAL only (PI instruction)
TEMP_TGTS=FI["TEMP_TARGETS"]
use_cols=[f"{c}_residual" for c in TEMP_TGTS if f"{c}_residual" in raw_df.columns]
if not use_cols: use_cols=[c for c in TEMP_TGTS if c in raw_df.columns]
NT=len(use_cols)

# Scalers — fit on SEEN train only (no Wetland in scaler fitting)
tr_df=raw_df[(raw_df["split"]=="train")&(raw_df["Site"]!="Wetland")]
feat_sc=RobustScaler(); feat_sc.fit(tr_df[V6F].fillna(0).values)
tgt_sc =RobustScaler(); tgt_sc.fit(tr_df[use_cols].dropna().values)
print(f"  Features: {N_FEATS} | Target: {use_cols} | NT: {NT}")
print(f"  Scalers fit on SEEN train only (no Wetland)")

# Location index
loc_to_idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS.iterrows()}

# SITE_LOCS — build from raw_df
SITE_LOCS={}
for site in SITES:
    rows=raw_df[raw_df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    idxs=sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                  for _,r in rows.iterrows()
                  if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])
    SITE_LOCS[site]=idxs

WETLAND=SITE_LOCS["Wetland"]
SEEN   =sorted(set(i for s,v in SITE_LOCS.items() if s!="Wetland" for i in v))
print(f"  SITE_LOCS: {[(s,len(v)) for s,v in SITE_LOCS.items()]}")
print(f"  WETLAND (val/test): {len(WETLAND)} | SEEN (train): {len(SEEN)}")

# Spatial graph (full 256x256 — masked per worker during training)
coords=LOCS[["Latitude","Longitude"]].values.astype(np.float32)
sc_=coords*np.array([111.0,63.0]); tree_=cKDTree(sc_)
d_,i_=tree_.query(sc_,k=7); sig_=np.median(d_[:,1:])+1e-8
A_np=np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1,d_.shape[1]):
        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
        A_np[i,j]+=w; A_np[j,i]+=w
A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5
A_full=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32)).to(DEVICE)

def make_subgraph(loc_subset):
    """Mask graph to only show edges within loc_subset — no leakage."""
    mask=torch.zeros(N_LOCS,dtype=torch.float32)
    mask[loc_subset]=1.0
    A_sub=(A_full*mask.unsqueeze(0).to(DEVICE)*mask.unsqueeze(1).to(DEVICE))
    D_sub=A_sub.sum(1,keepdim=True).clamp(min=1e-8)**0.5
    return A_sub/(D_sub*D_sub.t()+1e-8)

print(f"  Graph: σ={sig_:.2f}km | shape={A_full.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# IMPORT MODELS FROM V6
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Importing models from v6...")
_src=open(PROJECT/"train_soil_spatial_v6.py").read()
_pre=_src.split("if args.mode")[0]
_ns={"__name__":"__imported__"}
import unittest.mock as _mock
with _mock.patch("sys.argv",["train_soil_spatial_v6.py","--mode","train","--target","temp"]):
    try: exec(_pre,_ns)
    except SystemExit: pass
    except Exception as e: print(f"  Warning: {e}")

MODEL_MAP=_ns.get("MODEL_MAP",{})
MODELS_TO_RUN=["STGCN","SpatialMamba"]
for m in MODELS_TO_RUN:
    if m not in MODEL_MAP:
        print(f"  ERROR: {m} not in MODEL_MAP"); sys.exit(1)
    print(f"  {m}: loaded | Tier: {getattr(MODEL_MAP[m],'tier','?')}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# PI framework:
#   TRAIN loader: worker's spatial subregion only (SEEN, no Wetland)
#   VAL loader:   Wetland only (unseen — out of distribution)
# ══════════════════════════════════════════════════════════════════════════════
def build_loader(loc_subset,split,bs=4,max_s=600,lookback=24,stride=4):
    """
    PI framework:
    - split='train': use loc_subset locations, train period only
    - split='val':   use Wetland locations, val period only
    - split='test':  use Wetland locations, test period only
    Objective: each worker optimizes OUT-OF-DISTRIBUTION generalization
    """
    sub=raw_df[raw_df["split"]==split].copy()
    all_ts=sorted(sub["time_utc"].unique()); T=len(all_ts)
    ts_to_i={t:i for i,t in enumerate(all_ts)}
    sub["_ti"]=sub["time_utc"].map(ts_to_i)
    sub["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                 for la,lo in zip(sub["Latitude"],sub["Longitude"])]
    sub=sub.dropna(subset=["_ti","_ni"])
    sub["_ti"]=sub["_ti"].astype(int); sub["_ni"]=sub["_ni"].astype(int)
    sub=sub[sub["_ti"]<T]
    # Filter to requested locations
    sub=sub[sub["_ni"].isin(loc_subset)]
    if len(sub)==0: return None

    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)
    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32)
    mf=np.zeros((T,N_LOCS),dtype=np.float32)
    Xf[sub["_ti"].values,sub["_ni"].values]=\
        feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)
    yf[sub["_ti"].values,sub["_ni"].values]=\
        tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)
    mf[:,list(loc_subset)]=1.0

    rng=np.random.default_rng(SEED)
    tidxs=list(range(lookback,T,stride))
    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))
    Xl=[]; yl=[]; ml=[]
    for ti in tidxs:
        Xw=Xf[ti-lookback:ti]
        if np.isnan(Xw).mean()>0.5: continue
        Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti]); ml.append(mf[ti])
    if not Xl: return None
    ds=TensorDataset(torch.tensor(np.array(Xl)),
                      torch.tensor(np.array(yl)),
                      torch.tensor(np.array(ml)))
    return DataLoader(ds,batch_size=bs,shuffle=(split=="train"),
                       num_workers=0,pin_memory=False,drop_last=False)

# Test loaders
_tl=build_loader(SEEN,"train",max_s=50)
_vl=build_loader(WETLAND,"val",max_s=50)
print(f"  Train loader (SEEN): {_tl is not None}")
print(f"  Val loader (WETLAND): {_vl is not None}")

# ══════════════════════════════════════════════════════════════════════════════
# LOSS + EVALUATION
# PI: evaluate on RESIDUAL only, on WETLAND (unseen space)
# ══════════════════════════════════════════════════════════════════════════════
def nll_loss(mu,lsv,y,mask):
    sv=torch.exp(lsv).clamp(min=1e-6)
    loss=0.5*(lsv+(y-mu)**2/sv)
    me=mask.unsqueeze(-1).expand_as(loss)
    return (loss*me).sum()/(me.sum()+1e-8)

def evaluate(model,A_graph,locs=None,split="val",max_s=300):
    """Evaluate on unseen Wetland — RESIDUAL only (no reconstruction)."""
    if locs is None: locs=WETLAND
    loader=build_loader(locs,split,max_s=max_s)
    if loader is None: return float("nan"),float("nan"),float("nan")
    model.eval(); yt_l=[]; yp_l=[]; sig_l=[]
    with torch.no_grad():
        for X,y,mask in loader:
            X=X.to(DEVICE)
            out=model(X,A_graph); mu=out[0]
            lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])
            # RESIDUAL only — no approx reconstruction (PI instruction)
            mu_np=tgt_sc.inverse_transform(
                mu.cpu().float().numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            y_np=tgt_sc.inverse_transform(
                y.numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            sig_np=np.exp(0.5*lsv.cpu().float().numpy().reshape(X.shape[0],N_LOCS,NT))
            yt_l.append(y_np[:,locs,0].flatten())
            yp_l.append(mu_np[:,locs,0].flatten())
            sig_l.append(sig_np[:,locs,0].flatten())
    if not yt_l: return float("nan"),float("nan"),float("nan")
    yt=np.concatenate(yt_l); yp=np.concatenate(yp_l); sig=np.concatenate(sig_l)
    mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
    if len(yt)<5: return float("nan"),float("nan"),float("nan")
    rmse=float(np.sqrt(np.mean((yt-yp)**2)))
    r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
    unc=float(np.mean(sig))
    return rmse,r2,unc

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL SUBSETS (SEEN only — Wetland excluded)
# ══════════════════════════════════════════════════════════════════════════════
def get_subsets(n_workers):
    """
    PI: workers train on SEEN subregions only.
    Wetland is NEVER in any worker's training data.
    """
    if n_workers==1:
        return [SEEN]  # all 192 seen locations
    if n_workers==2:
        s1=SITE_LOCS["Bedrock"]+SITE_LOCS["Transition"]  # 128 locs
        s2=SITE_LOCS["Upland"]                             # 64 locs
        return [s1,s2]
    if n_workers==4:
        # Split each SEEN site in half
        subs=[]
        for s in ["Bedrock","Transition","Upland"]:
            locs=SITE_LOCS[s]; mid=max(1,len(locs)//2)
            subs.append(locs[:mid]); subs.append(locs[mid:])
        return subs[:4]  # 4 subsets of ~32 locs each
    if n_workers==8:
        subs=[]
        for s in ["Bedrock","Transition","Upland"]:
            locs=SITE_LOCS[s]; q=max(1,len(locs)//3)
            for i in range(3): subs.append(locs[i*q:(i+1)*q])
        return subs[:8]
    return [SEEN]

# ══════════════════════════════════════════════════════════════════════════════
# WORKER: local training
# PI: train on subregion, use subgraph (no leakage), minimize val loss
# ══════════════════════════════════════════════════════════════════════════════
def local_worker(worker_id,arch_name,theta_t_cpu,loc_subset,k_steps=50,lr=1e-3):
    arch_cls=MODEL_MAP[arch_name]
    model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    model.load_state_dict({k:v.to(DEVICE) for k,v in theta_t_cpu.items()})
    opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)

    # Subgraph: only edges within this worker's subset (no leakage)
    A_local=make_subgraph(loc_subset)

    # Train loader: worker's subregion, train split only
    loader=build_loader(loc_subset,"train",bs=4,max_s=300)
    if loader is None:
        print(f"    Worker {worker_id}: no data for subset ({len(loc_subset)} locs)")
        return None

    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}
    losses=[]; grad_dirs=[]; weight_traj=[]; uncertainties=[]; step=0
    w0=torch.cat([p.data.cpu().flatten() for p in model.parameters()]).numpy()
    weight_traj.append(w0[:200].copy()); model.train()

    for X,y,mask in loader:
        if step>=k_steps: break
        X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
        opt.zero_grad()
        out=model(X,A_local); mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])  # subgraph
        loss=nll_loss(mu,lsv,y,mask)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
        gv=torch.cat([p.grad.data.cpu().flatten()
                       for p in model.parameters() if p.grad is not None])
        gn=gv.norm()+1e-8; grad_dirs.append((gv/gn).numpy())
        with torch.no_grad():
            uncertainties.append(float(torch.exp(0.5*lsv).mean().item()))
        opt.step(); losses.append(float(loss.item())); step+=1
        if step%10==0:
            wt=torch.cat([p.data.cpu().flatten() for p in model.parameters()]).numpy()
            weight_traj.append(wt[:200].copy())

    if not losses: return None
    theta_final={k:v.clone().cpu() for k,v in model.state_dict().items()}
    delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}
    delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()
    stability=float(np.mean([np.dot(grad_dirs[i],grad_dirs[i-1])/
                               (np.linalg.norm(grad_dirs[i])*
                                np.linalg.norm(grad_dirs[i-1])+1e-8)
                               for i in range(1,len(grad_dirs))])) if len(grad_dirs)>1 else 1.0
    progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.
    wf=torch.cat([v.flatten() for v in theta_final.values()]).numpy()
    weight_traj.append(wf[:200].copy())
    return dict(worker_id=worker_id,arch=arch_name,
                 delta=delta,delta_flat=delta_flat,
                 losses=losses,grad_dirs=grad_dirs,weight_traj=weight_traj,
                 stability=stability,progress=progress,
                 movement=float(np.linalg.norm(delta_flat)),
                 uncertainty=float(np.mean(uncertainties)) if uncertainties else 0.1,
                 final_loss=losses[-1])

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════
def compute_alphas_betas(behaviors):
    N=len(behaviors); delta_flats=[b["delta_flat"] for b in behaviors]
    consensus=np.ones(N)
    for i in range(N):
        sims=[float(np.dot(delta_flats[i],delta_flats[j])/
                     (np.linalg.norm(delta_flats[i])*np.linalg.norm(delta_flats[j])+1e-8))
               for j in range(N) if j!=i]
        consensus[i]=float(np.mean(sims)) if sims else 0.
    stabilities=np.array([b["stability"] for b in behaviors])
    progresses =np.array([b["progress"]  for b in behaviors])
    scores=np.clip(consensus,0,None)*np.clip(stabilities,0,None)*np.clip(progresses,0,None)
    scores=np.clip(scores,1e-8,None); scores=scores-scores.max()
    alphas=np.exp(scores)/np.sum(np.exp(scores))
    mean_delta=np.mean(delta_flats,axis=0)
    betas=[]
    for df in delta_flats:
        residual=df-mean_delta
        sign=np.sign(np.dot(df,residual)+1e-8)
        betas.append(float(sign*np.linalg.norm(residual)/(np.linalg.norm(df)+1e-8)))
    U=float(np.mean([b["uncertainty"] for b in behaviors]))
    gamma=0.85*(1-min(U*0.1,0.5))
    return alphas,np.array(betas),gamma,U,dict(
        consensus=consensus,stability=stabilities,progress=progresses)

def aggregate(theta_t,behaviors,alphas,betas,gamma):
    N=len(behaviors); theta_new={}
    mean_delta={k:sum(behaviors[i]["delta"][k].float() for i in range(N))/N
                 for k in theta_t}
    for k in theta_t:
        shared=sum(alphas[i]*behaviors[i]["delta"][k].float() for i in range(N))
        residual_sum=sum(betas[i]*(behaviors[i]["delta"][k].float()-mean_delta[k])
                          for i in range(N))
        theta_new[k]=(theta_t[k].float()+gamma*(shared+0.1*residual_sum))
    return theta_new

# ── Ray remote worker for TRUE parallel execution ────────────────────────────

if HAS_RAY:

    @ray.remote(num_cpus=2)

    def ray_worker_remote(worker_id, arch_name, theta_t_cpu, loc_subset,

                           k_steps=50, lr=1e-3, seed=42):

        """Runs on separate Ray worker — truly parallel with other workers."""

        import torch, torch.nn as nn, numpy as np

        import pickle, pandas as pd, warnings, time

        from pathlib import Path

        from sklearn.preprocessing import RobustScaler

        from scipy.spatial import cKDTree

        from torch.optim import AdamW

        from torch.utils.data import DataLoader, TensorDataset

        warnings.filterwarnings("ignore")

        torch.manual_seed(seed+worker_id)



        PROJECT2 = Path("/home/emmanuel.keku")

        PREPROC2 = PROJECT2/"preprocessed_v3"

        with open(PREPROC2/"feature_info.pkl","rb") as f: FI2=pickle.load(f)

        raw2 = pd.read_csv(PREPROC2/"master_processed.csv", parse_dates=["time_utc"])

        LOCS2 = pd.DataFrame(FI2["LOCATIONS"]); N_LOCS2=FI2["N_LOCS"]

        ALL_TGTS2=FI2["ALL_TARGETS"]; SNAP2=FI2["SNAP_FEATURES"]

        CYCLICAL2=[c for c in raw2.columns if any(c.startswith(p) for p in ["sin_","cos_"])]

        CORE2=[f for f in SNAP2 if f not in CYCLICAL2 and f in raw2.columns]

        APPROX2=[f"{t}_approx" for t in ALL_TGTS2 if f"{t}_approx" in raw2.columns]

        RESID2=[f"{t}_residual" for t in ALL_TGTS2 if f"{t}_residual" in raw2.columns]

        UNC2=[]

        for feat in CORE2[:8]:

            vc=f"{feat}_unc_var"

            if vc not in raw2.columns: raw2[vc]=np.where(raw2[feat].isna(),1.0,0.01)

            UNC2.append(vc)

        V6F2=list(dict.fromkeys(CORE2+APPROX2+RESID2+UNC2))

        V6F2=[f for f in V6F2 if f in raw2.columns]

        TEMP2=FI2["TEMP_TARGETS"]

        use_cols2=[f"{c}_residual" for c in TEMP2 if f"{c}_residual" in raw2.columns]

        if not use_cols2: use_cols2=[c for c in TEMP2 if c in raw2.columns]

        NT2=len(use_cols2)

        tr2=raw2[(raw2["split"]=="train")&(raw2["Site"]!="Wetland")]

        feat_sc2=RobustScaler(); feat_sc2.fit(tr2[V6F2].fillna(0).values)

        tgt_sc2=RobustScaler(); tgt_sc2.fit(tr2[use_cols2].dropna().values)

        loc2idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS2.iterrows()}

        coords=LOCS2[["Latitude","Longitude"]].values.astype(np.float32)*np.array([111.0,63.0])

        tree_=cKDTree(coords); d_,i_=tree_.query(coords,k=7)

        sig_=np.median(d_[:,1:])+1e-8

        A_np=np.zeros((N_LOCS2,N_LOCS2),dtype=np.float32)

        for i in range(N_LOCS2):

            for jp in range(1,d_.shape[1]):

                j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))

                A_np[i,j]+=w; A_np[j,i]+=w

        A_np+=np.eye(N_LOCS2); D_=A_np.sum(1,keepdims=True)**0.5

        A_full2=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))

        # Subgraph mask

        mask=torch.zeros(N_LOCS2,dtype=torch.float32); mask[loc_subset]=1.0

        A_local2=(A_full2*mask.unsqueeze(0)*mask.unsqueeze(1))

        D2=A_local2.sum(1,keepdim=True).clamp(min=1e-8)**0.5

        A_local2=A_local2/(D2*D2.t()+1e-8)



        # Import model class from v6

        import unittest.mock as _mock, sys

        _src=open(PROJECT2/"train_soil_spatial_v6.py").read()

        _pre=_src.split("if args.mode")[0]

        _ns={"__name__":"__imported__"}

        with _mock.patch("sys.argv",["train_soil_spatial_v6.py","--mode","train","--target","temp"]):

            try: exec(_pre,_ns)

            except: pass

        MODEL_MAP2=_ns.get("MODEL_MAP",{})

        arch_cls2=MODEL_MAP2.get(arch_name)

        if arch_cls2 is None: return None



        nf2=len(V6F2)

        model=arch_cls2(nf=nf2,h=64,nl=2,gl=2,nt=NT2)

        model.load_state_dict({k:v.cpu() for k,v in theta_t_cpu.items()})

        opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)



        # Data

        sub2=raw2[raw2["split"]=="train"].copy()

        all_ts=sorted(sub2["time_utc"].unique()); T2=len(all_ts)

        ts2i={t:i for i,t in enumerate(all_ts)}

        sub2["_ti"]=sub2["time_utc"].map(ts2i)

        sub2["_ni"]=[loc2idx.get((float(la),float(lo))) for la,lo in zip(sub2["Latitude"],sub2["Longitude"])]

        sub2=sub2.dropna(subset=["_ti","_ni"])

        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)

        sub2=sub2[sub2["_ti"]<T2]; sub2=sub2[sub2["_ni"].isin(loc_subset)]

        if len(sub2)==0: return None

        Xf=np.zeros((T2,N_LOCS2,nf2),dtype=np.float32)

        yf=np.zeros((T2,N_LOCS2,NT2),dtype=np.float32)

        mf=np.zeros((T2,N_LOCS2),dtype=np.float32)

        Xf[sub2["_ti"].values,sub2["_ni"].values]=feat_sc2.transform(sub2[V6F2].fillna(0).values).astype(np.float32)

        yf[sub2["_ti"].values,sub2["_ni"].values]=tgt_sc2.transform(sub2[use_cols2].fillna(0).values).astype(np.float32)

        mf[:,loc_subset]=1.0

        LB=24; tidxs=list(range(LB,T2,6))[:300]

        Xl=[]; yl=[]; ml2=[]

        for ti in tidxs:

            Xw=Xf[ti-LB:ti]

            if np.isnan(Xw).mean()>0.5: continue

            Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti]); ml2.append(mf[ti])

        if not Xl: return None

        ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),torch.tensor(np.array(ml2)))

        loader=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0)



        theta_start={k:v.clone() for k,v in model.state_dict().items()}

        losses=[]; grad_dirs=[]; weight_traj=[]; uncertainties=[]; step=0

        w0=torch.cat([p.data.flatten() for p in model.parameters()]).numpy()

        weight_traj.append(w0[:200].copy()); model.train()

        for X,y,mask in loader:

            if step>=k_steps: break

            opt.zero_grad()

            out=model(X,A_local2); mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])

            sv=torch.exp(lsv).clamp(min=1e-6)

            loss=(0.5*(lsv+(y-mu)**2/sv)*mask.unsqueeze(-1)).sum()/(mask.sum()+1e-8)

            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)

            gv=torch.cat([p.grad.data.flatten() for p in model.parameters() if p.grad is not None])

            gn=gv.norm()+1e-8; grad_dirs.append((gv/gn).numpy())

            uncertainties.append(float(torch.exp(0.5*lsv).mean().item()))

            opt.step(); losses.append(float(loss.item())); step+=1

            if step%10==0:

                wt=torch.cat([p.data.flatten() for p in model.parameters()]).numpy()

                weight_traj.append(wt[:200].copy())

        if not losses: return None

        theta_final={k:v.clone() for k,v in model.state_dict().items()}

        delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}

        delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()

        stability=float(np.mean([np.dot(grad_dirs[i],grad_dirs[i-1])/(np.linalg.norm(grad_dirs[i])*np.linalg.norm(grad_dirs[i-1])+1e-8) for i in range(1,len(grad_dirs))])) if len(grad_dirs)>1 else 1.0

        progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.

        wf=torch.cat([v.flatten() for v in theta_final.values()]).numpy()

        weight_traj.append(wf[:200].copy())

        return dict(worker_id=worker_id,arch=arch_name,delta=delta,delta_flat=delta_flat,

                     losses=losses,grad_dirs=grad_dirs,weight_traj=weight_traj,

                     stability=stability,progress=progress,

                     movement=float(np.linalg.norm(delta_flat)),

                     uncertainty=float(np.mean(uncertainties)) if uncertainties else 0.1,

                     final_loss=losses[-1])

else:

    def ray_worker_remote(*args,**kwargs):

        return None  # Ray not available



# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — two models, N=1,2,4,8
# ══════════════════════════════════════════════════════════════════════════════
T_ROUNDS=18; K_STEPS=50; LR=1e-3
N_WORKERS_LIST=[2,4]  # Focus on N=2,4 for weight trajectories

all_results=[]; model_histories={}

for arch_name in MODELS_TO_RUN:
    print(f"\n{'#'*65}")
    print(f"  MODEL: {arch_name}")
    print(f"{'#'*65}")
    arch_cls=MODEL_MAP[arch_name]
    model_histories[arch_name]={}

    # Centralized baseline for this model
    print(f"\n  Centralized baseline ({arch_name})...")
    torch.manual_seed(SEED)
    cent=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    cent_loader=build_loader(SEEN,"train",max_s=600)
    cent_opt=AdamW(cent.parameters(),lr=LR,weight_decay=5e-4)
    cent_wt=[]; cent_val_history=[]
    t_cent=time.time()
    for ep in range(T_ROUNDS):
        cent.train(); ep_loss=0.; nb=0
        for X,y,mask in (cent_loader or []):
            X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
            cent_opt.zero_grad()
            out=cent(X,A_full); mu=out[0]
            lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])
            loss=nll_loss(mu,lsv,y,mask); loss.backward()
            nn.utils.clip_grad_norm_(cent.parameters(),1.0); cent_opt.step()
            ep_loss+=loss.item(); nb+=1
        ep_rmse,ep_r2,_=evaluate(cent,A_full,WETLAND,"val",max_s=150)
        cent_val_history.append(dict(epoch=ep+1,
                                      train_loss=ep_loss/max(nb,1),
                                      val_rmse=ep_rmse,val_r2=ep_r2))
        print(f"  Cent ep {ep+1:02d}/{T_ROUNDS} | loss={ep_loss/max(nb,1):.4f} | val_RMSE={ep_rmse:.4f} R2={ep_r2:.4f}")
        wc=torch.cat([p.data.cpu().flatten() for p in cent.parameters()]).numpy()[:200]
        cent_wt.append({"round":ep+1,"vec":wc})
    cent_time=time.time()-t_cent
    cent_rmse,cent_r2,cent_unc=evaluate(cent,A_full,WETLAND,"val")
    print(f"  Centralized final: RMSE={cent_rmse:.4f} R2={cent_r2:.4f} time={cent_time:.1f}s")

    for N_WORKERS in N_WORKERS_LIST:
        print(f"\n{'='*55}")
        print(f"  {arch_name} | N_WORKERS = {N_WORKERS}")
        print(f"{'='*55}")
        SUBSETS=get_subsets(N_WORKERS)
        print(f"  Subsets (SEEN only, no Wetland): {[len(s) for s in SUBSETS]}")

        torch.manual_seed(SEED)
        global_model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
        theta_t={k:v.cpu().clone() for k,v in global_model.state_dict().items()}

        round_history=[]; weight_snapshots=[]; worker_trajs={wi:[] for wi in range(N_WORKERS)}
        w0=torch.cat([v.flatten() for v in theta_t.values()]).numpy()[:200]
        weight_snapshots.append({"round":0,"who":"init","vec":w0.copy()})
        best_rmse=float("inf"); best_theta=None; prev_rmse=float("inf")
        t_start=time.time()

        for rnd in range(1,T_ROUNDS+1):
            t_round=time.time()
            theta_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}
            # TRUE PARALLEL: all N workers run simultaneously via Ray
            if HAS_RAY and N_WORKERS > 1:
                # Launch all workers at same time
                futures = [ray_worker_remote.remote(
                               wi, arch_name,
                               {k:v.clone() for k,v in theta_cpu.items()},
                               SUBSETS[wi], K_STEPS, LR, SEED)
                            for wi in range(N_WORKERS)]
                t_par = time.time()
                behaviors_raw = ray.get(futures)  # blocks until ALL done
                parallel_wall_s = time.time()-t_par
                behaviors = [b for b in behaviors_raw if b is not None]
                print(f"    True parallel wall: {parallel_wall_s:.1f}s | N={N_WORKERS} | ideal T/N")
            else:
                # N=1 sequential baseline
                behaviors = []
                b = local_worker(0, arch_name,
                                  {k:v.clone() for k,v in theta_cpu.items()},
                                  SUBSETS[0], K_STEPS, LR)
                if b is not None: behaviors.append(b)
                parallel_wall_s = time.time()-t_round


            if not behaviors:
                print(f"  Rnd {rnd:02d}: No workers returned")
                round_history.append(dict(round=rnd,rmse=float("nan"),r2=float("nan"),
                                           gamma=0.,U=float("nan"),alphas=[],betas=[],
                                           worker_losses=[],elapsed=time.time()-t_round))
                continue

            # Record trajectories
            for b in behaviors:
                wi=b["worker_id"]
                for wt in b["weight_traj"]: worker_trajs[wi].append({"round":rnd,"vec":wt})
                wend=torch.cat([v.flatten() for v in b["delta"].values()]).numpy()[:200]
                weight_snapshots.append({"round":rnd,"who":f"worker_{wi+1}","vec":wend})

            alphas,betas,gamma_u,U,scores=compute_alphas_betas(behaviors)
            theta_cand=aggregate(theta_t,behaviors,alphas,betas,gamma_u)

            # Evaluate on WETLAND (unseen) — PI: optimize out-of-distribution
            global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})
            rmse,r2,unc=evaluate(global_model,A_full,WETLAND,"val")

            wg=torch.cat([v.flatten() for v in theta_cand.values()]).numpy()[:200]
            weight_snapshots.append({"round":rnd,"who":"global","vec":wg})

            rmse_ok=not(np.isnan(rmse) or np.isinf(rmse))
            if not rmse_ok or rmse<=prev_rmse*1.02:
                theta_t=theta_cand
                if rmse_ok:
                    prev_rmse=rmse
                    if rmse<best_rmse:
                        best_rmse=rmse; best_theta=copy.deepcopy(theta_cand)
                status="ACC"
            else:
                gamma_u*=0.8; status="REJ"

            wl=[b["final_loss"] for b in behaviors]
            al=[f"{a:.2f}" for a in alphas]
            bl=[f"{b:.2f}" for b in betas]
            print(f"  Rnd {rnd:02d} | RMSE={rmse:.4f} R²={r2:.4f} U={U:.3f} "
                   f"γ={gamma_u:.3f} α={al} β={bl} | {status}")

            # Save parallel wall time
            p_wall = parallel_wall_s if "parallel_wall_s" in vars() else (time.time()-t_round)

            # Save absolute weight snapshots (500 dims) for real trajectory visualization
            worker_w_snaps=[]
            for b in behaviors:
                w_abs=torch.cat([(theta_cpu[k].float()+b["delta"][k].float()).flatten()
                                   for k in theta_cpu]).numpy()[:500]
                worker_w_snaps.append(w_abs.tolist())
            global_w=torch.cat([v.float().flatten() for v in theta_cand.values()]).numpy()[:500].tolist()

            round_history.append(dict(
                round=rnd,arch=arch_name,n_workers=N_WORKERS,
                rmse=rmse,r2=r2,gamma=gamma_u,U=U,
                alphas=alphas.tolist(),betas=betas.tolist(),
                consensus=scores["consensus"].tolist(),
                stability=scores["stability"].tolist(),
                progress=scores["progress"].tolist(),
                worker_losses=wl,
                parallel_wall_s=round(p_wall,2),
                worker_weight_snaps=str(worker_w_snaps),
                global_weight_snap=str(global_w),
                elapsed=time.time()-t_round))

        total_time=time.time()-t_start
        ideal_time=total_time/N_WORKERS
        valid_r2=[r["r2"] for r in round_history if not np.isnan(r.get("r2",float("nan")))]

        # Save rounds
        pd.DataFrame(round_history).to_csv(
            RESULTS/f"rounds_v3_{arch_name}_N{N_WORKERS}.csv",index=False)

        result=dict(
            arch=arch_name, n_workers=N_WORKERS,
            best_rmse=round(best_rmse,4) if not np.isinf(best_rmse) else float("nan"),
            best_r2=round(max(valid_r2),4) if valid_r2 else float("nan"),
            cent_rmse=round(cent_rmse,4), cent_r2=round(cent_r2,4),
            ideal_time_s=round(ideal_time,2),
            total_time_s=round(total_time,2),
            cent_time_s=round(cent_time,2))
        all_results.append(result)
        model_histories[arch_name][N_WORKERS]={
            "rounds":round_history,"snapshots":weight_snapshots,
            "worker_trajs":worker_trajs,"cent_wt":cent_wt}

        print(f"\n  {arch_name} N={N_WORKERS}: best_RMSE={best_rmse:.4f} "
               f"best_R²={max(valid_r2):.4f} if valid_r2 else nan"
               f" | ideal={ideal_time:.1f}s vs cent={cent_time:.1f}s")

# Save summary
summary=pd.DataFrame(all_results)
summary.to_csv(RESULTS/"behavior_guided_v3_summary.csv",index=False)
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(summary.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FIGURES (per model)
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from sklearn.decomposition import PCA

WORKER_COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                "#9467bd","#8c564b","#e377c2","#7f7f7f"]

for arch_name in MODELS_TO_RUN:
    print(f"\n  Generating figures for {arch_name}...")
    # Use N=4 for GIFs
    if 4 not in model_histories[arch_name]: continue
    h4=model_histories[arch_name][4]
    rh=h4["rounds"]; snaps=h4["snapshots"]; wtrajs=h4["worker_trajs"]
    cent_wt=h4["cent_wt"]
    cent_r=next((r for r in all_results if r["arch"]==arch_name and r["n_workers"]==4),{})
    general_rmse_=cent_r.get("cent_rmse",float("nan"))

    rounds_=[r["round"] for r in rh]; rmses_=[r["rmse"] for r in rh]
    r2s_=[r["r2"] for r in rh]; Us_=[r["U"] for r in rh]
    gammas_=[r["gamma"] for r in rh]
    valid_r=[r for r in rmses_ if not np.isnan(r) and not np.isinf(r)]

    # GIF_01: RMSE + R² convergence
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    ax1,ax2=axes
    l1,=ax1.plot([],[],color="#1f77b4",lw=2.5,marker="^",ms=8,
                  label=f"{arch_name} RMSE")
    l2,=ax1.plot([],[],color="#ff7f0e",lw=2.5,marker="o",ms=8,
                  label="General model RMSE")
    ylo=min(valid_r+[general_rmse_])*0.9 if valid_r and not np.isnan(general_rmse_) else 0
    yhi=max(valid_r+[general_rmse_])*1.1 if valid_r and not np.isnan(general_rmse_) else 5
    ax1.set_xlim(0,T_ROUNDS+1); ax1.set_ylim(ylo,yhi)
    ax1.set_xlabel("Global Round",fontsize=12)
    ax1.set_ylabel("Wetland Val RMSE (residual)",fontsize=11)
    ax1.set_title(f"Convergence | {arch_name} | N=4 workers",fontsize=12)
    ax1.legend(fontsize=10)
    box=ax1.text(0.05,0.75,"",transform=ax1.transAxes,fontsize=8,
                  bbox=dict(boxstyle="round",facecolor="lightblue",alpha=0.5))
    l3,=ax2.plot([],[],color="#2ca02c",lw=2,label="R² (Wetland val)")
    ax2.set_xlim(0,T_ROUNDS+1); ax2.set_ylim(-0.1,1.0)
    ax2.set_xlabel("Global Round",fontsize=12)
    ax2.set_ylabel("R² on unseen Wetland (residual)",fontsize=11)
    ax2.set_title("Generalization to Unseen Space",fontsize=12)
    ax2.legend(fontsize=10)
    def u1(frame):
        i=min(frame,len(rounds_)-1); r=rh[i]
        l1.set_data(rounds_[:i+1],rmses_[:i+1])
        l2.set_data(rounds_[:i+1],[general_rmse_]*(i+1))
        l3.set_data(rounds_[:i+1],r2s_[:i+1])
        box.set_text(f"Round {r['round']}/{T_ROUNDS}\n"
                      f"RMSE: {r['rmse']:.3f}\nR²: {r['r2']:.3f}\n"
                      f"Cent RMSE: {general_rmse_:.3f}")
        return l1,l2,l3,box
    ani1=animation.FuncAnimation(fig,u1,frames=len(rounds_),interval=400,blit=True)
    ani1.save(FIGS/f"gif_01_{arch_name}_convergence.gif",
               writer=PillowWriter(fps=2.5),dpi=100)
    plt.close(); print(f"    OK gif_01_{arch_name}_convergence.gif")

    # GIF_02: Weight space trajectories
    all_vecs=[s["vec"][:200] for s in snaps]
    if all_vecs:
        maxl=min(200,min(len(v) for v in all_vecs))
        all_np=np.array([v[:maxl] for v in all_vecs])
        pca=PCA(n_components=2,random_state=SEED)
        all_pca=pca.fit_transform(all_np)
        by_round={}
        for i,s in enumerate(snaps):
            rn=s["round"]; who=s["who"]
            if rn not in by_round: by_round[rn]={"workers":{},"global":None}
            if who.startswith("worker_"):
                wi=int(who.split("_")[1])-1
                if wi not in by_round[rn]["workers"]: by_round[rn]["workers"][wi]=[]
                by_round[rn]["workers"][wi].append(all_pca[i])
            elif who=="global": by_round[rn]["global"]=all_pca[i]
        # Centralized path
        cent_vecs=np.array([c["vec"][:maxl] for c in cent_wt])
        cent_pca=pca.transform(cent_vecs)
        fig,ax=plt.subplots(figsize=(10,8))
        ax.plot(cent_pca[:,0],cent_pca[:,1],color="saddlebrown",
                 lw=3,alpha=0.8,label="General model path path")
        ax.scatter(cent_pca[-1,0],cent_pca[-1,1],c="orange",s=200,
                    marker="*",zorder=10,label="General model path final")
        init_idx=[i for i,s in enumerate(snaps) if s["who"]=="init"]
        if init_idx:
            ax.scatter(all_pca[init_idx[0],0],all_pca[init_idx[0],1],
                        c="black",s=200,marker="X",zorder=10,label="Initialization")
        # Worker lines
        subset_names=[f"Spatial Worker {i+1}" for i in range(8)]
        wlines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,
                         label=f"Worker {wi+1}: {subset_names[wi] if wi<len(subset_names) else ''}")[0]
                 for wi in range(min(4,N_WORKERS))]
        gline,=ax.plot([],[],color="darkviolet",lw=2.5,ls="--",
                        label="Predicted global path")
        gdot=ax.scatter([],[],c="blue",s=150,marker="s",zorder=9,
                          label="Predicted final")
        xpad=(all_pca[:,0].max()-all_pca[:,0].min())*0.05+0.05
        ypad=(all_pca[:,1].max()-all_pca[:,1].min())*0.05+0.05
        ax.set_xlim(all_pca[:,0].min()-xpad,all_pca[:,0].max()+xpad)
        ax.set_ylim(all_pca[:,1].min()-ypad,all_pca[:,1].max()+ypad)
        ax.set_xlabel("Weight-space PC1",fontsize=11)
        ax.set_ylabel("Weight-space PC2",fontsize=11)
        ax.set_title(f"Weight Trajectories | {arch_name} | N=4\n"
                      f"Train: SEEN subregions | Val: Wetland (unseen)",fontsize=11)
        ax.legend(fontsize=8,loc="upper right")
        wpts={wi:[] for wi in range(4)}; gpts=[]
        sorted_rns=sorted(by_round.keys())
        def u2(frame):
            rn=sorted_rns[min(frame,len(sorted_rns)-1)]; rd=by_round[rn]
            for wi in range(min(4,N_WORKERS)):
                if wi in rd["workers"]:
                    for pt in rd["workers"][wi]: wpts[wi].append(pt)
                if len(wpts[wi])>1:
                    pts=np.array(wpts[wi]); wlines[wi].set_data(pts[:,0],pts[:,1])
            if rd["global"] is not None: gpts.append(rd["global"])
            if len(gpts)>1:
                gp=np.array(gpts); gline.set_data(gp[:,0],gp[:,1])
                gdot.set_offsets(gp[-1].reshape(1,2))
            return wlines+[gline,gdot]
        ani2=animation.FuncAnimation(fig,u2,frames=len(sorted_rns),interval=500,blit=True)
        ani2.save(FIGS/f"gif_02_{arch_name}_weight_space.gif",
                   writer=PillowWriter(fps=2),dpi=100)
        plt.close(); print(f"    OK gif_02_{arch_name}_weight_space.gif")

    # GIF_03: Loss evolution per worker
    all_wl=[[rh[ri]["worker_losses"][wi]
              if wi<len(rh[ri].get("worker_losses",[])) else float("nan")
              for ri in range(len(rh))] for wi in range(4)]
    gloss=[r["rmse"]**2 if not np.isnan(r.get("rmse",float("nan"))) else float("nan")
            for r in rh]
    closs=[general_rmse_**2]*len(rh) if not np.isnan(general_rmse_) else [float("nan")]*len(rh)
    all_v=[v for wl in all_wl for v in wl if not np.isnan(v)]
    ymax=max(all_v)*1.1 if all_v else 3
    fig,ax=plt.subplots(figsize=(10,6))
    ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(0,ymax)
    ax.set_xlabel("Global Round",fontsize=11)
    ax.set_ylabel("MSE Loss (residual)",fontsize=11)
    ax.set_title(f"Loss Evolution | {arch_name} | N=4\n"
                  f"Workers train on SEEN subregions | Global eval on Wetland",fontsize=11)
    subset_names=[f"Spatial Worker {i+1}" for i in range(8)]
    wlines2=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,marker="o",ms=5,
                      label=f"Worker {wi+1} ({subset_names[wi] if wi<len(subset_names) else ''})")[0]
              for wi in range(4)]
    gl,=ax.plot([],[],color="darkviolet",lw=2.5,marker="*",ms=10,
                 label="Global (Wetland val)")
    cl2,=ax.plot([],[],color="saddlebrown",lw=2,marker="s",ms=8,ls="--",
                  label="Centralized (Wetland val)")
    ax.legend(fontsize=8,loc="upper right")
    def u3(frame):
        i=min(frame,len(rh)-1); xs=rounds_[:i+1]
        for wi in range(4): wlines2[wi].set_data(xs,all_wl[wi][:i+1])
        gl.set_data(xs,gloss[:i+1]); cl2.set_data(xs,closs[:i+1])
        return wlines2+[gl,cl2]
    ani3=animation.FuncAnimation(fig,u3,frames=len(rh),interval=400,blit=True)
    ani3.save(FIGS/f"gif_03_{arch_name}_loss.gif",
               writer=PillowWriter(fps=2.5),dpi=100)
    plt.close(); print(f"    OK gif_03_{arch_name}_loss.gif")

# Final comparison figure — both models
fig,axes=plt.subplots(1,2,figsize=(18,7))
ax1,ax2=axes
# Performance
model_colors={"STGCN":"#1f77b4","SpatialMamba":"#ff7f0e"}
x_=np.arange(len(N_WORKERS_LIST)+1); w_=0.35
for mi,arch_name in enumerate(MODELS_TO_RUN):
    rmses_=[next((r["best_rmse"] for r in all_results
                   if r["arch"]==arch_name and r["n_workers"]==n),float("nan"))
             for n in N_WORKERS_LIST]
    cent_=[next((r["cent_rmse"] for r in all_results
                  if r["arch"]==arch_name),float("nan"))]
    all_rmse=rmses_+cent_
    all_rmse=[r if not np.isnan(r) else 0 for r in all_rmse]
    offset=(mi-0.5)*w_
    bars=ax1.bar(x_+offset,all_rmse,w_,color=model_colors[arch_name],
                  alpha=0.85,label=arch_name)
    for bar,v in zip(bars,all_rmse):
        if v>0: ax1.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.3f}",
                          ha="center",va="bottom",fontsize=8,fontweight="bold")
ax1.set_xticks(x_)
ax1.set_xticklabels([f"N={n}" for n in N_WORKERS_LIST]+["Centralized"],fontsize=10)
ax1.set_ylabel("Wetland Val RMSE (residual)",fontsize=11)
ax1.set_title("Final Performance | Residual-only | Unseen Wetland",fontsize=12)
ax1.legend(fontsize=10)

# Time
for mi,arch_name in enumerate(MODELS_TO_RUN):
    ideal=[next((r["ideal_time_s"] for r in all_results
                  if r["arch"]==arch_name and r["n_workers"]==n),0)
            for n in N_WORKERS_LIST]
    cent_t=[next((r["cent_time_s"] for r in all_results
                   if r["arch"]==arch_name),0)]
    offset=(mi-0.5)*w_
    ax2.bar(x_[:-1]+offset,ideal,w_,color=model_colors[arch_name],
             alpha=0.85,label=f"{arch_name} (ideal parallel)")
ax2.axhline(next((r["cent_time_s"] for r in all_results if r["arch"]=="STGCN"),0),
             color="red",ls="--",lw=2,label="Centralized STGCN")
ax2.set_xticks(x_[:-1])
ax2.set_xticklabels([f"N={n}" for n in N_WORKERS_LIST],fontsize=10)
ax2.set_ylabel("Ideal Parallel Time (s)",fontsize=11)
ax2.set_title("Training Time vs N Workers",fontsize=12)
ax2.legend(fontsize=9)
plt.suptitle("STGCN vs SpatialMamba | Behavior-Guided Distributed | v3\n"
              "Train: SEEN subregions only | Val: Wetland (unseen) | Residual-only evaluation",
              fontsize=12,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"final_comparison_v3.png",dpi=300,bbox_inches="tight")
plt.close()
print("\n  OK final_comparison_v3.png")

print(f"\n{'='*65}")
print(f"  EXPERIMENT v3 COMPLETE")
print(f"  Figures: {FIGS}")
figs_out=sorted(FIGS.glob("*v3*"))+sorted(FIGS.glob("*STGCN*"))+sorted(FIGS.glob("*Mamba*"))
for f in set(figs_out): print(f"  {f.name} ({f.stat().st_size//1024} KB)")
print(f"{'='*65}")
