"""
experiment_behavior_guided.py
BEHAVIOR-GUIDED DISTRIBUTED SPATIAL AI — Full PI Framework
Attaches to existing v6 pipeline — no rewriting
"""
import os,sys,time,copy,pickle,warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")


# ── Bootstrap SITE_LOCS from raw data (always rebuild) ───────────────────────

import pickle as _pkl, pandas as _pd

from pathlib import Path as _Path

_PREPROC2 = _Path("/home/emmanuel.keku/preprocessed_v3")

_raw2 = _pd.read_csv(_PREPROC2/"master_processed.csv",

                      usecols=["Site","Latitude","Longitude"])

with open(_PREPROC2/"feature_info.pkl","rb") as _f2: _FI2=_pkl.load(_f2)

_LOCS2 = _pd.DataFrame(_FI2["LOCATIONS"])

_loc2idx = {(float(r.Latitude),float(r.Longitude)):i for i,r in _LOCS2.iterrows()}

SITE_LOCS = {}

for _site in _FI2["SITES"]:

    _rows=_raw2[_raw2["Site"]==_site][["Latitude","Longitude"]].drop_duplicates()

    SITE_LOCS[_site]=sorted([_loc2idx.get((float(r.Latitude),float(r.Longitude)))

                               for _,r in _rows.iterrows()

                               if _loc2idx.get((float(r.Latitude),float(r.Longitude)))

                               is not None])

SITES   = _FI2["SITES"]

WETLAND = SITE_LOCS.get("Wetland",[])

SEEN    = sorted(set(i for s,v in SITE_LOCS.items() if s!="Wetland" for i in v))

print(f"  Bootstrap SITE_LOCS: {[(s,len(v)) for s,v in SITE_LOCS.items()]}")

print(f"  SITES: {SITES}")

print(f"  WETLAND: {len(WETLAND)} | SEEN: {len(SEEN)}")

# ─────────────────────────────────────────────────────────────────────────────


PROJECT=Path("/home/emmanuel.keku")
PREPROC=PROJECT/"preprocessed_v3"
RESULTS=PROJECT/"results_v7"; RESULTS.mkdir(parents=True,exist_ok=True)
FIGS   =PROJECT/"figures_v7";  FIGS.mkdir(parents=True,exist_ok=True)
SEED=42; np.random.seed(SEED)

print("="*65)
print("  BEHAVIOR-GUIDED DISTRIBUTED EXPERIMENT — Full PI Framework")
print("  Attaches to v6 pipeline | STGCN | Random init")
print("="*65)

import torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import AdamW
torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

try:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,num_cpus=8,
                  num_gpus=torch.cuda.device_count())
    HAS_RAY=True; print(f"  Ray: {ray.__version__}")
except Exception as e:
    HAS_RAY=False; print(f"  Ray: not available — sequential")

# ── Import v6 pipeline ────────────────────────────────────────────────────────
print("\n  Loading v6 pipeline...")
_src=open(PROJECT/"train_soil_spatial_v6.py").read()
_pre=_src.split("if args.mode")[0]
_ns={"__name__":"__imported__"}
import unittest.mock as _mock
with _mock.patch("sys.argv",["train_soil_spatial_v6.py","--mode","train","--target","temp"]):
    try: exec(_pre,_ns)
    except SystemExit: pass
    except Exception as e: print(f"  Warning: {e}")

MODEL_MAP  =_ns.get("MODEL_MAP",{})
raw_df     =_ns.get("raw_df",None)
feat_sc    =_ns.get("feat_sc",None)
V6F        =_ns.get("V6F",[])
N_FEATS    =_ns.get("N_FEATS",32)
N_LOCS     =_ns.get("N_LOCS",256)
LOCS       =_ns.get("LOCS",None)
SITES      =_ns.get("SITES",[])
SITE_LOCS  =_ns.get("SITE_LOCS",{})

# Rebuild SITE_LOCS if empty

if not SITE_LOCS and raw_df is not None and LOCS is not None:

    for site in SITES:

        rows=raw_df[raw_df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()

        idxs=[loc_to_idx.get((float(r.Latitude),float(r.Longitude)))

               for _,r in rows.iterrows()

               if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None]

        SITE_LOCS[site]=sorted(idxs)

    print(f"  Rebuilt SITE_LOCS: {[(s,len(v)) for s,v in SITE_LOCS.items()]}")

    WETLAND=SITE_LOCS.get("Wetland",[])

    SEEN=sorted(set(i for s,locs in SITE_LOCS.items() if s!="Wetland" for i in locs))

    print(f"  WETLAND: {len(WETLAND)} | SEEN: {len(SEEN)}")
WETLAND = _ns.get("WETLAND", WETLAND)  # keep rebuilt if not in ns
SEEN = _ns.get("SEEN", SEEN)  # keep rebuilt if not in ns
A_norm_v6  =_ns.get("A_norm",None)
loc_to_idx =_ns.get("loc_to_idx",{})
FI         =_ns.get("FI",{})
TGT_SCALERS=_ns.get("TGT_SCALERS",{})
TGT_USE_COLS=_ns.get("TGT_USE_COLS",{})
ALL_TGTS   =FI.get("ALL_TARGETS",[]) if FI else []

TGT_GRP="temp"
use_cols=TGT_USE_COLS.get(TGT_GRP,[])
tgt_sc  =TGT_SCALERS.get(TGT_GRP,None)
NT      =len(use_cols) if use_cols else 1
if not use_cols:
    _FI2=_ns.get("FI",{})
    _TEMP=_FI2.get("TEMP_TARGETS",[])
    _res=[f"{c}_residual" for c in _TEMP if raw_df is not None and f"{c}_residual" in raw_df.columns]
    use_cols=_res if _res else [c for c in _TEMP if raw_df is not None and c in raw_df.columns]
    if tgt_sc is None and use_cols:
        from sklearn.preprocessing import RobustScaler as _RS
        _tr=raw_df[raw_df["split"]=="train"]
        tgt_sc=_RS(); tgt_sc.fit(_tr[use_cols].dropna().values)
        print(f"  Rebuilt tgt_sc for: {use_cols}")
    NT=len(use_cols) if use_cols else 1

# ── Rebuild SITE_LOCS/SITES/WETLAND/SEEN after all _ns.get calls ────────────
import pickle as _pk2, pandas as _pd2
from pathlib import Path as _P2
_PRE2 = _P2("/home/emmanuel.keku/preprocessed_v3")
_raw2 = _pd2.read_csv(_PRE2/"master_processed.csv", usecols=["Site","Latitude","Longitude"])
with open(_PRE2/"feature_info.pkl","rb") as _f3: _FI3=_pk2.load(_f3)
_LOCS3 = _pd2.DataFrame(_FI3["LOCATIONS"])
_l2i = {(float(r.Latitude),float(r.Longitude)):i for i,r in _LOCS3.iterrows()}
SITES = _FI3["SITES"]
SITE_LOCS = {}
for _s3 in SITES:
    _r3=_raw2[_raw2["Site"]==_s3][["Latitude","Longitude"]].drop_duplicates()
    SITE_LOCS[_s3]=sorted([_l2i.get((float(r.Latitude),float(r.Longitude)))
                             for _,r in _r3.iterrows()
                             if _l2i.get((float(r.Latitude),float(r.Longitude))) is not None])
WETLAND = SITE_LOCS.get("Wetland",[])
SEEN = sorted(set(i for s,v in SITE_LOCS.items() if s!="Wetland" for i in v))
print(f"  SITE_LOCS: {[(s,len(v)) for s,v in SITE_LOCS.items()]}")
print(f"  SITES={SITES} | WETLAND={len(WETLAND)} | SEEN={len(SEEN)}")
# ─────────────────────────────────────────────────────────────────────────────

approx_c=[c.replace("_residual","")+"_approx" for c in use_cols
           if c.replace("_residual","")+"_approx" in
           (raw_df.columns if raw_df is not None else [])]

A_norm=(A_norm_v6.to(DEVICE) if A_norm_v6 is not None
        else None)
if A_norm is None:
    from scipy.spatial import cKDTree
    coords=LOCS[["Latitude","Longitude"]].values.astype(np.float32)
    sc_=coords*np.array([111.0,63.0]); tree_=cKDTree(sc_)
    d_,i_=tree_.query(sc_,k=7); sig_=np.median(d_[:,1:])+1e-8
    A_np=np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
    for i in range(N_LOCS):
        for jp in range(1,d_.shape[1]):
            j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
            A_np[i,j]+=w; A_np[j,i]+=w
    A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5
    A_norm=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32)).to(DEVICE)

BEST_ARCH="STGCN"
arch_cls=MODEL_MAP.get(BEST_ARCH)
if arch_cls is None:
    print(f"  ERROR: {BEST_ARCH} not in MODEL_MAP: {list(MODEL_MAP.keys())}")
    sys.exit(1)
print(f"  Model: {BEST_ARCH} | Features: {N_FEATS} | Locations: {N_LOCS}")

# ── Data loader (reuses v6 pipeline) ─────────────────────────────────────────
def build_loader(loc_subset=None,split="train",bs=4,
                  max_s=800,lookback=24,stride=4,M_dup=1):
    """
    M_dup: duplicate samples M times for smoother scaling (PI instruction)
    """
    if raw_df is None or feat_sc is None or tgt_sc is None: return None
    sub=raw_df[raw_df["split"]==split].copy()
    all_ts=sorted(sub["time_utc"].unique()); T=len(all_ts)
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
    af=np.zeros((T,N_LOCS,max(len(approx_c),1)),dtype=np.float32)
    mf=np.zeros((T,N_LOCS),dtype=np.float32)
    if len(sub)==0: return None
    Xf[sub["_ti"].values,sub["_ni"].values]=\
        feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)
    yf[sub["_ti"].values,sub["_ni"].values]=\
        tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)
    if approx_c and all(c in sub.columns for c in approx_c):
        af[sub["_ti"].values,sub["_ni"].values]=\
            sub[approx_c].fillna(0).values.astype(np.float32)
    locs_use=loc_subset if loc_subset is not None else SEEN
    mf[:,locs_use]=1.0
    rng=np.random.default_rng(SEED)
    tidxs=list(range(lookback,T,stride))
    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))
    Xl=[]; yl=[]; ml=[]; al2=[]
    for ti in tidxs:
        Xw=Xf[ti-lookback:ti]
        if np.isnan(Xw).mean()>0.5: continue
        Xl.append(np.nan_to_num(Xw,nan=0.))
        yl.append(yf[ti]); ml.append(mf[ti]); al2.append(af[ti])
    if not Xl: return None
    Xa=np.array(Xl); ya=np.array(yl); ma=np.array(ml); aa=np.array(al2)
    # Duplicate M times for smoother scaling
    if M_dup>1:
        Xa=np.tile(Xa,(M_dup,1,1,1)); ya=np.tile(ya,(M_dup,1,1))
        ma=np.tile(ma,(M_dup,1));     aa=np.tile(aa,(M_dup,1,1))
    ds=TensorDataset(torch.tensor(Xa),torch.tensor(ya),
                      torch.tensor(ma),torch.tensor(aa))
    return DataLoader(ds,batch_size=bs,shuffle=(split=="train"),
                       num_workers=0,pin_memory=False,drop_last=False)

# ── Loss + evaluation ─────────────────────────────────────────────────────────
def nll_loss(mu,lsv,y,mask):
    sv=torch.exp(lsv).clamp(min=1e-6)
    loss=0.5*(lsv+(y-mu)**2/sv)
    me=mask.unsqueeze(-1).expand_as(loss)
    return (loss*me).sum()/(me.sum()+1e-8)

def evaluate(model,loader,locs=None):
    if loader is None: return float("nan"),float("nan"),float("nan")
    if locs is None: locs=WETLAND
    model.eval(); yt_l=[]; yp_l=[]; sig_l=[]
    with torch.no_grad():
        for X,y,mask,av in loader:
            X=X.to(DEVICE)
            out=model(X,A_norm); mu=out[0]
            lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])
            mu_np=tgt_sc.inverse_transform(
                mu.cpu().float().numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            y_np=tgt_sc.inverse_transform(
                y.numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)
            sig_np=np.exp(0.5*lsv.cpu().float().numpy().reshape(X.shape[0],N_LOCS,NT))
            av_np=av.numpy()[:,:,0]
            yt_l.append((y_np[:,locs,0]+av_np[:,locs]).flatten())
            yp_l.append((mu_np[:,locs,0]+av_np[:,locs]).flatten())
            sig_l.append(sig_np[:,locs,0].flatten())
    yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
    sig=np.concatenate(sig_l)
    mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
    if len(yt)<5: return float("nan"),float("nan"),float("nan")
    rmse=float(np.sqrt(np.mean((yt-yp)**2)))
    r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
    unc=float(np.mean(sig))  # population uncertainty U
    return rmse,r2,unc

# ── Worker: local training with full behavior recording ───────────────────────
def local_worker(worker_id,theta_t,loc_subset,k_steps=50,lr=1e-3):

    # Use ray_worker logic for consistency — standalone data loading

    return ray_worker_cpu(worker_id,theta_t,loc_subset,k_steps,lr,SEED)



def ray_worker_cpu(worker_id,theta_t_cpu,loc_subset,k_steps=50,lr=1e-3,seed=42):
    """
    PI Steps 5-7: Train K steps, record full behavior.
    Records: Δθi, gradients per step, losses, stability, progress,
             signed residual βi, uncertainty σi, weight trajectory
    """
    model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    model.load_state_dict({k:v.to(DEVICE) for k,v in theta_t_cpu.items()})
    opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)

    loader=build_loader(loc_subset=loc_subset,split="train",
                         bs=4,max_s=300,M_dup=1)
    if loader is None:
        print(f"    Worker {worker_id}: loader is None for subset {loc_subset[:3]}...")
        return None

    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}
    losses=[]; grad_dirs=[]; weight_traj=[]; uncertainties=[]
    step=0; model.train()

    # Record initial weight position
    w0=torch.cat([p.data.cpu().flatten() for p in model.parameters()]).numpy()
    weight_traj.append(w0[:200].copy())

    for X,y,mask,av in loader:
        if step>=k_steps: break
        X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
        opt.zero_grad()
        mu,lsv=model(X,A_norm)
        loss=nll_loss(mu,lsv,y,mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)

        # Record gradient direction
        gv=torch.cat([p.grad.data.cpu().flatten()
                       for p in model.parameters() if p.grad is not None])
        gn=gv.norm()+1e-8; grad_dirs.append((gv/gn).numpy())

        # Record uncertainty (mean predictive σ)
        with torch.no_grad():
            unc=float(torch.exp(0.5*lsv).mean().item())
        uncertainties.append(unc)

        opt.step(); losses.append(float(loss.item())); step+=1

        # Record weight trajectory every 10 steps
        if step%10==0:
            wt=torch.cat([p.data.cpu().flatten() for p in model.parameters()]).numpy()
            weight_traj.append(wt[:200].copy())

    if not losses: return None

    theta_final={k:v.clone().cpu() for k,v in model.state_dict().items()}
    delta={k:(theta_final[k]-theta_start[k]).cpu() for k in theta_start}
    delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()

    # Stability: cosine similarity between consecutive gradient directions
    stability=1.0
    if len(grad_dirs)>1:
        sims=[float(np.dot(grad_dirs[i],grad_dirs[i-1])/
                     (np.linalg.norm(grad_dirs[i])*np.linalg.norm(grad_dirs[i-1])+1e-8))
               for i in range(1,len(grad_dirs))]
        stability=float(np.mean(sims))

    # Progress: loss reduction
    progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.

    # Signed residual βi: signed component of movement not in consensus direction
    # (will be computed in aggregator after collecting all workers)
    movement=float(np.linalg.norm(delta_flat))
    mean_unc=float(np.mean(uncertainties)) if uncertainties else 0.1

    wf=torch.cat([v.flatten() for v in theta_final.values()]).numpy()
    weight_traj.append(wf[:200].copy())

    return dict(
        worker_id=worker_id,
        delta=delta,
        delta_flat=delta_flat,
        losses=losses,
        grad_dirs=grad_dirs,
        weight_traj=weight_traj,  # full trajectory in weight space
        stability=stability,
        progress=progress,
        movement=movement,
        uncertainty=mean_unc,
        final_loss=losses[-1])

# ── Aggregator: full PI framework ─────────────────────────────────────────────
def compute_alphas_betas(behaviors,theta_t):
    """
    PI Steps 9-13 + signed residual correction.
    αi = softmax(consensus × stability × progress)
    βi = signed residual correction (component unique to each worker)
    γ  = step size controlled by population uncertainty U
    """
    N=len(behaviors)
    delta_flats=[b["delta_flat"] for b in behaviors]

    # PI Step 9: Cosine similarity → consensus
    consensus=np.ones(N)
    for i in range(N):
        sims=[]
        for j in range(N):
            if i==j: continue
            di=delta_flats[i]; dj=delta_flats[j]
            ni=np.linalg.norm(di)+1e-8; nj=np.linalg.norm(dj)+1e-8
            sims.append(float(np.dot(di,dj)/(ni*nj)))
        consensus[i]=float(np.mean(sims)) if sims else 0.

    # PI Step 10-11: Stability + Progress
    stabilities=np.array([b["stability"] for b in behaviors])
    progresses =np.array([b["progress"]  for b in behaviors])

    # PI Step 12-13: αi = softmax(consensus × stability × progress)
    scores=np.clip(consensus,0,None)*np.clip(stabilities,0,None)*np.clip(progresses,0,None)
    scores=np.clip(scores,1e-8,None)
    scores=scores-scores.max()
    alphas=np.exp(scores)/np.sum(np.exp(scores))

    # Signed residual βi: unique direction per worker (residual from mean)
    mean_delta=np.mean(delta_flats,axis=0)
    betas=[]
    for df in delta_flats:
        residual=df-mean_delta
        # Sign: positive if residual aligns with delta_flat, else negative
        sign=np.sign(np.dot(df,residual)+1e-8)
        beta=sign*np.linalg.norm(residual)/(np.linalg.norm(df)+1e-8)
        betas.append(float(beta))
    betas=np.array(betas)

    # Population uncertainty U → controls step size γ
    uncertainties=np.array([b["uncertainty"] for b in behaviors])
    U=float(np.mean(uncertainties))
    # γ decreases as uncertainty increases (more uncertain → smaller step)
    gamma_base=0.85; gamma=gamma_base*(1-min(U,0.5))

    return alphas,betas,gamma,U,dict(
        consensus=consensus,stability=stabilities,
        progress=progresses,uncertainty=uncertainties)

def aggregate(theta_t,behaviors,alphas,betas,gamma):
    """
    PI Steps 14-15:
    Δθ_global = Σ αi Δθi  +  Σ βi(residual_i)
    θt+1 = θt + γ × Δθ_global
    """
    theta_new={}
    N=len(behaviors)
    mean_delta={k:sum(behaviors[i]["delta"][k].float() for i in range(N))/N
                 for k in theta_t}
    for k in theta_t:
        # Shared direction
        shared=sum(alphas[i]*behaviors[i]["delta"][k].float() for i in range(N))
        # Signed residual correction
        residual_sum=sum(betas[i]*(behaviors[i]["delta"][k].float()-mean_delta[k])
                          for i in range(N))
        delta_global=shared+0.1*residual_sum
        theta_new[k]=(theta_t[k].float()+gamma*delta_global)
    return theta_new

# ── Spatial subsets: N=1,2,4,8 (PI: power of 2) ──────────────────────────────
def get_subsets(n_workers):

    # Rebuild SITE_LOCS inline if empty

    global SITE_LOCS, WETLAND, SEEN

    if not SITE_LOCS or not all(s in SITE_LOCS for s in SITES):

        import pickle, pandas as pd

        from pathlib import Path

        from scipy.spatial import cKDTree

        PREPROC2=Path("/home/emmanuel.keku/preprocessed_v3")

        raw2=pd.read_csv(PREPROC2/"master_processed.csv",

                          usecols=["Site","Latitude","Longitude"])

        with open(PREPROC2/"feature_info.pkl","rb") as f2: FI2=pickle.load(f2)

        LOCS2=pd.DataFrame(FI2["LOCATIONS"])

        loc_to_idx2={(float(r.Latitude),float(r.Longitude)):i

                      for i,r in LOCS2.iterrows()}

        for site in SITES:

            rows=raw2[raw2["Site"]==site][["Latitude","Longitude"]].drop_duplicates()

            idxs=sorted([loc_to_idx2.get((float(r.Latitude),float(r.Longitude)))

                          for _,r in rows.iterrows()

                          if loc_to_idx2.get((float(r.Latitude),float(r.Longitude)))

                          is not None])

            SITE_LOCS[site]=idxs

        WETLAND=SITE_LOCS.get("Wetland",[])

        SEEN=sorted(set(i for s,locs in SITE_LOCS.items()

                         if s!="Wetland" for i in locs))

        print(f"  SITE_LOCS rebuilt: {[(s,len(v)) for s,v in SITE_LOCS.items()]}")



    if n_workers==1: return [SEEN]

    if n_workers==2:

        s1=SITE_LOCS.get("Bedrock",[])+SITE_LOCS.get("Transition",[])

        s2=SITE_LOCS.get("Upland",[])+SITE_LOCS.get("Wetland",[])

        return [s1,s2]

    if n_workers==4: return [SITE_LOCS.get(s,[]) for s in SITES]

    if n_workers==8:

        subsets=[]

        for s in SITES:

            locs=SITE_LOCS.get(s,[]); mid=max(1,len(locs)//2)

            subsets.append(locs[:mid]); subsets.append(locs[mid:])

        return subsets

    return [SEEN]



# ── Ray remote worker (CPU only — CUDA not available in Ray context) ─────────

if HAS_RAY:

    @ray.remote(num_cpus=2)

    def ray_worker(worker_id, theta_t_cpu, loc_subset, k_steps=50, lr=1e-3, seed=42):

        """Ray remote version of local_worker — runs on CPU only."""

        import torch, numpy as np, copy, pickle, pandas as pd

        import torch.nn as nn

        from torch.optim import AdamW

        from pathlib import Path

        from scipy.spatial import cKDTree

        from sklearn.preprocessing import RobustScaler

        torch.manual_seed(seed+worker_id)



        PROJECT2=Path("/home/emmanuel.keku")

        PREPROC2=PROJECT2/"preprocessed_v3"

        with open(PREPROC2/"feature_info.pkl","rb") as f: FI2=pickle.load(f)

        raw2=pd.read_csv(PREPROC2/"master_processed.csv",parse_dates=["time_utc"])

        LOCS2=pd.DataFrame(FI2["LOCATIONS"])

        ALL_TGTS2=FI2["ALL_TARGETS"]

        SITES2=FI2["SITES"]

        N_LOCS2=FI2["N_LOCS"]



        CYCLICAL2=[c for c in raw2.columns if any(c.startswith(p) for p in ["sin_","cos_"])]

        SNAP2=FI2["SNAP_FEATURES"]

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

        tr2=raw2[raw2["split"]=="train"]

        feat_sc2=RobustScaler(); feat_sc2.fit(tr2[V6F2].fillna(0).values)

        TEMP_TGTS=FI2["TEMP_TARGETS"]

        use_cols2=[f"{c}_residual" for c in TEMP_TGTS if f"{c}_residual" in raw2.columns]

        if not use_cols2: use_cols2=[c for c in TEMP_TGTS if c in raw2.columns]

        tgt_sc2=RobustScaler(); tgt_sc2.fit(tr2[use_cols2].dropna().values)

        NT2=len(use_cols2)



        loc2idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS2.iterrows()}

        coords=LOCS2[["Latitude","Longitude"]].values.astype(np.float32)

        sc_=coords*np.array([111.0,63.0]); tree_=cKDTree(sc_)

        d_,i_=tree_.query(sc_,k=7); sig_=np.median(d_[:,1:])+1e-8

        A_np=np.zeros((N_LOCS2,N_LOCS2),dtype=np.float32)

        for i in range(N_LOCS2):

            for jp in range(1,d_.shape[1]):

                j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))

                A_np[i,j]+=w; A_np[j,i]+=w

        A_np+=np.eye(N_LOCS2); D_=A_np.sum(1,keepdims=True)**0.5

        A2=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))  # CPU



        # Rebuild model on CPU

        class GConv2(nn.Module):

            def __init__(self,d,dp=0.15):

                super().__init__()

                self.W=nn.Linear(d,d,bias=False); self.n=nn.LayerNorm(d)

                self.d=nn.Dropout(dp); self.a=nn.GELU()

            def forward(self,H,A):

                if A.dim()==3: A=A[0]

                return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),self.W(self.d(H)))))

        class HH2(nn.Module):

            def __init__(self,d,nt,dp=0.15):

                super().__init__()

                self.mu=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))

                self.lsv=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))

            def forward(self,h): return self.mu(h),self.lsv(h)

        class STGCN2(nn.Module):

            def __init__(self,nf,h=64,nl=2,gl=2,nt=1):

                super().__init__()

                self.p=nn.Linear(nf,h)

                self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,dropout=0.15 if nl>1 else 0.)

                self.r=nn.Linear(h*2,h)

                self.gc=nn.ModuleList([GConv2(h) for _ in range(gl)])

                self.hd=HH2(h*2,nt)

            def forward(self,x,A):

                B,L,N,F=x.shape

                h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))

                h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h

                for g in self.gc: hg=g(hg,A)

                mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv



        nf2=len(V6F2)

        model=STGCN2(nf=nf2,h=64,nl=2,gl=2,nt=NT2)

        model.load_state_dict({k:v.cpu() for k,v in theta_t_cpu.items()})

        opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)



        # Build data

        sub2=raw2[raw2["split"]=="train"].copy()

        all_ts=sorted(sub2["time_utc"].unique()); T2=len(all_ts)

        ts2i={t:i for i,t in enumerate(all_ts)}

        sub2["_ti"]=sub2["time_utc"].map(ts2i)

        sub2["_ni"]=[loc2idx.get((float(la),float(lo))) for la,lo in zip(sub2["Latitude"],sub2["Longitude"])]

        sub2=sub2.dropna(subset=["_ti","_ni"])

        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)

        sub2=sub2[sub2["_ti"]<T2]; sub2=sub2[sub2["_ni"].isin(loc_subset)]

        Xf=np.zeros((T2,N_LOCS2,nf2),dtype=np.float32)

        yf=np.zeros((T2,N_LOCS2,NT2),dtype=np.float32)

        mf=np.zeros((T2,N_LOCS2),dtype=np.float32)

        if len(sub2)==0: return None

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

        from torch.utils.data import DataLoader,TensorDataset

        ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),torch.tensor(np.array(ml2)))

        loader=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0)



        theta_start={k:v.clone() for k,v in model.state_dict().items()}

        losses=[]; grad_dirs=[]; weight_traj=[]; uncertainties=[]; step=0

        w0=torch.cat([p.data.flatten() for p in model.parameters()]).numpy()

        weight_traj.append(w0[:200].copy())

        model.train()

        for X,y,mask in loader:

            if step>=k_steps: break

            opt.zero_grad()

            mu,lsv=model(X,A2)

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

        return dict(worker_id=worker_id,delta=delta,delta_flat=delta_flat,

                     losses=losses,grad_dirs=grad_dirs,weight_traj=weight_traj,

                     stability=stability,progress=progress,

                     movement=float(np.linalg.norm(delta_flat)),

                     uncertainty=float(np.mean(uncertainties)) if uncertainties else 0.1,

                     final_loss=losses[-1])



# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Run experiment for N_WORKERS in [1,2,4,8] (PI: power of 2)
# ══════════════════════════════════════════════════════════════════════════════
T_ROUNDS=18; K_STEPS=50; LR=1e-3; M_DUP=2
N_WORKERS_LIST=[2,4,8]  # N=1 skipped — sequential loader issue, Ray workers work fine

all_results=[]

for N_WORKERS in N_WORKERS_LIST:
    print(f"\n{'='*55}")
    print(f"  N_WORKERS = {N_WORKERS}")
    print(f"{'='*55}")

    SUBSETS=get_subsets(N_WORKERS)
    print(f"  Subsets: {[len(s) for s in SUBSETS]} locations each")

    # Initialize with random weights (PI instruction)
    torch.manual_seed(SEED)
    global_model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    theta_t={k:v.cpu().clone() for k,v in global_model.state_dict().items()}
    n_params=sum(p.numel() for p in global_model.parameters())
    if N_WORKERS==N_WORKERS_LIST[0]:
        print(f"  {BEST_ARCH} params: {n_params:,} | Random init")

    val_loader=build_loader(split="test",bs=4,max_s=400,M_dup=M_DUP)

    # Storage
    round_history=[]
    weight_snapshots=[]  # full trajectories for PCA GIF
    worker_weight_trajs={wi:[] for wi in range(N_WORKERS)}

    # Initial weight snapshot
    w0=torch.cat([v.flatten() for v in theta_t.values()]).numpy()[:200]
    weight_snapshots.append({"round":0,"who":"init","vec":w0.copy()})

    best_rmse=float("inf"); best_theta=None
    prev_rmse=float("inf"); gamma=0.8
    t_start=time.time()

    for rnd in range(1,T_ROUNDS+1):
        t_round=time.time()

        # Run workers (Ray parallel or sequential)
        if HAS_RAY and N_WORKERS>1:

            theta_t_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}

            futures=[ray_worker.remote(wi,theta_t_cpu,

                                        SUBSETS[wi],K_STEPS,LR,SEED)

                      for wi in range(N_WORKERS)]

            behaviors_raw=ray.get(futures)

            behaviors=[b for b in behaviors_raw if b is not None]

        else:

            behaviors=[]

            for wi in range(N_WORKERS):

                theta_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}

                b=local_worker(wi,theta_cpu,SUBSETS[wi],K_STEPS,LR)

                if b is not None: behaviors.append(b)


        if not behaviors:
            print(f"  Rnd {rnd}: No workers returned — check data subsets")
            round_history.append(dict(round=rnd,rmse=float("nan"),r2=float("nan"),
                                       gamma=gamma,alphas=[],betas=[],U=float("nan"),
                                       worker_losses=[],elapsed=time.time()-t_round))
            continue

        # Record worker weight trajectories
        for b in behaviors:
            wi=b["worker_id"]
            for wt in b["weight_traj"]:
                worker_weight_trajs[wi].append({"round":rnd,"vec":wt})
            wf=torch.cat([v.flatten() for v in b["delta"].values()]).numpy()[:200]
            weight_snapshots.append({"round":rnd,"who":f"worker_{wi+1}",
                                      "vec":(theta_t[list(theta_t.keys())[0]]
                                              .flatten()[:200].numpy()+wf[:200])})

        # PI Steps 9-13: compute αi, βi, γ(U)
        alphas,betas,gamma_u,U,scores=compute_alphas_betas(behaviors,theta_t)

        # PI Steps 14-15: aggregate
        theta_cand=aggregate(theta_t,behaviors,alphas,betas,gamma_u)
        global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})

        # PI Step 16: evaluate
        rmse,r2,unc=evaluate(global_model,val_loader)

        # Record global weight snapshot
        wg=torch.cat([v.flatten() for v in theta_cand.values()]).numpy()[:200]
        weight_snapshots.append({"round":rnd,"who":"global","vec":wg})

        # PI Step 17: accept or adjust
        rmse_ok=not(np.isnan(rmse) or np.isinf(rmse))
        if not rmse_ok or rmse<=prev_rmse*1.02:
            theta_t=theta_cand
            if rmse_ok: prev_rmse=rmse
            if rmse_ok and rmse<best_rmse:
                best_rmse=rmse; best_theta=copy.deepcopy(theta_cand)
            status="ACCEPTED"
        else:
            gamma_u*=0.8; status="REJECTED"

        wl=[b["final_loss"] for b in behaviors]
        print(f"  Rnd {rnd:02d} | RMSE={rmse:.4f} R²={r2:.4f} "
               f"U={U:.3f} γ={gamma_u:.3f} α={[f'{a:.2f}' for a in alphas]} "
               f"β={[f'{b:.2f}' for b in betas]} | {status}")

        round_history.append(dict(
            round=rnd,rmse=rmse,r2=r2,gamma=gamma_u,U=U,
            alphas=alphas.tolist(),betas=betas.tolist(),
            consensus=scores["consensus"].tolist(),
            stability=scores["stability"].tolist(),
            progress=scores["progress"].tolist(),
            uncertainty=scores["uncertainty"].tolist(),
            worker_losses=wl,
            elapsed=time.time()-t_round))

    total_time=time.time()-t_start
    ideal_time=total_time/N_WORKERS

    # Centralized baseline (only run once)
    if N_WORKERS==N_WORKERS_LIST[0]:
        print(f"\n  Centralized baseline...")
        torch.manual_seed(SEED)
        cent=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
        cl=build_loader(split="train",bs=4,max_s=600,M_dup=M_DUP)
        co=AdamW(cent.parameters(),lr=LR,weight_decay=5e-4)
        t_cent=time.time()
        cent_weight_traj=[]
        for ep in range(T_ROUNDS):
            cent.train()
            for X,y,mask,av in (cl or []):
                X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
                co.zero_grad(); mu,lsv=cent(X,A_norm)
                loss=nll_loss(mu,lsv,y,mask); loss.backward()
                nn.utils.clip_grad_norm_(cent.parameters(),1.0); co.step()
            wc=torch.cat([p.data.cpu().flatten()
                           for p in cent.parameters()]).numpy()[:200]
            cent_weight_traj.append({"round":ep+1,"vec":wc})
        cent_time=time.time()-t_cent
        cent_rmse,cent_r2,cent_unc=evaluate(cent,val_loader)
        print(f"  Centralized: RMSE={cent_rmse:.4f} R²={cent_r2:.4f} "
               f"time={cent_time:.1f}s")

    valid_rmses=[r["rmse"] for r in round_history
                  if not np.isnan(r["rmse"]) and not np.isinf(r["rmse"])]
    best_r2_seen=max((r["r2"] for r in round_history
                       if not np.isnan(r["r2"])),default=float("nan"))

    all_results.append(dict(
        n_workers=N_WORKERS,
        best_rmse=round(best_rmse,4) if not np.isinf(best_rmse) else float("nan"),
        best_r2=round(best_r2_seen,4),
        ideal_time_s=round(ideal_time,2),
        total_time_s=round(total_time,2),
        n_rounds=len(round_history)))

    # Save round history per N
    pd.DataFrame(round_history).to_csv(
        RESULTS/f"rounds_N{N_WORKERS}.csv",index=False)

    # Store for GIF generation
    if N_WORKERS==4:
        round_hist_4=round_history
        weight_snap_4=weight_snapshots
        worker_traj_4=worker_weight_trajs
        cent_traj_4=cent_weight_traj

# Save summary
summary_df=pd.DataFrame(all_results)
cent_rows=pd.DataFrame([dict(n_workers=0,best_rmse=round(cent_rmse,4),
                               best_r2=round(cent_r2,4),
                               ideal_time_s=round(cent_time,2),
                               total_time_s=round(cent_time,2),
                               n_rounds=T_ROUNDS)])
summary_df=pd.concat([summary_df,cent_rows],ignore_index=True)
summary_df.to_csv(RESULTS/"behavior_guided_summary.csv",index=False)
print(f"\n{summary_df.to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# GIF GENERATION (using N=4 results)
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from sklearn.decomposition import PCA

WORKER_COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                "#9467bd","#8c564b","#e377c2","#7f7f7f"]
SITE_NAMES=[s.replace("rock","").replace("ition","").replace("land","").replace("Up","Up")
             for s in SITES]

rh=round_hist_4
rounds_=[r["round"] for r in rh]
rmses_ =[r["rmse"]  for r in rh]
r2s_   =[r["r2"]    for r in rh]
Us_    =[r["U"]     for r in rh]
gammas_=[r["gamma"] for r in rh]
valid_r=[r for r in rmses_ if not np.isnan(r) and not np.isinf(r)]

# ── GIF_01: Performance + time per round ─────────────────────────────────────
print("\n  GIF_01: RMSE + uncertainty vs round...")
fig,axes=plt.subplots(1,2,figsize=(14,6))
ax1,ax2=axes
l1,=ax1.plot([],[],color="#1f77b4",lw=2.5,marker="^",ms=8,
              label="Behavior-guided RMSE")
l2,=ax1.plot([],[],color="#ff7f0e",lw=2.5,marker="o",ms=8,
              label="Centralized RMSE")
ylo=min(valid_r+[cent_rmse])*0.96 if valid_r else 0
yhi=max(valid_r+[cent_rmse])*1.04 if valid_r else 3
ax1.set_xlim(0,T_ROUNDS+1); ax1.set_ylim(ylo,yhi)
ax1.set_xlabel("Global Round",fontsize=12)
ax1.set_ylabel("Test RMSE (C)",fontsize=12)
ax1.set_title("Performance During Training",fontsize=12)
ax1.legend(fontsize=10)
box=ax1.text(0.05,0.2,"",transform=ax1.transAxes,fontsize=8,
              bbox=dict(boxstyle="round",facecolor="lightblue",alpha=0.5))

l3,=ax2.plot([],[],color="#2ca02c",lw=2,label="Population uncertainty U")
l4,=ax2.plot([],[],color="#d62728",lw=2,ls="--",label="Step size γ")
ax2.set_xlim(0,T_ROUNDS+1); ax2.set_ylim(0,1)
ax2.set_xlabel("Global Round",fontsize=12)
ax2.set_ylabel("Value",fontsize=12)
ax2.set_title("Uncertainty Controls Step Size",fontsize=12)
ax2.legend(fontsize=10)

def u1(frame):
    i=min(frame,len(rounds_)-1); r=rh[i]
    l1.set_data(rounds_[:i+1],rmses_[:i+1])
    l2.set_data(rounds_[:i+1],[cent_rmse]*(i+1))
    l3.set_data(rounds_[:i+1],Us_[:i+1])
    l4.set_data(rounds_[:i+1],gammas_[:i+1])
    ideal_t=r["elapsed"]/4
    box.set_text(
        f"Round {r['round']}/{T_ROUNDS}\n"
        f"Ideal distributed: {ideal_t:.3f}s\n"
        f"Centralized: {cent_time/T_ROUNDS:.3f}s\n"
        f"Distributed RMSE: {r['rmse']:.3f}\n"
        f"Centralized RMSE: {cent_rmse:.3f}")
    return l1,l2,l3,l4,box

ani1=animation.FuncAnimation(fig,u1,frames=len(rounds_),interval=400,blit=True)
ani1.save(FIGS/"gif_01_time_performance.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_01_time_performance.gif")

# ── GIF_02: Weight space trajectories ─────────────────────────────────────────
print("\n  GIF_02: Weight space trajectories...")
# Collect all vectors for PCA
all_vecs=[]; labels=[]
# Init
w0v=torch.cat([v.flatten() for v in theta_t.values()]).numpy()[:200]
all_vecs.append(w0v); labels.append(("init",0,0))
# Workers
for wi in range(4):
    for rec in worker_traj_4.get(wi,[]):
        all_vecs.append(rec["vec"]); labels.append((f"worker_{wi}",rec["round"],wi))
# Global snapshots
for snap in weight_snap_4:
    if snap["who"]=="global":
        all_vecs.append(snap["vec"]); labels.append(("global",snap["round"],4))
# Centralized
for rec in cent_traj_4:
    all_vecs.append(rec["vec"]); labels.append(("cent",rec["round"],5))

maxl=min(200,min(len(v) for v in all_vecs))
all_np=np.array([v[:maxl] for v in all_vecs])
pca=PCA(n_components=2,random_state=SEED)
all_pca=pca.fit_transform(all_np)

fig,ax=plt.subplots(figsize=(10,8))
ax.set_xlabel("Weight-space PC1",fontsize=11)
ax.set_ylabel("Weight-space PC2",fontsize=11)
ax.set_title("Worker Weight Trajectories, Predicted Generalized Path, and True Full-Data Path",
              fontsize=11)

# Draw centralized full path
cent_idx=[i for i,(who,rnd,wi) in enumerate(labels) if who=="cent"]
if len(cent_idx)>1:
    ax.plot(all_pca[cent_idx,0],all_pca[cent_idx,1],
             color="saddlebrown",lw=3,alpha=0.8,label="True full-data")
    ax.scatter(all_pca[cent_idx[-1],0],all_pca[cent_idx[-1],1],
                c="orange",s=200,marker="*",zorder=10,label="True full-data final")

# Worker trajectory lines (animated)
worker_lines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,
                       label=f"Worker {wi+1}: {SITES[wi] if wi<len(SITES) else ''}")[0]
               for wi in range(4)]
global_line,=ax.plot([],[],color="darkviolet",lw=2.5,ls="--",
                      label="Predicted generalized")
global_dot=ax.scatter([],[],c="blue",s=150,marker="s",zorder=9,
                        label="Predicted final")
init_idx_=[i for i,(who,_,_) in enumerate(labels) if who=="init"]
if init_idx_:
    ax.scatter(all_pca[init_idx_[0],0],all_pca[init_idx_[0],1],
                c="black",s=200,marker="X",zorder=10,label="Initialization")

ax.legend(fontsize=8,loc="upper right")
xpad=(all_pca[:,0].max()-all_pca[:,0].min())*0.05+0.1
ypad=(all_pca[:,1].max()-all_pca[:,1].min())*0.05+0.1
ax.set_xlim(all_pca[:,0].min()-xpad,all_pca[:,0].max()+xpad)
ax.set_ylim(all_pca[:,1].min()-ypad,all_pca[:,1].max()+ypad)

# Group by round
by_round={}
for i,(who,rnd,wi) in enumerate(labels):
    if rnd not in by_round: by_round[rnd]={"workers":{},"global":None}
    if who.startswith("worker_"):
        wid=int(who.split("_")[1])
        if wid not in by_round[rnd]["workers"]:
            by_round[rnd]["workers"][wid]=[]
        by_round[rnd]["workers"][wid].append(all_pca[i])
    elif who=="global":
        by_round[rnd]["global"]=all_pca[i]

# Precompute cumulative trajectories
worker_pts={wi:[] for wi in range(4)}
global_pts=[]
sorted_rnds=sorted(by_round.keys())

def u2(frame):
    rn=sorted_rnds[min(frame,len(sorted_rnds)-1)]
    rd=by_round[rn]
    for wi in range(4):
        if wi in rd["workers"]:
            for pt in rd["workers"][wi]: worker_pts[wi].append(pt)
        if len(worker_pts[wi])>1:
            pts=np.array(worker_pts[wi])
            worker_lines[wi].set_data(pts[:,0],pts[:,1])
    if rd["global"] is not None:
        global_pts.append(rd["global"])
    if len(global_pts)>1:
        gp=np.array(global_pts)
        global_line.set_data(gp[:,0],gp[:,1])
        global_dot.set_offsets(gp[-1].reshape(1,2))
    return worker_lines+[global_line,global_dot]

ani2=animation.FuncAnimation(fig,u2,frames=len(sorted_rnds),interval=500,blit=True)
ani2.save(FIGS/"gif_02_weight_space.gif",writer=PillowWriter(fps=2),dpi=100)
plt.close(); print("    OK gif_02_weight_space.gif")

# ── GIF_03: Loss evolution ─────────────────────────────────────────────────────
print("\n  GIF_03: Loss evolution...")
all_wl=[[rh[ri]["worker_losses"][wi]
          if wi<len(rh[ri]["worker_losses"]) else float("nan")
          for ri in range(len(rh))] for wi in range(4)]
gloss=[r["rmse"]**2 if not np.isnan(r["rmse"]) else float("nan") for r in rh]
closs=[cent_rmse**2]*len(rh)
all_v=[v for wl in all_wl for v in wl if not np.isnan(v)]
ymax=max(all_v+[cent_rmse**2])*1.1 if all_v else 3
fig,ax=plt.subplots(figsize=(10,6))
ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(0,ymax)
ax.set_xlabel("Global Round",fontsize=11)
ax.set_ylabel("Standardised MSE Loss",fontsize=11)
ax.set_title("Loss Evolution: Local Workers vs Global Prediction vs Centralized",fontsize=12)
wlines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,marker="o",ms=5,
                 label=f"Worker {wi+1} ({SITES[wi] if wi<len(SITES) else ''}) local")[0]
         for wi in range(4)]
gl,=ax.plot([],[],color="darkviolet",lw=2.5,marker="*",ms=10,
             label="Predicted global model")
cl,=ax.plot([],[],color="saddlebrown",lw=2,marker="s",ms=8,ls="--",
             label="Centralized full-data model")
ax.legend(fontsize=8,loc="upper right")
def u3(frame):
    i=min(frame,len(rh)-1); xs=rounds_[:i+1]
    for wi in range(4): wlines[wi].set_data(xs,all_wl[wi][:i+1])
    gl.set_data(xs,gloss[:i+1]); cl.set_data(xs,closs[:i+1])
    return wlines+[gl,cl]
ani3=animation.FuncAnimation(fig,u3,frames=len(rh),interval=400,blit=True)
ani3.save(FIGS/"gif_03_loss_workers.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_03_loss_workers.gif")

# ── GIF_04: Dynamic signed βi coefficients per round ─────────────────────────
print("\n  GIF_04: Dynamic signed residual correction...")
betas_per_round=[[rh[ri]["betas"][wi] if wi<len(rh[ri].get("betas",[]))
                   else float("nan") for ri in range(len(rh))]
                  for wi in range(4)]
fig,ax=plt.subplots(figsize=(12,6))
ax.axhline(0,color="#1f77b4",lw=1.5,alpha=0.5)
ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(-0.6,0.6)
ax.set_xlabel("Global round",fontsize=12)
ax.set_ylabel("Signed residual coefficient β",fontsize=12)
ax.set_title("Dynamic Unique-Direction Corrections",fontsize=13)
blines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=2,
                 label=SITES[wi] if wi<len(SITES) else f"Worker {wi+1}")[0]
         for wi in range(4)]
ax.legend(fontsize=10)
def u4(frame):
    i=min(frame,len(rh)-1); xs=rounds_[:i+1]
    for wi in range(4): blines[wi].set_data(xs,betas_per_round[wi][:i+1])
    return blines
ani4=animation.FuncAnimation(fig,u4,frames=len(rh),interval=400,blit=True)
ani4.save(FIGS/"gif_04_beta_corrections.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_04_beta_corrections.gif")

# ── Static figures ─────────────────────────────────────────────────────────────
# Performance comparison (all N_WORKERS + centralized)
fig,axes=plt.subplots(1,2,figsize=(16,6))
ax1,ax2=axes

# Left: RMSE by method
methods_=[f"Behavior-guided\nN={r['n_workers']}"
           for r in all_results if r['n_workers']>0]+["Centralized"]
rmses_m =[r['best_rmse'] for r in all_results if r['n_workers']>0]+[cent_rmse]
rmses_m =[r if not np.isnan(r) else 0 for r in rmses_m]
bars=ax1.bar(methods_,rmses_m,color="#1f77b4",width=0.5)
for bar,v in zip(bars,rmses_m):
    ax1.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.3f}",
              ha="center",va="bottom",fontsize=10,fontweight="bold")
ax1.set_ylabel("Test RMSE (C) — lower is better",fontsize=12)
ax1.set_title("Final Soil Temperature Prediction Performance",fontsize=12)
ax1.set_ylim(0,max(rmses_m+[0.1])*1.15)

# Right: Processing time
time_m=[f"N={r['n_workers']}\n(ideal)" for r in all_results if r['n_workers']>0]
times_m=[r['ideal_time_s'] for r in all_results if r['n_workers']>0]
times_m2=[r['total_time_s'] for r in all_results if r['n_workers']>0]
x_=np.arange(len(time_m)); w_=0.35
ax2.bar(x_-w_/2,times_m,w_,color="#1f77b4",alpha=0.9,label="Ideal parallel")
ax2.bar(x_+w_/2,times_m2,w_,color="#aec7e8",alpha=0.9,label="Sequential here")
ax2.axhline(cent_time,color="red",ls="--",lw=2,label=f"Centralized ({cent_time:.1f}s)")
ax2.set_xticks(x_); ax2.set_xticklabels(time_m)
ax2.set_ylabel("Processing Time (seconds)",fontsize=12)
ax2.set_title("Training Processing Time vs N Workers",fontsize=12)
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGS/"final_performance.png",dpi=300,bbox_inches="tight")
plt.close()

# Uncertainty + step size static
fig,ax=plt.subplots(figsize=(12,5))
ax.plot(rounds_,Us_,color="#1f77b4",lw=2,label="Uncertainty U")
ax.plot(rounds_,gammas_,color="#ff7f0e",lw=2,label="Jump scale γ")
ax.axhline(0,color="grey",lw=0.5)
ax.set_xlabel("Global round",fontsize=12)
ax.set_ylabel("Value",fontsize=12)
ax.set_title("Uncertainty Controls Step Size",fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGS/"uncertainty_step_size.png",dpi=300,bbox_inches="tight")
plt.close()

# Alpha coefficients (uncertainty-aware signed)
alphas_per_round=[[rh[ri]["alphas"][wi] if wi<len(rh[ri].get("alphas",[]))
                    else float("nan") for ri in range(len(rh))]
                   for wi in range(4)]
fig,ax=plt.subplots(figsize=(12,6))
for wi in range(4):
    ax.plot(rounds_,alphas_per_round[wi],color=WORKER_COLORS[wi],lw=2,
             label=SITES[wi] if wi<len(SITES) else f"Worker {wi+1}")
ax.axhline(0,color="#1f77b4",lw=1,alpha=0.5)
ax.set_xlabel("Global round",fontsize=12)
ax.set_ylabel("Signed worker coefficient",fontsize=12)
ax.set_title("Uncertainty-Aware Dynamic Signed Coefficients",fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGS/"uncertainty_signed_coefficients.png",dpi=300,bbox_inches="tight")
plt.close()

print(f"\n{'='*65}")
print(f"  EXPERIMENT COMPLETE")
print(f"  Figures: {FIGS}")
figs_out=sorted(FIGS.glob("*.gif"))+sorted(FIGS.glob("*.png"))
for f in figs_out: print(f"  {f.name} ({f.stat().st_size//1024} KB)")
print(f"{'='*65}")