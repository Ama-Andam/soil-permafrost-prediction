"""
experiment_behavior_guided_v2.py
BEHAVIOR-GUIDED DISTRIBUTED EXPERIMENT — Clean standalone version
Imports STGCN architecture from train_soil_spatial_v6.py
Builds all data/scalers independently (no dependency on v6 namespace variables)
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
print("  BEHAVIOR-GUIDED DISTRIBUTED EXPERIMENT v2")
print("  STGCN | Soil Temperature | Random init | Standalone")
print("="*65)

import torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import AdamW
from sklearn.preprocessing import RobustScaler
from scipy.spatial import cKDTree
torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# ── Ray ───────────────────────────────────────────────────────────────────────
try:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,num_cpus=8,
                  num_gpus=0)  # workers use CPU
    HAS_RAY=True; print(f"  Ray: {ray.__version__}")
except Exception as e:
    HAS_RAY=False; print(f"  Ray: not available — sequential")

# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE DATA SETUP (no v6 namespace dependency)
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
APPROX=[f"{t}_approx"    for t in ALL_TGTS if f"{t}_approx"    in raw_df.columns]
RESID =[f"{t}_residual"  for t in ALL_TGTS if f"{t}_residual"  in raw_df.columns]
UNC=[]
for feat in CORE[:8]:
    vc=f"{feat}_unc_var"
    if vc not in raw_df.columns: raw_df[vc]=np.where(raw_df[feat].isna(),1.0,0.01)
    UNC.append(vc)
V6F=list(dict.fromkeys(CORE+APPROX+RESID+UNC))
V6F=[f for f in V6F if f in raw_df.columns]
N_FEATS=len(V6F)

# Target
TEMP_TGTS=FI["TEMP_TARGETS"]
use_cols=[f"{c}_residual" for c in TEMP_TGTS if f"{c}_residual" in raw_df.columns]
if not use_cols: use_cols=[c for c in TEMP_TGTS if c in raw_df.columns]
approx_c=[f"{c}_approx"  for c in TEMP_TGTS if f"{c}_approx"   in raw_df.columns]
NT=len(use_cols)

# Scalers
tr_df=raw_df[raw_df["split"]=="train"]
feat_sc=RobustScaler(); feat_sc.fit(tr_df[V6F].fillna(0).values)
tgt_sc =RobustScaler(); tgt_sc.fit(tr_df[use_cols].dropna().values)
print(f"  Features: {N_FEATS} | Target: {use_cols} | NT: {NT}")

# Location index
loc_to_idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS.iterrows()}

# SITE_LOCS
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
print(f"  WETLAND: {len(WETLAND)} | SEEN: {len(SEEN)}")

# Spatial graph
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
print(f"  Graph: σ={sig_:.2f}km | shape={A_norm.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL: Import STGCN from v6
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Importing STGCN from v6...")
_src=open(PROJECT/"train_soil_spatial_v6.py").read()
_pre=_src.split("if args.mode")[0]
_ns={"__name__":"__imported__"}
import unittest.mock as _mock
with _mock.patch("sys.argv",["train_soil_spatial_v6.py","--mode","train","--target","temp"]):
    try: exec(_pre,_ns)
    except SystemExit: pass
    except Exception as e: print(f"  Warning: {e}")

MODEL_MAP=_ns.get("MODEL_MAP",{})
arch_cls=MODEL_MAP.get("STGCN")
if arch_cls is None:
    print(f"  ERROR: STGCN not in MODEL_MAP: {list(MODEL_MAP.keys())}")
    sys.exit(1)
print(f"  STGCN loaded | Tier: {getattr(arch_cls,'tier','GRAPH')}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER (standalone — uses local scalers)
# ══════════════════════════════════════════════════════════════════════════════
def build_loader(loc_subset=None,split="train",bs=4,
                  max_s=800,lookback=24,stride=4,M_dup=1):
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
    if len(sub)==0: return None

    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)
    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32)
    af=np.zeros((T,N_LOCS,max(len(approx_c),1)),dtype=np.float32)
    mf=np.zeros((T,N_LOCS),dtype=np.float32)
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
    if M_dup>1:
        Xa=np.tile(Xa,(M_dup,1,1,1)); ya=np.tile(ya,(M_dup,1,1))
        ma=np.tile(ma,(M_dup,1));     aa=np.tile(aa,(M_dup,1,1))
    ds=TensorDataset(torch.tensor(Xa),torch.tensor(ya),
                      torch.tensor(ma),torch.tensor(aa))
    return DataLoader(ds,batch_size=bs,shuffle=(split=="train"),
                       num_workers=0,pin_memory=False,drop_last=False)

# Test loaders
_tl=build_loader(split="train",max_s=100)
_vl=build_loader(split="test", max_s=100)
print(f"  Train loader: {_tl is not None} | Val loader: {_vl is not None}")

# ══════════════════════════════════════════════════════════════════════════════
# LOSS + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
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
            # PI: evaluate on RESIDUAL only — no reconstruction
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
# WORKER: local training with behavior recording
# ══════════════════════════════════════════════════════════════════════════════
def local_worker(worker_id,theta_t_cpu,loc_subset,k_steps=50,lr=1e-3):
    model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    model.load_state_dict({k:v.to(DEVICE) for k,v in theta_t_cpu.items()})
    opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    loader=build_loader(loc_subset=loc_subset,split="train",bs=4,max_s=300)
    if loader is None:
        print(f"    Worker {worker_id}: loader None — subset={len(loc_subset)} locs")
        return None
    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}
    losses=[]; grad_dirs=[]; weight_traj=[]; uncertainties=[]; step=0
    w0=torch.cat([p.data.cpu().flatten() for p in model.parameters()]).numpy()
    weight_traj.append(w0[:200].copy()); model.train()
    for X,y,mask,av in loader:
        if step>=k_steps: break
        X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
        opt.zero_grad()
        mu,lsv=model(X,A_norm); loss=nll_loss(mu,lsv,y,mask)
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
    return dict(worker_id=worker_id,delta=delta,delta_flat=delta_flat,
                 losses=losses,grad_dirs=grad_dirs,weight_traj=weight_traj,
                 stability=stability,progress=progress,
                 movement=float(np.linalg.norm(delta_flat)),
                 uncertainty=float(np.mean(uncertainties)) if uncertainties else 0.1,
                 final_loss=losses[-1])

# ── Ray remote worker (CPU only) ──────────────────────────────────────────────
if HAS_RAY:
    @ray.remote(num_cpus=2)
    def ray_worker(worker_id,theta_t_cpu,loc_subset,k_steps=50,lr=1e-3,seed=42):
        import torch,torch.nn as nn,numpy as np,pickle,pandas as pd,warnings
        from pathlib import Path; from sklearn.preprocessing import RobustScaler
        from scipy.spatial import cKDTree; from torch.optim import AdamW
        from torch.utils.data import DataLoader,TensorDataset
        warnings.filterwarnings("ignore"); torch.manual_seed(seed+worker_id)

        PROJECT2=Path("/home/emmanuel.keku"); PREPROC2=PROJECT2/"preprocessed_v3"
        with open(PREPROC2/"feature_info.pkl","rb") as f: FI2=pickle.load(f)
        raw2=pd.read_csv(PREPROC2/"master_processed.csv",parse_dates=["time_utc"])
        LOCS2=pd.DataFrame(FI2["LOCATIONS"]); N_LOCS2=FI2["N_LOCS"]
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
        V6F2=list(dict.fromkeys(CORE2+APPROX2+RESID2+UNC2)); V6F2=[f for f in V6F2 if f in raw2.columns]
        TEMP2=FI2["TEMP_TARGETS"]
        use_cols2=[f"{c}_residual" for c in TEMP2 if f"{c}_residual" in raw2.columns]
        if not use_cols2: use_cols2=[c for c in TEMP2 if c in raw2.columns]
        approx2=[f"{c}_approx" for c in TEMP2 if f"{c}_approx" in raw2.columns]
        NT2=len(use_cols2)
        tr2=raw2[raw2["split"]=="train"]
        feat_sc2=RobustScaler(); feat_sc2.fit(tr2[V6F2].fillna(0).values)
        tgt_sc2=RobustScaler(); tgt_sc2.fit(tr2[use_cols2].dropna().values)
        loc2idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS2.iterrows()}
        coords=LOCS2[["Latitude","Longitude"]].values.astype(np.float32)*np.array([111.0,63.0])
        tree_=cKDTree(coords); d_,i_=tree_.query(coords,k=7); sig_=np.median(d_[:,1:])+1e-8
        A_np=np.zeros((N_LOCS2,N_LOCS2),dtype=np.float32)
        for i in range(N_LOCS2):
            for jp in range(1,d_.shape[1]):
                j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_)); A_np[i,j]+=w; A_np[j,i]+=w
        A_np+=np.eye(N_LOCS2); D_=A_np.sum(1,keepdims=True)**0.5
        A2=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))

        class GC(nn.Module):
            def __init__(self,d,dp=0.15):
                super().__init__()
                self.W=nn.Linear(d,d,bias=False); self.n=nn.LayerNorm(d); self.d=nn.Dropout(dp); self.a=nn.GELU()
            def forward(self,H,A):
                if A.dim()==3: A=A[0]
                return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),self.W(self.d(H)))))
        class HH(nn.Module):
            def __init__(self,d,nt,dp=0.15):
                super().__init__()
                self.mu=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
                self.lsv=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dp),nn.Linear(d//2,nt))
            def forward(self,h): return self.mu(h),self.lsv(h)
        class STGCN_W(nn.Module):
            def __init__(self,nf,h=64,nl=2,gl=2,nt=1):
                super().__init__()
                self.p=nn.Linear(nf,h)
                self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,dropout=0.15 if nl>1 else 0.)
                self.r=nn.Linear(h*2,h); self.gc=nn.ModuleList([GC(h) for _ in range(gl)]); self.hd=HH(h*2,nt)
            def forward(self,x,A):
                B,L,N,F=x.shape; h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F)); h,_=self.g(h)
                h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
                for g in self.gc: hg=g(hg,A)
                mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv

        nf2=len(V6F2); model=STGCN_W(nf=nf2,h=64,nl=2,gl=2,nt=NT2)
        model.load_state_dict({k:v.cpu() for k,v in theta_t_cpu.items()})
        opt=AdamW(model.parameters(),lr=lr,weight_decay=5e-4)

        sub2=raw2[raw2["split"]=="train"].copy()
        all_ts=sorted(sub2["time_utc"].unique()); T2=len(all_ts)
        ts2i={t:i for i,t in enumerate(all_ts)}
        sub2["_ti"]=sub2["time_utc"].map(ts2i)
        sub2["_ni"]=[loc2idx.get((float(la),float(lo))) for la,lo in zip(sub2["Latitude"],sub2["Longitude"])]
        sub2=sub2.dropna(subset=["_ti","_ni"]); sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
        sub2=sub2[sub2["_ti"]<T2]; sub2=sub2[sub2["_ni"].isin(loc_subset)]
        if len(sub2)==0: return None
        Xf=np.zeros((T2,N_LOCS2,nf2),dtype=np.float32); yf=np.zeros((T2,N_LOCS2,NT2),dtype=np.float32)
        mf=np.zeros((T2,N_LOCS2),dtype=np.float32)
        Xf[sub2["_ti"].values,sub2["_ni"].values]=feat_sc2.transform(sub2[V6F2].fillna(0).values).astype(np.float32)
        yf[sub2["_ti"].values,sub2["_ni"].values]=tgt_sc2.transform(sub2[use_cols2].fillna(0).values).astype(np.float32)
        mf[:,loc_subset]=1.0; LB=24; tidxs=list(range(LB,T2,6))[:300]
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
        w0=torch.cat([p.data.flatten() for p in model.parameters()]).numpy(); weight_traj.append(w0[:200].copy())
        model.train()
        for X,y,mask in loader:
            if step>=k_steps: break
            opt.zero_grad(); mu,lsv=model(X,A2)
            sv=torch.exp(lsv).clamp(min=1e-6)
            loss=(0.5*(lsv+(y-mu)**2/sv)*mask.unsqueeze(-1)).sum()/(mask.sum()+1e-8)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            gv=torch.cat([p.grad.data.flatten() for p in model.parameters() if p.grad is not None])
            gn=gv.norm()+1e-8; grad_dirs.append((gv/gn).numpy())
            uncertainties.append(float(torch.exp(0.5*lsv).mean().item()))
            opt.step(); losses.append(float(loss.item())); step+=1
            if step%10==0:
                wt=torch.cat([p.data.flatten() for p in model.parameters()]).numpy(); weight_traj.append(wt[:200].copy())
        if not losses: return None
        theta_final={k:v.clone() for k,v in model.state_dict().items()}
        delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}
        delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()
        stability=float(np.mean([np.dot(grad_dirs[i],grad_dirs[i-1])/(np.linalg.norm(grad_dirs[i])*np.linalg.norm(grad_dirs[i-1])+1e-8) for i in range(1,len(grad_dirs))])) if len(grad_dirs)>1 else 1.0
        progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.
        wf=torch.cat([v.flatten() for v in theta_final.values()]).numpy(); weight_traj.append(wf[:200].copy())
        return dict(worker_id=worker_id,delta=delta,delta_flat=delta_flat,losses=losses,
                     grad_dirs=grad_dirs,weight_traj=weight_traj,stability=stability,
                     progress=progress,movement=float(np.linalg.norm(delta_flat)),
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
    betas=np.array(betas)
    U=float(np.mean([b["uncertainty"] for b in behaviors]))
    gamma=0.85*(1-min(U*0.1,0.5))
    return alphas,betas,gamma,U,dict(consensus=consensus,stability=stabilities,progress=progresses)

def aggregate(theta_t,behaviors,alphas,betas,gamma):
    N=len(behaviors); theta_new={}
    mean_delta={k:sum(behaviors[i]["delta"][k].float() for i in range(N))/N for k in theta_t}
    for k in theta_t:
        shared=sum(alphas[i]*behaviors[i]["delta"][k].float() for i in range(N))
        residual_sum=sum(betas[i]*(behaviors[i]["delta"][k].float()-mean_delta[k]) for i in range(N))
        theta_new[k]=(theta_t[k].float()+gamma*(shared+0.1*residual_sum))
    return theta_new

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL SUBSETS
# ══════════════════════════════════════════════════════════════════════════════
def get_subsets(n_workers):
    if n_workers==1: return [SEEN]
    if n_workers==2:
        return [SITE_LOCS["Bedrock"]+SITE_LOCS["Transition"],
                SITE_LOCS["Upland"]+SITE_LOCS["Wetland"]]
    if n_workers==4: return [SITE_LOCS[s] for s in SITES]
    if n_workers==8:
        subs=[]
        for s in SITES:
            locs=SITE_LOCS[s]; mid=max(1,len(locs)//2)
            subs.append(locs[:mid]); subs.append(locs[mid:])
        return subs
    return [SEEN]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
T_ROUNDS=18; K_STEPS=50; LR=1e-3; M_DUP=2
N_WORKERS_LIST=[1,2,4,8]

all_results=[]; cent_rmse=float("nan"); cent_r2=float("nan"); cent_time=0.

# Centralized baseline first
print("\n  Centralized baseline...")
torch.manual_seed(SEED)
cent=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
cl=build_loader(split="train",bs=4,max_s=600,M_dup=M_DUP)
co=AdamW(cent.parameters(),lr=LR,weight_decay=5e-4)
vl=build_loader(split="test",bs=4,max_s=400,M_dup=M_DUP)
print(f"  Cent train loader: {cl is not None} | Val loader: {vl is not None}")
cent_weight_traj=[]
t_cent=time.time()
for ep in range(T_ROUNDS):
    cent.train()
    for X,y,mask,av in (cl or []):
        X=X.to(DEVICE); y=y.to(DEVICE); mask=mask.to(DEVICE)
        co.zero_grad(); mu,lsv=cent(X,A_norm)
        loss=nll_loss(mu,lsv,y,mask); loss.backward()
        nn.utils.clip_grad_norm_(cent.parameters(),1.0); co.step()
    wc=torch.cat([p.data.cpu().flatten() for p in cent.parameters()]).numpy()[:200]
    cent_weight_traj.append({"round":ep+1,"vec":wc})
cent_time=time.time()-t_cent
cent_rmse,cent_r2,cent_unc=evaluate(cent,vl)
print(f"  Centralized: RMSE={cent_rmse:.4f} R²={cent_r2:.4f} time={cent_time:.1f}s")

for N_WORKERS in N_WORKERS_LIST:
    print(f"\n{'='*55}")
    print(f"  N_WORKERS = {N_WORKERS}")
    print(f"{'='*55}")
    SUBSETS=get_subsets(N_WORKERS)
    print(f"  Subsets: {[len(s) for s in SUBSETS]} locations")

    torch.manual_seed(SEED)
    global_model=arch_cls(nf=N_FEATS,h=64,nl=2,gl=2,nt=NT).to(DEVICE)
    theta_t={k:v.cpu().clone() for k,v in global_model.state_dict().items()}
    vl2=build_loader(split="test",bs=4,max_s=400)

    round_history=[]; weight_snapshots=[]; worker_weight_trajs={wi:[] for wi in range(N_WORKERS)}
    w0=torch.cat([v.flatten() for v in theta_t.values()]).numpy()[:200]
    weight_snapshots.append({"round":0,"who":"init","vec":w0.copy()})
    best_rmse=float("inf"); best_theta=None; prev_rmse=float("inf")
    t_start=time.time()

    for rnd in range(1,T_ROUNDS+1):
        t_round=time.time()
        theta_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}

        if HAS_RAY and N_WORKERS>1:
            futures=[ray_worker.remote(wi,theta_cpu,SUBSETS[wi],K_STEPS,LR,SEED)
                      for wi in range(N_WORKERS)]
            behaviors_raw=ray.get(futures)
            behaviors=[b for b in behaviors_raw if b is not None]
        else:
            behaviors=[]
            for wi in range(N_WORKERS):
                b=local_worker(wi,{k:v.clone() for k,v in theta_cpu.items()},
                                SUBSETS[wi],K_STEPS,LR)
                if b is not None: behaviors.append(b)

        if not behaviors:
            print(f"  Rnd {rnd:02d}: No workers returned")
            round_history.append(dict(round=rnd,rmse=float("nan"),r2=float("nan"),
                                       gamma=0.,U=float("nan"),alphas=[],betas=[],
                                       worker_losses=[],elapsed=time.time()-t_round))
            continue

        for b in behaviors:
            wi=b["worker_id"]
            for wt in b["weight_traj"]: worker_weight_trajs[wi].append({"round":rnd,"vec":wt})
            wend=(theta_cpu[list(theta_cpu.keys())[0]].flatten()[:200].numpy())
            weight_snapshots.append({"round":rnd,"who":f"worker_{wi+1}","vec":wend})

        alphas,betas,gamma_u,U,scores=compute_alphas_betas(behaviors)
        theta_cand=aggregate(theta_t,behaviors,alphas,betas,gamma_u)
        global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})
        rmse,r2,unc=evaluate(global_model,vl2)

        wg=torch.cat([v.flatten() for v in theta_cand.values()]).numpy()[:200]
        weight_snapshots.append({"round":rnd,"who":"global","vec":wg})

        rmse_ok=not(np.isnan(rmse) or np.isinf(rmse))
        if not rmse_ok or rmse<=prev_rmse*1.02:
            theta_t=theta_cand
            if rmse_ok:
                prev_rmse=rmse
                if rmse<best_rmse: best_rmse=rmse; best_theta=copy.deepcopy(theta_cand)
            status="ACC"
        else:
            gamma_u*=0.8; status="REJ"

        wl=[b["final_loss"] for b in behaviors]
        print(f"  Rnd {rnd:02d} | RMSE={rmse:.4f} R²={r2:.4f} U={U:.3f} "
               f"γ={gamma_u:.3f} α={[f'{a:.2f}' for a in alphas]} "
               f"β={[f'{b:.2f}' for b in betas]} | {status}")

        round_history.append(dict(round=rnd,rmse=rmse,r2=r2,gamma=gamma_u,U=U,
                                   alphas=alphas.tolist(),betas=betas.tolist(),
                                   consensus=scores["consensus"].tolist(),
                                   stability=scores["stability"].tolist(),
                                   progress=scores["progress"].tolist(),
                                   worker_losses=wl,elapsed=time.time()-t_round))

    total_time=time.time()-t_start; ideal_time=total_time/N_WORKERS
    pd.DataFrame(round_history).to_csv(RESULTS/f"rounds_N{N_WORKERS}.csv",index=False)
    valid_r2=[r["r2"] for r in round_history if not np.isnan(r["r2"])]
    all_results.append(dict(n_workers=N_WORKERS,
                              best_rmse=round(best_rmse,4) if not np.isinf(best_rmse) else float("nan"),
                              best_r2=round(max(valid_r2),4) if valid_r2 else float("nan"),
                              ideal_time_s=round(ideal_time,2),
                              total_time_s=round(total_time,2)))

    # Save GIF data for N=4
    if N_WORKERS==4:
        round_hist_4=round_history; weight_snap_4=weight_snapshots
        worker_traj_4=worker_weight_trajs; cent_traj_4=cent_weight_traj

# Summary
summary=pd.DataFrame(all_results+[dict(n_workers=0,
    best_rmse=round(cent_rmse,4),best_r2=round(cent_r2,4),
    ideal_time_s=round(cent_time,2),total_time_s=round(cent_time,2))])
summary.to_csv(RESULTS/"behavior_guided_summary.csv",index=False)
print(f"\n{summary.to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# GIF GENERATION
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from sklearn.decomposition import PCA

WORKER_COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
rh=round_hist_4
rounds_=[r["round"] for r in rh]; rmses_=[r["rmse"] for r in rh]
Us_=[r["U"] for r in rh]; gammas_=[r["gamma"] for r in rh]
valid_r=[r for r in rmses_ if not np.isnan(r) and not np.isinf(r)]

# GIF_01: RMSE + uncertainty
print("\n  GIF_01...")
fig,axes=plt.subplots(1,2,figsize=(14,6))
ax1,ax2=axes
l1,=ax1.plot([],[],color="#1f77b4",lw=2.5,marker="^",ms=8,label="Behavior-guided RMSE")
l2,=ax1.plot([],[],color="#ff7f0e",lw=2.5,marker="o",ms=8,label="Centralized RMSE")
ylo=min(valid_r+[cent_rmse])*0.96 if valid_r else 0
yhi=max(valid_r+[cent_rmse])*1.04 if valid_r else 5
ax1.set_xlim(0,T_ROUNDS+1); ax1.set_ylim(ylo,yhi)
ax1.set_xlabel("Global Round",fontsize=12); ax1.set_ylabel("Test RMSE (C)",fontsize=12)
ax1.set_title("Performance During Training",fontsize=12); ax1.legend(fontsize=10)
box=ax1.text(0.05,0.2,"",transform=ax1.transAxes,fontsize=8,
              bbox=dict(boxstyle="round",facecolor="lightblue",alpha=0.5))
l3,=ax2.plot([],[],color="#2ca02c",lw=2,label="Population uncertainty U")
l4,=ax2.plot([],[],color="#d62728",lw=2,ls="--",label="Step size γ")
ax2.set_xlim(0,T_ROUNDS+1); ax2.set_ylim(0,max(Us_+[1])*1.1 if Us_ else 2)
ax2.set_xlabel("Global Round",fontsize=12); ax2.set_title("Uncertainty Controls Step Size",fontsize=12)
ax2.legend(fontsize=10)
def u1(frame):
    i=min(frame,len(rounds_)-1); r=rh[i]
    l1.set_data(rounds_[:i+1],rmses_[:i+1]); l2.set_data(rounds_[:i+1],[cent_rmse]*(i+1))
    l3.set_data(rounds_[:i+1],Us_[:i+1]); l4.set_data(rounds_[:i+1],gammas_[:i+1])
    box.set_text(f"Round {r['round']}/{T_ROUNDS}\nRMSE: {r['rmse']:.3f}\nCent: {cent_rmse:.3f}\nU={r['U']:.3f}")
    return l1,l2,l3,l4,box
ani1=animation.FuncAnimation(fig,u1,frames=len(rounds_),interval=400,blit=True)
ani1.save(FIGS/"gif_01_time_performance.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_01")

# GIF_02: Weight space
print("\n  GIF_02...")
all_vecs=[s["vec"][:200] for s in weight_snap_4]
maxl=min(200,min(len(v) for v in all_vecs))
all_np=np.array([v[:maxl] for v in all_vecs])
pca=PCA(n_components=2,random_state=SEED); all_pca=pca.fit_transform(all_np)
by_round={}
for i,s in enumerate(weight_snap_4):
    rn=s["round"]; who=s["who"]
    if rn not in by_round: by_round[rn]={"workers":{},"global":None}
    if who.startswith("worker_"):
        wi=int(who.split("_")[1])-1
        if wi not in by_round[rn]["workers"]: by_round[rn]["workers"][wi]=[]
        by_round[rn]["workers"][wi].append(all_pca[i])
    elif who=="global": by_round[rn]["global"]=all_pca[i]
wc_vec=torch.cat([p.data.cpu().flatten() for p in cent.parameters()]).numpy()[:maxl]
wc_pca=pca.transform(wc_vec.reshape(1,-1))
fig,ax=plt.subplots(figsize=(10,8))
ax.set_xlabel("Weight-space PC1",fontsize=11); ax.set_ylabel("Weight-space PC2",fontsize=11)
ax.set_title("Worker Weight Trajectories, Predicted Generalized Path, and True Full-Data Path",fontsize=10)
init_idx=[i for i,s in enumerate(weight_snap_4) if s["who"]=="init"]
if init_idx:
    ax.scatter(all_pca[init_idx[0],0],all_pca[init_idx[0],1],c="black",s=200,marker="X",
                zorder=10,label="Initialization")
ax.scatter(wc_pca[0,0],wc_pca[0,1],c="saddlebrown",s=200,marker="*",zorder=9,
            label="True full-data final")
wlines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,
                 label=f"Worker {wi+1}: {SITES[wi] if wi<len(SITES) else ''}")[0]
         for wi in range(4)]
gline,=ax.plot([],[],color="darkviolet",lw=2.5,ls="--",label="Predicted generalized")
gdot=ax.scatter([],[],c="blue",s=150,marker="s",zorder=9,label="Predicted final")
xpad=(all_pca[:,0].max()-all_pca[:,0].min())*0.05+0.05
ypad=(all_pca[:,1].max()-all_pca[:,1].min())*0.05+0.05
ax.set_xlim(all_pca[:,0].min()-xpad,all_pca[:,0].max()+xpad)
ax.set_ylim(all_pca[:,1].min()-ypad,all_pca[:,1].max()+ypad)
ax.legend(fontsize=8,loc="upper right")
wpts={wi:[] for wi in range(4)}; gpts=[]
sorted_rns=sorted(by_round.keys())
def u2(frame):
    rn=sorted_rns[min(frame,len(sorted_rns)-1)]; rd=by_round[rn]
    for wi in range(4):
        if wi in rd["workers"]:
            for pt in rd["workers"][wi]: wpts[wi].append(pt)
        if len(wpts[wi])>1:
            pts=np.array(wpts[wi]); wlines[wi].set_data(pts[:,0],pts[:,1])
    if rd["global"] is not None: gpts.append(rd["global"])
    if len(gpts)>1:
        gp=np.array(gpts); gline.set_data(gp[:,0],gp[:,1]); gdot.set_offsets(gp[-1].reshape(1,2))
    return wlines+[gline,gdot]
ani2=animation.FuncAnimation(fig,u2,frames=len(sorted_rns),interval=500,blit=True)
ani2.save(FIGS/"gif_02_weight_space.gif",writer=PillowWriter(fps=2),dpi=100)
plt.close(); print("    OK gif_02")

# GIF_03: Loss evolution
print("\n  GIF_03...")
all_wl=[[rh[ri]["worker_losses"][wi] if wi<len(rh[ri]["worker_losses"]) else float("nan")
          for ri in range(len(rh))] for wi in range(4)]
gloss=[r["rmse"]**2 if not np.isnan(r["rmse"]) else float("nan") for r in rh]
closs=[cent_rmse**2]*len(rh)
all_v=[v for wl in all_wl for v in wl if not np.isnan(v)]
ymax=max(all_v+[cent_rmse**2])*1.1 if all_v else 3
fig,ax=plt.subplots(figsize=(10,6))
ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(0,ymax)
ax.set_xlabel("Global Round",fontsize=11); ax.set_ylabel("MSE Loss",fontsize=11)
ax.set_title("Loss Evolution: Local Workers vs Global Prediction vs Centralized",fontsize=12)
wlines2=[ax.plot([],[],color=WORKER_COLORS[wi],lw=1.5,marker="o",ms=5,
                  label=f"Worker {wi+1} ({SITES[wi] if wi<len(SITES) else ''})")[0] for wi in range(4)]
gl,=ax.plot([],[],color="darkviolet",lw=2.5,marker="*",ms=10,label="Predicted global")
cl2,=ax.plot([],[],color="saddlebrown",lw=2,marker="s",ms=8,ls="--",label="Centralized")
ax.legend(fontsize=8,loc="upper right")
def u3(frame):
    i=min(frame,len(rh)-1); xs=rounds_[:i+1]
    for wi in range(4): wlines2[wi].set_data(xs,all_wl[wi][:i+1])
    gl.set_data(xs,gloss[:i+1]); cl2.set_data(xs,closs[:i+1])
    return wlines2+[gl,cl2]
ani3=animation.FuncAnimation(fig,u3,frames=len(rh),interval=400,blit=True)
ani3.save(FIGS/"gif_03_loss_workers.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_03")

# GIF_04: Beta corrections
print("\n  GIF_04...")
betas_pr=[[rh[ri]["betas"][wi] if wi<len(rh[ri].get("betas",[])) else float("nan")
            for ri in range(len(rh))] for wi in range(4)]
fig,ax=plt.subplots(figsize=(12,6))
ax.axhline(0,color="#1f77b4",lw=1.5,alpha=0.5)
ax.set_xlim(0,T_ROUNDS+1); ax.set_ylim(-0.8,0.8)
ax.set_xlabel("Global round",fontsize=12); ax.set_ylabel("Signed residual coefficient β",fontsize=12)
ax.set_title("Dynamic Unique-Direction Corrections",fontsize=13)
blines=[ax.plot([],[],color=WORKER_COLORS[wi],lw=2,
                 label=SITES[wi] if wi<len(SITES) else f"Worker {wi+1}")[0] for wi in range(4)]
ax.legend(fontsize=10)
def u4(frame):
    i=min(frame,len(rh)-1); xs=rounds_[:i+1]
    for wi in range(4): blines[wi].set_data(xs,betas_pr[wi][:i+1])
    return blines
ani4=animation.FuncAnimation(fig,u4,frames=len(rh),interval=400,blit=True)
ani4.save(FIGS/"gif_04_beta_corrections.gif",writer=PillowWriter(fps=2.5),dpi=100)
plt.close(); print("    OK gif_04")

# Static figures
fig,axes=plt.subplots(1,2,figsize=(16,6))
ax1,ax2=axes
methods_=[f"Behavior-guided\nN={r['n_workers']}" for r in all_results]+["Centralized"]
rmses_m=[r['best_rmse'] for r in all_results]+[cent_rmse]
rmses_m=[r if not np.isnan(r) else 0 for r in rmses_m]
bars=ax1.bar(methods_,rmses_m,color="#1f77b4",width=0.5)
for bar,v in zip(bars,rmses_m):
    ax1.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.3f}",
              ha="center",va="bottom",fontsize=10,fontweight="bold")
ax1.set_ylabel("Test RMSE (C)",fontsize=12); ax1.set_title("Final Prediction Performance",fontsize=12)
ax1.set_ylim(0,max(rmses_m+[0.1])*1.15)
tm=[f"N={r['n_workers']}\n(ideal)" for r in all_results]
tv=[r['ideal_time_s'] for r in all_results]
tv2=[r['total_time_s'] for r in all_results]
x_=np.arange(len(tm)); w_=0.35
ax2.bar(x_-w_/2,tv,w_,color="#1f77b4",alpha=0.9,label="Ideal parallel")
ax2.bar(x_+w_/2,tv2,w_,color="#aec7e8",alpha=0.9,label="Sequential here")
ax2.axhline(cent_time,color="red",ls="--",lw=2,label=f"Centralized ({cent_time:.1f}s)")
ax2.set_xticks(x_); ax2.set_xticklabels(tm)
ax2.set_ylabel("Processing Time (s)",fontsize=12); ax2.set_title("Training Time vs N Workers",fontsize=12)
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGS/"final_performance.png",dpi=300,bbox_inches="tight"); plt.close()

# Uncertainty + step size
fig,ax=plt.subplots(figsize=(12,5))
ax.plot(rounds_,Us_,color="#1f77b4",lw=2,label="Population uncertainty U")
ax.plot(rounds_,gammas_,color="#ff7f0e",lw=2,label="Jump scale γ")
ax.set_xlabel("Global round",fontsize=12); ax.set_ylabel("Value",fontsize=12)
ax.set_title("Uncertainty Controls Step Size",fontsize=13); ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(FIGS/"uncertainty_step_size.png",dpi=300,bbox_inches="tight"); plt.close()

# Signed coefficients
alphas_pr=[[rh[ri]["alphas"][wi] if wi<len(rh[ri].get("alphas",[])) else float("nan")
             for ri in range(len(rh))] for wi in range(4)]
fig,ax=plt.subplots(figsize=(12,6))
for wi in range(4):
    ax.plot(rounds_,alphas_pr[wi],color=WORKER_COLORS[wi],lw=2,
             label=SITES[wi] if wi<len(SITES) else f"Worker {wi+1}")
ax.axhline(0,color="#1f77b4",lw=1,alpha=0.5)
ax.set_xlabel("Global round",fontsize=12); ax.set_ylabel("Signed worker coefficient",fontsize=12)
ax.set_title("Uncertainty-Aware Dynamic Signed Coefficients",fontsize=13); ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(FIGS/"uncertainty_signed_coefficients.png",dpi=300,bbox_inches="tight"); plt.close()

print(f"\n{'='*65}")
print(f"  EXPERIMENT COMPLETE")
figs_out=sorted(FIGS.glob("*.gif"))+sorted(FIGS.glob("*.png"))
for f in figs_out: print(f"  {f.name} ({f.stat().st_size//1024} KB)")
print(f"{'='*65}")
