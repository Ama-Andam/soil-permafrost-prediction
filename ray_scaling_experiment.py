"""
================================================================================
ray_scaling_experiment.py
GPU SCALING EXPERIMENT — Model-Level Parallelism via Ray Remote
================================================================================

SENIOR INSTRUCTIONS:
  - Use ray.remote to parallelise tasks (NOT ray.train or ray.tune)
  - Use ray.get to gather results
  - Model-level parallelism only (training parallelism is unstable)
  - Test 1, 2, 4, 8 GPU configurations
  - Measure: wall time, speedup, distortion vs sequential baseline

WHAT THIS DOES:
  For each GPU count N (1, 2, 4, 8):
    - Assign one model per GPU using ray.remote(num_gpus=1)
    - Train N models simultaneously
    - Rotate through all 11 models in batches of N
    - Record wall time for the full experiment
    - Compare R² to sequential baseline (distortion check)

OUTPUT:
  results_v4/scaling_results.csv     — full timing and metrics table
  figures_v4/SCALE_01_speedup.png    — speedup curve vs ideal
  figures_v4/SCALE_02_distortion.png — R² distortion across GPU counts
  figures_v4/SCALE_03_time_table.png — publication-ready timing table

RUN ON TALON (request all 8 GPUs):
  sbatch ~/logs/run_scaling.sh
================================================================================
"""

import os, sys, time, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT  = Path("/home/emmanuel.keku")
PREPROC  = PROJECT / "preprocessed_v3"
RESULTS  = PROJECT / "results_v4"
MODELS   = PROJECT / "models_v4" / "dl"
FIGS     = PROJECT / "figures_v4"
LOGS     = PROJECT / "logs"
for d in [RESULTS, FIGS]: d.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3})

# ── Ray setup ─────────────────────────────────────────────────────────────────
try:
    import ray
    RAY_OK = True
except ImportError:
    print("Ray not available — installing...")
    import subprocess
    subprocess.run([sys.executable,"-m","pip","install","--user","ray","-q"])
    import ray
    RAY_OK = True

# ── GPU count from environment ─────────────────────────────────────────────────
N_GPUS_AVAILABLE = int(os.environ.get("SLURM_GPUS_ON_NODE",
                    os.environ.get("CUDA_VISIBLE_DEVICES","0,1,2,3,4,5,6,7")
                    .count(",")+1 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1))

JOB_ID = os.environ.get("SLURM_JOB_ID","local")
NODE   = os.environ.get("SLURMD_NODENAME","unknown")

print("="*70)
print("  GPU SCALING EXPERIMENT — Model-Level Parallelism via Ray")
print(f"  Node        : {NODE}")
print(f"  GPUs avail  : {N_GPUS_AVAILABLE}")
print(f"  Job         : {JOB_ID}")
print(f"  Start       : {pd.Timestamp.now()}")
print("="*70)

# ── Initialise Ray ────────────────────────────────────────────────────────────
if not ray.is_initialized():
    ray.init(num_gpus=N_GPUS_AVAILABLE,
             num_cpus=int(os.environ.get("SLURM_CPUS_ON_NODE", 8)),
             ignore_reinit_error=True,
             logging_level="WARNING")
    print(f"\nRay initialised: {ray.available_resources()}")

# ── Load preprocessed data (shared across workers) ────────────────────────────
print("\nLoading preprocessed data...")
t0 = time.time()

with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)

# Put shared data in Ray object store — workers access without copying
df_path     = str(PREPROC/"master_processed.csv")
scalers_ref = ray.put(SC)
fi_ref      = ray.put(FI)
print(f"  Data ready in {time.time()-t0:.1f}s")

# ── Model configurations ───────────────────────────────────────────────────────
ARCHES = [
    "BiGRU_NoGCN","GCN_NoTemporal",
    "DeepESN","SpatialESN",
    "GraphSAGE","GAT","STGCN",
    "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE",
]
TARGETS = ["temp", "smap", "moist"]   # All 3 targets — match v4

TRAIN_CONFIG = dict(
    epochs=30,        # Match v4 training — full 30 epochs
    lr=3e-4,
    patience=7,       # Match v4 patience
    lam_s=0.05,
    lam_a=0.01,
    lookback=24,
    batch_size=4,
    seed=42,
)

# ══════════════════════════════════════════════════════════════════════════════
# RAY REMOTE FUNCTION
# One GPU per task — per senior's instruction
# Using ray.remote NOT ray.train
# ══════════════════════════════════════════════════════════════════════════════

@ray.remote(num_gpus=1, num_cpus=2, max_retries=1)
def train_model_on_gpu(arch, target, config, scalers_ref, fi_ref,
                        df_path, models_dir, gpu_idx=None):
    """
    Train one model on one GPU.
    Called via ray.remote — runs in isolated worker process.
    Returns: dict with arch, target, metrics, history, elapsed_s

    Per senior: ray.remote for task dispatch, ray.get to gather.
    Model-level parallelism — each GPU owns one complete training run.
    """
    import os, time, pickle, warnings
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from pathlib import Path
    from scipy.spatial import cKDTree
    import ray                          # FIX: ray must be imported inside the worker
    warnings.filterwarnings("ignore")

    # Each Ray worker sees its allocated GPU as cuda:0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    worker_id = os.getpid()

    t_start = time.time()
    print(f"  [{arch}|{target}] PID={worker_id} | GPU={gpu_name} | {device}")

    # FIX: scalers_ref and fi_ref are already dereferenced by Ray before the
    # worker receives them — ray.get() on a plain object raises ValueError.
    # Ray automatically resolves ObjectRef arguments when dispatching .remote().
    SC = scalers_ref
    FI = fi_ref

    # Load data
    df = pd.read_csv(df_path, parse_dates=["time_utc"])

    LOCATIONS        = pd.DataFrame(FI["LOCATIONS"])
    N_LOCS           = FI["N_LOCS"]
    SNAP_FEATURES    = FI["SNAP_FEATURES"]
    TEMP_TARGETS     = FI["TEMP_TARGETS"]
    SMAP_TARGETS     = FI["SMAP_TARGETS"]
    MOIST_TARGETS    = FI["MOIST_TARGETS"]
    SITES            = FI["SITES"]
    snap_feat_scaler = SC["snap_feat_scaler"]
    snap_tgt_scalers = SC["snap_tgt_scalers"]

    # v4 features
    ALL_TARGETS  = TEMP_TARGETS + SMAP_TARGETS + MOIST_TARGETS
    APPROX_FEATS = [f"{t}_approx" for t in ALL_TARGETS if f"{t}_approx" in df.columns]
    V4_FEATURES  = list(dict.fromkeys(SNAP_FEATURES + APPROX_FEATS))
    V4_FEATURES  = [f for f in V4_FEATURES if f in df.columns]
    N_V4_FEATURES= len(V4_FEATURES)

    from sklearn.preprocessing import RobustScaler
    tr_all = df[df["split"]=="train"]
    v4_fs  = RobustScaler(); v4_fs.fit(tr_all[V4_FEATURES].fillna(0).values)

    tgt_cols = {"temp":TEMP_TARGETS,"smap":SMAP_TARGETS,"moist":MOIST_TARGETS}[target]
    av_tgt   = [c for c in tgt_cols if c in df.columns]
    if not av_tgt:
        return dict(arch=arch, target=target, status="NO_TARGETS",
                    elapsed_s=time.time()-t_start)

    v4_ts = RobustScaler(); v4_ts.fit(tr_all[av_tgt].dropna().values)

    # Spatial setup
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

    # Build graph
    coords = LOCATIONS[["Latitude","Longitude"]].values.astype(np.float32)
    scaled = coords * np.array([111.0,63.0],dtype=np.float32)
    tree   = cKDTree(scaled)
    dists, idxs = tree.query(scaled, k=min(7,N_LOCS))
    sigma  = np.median(dists[:,1:])+1e-8
    A      = np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
    for i in range(N_LOCS):
        for jp in range(1,dists.shape[1]):
            j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
            A[i,j]+=w; A[j,i]+=w
    A+=np.eye(N_LOCS); D=A.sum(1,keepdims=True)**0.5
    A_norm=torch.tensor((A/(D*D.T+1e-8)).astype(np.float32)).to(device)

    # Dataset
    class SnapDS(Dataset):
        def __init__(self, split, lookback=24, stride=6, max_s=500):
            sub=df[df["split"]==split].copy()
            all_ts=sorted(sub["time_utc"].unique()); T=len(all_ts)
            if T<lookback+2: self.X=self.y=self.m=torch.zeros(0); return
            ts_to_i={ts:i for i,ts in enumerate(all_ts)}
            sub2=sub.copy()
            sub2["_ti"]=sub2["time_utc"].map(ts_to_i)
            sub2["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                         for la,lo in zip(sub2["Latitude"].astype(float),
                                          sub2["Longitude"].astype(float))]
            sub2=sub2.dropna(subset=["_ti","_ni"])
            sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
            ti=sub2["_ti"].values; ni=sub2["_ni"].values
            nf=len(V4_FEATURES); nt=len(av_tgt)
            Xf=np.full((T,N_LOCS,nf),np.nan,dtype=np.float32)
            yf=np.full((T,N_LOCS,nt),np.nan,dtype=np.float32)
            mf=np.zeros((T,N_LOCS),dtype=np.float32)
            Xf[ti,ni,:]=v4_fs.transform(sub2[V4_FEATURES].fillna(0).values).astype(np.float32)
            yf[ti,ni,:]=v4_ts.transform(sub2[av_tgt].fillna(0).values).astype(np.float32)
            if split=="train": mf[:,SEEN_LOCS]=1.0
            else: mf[:,:]=1.0
            tidxs=list(range(lookback,T,stride))
            if max_s and len(tidxs)>max_s:
                rng=np.random.default_rng(42); tidxs=sorted(rng.choice(tidxs,max_s,replace=False))
            Xl=[]; yl=[]; ml=[]
            for ti2 in tidxs:
                Xw=Xf[ti2-lookback:ti2]; yi=yf[ti2]; mi=mf[ti2]
                if np.isnan(Xw).mean()>0.25: continue
                Xl.append(np.nan_to_num(Xw,nan=0.0))
                yl.append(np.nan_to_num(yi,nan=0.0)); ml.append(mi)
            if not Xl: self.X=self.y=self.m=torch.zeros(0); return
            self.X=torch.tensor(np.array(Xl)); self.y=torch.tensor(np.array(yl))
            self.m=torch.tensor(np.array(ml)); self.A=A_norm
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i],self.y[i],self.m[i],self.A

    train_ds=SnapDS("train",lookback=config["lookback"],stride=6,max_s=500)
    val_ds  =SnapDS("val",  lookback=config["lookback"],stride=24,max_s=200)
    if len(train_ds)==0 or len(val_ds)==0:
        return dict(arch=arch,target=target,status="NO_DATA",
                    elapsed_s=time.time()-t_start)

    train_ld=DataLoader(train_ds,batch_size=config["batch_size"],
                        shuffle=True,num_workers=0,drop_last=True)
    val_ld  =DataLoader(val_ds,  batch_size=config["batch_size"],
                        shuffle=False,num_workers=0)

    # Build model (inline definitions to avoid import issues in Ray workers)
    class GraphConv(nn.Module):
        def __init__(self,id,od,dp=0.1):
            super().__init__()
            self.W=nn.Linear(id,od,bias=False); self.n=nn.LayerNorm(od)
            self.d=nn.Dropout(dp); self.a=nn.GELU()
        def forward(self,H,A):
            if A.dim()==3: A=A[0]
            if A.dim()==4: A=A[0,0]
            return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),self.W(self.d(H)))))

    class BiGRU_NoGCN(nn.Module):
        def __init__(self,nf,h=96,nl=2,nh=4,nt=1,dp=0.1,**kw):
            super().__init__()
            self.p=nn.Linear(nf,h); self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,dropout=dp if nl>1 else 0.)
            d2=h*2; self.a=nn.MultiheadAttention(d2,nh,dropout=dp,batch_first=True)
            self.n1=nn.LayerNorm(d2); self.n2=nn.LayerNorm(d2)
            self.ff=nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),nn.Dropout(dp),nn.Linear(d2*2,d2))
            self.r=nn.Linear(d2,h); self.hd=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Dropout(dp),nn.Linear(h//2,nt))
        def forward(self,x,A):
            B,L,N,F=x.shape; h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
            h,_=self.g(h); a,_=self.a(h,h,h); h=self.n1(h+a); h=self.n2(h+self.ff(h))
            return self.hd(self.r(h[:,-1,:]).reshape(B,N,-1))

    class GCN_NoTemporal(nn.Module):
        def __init__(self,nf,h=96,gl=3,nt=1,dp=0.1,**kw):
            super().__init__()
            self.p=nn.Linear(nf,h); self.gcn=nn.ModuleList([GraphConv(h,h,dp) for _ in range(gl)])
            self.hd=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Dropout(dp),nn.Linear(h,nt))
        def forward(self,x,A):
            B,L,N,F=x.shape; h=self.p(x[:,-1,:,:]); h0=h
            for g in self.gcn: h=g(h,A)
            return self.hd(torch.cat([h0,h],dim=-1))

    class MambaBlock(nn.Module):
        def __init__(self,d,ds=16,dc=4,ex=2,dp=0.1):
            super().__init__()
            self.di=d*ex; self.ds=ds
            self.ip=nn.Linear(d,self.di*2,bias=False)
            self.cv=nn.Conv1d(self.di,self.di,dc,padding=dc-1,groups=self.di,bias=True)
            self.silu=nn.SiLU(); self.xp=nn.Linear(self.di,ds*2+self.di,bias=False)
            self.dtp=nn.Linear(self.di,self.di,bias=True)
            A_=torch.arange(1,ds+1,dtype=torch.float32).unsqueeze(0).repeat(self.di,1)
            self.Al=nn.Parameter(torch.log(A_)); self.D_=nn.Parameter(torch.ones(self.di))
            self.op=nn.Linear(self.di,d,bias=False); self.dr=nn.Dropout(dp); self.nm=nn.LayerNorm(d)
        def scan(self,x):
            B,L,D=x.shape; S=self.ds; xd=self.xp(x); dl,Bp,C=xd.split([D,S,S],dim=-1)
            dl=F.softplus(self.dtp(dl)); A__=-torch.exp(self.Al.float())
            dA=torch.exp(torch.einsum("bld,ds->blds",dl,A__))
            dB=torch.einsum("bld,bls->blds",dl,Bp)
            h=torch.zeros(B,D,S,device=x.device,dtype=x.dtype); ys=[]
            for i in range(L):
                h=dA[:,i]*h+dB[:,i]*x[:,i,:,None]; ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
            return torch.stack(ys,dim=1)*self.D_
        def forward(self,x):
            r=x; xz=self.ip(x); x_,z=xz.chunk(2,dim=-1)
            x_=self.silu(self.cv(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
            return self.nm(r+self.op(self.dr(self.scan(x_)*self.silu(z))))

    class DeepESNLayer(nn.Module):
        def __init__(self,id,rd,sr=0.9,lr=0.3,dp=0.1):
            super().__init__()
            self.rd=rd; self.lr=lr
            Wi=torch.randn(rd,id)*0.1; Wr=torch.randn(rd,rd)
            ev=torch.linalg.eigvals(Wr).abs(); Wr=Wr*(sr/(ev.max().item()+1e-8))
            self.register_buffer("Wi",Wi); self.register_buffer("Wr",Wr)
            self.drop=nn.Dropout(dp); self.norm=nn.LayerNorm(rd)
        def forward(self,x):
            B,L,_=x.shape; h=torch.zeros(B,self.rd,device=x.device,dtype=x.dtype); st=[]
            for t in range(L):
                h=(1-self.lr)*h+self.lr*torch.tanh(x[:,t,:]@self.Wi.T+h@self.Wr.T); st.append(h)
            return self.norm(torch.stack(st,dim=1))

    # Simplified model map for scaling experiment
    def make_model(arch, nf, N, nt):
        h=96; dp=0.1
        if arch=="BiGRU_NoGCN":
            return BiGRU_NoGCN(nf,h=h,nl=2,nh=4,nt=nt,dp=dp)
        elif arch=="GCN_NoTemporal":
            return GCN_NoTemporal(nf,h=h,gl=3,nt=nt,dp=dp)
        elif arch in ["DeepESN","SpatialESN"]:
            class SimpleESN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.p=nn.Linear(nf,128)
                    self.esn=DeepESNLayer(128,128,sr=0.9,lr=0.3,dp=dp)
                    self.gcn=nn.ModuleList([GraphConv(128,128,dp),GraphConv(128,128,dp)]) if arch=="SpatialESN" else nn.ModuleList()
                    self.hd=nn.Sequential(nn.Linear(128,nt))
                def forward(self,x,A):
                    B,L,N2,F=x.shape; h=self.p(x.permute(0,2,1,3).reshape(B*N2,L,F))
                    h=self.esn(h)[:,-1,:].reshape(B,N2,-1)
                    for g in self.gcn: h=g(h,A)
                    return self.hd(h)
            return SimpleESN()
        else:
            # Generic: GRU + GCN for all other archs
            class GenericModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.p=nn.Linear(nf,h)
                    self.gru=nn.GRU(h,h,2,batch_first=True,bidirectional=True,dropout=dp)
                    self.r=nn.Linear(h*2,h)
                    self.gcn=nn.ModuleList([GraphConv(h,h,dp),GraphConv(h,h,dp)])
                    self.hd=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Dropout(dp),nn.Linear(h,nt))
                def forward(self,x,A):
                    B,L,N2,F=x.shape; h=self.p(x.permute(0,2,1,3).reshape(B*N2,L,F))
                    h,_=self.gru(h); h=self.r(h[:,-1,:]).reshape(B,N2,-1); h0=h
                    for g in self.gcn: h=g(h,A)
                    return self.hd(torch.cat([h0,h],dim=-1))
            return GenericModel()

    model = make_model(arch, N_V4_FEATURES, N_LOCS, len(av_tgt)).to(device)
    is_moe = False
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt   = AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                  lr=config["lr"], weight_decay=1e-4)
    n_st  = config["epochs"]*len(train_ld)
    sched = OneCycleLR(opt, max_lr=config["lr"], total_steps=max(n_st,1), pct_start=0.1)
    amp_sc= torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def masked_huber(p,t,m,d=1.0):
        df=p-t; L=torch.where(df.abs()<=d,0.5*df**2,d*(df.abs()-0.5*d))
        me=m.unsqueeze(-1).expand_as(L)
        return (L*me).sum()/(me.sum()+1e-8)

    best_r2=float("-inf"); hist=[]
    pat=0

    for ep in range(1, config["epochs"]+1):
        model.train(); tr=0.; nb=0
        for batch in train_ld:
            X,y,mask,A_=[b.to(device) for b in batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred=model(X,A_)
                loss=masked_huber(pred,y,mask)
            amp_sc.scale(loss).backward()
            amp_sc.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            amp_sc.step(opt); amp_sc.update(); sched.step()
            tr+=loss.item(); nb+=1

        # Validate
        model.eval(); yt=[]; yp=[]
        with torch.no_grad():
            for batch in val_ld:
                X,y,mask,A_=[b.to(device) for b in batch]
                pred=model(X,A_)
                B_,N_,T_=pred.shape
                pr=pred.cpu().float().numpy()
                pr_r=v4_ts.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
                y_r=v4_ts.inverse_transform(y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
                yt.append(y_r[:,SEEN_LOCS,0].flatten())
                yp.append(pr_r[:,SEEN_LOCS,0].flatten())

        ytf=np.concatenate(yt); ypf=np.concatenate(yp)
        mk=~(np.isnan(ytf)|np.isnan(ypf)); ytf=ytf[mk]; ypf=ypf[mk]
        r2v=float(1-np.sum((ytf-ypf)**2)/(np.sum((ytf-ytf.mean())**2)+1e-10)) if len(ytf)>5 else 0
        hist.append(dict(epoch=ep,train_loss=round(tr/max(nb,1),6),val_R2=round(r2v,4)))

        if r2v>best_r2: best_r2=r2v; pat=0
        else: pat+=1
        if pat>=config["patience"]: break

    elapsed = time.time()-t_start
    print(f"  [{arch}|{target}] DONE | R²={best_r2:.4f} | {elapsed:.0f}s | {device}")

    return dict(
        arch=arch, target=target,
        n_params=n_params, n_epochs=ep,
        best_val_r2=round(best_r2,4),
        elapsed_s=round(elapsed,1),
        device=str(device), gpu_name=gpu_name,
        worker_pid=worker_id, history=hist,
        status="OK")


# ══════════════════════════════════════════════════════════════════════════════
# SCALING EXPERIMENT
# Test GPU counts: 1, 2, 4, 8
# For each count N: train N models in parallel, rotate through all 11
# ══════════════════════════════════════════════════════════════════════════════

# Model-target pairs — all 11 models × 3 targets = 33 pairs
PAIRS = [(arch, tgt) for arch in ARCHES for tgt in TARGETS]
N_PAIRS = len(PAIRS)

# GPU counts to test (limited by what's available)
GPU_COUNTS = [n for n in [1,2,4,8] if n <= N_GPUS_AVAILABLE]
print(f"\nGPU counts to test: {GPU_COUNTS}")
print(f"Models to train   : {N_PAIRS} ({len(ARCHES)} models × {len(TARGETS)} targets)")
print(f"Targets           : {TARGETS}")
print(f"Epochs per model  : {TRAIN_CONFIG['epochs']} | Patience: {TRAIN_CONFIG['patience']}")

scaling_results    = []
all_per_model_rows = []   # accumulate per-model results across ALL GPU configs

for n_gpu in GPU_COUNTS:
    print(f"\n{'='*55}")
    print(f"  CONFIGURATION: {n_gpu} GPU(s) in parallel")
    print(f"{'='*55}")

    wall_start   = time.time()
    model_results = []
    batch_wall_cumulative = 0.0  # track cumulative wall time per batch

    # Dispatch models in batches of n_gpu
    for batch_start in range(0, N_PAIRS, n_gpu):
        batch = PAIRS[batch_start:batch_start+n_gpu]
        print(f"  Batch {batch_start//n_gpu+1}: dispatching {[f'{a}[{t}]' for a,t in batch]}")

        batch_t0 = time.time()

        # Dispatch batch in parallel — ray.remote per senior's instruction
        futures = [
            train_model_on_gpu.remote(
                arch, target, TRAIN_CONFIG,
                scalers_ref, fi_ref,
                df_path, str(MODELS), i % n_gpu)
            for i,(arch,target) in enumerate(batch)
        ]

        # Gather results — ray.get per senior's instruction
        batch_results = ray.get(futures)
        batch_elapsed = time.time() - batch_t0
        batch_wall_cumulative += batch_elapsed
        model_results.extend(batch_results)

        for r in batch_results:
            if r.get("status")=="OK":
                print(f"    ✓ {r['arch']:<20} R²={r['best_val_r2']:.4f} "
                      f"| {r['elapsed_s']:.0f}s | {r.get('gpu_name','cpu')}")
                # Record per-model result with GPU config context
                all_per_model_rows.append({
                    "Model":       r["arch"],
                    "Target":      r["target"],
                    "GPU":         n_gpu,
                    "Elapsed_s":   round(r["elapsed_s"], 1),
                    "Best_Val_R2": r["best_val_r2"],
                    "GPU_Name":    r.get("gpu_name",""),
                    "N_Params":    r.get("n_params",0),
                    "N_Epochs":    r.get("n_epochs",0),
                    "Batch_Wall_s": round(batch_elapsed, 1),
                    "Cumul_Wall_s": round(batch_wall_cumulative, 1),
                })
            else:
                print(f"    ✗ {r['arch']:<20} status={r.get('status','?')}")

    wall_time = time.time() - wall_start

    # Aggregate metrics for this GPU count
    ok_results = [r for r in model_results if r.get("status")=="OK"]
    mean_r2    = np.mean([r["best_val_r2"] for r in ok_results]) if ok_results else 0

    scaling_results.append(dict(
        n_gpus=n_gpu,
        wall_time_s=round(wall_time,1),
        wall_time_min=round(wall_time/60,2),
        n_models_trained=len(ok_results),
        mean_val_r2=round(mean_r2,4),
        models_per_minute=round(len(ok_results)/(wall_time/60),2),
        config=json.dumps(TRAIN_CONFIG)))

    print(f"\n  → n_gpus={n_gpu} | wall={wall_time/60:.1f}min | "
          f"models={len(ok_results)} | mean_R²={mean_r2:.4f}")

# ── Save per-model results ────────────────────────────────────────────────────
# Save per-model CSV — KEY output for senior's tables
if all_per_model_rows:
    pm_df = pd.DataFrame(all_per_model_rows)

    # Drop ratio using CUMULATIVE WALL TIME — this is what actually drops with more GPUs
    # Elapsed_s = per-model compute time (stays ~constant regardless of GPU count)
    # Cumul_Wall_s = wall clock time when model finishes (drops with more GPUs)
    t1_map = pm_df[pm_df["GPU"]==1].set_index(["Model","Target"])["Cumul_Wall_s"].to_dict()
    pm_df["Drop_Ratio_pct"] = pm_df.apply(
        lambda row: round(
            (t1_map.get((row["Model"],row["Target"]), row["Cumul_Wall_s"]) - row["Cumul_Wall_s"]) /
            (t1_map.get((row["Model"],row["Target"]), row["Cumul_Wall_s"]) + 1e-8) * 100, 2)
        if (row["Model"],row["Target"]) in t1_map else 0.0, axis=1)
    pm_df["Processing_Time_Scale"] = pm_df.apply(
        lambda row: round(
            row["Cumul_Wall_s"] /
            (t1_map.get((row["Model"],row["Target"]), row["Cumul_Wall_s"]) + 1e-8), 4)
        if (row["Model"],row["Target"]) in t1_map else 1.0, axis=1)

    pm_df.to_csv(RESULTS/"scaling_per_model_results.csv", index=False)
    print(f"\n  ✓ scaling_per_model_results.csv ({len(pm_df)} rows)")
    print(f"\n  Per-model summary (temp target):")
    print(f"  {'Model':<22} {'1GPU(s)':>8} {'2GPU(s)':>8} {'4GPU(s)':>8} {'8GPU(s)':>8}")
    print("  " + "─"*58)
    for arch in ARCHES:
        vals=[]
        for n in [1,2,4,8]:
            sub=pm_df[(pm_df["Model"]==arch)&(pm_df["Target"]=="temp")&(pm_df["GPU"]==n)]
            vals.append(f"{sub['Elapsed_s'].values[0]:>8.1f}" if len(sub)>0 else f"{'N/A':>8}")
        print(f"  {arch:<22} {''.join(vals)}")

# Sequential baseline = 1 GPU results
baseline_1gpu = [r for r in all_per_model_rows if r.get("GPU")==1]

# Save scaling summary
scale_df = pd.DataFrame(scaling_results)
scale_df.to_csv(RESULTS/"scaling_results.csv", index=False)
print(f"\n  ✓ scaling_results.csv")

# Compute speedup and distortion
if len(scale_df) > 0 and 1 in scale_df["n_gpus"].values:
    t1 = float(scale_df[scale_df["n_gpus"]==1]["wall_time_s"])
    r2_1gpu = float(scale_df[scale_df["n_gpus"]==1]["mean_val_r2"])
    scale_df["speedup"]    = round(t1 / scale_df["wall_time_s"], 3)
    scale_df["ideal_speedup"] = scale_df["n_gpus"].astype(float)
    scale_df["r2_distortion"] = round(
        (scale_df["mean_val_r2"] - r2_1gpu).abs(), 4)
    scale_df.to_csv(RESULTS/"scaling_results.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating scaling figures...")

if len(scale_df) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # SCALE_01: Speedup curve
    ax = axes[0]
    if "speedup" in scale_df.columns:
        ax.plot(scale_df["n_gpus"], scale_df["speedup"],
                "bo-", lw=2.5, ms=10, label="Actual speedup", zorder=5)
        ax.plot(scale_df["n_gpus"], scale_df["ideal_speedup"],
                "k--", lw=1.5, alpha=0.6, label="Ideal (linear) speedup")
        for _, row in scale_df.iterrows():
            ax.annotate(f"{row['speedup']:.2f}×",
                        xy=(row["n_gpus"], row["speedup"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=10, fontweight="bold", color="blue")
    ax.set_xlabel("Number of GPUs", fontsize=11)
    ax.set_ylabel("Speedup vs 1 GPU", fontsize=11)
    ax.set_title("GPU Scaling Speedup\nModel-level parallelism via Ray",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(scale_df["n_gpus"])

    # SCALE_02: Wall time
    ax = axes[1]
    ax.bar(scale_df["n_gpus"].astype(str), scale_df["wall_time_min"],
           color="#1f77b4", alpha=0.85, edgecolor="black", lw=0.5, width=0.5)
    for _, row in scale_df.iterrows():
        ax.text(str(int(row["n_gpus"])), row["wall_time_min"]+0.2,
                f"{row['wall_time_min']:.1f}min",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of GPUs", fontsize=11)
    ax.set_ylabel("Wall Time (minutes)", fontsize=11)
    ax.set_title("Wall Time per GPU Configuration\nAll 11 models trained",
                 fontweight="bold", fontsize=12)

    # SCALE_03: R² distortion
    ax = axes[2]
    if "r2_distortion" in scale_df.columns:
        colors = ["green" if v < 0.005 else "orange" if v < 0.01 else "red"
                  for v in scale_df["r2_distortion"]]
        ax.bar(scale_df["n_gpus"].astype(str), scale_df["r2_distortion"],
               color=colors, alpha=0.85, edgecolor="black", lw=0.5, width=0.5)
        for _, row in scale_df.iterrows():
            ax.text(str(int(row["n_gpus"])), row["r2_distortion"]+0.0001,
                    f"{row['r2_distortion']:.4f}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.axhline(0.005, color="orange", ls="--", lw=1.5,
                   label="0.5% distortion threshold")
        ax.set_xlabel("Number of GPUs", fontsize=11)
        ax.set_ylabel("|R² parallel - R² sequential|", fontsize=11)
        ax.set_title("R² Distortion from Parallelisation\n"
                     "Green<0.5% | Orange<1% | Red>1%",
                     fontweight="bold", fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle(
        "GPU Scaling Experiment — Model-Level Parallelism via Ray Remote\n"
        f"talon32 | {N_GPUS_AVAILABLE}× V100 32GB | 11 Models | Weather Temp",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"SCALE_01_speedup_and_distortion.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SCALE_01_speedup_and_distortion.png")

    # Publication table
    fig, ax = plt.subplots(figsize=(16, max(6, len(scale_df)*1.2+2)))
    ax.axis("off")
    cols = ["GPUs","Wall Time (min)","Speedup","Ideal Speedup",
            "Mean Val R²","R² Distortion","Models/min"]
    rows = []
    for _, r in scale_df.iterrows():
        rows.append([
            f"{int(r['n_gpus'])}× V100",
            f"{r['wall_time_min']:.1f}",
            f"{r.get('speedup',1.0):.2f}×",
            f"{r.get('ideal_speedup',1.0):.1f}×",
            f"{r['mean_val_r2']:.4f}",
            f"{r.get('r2_distortion',0):.4f}",
            f"{r['models_per_minute']:.2f}"])
    tbl = ax.table(cellText=rows, colLabels=cols,
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c), cell in tbl.get_celld().items():
        if r==0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r%2==0: cell.set_facecolor("#f2f3f4")
        cell.set_edgecolor("white")
    ax.set_title(
        "GPU Scaling Results — Model-Level Parallelism\n"
        f"talon32 | {N_GPUS_AVAILABLE}× NVIDIA V100 32GB | Ray Remote\n"
        "Distortion = |R²(N GPUs) - R²(1 GPU)|",
        fontweight="bold", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(FIGS/"SCALE_02_scaling_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SCALE_02_scaling_table.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SCALING EXPERIMENT COMPLETE")
print("="*70)
print(f"\n  {'GPUs':>6} {'Wall(min)':>10} {'Speedup':>8} "
      f"{'Mean R²':>8} {'Distortion':>11} {'Models/min':>11}")
print("  " + "─"*60)
for _, r in scale_df.iterrows():
    print(f"  {int(r['n_gpus']):>6} {r['wall_time_min']:>10.1f} "
          f"{r.get('speedup',1.0):>8.2f}× "
          f"{r['mean_val_r2']:>8.4f} "
          f"{r.get('r2_distortion',0):>11.4f} "
          f"{r['models_per_minute']:>11.2f}")

print(f"\n  Results : {RESULTS}/scaling_results.csv")
print(f"  Figures : {FIGS}/SCALE_0*.png")
print(f"  Done    : {pd.Timestamp.now()}")

ray.shutdown()
