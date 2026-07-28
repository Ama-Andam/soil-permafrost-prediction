"""
================================================================================
postprocess_results.py — Per-Site Spatio-Temporal Evaluation
================================================================================
PURPOSE:
  Runs AFTER train_soil_spatial.py completes.
  Loads each checkpoint and evaluates per site (Bedrock, Transition, Upland, Wetland)
  to show true spatio-temporal coverage across all 4 sites × 4 models × 3 targets.

RUN ON TALON (login node — no GPU needed):
  python3 ~/postprocess_results.py

OUTPUT FILES (all in ~/figures_v3/):
  PP01_per_site_r2_heatmap.png     — R² heatmap: Model x Site x Target
  PP02_per_site_skill_heatmap.png  — Skill heatmap: Model x Site x Target
  PP03_per_site_bar_chart.png      — Grouped bars: all metrics per site
  PP04_per_site_timeseries.png     — Predicted vs Truth time series per site
  PP05_per_site_freeze_thaw.png    — Freeze-thaw accuracy per site
  PP06_spatial_field_snapshot.png  — Spatial field snapshot at one timestamp
  PP07_per_site_summary_table.png  — Publication-ready summary table
  postprocess_results.csv          — Full per-site results table
================================================================================
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/emmanuel.keku")
PREPROC_DIR = PROJECT_DIR / "preprocessed_v3"
RESULTS_DIR = PROJECT_DIR / "results_v3"
MODELS_DIR  = PROJECT_DIR / "models_v3" / "dl"
FIG_DIR     = PROJECT_DIR / "figures_v3"
LOG_PATH    = PROJECT_DIR / "logs" / "postprocess.log"

for d in [RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False
})

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("  POST-PROCESSING — Per-Site Spatio-Temporal Evaluation")
print(f"  Start: {pd.Timestamp.now()}")
print("=" * 70)

# ── Check PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    DEVICE = torch.device("cpu")  # CPU only — login node
    print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
except ImportError as e:
    print(f"FATAL: {e}"); sys.exit(1)

# ── Load preprocessed data ────────────────────────────────────────────────────
print("\nLoading preprocessed data...")
if not (PREPROC_DIR / "master_processed.csv").exists():
    print("FATAL: Run train_soil_spatial.py first"); sys.exit(1)

df = pd.read_csv(PREPROC_DIR / "master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC_DIR / "scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC_DIR / "feature_info.pkl", "rb") as f: FI = pickle.load(f)

snap_feat_scaler = SC["snap_feat_scaler"]
snap_tgt_scalers = SC["snap_tgt_scalers"]
SNAP_FEATURES    = FI["SNAP_FEATURES"]
TEMP_TARGETS     = FI["TEMP_TARGETS"]
SMAP_TARGETS     = FI["SMAP_TARGETS"]
MOIST_TARGETS    = FI["MOIST_TARGETS"]
TEMP_RESID_COLS  = FI["TEMP_RESID_COLS"]
SMAP_RESID_COLS  = FI["SMAP_RESID_COLS"]
MOIST_RESID_COLS = FI["MOIST_RESID_COLS"]
N_SNAP_FEATURES  = FI["N_SNAP_FEATURES"]

LOCATIONS = pd.DataFrame(FI["LOCATIONS"])
N_LOCS    = FI["N_LOCS"]
SITES     = sorted(df["Site"].unique().tolist())

print(f"  {len(df):,} rows | {N_LOCS} locations | Sites: {SITES}")

# ── Site → location index mapping ─────────────────────────────────────────────
# Build which location indices (0-255) belong to each site
loc_to_idx = {(float(r.Latitude), float(r.Longitude)): i
              for i, r in LOCATIONS.iterrows()}

site_loc_indices = {}
for site in SITES:
    site_df = df[df["Site"] == site][["Latitude","Longitude"]].drop_duplicates()
    idxs = []
    for _, row in site_df.iterrows():
        idx = loc_to_idx.get((float(row.Latitude), float(row.Longitude)))
        if idx is not None:
            idxs.append(idx)
    site_loc_indices[site] = sorted(set(idxs))
    print(f"  {site:<20}: {len(site_loc_indices[site])} locations")

# ── Model definitions (copy from train script — needed for checkpoint loading) ─
class GraphConv(nn.Module):
    def __init__(self, in_d, out_d, dp=0.1):
        super().__init__()
        self.W = nn.Linear(in_d, out_d, bias=False)
        self.n = nn.LayerNorm(out_d)
        self.d = nn.Dropout(dp)
        self.a = nn.GELU()
    def forward(self, H, A):
        if A.dim() == 3: A = A[0]
        if A.dim() == 4: A = A[0,0]
        A_b = A.unsqueeze(0).expand(H.shape[0], -1, -1)
        return self.a(self.n(torch.bmm(A_b, self.W(self.d(H)))))

class SpatialBiGRU(nn.Module):
    def __init__(self, nf, h=96, nl=2, nh=4, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,h)
        self.gru=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                        dropout=dp if nl>1 else 0.)
        d2=h*2
        self.attn=nn.MultiheadAttention(d2,nh,dropout=dp,batch_first=True)
        self.n1=nn.LayerNorm(d2); self.n2=nn.LayerNorm(d2)
        self.ffn=nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),
                                nn.Dropout(dp),nn.Linear(d2*2,d2))
        self.red=nn.Linear(d2,h)
        self.gcn=nn.ModuleList([GraphConv(h,h,dp) for _ in range(gl)])
        self.head=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),
                                 nn.Dropout(dp),nn.Linear(h,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.gru(h); a,_=self.attn(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ffn(h))
        h=self.red(h[:,-1,:]).reshape(B,N,-1)
        hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.head(torch.cat([h,hg],dim=-1))

class MambaBlock(nn.Module):
    def __init__(self, d, ds=16, dc=4, ex=2, dp=0.1):
        super().__init__()
        self.di=d*ex; self.ds=ds
        self.ip=nn.Linear(d,self.di*2,bias=False)
        self.cv=nn.Conv1d(self.di,self.di,dc,padding=dc-1,groups=self.di,bias=True)
        self.silu=nn.SiLU()
        self.xp=nn.Linear(self.di,ds*2+self.di,bias=False)
        self.dtp=nn.Linear(self.di,self.di,bias=True)
        A_=torch.arange(1,ds+1,dtype=torch.float32).unsqueeze(0).repeat(self.di,1)
        self.Al=nn.Parameter(torch.log(A_))
        self.D_=nn.Parameter(torch.ones(self.di))
        self.op=nn.Linear(self.di,d,bias=False)
        self.dr=nn.Dropout(dp); self.nm=nn.LayerNorm(d)
    def scan(self, x):
        B,L,D=x.shape; S=self.ds
        xd=self.xp(x); dl,Bp,C=xd.split([D,S,S],dim=-1)
        dl=F.softplus(self.dtp(dl))
        A__=-torch.exp(self.Al.float())
        dA=torch.exp(torch.einsum("bld,ds->blds",dl,A__))
        dB=torch.einsum("bld,bls->blds",dl,Bp)
        h=torch.zeros(B,D,S,device=x.device,dtype=x.dtype); ys=[]
        for i in range(L):
            h=dA[:,i]*h+dB[:,i]*x[:,i,:,None]
            ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
        return torch.stack(ys,dim=1)*self.D_
    def forward(self, x):
        r=x; xz=self.ip(x); x_,z=xz.chunk(2,dim=-1)
        x_=self.silu(self.cv(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        y=self.scan(x_)*self.silu(z)
        return self.nm(r+self.op(self.dr(y)))

class SpatialMamba(nn.Module):
    def __init__(self, nf, d=96, nl=4, ds=16, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.mb=nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gcn=nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for b in self.mb: h=b(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1)
        hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.hd(torch.cat([h,hg],dim=-1))

class S4Layer(nn.Module):
    def __init__(self, d, ds=64, dp=0.1):
        super().__init__()
        self.ds=ds
        def hippo(N):
            A=torch.zeros(N,N)
            for n in range(N):
                for m in range(n): A[n,m]=-(2*n+1)**.5*(2*m+1)**.5
                A[n,n]=-(n+1)
            return A
        self.A_=nn.Parameter(hippo(ds),requires_grad=False)
        self.B_=nn.Parameter(torch.randn(ds,1)*0.01)
        self.C_=nn.Parameter(torch.randn(d,ds))
        self.D_=nn.Parameter(torch.ones(d))
        self.nm=nn.LayerNorm(d); self.dr=nn.Dropout(dp)
        self.ot=nn.Linear(d,d); self.mx=nn.Linear(d*2,d)
    def scan(self, u):
        B,L,d=u.shape; dA=torch.matrix_exp(self.A_); dB=self.B_.squeeze(-1)
        h=torch.zeros(B,d,self.ds,device=u.device); ys=[]
        for t in range(L):
            h=h@dA.T+u[:,t,:,None]*dB
            ys.append((h*self.C_.unsqueeze(0)).sum(-1)+self.D_*u[:,t,:])
        return torch.stack(ys,dim=1)
    def forward(self, x):
        yf=self.scan(x); yr=self.scan(x.flip(1)).flip(1)
        return self.nm(x+self.dr(self.ot(self.mx(torch.cat([yf,yr],dim=-1)))))

class SpatialS4(nn.Module):
    def __init__(self, nf, d=96, nl=4, ds=64, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.ly=nn.ModuleList([S4Layer(d,ds,dp) for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gcn=nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for l in self.ly: h=l(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1)
        hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.hd(torch.cat([h,hg],dim=-1))

class SpatialFuseMoE(nn.Module):
    def __init__(self, nf, d=96, ne=4, tk=2, ds=16, nsl=2, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.ne=ne; self.tk=tk; self.d=d
        self.em=nn.Linear(nf,d)
        self.ex=nn.ModuleList([
            MambaBlock(d,ds,dp=dp),
            nn.GRU(d,d,batch_first=True),
            nn.Sequential(nn.Conv1d(d,d,7,padding=3,groups=d),nn.Conv1d(d,d,1),
                          nn.GELU(),nn.AdaptiveAvgPool1d(1)),
            nn.GRU(d,d,batch_first=True),
        ])
        self.enm=nn.ModuleList([nn.LayerNorm(d) for _ in range(ne)])
        self.gt=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ne))
        self.bb=nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nsl)])
        self.gcn=nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.nm=nn.LayerNorm(d)
        self.hd=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def _expert(self, i, h):
        ex=self.ex[i]
        if isinstance(ex,MambaBlock): return self.enm[i](ex(h)[:,-1,:])
        if isinstance(ex,nn.GRU): _,ht=ex(h); return self.enm[i](ht[-1])
        return self.enm[i](ex(h.transpose(1,2)).squeeze(-1))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        g_in=h.mean(1); lg=self.gt(g_in)
        tv,ti=lg.topk(self.tk,dim=-1)
        gs=torch.nn.functional.softmax(tv,dim=-1)
        gs_s=torch.nn.functional.softmax(lg,dim=-1)
        imp=gs_s.mean(0); ld=(gs_s>1/self.ne).float().mean(0)
        aux=(imp*ld).sum()*self.ne
        eo=[self._expert(i,h) for i in range(self.ne)]
        Es=torch.stack(eo,dim=1)
        sel=torch.gather(Es,1,ti.unsqueeze(-1).expand(-1,-1,self.d))
        fsd=(sel*gs.unsqueeze(-1)).sum(1)
        fs=fsd.unsqueeze(1).expand(-1,L,-1)+h
        for b in self.bb: fs=b(fs)
        ho=self.nm(fs[:,-1,:]).reshape(B,N,-1)
        hg=ho
        for g in self.gcn: hg=g(hg,A)
        return self.hd(torch.cat([ho,hg],dim=-1)), aux

ARCH_MAP = {
    "SpatialBiGRU"  : lambda nt: SpatialBiGRU(N_SNAP_FEATURES,h=96,nl=2,nh=4,N=N_LOCS,gl=2,nt=nt),
    "SpatialMamba"  : lambda nt: SpatialMamba( N_SNAP_FEATURES,d=96,nl=4,ds=16,N=N_LOCS,gl=2,nt=nt),
    "SpatialS4"     : lambda nt: SpatialS4(    N_SNAP_FEATURES,d=96,nl=4,ds=64,N=N_LOCS,gl=2,nt=nt),
    "SpatialFuseMoE": lambda nt: SpatialFuseMoE(N_SNAP_FEATURES,d=96,ne=4,tk=2,ds=16,nsl=2,N=N_LOCS,gl=2,nt=nt),
}

# ── Spatial graph ─────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree

def build_graph(locs_df, k=6):
    coords = locs_df[["Latitude","Longitude"]].values.astype(np.float32)
    N      = len(coords)
    scaled = coords * np.array([111.0, 63.0], dtype=np.float32)
    tree   = cKDTree(scaled)
    dists, idxs = tree.query(scaled, k=min(k+1, N))
    sigma  = np.median(dists[:,1:]) + 1e-8
    A      = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for jp in range(1, dists.shape[1]):
            j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
            A[i,j]+=w; A[j,i]+=w
    A += np.eye(N)
    D = A.sum(1, keepdims=True)**0.5
    return (A / (D * D.T + 1e-8)).astype(np.float32)

A_norm = torch.tensor(build_graph(LOCATIONS, k=6))
loc_coords = LOCATIONS[["Latitude","Longitude"]].values

# ── Dataset for per-site evaluation ───────────────────────────────────────────
class SiteEvalDataset(Dataset):
    """Builds spatial snapshot dataset for test split only."""
    def __init__(self, df, snap_features, resid_cols, approx_cols, raw_cols,
                 feat_scaler, tgt_scaler, lookback=24, stride=24):
        self.A = A_norm
        N=N_LOCS; nf=len(snap_features); nt=len(resid_cols)
        sub = df[df["split"]=="test"].copy()
        all_ts = sorted(sub["time_utc"].unique())
        T = len(all_ts)
        if T < lookback+2:
            self.X=self.yr=self.ya=self.yw=self.ts_idx=torch.zeros(0); return

        ts_to_i  = {ts:i for i,ts in enumerate(all_ts)}
        loc_to_i = {(float(r.Latitude),float(r.Longitude)):i
                    for i,r in LOCATIONS.iterrows()}

        sub2 = sub.copy()
        sub2["_ti"] = sub2["time_utc"].map(ts_to_i)
        sub2["_ni"] = [loc_to_i.get((float(la),float(lo)))
                       for la,lo in zip(sub2["Latitude"].astype(float),
                                        sub2["Longitude"].astype(float))]
        sub2 = sub2.dropna(subset=["_ti","_ni"])
        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
        ti_arr=sub2["_ti"].values; ni_arr=sub2["_ni"].values

        X_full =np.full((T,N,nf), np.nan,dtype=np.float32)
        yr_full=np.full((T,N,nt), np.nan,dtype=np.float32)
        ya_full=np.full((T,N,len(approx_cols)),np.nan,dtype=np.float32)
        yw_full=np.full((T,N,len(raw_cols)),np.nan,dtype=np.float32)

        X_full[ti_arr,ni_arr,:] = feat_scaler.transform(
            sub2[snap_features].fillna(0).values).astype(np.float32)
        yr_full[ti_arr,ni_arr,:] = tgt_scaler.transform(
            sub2[resid_cols].fillna(0).values).astype(np.float32)
        ya_full[ti_arr,ni_arr,:] = sub2[approx_cols].fillna(0).values.astype(np.float32)
        yw_full[ti_arr,ni_arr,:] = sub2[raw_cols].fillna(0).values.astype(np.float32)

        tidxs = list(range(lookback, T, stride))
        Xl=[]; yrl=[]; yal=[]; ywl=[]; tsl=[]
        for ti in tidxs:
            Xw=X_full[ti-lookback:ti]; yri=yr_full[ti]
            yai=ya_full[ti]; ywi=yw_full[ti]
            if np.isnan(Xw).mean()>0.2: continue
            Xl.append(np.nan_to_num(Xw,nan=0.0))
            yrl.append(np.nan_to_num(yri,nan=0.0))
            yal.append(np.nan_to_num(yai,nan=0.0))
            ywl.append(np.nan_to_num(ywi,nan=0.0))
            tsl.append(all_ts[ti])

        if not Xl:
            self.X=self.yr=self.ya=self.yw=self.ts_idx=torch.zeros(0); return

        self.X  =torch.tensor(np.array(Xl),dtype=torch.float32)
        self.yr =torch.tensor(np.array(yrl),dtype=torch.float32)
        self.ya =torch.tensor(np.array(yal),dtype=torch.float32)
        self.yw =torch.tensor(np.array(ywl),dtype=torch.float32)
        self.timestamps = tsl
        print(f"  Test dataset: {len(self.X)} samples | X={tuple(self.X.shape[1:])}")

    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return self.X[i],self.yr[i],self.ya[i],self.yw[i],self.A


def compute_site_metrics(yt_np, yp_np, site_idxs):
    """Compute metrics for locations belonging to one site."""
    if not site_idxs: return {}
    yt = yt_np[:,site_idxs,0].flatten()
    yp = yp_np[:,site_idxs,0].flatten()
    mk = ~(np.isnan(yt)|np.isnan(yp))
    yt=yt[mk]; yp=yp[mk]
    if len(yt)<5: return {}
    r2   = float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    r    = float(np.corrcoef(yt,yp)[0,1])
    kge  = float(1-np.sqrt((r-1)**2+(np.std(yp)/(np.std(yt)+1e-10)-1)**2+
                            (np.mean(yp)/(np.mean(yt)+1e-10)-1)**2))
    frz  = float(np.mean((yt<0).astype(int)==(yp<0).astype(int))*100)
    bias = float(np.mean(yp-yt))
    return dict(R2=round(r2,4),RMSE=round(rmse,4),KGE=round(kge,4),
                FreezeAcc=round(frz,2),Bias=round(bias,4),N=int(mk.sum()))


# ── Main evaluation loop ───────────────────────────────────────────────────────
TARGET_GROUPS = [
    ("temp",  TEMP_TARGETS,  TEMP_RESID_COLS,
     [f"{t}_approx" for t in TEMP_TARGETS if f"{t}_approx" in df.columns]),
    ("smap",  SMAP_TARGETS,  SMAP_RESID_COLS,
     [f"{t}_approx" for t in SMAP_TARGETS if f"{t}_approx" in df.columns]),
    ("moist", MOIST_TARGETS, MOIST_RESID_COLS,
     [f"{t}_approx" for t in MOIST_TARGETS if f"{t}_approx" in df.columns]),
]

TGT_LABELS   = {"temp":"Weather Temp (°C)","smap":"SMAP Temp L1 (K)","moist":"Soil Moisture"}
MODEL_COLORS = {"SpatialBiGRU":"#1f77b4","SpatialMamba":"#ff7f0e",
                "SpatialS4":"#2ca02c","SpatialFuseMoE":"#9467bd"}
SITE_COLORS  = {"Bedrock":"#1f77b4","Transition":"#ff7f0e",
                "Upland":"#2ca02c","Wetland":"#d62728"}
ARCHES = ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]

all_site_results = []
timeseries_store = {}   # store predictions for time series plots

print("\n" + "="*60)
print("  EVALUATING PER SITE")
print("="*60)

for (tgt_name, raw_cols, resid_cols, approx_cols) in TARGET_GROUPS:
    tgt_sc = snap_tgt_scalers.get(tgt_name)
    if tgt_sc is None: continue
    if not resid_cols: continue
    avail_approx = [c for c in approx_cols if c in df.columns]
    avail_raw    = [c for c in raw_cols    if c in df.columns]
    if not avail_raw: continue

    print(f"\n  Target: {TGT_LABELS.get(tgt_name,tgt_name)}")
    ds = SiteEvalDataset(df, SNAP_FEATURES, resid_cols, avail_approx, avail_raw,
                         snap_feat_scaler, tgt_sc, lookback=24, stride=24)
    if len(ds)==0: continue
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    for arch in ARCHES:
        ckpt_path = MODELS_DIR / f"{arch}_{tgt_name}_v3_best.pt"
        if not ckpt_path.exists():
            print(f"  ✗ {arch} [{tgt_name}] — checkpoint missing"); continue

        try:
            sv    = torch.load(ckpt_path, map_location=DEVICE)
            n_tgt = len(resid_cols)
            model = ARCH_MAP[arch](n_tgt).to(DEVICE)
            model.load_state_dict(sv["state_dict"])
            model.eval()
            is_moe = (arch=="SpatialFuseMoE")

            yt_list=[]; yp_list=[]; ya_list=[]
            with torch.no_grad():
                for batch in loader:
                    X,yr,ya,yw,A_=[b.to(DEVICE) for b in batch]
                    out=model(X,A_)
                    pred=out[0] if is_moe else out
                    B_,N_,T_=pred.shape
                    pr=pred.cpu().float().numpy()
                    pr_r=tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
                    yt_list.append(yw.cpu().numpy())
                    yp_list.append(ya.cpu().numpy()+pr_r)
                    ya_list.append(ya.cpu().numpy())

            yt_all=np.concatenate(yt_list,0)   # (S, N, T)
            yp_all=np.concatenate(yp_list,0)
            ya_all=np.concatenate(ya_list,0)

            # Store for time series plots
            key = f"{arch}_{tgt_name}"
            timeseries_store[key] = dict(yt=yt_all, yp=yp_all,
                                         timestamps=ds.timestamps)

            # Per-site metrics
            for site in SITES:
                s_idxs = site_loc_indices.get(site, [])
                m = compute_site_metrics(yt_all, yp_all, s_idxs)
                if not m: continue

                # Skill score vs seasonal
                yt_s = yt_all[:,s_idxs,0].flatten()
                yp_s = yp_all[:,s_idxs,0].flatten()
                ya_s = ya_all[:,s_idxs,0].flatten()
                mk   = ~(np.isnan(yt_s)|np.isnan(yp_s))
                skill= float(1-np.mean((yt_s[mk]-yp_s[mk])**2)/
                             (np.mean((yt_s[mk]-ya_s[mk])**2)+1e-10))

                all_site_results.append(dict(
                    Model=arch, Target=tgt_name, Site=site,
                    **m, Skill=round(skill,4),
                    n_locations=len(s_idxs)))
                print(f"  ✓ {arch:<20} [{tgt_name}] {site:<12} "
                      f"R²={m['R2']:.4f} Skill={skill:.4f} "
                      f"Freeze={m['FreezeAcc']:.1f}%")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {arch} [{tgt_name}]: {e}")

# ── Save results CSV ───────────────────────────────────────────────────────────
site_df = pd.DataFrame(all_site_results)
site_df.to_csv(RESULTS_DIR/"postprocess_results.csv", index=False)
print(f"\n  Saved: {RESULTS_DIR}/postprocess_results.csv")
print(f"  {len(site_df)} records")

if len(site_df) == 0:
    print("No results — check checkpoints exist in ~/models_v3/dl/")
    sys.exit(0)

targets  = sorted(site_df["Target"].unique())
sites    = sorted(site_df["Site"].unique())
models   = [m for m in ARCHES if m in site_df["Model"].unique()]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  GENERATING FIGURES")
print("="*55)

# ── PP01: R² Heatmap — Model × Site (per target) ─────────────────────────────
for tgt in targets:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty: continue
    pv = sub.pivot_table(index="Model", columns="Site",
                          values="R2", aggfunc="mean").round(4)
    if pv.empty: continue
    pv = pv.reindex(index=[m for m in ARCHES if m in pv.index],
                    columns=SITES)
    fig,ax=plt.subplots(figsize=(14,8))
    sns.heatmap(pv,ax=ax,cmap="RdYlGn",vmin=0.90,vmax=1.0,
                annot=True,fmt=".4f",linewidths=0.5,linecolor="white",
                annot_kws={"size":13,"weight":"bold"},
                cbar_kws={"label":"R²","shrink":0.85})
    ax.set_title(f"R² per Model × Site | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"True Spatial Field | All locations per site | Test 2025",
                 fontweight="bold",fontsize=13)
    ax.tick_params(axis="x",rotation=20,labelsize=11)
    ax.tick_params(axis="y",rotation=0, labelsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"PP01_r2_heatmap_model_site_{tgt}.png",
                dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP01 R² heatmap [{tgt}]")

# ── PP02: Skill Heatmap — Model × Site ───────────────────────────────────────
for tgt in targets:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty: continue
    pv = sub.pivot_table(index="Model", columns="Site",
                          values="Skill", aggfunc="mean").round(4)
    if pv.empty: continue
    pv = pv.reindex(index=[m for m in ARCHES if m in pv.index],
                    columns=SITES)
    fig,ax=plt.subplots(figsize=(14,8))
    sns.heatmap(pv,ax=ax,cmap="RdYlGn",vmin=-0.2,vmax=1.0,
                annot=True,fmt=".4f",linewidths=0.5,linecolor="white",
                annot_kws={"size":13,"weight":"bold"},
                cbar_kws={"label":"Skill vs Seasonal","shrink":0.85})
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v=pv.iloc[i,j]
            if not np.isnan(v) and v<0:
                ax.add_patch(plt.Rectangle((j,i),1,1,fill=False,
                                            edgecolor="red",lw=2.5))
    ax.set_title(f"Skill Score per Model × Site | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"Improvement over seasonal baseline | Test 2025",
                 fontweight="bold",fontsize=13)
    ax.tick_params(axis="x",rotation=20,labelsize=11)
    ax.tick_params(axis="y",rotation=0, labelsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"PP02_skill_heatmap_model_site_{tgt}.png",
                dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP02 Skill heatmap [{tgt}]")

# ── PP03: Grouped bar — all metrics per site ──────────────────────────────────
metrics_bar = [("R2","R²",0.90,1.0),("Skill","Skill",-0.2,1.0),
               ("KGE","KGE",0.85,1.0),("FreezeAcc","Freeze Acc (%)",85,101)]
for tgt in targets:
    sub = site_df[site_df["Target"]==tgt]
    if sub.empty: continue
    fig,axes=plt.subplots(2,2,figsize=(22,14))
    axf=axes.flatten()
    x=np.arange(len(sites)); w=0.8/max(len(models),1)
    for ai,(metric,mlbl,ymin,ymax) in enumerate(metrics_bar):
        ax=axf[ai]
        if metric not in sub.columns: continue
        for mi,model in enumerate(models):
            msub=sub[sub["Model"]==model]
            vals=[msub[msub["Site"]==s][metric].mean()
                  if len(msub[msub["Site"]==s])>0 else 0 for s in sites]
            bars=ax.bar(x+mi*w-0.4+w/2,vals,width=w*0.9,label=model,
                        color=MODEL_COLORS.get(model,"grey"),
                        alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_height()+0.002,f"{v:.3f}",
                                ha="center",va="bottom",fontsize=7,fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(sites,fontsize=10)
        ax.set_ylabel(mlbl,fontsize=10); ax.set_title(mlbl,fontweight="bold")
        ax.set_ylim(ymin,ymax); ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f"All Metrics per Site | {TGT_LABELS.get(tgt,tgt)}\n"
                 f"Bedrock | Transition | Upland | Wetland | Test 2025",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"PP03_grouped_bar_per_site_{tgt}.png",
                dpi=150,bbox_inches="tight")
    plt.close(); print(f"  ✓ PP03 grouped bars [{tgt}]")

# ── PP04: Time series — Predicted vs Truth per site ───────────────────────────
best_arch = (site_df[site_df["Target"]=="temp"]
             .groupby("Model")["R2"].mean()
             .idxmax() if "temp" in site_df["Target"].values else ARCHES[0])
key = f"{best_arch}_temp"
if key in timeseries_store:
    store = timeseries_store[key]
    yt_all = store["yt"]          # (S, N, T)
    yp_all = store["yp"]
    tss    = store["timestamps"]

    fig,axes=plt.subplots(len(SITES),1,figsize=(20,5*len(SITES)),sharex=True)
    for ax,site in zip(axes,SITES):
        s_idxs = site_loc_indices.get(site,[])
        if not s_idxs: continue
        yt_site = yt_all[:,s_idxs,0].mean(axis=1)   # mean over locations in site
        yp_site = yp_all[:,s_idxs,0].mean(axis=1)
        ts_plot = pd.to_datetime(tss)
        ax.plot(ts_plot, yt_site, lw=1.5, color=SITE_COLORS.get(site,"grey"),
                label="True", alpha=0.9)
        ax.plot(ts_plot, yp_site, lw=1.5, color="black",
                ls="--", label="Predicted", alpha=0.8)
        ax.axhline(0,color="grey",ls=":",lw=1,alpha=0.5)
        r2_site = site_df[(site_df["Model"]==best_arch) &
                           (site_df["Target"]=="temp") &
                           (site_df["Site"]==site)]["R2"].values
        r2_str = f"R²={r2_site[0]:.4f}" if len(r2_site)>0 else ""
        ax.set_ylabel(f"{site}\nSoil Temp (°C)",fontsize=10)
        ax.set_title(f"{site} — True vs Predicted | {r2_str}",
                     fontweight="bold",fontsize=11)
        ax.legend(fontsize=9,loc="upper right")
    axes[-1].set_xlabel("Date",fontsize=11)
    fig.suptitle(f"Predicted vs True Soil Temperature | {best_arch}\n"
                 f"All 4 Sites | Mean over site locations | Test 2025",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"PP04_timeseries_per_site.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP04 time series per site")

# ── PP05: Freeze-Thaw accuracy bar per site ───────────────────────────────────
fig,ax=plt.subplots(figsize=(18,9))
x=np.arange(len(sites)); w=0.8/max(len(models),1)
for mi,model in enumerate(models):
    for tgt in targets[:1]:   # Weather temp only for freeze
        sub=site_df[(site_df["Model"]==model)&(site_df["Target"]==tgt)]
        vals=[sub[sub["Site"]==s]["FreezeAcc"].mean()
              if len(sub[sub["Site"]==s])>0 else 0 for s in sites]
        bars=ax.bar(x+mi*w-0.4+w/2,vals,width=w*0.9,label=model,
                    color=MODEL_COLORS.get(model,"grey"),
                    alpha=0.85,edgecolor="black",lw=0.5)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                    f"{v:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sites,fontsize=12)
ax.set_ylabel("Freeze-Thaw Accuracy (%)",fontsize=12)
ax.set_ylim(85,102)
ax.axhline(95,color="orange",ls="--",lw=2,alpha=0.8,label="95% threshold")
ax.axhline(100,color="green",ls="--",lw=1.5,alpha=0.5,label="Perfect")
ax.set_title("Freeze-Thaw Transition Accuracy per Site\n"
             "Bedrock | Transition | Upland | Wetland | Test 2025",
             fontweight="bold",fontsize=13)
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
plt.savefig(FIG_DIR/"PP05_freeze_thaw_per_site.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ PP05 freeze-thaw per site")

# ── PP06: Spatial field snapshot at one timestamp ─────────────────────────────
key = f"{best_arch}_temp"
if key in timeseries_store:
    store  = timeseries_store[key]
    yt_all = store["yt"]
    yp_all = store["yp"]
    mid_idx= len(yt_all)//2   # pick middle test timestamp
    yt_snap= yt_all[mid_idx,:,0]
    yp_snap= yp_all[mid_idx,:,0]
    ts_str = str(store["timestamps"][mid_idx])[:10] if store["timestamps"] else "Test"

    fig,axes=plt.subplots(1,3,figsize=(24,9))
    vmin=min(np.nanmin(yt_snap),np.nanmin(yp_snap))
    vmax=max(np.nanmax(yt_snap),np.nanmax(yp_snap))

    for ax,data,title in [
        (axes[0],yt_snap,"True Spatial Field"),
        (axes[1],yp_snap,"Predicted Spatial Field"),
        (axes[2],yp_snap-yt_snap,"Residual (Pred - True)")]:
        cmap="RdBu_r" if "Residual" in title else "RdBu_r"
        vm = max(abs(np.nanmin(data)),abs(np.nanmax(data))) if "Residual" in title else None
        sc=ax.scatter(loc_coords[:,1],loc_coords[:,0],
                      c=data,cmap=cmap,
                      vmin=-vm if vm else vmin,
                      vmax=vm  if vm else vmax,
                      s=50,edgecolors="black",linewidth=0.3,zorder=5)
        plt.colorbar(sc,ax=ax,label="Soil Temp (°C)",shrink=0.85)
        ax.set_xlabel("Longitude",fontsize=10)
        ax.set_ylabel("Latitude",fontsize=10)
        ax.set_title(title,fontweight="bold",fontsize=12)
        for site,(lat,lon) in [("Bedrock",(66.25,-150.7)),
                                ("Transition",(67.5,-150.5)),
                                ("Upland",(68.5,-150.5)),
                                ("Wetland",(67.2,-151.0))]:
            ax.annotate(site,xy=(lon,lat),fontsize=8,fontweight="bold",
                        color="navy",ha="center",
                        bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.7))

    fig.suptitle(f"Spatial Field Snapshot | {best_arch} | {ts_str}\n"
                 f"True vs Predicted vs Residual | All 4 Sites | {N_LOCS} locations",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"PP06_spatial_field_snapshot.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  ✓ PP06 spatial field snapshot")

# ── PP07: Publication summary table ──────────────────────────────────────────
best_model = (site_df.groupby("Model")["R2"].mean().idxmax()
              if "Model" in site_df.columns else ARCHES[0])
summary = (site_df[site_df["Model"]==best_model]
           .groupby(["Site","Target"])
           [["R2","RMSE","KGE","Skill","FreezeAcc"]]
           .mean().round(4).reset_index())

fig,ax=plt.subplots(figsize=(18,max(6,len(summary)*0.6+2)))
ax.axis("off")
cols=["Site","Target","R²","RMSE","KGE","Skill","Freeze Acc (%)"]
cell_vals=[]
for _,row in summary.iterrows():
    tgt_lbl=TGT_LABELS.get(row["Target"],row["Target"])
    cell_vals.append([row["Site"],tgt_lbl,
                      f"{row['R2']:.4f}",f"{row['RMSE']:.4f}",
                      f"{row['KGE']:.4f}",f"{row['Skill']:.4f}",
                      f"{row['FreezeAcc']:.1f}%"])

tbl=ax.table(cellText=cell_vals,colLabels=cols,
              cellLoc="center",loc="center",
              bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
for (r,c),cell in tbl.get_celld().items():
    if r==0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white",fontweight="bold")
    elif r%2==0:
        cell.set_facecolor("#ecf0f1")
    cell.set_edgecolor("white")
ax.set_title(f"Publication Summary Table | Best Model: {best_model}\n"
             f"Per-Site Spatio-Temporal Results | All Target Groups | Test 2025",
             fontweight="bold",fontsize=13,pad=20)
plt.tight_layout()
plt.savefig(FIG_DIR/"PP07_publication_summary_table.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ PP07 publication summary table")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  POST-PROCESSING COMPLETE")
print("="*70)
print(f"\n  Best overall model: {best_model}")
print(f"\n  Per-site results summary:")
print(f"  {'Model':<20} {'Site':<12} {'Target':<8} {'R2':>8} {'Skill':>8} {'Freeze':>8}")
print("  " + "─"*68)
for _,row in site_df.sort_values(["Target","Site","R2"],ascending=[True,True,False]).iterrows():
    print(f"  {row['Model']:<20} {row['Site']:<12} {row['Target']:<8} "
          f"{row['R2']:>8.4f} {row['Skill']:>8.4f} {row['FreezeAcc']:>7.1f}%")

print(f"\n  Figures saved to: {FIG_DIR}")
figs = sorted(FIG_DIR.glob("PP*.png"))
for f in figs:
    print(f"    {f.name}")
print(f"\n  Results CSV: {RESULTS_DIR}/postprocess_results.csv")
print(f"  Completed  : {pd.Timestamp.now()}")
print("="*70)
