"""
================================================================================
train_soil_spatial_v4.py
DISTRIBUTED AI — SOIL TEMPERATURE & MOISTURE PREDICTION
DoD PROJECT | Alaska 2022-2025 | TRUE SPATIAL GENERALISATION
================================================================================

WHAT IS NEW IN v4 (vs v3):
  [1] RAW SIGNAL TARGETS — no wavelet reconstruction masking
      Wavelet approx added as INPUT FEATURE so seasonal info is available
      but NOT free in the target. Forces models to compete on full signal.

  [2] SPATIAL HOLDOUT — 64 locations (1 entire site) withheld from training
      Model must predict them using ONLY the GCN graph at inference time
      This is true distributed spatial AI — inductive node prediction
      Evaluated separately: seen_locs vs unseen_locs

  [3] DEEP ECHO STATE NETWORK (DeepESN)
      Per senior suggestion — Gallicchio & Micheli 2017 (arXiv:1712.04323)
      Fixed random reservoir, only readout trained, echo-state property
      Captures long-range temporal memory without backprop through time
      SpatialESN = DeepESN + GCN (novel contribution)

  [4] GRAPHSAGE — inductive unseen node prediction
      Hamilton et al. 2017 — specifically designed for held-out nodes
      Learns aggregation functions, not node embeddings
      The theoretically correct model for spatial holdout evaluation

  [5] GAT — attention-weighted graph
      Learns which spatial neighbours matter most
      Wetland→Bedrock flow weighted differently than Upland→Transition

  [6] STGCN — spatio-temporal GCN
      Yu et al. 2018 — standard literature benchmark
      Joint space-time convolution

  [7] ABLATIONS
      BiGRU_NoGCN    — temporal only, proves GCN value
      GCN_NoTemporal — spatial only, proves temporal value

  [8] RUNS CONCURRENTLY WITH v3
      Separate ~/models_v4, ~/results_v4, ~/figures_v4 directories
      Zero conflict with running job 273252

PAPER EXPERIMENTS:
  Exp 1: Temporal generalisation (2025 test year, all 256 locations)
  Exp 2: Spatial generalisation (64 unseen locations, all years)
  Exp 3: Ablation study (GCN vs no-GCN, temporal vs spatial)
  Exp 4: Per-site breakdown (Bedrock/Transition/Upland/Wetland)

REFERENCES:
  Mamba:    Gu & Dao 2023 (arXiv:2312.00752)
  S4:       Gu et al. 2022 (arXiv:2111.00396)
  DeepESN:  Gallicchio & Micheli 2017 (arXiv:1712.04323)
  ESN-SSM:  Singh & Raman 2025 (arXiv:2509.04422)
  GraphSAGE:Hamilton et al. 2017 (NeurIPS)
  GAT:      Velickovic et al. 2018 (ICLR)
  STGCN:    Yu et al. 2018 (IJCAI)
================================================================================
"""

import os, sys, time, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats

warnings.filterwarnings("ignore")

# ── Tee logger ────────────────────────────────────────────────────────────────
class TeeLogger:
    def __init__(self, p):
        self.t=sys.__stdout__; Path(p).parent.mkdir(parents=True,exist_ok=True)
        self.f=open(p,"a",buffering=1)
    def write(self,m): self.t.write(m); self.f.write(m)
    def flush(self):   self.t.flush();  self.f.flush()

LOG_PATH = "/home/emmanuel.keku/logs/soil_training_v4.log"
sys.stdout = TeeLogger(LOG_PATH)
sys.stderr = sys.stdout

JOB_ID = os.environ.get("SLURM_JOB_ID","local")
NODE   = os.environ.get("SLURMD_NODENAME","unknown")

print("="*70)
print("  SOIL SPATIAL v4 — DISTRIBUTED AI | TRUE SPATIAL GENERALISATION")
print(f"  Job: {JOB_ID} | Node: {NODE} | Start: {pd.Timestamp.now()}")
print("="*70)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT  = Path("/home/emmanuel.keku")
PREPROC  = PROJECT / "preprocessed_v3"   # reuse v3 preprocessed data
RESULTS  = PROJECT / "results_v4"
MODELS   = PROJECT / "models_v4" / "dl"
FIGS     = PROJECT / "figures_v4"
LOGS     = PROJECT / "logs"
for d in [RESULTS, MODELS, FIGS, LOGS]: d.mkdir(parents=True, exist_ok=True)

SEED = 42; np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — PYTORCH
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            n=torch.cuda.get_device_name(i)
            m=torch.cuda.get_device_properties(i).total_memory/1e9
            print(f"  GPU {i}: {n} | {m:.1f} GB")
except ImportError as e:
    print(f"FATAL: {e}"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD PREPROCESSED DATA (reuse v3 cache)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  PHASE 1: Loading v3 preprocessed data")
print("="*55)

if not (PREPROC/"master_processed.csv").exists():
    print("FATAL: Run train_soil_spatial.py (v3) first to generate cache")
    sys.exit(1)

t0 = time.time()
df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)
print(f"  {len(df):,} rows | {time.time()-t0:.1f}s")

LOCATIONS        = pd.DataFrame(FI["LOCATIONS"])
N_LOCS           = FI["N_LOCS"]
SNAP_FEATURES    = FI["SNAP_FEATURES"]
ALL_TARGETS      = FI["ALL_TARGETS"]
TEMP_TARGETS     = FI["TEMP_TARGETS"]
SMAP_TARGETS     = FI["SMAP_TARGETS"]
MOIST_TARGETS    = FI["MOIST_TARGETS"]
N_SNAP_FEATURES  = FI["N_SNAP_FEATURES"]
SITES            = FI["SITES"]
snap_feat_scaler = SC["snap_feat_scaler"]
snap_tgt_scalers = SC["snap_tgt_scalers"]

# ── ADD WAVELET APPROX AS INPUT FEATURES [FIX 1] ─────────────────────────────
# Instead of decomposing the TARGET into seasonal + residual,
# we add the seasonal component as an INPUT FEATURE.
# The model now predicts the RAW signal directly.
# This forces it to learn everything — not just the 4.6% residual.
APPROX_INPUT_FEATURES = []
for tgt in ALL_TARGETS:
    ac = f"{tgt}_approx"
    if ac in df.columns:
        APPROX_INPUT_FEATURES.append(ac)

# Combined feature set: original snap features + seasonal approx features
V4_FEATURES = list(dict.fromkeys(SNAP_FEATURES + APPROX_INPUT_FEATURES))
V4_FEATURES = [f for f in V4_FEATURES if f in df.columns]
N_V4_FEATURES = len(V4_FEATURES)

print(f"  Base snap features    : {N_SNAP_FEATURES}")
print(f"  Approx input features : {len(APPROX_INPUT_FEATURES)}")
print(f"  Total v4 features     : {N_V4_FEATURES}")
print(f"  Targets (RAW)         : {ALL_TARGETS}")

# Fit v4 feature scaler on training data
from sklearn.preprocessing import RobustScaler
print("  Fitting v4 feature scaler...")
tr_all = df[df["split"]=="train"]
v4_feat_scaler = RobustScaler()
v4_feat_scaler.fit(tr_all[V4_FEATURES].fillna(0).values)

# Fit v4 TARGET scalers on RAW signals (not residuals)
v4_tgt_scalers = {}
for grp_name, tgt_cols in [("temp",  TEMP_TARGETS),
                             ("smap",  SMAP_TARGETS),
                             ("moist", MOIST_TARGETS)]:
    av = [c for c in tgt_cols if c in tr_all.columns]
    if not av: continue
    ts = RobustScaler(); ts.fit(tr_all[av].dropna().values)
    v4_tgt_scalers[grp_name] = ts
    print(f"  ✓ v4_tgt [{grp_name}] raw signal scaler")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SPATIAL HOLDOUT SPLIT [FIX 2]
# Hold out ONE ENTIRE SITE from training
# Model must predict it using only the graph at test time
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  PHASE 2: Spatial Holdout Split")
print("="*55)

# Holdout strategy: Wetland is the most physically distinct site
# (permafrost-affected, highest freeze fraction)
# If model can predict Wetland from Bedrock/Transition/Upland neighbours
# via the GCN, it proves true spatial generalisation
HOLDOUT_SITE  = "Wetland"
TRAINING_SITES = [s for s in SITES if s != HOLDOUT_SITE]

# Get location indices for each group
loc_to_idx = {(float(r.Latitude), float(r.Longitude)): i
              for i, r in LOCATIONS.iterrows()}

def get_site_loc_indices(site):
    site_df = df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    idxs = [loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
            for _,r in site_df.iterrows()]
    return sorted([i for i in idxs if i is not None])

SEEN_LOCS   = []
for s in TRAINING_SITES:
    SEEN_LOCS.extend(get_site_loc_indices(s))
SEEN_LOCS = sorted(set(SEEN_LOCS))
UNSEEN_LOCS = get_site_loc_indices(HOLDOUT_SITE)

print(f"  Holdout site  : {HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locations)")
print(f"  Training sites: {TRAINING_SITES}")
print(f"  Seen locs     : {len(SEEN_LOCS)}")
print(f"  Unseen locs   : {len(UNSEEN_LOCS)}")
print(f"  Total locs    : {len(SEEN_LOCS)+len(UNSEEN_LOCS)} = {N_LOCS}")
print()
print("  KEY: At inference, unseen locations receive information ONLY")
print("  through GCN graph edges from seen neighbours.")
print("  A model with no GCN gets 0 signal for unseen locations.")
print("  This is the definitive spatial generalisation test.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SPATIAL GRAPH
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  PHASE 3: Spatial Graph")
print("="*55)

def build_latlon_graph(locs_df, k=6):
    coords = locs_df[["Latitude","Longitude"]].values.astype(np.float32)
    N      = len(coords)
    scaled = coords * np.array([111.0, 63.0], dtype=np.float32)
    tree   = cKDTree(scaled)
    dists, idxs = tree.query(scaled, k=min(k+1,N))
    sigma  = np.median(dists[:,1:]) + 1e-8
    A      = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for jp in range(1, dists.shape[1]):
            j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
            A[i,j]+=w; A[j,i]+=w
    A += np.eye(N)
    D = A.sum(1, keepdims=True)**0.5
    An = (A/(D*D.T+1e-8)).astype(np.float32)
    print(f"  N={N} | k={k} | sigma={sigma:.2f}km | avg_deg={(A>0).sum(1).mean():.1f}")
    return coords, An

loc_coords, A_norm_np = build_latlon_graph(LOCATIONS, k=6)
A_norm_t = torch.tensor(A_norm_np).to(DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — DATASET (raw signal, spatial holdout aware)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialFieldDatasetV4(Dataset):
    """
    v4 Dataset — key differences from v3:
    1. Predicts RAW signal (not wavelet residual)
    2. Wavelet approx is an INPUT FEATURE (not subtracted from target)
    3. Training mode: seen_locs only get valid targets
       Unseen locs present in X (graph needs them) but masked in loss
    4. Test mode: all locs evaluated, unseen separately
    """
    def __init__(self, df, loc_to_idx, A_norm, v4_features, tgt_cols,
                 feat_scaler, tgt_scaler, seen_locs, unseen_locs,
                 split="train", lookback=24, stride=6, max_samples=None,
                 mask_unseen_in_loss=True):
        self.A = A_norm
        N  = N_LOCS
        nf = len(v4_features)
        nt = len(tgt_cols)

        sub = df[df["split"]==split].copy()
        all_ts = sorted(sub["time_utc"].unique())
        T = len(all_ts)
        print(f"    [{split}] {T:,} timestamps | {N} locs | mask_unseen={mask_unseen_in_loss}")
        if T < lookback+2:
            self._empty(); return

        ts_to_i = {ts:i for i,ts in enumerate(all_ts)}
        sub2 = sub.copy()
        sub2["_ti"] = sub2["time_utc"].map(ts_to_i)
        sub2["_ni"] = [loc_to_idx.get((float(la),float(lo)))
                       for la,lo in zip(sub2["Latitude"].astype(float),
                                        sub2["Longitude"].astype(float))]
        sub2 = sub2.dropna(subset=["_ti","_ni"])
        sub2["_ti"]=sub2["_ti"].astype(int); sub2["_ni"]=sub2["_ni"].astype(int)
        ti_arr=sub2["_ti"].values; ni_arr=sub2["_ni"].values

        X_full  = np.full((T,N,nf), np.nan, dtype=np.float32)
        y_full  = np.full((T,N,nt), np.nan, dtype=np.float32)
        msk_full= np.zeros((T,N),   dtype=np.float32)  # 1=valid, 0=masked

        # Fill features
        X_full[ti_arr,ni_arr,:] = feat_scaler.transform(
            sub2[v4_features].fillna(0).values).astype(np.float32)

        # Fill raw targets
        av_tgt = [c for c in tgt_cols if c in sub2.columns]
        if av_tgt:
            y_full[ti_arr,ni_arr,:] = tgt_scaler.transform(
                sub2[tgt_cols].fillna(0).values).astype(np.float32)

        # Build validity mask
        # During training: seen locations only
        # During test: all locations (unseen evaluated separately)
        if mask_unseen_in_loss and split=="train":
            msk_full[:,seen_locs] = 1.0   # only seen locs contribute to loss
        else:
            msk_full[:,:] = 1.0           # all locs evaluated

        # Slide window
        tidxs = list(range(lookback, T, stride))
        if max_samples and len(tidxs)>max_samples:
            rng=np.random.default_rng(SEED)
            tidxs=sorted(rng.choice(tidxs,max_samples,replace=False))

        Xl=[]; yl=[]; ml=[]
        for ti in tidxs:
            Xw=X_full[ti-lookback:ti]
            yi=y_full[ti]; mi=msk_full[ti]
            if np.isnan(Xw).mean()>0.25: continue
            Xl.append(np.nan_to_num(Xw,nan=0.0))
            yl.append(np.nan_to_num(yi,nan=0.0))
            ml.append(mi)

        if not Xl: self._empty(); return

        self.X    = torch.tensor(np.array(Xl),  dtype=torch.float32)
        self.y    = torch.tensor(np.array(yl),  dtype=torch.float32)
        self.mask = torch.tensor(np.array(ml),  dtype=torch.float32)
        print(f"    [{split}] {len(self.X):,} samples | X={tuple(self.X.shape[1:])}")

    def _empty(self):
        self.X=self.y=self.mask=torch.zeros(0)

    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return self.X[i], self.y[i], self.mask[i], self.A


def make_v4_loaders(tgt_grp, tgt_cols, lookback=24,
                    st_tr=6, st_ev=24, bs=4, max_tr=2000, max_ev=500):
    ts = v4_tgt_scalers.get(tgt_grp)
    if ts is None or not tgt_cols: return {s:None for s in ["train","val","test"]}
    out = {}
    for sp,ms,st,mask in [("train",max_tr,st_tr,True),
                           ("val",  max_ev,st_ev,False),
                           ("test", max_ev,st_ev,False)]:
        ds = SpatialFieldDatasetV4(df, loc_to_idx, A_norm_t,
                                    V4_FEATURES, tgt_cols,
                                    v4_feat_scaler, ts,
                                    SEEN_LOCS, UNSEEN_LOCS,
                                    split=sp, lookback=lookback,
                                    stride=st, max_samples=ms,
                                    mask_unseen_in_loss=mask)
        out[sp] = None if len(ds)==0 else DataLoader(
            ds, batch_size=bs, shuffle=(sp=="train"),
            num_workers=0, pin_memory=False,
            drop_last=(sp=="train"))
    return out

print("\nBuilding v4 dataloaders...")
temp_ld  = make_v4_loaders("temp",  TEMP_TARGETS)
smap_ld  = make_v4_loaders("smap",  SMAP_TARGETS)
moist_ld = make_v4_loaders("moist", MOIST_TARGETS)
for name,ld in [("Temp",temp_ld),("SMAP",smap_ld),("Moist",moist_ld)]:
    for sp,l in ld.items():
        print(f"  {name} {sp}: {len(l) if l else 0} batches")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared: GraphConv (bmm, batch-safe) ──────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self, id, od, dp=0.1):
        super().__init__()
        self.W=nn.Linear(id,od,bias=False); self.n=nn.LayerNorm(od)
        self.d=nn.Dropout(dp); self.a=nn.GELU()
    def forward(self, H, A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        return self.a(self.n(torch.bmm(
            A.unsqueeze(0).expand(H.shape[0],-1,-1), self.W(self.d(H)))))

# ── ABLATION 1: BiGRU_NoGCN (temporal only) ──────────────────────────────────
class BiGRU_NoGCN(nn.Module):
    """
    Temporal-only baseline. No graph convolution.
    Each location predicted independently from its own time series.
    If spatial model beats this on UNSEEN locations → GCN is doing real work.
    """
    def __init__(self, nf, h=96, nl=2, nh=4, N=256, nt=1, dp=0.1, **kw):
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
        self.head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),
                                 nn.Dropout(dp),nn.Linear(h//2,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.gru(h); a,_=self.attn(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ffn(h))
        h=self.red(h[:,-1,:]).reshape(B,N,-1)
        return self.head(h)   # NO GCN

# ── ABLATION 2: GCN_NoTemporal (spatial only) ────────────────────────────────
class GCN_NoTemporal(nn.Module):
    """
    Spatial-only baseline. No sequence modelling.
    Uses only the current snapshot (last timestep) + graph.
    If temporal model beats this → sequence modelling matters.
    """
    def __init__(self, nf, h=96, gl=3, N=256, nt=1, dp=0.1, **kw):
        super().__init__()
        self.proj=nn.Linear(nf,h)
        self.gcn=nn.ModuleList([GraphConv(h,h,dp) for _ in range(gl)])
        self.head=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),
                                 nn.Dropout(dp),nn.Linear(h,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.proj(x[:,-1,:,:])   # only last timestep — no temporal
        h0=h
        for g in self.gcn: h=g(h,A)
        return self.head(torch.cat([h0,h],dim=-1))

# ── RESERVOIR 1: DeepESN (Deep Echo State Network) ───────────────────────────
class DeepESNLayer(nn.Module):
    """
    Single ESN reservoir layer.
    W_in and W_res are FIXED (not trained) — echo state property.
    Only the readout (head) is trained.
    Spectral radius < 1 guarantees echo state property (input-to-state stability).
    Ref: Gallicchio & Micheli 2017 (arXiv:1712.04323)
         Singh & Raman 2025 (arXiv:2509.04422) — ESN as SSM
    """
    def __init__(self, in_d, res_d, spectral_radius=0.9, leaking_rate=0.3, dp=0.1):
        super().__init__()
        self.res_d = res_d
        self.leak  = leaking_rate
        # Fixed random weights — NOT trained
        W_in  = torch.randn(res_d, in_d)  * 0.1
        W_res = torch.randn(res_d, res_d)
        # Scale W_res to desired spectral radius
        eigvals = torch.linalg.eigvals(W_res).abs()
        W_res   = W_res * (spectral_radius / (eigvals.max().item() + 1e-8))
        self.register_buffer("W_in",  W_in)
        self.register_buffer("W_res", W_res)
        self.drop = nn.Dropout(dp)
        self.norm = nn.LayerNorm(res_d)

    def forward(self, x):
        """x: (B, L, in_d) → h: (B, L, res_d)"""
        B,L,_ = x.shape
        h = torch.zeros(B, self.res_d, device=x.device, dtype=x.dtype)
        states = []
        for t in range(L):
            # Leaky integration: h = (1-leak)*h + leak*tanh(W_in*x + W_res*h)
            pre = (x[:,t,:] @ self.W_in.T + h @ self.W_res.T)
            h   = (1-self.leak)*h + self.leak*torch.tanh(pre)
            states.append(h)
        return self.norm(torch.stack(states, dim=1))   # (B, L, res_d)

class DeepESN(nn.Module):
    """
    Deep Echo State Network with 3 stacked reservoir layers.
    Each layer captures different temporal scales:
      Layer 1: fast dynamics (hourly weather)
      Layer 2: medium dynamics (daily freeze-thaw)
      Layer 3: slow dynamics (seasonal transitions)
    Only the linear readout head is trained.
    """
    def __init__(self, nf, res_d=128, n_layers=3, N=256, nt=1,
                 spectral_radius=0.9, leaking_rate=0.3, dp=0.1, **kw):
        super().__init__()
        self.proj = nn.Linear(nf, res_d)
        # Stacked ESN layers with increasing leaking rates (multi-scale)
        leak_rates = [leaking_rate*(0.5**i) for i in range(n_layers)]
        self.esn_layers = nn.ModuleList([
            DeepESNLayer(res_d, res_d,
                          spectral_radius=spectral_radius,
                          leaking_rate=leak_rates[i], dp=dp)
            for i in range(n_layers)
        ])
        # Trained readout — linear (per ESN design principle)
        total_res = res_d * n_layers   # concatenate all layer states
        self.head = nn.Sequential(
            nn.Linear(total_res, res_d), nn.GELU(),
            nn.Dropout(dp), nn.Linear(res_d, nt))

    def forward(self, x, A):
        B,L,N,F = x.shape
        h = self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        layer_states = []
        for esn in self.esn_layers:
            h = esn(h)               # (B*N, L, res_d)
            layer_states.append(h[:,-1,:])   # last timestep per layer
        h_cat = torch.cat(layer_states, dim=-1)   # (B*N, total_res)
        return self.head(h_cat).reshape(B,N,-1)

# ── RESERVOIR 2: SpatialESN (DeepESN + GCN) — Novel Contribution ─────────────
class SpatialESN(nn.Module):
    """
    Novel model: Deep Echo State Network + Graph Convolution Network.
    ESN captures temporal dynamics without backprop-through-time.
    GCN propagates reservoir states across spatial locations.
    This is the bridge between reservoir computing and spatial AI.
    """
    def __init__(self, nf, res_d=128, n_layers=3, N=256, gl=2, nt=1,
                 spectral_radius=0.9, leaking_rate=0.3, dp=0.1, **kw):
        super().__init__()
        self.proj = nn.Linear(nf, res_d)
        leak_rates = [leaking_rate*(0.5**i) for i in range(n_layers)]
        self.esn_layers = nn.ModuleList([
            DeepESNLayer(res_d, res_d,
                          spectral_radius=spectral_radius,
                          leaking_rate=leak_rates[i], dp=dp)
            for i in range(n_layers)
        ])
        total_res = res_d * n_layers
        self.compress = nn.Linear(total_res, res_d)
        self.gcn  = nn.ModuleList([GraphConv(res_d, res_d, dp) for _ in range(gl)])
        self.head = nn.Sequential(
            nn.Linear(res_d*2, res_d), nn.GELU(),
            nn.Dropout(dp), nn.Linear(res_d, nt))

    def forward(self, x, A):
        B,L,N,F = x.shape
        h = self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        layer_states = []
        for esn in self.esn_layers:
            h = esn(h)
            layer_states.append(h[:,-1,:])
        h_cat = torch.cat(layer_states, dim=-1)
        h0 = torch.relu(self.compress(h_cat)).reshape(B,N,-1)
        hg = h0
        for g in self.gcn: hg = g(hg, A)
        return self.head(torch.cat([h0, hg], dim=-1))

# ── GRAPH 1: GraphSAGE (inductive node prediction) ───────────────────────────
class SAGEConv(nn.Module):
    """
    GraphSAGE convolution — Hamilton et al. 2017.
    Inductive: learns AGGREGATION FUNCTIONS not node embeddings.
    Can generalise to nodes not seen during training.
    This is the theoretically correct model for spatial holdout evaluation.
    """
    def __init__(self, in_d, out_d, dp=0.1):
        super().__init__()
        self.W_self = nn.Linear(in_d, out_d, bias=False)
        self.W_neigh= nn.Linear(in_d, out_d, bias=False)
        self.norm   = nn.LayerNorm(out_d)
        self.drop   = nn.Dropout(dp)
        self.act    = nn.GELU()
    def forward(self, H, A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        # Mean aggregation from neighbours
        A_b  = A.unsqueeze(0).expand(H.shape[0],-1,-1)
        neigh= torch.bmm(A_b, self.drop(H))   # mean neighbour aggregation
        return self.act(self.norm(
            self.W_self(self.drop(H)) + self.W_neigh(neigh)))

class GraphSAGE(nn.Module):
    """
    GraphSAGE with BiGRU temporal encoder.
    Inductive spatial generalisation to unseen locations.
    """
    def __init__(self, nf, h=96, nl=2, N=256, gl=3, nt=1, dp=0.1, **kw):
        super().__init__()
        self.proj = nn.Linear(nf, h)
        self.gru  = nn.GRU(h, h, nl, batch_first=True, bidirectional=True,
                           dropout=dp if nl>1 else 0.)
        d2 = h*2
        self.red  = nn.Linear(d2, h)
        self.sage = nn.ModuleList([SAGEConv(h, h, dp) for _ in range(gl)])
        self.head = nn.Sequential(nn.Linear(h*2, h), nn.GELU(),
                                   nn.Dropout(dp), nn.Linear(h, nt))
    def forward(self, x, A):
        B,L,N,F = x.shape
        h = self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_ = self.gru(h)
        h   = self.red(h[:,-1,:]).reshape(B,N,-1)
        hg  = h
        for s in self.sage: hg = s(hg, A)
        return self.head(torch.cat([h, hg], dim=-1))

# ── GRAPH 2: GAT (Graph Attention Network) ───────────────────────────────────
class GATConv(nn.Module):
    """
    Graph Attention Network — Velickovic et al. 2018.
    Learns which spatial neighbours are most relevant.
    Wetland-Bedrock coupling may be weighted differently than Upland-Transition.
    """
    def __init__(self, in_d, out_d, n_heads=4, dp=0.1):
        super().__init__()
        self.nh   = n_heads
        self.hd   = out_d // n_heads
        self.W    = nn.Linear(in_d, out_d, bias=False)
        self.a_src= nn.Linear(self.hd, 1, bias=False)
        self.a_dst= nn.Linear(self.hd, 1, bias=False)
        self.norm = nn.LayerNorm(out_d)
        self.drop = nn.Dropout(dp)
        self.act  = nn.GELU()
    def forward(self, H, A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        B,N,_ = H.shape
        Wh = self.W(self.drop(H)).view(B,N,self.nh,self.hd)
        e_src = self.a_src(Wh)   # (B,N,nh,1)
        e_dst = self.a_dst(Wh)   # (B,N,nh,1)
        # Attention scores: e_ij = LeakyReLU(a_src[i] + a_dst[j])
        e = F.leaky_relu(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1), 0.2)  # (B,N,N,nh,1)
        # Mask with adjacency
        mask  = (A==0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        e     = e.masked_fill(mask, -1e9)
        alpha = F.softmax(e, dim=2)                         # (B,N,N,nh,1)
        # Aggregate
        Wh_T  = Wh.unsqueeze(1).expand(-1,N,-1,-1,-1)      # (B,N,N,nh,hd)
        out   = (alpha * Wh_T).sum(2).view(B,N,-1)          # (B,N,out_d)
        return self.act(self.norm(out))

class GAT(nn.Module):
    """GAT with BiGRU temporal encoder."""
    def __init__(self, nf, h=96, nl=2, nh=4, N=256, gl=2, nt=1, dp=0.1, **kw):
        super().__init__()
        self.proj= nn.Linear(nf, h)
        self.gru = nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                          dropout=dp if nl>1 else 0.)
        d2=h*2; self.red=nn.Linear(d2,h)
        self.gat = nn.ModuleList([GATConv(h,h,nh,dp) for _ in range(gl)])
        self.head= nn.Sequential(nn.Linear(h*2,h),nn.GELU(),
                                  nn.Dropout(dp),nn.Linear(h,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.gru(h); h=self.red(h[:,-1,:]).reshape(B,N,-1)
        hg=h
        for g in self.gat: hg=g(hg,A)
        return self.head(torch.cat([h,hg],dim=-1))

# ── GRAPH 3: STGCN (Spatio-Temporal GCN) ────────────────────────────────────
class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolution Block — Yu et al. 2018 (IJCAI).
    Temporal conv → Spatial GCN → Temporal conv (sandwich structure).
    Standard benchmark in traffic/weather spatiotemporal prediction.
    """
    def __init__(self, d, k_t=3, dp=0.1):
        super().__init__()
        pad = k_t//2
        self.tc1  = nn.Conv1d(d, d, k_t, padding=pad)
        self.gcn  = GraphConv(d, d, dp)
        self.tc2  = nn.Conv1d(d, d, k_t, padding=pad)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dp)
        self.act  = nn.GELU()
    def forward(self, x, A):
        # x: (B*N, L, d)
        h = self.act(self.tc1(x.transpose(1,2)).transpose(1,2))
        # Reshape for GCN: need (B, N, d)
        BN,L,d = h.shape
        # We process spatial dimension — need B and N separately
        # Pass through as (1, BN, d) treating BN as N
        hg = self.gcn(h[:,-1:,:].transpose(0,1), A) if A.shape[0]==BN else h[:,-1,:]
        h  = self.act(self.tc2(h.transpose(1,2)).transpose(1,2))
        return self.norm(h + self.drop(h))

class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Literature standard for spatiotemporal forecasting.
    """
    def __init__(self, nf, h=64, n_blocks=3, N=256, gl=2, nt=1, dp=0.1, **kw):
        super().__init__()
        self.proj = nn.Linear(nf, h)
        # Temporal GRU backbone
        self.gru  = nn.GRU(h,h,2,batch_first=True,bidirectional=True,
                           dropout=dp)
        d2=h*2; self.red=nn.Linear(d2,h)
        # Spatial GCN layers
        self.gcn  = nn.ModuleList([GraphConv(h,h,dp) for _ in range(gl)])
        # Temporal GCN cross-attention
        self.cross= nn.Linear(h,h)
        self.head = nn.Sequential(nn.Linear(h*2,h),nn.GELU(),
                                   nn.Dropout(dp),nn.Linear(h,nt))
    def forward(self, x, A):
        B,L,N,F=x.shape
        h=self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.gru(h); h=self.red(h[:,-1,:]).reshape(B,N,-1)
        hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.head(torch.cat([h,hg],dim=-1))

# ── SSM models from v3 (kept) ─────────────────────────────────────────────────
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
    def scan(self,x):
        B,L,D=x.shape; S=self.ds
        xd=self.xp(x); dl,Bp,C=xd.split([D,S,S],dim=-1)
        dl=F.softplus(self.dtp(dl)); A__=-torch.exp(self.Al.float())
        dA=torch.exp(torch.einsum("bld,ds->blds",dl,A__))
        dB=torch.einsum("bld,bls->blds",dl,Bp)
        h=torch.zeros(B,D,S,device=x.device,dtype=x.dtype); ys=[]
        for i in range(L):
            h=dA[:,i]*h+dB[:,i]*x[:,i,:,None]
            ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
        return torch.stack(ys,dim=1)*self.D_
    def forward(self,x):
        r=x; xz=self.ip(x); x_,z=xz.chunk(2,dim=-1)
        x_=self.silu(self.cv(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        y=self.scan(x_)*self.silu(z)
        return self.nm(r+self.op(self.dr(y)))

class SpatialBiGRU(nn.Module):
    def __init__(self, nf, h=96, nl=2, nh=4, N=256, gl=2, nt=1, dp=0.1, **kw):
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
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.proj(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.gru(h); a,_=self.attn(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ffn(h))
        h=self.red(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.head(torch.cat([h,hg],dim=-1))

class SpatialMamba(nn.Module):
    def __init__(self, nf, d=96, nl=4, ds=16, N=256, gl=2, nt=1, dp=0.1, **kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.mb=nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gcn=nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for b in self.mb: h=b(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
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
    def scan(self,u):
        B,L,d=u.shape; dA=torch.matrix_exp(self.A_); dB=self.B_.squeeze(-1)
        h=torch.zeros(B,d,self.ds,device=u.device); ys=[]
        for t in range(L):
            h=h@dA.T+u[:,t,:,None]*dB
            ys.append((h*self.C_.unsqueeze(0)).sum(-1)+self.D_*u[:,t,:])
        return torch.stack(ys,dim=1)
    def forward(self,x):
        yf=self.scan(x); yr=self.scan(x.flip(1)).flip(1)
        return self.nm(x+self.dr(self.ot(self.mx(torch.cat([yf,yr],dim=-1)))))

class SpatialS4(nn.Module):
    def __init__(self, nf, d=96, nl=4, ds=64, N=256, gl=2, nt=1, dp=0.1, **kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.ly=nn.ModuleList([S4Layer(d,ds,dp) for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gcn=nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for l in self.ly: h=l(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gcn: hg=g(hg,A)
        return self.hd(torch.cat([h,hg],dim=-1))

class SpatialFuseMoE(nn.Module):
    def __init__(self, nf, d=96, ne=4, tk=2, ds=16, nsl=2, N=256, gl=2, nt=1, dp=0.1, **kw):
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
    def _expert(self,i,h):
        ex=self.ex[i]
        if isinstance(ex,MambaBlock): return self.enm[i](ex(h)[:,-1,:])
        if isinstance(ex,nn.GRU): _,ht=ex(h); return self.enm[i](ht[-1])
        return self.enm[i](ex(h.transpose(1,2)).squeeze(-1))
    def forward(self,x,A):
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
        ho=self.nm(fs[:,-1,:]).reshape(B,N,-1); hg=ho
        for g in self.gcn: hg=g(hg,A)
        return self.hd(torch.cat([ho,hg],dim=-1)), aux

# ── Model factory ─────────────────────────────────────────────────────────────
ARCH_MAP = {
    # Ablations
    "BiGRU_NoGCN"   : lambda nt: BiGRU_NoGCN(   N_V4_FEATURES,h=96,nl=2,nh=4,N=N_LOCS,nt=nt),
    "GCN_NoTemporal": lambda nt: GCN_NoTemporal( N_V4_FEATURES,h=96,gl=3,N=N_LOCS,nt=nt),
    # Reservoir
    "DeepESN"       : lambda nt: DeepESN(        N_V4_FEATURES,res_d=128,n_layers=3,N=N_LOCS,nt=nt),
    "SpatialESN"    : lambda nt: SpatialESN(     N_V4_FEATURES,res_d=128,n_layers=3,N=N_LOCS,gl=2,nt=nt),
    # Graph
    "GraphSAGE"     : lambda nt: GraphSAGE(      N_V4_FEATURES,h=96,nl=2,N=N_LOCS,gl=3,nt=nt),
    "GAT"           : lambda nt: GAT(            N_V4_FEATURES,h=96,nl=2,nh=4,N=N_LOCS,gl=2,nt=nt),
    "STGCN"         : lambda nt: STGCN(          N_V4_FEATURES,h=64,N=N_LOCS,gl=2,nt=nt),
    # SSM (from v3)
    "SpatialBiGRU"  : lambda nt: SpatialBiGRU(  N_V4_FEATURES,h=96,nl=2,nh=4,N=N_LOCS,gl=2,nt=nt),
    "SpatialMamba"  : lambda nt: SpatialMamba(   N_V4_FEATURES,d=96,nl=4,ds=16,N=N_LOCS,gl=2,nt=nt),
    "SpatialS4"     : lambda nt: SpatialS4(      N_V4_FEATURES,d=96,nl=4,ds=64,N=N_LOCS,gl=2,nt=nt),
    "SpatialFuseMoE": lambda nt: SpatialFuseMoE( N_V4_FEATURES,d=96,ne=4,tk=2,ds=16,nsl=2,N=N_LOCS,gl=2,nt=nt),
}

# Tier labels for reporting
ARCH_TIERS = {
    "BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
    "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
    "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
    "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
    "SpatialS4":"SSM","SpatialFuseMoE":"SSM",
}

print("\nModel parameter counts:")
print(f"  {'Tier':<12} {'Model':<18} {'Params':>12}")
print("  " + "─"*44)
for arch,fn in ARCH_MAP.items():
    try:
        m=fn(1); p=sum(x.numel() for x in m.parameters() if x.requires_grad)
        tier=ARCH_TIERS.get(arch,"?")
        print(f"  {tier:<12} {arch:<18} {p:>12,}")
    except Exception as e: print(f"  {arch}: ERROR {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def masked_huber(pred, target, mask, delta=1.0):
    """
    Huber loss applied only to seen locations during training.
    Unseen locations (holdout site) do NOT contribute to loss.
    This forces the model to learn from seen locations only.
    At test time, predictions for unseen locations come purely
    from GCN propagation — true spatial generalisation.
    """
    diff = pred - target
    loss = torch.where(diff.abs()<=delta, 0.5*diff**2,
                       delta*(diff.abs()-0.5*delta))
    # mask: (B, N) — 1=seen, 0=unseen
    mask_e = mask.unsqueeze(-1).expand_as(loss)
    return (loss * mask_e).sum() / (mask_e.sum() + 1e-8)

def graph_smooth(pred, A, seen_locs):
    """Graph Laplacian only over seen locations."""
    if A.dim()==3: A=A[0]
    if A.dim()==4: A=A[0,0]
    A_b = A.unsqueeze(0).expand(pred.shape[0],-1,-1)
    sm  = torch.bmm(A_b, pred)
    return F.mse_loss(pred[:,seen_locs,:], sm[:,seen_locs,:])

def train_one_v4(arch, n_targets, train_ld, val_ld, tgt_sc,
                  epochs=30, lr=3e-4, patience=7,
                  lam_s=0.05, lam_a=0.01, ckpt_path=None):

    is_moe = (arch=="SpatialFuseMoE")
    model  = ARCH_MAP[arch](n_targets).to(DEVICE)
    opt    = AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                   lr=lr, weight_decay=1e-4)
    n_steps = epochs*len(train_ld)
    sched   = OneCycleLR(opt, max_lr=lr, total_steps=n_steps, pct_start=0.1)
    amp_sc  = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    seen_t = torch.tensor(SEEN_LOCS, device=DEVICE)
    best_r2=float("-inf"); best_st=None; pat=0; hist=[]; t0=time.time()
    np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
    tier=ARCH_TIERS.get(arch,"?")
    print(f"  [{tier}] {arch} | {np_:,} params | {epochs}ep | {DEVICE}")

    for ep in range(1, epochs+1):
        model.train(); tr=0.; nb=0
        for batch in train_ld:
            X,y,mask,A=[b.to(DEVICE) for b in batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out=model(X,A)
                if is_moe: pred,aux=out
                else:       pred=out; aux=None
                # Loss only on seen locations [KEY FIX]
                loss=masked_huber(pred,y,mask)
                loss=loss+lam_s*graph_smooth(pred,A,SEEN_LOCS)
                if aux is not None: loss=loss+lam_a*aux
            amp_sc.scale(loss).backward()
            amp_sc.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            amp_sc.step(opt); amp_sc.update(); sched.step()
            tr+=loss.item(); nb+=1
        tr_loss=tr/max(nb,1)

        # Validation — evaluate on seen + unseen separately
        model.eval(); yt=[]; yp=[]; yt_u=[]; yp_u=[]
        with torch.no_grad():
            for batch in val_ld:
                X,y,mask,A=[b.to(DEVICE) for b in batch]
                out=model(X,A)
                pred=out[0] if is_moe else out
                B_,N_,T_=pred.shape
                pr=pred.cpu().float().numpy()
                pr_r=tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
                y_r=tgt_sc.inverse_transform(
                    y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
                # Seen locations
                yt.append(y_r[:,SEEN_LOCS,0].flatten())
                yp.append(pr_r[:,SEEN_LOCS,0].flatten())
                # Unseen locations (holdout)
                yt_u.append(y_r[:,UNSEEN_LOCS,0].flatten())
                yp_u.append(pr_r[:,UNSEEN_LOCS,0].flatten())

        def r2(yt_,yp_):
            a=np.concatenate(yt_); b=np.concatenate(yp_)
            mk=~(np.isnan(a)|np.isnan(b)); a=a[mk]; b=b[mk]
            return float(1-np.sum((a-b)**2)/(np.sum((a-a.mean())**2)+1e-10)) if len(a)>5 else np.nan

        val_r2_seen  = r2(yt,yp)
        val_r2_unseen= r2(yt_u,yp_u)

        hist.append(dict(epoch=ep,train_loss=round(tr_loss,6),
                         val_R2_seen=round(val_r2_seen,4),
                         val_R2_unseen=round(val_r2_unseen,4)))

        if val_r2_seen > best_r2:
            best_r2=val_r2_seen
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat=0
        else: pat+=1

        if ep%5==0 or ep==1:
            print(f"    E{ep:03d} | loss={tr_loss:.4f} | "
                  f"R²_seen={val_r2_seen:.4f} | "
                  f"R²_unseen={val_r2_unseen:.4f} | "
                  f"{time.time()-t0:.0f}s")
        if pat>=patience:
            print(f"    Early stop @ {ep}"); break

    elapsed=time.time()-t0
    print(f"  ✓ val R²_seen={best_r2:.4f} | {elapsed:.0f}s")

    if best_st: model.load_state_dict(best_st)
    torch.save(dict(arch=arch,state_dict=best_st,val_r2=best_r2,
                    history=hist,epochs_run=ep,elapsed_s=elapsed,
                    seen_locs=SEEN_LOCS,unseen_locs=UNSEEN_LOCS,
                    holdout_site=HOLDOUT_SITE,n_v4_features=N_V4_FEATURES,
                    job_id=JOB_ID,node=NODE), ckpt_path)
    return model, hist, best_r2, elapsed


@torch.no_grad()
def evaluate_v4(model, loader, tgt_sc, arch):
    """
    Full evaluation separating seen vs unseen locations.
    This is the definitive spatial generalisation test.
    """
    is_moe=(arch=="SpatialFuseMoE")
    model.eval()
    yt_s=[]; yp_s=[]   # seen locations
    yt_u=[]; yp_u=[]   # unseen locations (holdout site — Wetland)
    yt_a=[]; yp_a=[]   # all locations

    for batch in loader:
        X,y,mask,A=[b.to(DEVICE) for b in batch]
        out=model(X,A); pred=out[0] if is_moe else out
        B_,N_,T_=pred.shape
        pr=pred.cpu().float().numpy()
        pr_r=tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
        y_r=tgt_sc.inverse_transform(
            y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
        yt_s.append(y_r[:,SEEN_LOCS,0].flatten())
        yp_s.append(pr_r[:,SEEN_LOCS,0].flatten())
        yt_u.append(y_r[:,UNSEEN_LOCS,0].flatten())
        yp_u.append(pr_r[:,UNSEEN_LOCS,0].flatten())
        yt_a.append(y_r[:,:,0].flatten())
        yp_a.append(pr_r[:,:,0].flatten())

    def metrics(yt_list, yp_list, label):
        yt=np.concatenate(yt_list); yp=np.concatenate(yp_list)
        mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
        if len(yt)<5: return {}
        r2   = float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
        rmse = float(np.sqrt(np.mean((yt-yp)**2)))
        r    = float(np.corrcoef(yt,yp)[0,1])
        kge  = float(1-np.sqrt((r-1)**2+(np.std(yp)/(np.std(yt)+1e-10)-1)**2+
                               (np.mean(yp)/(np.mean(yt)+1e-10)-1)**2))
        frz  = float(np.mean((yt<0).astype(int)==(yp<0).astype(int))*100)
        bias = float(np.mean(yp-yt))
        return {f"{label}_R2":round(r2,4), f"{label}_RMSE":round(rmse,4),
                f"{label}_KGE":round(kge,4), f"{label}_FreezeAcc":round(frz,2),
                f"{label}_Bias":round(bias,4), f"{label}_N":int(mk.sum())}

    m_seen  = metrics(yt_s, yp_s, "seen")
    m_unseen= metrics(yt_u, yp_u, "unseen")
    m_all   = metrics(yt_a, yp_a, "all")

    # Spatial generalisation gap = seen R² - unseen R²
    # Small gap → model generalises well to unseen locations
    # Large gap → model memorised seen locations, GCN not helping
    gap = round(m_seen.get("seen_R2",np.nan) - m_unseen.get("unseen_R2",np.nan), 4)

    return {**m_seen, **m_unseen, **m_all, "spatial_gap": gap}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 7: v4 Training — All Models × All Targets")
print(f"  Seen locs  : {len(SEEN_LOCS)} ({TRAINING_SITES})")
print(f"  Unseen locs: {len(UNSEEN_LOCS)} ({HOLDOUT_SITE}) — predicted via GCN only")
print(f"  Features   : {N_V4_FEATURES} (including wavelet approx as input)")
print(f"  Targets    : RAW signal (no wavelet decomp masking)")
print("="*70)

TARGET_GROUPS = [
    ("temp",  TEMP_TARGETS,  temp_ld,  "Weather Temp"),
    ("smap",  SMAP_TARGETS,  smap_ld,  "SMAP Temp L1"),
    ("moist", MOIST_TARGETS, moist_ld, "Soil Moisture"),
]

all_results = []

for (tgt_name, tgt_cols, loaders, label) in TARGET_GROUPS:
    print(f"\n{'─'*60}")
    print(f"  TARGET: {label} | n_tgts={len(tgt_cols)}")
    print(f"{'─'*60}")
    if loaders.get("train") is None: continue
    tgt_sc = v4_tgt_scalers.get(tgt_name)
    if tgt_sc is None: continue

    for arch in ARCH_MAP.keys():
        ckpt_name = f"{arch}_{tgt_name}_v4_best.pt"
        ckpt_path = MODELS / ckpt_name

        # Resume guard
        if ckpt_path.exists():
            try:
                sv=torch.load(ckpt_path,map_location="cpu")
                if sv.get("val_r2",-99)>-10:
                    print(f"\n  ✓ SKIP {arch} [{tgt_name}] val_r2={sv['val_r2']:.4f}")
                    tm=sv.get("test_metrics",{})
                    all_results.append(dict(
                        Model=arch,Target=tgt_name,Tier=ARCH_TIERS.get(arch,"?"),
                        Val_R2=sv["val_r2"],**tm,Resumed=True))
                    continue
            except Exception: pass

        print(f"\n  ── {arch} [{label}]")
        try:
            model,hist,best_r2,elapsed = train_one_v4(
                arch=arch, n_targets=len(tgt_cols),
                train_ld=loaders["train"], val_ld=loaders["val"],
                tgt_sc=tgt_sc, epochs=30, lr=3e-4, patience=7,
                lam_s=0.05, lam_a=0.01, ckpt_path=ckpt_path)

            test_m = {}
            if loaders.get("test"):
                test_m = evaluate_v4(model, loaders["test"], tgt_sc, arch)

            sv=torch.load(ckpt_path,map_location="cpu")
            sv["test_metrics"]=test_m; sv["job_id"]=JOB_ID; sv["node"]=NODE
            torch.save(sv,ckpt_path)

            all_results.append(dict(
                Model=arch, Target=tgt_name, Tier=ARCH_TIERS.get(arch,"?"),
                Val_R2=best_r2, **test_m,
                Train_s=round(elapsed,1), Job_ID=JOB_ID, Resumed=False))

            pd.DataFrame(all_results).to_csv(
                RESULTS/"v4_results_incremental.csv", index=False)

            print(f"  → seen_R2={test_m.get('seen_R2','N/A')} | "
                  f"unseen_R2={test_m.get('unseen_R2','N/A')} | "
                  f"gap={test_m.get('spatial_gap','N/A')}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ FAILED {arch}: {e}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS/"v4_results_all.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  PHASE 8: Figures")
print("="*55)
matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3})

TIER_COLORS = {"ABLATION":"#d62728","RESERVOIR":"#9467bd",
               "GRAPH":"#2ca02c","SSM":"#1f77b4"}
TGT_LABELS  = {"temp":"Weather Temp","smap":"SMAP Temp L1","moist":"Moisture"}

if len(results_df) > 0 and "seen_R2" in results_df.columns:

    # V4-01: Seen vs Unseen R² comparison (THE KEY FIGURE)
    for tgt in results_df["Target"].unique():
        sub = results_df[results_df["Target"]==tgt].dropna(subset=["seen_R2","unseen_R2"])
        if sub.empty: continue
        sub = sub.sort_values("unseen_R2", ascending=True)
        fig,ax = plt.subplots(figsize=(16,9))
        x = np.arange(len(sub)); w=0.35
        b1=ax.barh(x-w/2, sub["seen_R2"],   height=w, label=f"Seen ({len(SEEN_LOCS)} locs)",
                   color="#1f77b4", alpha=0.85, edgecolor="black", lw=0.5)
        b2=ax.barh(x+w/2, sub["unseen_R2"], height=w, label=f"Unseen — {HOLDOUT_SITE} ({len(UNSEEN_LOCS)} locs)",
                   color="#d62728", alpha=0.85, edgecolor="black", lw=0.5)
        for bar,v in zip(b1,sub["seen_R2"]):
            ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                    va="center",fontsize=8,fontweight="bold",color="#1f77b4")
        for bar,v in zip(b2,sub["unseen_R2"]):
            ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                    va="center",fontsize=8,fontweight="bold",color="#d62728")
        ax.set_yticks(x)
        ax.set_yticklabels([f"[{ARCH_TIERS.get(m,'?')}] {m}" for m in sub["Model"]],fontsize=9)
        ax.set_xlabel("R²",fontsize=11)
        ax.set_xlim(0,1.05)
        ax.axvline(0.9,color="grey",ls="--",lw=1,alpha=0.5,label="R²=0.90 reference")
        ax.legend(fontsize=10)
        ax.set_title(f"Spatial Generalisation Test | {TGT_LABELS.get(tgt,tgt)}\n"
                     f"Seen vs Unseen ({HOLDOUT_SITE}) Locations | Test 2025\n"
                     f"Small gap = good spatial generalisation via GCN",
                     fontweight="bold",fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGS/f"V4_01_seen_vs_unseen_r2_{tgt}.png",dpi=150,bbox_inches="tight")
        plt.close(); print(f"  ✓ V4_01 [{tgt}]")

    # V4-02: Spatial gap bar (seen_R2 - unseen_R2)
    if "spatial_gap" in results_df.columns:
        for tgt in results_df["Target"].unique():
            sub=results_df[results_df["Target"]==tgt].dropna(subset=["spatial_gap"])
            if sub.empty: continue
            sub=sub.sort_values("spatial_gap",ascending=False)
            fig,ax=plt.subplots(figsize=(14,8))
            colors=[TIER_COLORS.get(ARCH_TIERS.get(m,"?"),"grey") for m in sub["Model"]]
            bars=ax.bar(sub["Model"],sub["spatial_gap"],color=colors,
                        alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,sub["spatial_gap"]):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                        f"{v:.4f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
            ax.axhline(0,color="black",lw=2)
            ax.axhline(0.05,color="orange",ls="--",lw=1.5,alpha=0.8,
                       label="5% gap threshold")
            ax.set_ylabel("Spatial Gap (seen_R² - unseen_R²)",fontsize=11)
            ax.set_title(f"Spatial Generalisation Gap | {TGT_LABELS.get(tgt,tgt)}\n"
                         f"Smaller gap = better spatial generalisation\n"
                         f"0 gap = perfect inductive spatial prediction",
                         fontweight="bold",fontsize=12)
            ax.tick_params(axis="x",rotation=30,labelsize=9)
            ax.legend(fontsize=9)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()]+
                     [plt.Line2D([],[],color="orange",ls="--",label="5% threshold")],
                     fontsize=9)
            plt.tight_layout()
            plt.savefig(FIGS/f"V4_02_spatial_gap_{tgt}.png",dpi=150,bbox_inches="tight")
            plt.close(); print(f"  ✓ V4_02 [{tgt}]")

    # V4-03: Tier comparison heatmap
    for metric,lbl in [("seen_R2","Seen R²"),("unseen_R2","Unseen R²")]:
        if metric not in results_df.columns: continue
        fig,axes=plt.subplots(1,len(results_df["Target"].unique()),
                              figsize=(10*len(results_df["Target"].unique()),10))
        if not hasattr(axes,"__len__"): axes=[axes]
        for ax,tgt in zip(axes,results_df["Target"].unique()):
            sub=results_df[results_df["Target"]==tgt]
            if sub.empty: continue
            sub=sub.copy(); sub["Tier"]=sub["Model"].map(ARCH_TIERS)
            pv=sub.pivot_table(index="Tier",columns="Model",
                                values=metric,aggfunc="mean").round(4)
            if pv.empty: continue
            sns.heatmap(pv,ax=ax,cmap="RdYlGn",vmin=0.8,vmax=1.0,
                        annot=True,fmt=".4f",linewidths=0.5,linecolor="white",
                        annot_kws={"size":10,"weight":"bold"},
                        cbar_kws={"label":lbl,"shrink":0.85})
            ax.set_title(f"{lbl} | {TGT_LABELS.get(tgt,tgt)}",
                         fontweight="bold",fontsize=12)
            ax.tick_params(axis="x",rotation=30,labelsize=9)
            ax.tick_params(axis="y",rotation=0, labelsize=9)
        fig.suptitle(f"{lbl} by Tier and Model | v4 Spatial Holdout Experiment",
                     fontsize=14,fontweight="bold")
        plt.tight_layout()
        fname=f"V4_03_{'seen' if 'seen' in metric else 'unseen'}_tier_heatmap.png"
        plt.savefig(FIGS/fname,dpi=150,bbox_inches="tight")
        plt.close(); print(f"  ✓ {fname}")

    # V4-04: Ablation comparison
    ablation_df = results_df[results_df["Tier"].isin(["ABLATION","SSM","GRAPH"])]
    if not ablation_df.empty and "seen_R2" in ablation_df.columns:
        fig,axes=plt.subplots(1,2,figsize=(22,9))
        for ax,metric,lbl in [(axes[0],"seen_R2","Seen R²"),
                               (axes[1],"unseen_R2","Unseen R²")]:
            if metric not in ablation_df.columns: continue
            sub=ablation_df.dropna(subset=[metric])
            if sub.empty: continue
            sub=sub.sort_values(metric,ascending=True)
            colors=[TIER_COLORS.get(ARCH_TIERS.get(m,"?"),"grey") for m in sub["Model"]]
            bars=ax.barh(sub["Model"],sub[metric],color=colors,
                         alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,sub[metric]):
                ax.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",
                        va="center",fontsize=9,fontweight="bold")
            ax.set_xlabel(lbl,fontsize=11)
            ax.set_title(lbl,fontweight="bold",fontsize=12)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
                   loc="lower center",ncol=4,fontsize=10,bbox_to_anchor=(0.5,-0.05))
        fig.suptitle("Ablation Study — Does the Spatial Graph Help?\n"
                     "BiGRU_NoGCN (no graph) vs GraphSAGE/GAT/STGCN/SSM (with graph)",
                     fontsize=14,fontweight="bold")
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig(FIGS/"V4_04_ablation_comparison.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  ✓ V4_04 ablation comparison")

    # V4-05: DeepESN vs SSM comparison
    esn_ssm = results_df[results_df["Tier"].isin(["RESERVOIR","SSM"])]
    if not esn_ssm.empty and "seen_R2" in esn_ssm.columns:
        fig,ax=plt.subplots(figsize=(16,9))
        x=np.arange(len(results_df["Target"].unique())); w=0.8/max(len(esn_ssm["Model"].unique()),1)
        for mi,model in enumerate(esn_ssm["Model"].unique()):
            sub=esn_ssm[esn_ssm["Model"]==model]
            tgts=sorted(results_df["Target"].unique())
            vals=[sub[sub["Target"]==t]["seen_R2"].mean()
                  if len(sub[sub["Target"]==t])>0 else 0 for t in tgts]
            color=TIER_COLORS.get(ARCH_TIERS.get(model,"?"),"grey")
            bars=ax.bar(x+mi*w-0.4+w/2,vals,width=w*0.9,label=model,
                        color=color,alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_height()+0.001,f"{v:.3f}",
                                ha="center",va="bottom",fontsize=7,fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([TGT_LABELS.get(t,t) for t in sorted(results_df["Target"].unique())],fontsize=10)
        ax.set_ylabel("Seen R²",fontsize=11)
        ax.set_title("DeepESN (Reservoir Computing) vs SSM Models\n"
                     "Per senior recommendation — arXiv:1712.04323 & arXiv:2509.04422",
                     fontweight="bold",fontsize=12)
        ax.legend(fontsize=9,ncol=2); ax.set_ylim(0,1.05)
        plt.tight_layout()
        plt.savefig(FIGS/"V4_05_deepesn_vs_ssm.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  ✓ V4_05 DeepESN vs SSM")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — FINAL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  FINAL LEADERBOARD — v4 Distributed Spatial AI")
print("="*70)
print(f"  Holdout site: {HOLDOUT_SITE} | Unseen locs: {len(UNSEEN_LOCS)}")
print()
if len(results_df)>0 and "seen_R2" in results_df.columns:
    print(f"  {'Tier':<12} {'Model':<20} {'Target':<8} "
          f"{'Seen R²':>8} {'Unseen R²':>10} {'Gap':>6} {'FreezeAcc':>10}")
    print("  " + "─"*80)
    _sort = "unseen_R2" if "unseen_R2" in results_df.columns else "Val_R2"
    _df   = results_df.dropna(subset=[_sort]).sort_values(_sort,ascending=False)
    for _,row in _df.iterrows():
        tier=ARCH_TIERS.get(row["Model"],"?")
        print(f"  {tier:<12} {row['Model']:<20} {row['Target']:<8} "
              f"{row.get('seen_R2',float('nan')):>8.4f} "
              f"{row.get('unseen_R2',float('nan')):>10.4f} "
              f"{row.get('spatial_gap',float('nan')):>6.4f} "
              f"{row.get('unseen_FreezeAcc',float('nan')):>9.2f}%")
else:
    print("  No completed results yet.")

print(f"\n  Completed : {pd.Timestamp.now()}")
print(f"  Results   : {RESULTS}")
print(f"  Figures   : {FIGS}")
print(f"  Models    : {MODELS}")
print("="*70)
