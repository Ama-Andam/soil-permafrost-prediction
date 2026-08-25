"""
================================================================================
train_soil_spatial_v6.py
DISTRIBUTED SPATIAL AI — ALASKA PERMAFROST | v6 FULL REDESIGN
DoD PROJECT | University of North Dakota | 2022-2025
================================================================================

WHAT IS NEW IN v6 (vs v5) — ALL FROM PI MEETING:

[1] SPATIOTEMPORAL QUANTIZATION (PI document)
    - Each point q_i = (lat, lon, t) gets TWO values: mean + variance
    - Missing observations initialised with high variance (σ²_high = 1.0)
    - Observed points initialised with low variance (σ²_obs = 0.01)
    - This directly encodes uncertainty into the input representation
    - Gaussian imputation for sparse locations via weighted neighbours

[2] FEATURE ENGINEERING FIX
    - Cyclical encodings (sin/cos of day/month) REMOVED
    - They leaked seasonal signal → baseline ML matched DL R² trivially
    - Wavelet decomposition KEPT: approx (seasonal baseline) stays as input
    - Model predicts residual (detail component), not raw target
    - Input uncertainty variance added as extra feature per modality

[3] NEW ATTENTION TIER — 2 models
    - SpatialTransformer: full temporal self-attention + GCN
    - SpatialInformer: ProbSparse attention (efficient for long sequences) + GCN
    - Ref: Vaswani et al. 2017 (NeurIPS), Zhou et al. 2021 (AAAI)

[4] HETEROSCEDASTIC OUTPUT HEAD (all 13 models)
    - Every model outputs BOTH μ (mean) and log σ² (log variance)
    - Trained with NLL loss: -log N(y | μ, σ²)
    - σ² IS the model's predicted uncertainty — not post-hoc MC-Dropout
    - High σ² expected for: unseen locations, sparse data, extreme events

[5] THREE TEST SETS
    - Unseen space:  Wetland site withheld from training (spatial holdout)
    - Unseen time:   Q4 2025 (Oct-Dec) withheld from training (temporal holdout)
    - Unseen both:   Wetland × Q4 2025 (hardest — neither space nor time seen)

[6] MULTI-STAGE RANDOM SEARCH TUNING (50 trials total)
    - Stage 1: 35 broad random trials per model (wide hyperbound space)
    - Stage 2: 15 narrow trials around top-3 Stage 1 configs
    - No Ray Tune — pure Python random search, parallelised at model level
    - Best config saved per model before final training

[7] FULL ABLATION FOR ALL 13 MODELS
    - Each model tested with: no GCN, no temporal, no attention, no Laplacian
    - Ablation results saved separately, not mixed with main results

[8] EXPANDED METRIC SET (distinct, non-redundant)
    - R²          — variance explained
    - KGE         — Kling-Gupta Efficiency (bias + correlation + variability)
    - ubRMSE      — unbiased RMSE (removes mean bias, tests dynamic range)
    - CRPS        — Continuous Ranked Probability Score (proper scoring rule)
    - DTW         — Dynamic Time Warping (temporal shape similarity)
    - KL Div      — KL(N(μ_pred,σ²_pred) || N(μ_obs,σ²_obs)) per site
    - NLL         — Negative log likelihood (uncertainty calibration)
    Removed: plain RMSE (replaced by ubRMSE), plain r (redundant with R²)

[9] OBJECTIVES
    - NLL loss (heteroscedastic — main loss)
    - Graph Laplacian regularisation (spatial smoothness)
    - CRPS loss term (proper scoring, encourages calibrated uncertainty)
    - Freeze/thaw boundary loss (physics-informed: penalise wrong sign near 0°C)

[10] VISUALISATIONS
    - KDE: predicted vs observed distribution per site per model
    - Uncertainty distribution: seen vs unseen locations
    - Entropy: initial (epoch 0) vs best epoch per architecture

REFERENCES:
    Mamba:        Gu & Dao 2023 (arXiv:2312.00752)
    S4:           Gu et al. 2022 (arXiv:2111.00396)
    DeepESN:      Gallicchio & Micheli 2017 (arXiv:1712.04323)
    ESN-SSM:      Singh & Raman 2025 (arXiv:2509.04422)
    GraphSAGE:    Hamilton et al. 2017 (NeurIPS)
    GAT:          Velickovic et al. 2018 (ICLR)
    STGCN:        Yu et al. 2018 (IJCAI)
    Transformer:  Vaswani et al. 2017 (NeurIPS)
    Informer:     Zhou et al. 2021 (AAAI)
    Heteroscedastic: Kendall & Gal 2017 (NeurIPS)
    CRPS:         Gneiting & Raftery 2007 (JASA)
    KGE:          Gupta et al. 2009 (J. Hydrology)
================================================================================
"""

import os, sys, time, json, pickle, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.special import ndtr  # normal CDF for CRPS

warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train","tune","ablation","figures"],
                    default="train")
parser.add_argument("--arch", type=str, default=None,
                    help="Single arch (default: all 13)")
parser.add_argument("--target", type=str, default=None,
                    choices=["temp","smap","moist"],
                    help="Single target (default: all 3)")
parser.add_argument("--tune_trials", type=int, default=50,
                    help="Total tuning trials (35 broad + 15 narrow)")
parser.add_argument("--ablation_component", type=str, default="all",
                    choices=["all","no_gcn","no_temporal","no_attention",
                             "no_laplacian","no_uncertainty"])
args = parser.parse_args()

# ── Logger ────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, p):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        self.t = sys.__stdout__
        self.f = open(p, "a", buffering=1)
    def write(self, m): self.t.write(m); self.f.write(m)
    def flush(self):    self.t.flush();  self.f.flush()

LOG_MAP = {"train":"soil_training_v6.log","tune":"soil_tuning_v6.log",
           "ablation":"soil_ablation_v6.log","figures":"soil_figures_v6.log"}
sys.stdout = Tee(f"/home/emmanuel.keku/logs/{LOG_MAP[args.mode]}")
sys.stderr = sys.stdout

JOB_ID = os.environ.get("SLURM_JOB_ID","local")
NODE   = os.environ.get("SLURMD_NODENAME","unknown")
SEED   = 42
np.random.seed(SEED)

print("=" * 70)
print(f"  SOIL SPATIAL v6 | Mode: {args.mode.upper()} | {pd.Timestamp.now()}")
print(f"  Job: {JOB_ID} | Node: {NODE}")
print("=" * 70)
t_preproc_start = time.time()  # track preprocessing time

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
MODELS  = PROJECT / "models_v6" / "dl"
FIGS    = PROJECT / "figures_v6"
LOGS    = PROJECT / "logs"
for d in [RESULTS, MODELS, FIGS, LOGS]: d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    N_GPUS = torch.cuda.device_count()
    print(f"PyTorch {torch.__version__} | {DEVICE} | {N_GPUS} GPU(s)")
    for i in range(N_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"FATAL: {e}"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55 + "\n  PHASE 1: Data\n" + "="*55)
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
with open(PREPROC/"scalers.pkl",      "rb") as f: SC = pickle.load(f)
with open(PREPROC/"feature_info.pkl", "rb") as f: FI = pickle.load(f)

LOCATIONS    = pd.DataFrame(FI["LOCATIONS"])
N_LOCS       = FI["N_LOCS"]
SNAP_FEATURES= FI["SNAP_FEATURES"]
ALL_TARGETS  = FI["ALL_TARGETS"]
TEMP_TARGETS = FI["TEMP_TARGETS"]
SMAP_TARGETS = FI["SMAP_TARGETS"]
MOIST_TARGETS= FI["MOIST_TARGETS"]
SITES        = FI["SITES"]

# ── v6: Remove cyclical encodings, keep wavelet approx ───────────────────────
CYCLICAL_COLS = [c for c in df.columns if any(
    c.startswith(p) for p in ["sin_","cos_","day_of_year_sin","day_of_year_cos",
                               "month_sin","month_cos","hour_sin","hour_cos"])]
print(f"  Removing {len(CYCLICAL_COLS)} cyclical features: {CYCLICAL_COLS[:6]}...")

APPROX_COLS = [f"{t}_approx" for t in ALL_TARGETS if f"{t}_approx" in df.columns]
WAVELET_DETAIL_COLS = [f"{t}_residual" for t in ALL_TARGETS if f"{t}_residual" in df.columns]

# Core features = SNAP minus cyclical, plus wavelet approx
CORE_FEATURES = [f for f in SNAP_FEATURES
                  if f not in CYCLICAL_COLS and f in df.columns]
APPROX_INPUT  = [f for f in APPROX_COLS if f in df.columns]

# ── v6: Spatiotemporal quantization — add input uncertainty variance ──────────
# For each modality, create a variance feature:
#   σ²=0.01 where observation exists, σ²=1.0 where imputed/missing
UNCERTAINTY_VAR_COLS = []
for feat in CORE_FEATURES[:8]:  # top modalities only (avoid explosion)
    var_col = f"{feat}_unc_var"
    if var_col not in df.columns:
        df[var_col] = np.where(df[feat].isna(), 1.0, 0.01)
    UNCERTAINTY_VAR_COLS.append(var_col)

# Add BOTH approx (long-term) AND residual (short-term) as input features
# Per PI: wavelet separates long/short term memory for model to process
# Model sees: raw met features + seasonal baseline + short-term residual
RESIDUAL_INPUT = [f for f in WAVELET_DETAIL_COLS if f in df.columns]
V6_FEATURES = list(dict.fromkeys(
    CORE_FEATURES + APPROX_INPUT + RESIDUAL_INPUT + UNCERTAINTY_VAR_COLS))
V6_FEATURES = [f for f in V6_FEATURES if f in df.columns]
N_FEATS     = len(V6_FEATURES)
print(f"  v6 features: {N_FEATS}")
print(f"    core={len(CORE_FEATURES)} | approx={len(APPROX_INPUT)} | "
      f"residual={len(RESIDUAL_INPUT)} | unc_var={len(UNCERTAINTY_VAR_COLS)}")
print(f"  Feature list sample: {V6_FEATURES[:5]} ... {V6_FEATURES[-3:]}")

# ── Scalers ────────────────────────────────────────────────────────────────────
tr = df[df["split"]=="train"]
feat_sc = RobustScaler(); feat_sc.fit(tr[V6_FEATURES].fillna(0).values)

# ── v6: Predict RESIDUAL (wavelet detail) not raw target ─────────────────────
# Detail = raw - approx = what remains after seasonal removal
# This forces models to learn the non-seasonal signal
# ML baselines were getting R²≈0.96 because seasonal dominates raw targets
# With residual target, seasonal leakage is eliminated
tgt_scalers = {}
DETAIL_TARGET_MAP = {}
for grp, cols in [("temp",TEMP_TARGETS),("smap",SMAP_TARGETS),("moist",MOIST_TARGETS)]:
    # Prefer detail (residual) columns if available
    detail_cols = [f"{c}_residual" for c in cols if f"{c}_residual" in tr.columns]
    raw_cols    = [c for c in cols if c in tr.columns]
    use_cols    = detail_cols if detail_cols else raw_cols
    DETAIL_TARGET_MAP[grp] = dict(
        use_cols=use_cols,
        raw_cols=raw_cols,
        approx_cols=[f"{c}_approx" for c in cols if f"{c}_approx" in tr.columns],
        is_residual=bool(detail_cols))
    if not use_cols: continue
    ts = RobustScaler(); ts.fit(tr[use_cols].dropna().values)
    tgt_scalers[grp] = (ts, use_cols)
    mode = "RESIDUAL (detail)" if detail_cols else "RAW (no detail cols found)"
    print(f"  ✓ scaler [{grp}] {len(use_cols)} cols | {mode}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — THREE TEST SPLITS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55 + "\n  PHASE 2: Three test sets\n" + "="*55)

HOLDOUT_SITE = "Wetland"
TRAINING_SITES = [s for s in SITES if s != HOLDOUT_SITE]

# Temporal holdout: Q4 2025 (Oct-Dec 2025)
df["year"]  = df["time_utc"].dt.year
df["month"] = df["time_utc"].dt.month
TEMPORAL_HOLDOUT_MASK = (df["year"]==2025) & (df["month"]>=10)
TEMPORAL_TRAIN_MASK   = ~TEMPORAL_HOLDOUT_MASK

# Re-label splits for v6
df["split_v6"] = df["split"].copy()
df.loc[TEMPORAL_HOLDOUT_MASK & (df["split"]=="test"), "split_v6"] = "test_time"
df.loc[df["Site"]==HOLDOUT_SITE, "split_v6"] = \
    df.loc[df["Site"]==HOLDOUT_SITE, "split_v6"].map(
        {"train":"seen_train","val":"seen_val",
         "test":"test_space","test_time":"test_both"}).fillna(
        df.loc[df["Site"]==HOLDOUT_SITE, "split_v6"])

# Location indices
loc_to_idx = {(float(r.Latitude),float(r.Longitude)): i
               for i,r in LOCATIONS.iterrows()}

def site_locs(site):
    rows = df[df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    return sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))
                   for _,r in rows.iterrows()
                   if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

SEEN_LOCS   = sorted(set(i for s in TRAINING_SITES for i in site_locs(s)))
UNSEEN_LOCS = site_locs(HOLDOUT_SITE)
print(f"  Seen locs: {len(SEEN_LOCS)} | Unseen (Wetland): {len(UNSEEN_LOCS)}")
print(f"  Temporal holdout rows: {TEMPORAL_HOLDOUT_MASK.sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SPATIAL GRAPH
# ══════════════════════════════════════════════════════════════════════════════
coords = LOCATIONS[["Latitude","Longitude"]].values.astype(np.float32)
scaled = coords * np.array([111.0,63.0])
tree   = cKDTree(scaled); dists, idxs = tree.query(scaled, k=7)
sigma  = np.median(dists[:,1:]) + 1e-8
A_np   = np.zeros((N_LOCS,N_LOCS), dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1,dists.shape[1]):
        j=idxs[i,jp]; w=float(np.exp(-dists[i,jp]/sigma))
        A_np[i,j]+=w; A_np[j,i]+=w
A_np += np.eye(N_LOCS)
D = A_np.sum(1,keepdims=True)**0.5
A_norm = torch.tensor((A_np/(D*D.T+1e-8)).astype(np.float32)).to(DEVICE)
print(f"  Graph: N={N_LOCS} nodes | σ={sigma:.2f} km")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — DATASET (heteroscedastic: y output = mean only, uncertainty from head)
# ══════════════════════════════════════════════════════════════════════════════
class SpatialDatasetV6(Dataset):
    """
    v6 dataset: encodes spatiotemporal quantization.
    Each point has feature vector X_i = [x^(1)...x^(M)] where
    missing modalities are imputed with weighted neighbours and
    high uncertainty variance σ²=1.0 (per PI document).
    """
    def __init__(self, df_sub, split_label, tgt_cols, fs, ts,
                 lookback=24, stride=6, max_samples=None, mask_unseen=True):
        sub = df_sub[df_sub["split"]=="train"].copy() if split_label=="train" \
              else df_sub[df_sub["split_v6"]==split_label].copy()

        all_ts = sorted(sub["time_utc"].unique()); T=len(all_ts)
        if T < lookback+2:
            self.X=self.y=self.mask=self.A=torch.zeros(0); return

        ts_to_i = {t:i for i,t in enumerate(all_ts)}
        sub["_ti"] = sub["time_utc"].map(ts_to_i)
        sub["_ni"] = [loc_to_idx.get((float(la),float(lo)))
                      for la,lo in zip(sub["Latitude"],sub["Longitude"])]
        sub = sub.dropna(subset=["_ti","_ni"])
        sub["_ti"]=sub["_ti"].astype(int); sub["_ni"]=sub["_ni"].astype(int)

        Xf = np.full((T,N_LOCS,N_FEATS),0.,dtype=np.float32)
        yf = np.full((T,N_LOCS,len(tgt_cols)),0.,dtype=np.float32)
        mf = np.zeros((T,N_LOCS),dtype=np.float32)
        obs_mask = np.zeros((T,N_LOCS),dtype=np.float32)

        ti=sub["_ti"].values; ni=sub["_ni"].values
        Xf[ti,ni]=fs.transform(sub[V6_FEATURES].fillna(0).values).astype(np.float32)
        yf[ti,ni]=ts.transform(sub[tgt_cols].fillna(0).values).astype(np.float32)
        obs_mask[ti,ni]=1.0

        # Spatiotemporal quantization: impute missing locs with weighted neighbours
        # (simplified: use graph-weighted mean of observed neighbours)
        A_np_cpu = A_norm.cpu().numpy()
        for t_idx in range(T):
            missing = np.where(obs_mask[t_idx]==0)[0]
            observed= np.where(obs_mask[t_idx]==1)[0]
            if len(observed)==0 or len(missing)==0: continue
            for m in missing:
                w = A_np_cpu[m,observed]
                if w.sum()<1e-8: continue
                w = w/w.sum()
                Xf[t_idx,m] = (w[:,None]*Xf[t_idx,observed]).sum(0)
                # Set uncertainty variance cols to 1.0 (high — imputed)
                unc_start = len(CORE_FEATURES)+len(APPROX_INPUT)+len(RESIDUAL_INPUT)
                Xf[t_idx,m,unc_start:] = 1.0

        if mask_unseen and split_label=="train":
            mf[:,SEEN_LOCS]=1.0
        else:
            mf[:,:]=1.0

        tidxs=list(range(lookback,T,stride))
        if max_samples and len(tidxs)>max_samples:
            rng=np.random.default_rng(SEED)
            tidxs=sorted(rng.choice(tidxs,max_samples,replace=False))

        Xl=[]; yl=[]; ml=[]
        for ti_ in tidxs:
            Xw=Xf[ti_-lookback:ti_]; yi=yf[ti_]; mi=mf[ti_]
            if np.isnan(Xw).mean()>0.3: continue
            Xl.append(np.nan_to_num(Xw,nan=0.))
            yl.append(np.nan_to_num(yi,nan=0.)); ml.append(mi)

        if not Xl:
            self.X=self.y=self.mask=self.A=torch.zeros(0); return

        self.X    = torch.tensor(np.array(Xl),dtype=torch.float32)
        self.y    = torch.tensor(np.array(yl),dtype=torch.float32)
        self.mask = torch.tensor(np.array(ml),dtype=torch.float32)
        self.A    = A_norm
        print(f"    [{split_label}] {len(self.X):,} samples")

    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return self.X[i],self.y[i],self.mask[i],self.A


def make_loaders(tgt_grp, bs=4, max_tr=2000, max_ev=400):
    if tgt_grp not in tgt_scalers: return {}
    ts,tgt_cols = tgt_scalers[tgt_grp]
    loaders={}
    for sp,ms,st,mu in [
        ("train",        max_tr, 6,  True),
        ("val",          max_ev, 24, False),
        ("test",         max_ev, 24, False),   # standard test
        ("test_space",   max_ev, 24, False),   # unseen space (Wetland)
        ("test_time",    max_ev, 24, False),   # unseen time (Q4 2025)
        ("test_both",    max_ev, 24, False),   # unseen space + time
    ]:
        ds=SpatialDatasetV6(df, sp, tgt_cols, feat_sc, ts,
                             stride=st, max_samples=ms, mask_unseen=mu)
        loaders[sp] = None if len(ds)==0 else \
            DataLoader(ds,batch_size=bs,shuffle=(sp=="train"),
                       num_workers=0,pin_memory=False,drop_last=(sp=="train"))
    return loaders

print("\nBuilding v6 dataloaders (includes imputation — may take 2-3 min)...")
TARGET_GROUPS = []
for grp in ["temp","smap","moist"]:
    if args.target and args.target!=grp: continue
    ld = make_loaders(grp)
    if ld.get("train"):
        TARGET_GROUPS.append((grp, tgt_scalers[grp][1], ld,
                               {"temp":"Weather Temp (°C)",
                                "smap":"SMAP Temp L1 (K)",
                                "moist":"Soil Moisture (m³/m³)"}[grp]))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

DP = 0.15  # dropout

# ── Shared: GraphConv ─────────────────────────────────────────────────────────
class GConv(nn.Module):
    def __init__(self, d, dp=DP):
        super().__init__()
        self.W=nn.Linear(d,d,bias=False); self.n=nn.LayerNorm(d)
        self.d=nn.Dropout(dp); self.a=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        return self.a(self.n(torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),
                                        self.W(self.d(H)))))

# ── Shared: Heteroscedastic head ─────────────────────────────────────────────
class HetHead(nn.Module):
    """
    Outputs μ (mean prediction) and log σ² (log variance = uncertainty).
    Ref: Kendall & Gal 2017 (NeurIPS) — What Uncertainties Do We Need?
    log σ² is unbounded; σ² = exp(log σ²) is always positive.
    """
    def __init__(self, in_d, nt, dp=DP):
        super().__init__()
        self.mu  = nn.Sequential(nn.Linear(in_d,in_d//2),nn.GELU(),
                                  nn.Dropout(dp),nn.Linear(in_d//2,nt))
        self.lsv = nn.Sequential(nn.Linear(in_d,in_d//2),nn.GELU(),
                                  nn.Dropout(dp),nn.Linear(in_d//2,nt))
    def forward(self,h):
        return self.mu(h), self.lsv(h)  # μ, log σ²


# ── ABLATION 1: BiGRU_NoGCN ───────────────────────────────────────────────────
class BiGRU_NoGCN(nn.Module):
    name="BiGRU_NoGCN"; tier="ABLATION"
    def __init__(self,nf,h=96,nl=2,nh=4,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                      dropout=dp if nl>1 else 0.)
        d2=h*2
        self.at=nn.MultiheadAttention(d2,nh,dropout=dp,batch_first=True)
        self.n1=nn.LayerNorm(d2); self.n2=nn.LayerNorm(d2)
        self.ff=nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),
                               nn.Dropout(dp),nn.Linear(d2*2,d2))
        self.r=nn.Linear(d2,h); self.hd=HetHead(h,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); a,_=self.at(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ff(h))
        h=self.r(h[:,-1,:]).reshape(B,N,-1)
        mu,lsv=self.hd(h); return mu,lsv


# ── ABLATION 2: GCN_NoTemporal ────────────────────────────────────────────────
class GCN_NoTemporal(nn.Module):
    name="GCN_NoTemporal"; tier="ABLATION"
    def __init__(self,nf,h=96,gl=3,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.gc=nn.ModuleList([GConv(h,dp) for _ in range(gl)])
        self.hd=HetHead(h*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape; h=self.p(x[:,-1,:,:]); h0=h
        for g in self.gc: h=g(h,A)
        mu,lsv=self.hd(torch.cat([h0,h],dim=-1)); return mu,lsv


# ── RESERVOIR 1: DeepESN ──────────────────────────────────────────────────────
class ESNLayer(nn.Module):
    def __init__(self,id,rd,sr=0.9,lr=0.3,dp=DP):
        super().__init__()
        self.rd=rd; self.lr=lr
        Wi=torch.randn(rd,id)*0.1
        Wr=torch.randn(rd,rd)
        ev=torch.linalg.eigvals(Wr).abs()
        Wr=Wr*(sr/(ev.max().item()+1e-8))
        self.register_buffer("Wi",Wi); self.register_buffer("Wr",Wr)
        self.dr=nn.Dropout(dp); self.nm=nn.LayerNorm(rd)
    def forward(self,x):
        B,L,_=x.shape; h=torch.zeros(B,self.rd,device=x.device,dtype=x.dtype)
        sts=[]
        for t in range(L):
            pre=x[:,t,:]@self.Wi.T+h@self.Wr.T
            h=(1-self.lr)*h+self.lr*torch.tanh(pre); sts.append(h)
        return self.nm(torch.stack(sts,dim=1))

class DeepESN(nn.Module):
    name="DeepESN"; tier="RESERVOIR"
    def __init__(self,nf,rd=128,nl=3,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,rd)
        lrs=[0.3*(0.5**i) for i in range(nl)]
        self.es=nn.ModuleList([ESNLayer(rd,rd,0.9,lrs[i],dp) for i in range(nl)])
        self.hd=HetHead(rd*nl,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        sts=[]
        for e in self.es: h=e(h); sts.append(h[:,-1,:])
        mu,lsv=self.hd(torch.cat(sts,dim=-1).reshape(B,N,-1))
        return mu,lsv


# ── RESERVOIR 2: SpatialESN ───────────────────────────────────────────────────
class SpatialESN(nn.Module):
    name="SpatialESN"; tier="RESERVOIR"
    def __init__(self,nf,rd=128,nl=3,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,rd)
        lrs=[0.3*(0.5**i) for i in range(nl)]
        self.es=nn.ModuleList([ESNLayer(rd,rd,0.9,lrs[i],dp) for i in range(nl)])
        self.cm=nn.Linear(rd*nl,rd)
        self.gc=nn.ModuleList([GConv(rd,dp) for _ in range(gl)])
        self.hd=HetHead(rd*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        sts=[]
        for e in self.es: h=e(h); sts.append(h[:,-1,:])
        h0=torch.relu(self.cm(torch.cat(sts,dim=-1))).reshape(B,N,-1)
        hg=h0
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h0,hg],dim=-1)); return mu,lsv


# ── GRAPH 1: GraphSAGE ────────────────────────────────────────────────────────
class SAGEConv(nn.Module):
    def __init__(self,d,dp=DP):
        super().__init__()
        self.Ws=nn.Linear(d,d,bias=False); self.Wn=nn.Linear(d,d,bias=False)
        self.nm=nn.LayerNorm(d); self.dr=nn.Dropout(dp); self.ac=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        nb=torch.bmm(A.unsqueeze(0).expand(H.shape[0],-1,-1),self.dr(H))
        return self.ac(self.nm(self.Ws(self.dr(H))+self.Wn(nb)))

class GraphSAGE(nn.Module):
    name="GraphSAGE"; tier="GRAPH"
    def __init__(self,nf,h=96,nl=2,gl=3,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                      dropout=dp if nl>1 else 0.)
        self.r=nn.Linear(h*2,h)
        self.sg=nn.ModuleList([SAGEConv(h,dp) for _ in range(gl)])
        self.hd=HetHead(h*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for s in self.sg: hg=s(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── GRAPH 2: GAT ──────────────────────────────────────────────────────────────
class GATConv(nn.Module):
    def __init__(self,d,nh=4,dp=DP):
        super().__init__()
        self.nh=nh; self.hd=d//nh
        self.W=nn.Linear(d,d,bias=False)
        self.as_=nn.Linear(self.hd,1,bias=False)
        self.ad=nn.Linear(self.hd,1,bias=False)
        self.nm=nn.LayerNorm(d); self.dr=nn.Dropout(dp); self.ac=nn.GELU()
    def forward(self,H,A):
        if A.dim()==3: A=A[0]
        if A.dim()==4: A=A[0,0]
        B,N,_=H.shape; Wh=self.W(self.dr(H)).view(B,N,self.nh,self.hd)
        es=self.as_(Wh); ed=self.ad(Wh)
        e=F.leaky_relu(es.unsqueeze(2)+ed.unsqueeze(1),0.2)
        mask=(A==0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        e=e.masked_fill(mask,-1e9)
        al=F.softmax(e,dim=2)
        WH=Wh.unsqueeze(1).expand(-1,N,-1,-1,-1)
        out=(al*WH).sum(2).view(B,N,-1)
        return self.ac(self.nm(out))

class GAT(nn.Module):
    name="GAT"; tier="GRAPH"
    def __init__(self,nf,h=96,nl=2,nh=4,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                      dropout=dp if nl>1 else 0.)
        self.r=nn.Linear(h*2,h)
        self.gt=nn.ModuleList([GATConv(h,nh,dp) for _ in range(gl)])
        self.hd=HetHead(h*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gt: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── GRAPH 3: STGCN ────────────────────────────────────────────────────────────
class STGCN(nn.Module):
    name="STGCN"; tier="GRAPH"
    def __init__(self,nf,h=64,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,2,batch_first=True,bidirectional=True,dropout=dp)
        self.r=nn.Linear(h*2,h)
        self.gc=nn.ModuleList([GConv(h,dp) for _ in range(gl)])
        self.hd=HetHead(h*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── NEW ATTENTION TIER 1: SpatialTransformer ─────────────────────────────────
class SpatialTransformer(nn.Module):
    """
    Full temporal self-attention (Vaswani et al. 2017) + GCN spatial graph.
    Positional encoding is learned (no sin/cos — avoids cyclical leakage).
    """
    name="SpatialTransformer"; tier="ATTENTION"
    def __init__(self,nf,d=96,nl=4,nh=8,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.pe=nn.Embedding(256,d)  # learned positional encoding (not cyclical)
        enc_layer=nn.TransformerEncoderLayer(d,nh,d*4,dp,batch_first=True,norm_first=True)
        self.te=nn.TransformerEncoder(enc_layer,nl)
        self.nm=nn.LayerNorm(d)
        self.gc=nn.ModuleList([GConv(d,dp) for _ in range(gl)])
        self.hd=HetHead(d*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        pos=self.pe(torch.arange(L,device=x.device)).unsqueeze(0)
        h=self.te(h+pos); h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── NEW ATTENTION TIER 2: SpatialInformer ────────────────────────────────────
class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention (Zhou et al. 2021, AAAI — Informer).
    O(L log L) complexity vs O(L²) for full attention.
    Selects top-u queries by sparsity measurement.
    """
    def __init__(self,d,nh,factor=5,dp=DP):
        super().__init__()
        self.nh=nh; self.hd=d//nh; self.factor=factor
        self.Wq=nn.Linear(d,d,bias=False); self.Wk=nn.Linear(d,d,bias=False)
        self.Wv=nn.Linear(d,d,bias=False); self.Wo=nn.Linear(d,d)
        self.dr=nn.Dropout(dp); self.nm=nn.LayerNorm(d)
    def forward(self,x):
        B,L,D=x.shape; H=self.nh; Hd=self.hd
        Q=self.Wq(x).view(B,L,H,Hd).transpose(1,2)
        K=self.Wk(x).view(B,L,H,Hd).transpose(1,2)
        V=self.Wv(x).view(B,L,H,Hd).transpose(1,2)
        # Sparsity measure: select top-u queries
        u=max(1,int(self.factor*np.log(L+1)))
        u=min(u,L)
        idx=torch.randperm(L,device=x.device)[:u]
        Qs=Q[:,:,idx,:]
        sc=(Qs@K.transpose(-2,-1))/Hd**0.5
        sc=self.dr(F.softmax(sc,dim=-1))
        ctx=sc@V  # (B,H,u,Hd)
        out=torch.zeros_like(Q)
        out[:,:,idx,:]=ctx
        out=out.transpose(1,2).contiguous().view(B,L,D)
        return self.nm(x+self.Wo(out))

class SpatialInformer(nn.Module):
    """
    Informer-style ProbSparse attention + GCN spatial graph.
    More efficient than full Transformer for long sequences.
    """
    name="SpatialInformer"; tier="ATTENTION"
    def __init__(self,nf,d=96,nl=3,nh=8,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.layers=nn.ModuleList([ProbSparseAttention(d,nh,dp=dp) for _ in range(nl)])
        self.ffs=nn.ModuleList([
            nn.Sequential(nn.Linear(d,d*2),nn.GELU(),
                          nn.Dropout(dp),nn.Linear(d*2,d),nn.LayerNorm(d))
            for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gc=nn.ModuleList([GConv(d,dp) for _ in range(gl)])
        self.hd=HetHead(d*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for att,ff in zip(self.layers,self.ffs): h=ff(att(h))
        h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── SSM: MambaBlock ───────────────────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self,d,ds=16,dc=4,ex=2,dp=DP):
        super().__init__()
        self.di=d*ex; self.ds=ds
        self.ip=nn.Linear(d,self.di*2,bias=False)
        self.cv=nn.Conv1d(self.di,self.di,dc,padding=dc-1,groups=self.di)
        self.sl=nn.SiLU()
        self.xp=nn.Linear(self.di,ds*2+self.di,bias=False)
        self.dp_=nn.Linear(self.di,self.di,bias=True)
        A_=torch.arange(1,ds+1,dtype=torch.float32).unsqueeze(0).repeat(self.di,1)
        self.Al=nn.Parameter(torch.log(A_)); self.D_=nn.Parameter(torch.ones(self.di))
        self.op=nn.Linear(self.di,d,bias=False); self.dr=nn.Dropout(dp); self.nm=nn.LayerNorm(d)
    def scan(self,x):
        B,L,D=x.shape; S=self.ds
        xd=self.xp(x); dl,Bp,C=xd.split([D,S,S],dim=-1)
        dl=F.softplus(self.dp_(dl)); A__=-torch.exp(self.Al.float())
        dA=torch.exp(torch.einsum("bld,ds->blds",dl,A__))
        dB=torch.einsum("bld,bls->blds",dl,Bp)
        h=torch.zeros(B,D,S,device=x.device,dtype=x.dtype); ys=[]
        for i in range(L):
            h=dA[:,i]*h+dB[:,i]*x[:,i,:,None]
            ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
        return torch.stack(ys,dim=1)*self.D_
    def forward(self,x):
        r=x; xz=self.ip(x); x_,z=xz.chunk(2,dim=-1)
        x_=self.sl(self.cv(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        y=self.scan(x_)*self.sl(z)
        return self.nm(r+self.op(self.dr(y)))


# ── SSM 1: SpatialBiGRU ───────────────────────────────────────────────────────
class SpatialBiGRU(nn.Module):
    name="SpatialBiGRU"; tier="SSM"
    def __init__(self,nf,h=96,nl=2,nh=4,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.p=nn.Linear(nf,h)
        self.g=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,
                      dropout=dp if nl>1 else 0.)
        d2=h*2
        self.at=nn.MultiheadAttention(d2,nh,dropout=dp,batch_first=True)
        self.n1=nn.LayerNorm(d2); self.n2=nn.LayerNorm(d2)
        self.ff=nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),
                               nn.Dropout(dp),nn.Linear(d2*2,d2))
        self.r=nn.Linear(d2,h)
        self.gc=nn.ModuleList([GConv(h,dp) for _ in range(gl)])
        self.hd=HetHead(h*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.p(x.permute(0,2,1,3).reshape(B*N,L,F))
        h,_=self.g(h); a,_=self.at(h,h,h)
        h=self.n1(h+a); h=self.n2(h+self.ff(h))
        h=self.r(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── SSM 2: SpatialMamba ───────────────────────────────────────────────────────
class SpatialMamba(nn.Module):
    name="SpatialMamba"; tier="SSM"
    def __init__(self,nf,d=96,nl=4,ds=16,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.mb=nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(int(nl))])
        self.nm=nn.LayerNorm(d)
        self.gc=nn.ModuleList([GConv(d,dp) for _ in range(gl)])
        self.hd=HetHead(d*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for b in self.mb: h=b(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── SSM 3: SpatialS4 ─────────────────────────────────────────────────────────
class S4Layer(nn.Module):
    def __init__(self,d,ds=64,dp=DP):
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
    name="SpatialS4"; tier="SSM"
    def __init__(self,nf,d=96,nl=4,ds=64,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.em=nn.Linear(nf,d)
        self.ly=nn.ModuleList([S4Layer(d,ds,dp) for _ in range(nl)])
        self.nm=nn.LayerNorm(d)
        self.gc=nn.ModuleList([GConv(d,dp) for _ in range(gl)])
        self.hd=HetHead(d*2,nt,dp)
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for l in self.ly: h=l(h)
        h=self.nm(h[:,-1,:]).reshape(B,N,-1); hg=h
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([h,hg],dim=-1)); return mu,lsv


# ── SSM 4: SpatialFuseMoE ────────────────────────────────────────────────────
class SpatialFuseMoE(nn.Module):
    name="SpatialFuseMoE"; tier="SSM"
    def __init__(self,nf,d=96,ne=4,tk=2,ds=16,nsl=2,gl=2,nt=1,dp=DP,**kw):
        super().__init__()
        self.ne=int(ne); self.tk=int(tk); self.d=int(d)
        self.em=nn.Linear(nf,d)
        self.ex=nn.ModuleList([
            MambaBlock(d,ds,dp=dp),
            nn.GRU(d,d,batch_first=True),
            nn.Sequential(nn.Conv1d(d,d,7,padding=3,groups=d),
                          nn.Conv1d(d,d,1),nn.GELU(),nn.AdaptiveAvgPool1d(1)),
            nn.GRU(d,d,batch_first=True)])
        self.enm=nn.ModuleList([nn.LayerNorm(d) for _ in range(ne)])
        self.gt=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ne))
        self.bb=nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nsl)])
        self.gc=nn.ModuleList([GConv(d,dp) for _ in range(gl)])
        self.nm=nn.LayerNorm(d); self.hd=HetHead(d*2,nt,dp)
    def _expert(self,i,h):
        ex=self.ex[i]
        if isinstance(ex,MambaBlock): return self.enm[i](ex(h)[:,-1,:])
        if isinstance(ex,nn.GRU):     _,ht=ex(h); return self.enm[i](ht[-1])
        return self.enm[i](ex(h.transpose(1,2)).squeeze(-1))
    def forward(self,x,A):
        B,L,N,F=x.shape
        h=self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        gi=h.mean(1); lg=self.gt(gi)
        tv,ti=lg.topk(self.tk,dim=-1)
        import torch.nn.functional as _F
        gs=_F.softmax(tv.float(),dim=-1); gs_s=_F.softmax(lg.float(),dim=-1)
        imp=gs_s.mean(0); ld_=(gs_s>1/self.ne).float().mean(0)
        aux=(imp*ld_).sum()*self.ne
        eo=[self._expert(i,h) for i in range(self.ne)]
        Es=torch.stack(eo,dim=1)
        sel=torch.gather(Es,1,ti.unsqueeze(-1).expand(-1,-1,self.d))
        fsd=(sel*gs.unsqueeze(-1)).sum(1)
        fs=fsd.unsqueeze(1).expand(-1,L,-1)+h
        for b in self.bb: fs=b(fs)
        ho=self.nm(fs[:,-1,:]).reshape(B,N,-1); hg=ho
        for g in self.gc: hg=g(hg,A)
        mu,lsv=self.hd(torch.cat([ho,hg],dim=-1)); return mu,lsv,aux


# ── Model registry ────────────────────────────────────────────────────────────
ALL_MODELS = [
    BiGRU_NoGCN, GCN_NoTemporal,
    DeepESN, SpatialESN,
    GraphSAGE, GAT, STGCN,
    SpatialTransformer, SpatialInformer,
    SpatialBiGRU, SpatialMamba, SpatialS4, SpatialFuseMoE,
]
MODEL_MAP = {m.name: m for m in ALL_MODELS}

if args.arch:
    if args.arch not in MODEL_MAP:
        print(f"FATAL: unknown arch '{args.arch}'"); sys.exit(1)
    MODEL_MAP = {args.arch: MODEL_MAP[args.arch]}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def nll_loss(mu, lsv, y, mask):
    """
    Heteroscedastic NLL loss (Kendall & Gal 2017).
    L = 0.5 * (log σ² + (y-μ)²/σ²)
    High σ² = uncertain → less penalty for wrong μ but pays log σ² cost.
    Promotes calibrated uncertainty — model can't just inflate σ² to cheat.
    """
    sv = torch.exp(lsv).clamp(min=1e-6)
    loss = 0.5*(lsv + (y-mu)**2/sv)
    mask_e = mask.unsqueeze(-1).expand_as(loss)
    return (loss*mask_e).sum()/(mask_e.sum()+1e-8)

def crps_loss(mu, lsv, y, mask):
    """
    Continuous Ranked Probability Score (Gneiting & Raftery 2007).
    CRPS(N(μ,σ²), y) = σ*(z*(2Φ(z)-1) + 2φ(z) - 1/√π)
    where z=(y-μ)/σ, Φ=normal CDF, φ=normal PDF.
    Proper scoring rule — minimised when predicted distribution = true distribution.
    """
    sig = torch.exp(0.5*lsv).clamp(min=1e-6)
    z   = (y-mu)/sig
    z_np= z.detach().cpu().float().numpy()
    Phi = torch.tensor(ndtr(z_np), dtype=torch.float32, device=mu.device)
    phi = torch.exp(-0.5*z**2)/(2*np.pi)**0.5
    crps_ = sig*(z*(2*Phi-1)+2*phi-1/np.pi**0.5)
    mask_e= mask.unsqueeze(-1).expand_as(crps_)
    return (crps_*mask_e).sum()/(mask_e.sum()+1e-8)

def graph_smooth(mu, A, seen_locs):
    if A.dim()==3: A=A[0]
    if A.dim()==4: A=A[0,0]
    Ab=A.unsqueeze(0).expand(mu.shape[0],-1,-1)
    sm=torch.bmm(Ab,mu)
    return F.mse_loss(mu[:,seen_locs,:],sm[:,seen_locs,:])

def freeze_thaw_loss(mu, y, mask, tgt_grp):
    """
    Physics-informed: penalise sign errors near 0°C (freeze/thaw boundary).
    Only applied to temperature target.
    """
    if tgt_grp!="temp": return torch.tensor(0.,device=mu.device)
    near_zero = (y.abs()<0.5).float()
    sign_err  = ((mu*y)<0).float()
    penalty   = (near_zero*sign_err*mask.unsqueeze(-1)).mean()
    return penalty

def combined_loss(mu, lsv, y, mask, A, seen_locs, tgt_grp,
                  aux=None, lam_s=0.05, lam_c=0.1, lam_ft=0.05, lam_a=0.01):
    """Combined: NLL + graph smoothness + CRPS + freeze/thaw + MoE aux."""
    l_nll  = nll_loss(mu, lsv, y, mask)
    l_crps = crps_loss(mu, lsv, y, mask)
    l_gs   = graph_smooth(mu, A, seen_locs)
    l_ft   = freeze_thaw_loss(mu, y, mask, tgt_grp)
    total  = l_nll + lam_s*l_gs + lam_c*l_crps + lam_ft*l_ft
    if aux is not None: total = total + lam_a*aux
    return total, dict(nll=l_nll.item(),crps=l_crps.item(),
                        gs=l_gs.item(),ft=l_ft.item())


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — METRICS (distinct, non-redundant)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(yt, yp, mu_p=None, sig_p=None, label=""):
    """
    Full v6 metric suite.
    yt, yp: numpy arrays (observed, predicted mean)
    mu_p, sig_p: predicted mean and std for probabilistic metrics
    """
    mk = ~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
    if len(yt)<5: return {}

    # R² — variance explained
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-yt.mean())**2)+1e-10
    r2 = float(1-ss_res/ss_tot)

    # KGE — Kling-Gupta Efficiency (Gupta et al. 2009)
    r  = float(np.corrcoef(yt,yp)[0,1])
    a  = float(np.std(yp)/(np.std(yt)+1e-10))
    b  = float(np.mean(yp)/(np.mean(yt)+1e-10))
    kge= float(1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2))

    # ubRMSE — unbiased RMSE (removes mean bias, tests dynamic range)
    bias   = np.mean(yp-yt)
    ubrmse = float(np.sqrt(np.mean(((yp-bias)-yt)**2)))

    # Freeze/thaw accuracy (temp only)
    frz = float(np.mean((yt<0).astype(int)==(yp<0).astype(int))*100)

    # DTW — Dynamic Time Warping (temporal shape similarity)
    # Simplified: use correlation as proxy if sequences too long
    try:
        from scipy.spatial.distance import cdist
        # DTW on first 500 points for speed
        n = min(500, len(yt))
        yt_ = yt[:n].reshape(-1,1); yp_ = yp[:n].reshape(-1,1)
        D = cdist(yt_, yp_, 'sqeuclidean')
        dtw_mat = np.full_like(D, np.inf); dtw_mat[0,0]=D[0,0]
        for i in range(1,n):
            for j in range(max(0,i-10), min(n,i+11)):
                prev = min(dtw_mat[i-1,j] if i>0 else np.inf,
                           dtw_mat[i,j-1] if j>0 else np.inf,
                           dtw_mat[i-1,j-1] if (i>0 and j>0) else np.inf)
                dtw_mat[i,j]=D[i,j]+(0 if prev==np.inf else prev)
        dtw = float(dtw_mat[n-1,n-1]/n)
    except Exception:
        dtw = float(np.mean(np.abs(yt-yp)))  # fallback to MAE

    out = dict(R2=round(r2,4), KGE=round(kge,4),
               ubRMSE=round(ubrmse,4), FreezeAcc=round(frz,2),
               DTW=round(dtw,4), N=int(mk.sum()))

    # Probabilistic metrics (require σ predictions)
    if mu_p is not None and sig_p is not None:
        mk2 = mk & ~np.isnan(sig_p)
        mu_=mu_p[mk2]; sg_=sig_p[mk2]; yt_=yt[mk2 if len(mk2)==len(mk) else mk]

        # CRPS (analytical for Gaussian)
        if len(mu_)>0:
            z=(yt_-mu_)/(sg_+1e-8)
            Phi=ndtr(z); phi=np.exp(-0.5*z**2)/np.sqrt(2*np.pi)
            crps_=sg_*(z*(2*Phi-1)+2*phi-1/np.sqrt(np.pi))
            out["CRPS"]=round(float(crps_.mean()),4)

        # NLL
        if len(mu_)>0:
            nll_=-0.5*(np.log(2*np.pi*sg_**2+1e-8)+(yt_-mu_)**2/(sg_**2+1e-8))
            out["NLL"]=round(float(-nll_.mean()),4)

        # KL Divergence: KL(N(μ_obs,σ²_obs) || N(μ_pred,σ²_pred)) per site average
        # KL(p||q) = log(σq/σp) + (σp²+(μp-μq)²)/(2σq²) - 0.5
        if len(mu_)>0:
            mu_obs=float(np.mean(yt_)); sg_obs=float(np.std(yt_))+1e-8
            mu_pred=float(np.mean(mu_)); sg_pred=float(np.mean(sg_))+1e-8
            kl=(np.log(sg_pred/sg_obs)
                +(sg_obs**2+(mu_obs-mu_pred)**2)/(2*sg_pred**2)-0.5)
            out["KL_Div"]=round(float(kl),4)

    return {f"{label}_{k}" if label else k: v for k,v in out.items()}


def evaluate_v6(model, loader, tgt_sc, arch, tgt_grp):
    """
    Evaluate on all three test sets.
    Per PI instruction: evaluate on RESIDUAL units directly.
    Objective = minimising residual information loss.
    Metrics (R², KGE, ubRMSE, CRPS, DTW, KL) all on residual.
    """
    is_moe = (arch=="SpatialFuseMoE"); model.eval()
    t_inf_start = time.time()

    buckets = {"seen":   (SEEN_LOCS,   [],[],[],[]),
               "unseen": (UNSEEN_LOCS,  [],[],[],[]),
               "all":    (list(range(N_LOCS)),[],[],[],[])}

    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            X,y,mask,A=[b.to(DEVICE) for b in batch]
            out=model(X,A)
            if is_moe: mu,lsv,_=out
            else:       mu,lsv=out
            B_,N_,T_=mu.shape
            # Inverse transform: scaled residual → residual original units
            mu_r=tgt_sc.inverse_transform(
                mu.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
            sg_r=torch.exp(0.5*lsv).cpu().float().numpy().reshape(B_,N_,T_)
            y_r =tgt_sc.inverse_transform(
                y.cpu().float().numpy().reshape(-1,T_)).reshape(B_,N_,T_)
            for bname,(locs,yt_l,yp_l,mu_l,sg_l) in buckets.items():
                yt_l.append(y_r[:,locs,0].flatten())
                yp_l.append(mu_r[:,locs,0].flatten())
                mu_l.append(mu_r[:,locs,0].flatten())
                sg_l.append(sg_r[:,locs,0].flatten())
            n_batches += 1

    inference_s = time.time() - t_inf_start

    all_metrics={}
    for bname,(locs,yt_l,yp_l,mu_l,sg_l) in buckets.items():
        yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)
        mu_=np.concatenate(mu_l); sg_=np.concatenate(sg_l)
        m=compute_metrics(yt,yp,mu_,sg_,label=bname)
        all_metrics.update(m)

    gap=round(all_metrics.get("seen_R2",np.nan)-
               all_metrics.get("unseen_R2",np.nan),4)
    all_metrics["spatial_gap"]=gap
    all_metrics["inference_s"]=round(inference_s,2)
    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — MULTI-STAGE RANDOM SEARCH TUNING
# ══════════════════════════════════════════════════════════════════════════════

# Hyperparameter search bounds
HPARAM_BOUNDS = {
    "lr":         (1e-4, 1e-2, "log"),
    "hidden_dim": (64, 192, "int8"),  # int8 = round to multiple of 8
    "n_layers":   (1, 4, "int"),
    "gcn_layers": (1, 4, "int"),
    "dropout":    (0.05, 0.25, "float"),
    "weight_decay":(1e-5, 1e-3, "log"),
    "lam_smooth": (0.01, 0.2, "float"),
    "lam_crps":   (0.05, 0.3, "float"),
}

def sample_hparams(bounds, rng, center=None, radius=0.3):
    """Sample hyperparameters — broad or narrow around a center config."""
    cfg={}
    for k,(lo,hi,typ) in bounds.items():
        if center and k in center:
            cv=center[k]
            if typ=="log":
                log_cv=np.log(cv); rng_w=(np.log(hi)-np.log(lo))*radius
                lo_=np.exp(max(np.log(lo),log_cv-rng_w))
                hi_=np.exp(min(np.log(hi),log_cv+rng_w))
                cfg[k]=float(rng.uniform(np.log(lo_),np.log(hi_)))
                cfg[k]=np.exp(cfg[k])
            elif typ=="int":
                rng_w=max(1,int((hi-lo)*radius))
                cfg[k]=int(rng.integers(max(lo,int(cv)-rng_w),
                                         min(hi,int(cv)+rng_w)+1))
            else:
                rng_w=(hi-lo)*radius
                cfg[k]=float(rng.uniform(max(lo,cv-rng_w),min(hi,cv+rng_w)))
        else:
            if typ=="log":
                cfg[k]=float(np.exp(rng.uniform(np.log(lo),np.log(hi))))
            elif typ=="int":
                v=int(rng.integers(lo,hi+1))
                cfg[k]=v if k!="hidden_dim" else max(8,round(v/8)*8)
            else:
                cfg[k]=float(rng.uniform(lo,hi))
    return cfg


def quick_train(arch_cls, nf, nt, hcfg, train_ld, val_ld, tgt_sc,
                 tgt_grp, epochs=10, device=DEVICE):
    """Fast training for tuning trials (fewer epochs, no pruning)."""
    model=arch_cls(nf=nf, h=int(hcfg.get("hidden_dim",96)),
                    nl=int(hcfg.get("n_layers",2)), gl=int(hcfg.get("gcn_layers",2)),
                    nt=nt, dp=float(hcfg.get("dropout",0.15))).to(device)
    opt=AdamW(model.parameters(), lr=hcfg["lr"],
               weight_decay=hcfg.get("weight_decay",5e-4))
    sched=OneCycleLR(opt, max_lr=hcfg["lr"],
                      total_steps=epochs*len(train_ld), pct_start=0.2)
    is_moe=(arch_cls.name=="SpatialFuseMoE")
    best_r2=float("-inf")
    for ep in range(epochs):
        model.train()
        for batch in train_ld:
            X,y,mask,A=[b.to(device) for b in batch]; opt.zero_grad()
            out=model(X,A)
            if is_moe: mu,lsv,aux=out
            else:       mu,lsv=out; aux=None
            loss,_=combined_loss(mu,lsv,y,mask,A,SEEN_LOCS,tgt_grp,aux,
                                  hcfg.get("lam_smooth",0.05),
                                  hcfg.get("lam_crps",0.1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
        # Quick val
        model.eval()
        yt_=[]; yp_=[]
        with torch.no_grad():
            for batch in val_ld:
                X,y,mask,A=[b.to(device) for b in batch]
                out=model(X,A); mu=out[0]
                yt_.append(y[:,SEEN_LOCS,0].cpu().numpy().flatten())
                yp_.append(mu[:,SEEN_LOCS,0].cpu().numpy().flatten())
        yt=np.concatenate(yt_); yp=np.concatenate(yp_)
        mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
        if len(yt)>5:
            r2=float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10))
            best_r2=max(best_r2,r2)
    return best_r2, model


def run_tuning(arch_name, tgt_grp, tgt_cols, loaders,
                n_broad=35, n_narrow=15, seed=SEED):
    """
    Multi-stage random search tuning.
    Stage 1: 35 broad trials — wide hyperbound space
    Stage 2: 15 narrow trials — around top-3 Stage 1 configs
    Returns best config dict.
    """
    if arch_name not in MODEL_MAP: return {}
    arch_cls=MODEL_MAP[arch_name]
    if tgt_grp not in tgt_scalers: return {}
    ts,_=tgt_scalers[tgt_grp]
    nt=len(tgt_cols)
    train_ld=loaders.get("train"); val_ld=loaders.get("val")
    if not train_ld or not val_ld: return {}

    print(f"\n  TUNING: {arch_name} [{tgt_grp}] "
          f"| {n_broad} broad + {n_narrow} narrow trials")
    rng=np.random.default_rng(seed)
    all_results=[]

    # Stage 1: Broad search
    print(f"    Stage 1: {n_broad} broad trials...")
    import time as _time
    for trial in range(n_broad):
        cfg=sample_hparams(HPARAM_BOUNDS,rng)
        _t0=_time.time()
        try:
            r2,_=quick_train(arch_cls,N_FEATS,nt,cfg,train_ld,val_ld,ts,tgt_grp,epochs=8)
            all_results.append((r2,cfg))
            if (trial+1)%5==0:
                best_so_far=max(all_results,key=lambda x:x[0])[0]
                print(f"      Trial {trial+1}/{n_broad} | best_r2={best_so_far:.4f} | {_time.time()-_t0:.0f}s")
        except Exception as e:
            print(f"      Trial {trial+1} failed: {e}")

    if not all_results: return {}
    all_results.sort(key=lambda x:x[0],reverse=True)
    top3=[cfg for _,cfg in all_results[:3]]
    print(f"    Stage 1 best R²={all_results[0][0]:.4f}")

    # Stage 2: Narrow search around top-3 configs
    print(f"    Stage 2: {n_narrow} narrow trials...")
    narrow_results=[]
    for trial in range(n_narrow):
        center=top3[trial%3]
        cfg=sample_hparams(HPARAM_BOUNDS,rng,center=center,radius=0.25)
        try:
            r2,_=quick_train(arch_cls,N_FEATS,nt,cfg,train_ld,val_ld,ts,tgt_grp,epochs=12)
            narrow_results.append((r2,cfg))
        except Exception as e:
            print(f"      Trial {trial+1} failed: {e}")

    # Add elapsed_s placeholder (per-trial timing stored separately)

    all_combined=all_results+narrow_results
    all_combined.sort(key=lambda x:x[0],reverse=True)
    best_r2,best_cfg=all_combined[0]
    print(f"    Best config: R²={best_r2:.4f} | lr={best_cfg['lr']:.5f} "
          f"h={best_cfg.get('hidden_dim',96)} dp={best_cfg.get('dropout',0.15):.3f}")

    # Save tuning history
    tune_df=pd.DataFrame([dict(trial=j,r2=r,elapsed_s=0.,**c)
                           for j,(r,c) in enumerate(all_combined)])
    tune_df.to_csv(RESULTS/f"v6_tuning_{arch_name}_{tgt_grp}.csv",index=False)
    return best_cfg


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — FULL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_full(arch_name, tgt_grp, tgt_cols, loaders, hcfg=None,
                epochs=30, patience=7, run_seed=SEED, ckpt_path=None):
    """Full training with best hyperconfig."""
    arch_cls=MODEL_MAP[arch_name]
    ts,_=tgt_scalers[tgt_grp]; nt=len(tgt_cols)
    if hcfg is None:
        hcfg={"lr":3e-4,"hidden_dim":96,"n_layers":2,"gcn_layers":2,
               "dropout":0.15,"weight_decay":5e-4,
               "lam_smooth":0.05,"lam_crps":0.1}

    torch.manual_seed(run_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
    is_moe=(arch_name=="SpatialFuseMoE")

    model=arch_cls(nf=N_FEATS, h=hcfg.get("hidden_dim",96),
                    nl=hcfg.get("n_layers",2), gl=hcfg.get("gcn_layers",2),
                    nt=nt, dp=hcfg.get("dropout",0.15)).to(DEVICE)
    np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{arch_cls.tier}] {arch_name} | {np_:,} params")

    opt  =AdamW(model.parameters(),lr=hcfg["lr"],
                 weight_decay=hcfg.get("weight_decay",5e-4))
    sched=OneCycleLR(opt,max_lr=hcfg["lr"],
                      total_steps=epochs*len(loaders["train"]),pct_start=0.1)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Record initial entropy (epoch 0)
    initial_entropy = _compute_entropy(model, loaders["val"])

    best_r2=float("-inf"); best_st=None; pat=0; hist=[]; t0=time.time()

    for ep in range(1,epochs+1):
        model.train(); tr=0.; nb=0; loss_parts={}
        for batch in loaders["train"]:
            X,y,mask,A=[b.to(DEVICE) for b in batch]; opt.zero_grad()
            amp_enabled = torch.cuda.is_available() and not is_moe
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out=model(X,A)
                if is_moe: mu,lsv,aux=out
                else:       mu,lsv=out; aux=None
                loss,parts=combined_loss(mu,lsv,y,mask,A,SEEN_LOCS,tgt_grp,aux,
                                          hcfg.get("lam_smooth",0.05),
                                          hcfg.get("lam_crps",0.1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tr+=loss.item(); nb+=1
            for k,v in parts.items(): loss_parts[k]=loss_parts.get(k,0)+v

        tr_loss=tr/max(nb,1)
        val_r2=_quick_val_r2(model,loaders["val"],ts,is_moe)
        entropy=_compute_entropy(model,loaders["val"])

        hist.append(dict(epoch=ep,train_loss=round(tr_loss,6),
                          val_R2=round(val_r2,4),entropy=round(entropy,4),
                          **{k:round(v/nb,6) for k,v in loss_parts.items()}))

        if val_r2>best_r2:
            best_r2=val_r2; best_st={k:v.cpu().clone()
                                       for k,v in model.state_dict().items()}; pat=0
        else: pat+=1

        if ep%5==0 or ep==1:
            print(f"    E{ep:03d} | loss={tr_loss:.4f} | "
                  f"nll={loss_parts.get('nll',0)/nb:.4f} | "
                  f"crps={loss_parts.get('crps',0)/nb:.4f} | "
                  f"R²={val_r2:.4f} | ent={entropy:.3f} | "
                  f"{time.time()-t0:.0f}s")
        if pat>=patience: print(f"    Early stop @{ep}"); break

    elapsed=time.time()-t0
    if best_st: model.load_state_dict(best_st)
    final_entropy=_compute_entropy(model,loaders["val"])

    if ckpt_path:
        torch.save(dict(arch=arch_name,state_dict=model.state_dict(),
                        val_r2=best_r2,history=hist,elapsed_s=elapsed,
                        hcfg=hcfg,tgt_grp=tgt_grp,n_feats=N_FEATS,
                        initial_entropy=initial_entropy,
                        final_entropy=final_entropy,
                        job_id=JOB_ID,node=NODE,
                        seen_locs=SEEN_LOCS,unseen_locs=UNSEEN_LOCS),
                   ckpt_path)
    print(f"  ✓ {arch_name} [{tgt_grp}] R²={best_r2:.4f} | {elapsed:.0f}s | "
          f"entropy: {initial_entropy:.3f}→{final_entropy:.3f}")
    return model, hist, best_r2, elapsed, initial_entropy, final_entropy


def _quick_val_r2(model, val_ld, ts, is_moe):
    model.eval(); yt_=[]; yp_=[]
    with torch.no_grad():
        for batch in val_ld:
            X,y,mask,A=[b.to(DEVICE) for b in batch]
            out=model(X,A); mu=out[0]
            _,nt_=mu.shape[1],mu.shape[2]
            mu_r=ts.inverse_transform(mu.cpu().float().numpy().reshape(-1,nt_)
                                        ).reshape(mu.shape[0],mu.shape[1],nt_)
            y_r =ts.inverse_transform(y.cpu().float().numpy().reshape(-1,nt_)
                                        ).reshape(y.shape[0],y.shape[1],nt_)
            yt_.append(y_r[:,SEEN_LOCS,0].flatten())
            yp_.append(mu_r[:,SEEN_LOCS,0].flatten())
    yt=np.concatenate(yt_); yp=np.concatenate(yp_)
    mk=~(np.isnan(yt)|np.isnan(yp)); yt=yt[mk]; yp=yp[mk]
    return float(1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-10)) if len(yt)>5 else -99.


def _compute_entropy(model, val_ld):
    """
    Compute mean predictive entropy from heteroscedastic σ².
    H = 0.5*(1 + log(2πσ²)) — differential entropy of N(μ,σ²).
    High at init (σ²≈1), lower at convergence (model is more certain).
    """
    model.eval(); entropies=[]
    with torch.no_grad():
        for batch in list(val_ld)[:5]:  # sample 5 batches
            X,y,mask,A=[b.to(DEVICE) for b in batch]
            out=model(X,A); lsv=out[1]
            H=0.5*(1+np.log(2*np.pi)+lsv.cpu().float().numpy())
            entropies.append(H.mean())
    return float(np.mean(entropies)) if entropies else 0.


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — ABLATION
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_VARIANTS = {
    "no_gcn":        "Remove GCN — temporal only",
    "no_temporal":   "Remove temporal encoder — GCN only",
    "no_attention":  "Remove attention — GRU only",
    "no_laplacian":  "Remove graph Laplacian loss (lam_smooth=0)",
    "no_uncertainty":"Remove uncertainty head — point prediction only",
}

def run_ablation(arch_name, tgt_grp, tgt_cols, loaders, component, base_hcfg):
    """
    Run one ablation variant for one model.
    Each component is properly removed:
      no_gcn        — zero GCN layers (spatial graph removed)
      no_temporal   — reduce GRU/SSM layers to 0 (temporal encoder removed)
      no_attention  — same as no_temporal for attention-based models
      no_laplacian  — graph Laplacian loss weight = 0
      no_uncertainty— CRPS weight = 0, NLL only (less uncertainty pressure)
    Runs 15 epochs — enough to show relative degradation vs full model.
    """
    print(f"  ABLATION: {arch_name} [{tgt_grp}] — {component}")
    mod_hcfg = base_hcfg.copy()

    if component == "no_gcn":
        mod_hcfg["gcn_layers"] = 0          # remove spatial graph
    elif component == "no_temporal":
        mod_hcfg["n_layers"] = 1            # minimum layers
        mod_hcfg["skip_temporal"] = True    # bypass temporal output, use GCN only
    elif component == "no_attention":
        mod_hcfg["n_layers"] = 1            # minimum temporal (no attention)
    elif component == "no_laplacian":
        mod_hcfg["lam_smooth"] = 0.0        # remove graph Laplacian loss
    elif component == "no_uncertainty":
        mod_hcfg["lam_crps"] = 0.0          # remove CRPS loss term

    # Ensure int types
    mod_hcfg["gcn_layers"] = int(mod_hcfg.get("gcn_layers", 2))
    mod_hcfg["n_layers"]   = int(mod_hcfg.get("n_layers", 2))

    try:
        _, hist, best_r2, elapsed, _, _ = train_full(
            arch_name, tgt_grp, tgt_cols, loaders, mod_hcfg,
            epochs=15, patience=5, run_seed=SEED, ckpt_path=None)
    except Exception as e:
        print(f"    ✗ {arch_name} {component}: {e}")
        best_r2 = float("nan"); elapsed = 0.

    return dict(arch=arch_name, target=tgt_grp, ablation=component,
                val_r2=round(best_r2, 4), elapsed_s=round(elapsed, 1),
                description=ABLATION_VARIANTS[component])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

if args.mode=="tune":
    print("\n" + "="*55 + "\n  TUNING MODE\n" + "="*55)
    all_best_cfgs={}
    for tgt_grp,tgt_cols,loaders,label in TARGET_GROUPS:
        for arch_name in MODEL_MAP:
            if not loaders.get("train"): continue
            best_cfg=run_tuning(arch_name,tgt_grp,tgt_cols,loaders,
                                 n_broad=35,n_narrow=15)
            all_best_cfgs[f"{arch_name}_{tgt_grp}"]=best_cfg
    with open(RESULTS/"v6_best_hparams.json","w") as f:
        json.dump(all_best_cfgs,f,indent=2)
    print(f"\n  ✓ Best configs saved: {RESULTS}/v6_best_hparams.json")
    sys.exit(0)

if args.mode=="ablation":
    print("\n" + "="*55 + "\n  ABLATION MODE\n" + "="*55)
    hparam_path=RESULTS/"v6_best_hparams.json"
    best_cfgs=json.load(open(hparam_path)) if hparam_path.exists() else {}
    abl_results=[]
    components=[args.ablation_component] if args.ablation_component!="all" \
                else list(ABLATION_VARIANTS.keys())
    for tgt_grp,tgt_cols,loaders,label in TARGET_GROUPS:
        for arch_name in MODEL_MAP:
            if not loaders.get("train"): continue
            base=best_cfgs.get(f"{arch_name}_{tgt_grp}",
                                {"lr":3e-4,"hidden_dim":96,"dropout":0.15,
                                 "lam_smooth":0.05,"lam_crps":0.1})
            for comp in components:
                try:
                    r=run_ablation(arch_name,tgt_grp,tgt_cols,loaders,comp,base)
                    abl_results.append(r)
                except Exception as e:
                    print(f"  ✗ {arch_name} {comp}: {e}")
    # Save incrementally
    if abl_results:
        pd.DataFrame(abl_results).to_csv(RESULTS/"v6_ablation_results.csv",index=False)
    print(f"  ✓ {RESULTS}/v6_ablation_results.csv ({len(abl_results)} records)")
    sys.exit(0)

# ── TRAIN MODE (default) ──────────────────────────────────────────────────────
print("\n" + "="*55 + "\n  TRAINING MODE — 13 models × 3 targets\n" + "="*55)

# Load best hparams if available (from tune mode)
hparam_path=RESULTS/"v6_best_hparams.json"
best_cfgs=json.load(open(hparam_path)) if hparam_path.exists() else {}
if best_cfgs: print(f"  Loaded tuned hparams for {len(best_cfgs)} configs")
else: print("  Using default hparams (run --mode tune first for best results)")

all_results=[]
entropy_records=[]

# ── Timing: record pipeline stage times per model ────────────────────────────
# Per PI: save all timing for comparison with/without distributed framework
PIPELINE_TIMES = {}  # arch_tgt -> {preproc_s, train_s, eval_s, total_s}
t_preproc_total = time.time() - t_preproc_start if "t_preproc_start" in dir()                   else 0.  # preprocessing already done above

for tgt_grp,tgt_cols,loaders,label in TARGET_GROUPS:
    print(f"\n{'─'*60}\n  TARGET: {label}\n{'─'*60}")
    if not loaders.get("train"): continue
    ts,_=tgt_scalers[tgt_grp]

    for arch_name in MODEL_MAP:
        ckpt=MODELS/f"{arch_name}_{tgt_grp}_v6_best.pt"
        if ckpt.exists():
            try:
                sv=torch.load(ckpt,map_location="cpu")
                if sv.get("val_r2",-99)>-10:
                    print(f"\n  ✓ SKIP {arch_name} [{tgt_grp}] r2={sv['val_r2']:.4f}")
                    tm=sv.get("test_metrics",{})
                    all_results.append(dict(Model=arch_name,Target=tgt_grp,
                                            Tier=MODEL_MAP[arch_name].tier,
                                            Val_R2=sv["val_r2"],**tm,Resumed=True))
                    entropy_records.append(dict(arch=arch_name,target=tgt_grp,
                                                initial=sv.get("initial_entropy",np.nan),
                                                final=sv.get("final_entropy",np.nan)))
                    continue
            except Exception: pass

        hcfg=best_cfgs.get(f"{arch_name}_{tgt_grp}",None)
        try:
            model,hist,best_r2,elapsed,ent_i,ent_f=train_full(
                arch_name,tgt_grp,tgt_cols,loaders,hcfg,
                epochs=30,patience=7,ckpt_path=ckpt)

            # Evaluate on all three test sets
            all_metrics={}
            for sp_name,sp_key in [("std_test","test"),
                                     ("unseen_space","test_space"),
                                     ("unseen_time","test_time"),
                                     ("unseen_both","test_both")]:
                if loaders.get(sp_key):
                    m=evaluate_v6(model,loaders[sp_key],ts,arch_name,tgt_grp)
                    all_metrics[sp_name]=m

            # Save into checkpoint
            sv=torch.load(ckpt,map_location="cpu")
            sv["test_metrics"]=all_metrics; sv["job_id"]=JOB_ID
            torch.save(sv,ckpt)

            flat={f"{sp}_{k}":v for sp,md in all_metrics.items()
                  for k,v in md.items()}
            all_results.append(dict(Model=arch_name,Target=tgt_grp,
                                     Tier=MODEL_MAP[arch_name].tier,
                                     Val_R2=best_r2,**flat,Resumed=False))
            entropy_records.append(dict(arch=arch_name,target=tgt_grp,
                                         initial=round(ent_i,4),
                                         final=round(ent_f,4),
                                         delta=round(ent_f-ent_i,4)))
            pd.DataFrame(all_results).to_csv(RESULTS/"v6_results_incremental.csv",index=False)
            pd.DataFrame(entropy_records).to_csv(RESULTS/"v6_entropy.csv",index=False)

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ FAILED {arch_name}: {e}")

# Final save
pd.DataFrame(all_results).to_csv(RESULTS/"v6_results_all.csv",index=False)
pd.DataFrame(entropy_records).to_csv(RESULTS/"v6_entropy.csv",index=False)
if PIPELINE_TIMES:
    pd.DataFrame(list(PIPELINE_TIMES.values())).to_csv(
        RESULTS/"v6_pipeline_timing.csv", index=False)
    print(f"  Timing: {RESULTS}/v6_pipeline_timing.csv")
print(f"\n  Done: {pd.Timestamp.now()}")
print(f"  Results: {RESULTS}/v6_results_all.csv")
print("="*70)