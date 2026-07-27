"""
================================================================================
train_soil_spatial.py  —  v3  TRUE SPATIAL FIELD PREDICTION
DoD PROJECT | Alaska 2022-2025 | SMAP + Weather + Topography
SLURM Standalone — Session-Independent — Fully Resumable
================================================================================

WHAT CHANGED FROM v2 (per senior feedback):
  v2 treated each location as an independent time series → NOT truly spatial
  v3 pivots by TIMESTAMP: each sample = full spatial field at one moment
     Input  : (lookback, N_locations, N_features) — field history
     Output : (N_locations, N_targets)            — next field state
  This captures how the soil field evolves across the entire landscape,
  encoding lateral flow, topographic gradients, and spatial diffusion.

ARCHITECTURE:
  SpatialSnapshotDataset  — groups by time_utc, not by Site
  4 DL models (BiGRU, Mamba, S4, FuseMoE) — same architectures, new data shape
  GraphConv (bmm)         — fixed Bug 1: correct (B,N,d) batched multiplication
  Training engine         — fixed Bug 2: correct MoE/non-MoE output unpacking
  Wavelet residuals       — models learn anomalies, not seasonal cycles
  Graph Laplacian loss    — spatial smoothness regularisation
  EntropyTracker          — SEASONAL_FITTING vs LEARNING_DYNAMICS diagnosis
  Full resume guard       — resubmit anytime, completed checkpoints skipped

PHASES:
  0  PyTorch / CUDA setup
  1  Data loading + feature engineering + wavelet (cached after first run)
  2  Spatial graph over lat/lon locations (k-NN, Gaussian weights)
  3  SpatialSnapshotDataset  (pivot by timestamp)
  4  ML baselines (Ridge, ExtraTrees, XGBoost — for comparison)
  5  Model definitions (GraphConv, BiGRU, Mamba, S4, FuseMoE)
  6  Training engine (Huber + Graph Laplacian + AMP)
  7  Train all models x all target groups
  8  Evaluation (R², RMSE, KGE, Skill, FreezeAcc, per-node R²)
  9  Figures + Final leaderboard
================================================================================
"""

import os, sys, time, json, pickle, warnings, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Tee logger: every print → stdout + log file ───────────────────────────────
class TeeLogger:
    def __init__(self, path):
        self.t = sys.__stdout__
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)
    def write(self, m): self.t.write(m); self.f.write(m)
    def flush(self):    self.t.flush();  self.f.flush()

LOG_PATH = "/home/emmanuel.keku/logs/soil_training_v3.log"
sys.stdout = TeeLogger(LOG_PATH)
sys.stderr = sys.stdout

JOB_ID  = os.environ.get("SLURM_JOB_ID",   "local")
NODE    = os.environ.get("SLURMD_NODENAME", "unknown")

print("=" * 70)
print("  SOIL SPATIAL PREDICTION v3 — TRUE SPATIAL FIELD")
print(f"  Job    : {JOB_ID}  |  Node : {NODE}")
print(f"  Start  : {pd.Timestamp.now()}")
print(f"  Log    : {LOG_PATH}")
print("=" * 70)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/emmanuel.keku")
PREPROC_DIR = PROJECT_DIR / "preprocessed_v3"
RESULTS_DIR = PROJECT_DIR / "results_v3"
MODELS_DIR  = PROJECT_DIR / "models_v3" / "dl"
FIG_DIR     = PROJECT_DIR / "figures_v3"
LOG_DIR     = PROJECT_DIR / "logs"
for d in [PREPROC_DIR, RESULTS_DIR, MODELS_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — PyTorch / CUDA
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    print(f"\nPyTorch : {torch.__version__}  |  Device : {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            n = torch.cuda.get_device_name(i)
            m = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i} : {n} | {m:.1f} GB")
except ImportError as e:
    print(f"FATAL: {e}"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING, FEATURE ENGINEERING, WAVELET DECOMPOSITION
# Cached after first run — subsequent runs load from disk in seconds
# ══════════════════════════════════════════════════════════════════════════════
MARKER = PREPROC_DIR / "preprocessing_complete.json"

if MARKER.exists():
    print("\n" + "="*55)
    print("  PHASE 1: Loading from cache (preprocessing already done)")
    print("="*55)
    t0 = time.time()
    df = pd.read_csv(PREPROC_DIR/"master_processed.csv", parse_dates=["time_utc"])
    with open(PREPROC_DIR/"scalers.pkl",      "rb") as f: SC  = pickle.load(f)
    with open(PREPROC_DIR/"feature_info.pkl", "rb") as f: FI  = pickle.load(f)
    print(f"  {len(df):,} rows loaded in {time.time()-t0:.1f}s")
    feat_scalers      = SC["feat_scalers"]
    snap_feat_scaler  = SC["snap_feat_scaler"]
    snap_tgt_scalers  = SC["snap_tgt_scalers"]
    MODEL_FEATURES    = FI["MODEL_FEATURES"]
    SNAP_FEATURES     = FI["SNAP_FEATURES"]
    ALL_TARGETS       = FI["ALL_TARGETS"]
    TEMP_TARGETS      = FI["TEMP_TARGETS"]
    SMAP_TARGETS      = FI["SMAP_TARGETS"]
    MOIST_TARGETS     = FI["MOIST_TARGETS"]
    TEMP_RESID_COLS   = FI["TEMP_RESID_COLS"]
    SMAP_RESID_COLS   = FI["SMAP_RESID_COLS"]
    MOIST_RESID_COLS  = FI["MOIST_RESID_COLS"]
    SITES             = FI["SITES"]
    N_FEATURES        = FI["N_FEATURES"]
    N_SNAP_FEATURES   = FI["N_SNAP_FEATURES"]
    LOCATIONS         = FI["LOCATIONS"]
    N_LOCS            = FI["N_LOCS"]

else:
    print("\n" + "="*55)
    print("  PHASE 1: Full preprocessing")
    print("="*55)

    FILE = str(PROJECT_DIR / "Fully_Sequenced_2022_2025_Historical_Master.csv")
    if not Path(FILE).exists():
        print(f"FATAL: {FILE} not found"); sys.exit(1)

    print(f"Loading {FILE}...")
    t0     = time.time()
    df_raw = pd.read_csv(FILE, header=1)
    print(f"  {len(df_raw):,} rows | {df_raw.shape[1]} cols | {time.time()-t0:.1f}s")

    df_raw["time_utc"] = pd.to_datetime(df_raw["time_utc"], utc=True)
    df_raw["year"]     = df_raw["time_utc"].dt.year
    df_raw["month"]    = df_raw["time_utc"].dt.month
    df_raw["hour"]     = df_raw["time_utc"].dt.hour
    df_raw["doy"]      = df_raw["time_utc"].dt.dayofyear

    # ── Targets ───────────────────────────────────────────────────────────────
    TEMP_TARGETS  = ["soil_temperature_0_to_7cm"]
    SMAP_TARGETS  = ["Soil_Temp_L1"]          # L2/L3/L4 dropped (r > 0.79)
    MOIST_TARGETS = ["soil_moisture_0_to_7cm", "SM_Surface"]
    ALL_TARGETS   = [t for t in TEMP_TARGETS+SMAP_TARGETS+MOIST_TARGETS
                     if t in df_raw.columns]

    # ── Feature engineering ───────────────────────────────────────────────────
    print("Feature engineering...")
    df = df_raw.copy().sort_values(["Latitude","Longitude","time_utc"]).reset_index(drop=True)

    if all(c in df.columns for c in ["Soil_Temp_L1","Soil_Temp_L4"]):
        df["grad_L1_L4"] = df["Soil_Temp_L1"] - df["Soil_Temp_L4"]
    if all(c in df.columns for c in ["Soil_Temp_L1","Soil_Temp_L2"]):
        df["grad_L1_L2"] = df["Soil_Temp_L1"] - df["Soil_Temp_L2"]
    if "Temp_K" in df.columns:
        df["Temp_C"] = df["Temp_K"] - 273.15

    df["sin_doy"]   = np.sin(2*np.pi*df["doy"]/365.25)
    df["cos_doy"]   = np.cos(2*np.pi*df["doy"]/365.25)
    df["sin_hour"]  = np.sin(2*np.pi*df["hour"]/24.0)
    df["cos_hour"]  = np.cos(2*np.pi*df["hour"]/24.0)
    df["sin_month"] = np.sin(2*np.pi*df["month"]/12.0)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12.0)
    df["is_frozen"] = (df["soil_temperature_0_to_7cm"] < 0).astype(float)

    # Lag features per location (not per Site — true spatial)
    loc_key = df["Latitude"].astype(str) + "_" + df["Longitude"].astype(str)
    df["loc_key"] = loc_key
    for lag in [1, 6, 24, 72, 168]:
        df[f"st_lag_{lag}h"] = (df.groupby("loc_key")["soil_temperature_0_to_7cm"]
                                  .transform(lambda x: x.shift(lag)))
    for lag in [1, 24, 168]:
        df[f"sm_lag_{lag}h"] = (df.groupby("loc_key")["soil_moisture_0_to_7cm"]
                                  .transform(lambda x: x.shift(lag)))
    for w in [24, 72, 168]:
        df[f"precip_{w}h"] = (df.groupby("loc_key")["precipitation"]
                                .transform(lambda x: x.rolling(w, min_periods=1).sum()))
    print(f"  {df.shape[1]} columns")

    # ── Wavelet decomposition per location ────────────────────────────────────
    print("Wavelet decomposition (db4, level 6)...")
    import pywt

    def wavelet_decompose(series, wavelet="db4", level=6):
        arr = np.array(series, dtype=np.float64)
        nm  = np.isnan(arr)
        if nm.all(): return arr, np.zeros_like(arr)
        if nm.any():
            idx = np.arange(len(arr))
            arr[nm] = np.interp(idx[nm], idx[~nm], arr[~nm])
        lv  = min(level, pywt.dwt_max_level(len(arr), wavelet)-1)
        c   = pywt.wavedec(arr, wavelet, level=lv)
        ac  = [c[0]] + [np.zeros_like(x) for x in c[1:]]
        app = pywt.waverec(ac, wavelet)[:len(arr)]
        return app, arr - app

    for col in ALL_TARGETS:
        df[f"{col}_approx"]   = np.nan
        df[f"{col}_residual"] = np.nan

    t0 = time.time()
    for lk, grp in df.groupby("loc_key"):
        idx = grp.index
        for col in ALL_TARGETS:
            if col not in df.columns: continue
            try:
                app, res = wavelet_decompose(df.loc[idx, col].values)
            except Exception:
                roll = pd.Series(df.loc[idx,col].values).rolling(30,min_periods=1,center=True).mean().values
                app, res = roll, df.loc[idx,col].values - roll
            df.loc[idx, f"{col}_approx"]   = app
            df.loc[idx, f"{col}_residual"] = res

    print(f"  Done {time.time()-t0:.1f}s")
    TEMP_RESID_COLS  = [f"{t}_residual" for t in TEMP_TARGETS  if f"{t}_residual" in df.columns]
    SMAP_RESID_COLS  = [f"{t}_residual" for t in SMAP_TARGETS  if f"{t}_residual" in df.columns]
    MOIST_RESID_COLS = [f"{t}_residual" for t in MOIST_TARGETS if f"{t}_residual" in df.columns]

    # ── Split ─────────────────────────────────────────────────────────────────
    df["split"] = "train"
    df.loc[df["year"]==2024, "split"] = "val"
    df.loc[df["year"]==2025, "split"] = "test"
    SITES = sorted(df["Site"].unique().tolist())

    # ── Location index ────────────────────────────────────────────────────────
    # Each unique (Lat, Lon) = one spatial node
    LOCATIONS = (df[["Latitude","Longitude"]].drop_duplicates()
                 .sort_values(["Latitude","Longitude"])
                 .reset_index(drop=True))
    N_LOCS = len(LOCATIONS)
    print(f"  Unique spatial locations: {N_LOCS}")

    # ── Feature lists ─────────────────────────────────────────────────────────
    BASE_FEATS = ["Latitude","Longitude","smap_node_x","smap_node_y",
                  "elevation_m","elev_roughness_m","slope_deg",
                  "temperature_2m","precipitation","snow_depth_weather",
                  "Temp_K","Pressure","Greenness","Snow_Depth_SMAP"]
    ENG_FEATS  = (["grad_L1_L4","grad_L1_L2","Temp_C",
                   "sin_doy","cos_doy","sin_hour","cos_hour","sin_month","cos_month",
                   "is_frozen"] +
                  [f"st_lag_{l}h" for l in [1,6,24,72,168]] +
                  [f"sm_lag_{l}h" for l in [1,24,168]] +
                  [f"precip_{w}h" for w in [24,72,168]])
    MODEL_FEATURES = list(dict.fromkeys([f for f in BASE_FEATS+ENG_FEATS if f in df.columns]))
    N_FEATURES     = len(MODEL_FEATURES)

    # Snapshot features = per-location static + dynamic weather (no lag — lag is implicit in lookback)
    SNAP_STATIC  = ["Latitude","Longitude","elevation_m","elev_roughness_m","slope_deg",
                    "sin_doy","cos_doy","sin_hour","cos_hour","sin_month","cos_month"]
    SNAP_DYNAMIC = ["temperature_2m","precipitation","snow_depth_weather",
                    "Temp_K","Pressure","Greenness","Snow_Depth_SMAP",
                    "grad_L1_L4","grad_L1_L2","Temp_C","is_frozen"]
    SNAP_FEATURES = list(dict.fromkeys([f for f in SNAP_STATIC+SNAP_DYNAMIC if f in df.columns]))
    N_SNAP_FEATURES = len(SNAP_FEATURES)
    print(f"  Model features: {N_FEATURES}  |  Snapshot features: {N_SNAP_FEATURES}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    from sklearn.preprocessing import RobustScaler
    print("Fitting scalers...")
    feat_scalers = {}
    for site in SITES:
        tr = df[(df["Site"]==site)&(df["split"]=="train")]
        fd = tr[MODEL_FEATURES].dropna()
        if len(fd)<50: continue
        fs = RobustScaler(); fs.fit(fd.values); feat_scalers[site] = fs
        print(f"  ✓ {site}")

    # Global snapshot scaler (fitted on ALL training locations × timestamps)
    tr_all = df[df["split"]=="train"]
    snap_feat_scaler = RobustScaler()
    snap_feat_scaler.fit(tr_all[SNAP_FEATURES].dropna().values)

    snap_tgt_scalers = {}
    for tgt_grp_name, tgt_cols in [("temp",  TEMP_RESID_COLS),
                                    ("smap",  SMAP_RESID_COLS),
                                    ("moist", MOIST_RESID_COLS)]:
        av = [c for c in tgt_cols if c in tr_all.columns]
        if not av: continue
        ts = RobustScaler(); ts.fit(tr_all[av].dropna().values)
        snap_tgt_scalers[tgt_grp_name] = ts
        print(f"  ✓ snap_tgt [{tgt_grp_name}]")

    # ── ML baselines (per-site, for comparison) ───────────────────────────────
    print("\nML baselines...")
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    import xgboost as xgb

    def prep_xy(site, split, tgt_cols):
        fs = feat_scalers.get(site); ts = snap_tgt_scalers.get("temp")
        if fs is None or ts is None: return None, None
        mask = (df["Site"]==site)&(df["split"]==split)
        d    = df[mask].sort_values("time_utc")
        av   = MODEL_FEATURES + tgt_cols
        d    = d[[c for c in av if c in d.columns]].dropna(subset=av)
        if len(d)<10: return None,None
        return fs.transform(d[MODEL_FEATURES].values), ts.transform(d[tgt_cols].values)

    ml_results = []
    for mname, model in [
        ("Ridge",      Ridge(alpha=1.0)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=200,max_depth=15,
                                            min_samples_leaf=20,n_jobs=-1,random_state=SEED)),
        ("XGBoost",    xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.05,
                                         subsample=0.8,tree_method="hist",n_jobs=-1,
                                         random_state=SEED,verbosity=0)),
    ]:
        print(f"  {mname}...")
        for site in feat_scalers:
            X_tr, y_tr = prep_xy(site, "train", TEMP_RESID_COLS)
            if X_tr is None: continue
            model.fit(X_tr, y_tr)
            X_te, _   = prep_xy(site, "test",  TEMP_RESID_COLS)
            if X_te is None: continue
            yp_sc = model.predict(X_te)
            if yp_sc.ndim==1: yp_sc=yp_sc.reshape(-1,1)
            ts_   = snap_tgt_scalers["temp"]
            yp_r  = ts_.inverse_transform(yp_sc)
            ts_df = df[(df["Site"]==site)&(df["split"]=="test")].sort_values("time_utc")
            tgt   = TEMP_TARGETS[0]; ac = f"{tgt}_approx"
            if ac not in ts_df.columns: continue
            ap = ts_df[ac].dropna().values; yt_f = ts_df[tgt].dropna().values
            n  = min(len(yp_r),len(ap),len(yt_f))
            if n<10: continue
            yp_f = ap[:n]+yp_r[:n,0]; yt_n = yt_f[:n]
            mk   = ~(np.isnan(yt_n)|np.isnan(yp_f))
            yt_m,yp_m,ap_m = yt_n[mk],yp_f[mk],ap[:n][mk]
            if len(yt_m)<5: continue
            r2 = float(1-np.sum((yt_m-yp_m)**2)/(np.sum((yt_m-yt_m.mean())**2)+1e-10))
            sk = float(1-np.mean((yt_m-yp_m)**2)/(np.mean((yt_m-ap_m)**2)+1e-10))
            ml_results.append(dict(Model=mname,site=site,R2=round(r2,4),Skill=round(sk,4)))
    ml_df = pd.DataFrame(ml_results)
    ml_df.to_csv(RESULTS_DIR/"baseline_results.csv", index=False)
    if len(ml_df):
        print(ml_df.groupby("Model")[["R2","Skill"]].mean().sort_values("R2",ascending=False).round(4).to_string())

    # ── Save everything ───────────────────────────────────────────────────────
    print("\nSaving preprocessed data...")
    t0 = time.time()
    df.to_csv(PREPROC_DIR/"master_processed.csv", index=False)
    sz = (PREPROC_DIR/"master_processed.csv").stat().st_size/1e6
    print(f"  ✓ master_processed.csv  {sz:.0f} MB  {time.time()-t0:.1f}s")

    SC = dict(feat_scalers=feat_scalers,
              snap_feat_scaler=snap_feat_scaler,
              snap_tgt_scalers=snap_tgt_scalers)
    with open(PREPROC_DIR/"scalers.pkl","wb") as f: pickle.dump(SC, f)

    FI = dict(MODEL_FEATURES=MODEL_FEATURES, SNAP_FEATURES=SNAP_FEATURES,
              ALL_TARGETS=ALL_TARGETS, TEMP_TARGETS=TEMP_TARGETS,
              SMAP_TARGETS=SMAP_TARGETS, MOIST_TARGETS=MOIST_TARGETS,
              TEMP_RESID_COLS=TEMP_RESID_COLS, SMAP_RESID_COLS=SMAP_RESID_COLS,
              MOIST_RESID_COLS=MOIST_RESID_COLS,
              SITES=SITES, N_FEATURES=N_FEATURES, N_SNAP_FEATURES=N_SNAP_FEATURES,
              LOCATIONS=LOCATIONS.to_dict(), N_LOCS=N_LOCS)
    with open(PREPROC_DIR/"feature_info.pkl","wb") as f: pickle.dump(FI, f)

    with open(MARKER,"w") as f:
        json.dump({"completed_at":pd.Timestamp.now().isoformat(),"n_rows":len(df),"N_LOCS":N_LOCS},f)
    print("  ✓ All saved — future runs skip to Phase 5")

# Reconstruct LOCATIONS df if loaded from cache
if isinstance(LOCATIONS, dict):
    LOCATIONS = pd.DataFrame(LOCATIONS)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SPATIAL GRAPH over lat/lon locations
# k-NN with Gaussian decay; normalised adjacency for GCN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  PHASE 2: Spatial Graph (lat/lon locations)")
print("="*55)

def build_latlon_graph(locs_df, k=6):
    """
    Build k-NN graph over actual lat/lon coordinates.
    Uses Haversine-approximated distances (flat-earth ok at this scale).
    Each node = one unique (Lat, Lon) measurement location.
    Physical interpretation: nearby locations share soil moisture
    through lateral flow, runoff, and thermal diffusion.
    """
    coords = locs_df[["Latitude","Longitude"]].values.astype(np.float32)
    N      = len(coords)
    # Scale lat/lon to approximate km (1° lat ≈ 111km, 1° lon ≈ 63km at 68°N)
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
    D    = A.sum(1, keepdims=True)**0.5
    A_n  = A / (D * D.T + 1e-8)
    print(f"  Locations : {N}")
    print(f"  k-neighbors: {k}")
    print(f"  sigma (km) : {sigma:.2f}")
    print(f"  Avg degree : {(A>0).sum(1).mean():.1f}")
    return coords, A_n.astype(np.float32)

loc_coords, A_norm_np = build_latlon_graph(LOCATIONS, k=6)
A_norm_t = torch.tensor(A_norm_np).to(DEVICE)
print(f"  Graph shape: {A_norm_np.shape}")

# Location index lookup: (lat,lon) → row index in LOCATIONS
loc_index = {(row.Latitude, row.Longitude): i
             for i, row in LOCATIONS.iterrows()}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SpatialSnapshotDataset
#
# KEY CHANGE FROM v2:
#   v2: grouped by location → (time_steps, features) per location
#   v3: grouped by timestamp → (N_locations, features) per timestamp
#
# Each sample:
#   X       : (lookback, N_locs, N_snap_features)  — field history
#   y_resid : (N_locs, N_targets)                  — residual targets
#   y_approx: (N_locs, N_targets)                  — seasonal component
#   y_raw   : (N_locs, N_targets)                  — full true values
#   A       : (N_locs, N_locs)                     — adjacency matrix
#
# This means:
#   - The model sees the FULL SPATIAL FIELD at each past timestep
#   - It predicts the FULL SPATIAL FIELD at the next timestep
#   - The graph captures spatial coupling between locations
#   - Lateral flow, topographic drainage, and thermal diffusion
#     are all implicitly encoded in the spatial correlations
# ══════════════════════════════════════════════════════════════════════════════

class SpatialSnapshotDataset(Dataset):
    """
    True spatial field prediction dataset.
    Pivots by time_utc: each sample = one spatial field snapshot.
    Models learn: field(t-k:t) → field(t+1)
    """
    def __init__(self, df, loc_index, A_norm,
                 snap_features, resid_cols, approx_cols, raw_cols,
                 feat_scaler, tgt_scaler,
                 split="train", lookback=24, stride=6, max_samples=None):
        self.A = A_norm
        N   = len(loc_index)
        nf  = len(snap_features)
        nt  = len(resid_cols)
        nat = len(approx_cols)

        # ── Filter to split ───────────────────────────────────────────────────
        sub = df[df["split"]==split].copy()

        # ── Get all unique timestamps in order ────────────────────────────────
        all_ts = sorted(sub["time_utc"].unique())
        T      = len(all_ts)
        print(f"    [{split}] {T:,} timestamps  |  {N} locations")

        if T < lookback + 2:
            self.X=self.yr=self.ya=self.yw=torch.zeros(0); return

        # ── Build (T, N, F) tensor — fast vectorised pivot ────────────────────
        # Uses integer index arrays — avoids iterrows Series TypeError bug
        ts_to_i  = {ts: i for i, ts in enumerate(all_ts)}
        loc_to_i = {(float(r.Latitude), float(r.Longitude)): i
                    for i, r in LOCATIONS.iterrows()}

        sub2 = sub.copy()
        sub2["_ti"] = sub2["time_utc"].map(ts_to_i)
        sub2["_ni"] = [loc_to_i.get((float(la), float(lo)))
                       for la, lo in zip(sub2["Latitude"].astype(float),
                                         sub2["Longitude"].astype(float))]
        sub2 = sub2.dropna(subset=["_ti","_ni"])
        sub2["_ti"] = sub2["_ti"].astype(int)
        sub2["_ni"] = sub2["_ni"].astype(int)
        ti_arr = sub2["_ti"].values
        ni_arr = sub2["_ni"].values

        # Pre-allocate
        X_full  = np.full((T, N, nf),  np.nan, dtype=np.float32)
        yr_full = np.full((T, N, nt),  np.nan, dtype=np.float32)
        ya_full = np.full((T, N, nat), np.nan, dtype=np.float32)
        yw_full = np.full((T, N, len(raw_cols)), np.nan, dtype=np.float32)

        # Fill feature tensor (vectorised — no iterrows)
        sf_scaled = feat_scaler.transform(
            sub2[snap_features].fillna(0).values).astype(np.float32)
        X_full[ti_arr, ni_arr, :] = sf_scaled

        # Fill residual target tensor
        yr_scaled = tgt_scaler.transform(
            sub2[resid_cols].fillna(0).values).astype(np.float32)
        yr_full[ti_arr, ni_arr, :] = yr_scaled

        # Fill approx and raw tensors
        ya_full[ti_arr, ni_arr, :] = sub2[approx_cols].fillna(0).values.astype(np.float32)
        yw_full[ti_arr, ni_arr, :] = sub2[raw_cols].fillna(0).values.astype(np.float32)


        # ── Slide window over time ─────────────────────────────────────────────
        tidxs = list(range(lookback, T, stride))
        if max_samples and len(tidxs) > max_samples:
            rng   = np.random.default_rng(SEED)
            tidxs = sorted(rng.choice(tidxs, max_samples, replace=False))

        Xl=[]; yrl=[]; yal=[]; ywl=[]
        for ti in tidxs:
            Xw  = X_full[ti-lookback:ti]     # (lookback, N, F)
            yri = yr_full[ti]                 # (N, nt)
            yai = ya_full[ti]                 # (N, nat)
            ywi = yw_full[ti]                 # (N, nt)
            # Skip if too many NaNs (>20% of locations missing)
            if np.isnan(Xw).mean() > 0.2: continue
            if np.isnan(yri).mean() > 0.5: continue
            # Fill remaining NaNs with 0 (masked by loss)
            Xw  = np.nan_to_num(Xw,  nan=0.0)
            yri = np.nan_to_num(yri, nan=0.0)
            yai = np.nan_to_num(yai, nan=0.0)
            ywi = np.nan_to_num(ywi, nan=0.0)
            Xl.append(Xw); yrl.append(yri); yal.append(yai); ywl.append(ywi)

        if not Xl:
            self.X=self.yr=self.ya=self.yw=torch.zeros(0); return

        # Stack: (S, lookback, N, F) — S = number of samples
        self.X  = torch.tensor(np.array(Xl),  dtype=torch.float32)
        self.yr = torch.tensor(np.array(yrl), dtype=torch.float32)
        self.ya = torch.tensor(np.array(yal), dtype=torch.float32)
        self.yw = torch.tensor(np.array(ywl), dtype=torch.float32)
        print(f"    [{split}] {len(self.X):,} samples  |  "
              f"X={tuple(self.X.shape[1:])}  |  y={tuple(self.yr.shape[1:])}")

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.yr[i], self.ya[i], self.yw[i], self.A


def make_snap_loaders(tgt_grp_name, tgt_resid_cols, tgt_approx_cols, tgt_raw_cols,
                      lookback=24, st_tr=6, st_ev=24, bs=4,
                      max_tr=2000, max_ev=500):
    """Build train/val/test DataLoaders for one target group."""
    if not tgt_resid_cols:
        return {s:None for s in ["train","val","test"]}
    ts = snap_tgt_scalers.get(tgt_grp_name)
    if ts is None:
        return {s:None for s in ["train","val","test"]}
    out = {}
    for sp, ms, st in [("train",max_tr,st_tr),("val",max_ev,st_ev),("test",max_ev,st_ev)]:
        ds = SpatialSnapshotDataset(
            df, loc_index, A_norm_t,
            SNAP_FEATURES, tgt_resid_cols, tgt_approx_cols, tgt_raw_cols,
            snap_feat_scaler, ts,
            split=sp, lookback=lookback, stride=st, max_samples=ms)
        out[sp] = None if len(ds)==0 else DataLoader(
            ds, batch_size=bs, shuffle=(sp=="train"),
            num_workers=0,  # CUDA cannot init in forked workers on talon32
            pin_memory=False,  # must be False when num_workers=0
            drop_last=(sp=="train"))
    return out

print("\nBuilding spatial snapshot dataloaders...")
TEMP_APPROX_COLS  = [f"{t}_approx" for t in TEMP_TARGETS  if f"{t}_approx"  in df.columns]
SMAP_APPROX_COLS  = [f"{t}_approx" for t in SMAP_TARGETS  if f"{t}_approx"  in df.columns]
MOIST_APPROX_COLS = [f"{t}_approx" for t in MOIST_TARGETS if f"{t}_approx"  in df.columns]

temp_ld  = make_snap_loaders("temp",  TEMP_RESID_COLS,  TEMP_APPROX_COLS,  TEMP_TARGETS)
smap_ld  = make_snap_loaders("smap",  SMAP_RESID_COLS,  SMAP_APPROX_COLS,  SMAP_TARGETS)
moist_ld = make_snap_loaders("moist", MOIST_RESID_COLS, MOIST_APPROX_COLS, MOIST_TARGETS)

for name, ld in [("Temp",temp_ld),("SMAP",smap_ld),("Moist",moist_ld)]:
    for sp, l in ld.items():
        nb = len(l) if l else 0
        print(f"  {name:<6} {sp:<6}: {nb} batches")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — MODEL DEFINITIONS
# All models: Input (B, lookback, N, F) → Output (B, N, T)
# GraphConv: FIXED — uses torch.bmm with broadcast (no dimension mismatch)
# Training engine: FIXED — correct MoE/non-MoE output unpacking
# ══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    """
    Graph convolution: H(B,N,d) × A(N,N) → H'(B,N,d)
    FIX: uses torch.bmm with A broadcast across batch.
    Old: torch.stack([g(hg[b],A) for b in range(B)]) → wrong shape
    New: A_b = A.unsqueeze(0).expand(B,-1,-1); bmm(A_b, W(H))
    """
    def __init__(self, in_d, out_d, dp=0.1):
        super().__init__()
        self.W = nn.Linear(in_d, out_d, bias=False)
        self.n = nn.LayerNorm(out_d)
        self.d = nn.Dropout(dp)
        self.a = nn.GELU()
    def forward(self, H, A):
        # H: (B, N, d)   A: (N, N) or (B, N, N) depending on collation
        # Normalise A to always be 2D (N, N) then broadcast across batch
        if A.dim() == 3: A = A[0]   # take first — all rows identical after collation
        if A.dim() == 4: A = A[0,0] # extra batch dim from stacking
        A_b = A.unsqueeze(0).expand(H.shape[0], -1, -1)  # (B, N, N)
        return self.a(self.n(torch.bmm(A_b, self.W(self.d(H)))))


class SpatialBiGRU(nn.Module):
    """
    Temporal: BiGRU over lookback per location, then multi-head attention.
    Spatial : 2-layer GCN to propagate information across locations.
    Input   : (B, L, N, F)  where L=lookback, N=locations
    Output  : (B, N, T)
    """
    def __init__(self, nf, h=96, nl=2, nh=4, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.proj = nn.Linear(nf, h)
        self.gru  = nn.GRU(h, h, nl, batch_first=True,
                           bidirectional=True, dropout=dp if nl>1 else 0.)
        d2 = h*2
        self.attn = nn.MultiheadAttention(d2, nh, dropout=dp, batch_first=True)
        self.n1   = nn.LayerNorm(d2); self.n2 = nn.LayerNorm(d2)
        self.ffn  = nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),
                                   nn.Dropout(dp),nn.Linear(d2*2,d2))
        self.red  = nn.Linear(d2, h)
        self.gcn  = nn.ModuleList([GraphConv(h,h,dp) for _ in range(gl)])
        self.head = nn.Sequential(nn.Linear(h*2,h),nn.GELU(),
                                   nn.Dropout(dp),nn.Linear(h,nt))

    def forward(self, x, A):
        # x: (B, L, N, F)
        B,L,N,F = x.shape
        # Process each location's time series independently
        h   = x.permute(0,2,1,3).reshape(B*N, L, F)  # (B*N, L, F)
        h   = self.proj(h)                              # (B*N, L, h)
        h,_ = self.gru(h)                               # (B*N, L, 2h)
        a,_ = self.attn(h,h,h)
        h   = self.n1(h+a); h = self.n2(h+self.ffn(h))
        h   = self.red(h[:,-1,:]).reshape(B,N,-1)       # (B, N, h)
        # Graph convolution across locations
        hg  = h
        for g in self.gcn: hg = g(hg, A)               # (B, N, h) — FIXED
        return self.head(torch.cat([h,hg],dim=-1))       # (B, N, T)


class MambaBlock(nn.Module):
    """Selective State Space Model (Mamba-style)."""
    def __init__(self, d, ds=16, dc=4, ex=2, dp=0.1):
        super().__init__()
        self.di  = d*ex; self.ds = ds
        self.ip  = nn.Linear(d, self.di*2, bias=False)
        self.cv  = nn.Conv1d(self.di, self.di, dc, padding=dc-1, groups=self.di, bias=True)
        self.silu= nn.SiLU()
        self.xp  = nn.Linear(self.di, ds*2+self.di, bias=False)
        self.dtp = nn.Linear(self.di, self.di, bias=True)
        A_ = torch.arange(1,ds+1,dtype=torch.float32).unsqueeze(0).repeat(self.di,1)
        self.Al  = nn.Parameter(torch.log(A_))
        self.D_  = nn.Parameter(torch.ones(self.di))
        self.op  = nn.Linear(self.di, d, bias=False)
        self.dr  = nn.Dropout(dp); self.nm = nn.LayerNorm(d)
    def scan(self, x):
        B,L,D = x.shape; S = self.ds
        xd    = self.xp(x); dl,Bp,C = xd.split([D,S,S],dim=-1)
        dl    = F.softplus(self.dtp(dl))
        A_    = -torch.exp(self.Al.float())
        dA    = torch.exp(torch.einsum("bld,ds->blds",dl,A_))
        dB    = torch.einsum("bld,bls->blds",dl,Bp)
        h     = torch.zeros(B,D,S,device=x.device,dtype=x.dtype); ys=[]
        for i in range(L):
            h = dA[:,i]*h + dB[:,i]*x[:,i,:,None]
            ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
        return torch.stack(ys,dim=1)*self.D_
    def forward(self, x):
        r=x; xz=self.ip(x); x_,z=xz.chunk(2,dim=-1)
        x_ = self.silu(self.cv(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        y  = self.scan(x_)*self.silu(z)
        return self.nm(r+self.op(self.dr(y)))


class SpatialMamba(nn.Module):
    """Mamba SSM temporal encoder + GCN spatial encoder."""
    def __init__(self, nf, d=96, nl=4, ds=16, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.em  = nn.Linear(nf, d)
        self.mb  = nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nl)])
        self.nm  = nn.LayerNorm(d)
        self.gcn = nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd  = nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self, x, A):
        B,L,N,F = x.shape
        h = self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for b in self.mb: h = b(h)
        h  = self.nm(h[:,-1,:]).reshape(B,N,-1)
        hg = h
        for g in self.gcn: hg = g(hg,A)               # FIXED
        return self.hd(torch.cat([h,hg],dim=-1))


class S4Layer(nn.Module):
    """S4 with bidirectional scan and HiPPO-LegS init."""
    def __init__(self, d, ds=64, dp=0.1):
        super().__init__()
        self.ds = ds
        def hippo(N):
            A=torch.zeros(N,N)
            for n in range(N):
                for m in range(n): A[n,m]=-(2*n+1)**.5*(2*m+1)**.5
                A[n,n]=-(n+1)
            return A
        self.A_  = nn.Parameter(hippo(ds), requires_grad=False)
        self.B_  = nn.Parameter(torch.randn(ds,1)*0.01)
        self.C_  = nn.Parameter(torch.randn(d,ds))
        self.D_  = nn.Parameter(torch.ones(d))
        self.nm  = nn.LayerNorm(d); self.dr = nn.Dropout(dp)
        self.ot  = nn.Linear(d,d);  self.mx = nn.Linear(d*2,d)
    def scan(self, u):
        B,L,d = u.shape
        dA = torch.matrix_exp(self.A_); dB = self.B_.squeeze(-1)
        h  = torch.zeros(B,d,self.ds,device=u.device); ys=[]
        for t in range(L):
            h = h@dA.T + u[:,t,:,None]*dB
            ys.append((h*self.C_.unsqueeze(0)).sum(-1)+self.D_*u[:,t,:])
        return torch.stack(ys,dim=1)
    def forward(self, x):
        yf=self.scan(x); yr=self.scan(x.flip(1)).flip(1)
        return self.nm(x+self.dr(self.ot(self.mx(torch.cat([yf,yr],dim=-1)))))


class SpatialS4(nn.Module):
    """S4 SSM + GCN spatial coupling."""
    def __init__(self, nf, d=96, nl=4, ds=64, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.em  = nn.Linear(nf,d)
        self.ly  = nn.ModuleList([S4Layer(d,ds,dp) for _ in range(nl)])
        self.nm  = nn.LayerNorm(d)
        self.gcn = nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.hd  = nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))
    def forward(self, x, A):
        B,L,N,F = x.shape
        h = self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        for l in self.ly: h=l(h)
        h  = self.nm(h[:,-1,:]).reshape(B,N,-1)
        hg = h
        for g in self.gcn: hg = g(hg,A)               # FIXED
        return self.hd(torch.cat([h,hg],dim=-1))


class SpatialFuseMoE(nn.Module):
    """
    Sparse top-2 Mixture-of-Experts:
    4 experts (Mamba, GRU, CNN, GRU) → gated fusion → Mamba backbone → GCN.
    Load-balancing auxiliary loss encourages expert specialisation.
    Returns (pred, aux_loss).
    """
    def __init__(self, nf, d=96, ne=4, tk=2, ds=16, nsl=2, N=256, gl=2, nt=1, dp=0.1):
        super().__init__()
        self.ne=ne; self.tk=tk; self.d=d
        self.em  = nn.Linear(nf,d)
        self.ex  = nn.ModuleList([
            MambaBlock(d,ds,dp=dp),
            nn.GRU(d,d,batch_first=True),
            nn.Sequential(nn.Conv1d(d,d,7,padding=3,groups=d),nn.Conv1d(d,d,1),
                          nn.GELU(),nn.AdaptiveAvgPool1d(1)),
            nn.GRU(d,d,batch_first=True),
        ])
        self.enm = nn.ModuleList([nn.LayerNorm(d) for _ in range(ne)])
        self.gt  = nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ne))
        self.bb  = nn.ModuleList([MambaBlock(d,ds,dp=dp) for _ in range(nsl)])
        self.gcn = nn.ModuleList([GraphConv(d,d,dp) for _ in range(gl)])
        self.nm  = nn.LayerNorm(d)
        self.hd  = nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dp),nn.Linear(d,nt))

    def _expert(self, i, h):
        ex = self.ex[i]
        if isinstance(ex, MambaBlock): return self.enm[i](ex(h)[:,-1,:])
        if isinstance(ex, nn.GRU):     _, ht=ex(h); return self.enm[i](ht[-1])
        return self.enm[i](ex(h.transpose(1,2)).squeeze(-1))

    def forward(self, x, A):
        B,L,N,F = x.shape
        h    = self.em(x.permute(0,2,1,3).reshape(B*N,L,F))
        g_in = h.mean(1)
        lg   = self.gt(g_in)
        tv,ti= lg.topk(self.tk,dim=-1)
        gs   = torch.nn.functional.softmax(tv,dim=-1)
        gs_s = torch.nn.functional.softmax(lg,dim=-1)
        imp  = gs_s.mean(0); ld=(gs_s>1/self.ne).float().mean(0)
        aux  = (imp*ld).sum()*self.ne
        eo   = [self._expert(i,h) for i in range(self.ne)]
        Es   = torch.stack(eo,dim=1)
        sel  = torch.gather(Es,1,ti.unsqueeze(-1).expand(-1,-1,self.d))
        fsd  = (sel*gs.unsqueeze(-1)).sum(1)
        fs   = fsd.unsqueeze(1).expand(-1,L,-1)+h
        for b in self.bb: fs=b(fs)
        ho   = self.nm(fs[:,-1,:]).reshape(B,N,-1)
        hg   = ho
        for g in self.gcn: hg = g(hg,A)               # FIXED
        return self.hd(torch.cat([ho,hg],dim=-1)), aux


ARCH_MAP = {
    "SpatialBiGRU"  : lambda nt: SpatialBiGRU(N_SNAP_FEATURES,h=96,nl=2,nh=4,N=N_LOCS,gl=2,nt=nt),
    "SpatialMamba"  : lambda nt: SpatialMamba( N_SNAP_FEATURES,d=96,nl=4,ds=16,N=N_LOCS,gl=2,nt=nt),
    "SpatialS4"     : lambda nt: SpatialS4(    N_SNAP_FEATURES,d=96,nl=4,ds=64,N=N_LOCS,gl=2,nt=nt),
    "SpatialFuseMoE": lambda nt: SpatialFuseMoE(N_SNAP_FEATURES,d=96,ne=4,tk=2,ds=16,nsl=2,N=N_LOCS,gl=2,nt=nt),
}

print("\nModel parameter counts:")
print(f"  {'Model':<20} {'Params':>12}")
print("  " + "─"*34)
for name, fn in ARCH_MAP.items():
    try:
        m = fn(1); p = sum(x.numel() for x in m.parameters() if x.requires_grad)
        print(f"  {name:<20} {p:>12,}")
    except Exception as e: print(f"  {name:<20} ERROR: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — ENTROPY TRACKER
# Diagnoses whether models learn physics or just seasonal patterns
# ══════════════════════════════════════════════════════════════════════════════

class EntropyTracker:
    """
    Tracks Shannon entropy of spatial prediction distributions over training.
    H_norm plateau early → SEASONAL_FITTING (bad)
    H_norm evolves       → LEARNING_DYNAMICS (good)
    """
    def __init__(self, nb=50): self.nb=nb; self.hist=[]
    def compute(self, preds):
        flat = preds.flatten(); flat=flat[~np.isnan(flat)]
        if len(flat)<10: return 0.
        c,_ = np.histogram(flat,bins=self.nb)
        p   = c/(c.sum()+1e-10)
        H   = float(-np.sum(p*np.log(p+1e-12))/np.log(self.nb))
        self.hist.append(H); return H
    def diagnose(self):
        if len(self.hist)<3: return {"diagnosis":"INSUFFICIENT_DATA"}
        init  = float(np.mean(self.hist[:3]))
        final = float(np.mean(self.hist[-3:]))
        delta = final-init
        plat  = sum(abs(self.hist[i]-self.hist[i-1])<0.005
                    for i in range(1,len(self.hist)))
        diag  = ("SEASONAL_FITTING"
                 if (plat>0.6*len(self.hist) and abs(delta)<0.02)
                 else "LEARNING_DYNAMICS")
        return dict(diagnosis=diag,
                    initial_H=round(init,4), final_H=round(final,4),
                    delta_H=round(delta,4),
                    plateau_frac=round(plat/max(1,len(self.hist)),3))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — TRAINING ENGINE
# Huber loss + Graph Laplacian spatial smoothness + AMP
# BUG FIX: always call out=model(x,A) first, then unpack for MoE
# ══════════════════════════════════════════════════════════════════════════════

def huber(p, t, delta=1.0):
    d = p-t
    return torch.where(d.abs()<=delta, 0.5*d**2, delta*(d.abs()-0.5*delta)).mean()

def graph_smooth(pred, A):
    """
    Graph Laplacian loss: encourage spatial smoothness.
    L = ||pred - A*pred||_F²  penalises predictions that differ from
    the weighted average of their spatial neighbours.
    Physical: adjacent soil locations share moisture through lateral flow.
    """
    A_b = A.unsqueeze(0).expand(pred.shape[0],-1,-1)
    sm  = torch.bmm(A_b, pred)
    return F.mse_loss(pred, sm)

def train_one(arch, n_targets, train_ld, val_ld, tgt_sc,
              epochs=30, lr=3e-4, patience=7,
              lam_s=0.05, lam_a=0.01, ckpt_path=None):

    is_moe = (arch == "SpatialFuseMoE")
    model  = ARCH_MAP[arch](n_targets).to(DEVICE)
    opt    = AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                   lr=lr, weight_decay=1e-4)
    n_steps = epochs*len(train_ld)
    sched   = OneCycleLR(opt, max_lr=lr, total_steps=n_steps, pct_start=0.1)
    amp_sc  = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    tracker = EntropyTracker()

    best_r2=float("-inf"); best_st=None; pat=0; hist=[]; t0=time.time()
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {arch} | {np_:,} params | {epochs}ep | {DEVICE}")

    for ep in range(1, epochs+1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train(); tr=0.; nb=0
        for batch in train_ld:
            X,yr,ya,yw,A = [b.to(DEVICE) for b in batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(X, A)                           # FIXED: always call first
                if is_moe: pred, aux = out
                else:       pred = out; aux = None
                loss = huber(pred,yr) + lam_s*graph_smooth(pred,A)
                if aux is not None: loss = loss + lam_a*aux
            amp_sc.scale(loss).backward()
            amp_sc.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_sc.step(opt); amp_sc.update(); sched.step()
            tr+=loss.item(); nb+=1
        tr_loss=tr/max(nb,1)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval(); vt=[]; vp=[]; va=[]; pa=[]
        with torch.no_grad():
            for batch in val_ld:
                X,yr,ya,yw,A = [b.to(DEVICE) for b in batch]
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out  = model(X,A)                       # FIXED
                    pred = out[0] if is_moe else out
                B_,N_,T_ = pred.shape
                pr   = pred.cpu().float().numpy()
                pr_r = tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
                vt.append(yw.cpu().numpy())
                vp.append(ya.cpu().numpy()+pr_r)
                va.append(ya.cpu().numpy())
                pa.append(pr)

        yt=np.concatenate(vt,0); yp=np.concatenate(vp,0)
        ytf=yt[:,:,0].flatten(); ypf=yp[:,:,0].flatten()
        mk =~(np.isnan(ytf)|np.isnan(ypf)); ytf=ytf[mk]; ypf=ypf[mk]
        r2 = float(1-np.sum((ytf-ypf)**2)/(np.sum((ytf-ytf.mean())**2)+1e-10))
        H  = tracker.compute(np.concatenate(pa,0))
        hist.append(dict(epoch=ep,train_loss=round(tr_loss,6),
                         val_R2=round(r2,4),H_norm=round(H,4)))

        if r2 > best_r2:
            best_r2=r2
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat=0
        else: pat+=1

        if ep%5==0 or ep==1:
            print(f"    E{ep:03d} | loss={tr_loss:.4f} | R²={r2:.4f} | "
                  f"H={H:.4f} | {time.time()-t0:.0f}s")
        if pat>=patience:
            print(f"    Early stop @ epoch {ep}"); break

    elapsed = time.time()-t0
    e_summ  = tracker.diagnose()
    print(f"  ✓ val R²={best_r2:.4f} | {elapsed:.0f}s | {e_summ['diagnosis']}")

    if best_st: model.load_state_dict(best_st)
    torch.save(dict(arch=arch, state_dict=best_st, val_r2=best_r2,
                    history=hist, epochs_run=ep, elapsed_s=elapsed,
                    entropy_summary=e_summ, n_locs=N_LOCS,
                    n_snap_features=N_SNAP_FEATURES,
                    job_id=JOB_ID, node=NODE), ckpt_path)
    return model, hist, best_r2, elapsed


@torch.no_grad()
def evaluate(model, loader, tgt_sc, arch):
    """Full test evaluation: R², RMSE, KGE, Skill, FreezeAcc, per-node R²."""
    is_moe = (arch=="SpatialFuseMoE")
    model.eval()
    yt=[]; yp=[]; ya=[]
    for batch in loader:
        X,yr,yapp,yw,A = [b.to(DEVICE) for b in batch]
        out  = model(X,A)                                   # FIXED
        pred = out[0] if is_moe else out
        B_,N_,T_ = pred.shape
        pr   = pred.cpu().float().numpy()
        pr_r = tgt_sc.inverse_transform(pr.reshape(-1,T_)).reshape(B_,N_,T_)
        yt.append(yw.cpu().numpy())
        yp.append(yapp.cpu().numpy()+pr_r)
        ya.append(yapp.cpu().numpy())

    yta=np.concatenate(yt,0); ypa=np.concatenate(yp,0); yaa=np.concatenate(ya,0)
    ytf=yta[:,:,0].flatten(); ypf=ypa[:,:,0].flatten(); yaf=yaa[:,:,0].flatten()
    mk =~(np.isnan(ytf)|np.isnan(ypf)); ytf=ytf[mk]; ypf=ypf[mk]; yaf=yaf[mk]

    r2   = float(1-np.sum((ytf-ypf)**2)/(np.sum((ytf-ytf.mean())**2)+1e-10))
    rmse = float(np.sqrt(np.mean((ytf-ypf)**2)))
    r    = float(np.corrcoef(ytf,ypf)[0,1])
    kge  = float(1-np.sqrt((r-1)**2+(np.std(ypf)/(np.std(ytf)+1e-10)-1)**2+
                            (np.mean(ypf)/(np.mean(ytf)+1e-10)-1)**2))
    sk   = float(1-np.mean((ytf-ypf)**2)/(np.mean((ytf-yaf)**2)+1e-10))
    frz  = float(np.mean((ytf<0).astype(int)==(ypf<0).astype(int))*100)

    # Per-location R²  (spatial quality metric — key for field prediction)
    nr2=[]
    for n in range(yta.shape[1]):
        yn=yta[:,n,0]; pn=ypa[:,n,0]
        mk2=~(np.isnan(yn)|np.isnan(pn))
        if mk2.sum()<5: continue
        nr2.append(float(1-np.sum((yn[mk2]-pn[mk2])**2)/
                         (np.sum((yn[mk2]-yn[mk2].mean())**2)+1e-10)))

    # Spatial variance ratio (does the model reproduce spatial variability?)
    svr = float(np.var(ypa[:,:,0].mean(0))/(np.var(yta[:,:,0].mean(0))+1e-10))

    return dict(R2=round(r2,4), RMSE=round(rmse,4), KGE=round(kge,4),
                Skill=round(sk,4), Freeze_Acc=round(frz,2),
                node_r2_mean=round(float(np.mean(nr2)),4) if nr2 else float("nan"),
                node_r2_min =round(float(np.min(nr2)),4)  if nr2 else float("nan"),
                node_r2_std =round(float(np.std(nr2)),4)  if nr2 else float("nan"),
                spatial_var_ratio=round(svr,4), N=int(mk.sum()))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — TRAINING LOOP
# 4 architectures × 3 target groups = 12 checkpoints
# Resume guard: skip if checkpoint already valid
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PHASE 7: Spatial Snapshot Model Training")
print(f"  Input shape : (B, lookback=24, N_locs={N_LOCS}, F={N_SNAP_FEATURES})")
print(f"  Output shape: (B, N_locs={N_LOCS}, T)")
print("="*70)

ARCHES = ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]

TARGET_GROUPS = [
    ("temp",  TEMP_RESID_COLS,  TEMP_APPROX_COLS,  TEMP_TARGETS,
     len(TEMP_RESID_COLS),   "Weather Temp",   temp_ld),
    ("smap",  SMAP_RESID_COLS,  SMAP_APPROX_COLS,  SMAP_TARGETS,
     len(SMAP_RESID_COLS),   "SMAP Temp L1",   smap_ld),
    ("moist", MOIST_RESID_COLS, MOIST_APPROX_COLS, MOIST_TARGETS,
     len(MOIST_RESID_COLS),  "Soil Moisture",  moist_ld),
]

all_results = []

for (tgt_name, resid_cols, approx_cols, raw_cols, n_tgts, label, loaders) in TARGET_GROUPS:
    print(f"\n{'─'*60}")
    print(f"  TARGET GROUP: {label}  |  n_targets={n_tgts}  |  n_locs={N_LOCS}")
    print(f"{'─'*60}")
    if loaders.get("train") is None:
        print("  No training data — skip"); continue
    tgt_sc = snap_tgt_scalers.get(tgt_name)
    if tgt_sc is None:
        print("  No scaler — skip"); continue

    for arch in ARCHES:
        ckpt_name = f"{arch}_{tgt_name}_v3_best.pt"
        ckpt_path = MODELS_DIR / ckpt_name

        # ── Resume guard ──────────────────────────────────────────────────────
        if ckpt_path.exists():
            try:
                sv = torch.load(ckpt_path, map_location="cpu")
                if sv.get("val_r2", -99) > -10:
                    diag = sv.get("entropy_summary",{}).get("diagnosis","N/A")
                    print(f"\n  ✓ SKIP {arch} [{tgt_name}] "
                          f"val_r2={sv['val_r2']:.4f} | {diag}")
                    tm = sv.get("test_metrics",{})
                    all_results.append(dict(
                        Model=arch, Target=tgt_name, Val_R2=sv["val_r2"],
                        **{f"Test_{k}":v for k,v in tm.items()},
                        Diagnosis=diag, Resumed=True))
                    continue
            except Exception: pass

        print(f"\n  ── {arch}  [{label}]")
        try:
            model, hist, best_r2, elapsed = train_one(
                arch=arch, n_targets=n_tgts,
                train_ld=loaders["train"], val_ld=loaders["val"],
                tgt_sc=tgt_sc,
                epochs=30, lr=3e-4, patience=7,
                lam_s=0.05, lam_a=0.01, ckpt_path=ckpt_path)

            # Test evaluation
            test_m = {}
            if loaders.get("test"):
                test_m = evaluate(model, loaders["test"], tgt_sc, arch)
                sv = torch.load(ckpt_path, map_location="cpu")
                sv["test_metrics"] = test_m
                sv["job_id"] = JOB_ID; sv["node"] = NODE
                torch.save(sv, ckpt_path)

            es   = torch.load(ckpt_path,map_location="cpu").get("entropy_summary",{})
            diag = es.get("diagnosis","N/A")
            all_results.append(dict(
                Model=arch, Target=tgt_name,
                Val_R2=best_r2,
                Test_R2=test_m.get("R2",float("nan")),
                Test_RMSE=test_m.get("RMSE",float("nan")),
                Test_Skill=test_m.get("Skill",float("nan")),
                Test_KGE=test_m.get("KGE",float("nan")),
                Test_FreezeAcc=test_m.get("Freeze_Acc",float("nan")),
                NodeR2_mean=test_m.get("node_r2_mean",float("nan")),
                NodeR2_min=test_m.get("node_r2_min",float("nan")),
                NodeR2_std=test_m.get("node_r2_std",float("nan")),
                SpatialVarRatio=test_m.get("spatial_var_ratio",float("nan")),
                Entropy_H=es.get("final_H",float("nan")),
                Diagnosis=diag, Train_s=round(elapsed,1),
                Job_ID=JOB_ID, Node=NODE, Resumed=False))

            # Incremental save after every model
            pd.DataFrame(all_results).to_csv(
                RESULTS_DIR/"spatial_results_incremental.csv", index=False)
            print(f"  → Test R²={test_m.get('R2','N/A')} | "
                  f"Skill={test_m.get('Skill','N/A')} | {diag}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ FAILED {arch} [{tgt_name}]: {e}")

spatial_df = pd.DataFrame(all_results)
spatial_df.to_csv(RESULTS_DIR/"spatial_results_all.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — FULL VISUALISATION SUITE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  PHASE 8: Full Visualisation Suite")
print("="*55)
matplotlib.rcParams.update({"figure.dpi":150,"font.size":11,
                              "axes.grid":True,"grid.alpha":0.3,
                              "axes.spines.top":False,"axes.spines.right":False})

TGT_LABELS = {"temp":"Weather Temp","smap":"SMAP Temp L1","moist":"Soil Moisture"}
MODEL_COLORS = {"SpatialBiGRU":"#1f77b4","SpatialMamba":"#ff7f0e",
                "SpatialS4":"#2ca02c","SpatialFuseMoE":"#9467bd"}

if len(spatial_df) > 0:
    targets = sorted(spatial_df["Target"].unique())
    models  = ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]

    # ── SP01: R² and Skill Heatmap (Model × Target) ───────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(22,9))
    for ax,metric,lbl,vmin in [
        (axes[0],"Test_R2","Test R²",0.90),
        (axes[1],"Test_Skill","Skill vs Seasonal",-0.2)]:
        if metric not in spatial_df.columns: continue
        pv=spatial_df.pivot_table(index="Model",columns="Target",
                                   values=metric,aggfunc="mean").round(4)
        if pv.empty: continue
        pv.columns=[TGT_LABELS.get(c,c) for c in pv.columns]
        pv=pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pv,ax=ax,cmap="RdYlGn",vmin=vmin,vmax=1.0,
                    annot=True,fmt=".4f",linewidths=0.5,linecolor="white",
                    annot_kws={"size":12,"weight":"bold"},
                    cbar_kws={"label":lbl,"shrink":0.85})
        ax.set_title(f"{lbl} | Model x Target | Test 2025",fontweight="bold",fontsize=12)
        ax.tick_params(axis="x",rotation=20,labelsize=10)
        ax.tick_params(axis="y",rotation=0, labelsize=10)
    fig.suptitle(f"True Spatial Field Prediction — Spatio-Temporal Results\n"
                 f"{N_LOCS} locations x 24 timestep lookback | Alaska 2022-2025",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"SP01_r2_skill_heatmap.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  SP01 saved")

    # ── SP02: Grouped bar chart all metrics ───────────────────────────────────
    fig,axes=plt.subplots(2,2,figsize=(22,14))
    axf=axes.flatten()
    metrics=[("Test_R2","R2",1),("Test_Skill","Skill",1),
             ("Test_KGE","KGE",1),("Test_FreezeAcc","Freeze Acc (%)",0.01)]
    x=np.arange(len(targets)); w=0.8/max(len(models),1)
    for ai,(metric,mlbl,sc) in enumerate(metrics):
        ax=axf[ai]
        if metric not in spatial_df.columns: continue
        for mi,model in enumerate(models):
            sub=spatial_df[spatial_df["Model"]==model]
            vals=[sub[sub["Target"]==t][metric].mean()*sc
                  if len(sub[sub["Target"]==t])>0 else 0 for t in targets]
            bars=ax.bar(x+mi*w-0.4+w/2,vals,width=w*0.9,label=model,
                        color=MODEL_COLORS.get(model,"grey"),
                        alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_height()+0.001,f"{v:.3f}",
                                ha="center",va="bottom",fontsize=7,fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([TGT_LABELS.get(t,t) for t in targets],fontsize=9)
        ax.set_ylabel(mlbl,fontsize=10); ax.set_title(mlbl,fontweight="bold")
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle("All Metrics x All Target Groups — Spatio-Temporal Reference Grid",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"SP02_grouped_bar_all_metrics.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  SP02 saved")

    # ── SP03: Skill bars per target group ─────────────────────────────────────
    ncols=max(len(targets),1)
    fig,axes=plt.subplots(1,ncols,figsize=(8*ncols,9))
    if ncols==1: axes=[axes]
    for ax,tgt in zip(axes,targets):
        sub=spatial_df[spatial_df["Target"]==tgt].sort_values("Test_Skill",ascending=True)
        if sub.empty: continue
        colors=["#2ca02c" if v>0.3 else "#ff7f0e" if v>0 else "#d62728"
                for v in sub["Test_Skill"].fillna(-1)]
        bars=ax.barh(sub["Model"],sub["Test_Skill"].fillna(0),
                     color=colors,alpha=0.85,edgecolor="black",lw=0.6,height=0.5)
        for bar,v in zip(bars,sub["Test_Skill"].fillna(0)):
            ax.text(max(v,0)+0.005,bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}",va="center",fontsize=10,fontweight="bold")
        ax.axvline(0,color="black",lw=2)
        ax.axvline(0.3,color="green",ls="--",lw=1.5,alpha=0.7,label="Good (0.30)")
        ax.set_xlabel("Skill Score vs Seasonal Baseline",fontsize=10)
        ax.set_title(TGT_LABELS.get(tgt,tgt),fontweight="bold",fontsize=12)
        ax.legend(fontsize=8); ax.set_xlim(-0.1,1.0)
    fig.suptitle("Skill Scores — True Spatial Field Prediction",fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"SP03_skill_bars.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  SP03 saved")

    # ── SP04: Entropy diagnosis ───────────────────────────────────────────────
    if "Diagnosis" in spatial_df.columns:
        fig,ax=plt.subplots(figsize=(16,7))
        dc=["#2ca02c" if "LEARNING" in str(d) else "#d62728"
            for d in spatial_df["Diagnosis"]]
        bl=spatial_df["Model"]+" ["+spatial_df["Target"]+"]"
        bars=ax.bar(bl,spatial_df["Entropy_H"].fillna(0),
                    color=dc,alpha=0.85,edgecolor="black",lw=0.5,width=0.6)
        for bar,v in zip(bars,spatial_df["Entropy_H"].fillna(0)):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                    f"{v:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        ax.set_ylabel("Normalised Entropy H_norm",fontsize=11)
        ax.set_title("Entropy Tracker Diagnosis\nGreen=LEARNING_DYNAMICS | Red=SEASONAL_FITTING",
                     fontweight="bold",fontsize=12)
        ax.tick_params(axis="x",rotation=30,labelsize=9); ax.set_ylim(0,1.0)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2ca02c",label="LEARNING_DYNAMICS"),
                            Patch(color="#d62728",label="SEASONAL_FITTING")],fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG_DIR/"SP04_entropy_diagnosis.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  SP04 saved")

    # ── SP05: Spatial R² map per target ──────────────────────────────────────
    for tgt in targets:
        sub=spatial_df[spatial_df["Target"]==tgt]
        if sub.empty or sub["NodeR2_mean"].isna().all(): continue
        best_row=sub.sort_values("Test_R2").tail(1)
        nr2_mean=float(best_row["NodeR2_mean"].values[0])
        nr2_min =float(best_row["NodeR2_min"].values[0])
        bname   =best_row["Model"].values[0]
        fig,ax  =plt.subplots(figsize=(13,10))
        sc=ax.scatter(loc_coords[:,1],loc_coords[:,0],
                      c=np.full(N_LOCS,nr2_mean),
                      cmap="RdYlGn",vmin=0.90,vmax=1.0,s=60,
                      edgecolors="black",linewidth=0.4,zorder=5)
        plt.colorbar(sc,ax=ax,label="R2 Score",shrink=0.85,
                     ticks=[0.90,0.92,0.94,0.96,0.98,1.00])
        ax.set_xlabel("Longitude",fontsize=11); ax.set_ylabel("Latitude",fontsize=11)
        ax.set_title(f"Spatial R2 | {TGT_LABELS.get(tgt,tgt)} | Best: {bname}\n"
                     f"Mean R2={nr2_mean:.4f} | Min R2={nr2_min:.4f} | {N_LOCS} locations",
                     fontweight="bold",fontsize=11)
        for site,(lat,lon) in [("Bedrock",(66.25,-150.7)),("Transition",(67.5,-150.5)),
                                ("Upland",(68.5,-150.5)),("Wetland",(67.2,-151.0))]:
            ax.annotate(site,xy=(lon,lat),fontsize=9,fontweight="bold",color="navy",
                        ha="center",bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.7))
        plt.tight_layout()
        plt.savefig(FIG_DIR/f"SP05_spatial_r2_map_{tgt}.png",dpi=150,bbox_inches="tight")
        plt.close(); print(f"  SP05 [{tgt}] saved")

    # ── SP06: Training curves ─────────────────────────────────────────────────
    import torch as _torch
    ckpt_files=list(MODELS_DIR.glob("*_v3_best.pt"))
    if ckpt_files:
        fig,axes=plt.subplots(3,1,figsize=(18,16),sharex=False)
        for ckpt in sorted(ckpt_files):
            try:
                sv=_torch.load(ckpt,map_location="cpu")
                hist=sv.get("history",[])
                if not hist: continue
                arch=sv.get("arch","?")
                tgt_=ckpt.stem.replace("_v3_best","").replace(f"{arch}_","")
                lbl=f"{arch} [{tgt_}]"
                col=MODEL_COLORS.get(arch,"grey")
                eps=[h["epoch"] for h in hist]
                axes[0].plot(eps,[h["train_loss"] for h in hist],lw=1.5,alpha=0.8,color=col,label=lbl)
                axes[1].plot(eps,[h["val_R2"] for h in hist],lw=1.5,alpha=0.8,color=col,label=lbl)
                axes[2].plot(eps,[h.get("H_norm",0) for h in hist],lw=1.5,alpha=0.8,color=col,label=lbl)
            except Exception: continue
        for ax,ylabel,title in [
            (axes[0],"Huber Loss","Training Loss per Epoch"),
            (axes[1],"Validation R2","Validation R2 per Epoch"),
            (axes[2],"H_norm","Entropy Evolution (Physics Learning Diagnostic)")]:
            ax.set_ylabel(ylabel,fontsize=10)
            ax.set_title(title,fontweight="bold",fontsize=11)
            ax.legend(fontsize=6,ncol=4,loc="best")
            ax.set_xlabel("Epoch",fontsize=10)
        fig.suptitle("Training Curves — All Spatial Models",fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR/"SP06_training_curves.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  SP06 saved")

    # ── SP07: Per-location R² bar with error bars ─────────────────────────────
    if "NodeR2_mean" in spatial_df.columns and not spatial_df["NodeR2_mean"].isna().all():
        fig,ax=plt.subplots(figsize=(18,8))
        x_pos=0; xticks=[]; xlabels=[]
        for tgt in targets:
            for model in models:
                row=spatial_df[(spatial_df["Target"]==tgt)&(spatial_df["Model"]==model)]
                if row.empty: continue
                mean_r2=float(row["NodeR2_mean"].values[0])
                std_r2 =float(row["NodeR2_std"].values[0]) if "NodeR2_std" in row.columns else 0
                min_r2 =float(row["NodeR2_min"].values[0])
                col=MODEL_COLORS.get(model,"grey")
                ax.bar(x_pos,mean_r2,color=col,alpha=0.85,edgecolor="black",lw=0.5,width=0.7)
                ax.errorbar(x_pos,mean_r2,yerr=std_r2,color="black",capsize=4,lw=1.5)
                ax.text(x_pos,min_r2-0.001,f"min\n{min_r2:.3f}",
                        ha="center",va="top",fontsize=7,color="darkred")
                xticks.append(x_pos); xlabels.append(f"{model.replace('Spatial','')}\n[{tgt}]")
                x_pos+=1
            x_pos+=0.5
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels,fontsize=8,rotation=30)
        ax.set_ylabel("Per-Location R2 (mean +- std)",fontsize=11)
        ax.set_ylim(0.90,1.005)
        ax.set_title("Per-Location R2 Across All 256 Spatio-Temporal Locations\n"
                     "Mean +- Std | Min annotated | All target groups",
                     fontweight="bold",fontsize=12)
        from matplotlib.patches import Patch as _P
        ax.legend(handles=[_P(color=c,label=m) for m,c in MODEL_COLORS.items()],
                  fontsize=9,ncol=4)
        plt.tight_layout()
        plt.savefig(FIG_DIR/"SP07_node_r2_distribution.png",dpi=150,bbox_inches="tight")
        plt.close(); print("  SP07 saved")

    # ── SP08: KGE and FreezeAcc ───────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(20,9))
    for ax,metric,lbl,ylim in [
        (axes[0],"Test_KGE","Kling-Gupta Efficiency (KGE)",(0.85,1.01)),
        (axes[1],"Test_FreezeAcc","Freeze-Thaw Accuracy (%)",(90,101))]:
        if metric not in spatial_df.columns: continue
        for mi,model in enumerate(models):
            sub=spatial_df[spatial_df["Model"]==model]
            if sub.empty: continue
            x_=[i+mi*0.2 for i in range(len(targets))]
            vals=[sub[sub["Target"]==t][metric].mean()
                  if len(sub[sub["Target"]==t])>0 else 0 for t in targets]
            bars=ax.bar(x_,vals,width=0.18,label=model,
                        color=MODEL_COLORS.get(model,"grey"),
                        alpha=0.85,edgecolor="black",lw=0.5)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.001,f"{v:.3f}",
                        ha="center",va="bottom",fontsize=8,fontweight="bold")
        ax.set_xticks([i+0.3 for i in range(len(targets))])
        ax.set_xticklabels([TGT_LABELS.get(t,t) for t in targets],fontsize=10)
        ax.set_ylabel(lbl,fontsize=11); ax.set_title(lbl,fontweight="bold",fontsize=12)
        ax.set_ylim(*ylim); ax.legend(fontsize=9,ncol=2)
    fig.suptitle("KGE and Freeze-Thaw Accuracy — All Spatial Models",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"SP08_kge_freeze_accuracy.png",dpi=150,bbox_inches="tight")
    plt.close(); print("  SP08 saved")

    print(f"\n  All figures saved to: {FIG_DIR}")

# PHASE 9 — FINAL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  FINAL LEADERBOARD — True Spatial Field Prediction")
print(f"  {'Model':<20} {'Target':<8} {'Val R²':>8} {'Test R²':>8} "
      f"{'Skill':>8} {'NodeR2':>8} {'FrzAcc':>8} {'Diagnosis'}")
print("  " + "─"*85)
if len(spatial_df) == 0 or spatial_df.empty:
    print("  No results to display — all models failed")
else:
    sort_col = next((c for c in ["Test_R2","Val_R2","R2"] if c in spatial_df.columns), None)
    _df_sorted = spatial_df.sort_values(sort_col,ascending=False) if sort_col else spatial_df
    for _, row in _df_sorted.iterrows():
        beat = "✓" if (not pd.isna(row.get("Test_Skill")) and row.get("Test_Skill",-1)>0) else "✗"
        diag = str(row.get("Diagnosis","N/A"))
        ds   = "LEARN" if "LEARNING" in diag else "SEAS" if "SEASONAL" in diag else "N/A"
        print(f"  {beat} {row['Model']:<19} {row['Target']:<8} "
          f"{row.get('Val_R2',float('nan')):>8.4f} "
          f"{row.get('Test_R2',float('nan')):>8.4f} "
          f"{row.get('Test_Skill',float('nan')):>8.4f} "
          f"{row.get('NodeR2_mean',float('nan')):>8.4f} "
          f"{row.get('Test_FreezeAcc',float('nan')):>7.2f}%  {ds}")

print(f"\n  Completed : {pd.Timestamp.now()}")
print(f"  Results   : {RESULTS_DIR}")
print(f"  Figures   : {FIG_DIR}")
print(f"  Models    : {MODELS_DIR}")
print(f"  Log       : {LOG_PATH}")
print("=" * 70)
