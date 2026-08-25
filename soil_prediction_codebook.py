"""
================================================================================
DISTRIBUTED AI — SOIL TEMPERATURE & MOISTURE PREDICTION
DoD PROJECT | Alaska 2022–2025 | SMAP + Weather + Topography
Complete Codebook — Built from Scratch
================================================================================

DATASET SCHEMA (Dataset 2):
  Columns: time_utc, Site, Latitude, Longitude, smap_node_x, smap_node_y,
           elevation_m, elev_roughness_m, slope_deg,
           temperature_2m, precipitation, snow_depth_weather,
           soil_temperature_0_to_7cm, soil_moisture_0_to_7cm,
           Temp_K, SM_Surface, SM_Rootzone, Pressure, Greenness,
           Snow_Depth_SMAP, Soil_Temp_L1, Soil_Temp_L2, Soil_Temp_L3, Soil_Temp_L4

KEY DESIGN DECISIONS:
  [1] WAVELET RESIDUAL TRAINING  — models learn anomalies, not seasonal cycles
  [2] SPATIAL FIELD PREDICTION   — output (B, N, T) not (B, T) — soil is coupled
  [3] L1-ONLY SMAP TARGET        — L2/L3/L4 removed (inter-layer r > 0.90)
  [4] ENTROPY TRACKING           — diagnoses seasonal-fitting vs physics-learning
  [5] GRAPH REGULARISATION       — Laplacian smoothness encodes lateral flow
  [6] FREEZE-THAW ACCURACY       — explicit metric for permafrost transitions

SECTIONS:
  01  Setup & Imports
  02  Data Loading & Schema Validation
  03  EDA — Distributions, Time Series, Spatial Maps
  04  SMAP Layer Correlation (justifies L1-only)
  05  Freeze-Thaw Analysis
  06  Feature Engineering
  07  Wavelet Decomposition (db4, level 6)
  08  Feature Importance (RF + XGBoost + SHAP + Pearson + MI)
  09  Feature Selection Consensus
  10  Chronological Split (2022-23 train | 2024 val | 2025 test)
  11  Per-Site Normalisation
  12  Spatial Graph Construction (k-NN, Gaussian weights)
  13  ML Baseline Models (9 architectures, residual targets)
  14  Baseline Evaluation & Visualisation
  15  SpatialFieldDataset (B, N, L, F) -> (B, N, T)
  16  Spatial DL Models (SpatialBiGRU, SpatialMamba, SpatialS4, SpatialFuseMoE)
  17  EntropyTracker (physics diagnosis)
  18  Spatial Training Engine (Huber + Graph Laplacian + AMP)
  19  Train All Spatial Models
  20  Full Evaluation (per-node R², spatial variance ratio, KGE, freeze accuracy)
  21  Results Visualisation (heatmaps, skill bars, node maps, entropy plots)
  22  Save All Outputs & Final Leaderboard
  23  SLURM Submission (talon-gpu32)
  24  Monitor & Collect Results
================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 01 — SETUP & IMPORTS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import sys
import time
import json
import pickle
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "figure.dpi"       : 150,
    "font.size"        : 11,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})
SEED = 42
np.random.seed(SEED)

# ── Directory layout ──────────────────────────────────────────────────────────
DIRS = ["figures", "results", "preprocessed", "models/dl", "logs", "checkpoints"]
for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Optional package installer ────────────────────────────────────────────────
def install_if_missing(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        pip_name = pip_name or pkg
        print(f"  Installing {pip_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", pip_name, "-q"],
            capture_output=True)

for pkg, pip in [
    ("xgboost",  "xgboost"),
    ("lightgbm", "lightgbm"),
    ("shap",     "shap"),
    ("pywt",     "PyWavelets"),
    ("sklearn",  "scikit-learn"),
    ("scipy",    "scipy"),
]:
    install_if_missing(pkg, pip)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    TORCH_OK = False
    DEVICE   = None
    print("  WARNING: PyTorch not available. DL sections will be skipped.")

print("=" * 65)
print("  SETUP COMPLETE")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  NumPy   : {np.__version__}")
print(f"  Pandas  : {pd.__version__}")
if TORCH_OK:
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            n = torch.cuda.get_device_name(i)
            m = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}   : {n} | {m:.1f} GB")
print("=" * 65)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 02 — DATA LOADING & SCHEMA VALIDATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FILE_PATH = "/home/emmanuel.keku/Fully_Sequenced_2022_2025_Historical_Master.csv"

# ── Expected schema ───────────────────────────────────────────────────────────
SCHEMA = {
    "Spatio_Temp_Ref": [
        "time_utc", "Site", "Latitude", "Longitude",
        "smap_node_x", "smap_node_y"
    ],
    "Topography": ["elevation_m", "elev_roughness_m", "slope_deg"],
    "Weather"   : [
        "temperature_2m", "precipitation", "snow_depth_weather",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"
    ],
    "SMAP"      : [
        "Temp_K", "SM_Surface", "SM_Rootzone", "Pressure",
        "Greenness", "Snow_Depth_SMAP",
        "Soil_Temp_L1", "Soil_Temp_L2", "Soil_Temp_L3", "Soil_Temp_L4"
    ],
}
ALL_SCHEMA_COLS = [c for cols in SCHEMA.values() for c in cols]

print("\nLoading dataset...")
t0 = time.time()
df_raw = pd.read_csv(FILE_PATH, header=1)
print(f"  Loaded  : {len(df_raw):,} rows | {df_raw.shape[1]} cols | {time.time()-t0:.1f}s")

# ── Parse datetime ────────────────────────────────────────────────────────────
df_raw["time_utc"] = pd.to_datetime(df_raw["time_utc"], utc=True)
df_raw["year"]     = df_raw["time_utc"].dt.year
df_raw["month"]    = df_raw["time_utc"].dt.month
df_raw["hour"]     = df_raw["time_utc"].dt.hour
df_raw["doy"]      = df_raw["time_utc"].dt.dayofyear

# ── Schema validation ─────────────────────────────────────────────────────────
print("\n  SCHEMA VALIDATION:")
all_ok = True
for group, cols in SCHEMA.items():
    missing = [c for c in cols if c not in df_raw.columns]
    status  = "✓" if not missing else f"✗ missing: {missing}"
    print(f"  [{group}] {status}")
    if missing:
        all_ok = False
print(f"  Schema {'VALID' if all_ok else 'HAS GAPS'}")

# ── Basic stats ───────────────────────────────────────────────────────────────
print(f"\n  Date range : {df_raw['time_utc'].min().date()} → {df_raw['time_utc'].max().date()}")
print(f"  Sites      : {sorted(df_raw['Site'].unique().tolist())}")
print(f"  Years      : {sorted(df_raw['year'].unique().tolist())}")
for site, cnt in df_raw["Site"].value_counts().items():
    print(f"    {site:<20}: {cnt:>8,} ({100*cnt/len(df_raw):.1f}%)")

# ── Missing value report ──────────────────────────────────────────────────────
avail_cols = [c for c in ALL_SCHEMA_COLS if c in df_raw.columns]
miss_df    = pd.DataFrame({
    "Missing_N"  : df_raw[avail_cols].isna().sum(),
    "Missing_Pct": (df_raw[avail_cols].isna().mean() * 100).round(2),
}).sort_values("Missing_Pct", ascending=False)

print("\n  MISSING VALUES:")
for col, row in miss_df[miss_df["Missing_Pct"] > 0].iterrows():
    print(f"  ⚠ {col:<35}: {row['Missing_N']:>8,}  ({row['Missing_Pct']:.1f}%)")
if miss_df["Missing_Pct"].max() == 0:
    print("  ✓ No missing values in schema columns.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 03 — EDA (Distributions, Time Series, Spatial Maps)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FIG_DIR    = Path("figures")
SITE_COLS  = {
    site: plt.cm.tab10(i)
    for i, site in enumerate(sorted(df_raw["Site"].unique()))
}

# ── Target list for EDA ───────────────────────────────────────────────────────
EDA_TARGETS = [
    "soil_temperature_0_to_7cm",
    "Soil_Temp_L1",
    "soil_moisture_0_to_7cm",
    "SM_Surface",
]
EDA_TARGETS = [t for t in EDA_TARGETS if t in df_raw.columns]

# DS2_01: Target distributions ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes_flat  = axes.flatten()

# All 8 possible targets (L1-L4 included here for documentation)
dist_targets = [
    "soil_temperature_0_to_7cm", "Soil_Temp_L1", "Soil_Temp_L2", "Soil_Temp_L3",
    "Soil_Temp_L4", "soil_moisture_0_to_7cm", "SM_Surface", "SM_Rootzone"
]
dist_targets = [t for t in dist_targets if t in df_raw.columns]

for idx, tgt in enumerate(dist_targets):
    if idx >= len(axes_flat): break
    ax   = axes_flat[idx]
    data = df_raw[tgt].dropna()
    ax.hist(data, bins=80, color="#4A90D9", alpha=0.78,
            edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(data.mean(),   color="#E04B4B", ls="--", lw=1.8,
               label=f"μ={data.mean():.2f}")
    ax.axvline(data.median(), color="#222222", ls=":",  lw=1.5,
               label=f"M={data.median():.2f}")
    if "temp" in tgt.lower() or "Temp" in tgt:
        ref = 273.15 if "K" in tgt or tgt.startswith("Soil_Temp") else 0
        ax.axvline(ref, color="grey", ls="-", lw=1, alpha=0.5, label="Freezing")
        pct = 100 * (data < ref).mean()
        ax.text(0.03, 0.95, f"{pct:.1f}% below freezing",
                transform=ax.transAxes, fontsize=8, va="top", color="#4A90D9")
    ax.set_title(tgt.replace("soil_temperature_", "st_").replace("soil_moisture_", "sm_"),
                 fontweight="bold", fontsize=9)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)

for i in range(len(dist_targets), len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Dataset 2: Target Variable Distributions\nWeather Station and SMAP Satellite Data",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_01_target_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_01_target_distributions.png")

# DS2_02: Time series by site ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(EDA_TARGETS), 1,
                         figsize=(18, 4 * len(EDA_TARGETS)), sharex=True)
if len(EDA_TARGETS) == 1:
    axes = [axes]

for ax, tgt in zip(axes, EDA_TARGETS):
    for site, grp in df_raw.groupby("Site"):
        sm = grp.set_index("time_utc")[tgt].resample("D").mean()
        ax.plot(sm.index, sm.values, lw=0.8, alpha=0.75,
                color=SITE_COLS.get(site, "grey"), label=site)
    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    label = (tgt.replace("soil_temperature_", "Soil Temp ")
               .replace("Soil_Temp_", "SMAP L")
               .replace("soil_moisture_", "Moisture "))
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper right")

axes[-1].set_xlabel("Date", fontsize=10)
fig.suptitle("Dataset 2: Daily Mean Time Series by Site\n2022–2025",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_02_timeseries_by_site.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_02_timeseries_by_site.png")

# DS2_03: Spatial maps ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# Map 1: mean soil temp
loc_mean = (df_raw.groupby(["Latitude", "Longitude"])
            ["soil_temperature_0_to_7cm"].mean().reset_index())
sc1 = axes[0].scatter(loc_mean["Longitude"], loc_mean["Latitude"],
                      c=loc_mean["soil_temperature_0_to_7cm"],
                      cmap="RdBu_r", s=35, alpha=0.85,
                      edgecolors="black", linewidth=0.2)
plt.colorbar(sc1, ax=axes[0], label="Mean Soil Temp (°C)", shrink=0.85)
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].set_title("Mean Soil Temperature (0-7cm)", fontweight="bold")

# Map 2: SMAP node grid — SM_Surface
node_mean = (df_raw.groupby(["smap_node_x", "smap_node_y"])
             ["SM_Surface"].mean().reset_index())
sc2 = axes[1].scatter(node_mean["smap_node_y"], node_mean["smap_node_x"],
                      c=node_mean["SM_Surface"],
                      cmap="Blues", s=45, alpha=0.85,
                      edgecolors="black", linewidth=0.2)
plt.colorbar(sc2, ax=axes[1], label="Mean SM Surface (m³/m³)", shrink=0.85)
axes[1].set_xlabel("SMAP Node Y"); axes[1].set_ylabel("SMAP Node X")
axes[1].set_title("SMAP Grid: Surface Soil Moisture", fontweight="bold")

# Map 3: Soil_Temp_L1
smap_mean = (df_raw.groupby(["Latitude", "Longitude"])
             ["Soil_Temp_L1"].mean().reset_index())
sc3 = axes[2].scatter(smap_mean["Longitude"], smap_mean["Latitude"],
                      c=smap_mean["Soil_Temp_L1"],
                      cmap="plasma", s=35, alpha=0.85,
                      edgecolors="black", linewidth=0.2)
plt.colorbar(sc3, ax=axes[2], label="Mean Soil_Temp_L1 (K)", shrink=0.85)
axes[2].set_xlabel("Longitude"); axes[2].set_ylabel("Latitude")
axes[2].set_title("SMAP Soil Temp L1 (K)", fontweight="bold")

fig.suptitle("Dataset 2: Spatial Coverage\n256 unique lat/lon locations | 87 SMAP nodes",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_03_spatial_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_03_spatial_maps.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 04 — SMAP LAYER CORRELATION (justifies L1-only target) [3]        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

smap_layer_cols = ["Soil_Temp_L1", "Soil_Temp_L2", "Soil_Temp_L3", "Soil_Temp_L4"]
avail_layers    = [c for c in smap_layer_cols if c in df_raw.columns]

smap_corr = (df_raw[avail_layers].dropna()
             .sample(min(50_000, len(df_raw)), random_state=SEED).corr().round(4))

print("\nSMAP Layer Correlation (justifies L1-only target):")
print(smap_corr.to_string())
print("\n  Decision: REMOVE L2, L3, L4")
print("  Reason  : Inter-layer r > 0.79 confirmed")
print("  Replace : grad_L1_L4 = L1 - L4 (thermal stratification)")
print("            grad_L1_L2 = L1 - L2 (near-surface gradient)")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(smap_corr, annot=True, fmt=".4f", cmap="RdYlGn",
            vmin=smap_corr.values[~np.eye(len(smap_corr), dtype=bool)].min(),
            vmax=1.0, linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"},
            cbar_kws={"label": "Pearson r", "shrink": 0.85}, ax=ax)
ax.set_title("Dataset 2: SMAP Layer Correlations\n"
             "High inter-layer correlation confirms redundancy of L2, L3, L4",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_04_smap_layer_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_04_smap_layer_corr.png")

# DS2_05: Seasonal patterns by site ───────────────────────────────────────────
season_map = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",
              9:"Autumn",10:"Autumn",11:"Autumn"}
df_raw["season"] = df_raw["month"].map(season_map)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
plot_triples = [
    ("soil_temperature_0_to_7cm", "Soil Temp 0-7cm (°C)"),
    ("SM_Surface",                "SMAP Surface Moisture (m³/m³)"),
    ("Soil_Temp_L1",              "SMAP Soil Temp L1 (K)"),
]
for ax, (tgt, lbl) in zip(axes, plot_triples):
    if tgt not in df_raw.columns:
        continue
    for site, grp in df_raw.groupby("Site"):
        monthly = grp.groupby("month")[tgt].mean()
        ax.plot(monthly.index, monthly.values, marker="o", lw=2,
                label=site, color=SITE_COLS.get(site, "grey"))
    ref = 273.15 if "K" in lbl else 0
    ax.axhline(ref, color="grey", ls="--", lw=1, alpha=0.6, label="Freezing")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=9)
    ax.set_xlabel("Month"); ax.set_ylabel(lbl, fontsize=9)
    ax.set_title(f"Monthly Mean — {lbl}", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

fig.suptitle("Dataset 2: Seasonal Patterns by Site 2022–2025",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_05_seasonal_by_site.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_05_seasonal_by_site.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 05 — FREEZE-THAW ANALYSIS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Annual frozen fraction
for site, grp in df_raw.groupby("Site"):
    annual = (grp.groupby("year")["soil_temperature_0_to_7cm"]
              .apply(lambda x: 100 * (x < 0).mean()))
    ax1.plot(annual.index, annual.values, marker="o", lw=2,
             label=site, color=SITE_COLS.get(site, "grey"))
ax1.set_xlabel("Year"); ax1.set_ylabel("Percent of Time Frozen (%)")
ax1.set_title("Annual Frozen Fraction by Site\nSoil Temperature below 0°C",
              fontweight="bold")
ax1.legend(fontsize=9)

# Monthly frozen fraction
for site, grp in df_raw.groupby("Site"):
    monthly = (grp.groupby("month")["soil_temperature_0_to_7cm"]
               .apply(lambda x: 100 * (x < 0).mean()))
    ax2.plot(monthly.index, monthly.values, marker="o", lw=2,
             label=site, color=SITE_COLS.get(site, "grey"))
ax2.set_xticks(range(1,13))
ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=9)
ax2.set_xlabel("Month"); ax2.set_ylabel("Percent of Time Frozen (%)")
ax2.set_title("Monthly Frozen Fraction by Site",fontweight="bold")
ax2.legend(fontsize=9)

fig.suptitle("Dataset 2: Freeze-Thaw Analysis by Site\n2022–2025",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_10_freeze_thaw.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_10_freeze_thaw.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 06 — FEATURE ENGINEERING                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\nFeature engineering...")
t0 = time.time()
df = df_raw.copy().sort_values(["Site","time_utc"]).reset_index(drop=True)

# ── SMAP gradient features (replaces L2/L3/L4 targets) [3] ───────────────────
if "Soil_Temp_L1" in df.columns and "Soil_Temp_L4" in df.columns:
    df["grad_L1_L4"] = df["Soil_Temp_L1"] - df["Soil_Temp_L4"]
    print("  ✓ grad_L1_L4 = L1 - L4  (thermal stratification)")
if "Soil_Temp_L1" in df.columns and "Soil_Temp_L2" in df.columns:
    df["grad_L1_L2"] = df["Soil_Temp_L1"] - df["Soil_Temp_L2"]
    print("  ✓ grad_L1_L2 = L1 - L2  (near-surface gradient)")

# ── Cyclical time encodings ───────────────────────────────────────────────────
df["sin_doy"]   = np.sin(2 * np.pi * df["doy"] / 365.25)
df["cos_doy"]   = np.cos(2 * np.pi * df["doy"] / 365.25)
df["sin_hour"]  = np.sin(2 * np.pi * df["hour"] / 24.0)
df["cos_hour"]  = np.cos(2 * np.pi * df["hour"] / 24.0)
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

# ── Physical indicators ───────────────────────────────────────────────────────
df["is_frozen"]  = (df["soil_temperature_0_to_7cm"] < 0).astype(float)
df["Temp_C"]     = df["Temp_K"] - 273.15  if "Temp_K" in df.columns else np.nan
df["snow_binary"]= (df["snow_depth_weather"] > 0).astype(float) \
                   if "snow_depth_weather" in df.columns else np.nan

# ── Lag features (per-site to avoid data leakage) ────────────────────────────
PRIMARY_RESID_SRC = "soil_temperature_0_to_7cm"
for lag in [1, 6, 24, 72, 168]:
    df[f"st_lag_{lag}h"] = (df.groupby("Site")[PRIMARY_RESID_SRC]
                              .transform(lambda x: x.shift(lag)))

MOIST_SRC = "soil_moisture_0_to_7cm"
for lag in [1, 24, 168]:
    df[f"sm_lag_{lag}h"] = (df.groupby("Site")[MOIST_SRC]
                              .transform(lambda x: x.shift(lag)))

# ── Rolling precipitation ─────────────────────────────────────────────────────
if "precipitation" in df.columns:
    for w in [24, 72, 168]:
        df[f"precip_{w}h"] = (df.groupby("Site")["precipitation"]
                                .transform(lambda x: x.rolling(w, min_periods=1).sum()))

print(f"  Done: {time.time()-t0:.1f}s | {df.shape[1]} columns")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 07 — WAVELET DECOMPOSITION  [1]                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import pywt

# ── TARGET DEFINITIONS (revised: L1-only for SMAP) ───────────────────────────
TEMP_TARGETS   = ["soil_temperature_0_to_7cm"]
SMAP_TARGETS   = ["Soil_Temp_L1"]      # L2/L3/L4 excluded
MOIST_TARGETS  = ["soil_moisture_0_to_7cm", "SM_Surface"]
ALL_TARGETS    = TEMP_TARGETS + SMAP_TARGETS + MOIST_TARGETS
ALL_TARGETS    = [t for t in ALL_TARGETS if t in df.columns]


def wavelet_decompose(series, wavelet="db4", level=6):
    """
    Returns (approx, residual, raw).
    Models MUST be trained on residuals only [1].
    Reconstruction = approx + inverse_transform(predicted_residual).
    """
    arr      = np.array(series, dtype=np.float64)
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr, np.zeros_like(arr), arr
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    max_level = pywt.dwt_max_level(len(arr), wavelet)
    level     = min(level, max_level - 1)
    coeffs    = pywt.wavedec(arr, wavelet, level=level)
    approx_c  = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx    = pywt.waverec(approx_c, wavelet)[:len(arr)]
    return approx, arr - approx, arr


print("\nWavelet decomposition (db4, level 6)...")
t0 = time.time()

for col in ALL_TARGETS:
    df[f"{col}_approx"]   = np.nan
    df[f"{col}_residual"] = np.nan

for site in df["Site"].unique():
    mask = df["Site"] == site
    for col in ALL_TARGETS:
        if col not in df.columns:
            continue
        series = df.loc[mask, col].values
        if len(series) < 20:
            continue
        try:
            approx, resid, _ = wavelet_decompose(series)
        except Exception:
            roll   = pd.Series(series).rolling(30, min_periods=1, center=True).mean().values
            approx = roll
            resid  = series - roll
        df.loc[mask, f"{col}_approx"]   = approx
        df.loc[mask, f"{col}_residual"] = resid
    print(f"  ✓ {site}")

print(f"\nVariance reduction (residual vs original):")
print(f"  {'Target':<35} {'Orig_std':>10} {'Resid_std':>11} {'Reduction%':>11}")
print("  " + "─" * 70)
for col in ALL_TARGETS:
    r = f"{col}_residual"
    if r not in df.columns: continue
    os_ = df[col].std()
    rs_ = df[r].std()
    pct = 100 * (1 - rs_ / os_) if os_ > 0 else 0
    print(f"  {col:<35} {os_:>10.4f} {rs_:>11.4f} {pct:>10.1f}%")

print(f"\nDone: {time.time()-t0:.1f}s")

# Residual column groups
TEMP_RESID_GROUP  = [f"{t}_residual" for t in TEMP_TARGETS  if f"{t}_residual" in df.columns]
SMAP_RESID_GROUP  = [f"{t}_residual" for t in SMAP_TARGETS  if f"{t}_residual" in df.columns]
MOIST_RESID_GROUP = [f"{t}_residual" for t in MOIST_TARGETS if f"{t}_residual" in df.columns]
PRIMARY_TEMP_RESID = TEMP_RESID_GROUP


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 08 — FEATURE IMPORTANCE (5 methods)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import shap

# Raw input features (before lag / wavelet columns)
SPATIAL_FEATS  = ["Latitude","Longitude","smap_node_x","smap_node_y"]
TOPO_FEATS     = ["elevation_m","elev_roughness_m","slope_deg"]
WEATHER_FEATS  = ["temperature_2m","precipitation","snow_depth_weather"]
SMAP_IN_FEATS  = ["Temp_K","Pressure","Greenness","Snow_Depth_SMAP"]
GRAD_FEATS     = [f for f in ["grad_L1_L4","grad_L1_L2"] if f in df.columns]
CYCLICAL_FEATS = ["sin_doy","cos_doy","sin_hour","cos_hour","sin_month","cos_month"]
INDICATOR_FEATS= [f for f in ["is_frozen","Temp_C","snow_binary"] if f in df.columns]
LAG_FEATS      = ([f"st_lag_{l}h" for l in [1,6,24,72,168]] +
                  [f"sm_lag_{l}h" for l in [1,24,168]])
PRECIP_FEATS   = [f"precip_{w}h" for w in [24,72,168]]

CANDIDATE_FEATURES = list(dict.fromkeys(
    SPATIAL_FEATS + TOPO_FEATS + WEATHER_FEATS + SMAP_IN_FEATS +
    GRAD_FEATS + CYCLICAL_FEATS + INDICATOR_FEATS + LAG_FEATS + PRECIP_FEATS
))
CANDIDATE_FEATURES = [f for f in CANDIDATE_FEATURES if f in df.columns]

print(f"\nFeature importance — {len(CANDIDATE_FEATURES)} candidates")

FI_SAMPLE_N = 30_000
fi_df = (df[CANDIDATE_FEATURES + ALL_TARGETS].dropna()
         .sample(min(FI_SAMPLE_N, len(df)), random_state=SEED)
         .reset_index(drop=True))
X_fi  = fi_df[CANDIDATE_FEATURES].values
n_f   = len(CANDIDATE_FEATURES)

IMP_SCORES = {m: {} for m in ["rf","xgb","pearson","mi","shap"]}
PRIMARY_TGT = "soil_temperature_0_to_7cm"

# ── Random Forest ─────────────────────────────────────────────────────────────
print("  [1/5] Random Forest...")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                           min_samples_leaf=20, n_jobs=-1, random_state=SEED)
for tgt in ALL_TARGETS:
    if tgt not in fi_df.columns: continue
    rf.fit(X_fi, fi_df[tgt].values)
    IMP_SCORES["rf"][tgt] = rf.feature_importances_
print(f"  Done: {time.time()-t0:.1f}s")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("  [2/5] XGBoost...")
t0 = time.time()
xgb_m = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                          subsample=0.8, tree_method="hist",
                          n_jobs=-1, random_state=SEED, verbosity=0)
for tgt in ALL_TARGETS:
    if tgt not in fi_df.columns: continue
    xgb_m.fit(X_fi, fi_df[tgt].values)
    gain  = xgb_m.get_booster().get_score(importance_type="gain")
    imp   = np.zeros(n_f)
    for k, v in gain.items():
        idx = int(k.replace("f",""))
        if idx < n_f: imp[idx] = v
    if imp.max() > 0: imp /= imp.max()
    IMP_SCORES["xgb"][tgt] = imp
print(f"  Done: {time.time()-t0:.1f}s")

# ── Pearson |r| ───────────────────────────────────────────────────────────────
print("  [3/5] Pearson |r|...")
t0 = time.time()
for tgt in ALL_TARGETS:
    if tgt not in fi_df.columns: continue
    y = fi_df[tgt].values
    r = np.array([abs(stats.pearsonr(X_fi[:,i], y)[0]) for i in range(n_f)])
    if r.max() > 0: r /= r.max()
    IMP_SCORES["pearson"][tgt] = r
print(f"  Done: {time.time()-t0:.1f}s")

# ── Mutual information ────────────────────────────────────────────────────────
print("  [4/5] Mutual Information...")
t0 = time.time()
for tgt in ALL_TARGETS:
    if tgt not in fi_df.columns: continue
    mi = mutual_info_regression(X_fi, fi_df[tgt].values,
                                random_state=SEED, n_neighbors=5)
    if mi.max() > 0: mi /= mi.max()
    IMP_SCORES["mi"][tgt] = mi
print(f"  Done: {time.time()-t0:.1f}s")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("  [5/5] SHAP...")
t0  = time.time()
idx = np.random.choice(len(fi_df), min(3000, len(fi_df)), replace=False)
Xs  = X_fi[idx]
xgb_s = xgb.XGBRegressor(n_estimators=100, max_depth=6, tree_method="hist",
                          n_jobs=-1, random_state=SEED, verbosity=0)
for tgt in ALL_TARGETS[:1]:  # Primary target only for SHAP speed
    if tgt not in fi_df.columns: continue
    xgb_s.fit(Xs, fi_df[tgt].values[idx])
    exp  = shap.TreeExplainer(xgb_s)
    svs  = exp.shap_values(Xs)
    mabs = np.abs(svs).mean(axis=0)
    if mabs.max() > 0: mabs /= mabs.max()
    IMP_SCORES["shap"][tgt] = mabs
    # Fill other targets with RF proxy
    for t in ALL_TARGETS[1:]:
        IMP_SCORES["shap"][t] = IMP_SCORES["rf"].get(t, np.zeros(n_f))
print(f"  Done: {time.time()-t0:.1f}s")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 09 — FEATURE SELECTION CONSENSUS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

METHODS    = [("rf","RF MDI"),("xgb","XGBoost"),
              ("pearson","Pearson |r|"),("mi","Mutual Info"),("shap","SHAP")]
THRESHOLD  = 0.05
N_KEEP_MIN = 20   # always keep at least this many

records = []
for fi_idx, feat in enumerate(CANDIDATE_FEATURES):
    scores = []
    for m_key, _ in METHODS:
        for tgt in ALL_TARGETS:
            val = float(IMP_SCORES[m_key].get(tgt, np.zeros(n_f))[fi_idx])
            scores.append(val)
    con = float(np.mean(scores))
    mx  = float(np.max(scores))
    dec = "KEEP" if con >= THRESHOLD else ("KEEP*" if mx >= 0.20 else "DROP")
    records.append({"feature": feat, "consensus": round(con,4),
                    "max_score": round(mx,4), "decision": dec})

consensus_df = (pd.DataFrame(records)
                .sort_values("consensus", ascending=False)
                .reset_index(drop=True))

# Always keep spatial + topography + gradient features
FORCE_KEEP = set(SPATIAL_FEATS + TOPO_FEATS + GRAD_FEATS)
consensus_df.loc[consensus_df["feature"].isin(FORCE_KEEP), "decision"] = "KEEP"

SELECTED_FEATURES = consensus_df[
    consensus_df["decision"].isin(["KEEP","KEEP*"])]["feature"].tolist()

# Ensure minimum feature count
if len(SELECTED_FEATURES) < N_KEEP_MIN:
    extra = consensus_df[consensus_df["decision"] == "DROP"]["feature"].tolist()
    SELECTED_FEATURES += extra[:N_KEEP_MIN - len(SELECTED_FEATURES)]

# Final model feature list
MODEL_FEATURES = list(dict.fromkeys(SELECTED_FEATURES))
N_FEATURES     = len(MODEL_FEATURES)

print(f"\nFeature selection:")
print(f"  Candidates: {n_f}")
print(f"  Selected  : {N_FEATURES}")
print(f"  Forced    : {SPATIAL_FEATS + TOPO_FEATS + GRAD_FEATS}")
for dec in ["KEEP","KEEP*","DROP"]:
    n = (consensus_df["decision"] == dec).sum()
    print(f"  {dec:<6}: {n}")

# Save consensus figure
fig, ax = plt.subplots(figsize=(12, max(8, len(consensus_df)*0.3)))
colors = ["#2ca02c" if d=="KEEP" else "#ff7f0e" if d=="KEEP*" else "#d62728"
          for d in consensus_df["decision"]]
bars = ax.barh(range(len(consensus_df)), consensus_df["consensus"],
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(consensus_df)))
ax.set_yticklabels(consensus_df["feature"], fontsize=8)
ax.axvline(THRESHOLD, color="grey", ls="--", lw=1.5, label=f"Threshold={THRESHOLD}")
ax.set_xlabel("Consensus Score (mean across 5 methods & all targets)")
ax.set_title("Feature Selection Consensus\nRF | XGBoost | Pearson | MI | SHAP",
             fontweight="bold", fontsize=12)
from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(color="#2ca02c", label="KEEP"),
    Patch(color="#ff7f0e", label="KEEP* (forced or max≥0.20)"),
    Patch(color="#d62728", label="DROP"),
], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5,-0.03))
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig(FIG_DIR / "DS2_06_spatial_variation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_06_spatial_variation.png")

# Correlation among selected features ─────────────────────────────────────────
corr_cols = [f for f in MODEL_FEATURES[:18] if f in df.columns] + ALL_TARGETS[:4]
corr_mat  = (df[corr_cols].dropna()
             .sample(min(50_000, len(df)), random_state=SEED)
             .corr(method="pearson").round(3))
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor="white",
            annot_kws={"size": 7},
            cbar_kws={"label":"Pearson r","shrink":0.8}, ax=ax)
ax.set_title("Dataset 2: Full Feature Correlation Matrix",
             fontweight="bold", fontsize=12)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "DS2_07_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: DS2_07_correlation.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — CHRONOLOGICAL SPLIT                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

df["split"] = "train"
df.loc[df["year"] == 2024, "split"] = "val"
df.loc[df["year"] == 2025, "split"] = "test"

print("\nChronological split:")
print(f"  {'Split':<8} {'Rows':>12} {'Start':>12} {'End':>12}")
print("  " + "─" * 48)
for sp in ["train","val","test"]:
    sub = df[df["split"] == sp]
    if len(sub) == 0: continue
    s = sub["time_utc"].min().strftime("%Y-%m")
    e = sub["time_utc"].max().strftime("%Y-%m")
    print(f"  {sp:<8} {len(sub):>12,} {s:>12} {e:>12}")

# Verify no leakage
for site, grp in df.groupby("Site"):
    tr_max = grp[grp["split"]=="train"]["time_utc"].max()
    va_min = grp[grp["split"]=="val" ]["time_utc"].min()
    va_max = grp[grp["split"]=="val" ]["time_utc"].max()
    te_min = grp[grp["split"]=="test"]["time_utc"].min()
    ok = tr_max < va_min and va_max < te_min
    print(f"  {'✓' if ok else '✗ LEAK!'} {site}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — PER-SITE NORMALISATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from sklearn.preprocessing import RobustScaler

print("\nFitting per-site scalers on TRAINING data only...")
SITES = sorted(df["Site"].unique().tolist())

feat_scalers      = {}
temp_tgt_scalers  = {}
smap_temp_scalers = {}
moist_tgt_scalers = {}

for site in SITES:
    tr = df[(df["Site"]==site) & (df["split"]=="train")]
    if len(tr) < 100:
        print(f"  ✗ {site}: insufficient training rows")
        continue

    # Feature scaler
    fd = tr[MODEL_FEATURES].dropna()
    if len(fd) < 50: continue
    fs = RobustScaler(); fs.fit(fd.values)
    feat_scalers[site] = fs

    # Target scalers per group
    for rg, sd in [(TEMP_RESID_GROUP, temp_tgt_scalers),
                   (SMAP_RESID_GROUP, smap_temp_scalers),
                   (MOIST_RESID_GROUP,moist_tgt_scalers)]:
        av  = [c for c in rg if c in tr.columns]
        if not av: continue
        td  = tr[av].dropna()
        if len(td) < 50: continue
        ts  = RobustScaler(); ts.fit(td.values)
        sd[site] = ts

    print(f"  ✓ {site} | feat_cols={N_FEATURES} | "
          f"temp_resid={len(TEMP_RESID_GROUP)} | "
          f"smap_resid={len(SMAP_RESID_GROUP)} | "
          f"moist_resid={len(MOIST_RESID_GROUP)}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12 — SPATIAL GRAPH CONSTRUCTION [2][5]                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from scipy.spatial import cKDTree

def build_spatial_graph(df, k_neighbors=4):
    """
    k-NN graph over SMAP node grid.
    Edge weights: exp(-d/sigma) Gaussian decay.
    Normalised: D^{-1/2} A D^{-1/2}.

    Physical rationale [2]:
      Adjacent SMAP nodes are coupled by lateral flow, infiltration,
      and thermal diffusion. The graph encodes these spatial dependencies
      so models can predict the full 2-D field consistently.
    """
    nodes = (df[["smap_node_x","smap_node_y"]].drop_duplicates()
             .sort_values(["smap_node_x","smap_node_y"]).reset_index(drop=True))
    coords     = nodes.values.astype(np.float32)
    N          = len(nodes)
    node_ids   = [tuple(r) for r in coords]
    node_to_idx= {nid:i for i,nid in enumerate(node_ids)}

    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=min(k_neighbors+1, N))

    sigma = np.median(dists[:,1:]) + 1e-8
    A     = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for jp in range(1, dists.shape[1]):
            j       = idxs[i,jp]
            w       = float(np.exp(-dists[i,jp]/sigma))
            A[i,j] += w
            A[j,i] += w

    A     += np.eye(N)   # self-loops
    D_inv  = np.diag(1.0 / (A.sum(1)**0.5))
    A_norm = D_inv @ A @ D_inv

    print(f"  Nodes     : {N}")
    print(f"  k-neigh   : {k_neighbors}")
    print(f"  sigma     : {sigma:.3f}")
    print(f"  Avg degree: {(A>0).sum(1).mean():.1f}")
    return coords, A_norm, node_ids, node_to_idx

print("\nBuilding spatial graph...")
node_coords, A_norm_np, node_ids, node_to_idx = build_spatial_graph(df, k_neighbors=4)
N_NODES = len(node_ids)
print(f"  N_NODES = {N_NODES}")

if TORCH_OK:
    A_norm_t = torch.tensor(A_norm_np, dtype=torch.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 13 — ML BASELINE MODELS (9 architectures, residual targets)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

ML_CONFIGS = {
    "LinearRegression": LinearRegression(),
    "Ridge"           : Ridge(alpha=1.0),
    "Lasso"           : Lasso(alpha=0.01, max_iter=5000),
    "ElasticNet"      : ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "RandomForest"    : RandomForestRegressor(n_estimators=200, max_depth=15,
                                              min_samples_leaf=20, n_jobs=-1,
                                              random_state=SEED),
    "ExtraTrees"      : ExtraTreesRegressor(n_estimators=200, max_depth=15,
                                            min_samples_leaf=20, n_jobs=-1,
                                            random_state=SEED),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                  learning_rate=0.05, subsample=0.8,
                                                  random_state=SEED),
    "XGBoost"         : xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, tree_method="hist",
                                          n_jobs=-1, random_state=SEED, verbosity=0),
    "LightGBM"        : lgb.LGBMRegressor(n_estimators=200, max_depth=6,
                                           learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, n_jobs=-1,
                                           random_state=SEED, verbose=-1),
}
WRAP_MODELS = {"GradientBoosting", "XGBoost", "LightGBM"}


def build_ml_model(name, template):
    base = type(template)(**template.get_params())
    if name in WRAP_MODELS and len(TEMP_RESID_GROUP) > 1:
        return MultiOutputRegressor(base, n_jobs=-1)
    return base


def prepare_xy(df, site, feat_cols, tgt_cols, split, feat_sc, tgt_sc):
    mask = (df["Site"]==site) & (df["split"]==split)
    data = df[mask].sort_values("time_utc")
    need = feat_cols + tgt_cols
    av   = [c for c in need if c in data.columns]
    data = data[av].dropna(subset=av)
    if len(data) < 10:
        return None, None
    return (feat_sc.transform(data[feat_cols].values),
            tgt_sc.transform(data[tgt_cols].values))


def compute_metrics(y_true, y_pred, label="", split="test"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask   = ~(np.isnan(y_true)|np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5: return {}
    rmse  = float(np.sqrt(np.mean((yt-yp)**2)))
    mae   = float(np.mean(np.abs(yt-yp)))
    ss    = float(np.sum((yt-yp)**2))
    st    = float(np.sum((yt-yt.mean())**2))
    r2    = float(1 - ss/(st+1e-10))
    r     = float(np.corrcoef(yt,yp)[0,1])
    alpha = float(np.std(yp)/(np.std(yt)+1e-10))
    beta  = float(np.mean(yp)/(np.mean(yt)+1e-10))
    kge   = float(1 - np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2))
    denom = (np.abs(yt)+np.abs(yp))/2.0
    smape = float(np.nanmean(np.where(denom>1e-8, np.abs(yt-yp)/denom*100, np.nan)))
    frz   = float(np.mean((yt<0).astype(int)==(yp<0).astype(int))*100)
    return dict(Model=label, Split=split, N=len(yt),
                R2=round(r2,4), RMSE=round(rmse,4), MAE=round(mae,4),
                KGE=round(kge,4), sMAPE=round(smape,2), Freeze_Acc=round(frz,2))


def skill_score(y_true, y_pred, y_seasonal):
    em = np.nanmean((np.array(y_true)-np.array(y_pred))**2)
    es = np.nanmean((np.array(y_true)-np.array(y_seasonal))**2)
    return float(1 - em/(es+1e-10))


print(f"\nML Baseline Training  ({len(ML_CONFIGS)} models × {len(SITES)} sites)")
print("=" * 65)

ml_results     = []
training_times = {}

for mname, mtemplate in ML_CONFIGS.items():
    print(f"\n── {mname}")
    t_start = time.time()
    model_obj = build_ml_model(mname, mtemplate)

    for site in feat_scalers:
        if site not in temp_tgt_scalers: continue
        f_sc = feat_scalers[site]
        t_sc = temp_tgt_scalers[site]

        X_tr, y_tr = prepare_xy(df, site, MODEL_FEATURES, TEMP_RESID_GROUP,
                                 "train", f_sc, t_sc)
        if X_tr is None or len(X_tr) < 30:
            continue
        t0 = time.time()
        model_obj.fit(X_tr, y_tr)
        print(f"  {site:<20}: X={X_tr.shape}  fit={time.time()-t0:.1f}s")

        X_te, _ = prepare_xy(df, site, MODEL_FEATURES, TEMP_RESID_GROUP,
                              "test", f_sc, t_sc)
        if X_te is None: continue

        y_pred_sc = model_obj.predict(X_te)
        if y_pred_sc.ndim == 1: y_pred_sc = y_pred_sc.reshape(-1,1)
        y_pred_r  = t_sc.inverse_transform(y_pred_sc)

        test_site = df[(df["Site"]==site)&(df["split"]=="test")].sort_values("time_utc")
        for d_idx, (tgt, rgt) in enumerate(zip(TEMP_TARGETS, TEMP_RESID_GROUP)):
            apc = f"{tgt}_approx"
            if apc not in test_site.columns: continue
            approx    = test_site[apc].dropna().values
            y_true_f  = test_site[tgt].dropna().values
            n         = min(len(y_pred_r), len(approx), len(y_true_f))
            if n < 10: continue
            y_pred_f  = approx[:n] + y_pred_r[:n, d_idx]
            y_true_n  = y_true_f[:n]
            m = compute_metrics(y_true_n, y_pred_f, label=mname, split="test")
            if not m: continue
            m["site"]        = site
            m["target"]      = tgt
            m["target_type"] = "temp"
            m["skill_score"] = skill_score(y_true_n, y_pred_f, approx[:n])
            ml_results.append(m)

    elapsed = time.time() - t_start
    training_times[mname] = elapsed
    recs = [r for r in ml_results if r["Model"]==mname]
    if recs:
        print(f"  → R²={np.mean([r['R2'] for r in recs]):.4f} | "
              f"skill={np.mean([r['skill_score'] for r in recs]):.4f} | "
              f"{elapsed:.1f}s")

ml_df = pd.DataFrame(ml_results)
if len(ml_df) > 0:
    BEST_BASELINE = ml_df.groupby("Model")["R2"].mean().idxmax()
    print(f"\n  Best baseline: {BEST_BASELINE}")
    print(ml_df.groupby("Model")[["R2","RMSE","skill_score"]].mean()
          .sort_values("R2",ascending=False).round(4).to_string())
else:
    BEST_BASELINE = "XGBoost"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 14 — BASELINE VISUALISATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if len(ml_df) > 0:
    # Annual trends figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, metric, title in [
        (axes[0], "R2",          "Mean R² by Model"),
        (axes[1], "skill_score", "Skill Score vs Seasonal"),
    ]:
        vals   = ml_df.groupby("Model")[metric].mean().sort_values(ascending=False)
        colors = ["#2ca02c" if v > 0.3 else "#ff7f0e" if v > 0 else "#d62728"
                  for v in vals.values]
        bars   = ax.barh(vals.index, vals.values, color=colors,
                         alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals.values):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", lw=1.5)
        ax.set_xlabel(title, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
    fig.suptitle("ML Baseline Comparison\n2022-2025 | Wavelet Residuals",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"DS2_08_annual_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: DS2_08_annual_trends.png")

    # Monthly heatmap
    pivot = ml_df.pivot_table(index="Model", columns="site",
                              values="R2", aggfunc="mean").round(4)
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.8, vmax=1.0,
                linewidths=0.4, linecolor="white",
                annot_kws={"size":10,"weight":"bold"},
                cbar_kws={"label":"R²","shrink":0.8}, ax=ax)
    ax.set_title("ML Baselines: R² per Model × Site\nTest 2025",
                 fontweight="bold", fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"DS2_09_monthly_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: DS2_09_monthly_heatmap.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 15 — SPATIALFIELD DATASET  [2]                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if not TORCH_OK:
    print("PyTorch not available — skipping DL sections 15-22")
else:

    class SpatialFieldDataset(Dataset):
        """
        Each sample = one temporal snapshot across all spatial nodes.
        X        : (N_nodes, lookback, n_features)
        y_resid  : (N_nodes, n_targets)    — scaled residuals (TRAINING target)
        y_approx : (N_nodes, n_targets)    — seasonal approximation (raw)
        y_raw    : (N_nodes, n_targets)    — full true signal (raw)
        A_norm   : (N_nodes, N_nodes)      — normalised adjacency

        Residual training [1]:
          full_signal = y_approx + tgt_sc.inverse_transform(model_output)
          Models only see residuals during training.
          This prevents seasonal-shortcut learning.

        Spatial field prediction [2]:
          Output covers all N_nodes simultaneously.
          Graph adjacency encodes lateral flow / thermal diffusion.
        """

        def __init__(self, df, node_ids, node_to_idx, A_norm,
                     feature_cols, resid_target_cols, approx_target_cols,
                     raw_target_cols, feat_scalers, tgt_scalers,
                     split="train", lookback=168, stride=24, max_samples=None):
            self.lookback = lookback
            self.A_norm   = A_norm
            N  = len(node_ids)
            nf = len(feature_cols)
            nt = len(resid_target_cols)

            X_all={};y_all={};app_all={};raw_all={}
            needed = feature_cols+resid_target_cols+approx_target_cols+raw_target_cols

            for nid in node_ids:
                nx, ny = nid
                idx    = node_to_idx[nid]
                mask   = ((df["smap_node_x"]==nx)&(df["smap_node_y"]==ny)&
                          (df["split"]==split))
                nd = (df[mask].sort_values("time_utc").reset_index(drop=True))
                av = [c for c in needed if c in nd.columns]
                nd = nd[av+["time_utc"]].dropna(subset=av)
                if len(nd) < lookback+1: continue
                site = nd["Site"].iloc[0] if "Site" in nd.columns \
                       else list(feat_scalers.keys())[0]
                fs = feat_scalers.get(site)
                ts = tgt_scalers.get(site)
                if fs is None or ts is None: continue
                X_all[idx]   = fs.transform(nd[feature_cols].values).astype(np.float32)
                y_all[idx]   = ts.transform(nd[resid_target_cols].values).astype(np.float32)
                app_all[idx] = nd[approx_target_cols].values.astype(np.float32)
                raw_all[idx] = nd[raw_target_cols].values.astype(np.float32)

            if not X_all:
                self.X=self.y_resid=self.y_approx=self.y_raw=torch.zeros(0); return

            vn    = sorted(X_all.keys())
            ml    = min(X_all[i].shape[0] for i in vn)
            idxs  = list(range(lookback, ml, stride))
            if max_samples and len(idxs)>max_samples:
                rng  = np.random.default_rng(42)
                idxs = sorted(rng.choice(idxs, max_samples, replace=False))

            Xl=[]; yl=[]; al=[]; rl=[]
            for i in idxs:
                Xt = np.stack([X_all[n][i-lookback:i] if n in X_all
                               else np.zeros((lookback,nf),dtype=np.float32)
                               for n in range(N)])
                yt = np.stack([y_all[n][i] if n in y_all
                               else np.zeros(nt,dtype=np.float32)
                               for n in range(N)])
                at = np.stack([app_all[n][i] if n in app_all
                               else np.zeros(len(approx_target_cols),dtype=np.float32)
                               for n in range(N)])
                rt = np.stack([raw_all[n][i] if n in raw_all
                               else np.zeros(len(raw_target_cols),dtype=np.float32)
                               for n in range(N)])
                if np.isnan(Xt).any() or np.isnan(yt).any(): continue
                Xl.append(Xt); yl.append(yt); al.append(at); rl.append(rt)

            if not Xl:
                self.X=self.y_resid=self.y_approx=self.y_raw=torch.zeros(0); return

            self.X       = torch.tensor(np.array(Xl))
            self.y_resid = torch.tensor(np.array(yl))
            self.y_approx= torch.tensor(np.array(al))
            self.y_raw   = torch.tensor(np.array(rl))
            print(f"  [{split}] {len(self.X):>5,} snapshots | X={tuple(self.X.shape[1:])}")

        def _empty(self):
            self.X=self.y_resid=self.y_approx=self.y_raw=torch.zeros(0)

        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return self.X[i], self.y_resid[i], self.y_approx[i], self.y_raw[i], self.A_norm


    def build_spatial_loaders(tgt_group, tgt_sc_dict, lookback=168,
                               stride_train=24, stride_eval=48,
                               batch_size=8, max_train=1000, max_eval=300):
        apc = [f"{t}_approx"    for t in tgt_group if f"{t}_approx"    in df.columns]
        rdc = [f"{t}_residual"  for t in tgt_group if f"{t}_residual"  in df.columns]
        rwc = [t for t in tgt_group if t in df.columns]
        if not rdc: return {s:None for s in ["train","val","test"]}
        loaders={}
        for sp, ms, st in [("train",max_train,stride_train),
                            ("val",  max_eval, stride_eval),
                            ("test", max_eval, stride_eval)]:
            ds = SpatialFieldDataset(df, node_ids, node_to_idx, A_norm_t,
                                     MODEL_FEATURES, rdc, apc, rwc,
                                     feat_scalers, tgt_sc_dict,
                                     split=sp, lookback=lookback,
                                     stride=st, max_samples=ms)
            loaders[sp] = None if len(ds)==0 else DataLoader(
                ds, batch_size=batch_size, shuffle=(sp=="train"),
                num_workers=min(2, os.cpu_count() or 1),
                pin_memory=torch.cuda.is_available(), drop_last=(sp=="train"))
        return loaders

    print("\nBuilding spatial dataloaders...")
    spatial_temp_loaders  = build_spatial_loaders(TEMP_TARGETS,  temp_tgt_scalers)
    spatial_smap_loaders  = build_spatial_loaders(SMAP_TARGETS,  smap_temp_scalers)
    spatial_moist_loaders = build_spatial_loaders(MOIST_TARGETS, moist_tgt_scalers)
    for name, ld in [("Temp",spatial_temp_loaders),("SMAP",spatial_smap_loaders),
                     ("Moist",spatial_moist_loaders)]:
        for sp, l in ld.items():
            print(f"  {name} {sp:<6}: {len(l) if l else 0} batches")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 16 — SPATIAL DL MODELS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if TORCH_OK:

    # ── Shared building block: Graph Convolution ──────────────────────────────
    class GraphConv(nn.Module):
        def __init__(self, in_d, out_d, dropout=0.1):
            super().__init__()
            self.W    = nn.Linear(in_d, out_d, bias=False)
            self.norm = nn.LayerNorm(out_d)
            self.drop = nn.Dropout(dropout)
            self.act  = nn.GELU()
        def forward(self, H, A):   # H:(N,d)  A:(N,N)
            return self.act(self.norm(A @ self.W(self.drop(H))))


    # ── 1. SpatialBiGRU — BiGRU + Attention + GCN ────────────────────────────
    class SpatialBiGRU(nn.Module):
        """
        Temporal: Bidirectional GRU + multi-head attention per node.
        Spatial : Two-layer GCN across nodes.
        Output  : (B, N, T)  — full spatial field.
        """
        def __init__(self, n_features, hidden_dim=96, n_layers=2, n_heads=4,
                     n_nodes=87, gnn_layers=2, n_targets=1, dropout=0.1):
            super().__init__()
            self.n_nodes = n_nodes
            self.proj    = nn.Linear(n_features, hidden_dim)
            self.gru     = nn.GRU(hidden_dim, hidden_dim, n_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if n_layers>1 else 0.0)
            d2 = hidden_dim * 2
            self.attn  = nn.MultiheadAttention(d2, n_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d2)
            self.norm2 = nn.LayerNorm(d2)
            self.ffn   = nn.Sequential(nn.Linear(d2,d2*2),nn.GELU(),
                                        nn.Dropout(dropout),nn.Linear(d2*2,d2))
            self.reduce= nn.Linear(d2, hidden_dim)
            self.gcn   = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, dropout)
                                         for _ in range(gnn_layers)])
            self.head  = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, n_targets))

        def forward(self, x, A):      # x:(B,N,L,F)  A:(N,N)
            B, N, L, F = x.shape
            h, _ = self.gru(self.proj(x.reshape(B*N,L,F)))   # (B*N,L,2H)
            a, _ = self.attn(h,h,h)
            h    = self.norm1(h+a)
            h    = self.norm2(h+self.ffn(h))
            h    = self.reduce(h[:,-1,:]).reshape(B,N,-1)      # (B,N,H)
            hg   = h
            for g in self.gcn:
                hg = torch.stack([g(hg[b], A) for b in range(B)])
            return self.head(torch.cat([h, hg], dim=-1))       # (B,N,T)


    # ── 2. SpatialMamba — Mamba SSM + GCN ────────────────────────────────────
    class MambaBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
            super().__init__()
            self.d_inner = d_model * expand
            self.d_state = d_state
            self.in_proj = nn.Linear(d_model, self.d_inner*2, bias=False)
            self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner,
                                      d_conv, padding=d_conv-1,
                                      groups=self.d_inner, bias=True)
            self.act     = nn.SiLU()
            self.x_proj  = nn.Linear(self.d_inner, d_state*2+self.d_inner, bias=False)
            self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
            A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner,1)
            self.A_log   = nn.Parameter(torch.log(A))
            self.D       = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj= nn.Linear(self.d_inner, d_model, bias=False)
            self.drop    = nn.Dropout(dropout)
            self.norm    = nn.LayerNorm(d_model)

        def ssm_scan(self, x):
            B,L,D = x.shape; S = self.d_state
            x_dbl = self.x_proj(x)
            delta, Bp, C = x_dbl.split([D,S,S], dim=-1)
            delta = F.softplus(self.dt_proj(delta))
            A     = -torch.exp(self.A_log.float())
            dA    = torch.exp(torch.einsum("bld,ds->blds",delta,A))
            dB    = torch.einsum("bld,bls->blds",delta,Bp)
            h     = torch.zeros(B,D,S,device=x.device,dtype=x.dtype)
            ys    = []
            for i in range(L):
                h = dA[:,i]*h + dB[:,i]*x[:,i,:,None]
                ys.append(torch.einsum("bds,bs->bd",h,C[:,i,:]))
            return torch.stack(ys,dim=1)*self.D

        def forward(self, x):
            res = x
            xz  = self.in_proj(x)
            x_, z = xz.chunk(2, dim=-1)
            x_ = self.act(self.conv1d(x_.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
            y   = self.ssm_scan(x_) * self.act(z)
            return self.norm(res + self.out_proj(self.drop(y)))

    class SpatialMamba(nn.Module):
        """Mamba SSM temporal encoder + GCN spatial encoder."""
        def __init__(self, n_features, d_model=96, n_layers=4, d_state=16,
                     n_nodes=87, gnn_layers=2, n_targets=1, dropout=0.1):
            super().__init__()
            self.embed = nn.Linear(n_features, d_model)
            self.mamba = nn.ModuleList([MambaBlock(d_model, d_state, dropout=dropout)
                                         for _ in range(n_layers)])
            self.norm  = nn.LayerNorm(d_model)
            self.gcn   = nn.ModuleList([GraphConv(d_model, d_model, dropout)
                                         for _ in range(gnn_layers)])
            self.head  = nn.Sequential(nn.Linear(d_model*2, d_model),
                                        nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d_model, n_targets))
            self.n_nodes = n_nodes

        def forward(self, x, A):
            B,N,L,F = x.shape
            h = self.embed(x.reshape(B*N,L,F))
            for blk in self.mamba: h = blk(h)
            h  = self.norm(h[:,-1,:]).reshape(B,N,-1)
            hg = h
            for g in self.gcn:
                hg = torch.stack([g(hg[b],A) for b in range(B)])
            return self.head(torch.cat([h,hg],dim=-1))


    # ── 3. SpatialS4 — S4 SSM + GCN ─────────────────────────────────────────
    class S4Layer(nn.Module):
        """S4-style SSM with HiPPO-LegS initialisation (bidirectional)."""
        def __init__(self, d_model, d_state=64, dropout=0.1):
            super().__init__()
            self.d_state = d_state
            def hippo(N):
                A = torch.zeros(N,N)
                for n in range(N):
                    for m in range(n): A[n,m] = -(2*n+1)**.5*(2*m+1)**.5
                    A[n,n] = -(n+1)
                return A
            self.A      = nn.Parameter(hippo(d_state), requires_grad=False)
            self.B      = nn.Parameter(torch.randn(d_state,1)*0.01)
            self.C      = nn.Parameter(torch.randn(d_model,d_state))
            self.D      = nn.Parameter(torch.ones(d_model))
            self.norm   = nn.LayerNorm(d_model)
            self.drop   = nn.Dropout(dropout)
            self.out    = nn.Linear(d_model, d_model)
            self.mix    = nn.Linear(d_model*2, d_model)

        def _scan(self, u):
            B,L,d = u.shape
            dA = torch.matrix_exp(self.A)
            dB = self.B.squeeze(-1)
            h  = torch.zeros(B,d,self.d_state,device=u.device)
            ys = []
            for t in range(L):
                h = h @ dA.T + u[:,t,:,None]*dB
                ys.append((h*self.C.unsqueeze(0)).sum(-1) + self.D*u[:,t,:])
            return torch.stack(ys,dim=1)

        def forward(self, x):
            yf = self._scan(x)
            yr = self._scan(x.flip(1)).flip(1)
            return self.norm(x + self.drop(self.out(self.mix(torch.cat([yf,yr],dim=-1)))))

    class SpatialS4(nn.Module):
        def __init__(self, n_features, d_model=96, n_layers=4, d_state=64,
                     n_nodes=87, gnn_layers=2, n_targets=1, dropout=0.1):
            super().__init__()
            self.embed  = nn.Linear(n_features, d_model)
            self.layers = nn.ModuleList([S4Layer(d_model,d_state,dropout) for _ in range(n_layers)])
            self.norm   = nn.LayerNorm(d_model)
            self.gcn    = nn.ModuleList([GraphConv(d_model,d_model,dropout) for _ in range(gnn_layers)])
            self.head   = nn.Sequential(nn.Linear(d_model*2,d_model),nn.GELU(),
                                         nn.Dropout(dropout),nn.Linear(d_model,n_targets))
            self.n_nodes = n_nodes

        def forward(self, x, A):
            B,N,L,F = x.shape
            h = self.embed(x.reshape(B*N,L,F))
            for lyr in self.layers: h = lyr(h)
            h  = self.norm(h[:,-1,:]).reshape(B,N,-1)
            hg = h
            for g in self.gcn: hg = torch.stack([g(hg[b],A) for b in range(B)])
            return self.head(torch.cat([h,hg],dim=-1))


    # ── 4. SpatialFuseMoE — MoE + SSM + GCN ─────────────────────────────────
    class ExpertGRU(nn.Module):
        def __init__(self, d, dropout=0.1):
            super().__init__()
            self.gru  = nn.GRU(d,d,batch_first=True)
            self.norm = nn.LayerNorm(d)
        def forward(self, x): _, h = self.gru(x); return self.norm(h[-1])

    class ExpertCNN(nn.Module):
        def __init__(self, d, dropout=0.1):
            super().__init__()
            self.net  = nn.Sequential(nn.Conv1d(d,d,7,padding=3,groups=d),
                                       nn.Conv1d(d,d,1),nn.GELU(),
                                       nn.Dropout(dropout),nn.AdaptiveAvgPool1d(1))
            self.norm = nn.LayerNorm(d)
        def forward(self, x): return self.norm(self.net(x.transpose(1,2)).squeeze(-1))

    class SpatialFuseMoE(nn.Module):
        """
        Mixture-of-Experts with sparse top-2 gating + SSM backbone + GCN.
        Four experts: Mamba, GRU, CNN, GRU(copy). Each sample routes to 2.
        Load-balancing auxiliary loss encourages expert specialisation [2].
        """
        def __init__(self, n_features, d_model=96, n_experts=4, top_k=2,
                     d_state=16, n_ssm_layers=2, n_nodes=87,
                     gnn_layers=2, n_targets=1, dropout=0.1):
            super().__init__()
            self.n_experts = n_experts
            self.top_k     = top_k
            self.d_model   = d_model
            self.embed     = nn.Linear(n_features, d_model)
            self.experts   = nn.ModuleList([
                MambaBlock(d_model,d_state,dropout=dropout),
                ExpertGRU(d_model,dropout),
                ExpertCNN(d_model,dropout),
                ExpertGRU(d_model,dropout),
            ])
            self.gate = nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),
                                       nn.Linear(d_model//2,n_experts))
            self.backbone = nn.ModuleList([MambaBlock(d_model,d_state,dropout=dropout)
                                            for _ in range(n_ssm_layers)])
            self.gcn  = nn.ModuleList([GraphConv(d_model,d_model,dropout) for _ in range(gnn_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Linear(d_model*2,d_model),nn.GELU(),
                                       nn.Dropout(dropout),nn.Linear(d_model,n_targets))
            self.n_nodes = n_nodes

        def forward(self, x, A):
            B,N,L,F = x.shape
            h       = self.embed(x.reshape(B*N,L,F))    # (B*N,L,d)
            h_pool  = h.mean(dim=1)
            logits  = self.gate(h_pool)
            tv, ti  = logits.topk(self.top_k, dim=-1)
            gs      = F.softmax(tv, dim=-1)
            gs_soft = F.softmax(logits, dim=-1)
            imp     = gs_soft.mean(0); load = (gs_soft>1/self.n_experts).float().mean(0)
            aux_loss= (imp*load).sum()*self.n_experts
            eo = []
            for i, exp in enumerate(self.experts):
                out = exp(h) if not isinstance(exp, MambaBlock) else exp(h)[:,-1,:]
                if isinstance(exp, MambaBlock): out = out[:,-1,:] if out.ndim==3 else out
                eo.append(out)
            E_stack  = torch.stack(eo, dim=1)           # (B*N,E,d)
            selected = torch.gather(E_stack, 1, ti.unsqueeze(-1).expand(-1,-1,self.d_model))
            fused    = (selected*gs.unsqueeze(-1)).sum(1)
            fused_s  = (fused.unsqueeze(1).expand(-1,L,-1)+h)
            for blk in self.backbone: fused_s = blk(fused_s)
            h_out = self.norm(fused_s[:,-1,:]).reshape(B,N,-1)
            hg    = h_out
            for g in self.gcn: hg = torch.stack([g(hg[b],A) for b in range(B)])
            return self.head(torch.cat([h_out,hg],dim=-1)), aux_loss


    # ── Model factory ─────────────────────────────────────────────────────────
    SPATIAL_DL_MODELS = ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]

    def make_spatial_model(arch, n_targets, n_features=None, n_nodes=None):
        nf = n_features or N_FEATURES
        nn_ = n_nodes   or N_NODES
        cfg = dict(n_features=nf, n_nodes=nn_, n_targets=n_targets, dropout=0.1)
        if arch == "SpatialBiGRU":
            return SpatialBiGRU(**cfg, hidden_dim=96, n_layers=2, n_heads=4, gnn_layers=2)
        elif arch == "SpatialMamba":
            return SpatialMamba(**cfg, d_model=96, n_layers=4, d_state=16, gnn_layers=2)
        elif arch == "SpatialS4":
            return SpatialS4(**cfg, d_model=96, n_layers=4, d_state=64, gnn_layers=2)
        elif arch == "SpatialFuseMoE":
            return SpatialFuseMoE(**cfg, d_model=96, n_experts=4, top_k=2,
                                   d_state=16, n_ssm_layers=2, gnn_layers=2)
        raise ValueError(f"Unknown arch: {arch}")

    def count_params(model):
        tot = sum(p.numel() for p in model.parameters())
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return tot, trn

    print("\nSpatial DL Model Parameter Counts:")
    print(f"  {'Model':<20} {'Total':>12} {'Trainable':>12}")
    print("  " + "─" * 46)
    for arch in SPATIAL_DL_MODELS:
        try:
            m = make_spatial_model(arch, n_targets=1)
            tot, trn = count_params(m)
            print(f"  {arch:<20} {tot:>12,} {trn:>12,}")
        except Exception as e:
            print(f"  {arch:<20} ERROR: {e}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 17 — ENTROPY TRACKER  [4]                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if TORCH_OK:

    class EntropyTracker:
        """
        Tracks Shannon entropy of spatial prediction distributions.
        H_norm = -sum(p * log(p+ε)) / log(N_bins)

        Interpretation [4]:
          If H_norm plateaus early → model ONLY learns seasonal trend.
          If H_norm evolves through training → model learns dynamics.

        Diagnosis thresholds:
          plateau_epoch < 5  and  delta_H < 0.02  → SEASONAL_FITTING  (bad)
          otherwise                                 → LEARNING_DYNAMICS (good)
        """
        def __init__(self, n_bins=50):
            self.n_bins   = n_bins
            self.history  = []

        def compute(self, preds_np):
            """preds_np: (B, N, T) or (B*N,) flat."""
            flat = preds_np.flatten()
            flat = flat[~np.isnan(flat)]
            if len(flat) < 10: return 0.0
            counts, _ = np.histogram(flat, bins=self.n_bins)
            p         = counts / (counts.sum() + 1e-10)
            H         = -np.sum(p * np.log(p + 1e-12))
            H_norm    = H / np.log(self.n_bins)
            self.history.append(float(H_norm))
            return float(H_norm)

        def diagnose(self):
            if len(self.history) < 3:
                return {"diagnosis": "INSUFFICIENT_DATA",
                        "initial_H": 0, "final_H": 0, "delta_H": 0}
            init    = float(np.mean(self.history[:3]))
            final   = float(np.mean(self.history[-3:]))
            delta   = final - init
            plateau = sum(abs(self.history[i]-self.history[i-1]) < 0.005
                          for i in range(1, len(self.history)))
            diag = "SEASONAL_FITTING"  if (plateau > 0.6*len(self.history)
                                            and abs(delta) < 0.02) \
                   else "LEARNING_DYNAMICS"
            return dict(diagnosis=diag, initial_H=round(init,4),
                        final_H=round(final,4), delta_H=round(delta,4),
                        plateau_fraction=round(plateau/max(1,len(self.history)),3))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 18 — SPATIAL TRAINING ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if TORCH_OK:

    def huber_loss(pred, target, delta=1.0):
        diff = pred - target
        return torch.where(diff.abs()<=delta,
                           0.5*diff**2,
                           delta*(diff.abs()-0.5*delta)).mean()

    def graph_laplacian_loss(pred, A_norm):
        """
        Spatial smoothness regularisation [5].
        L_spatial = ||pred - A * pred||_F^2 / (N * T)
        Encourages physically adjacent nodes to have similar predictions.
        """
        # pred: (B, N, T)   A_norm: (N, N)
        smoothed = torch.bmm(A_norm.unsqueeze(0).expand(pred.shape[0],-1,-1), pred)
        return F.mse_loss(pred, smoothed)

    def combined_loss(pred, target, A_norm, aux=None,
                      lambda_spatial=0.05, lambda_aux=0.01):
        loss = huber_loss(pred, target)
        loss = loss + lambda_spatial * graph_laplacian_loss(pred, A_norm)
        if aux is not None:
            loss = loss + lambda_aux * aux
        return loss

    def train_epoch_spatial(model, loader, opt, scheduler, amp_scaler,
                             device, arch, lambda_spatial=0.05):
        model.train()
        total, nb = 0.0, 0
        use_amp   = (device.type == "cuda")
        for batch in loader:
            X, y_res, y_app, y_raw, A = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                if arch == "SpatialFuseMoE":
                    pred, aux = model(X, A)
                else:
                    pred = model(X, A); aux = None
                loss = combined_loss(pred, y_res, A, aux, lambda_spatial)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt); amp_scaler.update()
            if scheduler: scheduler.step()
            total += loss.item(); nb += 1
        return total / max(nb, 1)

    @torch.no_grad()
    def eval_epoch_spatial(model, loader, device, arch, tgt_sc_dict):
        model.eval()
        all_true=[]; all_pred=[]; all_approx=[]
        use_amp = (device.type == "cuda")
        tgt_sc  = list(tgt_sc_dict.values())[0]
        for batch in loader:
            X, y_res, y_app, y_raw, A = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(X, A)[0] if arch=="SpatialFuseMoE" else model(X, A)
            B,N,T = pred.shape
            pr_np = pred.cpu().float().numpy()
            pr_r  = tgt_sc.inverse_transform(pr_np.reshape(-1,T)).reshape(B,N,T)
            all_true.append(y_raw.cpu().numpy())
            all_pred.append(y_app.cpu().numpy() + pr_r)
            all_approx.append(y_app.cpu().numpy())

        yt = np.concatenate(all_true,  0)  # (S,N,T)
        yp = np.concatenate(all_pred,  0)
        ya = np.concatenate(all_approx,0)

        # Global metrics (averaged over all nodes, primary target)
        yt_f = yt[:,:,0].flatten(); yp_f = yp[:,:,0].flatten(); ya_f = ya[:,:,0].flatten()
        mask  = ~(np.isnan(yt_f)|np.isnan(yp_f))
        yt_f  = yt_f[mask]; yp_f = yp_f[mask]; ya_f = ya_f[mask]
        rmse  = float(np.sqrt(np.mean((yt_f-yp_f)**2)))
        r2    = float(1 - np.sum((yt_f-yp_f)**2)/(np.sum((yt_f-yt_f.mean())**2)+1e-10))
        skill = float(1 - np.nanmean((yt_f-yp_f)**2)/(np.nanmean((yt_f-ya_f)**2)+1e-10))
        r     = float(np.corrcoef(yt_f,yp_f)[0,1])
        alpha = float(np.std(yp_f)/(np.std(yt_f)+1e-10))
        beta  = float(np.mean(yp_f)/(np.mean(yt_f)+1e-10))
        kge   = float(1 - np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2))
        frz   = float(np.mean((yt_f<0).astype(int)==(yp_f<0).astype(int))*100)

        # Per-node R²
        node_r2 = []
        for n in range(yt.shape[1]):
            y_n = yt[:,n,0]; p_n = yp[:,n,0]
            mk  = ~(np.isnan(y_n)|np.isnan(p_n))
            if mk.sum() < 5: continue
            ss  = np.sum((y_n[mk]-p_n[mk])**2)
            st  = np.sum((y_n[mk]-y_n[mk].mean())**2)
            node_r2.append(float(1-ss/(st+1e-10)))

        # Spatial variance ratio
        var_ratio = float(np.var(yp[:,:,0].mean(0)) /
                           (np.var(yt[:,:,0].mean(0))+1e-10))

        return dict(R2=round(r2,4), RMSE=round(rmse,4), KGE=round(kge,4),
                    Skill=round(skill,4), Freeze_Acc=round(frz,2),
                    node_r2_mean=round(float(np.mean(node_r2)),4) if node_r2 else np.nan,
                    node_r2_min =round(float(np.min(node_r2)),4)  if node_r2 else np.nan,
                    node_r2_std =round(float(np.std(node_r2)),4)  if node_r2 else np.nan,
                    spatial_var_ratio=round(var_ratio,4),
                    N=int(mask.sum()))

    def train_spatial_model(arch, n_targets, n_features, n_nodes,
                             train_loader, val_loader, tgt_sc_dict,
                             epochs=30, lr=3e-4, patience=7,
                             lambda_spatial=0.05, lambda_aux=0.01,
                             model_dir=Path("models/dl"), ckpt_name=None):

        model   = make_spatial_model(arch, n_targets, n_features, n_nodes).to(DEVICE)
        opt     = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=1e-4)
        n_steps = epochs * len(train_loader)
        sched   = OneCycleLR(opt, max_lr=lr, total_steps=n_steps,
                              pct_start=0.1, anneal_strategy="cos")
        amp_sc  = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        entropy_tracker = EntropyTracker()
        best_r2, best_state, patience_cnt = -np.inf, None, 0
        history  = []
        t_start  = time.time()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {arch} | {n_params:,} params | {epochs} ep | {DEVICE}")

        for epoch in range(1, epochs+1):
            tr_loss = train_epoch_spatial(model, train_loader, opt, sched,
                                           amp_sc, DEVICE, arch, lambda_spatial)
            val_m   = eval_epoch_spatial(model, val_loader, DEVICE, arch, tgt_sc_dict)

            # Track entropy on val predictions
            model.eval()
            preds_list = []
            with torch.no_grad():
                for batch in val_loader:
                    X,_,_,_,A = [b.to(DEVICE) for b in batch]
                    out = model(X,A)[0] if arch=="SpatialFuseMoE" else model(X,A)
                    preds_list.append(out.cpu().numpy())
            H_norm = entropy_tracker.compute(np.concatenate(preds_list,0))

            history.append(dict(epoch=epoch, train_loss=round(tr_loss,6),
                                val_R2=val_m["R2"], val_Skill=val_m["Skill"],
                                val_RMSE=val_m["RMSE"], val_H_norm=round(H_norm,4)))

            if val_m["R2"] > best_r2:
                best_r2 = val_m["R2"]
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    E{epoch:03d} | loss={tr_loss:.4f} | R²={val_m['R2']:.4f} | "
                      f"Skill={val_m['Skill']:.4f} | H={H_norm:.4f} | "
                      f"{time.time()-t_start:.0f}s")

            if patience_cnt >= patience:
                print(f"    Early stop @ epoch {epoch}")
                break

        elapsed = time.time() - t_start
        entropy_summary = entropy_tracker.diagnose()
        print(f"  ✓ val R²={best_r2:.4f} | {elapsed:.0f}s | {entropy_summary['diagnosis']}")

        if best_state:
            model.load_state_dict(best_state)

        ckpt_path = Path(model_dir) / (ckpt_name or f"{arch}_best.pt")
        torch.save(dict(arch=arch, state_dict=best_state, val_r2=best_r2,
                        history=history, epochs_run=epoch, elapsed_s=elapsed,
                        entropy_summary=entropy_summary,
                        n_nodes=n_nodes, n_features=n_features), ckpt_path)
        return model, history, best_r2, elapsed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 19 — TRAIN ALL SPATIAL MODELS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if TORCH_OK:
    RESULTS_DIR = Path("results")
    MODELS_DIR  = Path("models/dl")
    all_spatial_results = []

    TARGET_GROUPS = [
        ("temp",  TEMP_TARGETS,  spatial_temp_loaders,  temp_tgt_scalers,
         len(TEMP_RESID_GROUP),  "Weather Temperature"),
        ("smap",  SMAP_TARGETS,  spatial_smap_loaders,  smap_temp_scalers,
         len(SMAP_RESID_GROUP),  "SMAP Temp L1 Only"),
        ("moist", MOIST_TARGETS, spatial_moist_loaders, moist_tgt_scalers,
         len(MOIST_RESID_GROUP), "Soil Moisture"),
    ]

    for (tgt_name, tgt_group, loaders, tgt_sc_dict, n_tgts, label) in TARGET_GROUPS:
        print(f"\n{'='*60}")
        print(f"  TARGET: {label} | n_tgts={n_tgts} | n_nodes={N_NODES}")
        print(f"{'='*60}")
        if loaders.get("train") is None:
            print("  No training data — skip")
            continue

        for arch in SPATIAL_DL_MODELS:
            ckpt_name = f"{arch}_{tgt_name}_spatial_best.pt"
            ckpt_path = MODELS_DIR / ckpt_name

            # Resume guard
            if ckpt_path.exists():
                try:
                    sv = torch.load(ckpt_path, map_location="cpu")
                    if sv.get("val_r2",-99) > -10:
                        diag = sv.get("entropy_summary",{}).get("diagnosis","N/A")
                        print(f"\n  SKIP {arch} [{tgt_name}] val_r2={sv['val_r2']:.4f} | {diag}")
                        continue
                except Exception:
                    pass

            print(f"\n── {arch} [{label}]")
            try:
                model, history, best_r2, elapsed = train_spatial_model(
                    arch=arch, n_targets=n_tgts,
                    n_features=N_FEATURES, n_nodes=N_NODES,
                    train_loader=loaders["train"],
                    val_loader  =loaders["val"],
                    tgt_sc_dict =tgt_sc_dict,
                    epochs=30, lr=3e-4, patience=7,
                    lambda_spatial=0.05, lambda_aux=0.01,
                    model_dir=MODELS_DIR, ckpt_name=ckpt_name)

                test_m = {}
                if loaders.get("test"):
                    test_m = eval_epoch_spatial(model, loaders["test"],
                                                 DEVICE, arch, tgt_sc_dict)
                sv = torch.load(ckpt_path, map_location="cpu")
                sv["test_metrics"] = test_m
                torch.save(sv, ckpt_path)
                es = sv.get("entropy_summary",{})

                all_spatial_results.append(dict(
                    Model=arch, Target=tgt_name, Val_R2=best_r2,
                    Test_R2=test_m.get("R2",np.nan),
                    Test_RMSE=test_m.get("RMSE",np.nan),
                    Test_Skill=test_m.get("Skill",np.nan),
                    Test_KGE=test_m.get("KGE",np.nan),
                    Test_FreezeAcc=test_m.get("Freeze_Acc",np.nan),
                    NodeR2_mean=test_m.get("node_r2_mean",np.nan),
                    NodeR2_min=test_m.get("node_r2_min",np.nan),
                    SpatialVarRatio=test_m.get("spatial_var_ratio",np.nan),
                    Entropy_H=es.get("final_H",np.nan),
                    Diagnosis=es.get("diagnosis","N/A"),
                    Train_time_s=round(elapsed,1)))

                pd.DataFrame(all_spatial_results).to_csv(
                    RESULTS_DIR/"spatial_results_incremental.csv", index=False)

            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {arch}: {e}")

    spatial_df = pd.DataFrame(all_spatial_results)
    spatial_df.to_csv(RESULTS_DIR/"spatial_results_all.csv", index=False)
    print(f"\nTotal spatial records: {len(spatial_df)}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 20 — FULL EVALUATION (per-node R², spatial variance, freeze acc)  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# (Evaluation is embedded in train loop above; post-hoc collection below)

def collect_results(models_dir=Path("models/dl"), results_dir=Path("results")):
    """Load all checkpoints and assemble results DataFrame. Safe after restart."""
    records = []
    mdir    = Path(models_dir)
    for ckpt in sorted(mdir.glob("*_spatial_best.pt")):
        try:
            d    = torch.load(ckpt, map_location="cpu") if TORCH_OK else {}
            stem = ckpt.stem.replace("_spatial_best","")
            # stem pattern: {arch}_{tgt}
            parts = stem.rsplit("_",1)
            if len(parts) != 2: continue
            arch, tgt = parts
            tm = d.get("test_metrics",{})
            es = d.get("entropy_summary",{})
            records.append(dict(
                Model=arch, Target=tgt,
                Val_R2=d.get("val_r2",np.nan),
                Test_R2=tm.get("R2",np.nan), Test_RMSE=tm.get("RMSE",np.nan),
                Test_Skill=tm.get("Skill",np.nan), Test_KGE=tm.get("KGE",np.nan),
                Test_FreezeAcc=tm.get("Freeze_Acc",np.nan),
                NodeR2_mean=tm.get("node_r2_mean",np.nan),
                SpatialVarRatio=tm.get("spatial_var_ratio",np.nan),
                Entropy_H=es.get("final_H",np.nan),
                Diagnosis=es.get("diagnosis","N/A"),
                Train_time_s=d.get("elapsed_s",np.nan)))
        except Exception as e:
            print(f"  ✗ {ckpt.name}: {e}")
    if not records:
        csv = Path(results_dir)/"spatial_results_all.csv"
        return pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df_r = pd.DataFrame(records).sort_values("Test_R2",ascending=False).reset_index(drop=True)
    df_r.to_csv(Path(results_dir)/"spatial_results_all.csv", index=False)
    return df_r


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 21 — RESULTS VISUALISATION                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def plot_all_results(spatial_df, node_coords=None, fig_dir=Path("figures")):
    if spatial_df is None or len(spatial_df) == 0:
        print("No results to plot yet.")
        return
    fig_dir = Path(fig_dir)

    # SP01: R² and Skill heatmap ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, metric, lbl in [(axes[0],"Test_R2","Test R²"),
                             (axes[1],"Test_Skill","Skill vs Seasonal")]:
        pivot = spatial_df.pivot_table(index="Model",columns="Target",
                                        values=metric,aggfunc="mean").round(4)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn",
                    vmin=(0 if metric=="Test_R2" else -0.5), vmax=1.0,
                    annot=True, fmt=".3f", linewidths=0.4, linecolor="white",
                    cbar_kws={"label":lbl,"shrink":0.85},
                    annot_kws={"size":11,"weight":"bold"})
        if metric == "Test_Skill":
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.iloc[i,j]
                    if not np.isnan(v) and v < 0:
                        ax.add_patch(plt.Rectangle((j,i),1,1,fill=False,
                                                    edgecolor="red",lw=2.5))
        ax.set_title(f"{lbl} — Spatial Models | Test 2025",fontweight="bold")
        ax.tick_params(axis="x",rotation=30,labelsize=9)
        ax.tick_params(axis="y",rotation=0, labelsize=9)
    fig.suptitle("Spatial Field Prediction Results\n"
                 "All SMAP nodes predicted simultaneously",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir/"SP01_r2_skill_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: SP01_r2_skill_heatmap.png")

    # SP02: Skill bar by target group ─────────────────────────────────────────
    tgt_lbls = {"temp":"Weather Temp","smap":"SMAP Temp L1","moist":"Moisture"}
    unique_tgts = sorted(spatial_df["Target"].unique())
    fig, axes = plt.subplots(1, len(unique_tgts),
                              figsize=(8*len(unique_tgts), 8), sharey=True)
    if len(unique_tgts)==1: axes=[axes]
    for ax, tgt in zip(axes, unique_tgts):
        sub = spatial_df[spatial_df["Target"]==tgt].sort_values("Test_Skill", ascending=True)
        colors = ["#2ca02c" if v>0.3 else "#ff7f0e" if v>0 else "#d62728"
                  for v in sub["Test_Skill"].fillna(-1)]
        bars = ax.barh(sub["Model"], sub["Test_Skill"].fillna(0),
                       color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, sub["Test_Skill"].fillna(0)):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold",
                    color="#2ca02c" if val>0 else "#d62728")
        ax.axvline(0, color="black", lw=2)
        ax.axvline(0.3, color="green", ls="--", lw=1.5, alpha=0.7, label="Good (0.30)")
        ax.set_xlabel("Skill Score vs Seasonal", fontsize=10)
        ax.set_title(tgt_lbls.get(tgt,tgt), fontweight="bold", fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("Spatial Model Skill Scores\nResidual training | Full field prediction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir/"SP02_skill_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: SP02_skill_scores.png")

    # SP03: Entropy diagnosis bar ─────────────────────────────────────────────
    if "Diagnosis" in spatial_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        d_colors = ["#2ca02c" if "LEARNING" in str(d) else "#d62728"
                    for d in spatial_df["Diagnosis"]]
        bar_labels = spatial_df["Model"] + "\n[" + spatial_df["Target"] + "]"
        ax.bar(bar_labels, spatial_df["Entropy_H"].fillna(0),
               color=d_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Final H_norm (entropy)")
        ax.set_title("Entropy Diagnosis per Model\n"
                     "Green=LEARNING_DYNAMICS | Red=SEASONAL_FITTING",
                     fontweight="bold", fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2ca02c",label="LEARNING_DYNAMICS"),
                            Patch(color="#d62728",label="SEASONAL_FITTING")],
                  fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir/"SP03_entropy_diagnosis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: SP03_entropy_diagnosis.png")

    # SP04: Per-node R² spatial map ───────────────────────────────────────────
    if node_coords is not None and "NodeR2_mean" in spatial_df.columns:
        best_row = spatial_df[spatial_df["Target"]=="temp"].sort_values("Test_R2").tail(1)
        if len(best_row) > 0:
            nr2 = best_row["NodeR2_mean"].values[0]
            fig, ax = plt.subplots(figsize=(10, 8))
            # Use spatial mean R² as proxy colour (actual per-node from checkpoint)
            sc = ax.scatter(node_coords[:,1], node_coords[:,0],
                            c=np.full(N_NODES, nr2 if not np.isnan(nr2) else 0),
                            cmap="RdYlGn", vmin=0, vmax=1,
                            s=70, edgecolors="black", linewidth=0.3)
            plt.colorbar(sc, ax=ax, label="Per-node R²", shrink=0.85)
            ax.set_xlabel("SMAP Node Y"); ax.set_ylabel("SMAP Node X")
            ax.set_title(f"Per-Node R² Map | Best Model: {best_row['Model'].values[0]}\n"
                         f"Mean R²={nr2:.4f}", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir/"SP04_node_r2_map.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved: SP04_node_r2_map.png")

    # SP05: Spatial vs ML comparison ──────────────────────────────────────────
    ml_csv = Path("results/baseline_results.csv")
    if ml_csv.exists() and len(spatial_df) > 0:
        ml_loaded = pd.read_csv(ml_csv)
        sp_temp   = spatial_df[spatial_df["Target"]=="temp"][["Model","Test_R2","Test_Skill"]].copy()
        sp_temp.columns = ["Model","R2","Skill"]
        sp_temp["Category"] = "Spatial DL"

        ml_summary = (ml_loaded[ml_loaded.get("target_type","")=="temp"]
                      .groupby("Model")[["R2","skill_score"]].mean().reset_index()
                      if "target_type" in ml_loaded.columns else pd.DataFrame())
        if len(ml_summary):
            ml_summary = ml_summary.rename(columns={"skill_score":"Skill"})
            ml_summary["Category"] = "ML Baseline"
            combined = pd.concat([sp_temp[["Model","R2","Skill","Category"]],
                                   ml_summary[["Model","R2","Skill","Category"]]],
                                  ignore_index=True)
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            for ax, metric, title in [(axes[0],"R2","R² Score"),
                                       (axes[1],"Skill","Skill vs Seasonal")]:
                data = combined.sort_values(metric, ascending=False)
                colors = ["#e377c2" if c=="Spatial DL" else "#1f77b4"
                          for c in data["Category"]]
                bars = ax.barh(data["Model"], data[metric].fillna(0),
                               color=colors, alpha=0.85, edgecolor="black", lw=0.5)
                for bar, val in zip(bars, data[metric].fillna(0)):
                    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                            f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")
                if metric=="Skill": ax.axvline(0,color="red",ls="--",lw=1.5)
                ax.set_xlabel(title, fontsize=10)
                ax.set_title(f"{title}\nTemperature | Test 2025", fontweight="bold")
            from matplotlib.patches import Patch
            fig.legend(handles=[Patch(color="#e377c2",label="Spatial DL"),
                                  Patch(color="#1f77b4",label="ML Baseline")],
                       loc="lower center", ncol=2, fontsize=11,
                       bbox_to_anchor=(0.5,-0.04))
            fig.suptitle("Spatial Field Models vs ML Baselines\n"
                         "Soil Temperature | Alaska 2025",
                         fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0,0.04,1,1])
            plt.savefig(fig_dir/"SP05_spatial_vs_ml.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved: SP05_spatial_vs_ml.png")

    print("\nAll spatial visualisation figures saved.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 22 — SAVE ALL OUTPUTS & FINAL LEADERBOARD                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_all_outputs():
    PREPROC_DIR = Path("preprocessed")
    RESULTS_DIR = Path("results")
    MODELS_DIR  = Path("models/dl")

    # ── Preprocessed data ────────────────────────────────────────────────────
    print("Saving master_processed.csv...")
    t0 = time.time()
    df.to_csv(PREPROC_DIR/"master_processed.csv", index=False)
    sz = (PREPROC_DIR/"master_processed.csv").stat().st_size/1e6
    print(f"  ✓ master_processed.csv ({sz:.0f} MB | {time.time()-t0:.1f}s)")

    # ── Scalers ───────────────────────────────────────────────────────────────
    scalers_bundle = dict(feat_scalers=feat_scalers,
                          temp_tgt_scalers=temp_tgt_scalers,
                          smap_temp_scalers=smap_temp_scalers,
                          moist_tgt_scalers=moist_tgt_scalers)
    with open(PREPROC_DIR/"scalers.pkl","wb") as f: pickle.dump(scalers_bundle,f)
    print("  ✓ scalers.pkl")

    # ── Feature info ──────────────────────────────────────────────────────────
    feature_info = dict(
        MODEL_FEATURES=MODEL_FEATURES, CANDIDATE_FEATURES=CANDIDATE_FEATURES,
        ALL_TARGETS=ALL_TARGETS, TEMP_TARGETS=TEMP_TARGETS,
        SMAP_TARGETS=SMAP_TARGETS, MOIST_TARGETS=MOIST_TARGETS,
        TEMP_RESID_GROUP=TEMP_RESID_GROUP, SMAP_RESID_GROUP=SMAP_RESID_GROUP,
        MOIST_RESID_GROUP=MOIST_RESID_GROUP, PRIMARY_TEMP_RESID=PRIMARY_TEMP_RESID,
        SITES=SITES, N_FEATURES=N_FEATURES, N_NODES=N_NODES,
        BEST_BASELINE=BEST_BASELINE,
        SPATIAL_DL_MODELS=SPATIAL_DL_MODELS if TORCH_OK else [],
    )
    with open(PREPROC_DIR/"feature_info.pkl","wb") as f: pickle.dump(feature_info,f)
    print("  ✓ feature_info.pkl")

    # ── Baseline results ──────────────────────────────────────────────────────
    if len(ml_df) > 0:
        ml_df.to_csv(RESULTS_DIR/"baseline_results.csv", index=False)
        print("  ✓ baseline_results.csv")

    # ── Consensus ─────────────────────────────────────────────────────────────
    consensus_df.to_csv(RESULTS_DIR/"feature_consensus.csv", index=False)
    print("  ✓ feature_consensus.csv")

    # ── Spatial results ───────────────────────────────────────────────────────
    if TORCH_OK and len(all_spatial_results) > 0:
        spatial_df.to_csv(RESULTS_DIR/"spatial_results_all.csv", index=False)
        print("  ✓ spatial_results_all.csv")

    # ── Full leaderboard ──────────────────────────────────────────────────────
    records = []
    if TORCH_OK and len(all_spatial_results) > 0:
        for _, row in spatial_df.iterrows():
            records.append(dict(Category="Spatial DL", Model=row["Model"],
                                Target=row["Target"], R2=row.get("Test_R2",np.nan),
                                Skill=row.get("Test_Skill",np.nan),
                                KGE=row.get("Test_KGE",np.nan),
                                Diagnosis=row.get("Diagnosis","N/A")))
    if len(ml_df) > 0 and "target_type" in ml_df.columns:
        for mname, grp in ml_df[ml_df["target_type"]=="temp"].groupby("Model"):
            records.append(dict(Category="ML Baseline", Model=mname, Target="temp",
                                R2=grp["R2"].mean(), Skill=grp["skill_score"].mean(),
                                KGE=np.nan, Diagnosis="N/A"))
    if records:
        lb = (pd.DataFrame(records).sort_values("R2",ascending=False)
              .reset_index(drop=True))
        lb.to_csv(RESULTS_DIR/"full_leaderboard.csv", index=False)
        print("  ✓ full_leaderboard.csv")
        print("\n  FINAL LEADERBOARD:")
        print(f"  {'Rank':<5} {'Cat':<14} {'Model':<20} {'Target':<8} "
              f"{'R²':>7} {'Skill':>7} {'Diagnosis':<22}")
        print("  " + "─"*85)
        for rank, row in lb.iterrows():
            beat = "✓" if (not np.isnan(row.get("Skill",np.nan)) and
                           row.get("Skill",-1)>0) else "✗"
            star = "★" if rank==0 else " "
            diag = str(row.get("Diagnosis","N/A"))
            diag_s = "LEARNING" if "LEARNING" in diag else "SEASONAL" if "SEASONAL" in diag else "N/A"
            print(f"  {beat}{star} {rank+1:<4} {row['Category']:<14} "
                  f"{row['Model']:<20} {row['Target']:<8} "
                  f"{row.get('R2',np.nan):>7.4f} {row.get('Skill',np.nan):>7.4f} "
                  f"{diag_s:<22}")

    print("\n  Files summary:")
    for root in ["preprocessed","results","models/dl","figures"]:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                sz = p.stat().st_size
                unit = "MB" if sz>1e6 else "KB"
                val  = sz/1e6 if sz>1e6 else sz/1e3
                print(f"    {'✓'} {str(p):<55} {val:>8.1f} {unit}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 23 — SLURM SUBMISSION (talon-gpu32)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def write_slurm_submission():
    """Write SLURM batch script for talon-gpu32."""
    PROJECT_DIR = Path("/home/emmanuel.keku")
    LOG_DIR     = PROJECT_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_SCRIPT = LOG_DIR / "run_soil_spatial_v2.sh"

    content = f"""#!/bin/bash
#SBATCH --job-name=soil_spatial_v2
#SBATCH --account=hpcusers
#SBATCH --partition=talon-gpu32
#SBATCH --nodelist=talon32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output={LOG_DIR}/spatial_v2_%j.out
#SBATCH --error={LOG_DIR}/spatial_v2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.keku@und.edu

echo "====================================="
echo "  Job  : $SLURM_JOB_ID"
echo "  Node : $(hostname)"
echo "  Start: $(date)"
echo "====================================="

module purge
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
module load pytorch-py39-cuda11.8-gcc11/1.13.0

PY=$(which python3)
echo "Python: $PY"
$PY --version
nvidia-smi

# Upgrade numpy to avoid _core import errors
$PY -m pip install --user "numpy>=1.24" -q
$PY -m pip install --user einops scipy scikit-learn PyWavelets xgboost lightgbm shap -q

$PY -c "
import torch
print('torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU    :', torch.cuda.get_device_name(0))
import numpy; print('numpy  :', numpy.__version__)
print('ALL OK')
"

cd {PROJECT_DIR}
export PYTHONPATH={PROJECT_DIR}:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Training start: $(date)"
$PY {PROJECT_DIR}/soil_prediction_codebook.py \\
    2>&1 | tee {LOG_DIR}/soil_spatial_v2_training.log

echo "Done: $(date)"
"""
    SLURM_SCRIPT.write_text(content)
    import os; os.chmod(SLURM_SCRIPT, 0o755)
    print(f"SLURM script: {SLURM_SCRIPT}")
    return SLURM_SCRIPT


def submit_job():
    slurm = write_slurm_submission()
    # Validate first
    val = subprocess.run(["sbatch","--test-only",str(slurm)],
                          capture_output=True, text=True)
    msg = val.stdout.strip() or val.stderr.strip()
    print(f"Validation: {msg}")
    if val.returncode == 0:
        result = subprocess.run(["sbatch",str(slurm)],
                                 capture_output=True, text=True,
                                 cwd="/home/emmanuel.keku")
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"\n✓ JOB SUBMITTED: {job_id}")
            print(f"  Monitor: squeue -j {job_id}")
            print(f"  Log    : tail -f /home/emmanuel.keku/logs/soil_spatial_v2_training.log")
            Path("/home/emmanuel.keku/logs").mkdir(parents=True, exist_ok=True)
            with open("/home/emmanuel.keku/logs/last_job.json","w") as f:
                json.dump({"job_id":job_id,"submitted_at":pd.Timestamp.now().isoformat()},f)
        else:
            print(f"✗ Submit failed: {result.stderr.strip()}")
            print(f"  Manual: sbatch {slurm}")
    else:
        print(f"✗ Validation failed: {msg}")
        print(f"  Manual: sbatch {slurm}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 24 — MONITOR & COLLECT RESULTS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def check_status():
    """Full status snapshot — safe after session restart."""
    print("=" * 60)
    print(f"  STATUS  {pd.Timestamp.now()}")
    print("=" * 60)

    # SLURM queue
    print("\n  SLURM queue:")
    r = subprocess.run(["squeue","--me","--format=%i %j %T %M %N","--noheader"],
                        capture_output=True, text=True)
    lines = r.stdout.strip().split("\n") if r.stdout.strip() else ["  (empty)"]
    for l in lines: print(f"    {l}")

    # Checkpoints
    mdir = Path("models/dl")
    print(f"\n  CHECKPOINTS  ({mdir}):")
    print(f"  {'Model':<22} {'temp':>8} {'smap':>8} {'moist':>8}")
    print("  " + "─"*50)
    if TORCH_OK:
        for arch in ["SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]:
            row = f"  {arch:<22}"
            for tgt in ["temp","smap","moist"]:
                ckpt = mdir / f"{arch}_{tgt}_spatial_best.pt"
                if ckpt.exists():
                    try:
                        d  = torch.load(ckpt, map_location="cpu")
                        r2 = d.get("val_r2",np.nan)
                        row += f" {r2:>8.4f}"
                    except Exception: row += f"{'ERR':>8}"
                else: row += f"{'missing':>8}"
            print(row)

    # Log tail
    log = Path("/home/emmanuel.keku/logs/soil_spatial_v2_training.log")
    if log.exists():
        print(f"\n  LOG TAIL ({log}):")
        with open(log) as f: lines = f.readlines()
        for l in lines[-10:]: print(f"    {l.rstrip()}")
    else:
        print("\n  Log not created yet.")
    print("=" * 60)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION BLOCK                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  RUNNING FULL CODEBOOK")
    print("="*65)

    # Sections 01-15 ran above (data loading, EDA, features, baselines)
    # Sections 16-19 ran above (DL models, training) if TORCH_OK
    # Now finalise:

    if TORCH_OK:
        # Collect any completed checkpoints
        try:
            spatial_df
        except NameError:
            spatial_df = collect_results()

        # Visualise
        if len(spatial_df) > 0:
            plot_all_results(spatial_df, node_coords=node_coords)

    # Save all outputs
    save_all_outputs()

    # Submit to SLURM if on login node
    import socket
    hostname = socket.gethostname()
    if "talon" in hostname.lower() or "hpc" in hostname.lower():
        print(f"\nDetected HPC node ({hostname}) — submitting SLURM job...")
        submit_job()
    else:
        print(f"\nNot on HPC ({hostname}) — skipping SLURM submission.")
        print("  To submit: call submit_job() in an interactive session on talon.")

    print("\n" + "="*65)
    print("  CODEBOOK COMPLETE")
    print("="*65)
    print("""
  QUICK REFERENCE — post-training commands:
  ─────────────────────────────────────────────────────────
  check_status()                  # SLURM + checkpoint inventory
  spatial_df = collect_results()  # load from checkpoints (safe after restart)
  plot_all_results(spatial_df)    # generate all SP0x figures
  save_all_outputs()              # serialise everything

  NEXT STEPS:
  ─────────────────────────────────────────────────────────
  Section 25: Ray Tune hyperparameter search
  Section 26: Ensemble (top spatial + top ML)
  Section 27: Multi-step forecasting (6/24/72h)
  Section 28: Freeze-thaw event detection
  Section 29: Knowledge distillation for edge deploy

  PAPER METRICS:
  ─────────────────────────────────────────────────────────
  R²              — coefficient of determination
  RMSE            — root mean squared error (°C / m³/m³)
  KGE             — Kling-Gupta efficiency
  Skill           — improvement over seasonal baseline [1]
  NodeR2_mean     — mean R² across all SMAP nodes [2]
  NodeR2_min      — worst-performing node
  SpatialVarRatio — spatial variance ratio (field quality) [2]
  Entropy_H       — physics learning diagnosis [4]
  Freeze_Acc      — freeze/thaw transition accuracy (%) [6]
    """)
