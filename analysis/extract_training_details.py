"""
extract_training_details.py
Extracts and saves all training details requested by senior:
1. Loss over epoch (train + val) per model — CSV + figures
2. Input variables table — CSV
3. Hyperparameter configuration — CSV (honest: fixed, not tuned)
4. Full training log summary — CSV

RUN ON TALON:
  module load pytorch-py39-cuda11.8-gcc11/1.13.0
  python3 ~/extract_training_details.py
"""

import pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT  = Path("/home/emmanuel.keku")
MODELS   = PROJECT / "models_v4" / "dl"
RESULTS  = PROJECT / "results_v4"
FIGS     = PROJECT / "figures_v4"
PREPROC  = PROJECT / "preprocessed_v3"
for d in [RESULTS, FIGS]: d.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi":150, "font.size":11,
                              "axes.grid":True, "grid.alpha":0.3})

try:
    import torch
    DEVICE = torch.device("cpu")
except ImportError:
    print("Load modules first"); exit(1)

print("="*65)
print("  EXTRACTING TRAINING DETAILS FOR SENIOR")
print("="*65)

ARCHES = ["BiGRU_NoGCN","GCN_NoTemporal","DeepESN","SpatialESN",
          "GraphSAGE","GAT","STGCN",
          "SpatialBiGRU","SpatialMamba","SpatialS4","SpatialFuseMoE"]
TARGETS = ["temp","smap","moist"]
TIERS   = {"BiGRU_NoGCN":"ABLATION","GCN_NoTemporal":"ABLATION",
           "DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR",
           "GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH",
           "SpatialBiGRU":"SSM","SpatialMamba":"SSM",
           "SpatialS4":"SSM","SpatialFuseMoE":"SSM"}
TIER_COLORS = {"ABLATION":"#d62728","RESERVOIR":"#9467bd",
               "GRAPH":"#2ca02c","SSM":"#1f77b4"}
TGT_LABELS  = {"temp":"Weather Temp (°C)",
               "smap":"SMAP Temp L1 (K)",
               "moist":"Soil Moisture (m³/m³)"}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOSS OVER EPOCH — CSV + FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Extracting loss curves per epoch...")

all_history = []
for arch in ARCHES:
    for tgt in TARGETS:
        ckpt = MODELS / f"{arch}_{tgt}_v4_best.pt"
        if not ckpt.exists(): continue
        try:
            sv   = torch.load(ckpt, map_location="cpu")
            hist = sv.get("history", [])
            if not hist: continue
            for h in hist:
                all_history.append(dict(
                    Model=arch, Target=tgt,
                    Tier=TIERS.get(arch,"?"),
                    Epoch=h["epoch"],
                    Train_Loss=round(h["train_loss"],6),
                    Val_R2_Seen=round(h.get("val_R2_seen",0),4),
                    Val_R2_Unseen=round(h.get("val_R2_unseen",0),4),
                    H_norm=round(h.get("H_norm",0),4),
                    Epochs_Run=sv.get("epochs_run","?"),
                    Val_R2_Best=round(sv.get("val_r2",0),4),
                    Holdout_Site=sv.get("holdout_site","Wetland"),
                    Job_ID=sv.get("job_id","?"),
                    Node=sv.get("node","?"),
                    Train_Time_s=round(sv.get("elapsed_s",0),1)))
        except Exception as e:
            print(f"  ✗ {arch}[{tgt}]: {e}")

hist_df = pd.DataFrame(all_history)
if len(hist_df) > 0:
    hist_df.to_csv(RESULTS/"training_loss_per_epoch.csv", index=False)
    print(f"  ✓ training_loss_per_epoch.csv — {len(hist_df):,} epoch records")
    print(f"    Models: {hist_df['Model'].nunique()} | "
          f"Targets: {hist_df['Target'].nunique()} | "
          f"Total epochs: {len(hist_df)}")

    # ── Figure: Loss curves per target group ─────────────────────────────────
    for tgt in TARGETS:
        sub = hist_df[hist_df["Target"]==tgt]
        if sub.empty: continue

        fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=False)

        # Panel 1: Training loss
        for arch in ARCHES:
            m = sub[sub["Model"]==arch]
            if m.empty: continue
            col = TIER_COLORS.get(TIERS.get(arch,"?"), "grey")
            ls  = "--" if TIERS.get(arch) == "ABLATION" else "-"
            axes[0].plot(m["Epoch"], m["Train_Loss"],
                         lw=1.8, alpha=0.85, color=col,
                         ls=ls, label=f"[{TIERS.get(arch,'?')}] {arch}")
        axes[0].set_ylabel("Huber Training Loss", fontsize=11)
        axes[0].set_title(f"Training Loss per Epoch | {TGT_LABELS.get(tgt,tgt)}",
                          fontweight="bold", fontsize=12)
        axes[0].legend(fontsize=7, ncol=3, loc="upper right")

        # Panel 2: Val R² seen
        for arch in ARCHES:
            m = sub[sub["Model"]==arch]
            if m.empty: continue
            col = TIER_COLORS.get(TIERS.get(arch,"?"), "grey")
            ls  = "--" if TIERS.get(arch) == "ABLATION" else "-"
            axes[1].plot(m["Epoch"], m["Val_R2_Seen"],
                         lw=1.8, alpha=0.85, color=col,
                         ls=ls, label=arch)
        axes[1].set_ylabel("Validation R² (Seen — 192 locs)", fontsize=11)
        axes[1].set_title("Validation R² — Seen Locations",
                          fontweight="bold", fontsize=12)
        axes[1].set_ylim(0.85, 1.01)

        # Panel 3: Val R² unseen (KEY — Wetland holdout)
        for arch in ARCHES:
            m = sub[sub["Model"]==arch]
            if m.empty: continue
            col = TIER_COLORS.get(TIERS.get(arch,"?"), "grey")
            ls  = "--" if TIERS.get(arch) == "ABLATION" else "-"
            axes[2].plot(m["Epoch"], m["Val_R2_Unseen"],
                         lw=1.8, alpha=0.85, color=col,
                         ls=ls, label=arch)
        axes[2].set_ylabel("Validation R² (Unseen — Wetland 64 locs)", fontsize=11)
        axes[2].set_title("Validation R² — Unseen Wetland Locations (Spatial Generalisation)",
                          fontweight="bold", fontsize=12,
                          color="#d62728")
        axes[2].set_ylim(0.70, 1.01)
        axes[2].axhline(0.90, color="grey", ls=":", lw=1.5,
                        alpha=0.7, label="R²=0.90 threshold")
        axes[2].legend(fontsize=7, ncol=3, loc="lower right")

        for ax in axes:
            ax.set_xlabel("Epoch", fontsize=10)

        from matplotlib.patches import Patch
        fig.legend(
            handles=[Patch(color=c,label=t) for t,c in TIER_COLORS.items()],
            loc="lower center", ncol=4, fontsize=10,
            bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Training and Validation Curves | {TGT_LABELS.get(tgt,tgt)}\n"
            f"v4 Distributed Spatial AI | Wetland Spatial Holdout | "
            f"11 Models × 4 Tiers",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(FIGS/f"DETAIL_01_loss_curves_{tgt}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ DETAIL_01_loss_curves_{tgt}.png")

    # ── Figure: Entropy (H_norm) evolution ───────────────────────────────────
    for tgt in TARGETS:
        sub = hist_df[hist_df["Target"]==tgt]
        if sub.empty or sub["H_norm"].max() == 0: continue
        fig, ax = plt.subplots(figsize=(18, 8))
        for arch in ARCHES:
            m = sub[sub["Model"]==arch]
            if m.empty: continue
            col = TIER_COLORS.get(TIERS.get(arch,"?"), "grey")
            ax.plot(m["Epoch"], m["H_norm"], lw=1.8, alpha=0.85,
                    color=col, label=f"[{TIERS.get(arch,'?')}] {arch}")
        ax.axhline(0.5, color="orange", ls="--", lw=1.5, alpha=0.7,
                   label="H=0.5 (moderate diversity)")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Normalised Entropy H_norm", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Entropy Evolution | {TGT_LABELS.get(tgt,tgt)}\n"
            f"Rising/stable H → LEARNING_DYNAMICS | "
            f"Flat H → SEASONAL_FITTING",
            fontweight="bold", fontsize=12)
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGS/f"DETAIL_02_entropy_{tgt}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ DETAIL_02_entropy_{tgt}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. INPUT VARIABLES TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Building input variables table...")

input_vars = [
    # Spatio-temporal reference
    ("Latitude",              "Spatio-Temporal Ref", "°N",        "Location latitude"),
    ("Longitude",             "Spatio-Temporal Ref", "°E",        "Location longitude"),
    ("smap_node_x",           "Spatio-Temporal Ref", "index",     "SMAP grid node X index"),
    ("smap_node_y",           "Spatio-Temporal Ref", "index",     "SMAP grid node Y index"),
    # Topography
    ("elevation_m",           "Topography",          "m",         "Elevation above sea level"),
    ("elev_roughness_m",      "Topography",          "m",         "Elevation roughness (local std)"),
    ("slope_deg",             "Topography",          "degrees",   "Terrain slope angle"),
    # Weather station
    ("temperature_2m",        "Weather",             "°C",        "Air temperature at 2m"),
    ("precipitation",         "Weather",             "mm/hr",     "Precipitation rate"),
    ("snow_depth_weather",    "Weather",             "m",         "Snow depth from weather station"),
    # SMAP satellite
    ("Temp_K",                "SMAP Satellite",      "K",         "SMAP soil temperature (Kelvin)"),
    ("Pressure",              "SMAP Satellite",      "Pa",        "Surface pressure"),
    ("Greenness",             "SMAP Satellite",      "index",     "Vegetation greenness index"),
    ("Snow_Depth_SMAP",       "SMAP Satellite",      "m",         "Snow depth from SMAP"),
    # Cyclical time encodings
    ("sin_doy",               "Cyclical Encoding",   "-",         "Sine of day-of-year (annual cycle)"),
    ("cos_doy",               "Cyclical Encoding",   "-",         "Cosine of day-of-year"),
    ("sin_hour",              "Cyclical Encoding",   "-",         "Sine of hour (diurnal cycle)"),
    ("cos_hour",              "Cyclical Encoding",   "-",         "Cosine of hour"),
    ("sin_month",             "Cyclical Encoding",   "-",         "Sine of month"),
    ("cos_month",             "Cyclical Encoding",   "-",         "Cosine of month"),
    # Physical indicators
    ("is_frozen",             "Physical Indicator",  "binary",    "1 if soil temp < 0°C else 0"),
    ("Temp_C",                "Physical Indicator",  "°C",        "SMAP temp converted to Celsius"),
    # Wavelet seasonal approx (v4 specific — added as input feature)
    ("soil_temperature_approx","Wavelet Approx (v4)","°C",        "Seasonal component of soil temperature"),
    ("Soil_Temp_L1_approx",   "Wavelet Approx (v4)", "K",         "Seasonal component of SMAP L1 temp"),
    ("soil_moisture_approx",  "Wavelet Approx (v4)", "m³/m³",     "Seasonal component of soil moisture"),
    ("SM_Surface_approx",     "Wavelet Approx (v4)", "m³/m³",     "Seasonal component of SMAP surface SM"),
]

iv_df = pd.DataFrame(input_vars,
                     columns=["Variable","Group","Unit","Description"])
iv_df.index = range(1, len(iv_df)+1)
iv_df.index.name = "No."
iv_df.to_csv(RESULTS/"input_variables.csv")
print(f"  ✓ input_variables.csv — {len(iv_df)} variables")
print(f"    Groups: {iv_df['Group'].value_counts().to_dict()}")

# ── Input variables figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, max(8, len(iv_df)*0.45+2)))
ax.axis("off")
tbl = ax.table(
    cellText=[[str(i), r["Variable"], r["Group"], r["Unit"], r["Description"]]
              for i, r in iv_df.reset_index().iterrows()],
    colLabels=["No.", "Variable", "Group", "Unit", "Description"],
    cellLoc="left", loc="center", bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
group_colors = {
    "Spatio-Temporal Ref": "#d4e6f1",
    "Topography":          "#d5f5e3",
    "Weather":             "#fdebd0",
    "SMAP Satellite":      "#e8daef",
    "Cyclical Encoding":   "#fdfefe",
    "Physical Indicator":  "#fadbd8",
    "Wavelet Approx (v4)": "#d0ece7",
}
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r <= len(iv_df):
        grp = iv_df.iloc[r-1]["Group"]
        cell.set_facecolor(group_colors.get(grp, "white"))
    cell.set_edgecolor("white")
ax.set_title(
    "Input Variables — v4 Distributed Spatial AI Framework\n"
    "26 features per location per timestep | "
    "Input shape: (batch, 24 timesteps, 256 locations, 26 features)",
    fontweight="bold", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(FIGS/"DETAIL_03_input_variables_table.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ DETAIL_03_input_variables_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER CONFIGURATION
# (honest: fixed hyperparameters, no systematic tuning)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building hyperparameter configuration table...")

hp_rows = []

# Shared training hyperparameters
shared_hp = [
    ("ALL", "ALL", "Optimizer",        "AdamW",   "Standard adaptive optimizer with decoupled weight decay"),
    ("ALL", "ALL", "Learning Rate",    "3e-4",    "OneCycleLR with max_lr=3e-4, cosine annealing"),
    ("ALL", "ALL", "LR Schedule",      "OneCycleLR", "pct_start=0.1, anneal_strategy=cosine"),
    ("ALL", "ALL", "Weight Decay",     "1e-4",    "L2 regularisation"),
    ("ALL", "ALL", "Max Epochs",       "30",      "With early stopping (patience=7 on val_R2_seen)"),
    ("ALL", "ALL", "Early Stopping",   "patience=7","Triggered on validation R² (seen locations)"),
    ("ALL", "ALL", "Batch Size",       "4",       "GPU memory constrained (V100 32GB)"),
    ("ALL", "ALL", "Lookback Window",  "24 hours","24 hourly timesteps of history"),
    ("ALL", "ALL", "Dropout",          "0.1",     "Applied to all trainable layers"),
    ("ALL", "ALL", "Grad Clip",        "1.0",     "Global gradient norm clipping"),
    ("ALL", "ALL", "Loss Function",    "Huber δ=1.0","Robust to outliers in soil moisture"),
    ("ALL", "ALL", "λ_spatial",        "0.05",    "Graph Laplacian smoothness regulariser weight"),
    ("ALL", "ALL", "λ_aux (MoE only)", "0.01",    "Load-balancing auxiliary loss for FuseMoE"),
    ("ALL", "ALL", "Training Strategy","Masked Huber","Loss computed on seen locations only (192/256)"),
    ("ALL", "ALL", "Spatial Holdout",  "Wetland", "64 locations withheld from training entirely"),
    ("ALL", "ALL", "GCN k-neighbours", "6",       "k-NN spatial graph, Gaussian decay weights"),
    ("ALL", "ALL", "GCN sigma",        "9.99 km", "Gaussian decay bandwidth (median pairwise dist)"),
    ("ALL", "ALL", "AMP",              "float16", "Automatic Mixed Precision on V100 GPU"),
    ("ALL", "ALL", "num_workers",      "0",       "DataLoader workers (CUDA fork constraint on talon32)"),
    ("ALL", "ALL", "Seed",             "42",      "Global random seed for reproducibility"),
]

# Model-specific hyperparameters
model_hp = [
    # BiGRU_NoGCN
    ("BiGRU_NoGCN", "ABLATION", "Hidden Dim",    "96",     "BiGRU hidden size per direction"),
    ("BiGRU_NoGCN", "ABLATION", "GRU Layers",    "2",      "Bidirectional GRU depth"),
    ("BiGRU_NoGCN", "ABLATION", "Attention Heads","4",     "Multi-head attention"),
    ("BiGRU_NoGCN", "ABLATION", "GCN Layers",    "0",      "NO graph convolution — ablation"),
    # GCN_NoTemporal
    ("GCN_NoTemporal","ABLATION","Hidden Dim",   "96",     "GCN hidden size"),
    ("GCN_NoTemporal","ABLATION","GCN Layers",   "3",      "Graph convolution layers"),
    ("GCN_NoTemporal","ABLATION","Temporal",     "None",   "NO temporal encoder — ablation"),
    # DeepESN
    ("DeepESN",    "RESERVOIR", "Reservoir Dim", "128",    "ESN reservoir size per layer"),
    ("DeepESN",    "RESERVOIR", "ESN Layers",    "3",      "Stacked reservoir layers"),
    ("DeepESN",    "RESERVOIR", "Spectral Radius","0.9",   "Controls echo state memory length"),
    ("DeepESN",    "RESERVOIR", "Leaking Rate",  "0.3/layer","Decreasing per layer (multi-scale)"),
    ("DeepESN",    "RESERVOIR", "Reservoir Weights","FIXED","Never trained — echo state property"),
    ("DeepESN",    "RESERVOIR", "Trainable Params","Readout only","Linear head only"),
    # SpatialESN
    ("SpatialESN", "RESERVOIR", "Reservoir Dim", "128",    "ESN reservoir size"),
    ("SpatialESN", "RESERVOIR", "ESN Layers",    "3",      "Stacked reservoirs"),
    ("SpatialESN", "RESERVOIR", "GCN Layers",    "2",      "Graph convolution after ESN"),
    ("SpatialESN", "RESERVOIR", "Spectral Radius","0.9",   "Echo state memory"),
    # GraphSAGE
    ("GraphSAGE",  "GRAPH",     "Hidden Dim",    "96",     "SAGE hidden size"),
    ("GraphSAGE",  "GRAPH",     "GRU Layers",    "2",      "Temporal BiGRU encoder"),
    ("GraphSAGE",  "GRAPH",     "SAGE Layers",   "3",      "Inductive aggregation layers"),
    ("GraphSAGE",  "GRAPH",     "Aggregation",   "Mean",   "Neighbourhood mean aggregation"),
    # GAT
    ("GAT",        "GRAPH",     "Hidden Dim",    "96",     "GAT hidden size"),
    ("GAT",        "GRAPH",     "Attention Heads","4",     "Multi-head graph attention"),
    ("GAT",        "GRAPH",     "GAT Layers",    "2",      "Graph attention layers"),
    # STGCN
    ("STGCN",      "GRAPH",     "Hidden Dim",    "64",     "STGCN channel size"),
    ("STGCN",      "GRAPH",     "GCN Layers",    "2",      "Spatial graph convolutions"),
    # SpatialBiGRU
    ("SpatialBiGRU","SSM",      "Hidden Dim",    "96",     "BiGRU hidden size per direction"),
    ("SpatialBiGRU","SSM",      "GRU Layers",    "2",      "Bidirectional GRU depth"),
    ("SpatialBiGRU","SSM",      "Attention Heads","4",     "Multi-head temporal attention"),
    ("SpatialBiGRU","SSM",      "GCN Layers",    "2",      "Spatial graph convolution layers"),
    # SpatialMamba
    ("SpatialMamba","SSM",      "Model Dim",     "96",     "Mamba SSM hidden dimension"),
    ("SpatialMamba","SSM",      "Mamba Layers",  "4",      "Selective SSM blocks"),
    ("SpatialMamba","SSM",      "State Dim",     "16",     "SSM state space dimension"),
    ("SpatialMamba","SSM",      "GCN Layers",    "2",      "Spatial graph convolution layers"),
    # SpatialS4
    ("SpatialS4",  "SSM",       "Model Dim",     "96",     "S4 hidden dimension"),
    ("SpatialS4",  "SSM",       "S4 Layers",     "4",      "Bidirectional S4 blocks"),
    ("SpatialS4",  "SSM",       "State Dim",     "64",     "HiPPO-LegS state dimension"),
    ("SpatialS4",  "SSM",       "GCN Layers",    "2",      "Spatial graph convolution layers"),
    # SpatialFuseMoE
    ("SpatialFuseMoE","SSM",    "Model Dim",     "96",     "MoE hidden dimension"),
    ("SpatialFuseMoE","SSM",    "N Experts",     "4",      "Mamba/GRU/CNN/GRU experts"),
    ("SpatialFuseMoE","SSM",    "Top-K Gate",    "2",      "Sparse top-2 expert selection"),
    ("SpatialFuseMoE","SSM",    "SSM Backbone",  "2 layers","Mamba backbone after fusion"),
    ("SpatialFuseMoE","SSM",    "GCN Layers",    "2",      "Spatial graph convolution layers"),
]

for m,t,p,v,d in shared_hp:
    hp_rows.append({"Model":m,"Tier":t,"Parameter":p,"Value":v,"Description":d})
for m,t,p,v,d in model_hp:
    hp_rows.append({"Model":m,"Tier":t,"Parameter":p,"Value":v,"Description":d})

hp_df = pd.DataFrame(hp_rows)
hp_df.to_csv(RESULTS/"hyperparameter_config.csv", index=False)
print(f"  ✓ hyperparameter_config.csv — {len(hp_df)} entries")
print(f"  NOTE: All hyperparameters are FIXED (literature defaults)")
print(f"        No systematic tuning was performed")
print(f"        Systematic tuning via Ray Tune is planned as future work")

# ── Hyperparameter figure ─────────────────────────────────────────────────────
shared_only = hp_df[hp_df["Model"]=="ALL"]
fig, ax = plt.subplots(figsize=(18, max(8, len(shared_only)*0.55+2)))
ax.axis("off")
tbl = ax.table(
    cellText=[[r["Parameter"], r["Value"], r["Description"]]
              for _, r in shared_only.iterrows()],
    colLabels=["Parameter", "Value", "Description"],
    cellLoc="left", loc="center", bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f2f3f4")
    cell.set_edgecolor("white")
ax.set_title(
    "Shared Training Hyperparameters — All 11 Models\n"
    "Fixed values (literature standards) — No systematic tuning performed\n"
    "Hyperparameter optimisation via Ray Tune is planned as future work",
    fontweight="bold", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(FIGS/"DETAIL_04_hyperparameter_config.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ DETAIL_04_hyperparameter_config.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING SUMMARY LOG
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Building training summary log...")

summary_rows = []
for arch in ARCHES:
    for tgt in TARGETS:
        ckpt = MODELS / f"{arch}_{tgt}_v4_best.pt"
        if not ckpt.exists(): continue
        try:
            sv   = torch.load(ckpt, map_location="cpu")
            hist = sv.get("history", [])
            tm   = sv.get("test_metrics", {})
            summary_rows.append(dict(
                Model=arch, Tier=TIERS.get(arch,"?"), Target=tgt,
                N_Epochs=sv.get("epochs_run","?"),
                Train_Time_s=round(sv.get("elapsed_s",0),1),
                Train_Time_min=round(sv.get("elapsed_s",0)/60,1),
                Best_Val_R2=round(sv.get("val_r2",0),4),
                Final_Train_Loss=round(hist[-1]["train_loss"],6) if hist else None,
                Initial_Train_Loss=round(hist[0]["train_loss"],6) if hist else None,
                Loss_Reduction_Pct=round(
                    (1-hist[-1]["train_loss"]/hist[0]["train_loss"])*100,1)
                    if hist and hist[0]["train_loss"]>0 else None,
                Final_Val_R2_Seen=round(hist[-1].get("val_R2_seen",0),4) if hist else None,
                Final_Val_R2_Unseen=round(hist[-1].get("val_R2_unseen",0),4) if hist else None,
                Test_Seen_R2=tm.get("seen_R2","?"),
                Test_Unseen_R2=tm.get("unseen_R2","?"),
                Spatial_Gap=tm.get("spatial_gap","?"),
                Holdout_Site=sv.get("holdout_site","Wetland"),
                N_Features=sv.get("n_v4_features",26),
                Job_ID=sv.get("job_id","?"),
                Node=sv.get("node","?")))
        except Exception as e:
            print(f"  ✗ {arch}[{tgt}]: {e}")

summary_df = pd.DataFrame(summary_rows)
if len(summary_df) > 0:
    summary_df.to_csv(RESULTS/"training_summary_log.csv", index=False)
    print(f"  ✓ training_summary_log.csv — {len(summary_df)} model×target combinations")

    # Print compact summary
    print(f"\n  {'Tier':<12} {'Model':<20} {'Tgt':<6} "
          f"{'Epochs':>7} {'Time(min)':>10} {'BestValR2':>10} "
          f"{'LossRed%':>9} {'TestUnseen':>11}")
    print("  " + "─"*85)
    for _,r in summary_df.sort_values(["Tier","Model"]).iterrows():
        print(f"  {r['Tier']:<12} {r['Model']:<20} {r['Target']:<6} "
              f"{str(r['N_Epochs']):>7} {str(r['Train_Time_min']):>10} "
              f"{str(r['Best_Val_R2']):>10} "
              f"{str(r.get('Loss_Reduction_Pct','?')):>9}% "
              f"{str(r.get('Test_Unseen_R2','?')):>11}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  ALL TRAINING DETAILS EXTRACTED")
print("="*65)
print(f"\n  CSVs saved to: {RESULTS}")
for f in sorted(RESULTS.glob("*.csv")):
    print(f"    {f.name:<45} {f.stat().st_size/1e3:.1f} KB")

print(f"\n  Figures saved to: {FIGS}")
for f in sorted(FIGS.glob("DETAIL_*.png")):
    print(f"    {f.name}")

print(f"""
  SHARE WITH SENIOR:
    training_loss_per_epoch.csv    — loss + R²_seen + R²_unseen per epoch
    input_variables.csv            — all 26 input features with descriptions
    hyperparameter_config.csv      — all fixed hyperparameters per model
    training_summary_log.csv       — epochs, time, loss reduction per model

  FIGURES:
    DETAIL_01_loss_curves_*.png    — train loss + val R² per epoch
    DETAIL_02_entropy_*.png        — entropy evolution (physics diagnostic)
    DETAIL_03_input_variables_table.png
    DETAIL_04_hyperparameter_config.png

  NOTE ON HYPERPARAMETER TUNING:
    All hyperparameters are FIXED literature defaults.
    No systematic tuning was performed.
    Tuning via Ray Tune is planned as future work.
    Be transparent about this with the senior.
""")
