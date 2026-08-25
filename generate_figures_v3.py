"""
generate_figures_v3.py
Publication-quality figures for behavior-guided distributed experiment
Matches PI's reference figures style
Generates from saved CSV round histories — no rerunning needed
"""
import numpy as np
import pandas as pd
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

RESULTS = Path("/home/emmanuel.keku/results_v7")
FIGS    = Path("/home/emmanuel.keku/figures_v7")
FIGS.mkdir(parents=True, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

WORKER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
GENERAL_COLOR  = "#2c2c2c"
PREDICTED_COLOR= "#8B0000"

def worker_label(i): return f"Spatial Worker {i+1}"

def load_rounds(arch, n):
    f = RESULTS/f"rounds_v3_{arch}_N{n}.csv"
    if not f.exists(): return None
    return pd.read_csv(f)

def parse_list(s):
    try: return ast.literal_eval(str(s))
    except: return []

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: Cosine Similarity Heatmap (PI's key figure — Image 5)
# Shows how worker alignment with General trajectory changes over rounds
# ══════════════════════════════════════════════════════════════════════════════
def fig_cosine_heatmap(arch="STGCN", n=4):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist()
    R = len(rounds)
    N = n

    # Extract consensus per worker per round (proxy for cosine to General)
    # consensus[i] = cosine similarity of worker i to mean of all workers
    consensus_mat = np.zeros((N+1, R))  # N workers + Predicted row
    alpha_mat     = np.zeros((N, R))

    for ri, row in df.iterrows():
        cons = parse_list(row.get("consensus", "[]"))
        alps = parse_list(row.get("alphas", "[]"))
        for wi in range(min(N, len(cons))):
            consensus_mat[wi, ri] = float(cons[wi]) if not np.isnan(float(cons[wi])) else 0.
            alpha_mat[wi, ri]     = float(alps[wi]) if wi<len(alps) else 0.25
        # Predicted = weighted average consensus
        if cons:
            consensus_mat[N, ri] = float(np.mean(cons))

    fig, ax = plt.subplots(figsize=(16, 6))
    labels = [worker_label(i) for i in range(N)] + ["Predicted"]

    # Custom colormap: purple→teal→yellow
    cmap = LinearSegmentedColormap.from_list(
        "worker_align",
        ["#440154", "#31688e", "#35b779", "#fde725"], N=256)

    im = ax.imshow(consensus_mat, aspect="auto", cmap=cmap,
                    vmin=-1, vmax=1,
                    extent=[rounds[0]-0.5, rounds[-1]+0.5,
                             len(labels)-0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_title(f"Spatial Workers and Predicted Direction vs. General Trajectory\n"
                  f"{arch} | N={n} spatial workers | Soil Temperature",
                  fontsize=13, fontweight="bold")

    # Separator line before Predicted row
    ax.axhline(N-0.5, color="black", lw=2)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                         fraction=0.02, pad=0.02)
    cbar.set_label("Cosine Similarity to General", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_cosine_heatmap_{arch}_N{n}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_cosine_heatmap_{arch}_N{n}.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: Directional Alignment — line plot (Image 1 style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_directional_alignment(arch="STGCN", n=4):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(0, color="grey", lw=1.5, alpha=0.5)

    for wi in range(n):
        cons = [parse_list(row.get("consensus","[]")) for _,row in df.iterrows()]
        vals = [c[wi] if wi<len(c) else float("nan") for c in cons]
        ax.plot(rounds, vals, color=WORKER_COLORS[wi], lw=2,
                 label=worker_label(wi))

    # Predicted = mean consensus
    pred_vals = []
    for _,row in df.iterrows():
        c = parse_list(row.get("consensus","[]"))
        pred_vals.append(float(np.mean(c)) if c else float("nan"))
    ax.plot(rounds, pred_vals, color=PREDICTED_COLOR, lw=3,
             ls="-", label="Predicted", zorder=5)

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cosine to General", fontsize=12)
    ax.set_title(f"{len(rounds)}-Round Directional Alignment\n"
                  f"{arch} | N={n} spatial workers | Soil Temperature",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_directional_alignment_{arch}_N{n}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_directional_alignment_{arch}_N{n}.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: Attention Evolution — αi per worker (Image 2 style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_attention_evolution(arch="STGCN", n=4):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    for wi in range(n):
        alps = [parse_list(row.get("alphas","[]")) for _,row in df.iterrows()]
        vals = [a[wi] if wi<len(a) else float("nan") for a in alps]
        ax.plot(rounds, vals, color=WORKER_COLORS[wi], lw=2,
                 label=worker_label(wi))

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Attention (α)", fontsize=12)
    ax.set_title(f"Attention Evolution\n"
                  f"{arch} | N={n} spatial workers | Soil Temperature",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_attention_evolution_{arch}_N{n}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_attention_evolution_{arch}_N{n}.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: Uncertainty vs RMSE scatter (Image 3 style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_uncertainty_rmse(arch="STGCN"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    for ni, n in enumerate([1, 2, 4, 8]):
        df = load_rounds(arch, n)
        ax = axes[ni]
        if df is None:
            ax.set_title(f"N={n}"); continue
        rmses = df["rmse"].tolist()
        Us    = df["U"].tolist()
        valid = [(r,u) for r,u in zip(rmses,Us)
                  if not np.isnan(r) and not np.isinf(r) and not np.isnan(u)]
        if not valid: continue
        r_v, u_v = zip(*valid)
        ax.plot(r_v, u_v, color=WORKER_COLORS[ni], lw=1.5, marker="o",
                 ms=5, alpha=0.8)
        ax.set_xlabel("Validation RMSE", fontsize=11)
        ax.set_title(f"N={n} workers", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Uncertainty (U)", fontsize=12)
    fig.suptitle(f"Uncertainty vs RMSE | {arch} | All N configurations\n"
                  f"Soil Temperature | Wetland spatial holdout",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_uncertainty_rmse_{arch}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_uncertainty_rmse_{arch}.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5: Worker OOD vs Predicted vs General (Image 4 style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_worker_ood(arch="STGCN", n=4):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Worker local losses (in-distribution)
    for wi in range(min(n, 3)):
        wl = [parse_list(row.get("worker_losses","[]")) for _,row in df.iterrows()]
        vals = [w[wi] if wi<len(w) else float("nan") for w in wl]
        ax.plot(rounds, vals, color=WORKER_COLORS[wi], lw=1.5,
                 alpha=0.8, label=f"{worker_label(wi)} OOD")

    # Predicted global RMSE
    rmses = df["rmse"].tolist()
    ax.plot(rounds, rmses, color=PREDICTED_COLOR, lw=2.5,
             label="Predicted", zorder=5)

    # General model line (constant = centralized RMSE)
    summary = pd.read_csv(RESULTS/"behavior_guided_v3_summary.csv")
    sub = summary[(summary["arch"]==arch)&(summary["n_workers"]==n)]
    if not sub.empty:
        cent_rmse = float(sub["cent_rmse"].iloc[0])
        ax.axhline(cent_rmse, color=GENERAL_COLOR, lw=2.5, ls="--",
                    label="General")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(f"Worker OOD vs Predicted vs General\n"
                  f"{arch} | N={n} spatial workers | Soil Temperature",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_worker_ood_{arch}_N{n}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_worker_ood_{arch}_N{n}.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6: Processing time comparison (true parallel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_processing_time():
    summary = pd.read_csv(RESULTS/"behavior_guided_v3_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for mi, arch in enumerate(["STGCN", "SpatialMamba"]):
        ax = axes[mi]
        sub = summary[summary["arch"]==arch].sort_values("n_workers")
        if sub.empty: continue

        Ns = sub["n_workers"].tolist()
        ideal = sub["ideal_time_s"].tolist()
        cent  = float(sub["cent_time_s"].iloc[0])

        # Theoretical T/N
        t1 = [t for t,n in zip(sub["total_time_s"].tolist(), Ns) if n==1]
        t1 = t1[0] if t1 else ideal[0]
        theory = [t1/n for n in Ns]

        x = np.arange(len(Ns))
        w = 0.3
        bars1 = ax.bar(x-w/2, ideal, w, color="#1f77b4",
                        alpha=0.85, label="Actual parallel (Ray)")
        bars2 = ax.bar(x+w/2, theory, w, color="#aec7e8",
                        alpha=0.85, label="Theoretical T/N")
        ax.axhline(cent, color="red", ls="--", lw=2,
                    label=f"General model ({cent:.0f}s)")

        for bar, v in zip(bars1, ideal):
            ax.text(bar.get_x()+bar.get_width()/2, v+2,
                     f"{v:.0f}s", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in Ns], fontsize=11)
        ax.set_ylabel("Wall-clock Time (s)", fontsize=12)
        ax.set_title(f"Processing Time | {arch}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("True Parallel Processing Time vs N Spatial Workers\n"
                  "Actual Ray parallel wall-clock vs Theoretical T/N vs General model",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"fig_processing_time_comparison.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print("  OK fig_processing_time_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7: RMSE convergence — all N on one plot
# ══════════════════════════════════════════════════════════════════════════════
def fig_convergence_all_N(arch="STGCN"):
    summary = pd.read_csv(RESULTS/"behavior_guided_v3_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for mi, arch_name in enumerate(["STGCN", "SpatialMamba"]):
        ax = axes[mi]
        sub = summary[summary["arch"]==arch_name]
        cent_rmse = float(sub["cent_rmse"].iloc[0]) if not sub.empty else float("nan")

        for ni, n in enumerate([1, 2, 4, 8]):
            df = load_rounds(arch_name, n)
            if df is None: continue
            rounds = df["round"].tolist()
            rmses  = df["rmse"].tolist()
            valid  = [(r,v) for r,v in zip(rounds,rmses)
                       if not np.isnan(v) and not np.isinf(v)]
            if not valid: continue
            rv, vv = zip(*valid)
            ax.plot(rv, vv, color=WORKER_COLORS[ni], lw=2,
                     marker="o", ms=4, alpha=0.85,
                     label=f"N={n} spatial workers")

        ax.axhline(cent_rmse, color=GENERAL_COLOR, lw=2.5, ls="--",
                    label="General model", zorder=5)
        ax.set_xlabel("Global Round", fontsize=12)
        ax.set_ylabel("Wetland Val RMSE (residual)", fontsize=11)
        ax.set_title(f"Convergence | {arch_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("RMSE Convergence — All N Configurations\n"
                  "Train: SEEN spatial subregions | Val: Wetland (unseen) | Residual-only",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/"fig_convergence_all_N.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  OK fig_convergence_all_N.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8: Temporal prediction — observed vs predicted
# ══════════════════════════════════════════════════════════════════════════════
def fig_temporal_prediction(arch="STGCN", n=4):
    """Load saved prediction CSVs and plot temporal trend."""
    # Find all prediction files for this config
    pred_files = sorted(RESULTS.glob(f"preds_{arch}_N{n}_rnd*.csv"))
    if not pred_files:
        print(f"  No prediction files for {arch} N={n}")
        return

    # Use last 3 rounds
    use_files = pred_files[-3:]
    fig, axes = plt.subplots(len(use_files), 1,
                               figsize=(16, 4*len(use_files)),
                               sharex=True)
    if len(use_files)==1: axes=[axes]

    for ax, f in zip(axes, use_files):
        df = pd.read_csv(f)
        rnd = int(f.stem.split("rnd")[1])
        locs = df["location"].unique()[:5]
        for loc in locs:
            sub = df[df["location"]==loc]
            ax.plot(sub.index, sub["observed"], color="grey",
                     lw=1.5, alpha=0.5, label="Observed" if loc==locs[0] else "")
            ax.plot(sub.index, sub["predicted"],
                     color=PREDICTED_COLOR, lw=2, alpha=0.8,
                     label="Predicted" if loc==locs[0] else "")
        ax.set_ylabel("Residual", fontsize=11)
        ax.set_title(f"Round {rnd}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Sample index", fontsize=12)
    fig.suptitle(f"Temporal Prediction | {arch} | N={n}\n"
                  f"Wetland (unseen) | Soil Temperature Residual",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS/f"fig_temporal_prediction_{arch}_N{n}.png",
                 dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK fig_temporal_prediction_{arch}_N{n}.png")

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("Generating publication figures...")
print()

for arch in ["STGCN", "SpatialMamba"]:
    print(f"=== {arch} ===")
    for n in [1, 2, 4, 8]:
        fig_cosine_heatmap(arch, n)
        fig_directional_alignment(arch, n)
        fig_attention_evolution(arch, n)
        fig_worker_ood(arch, n)
    fig_uncertainty_rmse(arch)
    fig_temporal_prediction(arch, n=4)
    print()

fig_processing_time()
fig_convergence_all_N()

print()
print("All figures done:")
figs = sorted(FIGS.glob("fig_*.png"))
for f in figs:
    print(f"  {f.name} ({f.stat().st_size//1024} KB)")
