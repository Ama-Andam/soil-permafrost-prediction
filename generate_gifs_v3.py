"""
generate_gifs_v3.py
GIF generation for behavior-guided distributed experiment
Run after generate_figures_v3.py
"""
import numpy as np
import pandas as pd
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

RESULTS = Path("/home/emmanuel.keku/results_v7")
FIGS    = Path("/home/emmanuel.keku/figures_v7")
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
})

WORKER_COLORS  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                   "#9467bd","#8c564b","#e377c2","#7f7f7f"]
GENERAL_COLOR  = "#2c2c2c"
PREDICTED_COLOR= "#8B0000"

def load_rounds(arch, n):
    f = RESULTS/f"rounds_v3_{arch}_N{n}.csv"
    if not f.exists(): return None
    return pd.read_csv(f)

def parse_list(s):
    try: return ast.literal_eval(str(s))
    except: return []

def worker_label(i): return f"Spatial Worker {i+1}"

# ── GIF 1: Cosine heatmap animated ───────────────────────────────────────────
def make_gif_cosine_heatmap(arch, n):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist(); R=len(rounds); N=n
    consensus_mat = np.zeros((N+1, R))
    for ri,row in df.iterrows():
        cons = parse_list(row.get("consensus","[]"))
        for wi in range(min(N,len(cons))):
            v = float(cons[wi])
            consensus_mat[wi,ri] = v if not np.isnan(v) else 0.
        if cons: consensus_mat[N,ri] = float(np.mean(cons))
    labels = [worker_label(i) for i in range(N)] + ["Predicted"]
    cmap = LinearSegmentedColormap.from_list(
        "wa",["#440154","#31688e","#35b779","#fde725"],N=256)

    fig, ax = plt.subplots(figsize=(14,5))
    fig.colorbar_ax = None

    def update(frame):
        ax.clear()
        data = consensus_mat[:,:frame+1].copy()
        pad  = np.full((N+1, R-frame-1), np.nan)
        full = np.hstack([data, pad])
        im = ax.imshow(full, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                        extent=[0.5, R+0.5, N+0.5, -0.5])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axhline(N-0.5, color="black", lw=2)
        ax.set_xlabel("Training Round", fontsize=11)
        ax.set_title(
            f"Spatial Workers and Predicted Direction vs. General | {arch} N={n} | Round {frame+1}/{R}",
            fontsize=11, fontweight="bold")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=R, interval=300, blit=False)
    ani.save(FIGS/f"gif_cosine_heatmap_{arch}_N{n}.gif",
              writer=PillowWriter(fps=3), dpi=100)
    plt.close()
    print(f"  OK gif_cosine_heatmap_{arch}_N{n}.gif")

# ── GIF 2: Convergence all N ─────────────────────────────────────────────────
def make_gif_convergence(arch):
    summary = pd.read_csv(RESULTS/"behavior_guided_v3_summary.csv")
    sub = summary[summary["arch"]==arch]
    cent_rmse = float(sub["cent_rmse"].iloc[0]) if not sub.empty else float("nan")
    all_data = {}
    for n in [1,2,4,8]:
        df = load_rounds(arch, n)
        if df is not None: all_data[n] = df
    if not all_data: return
    max_r = max(len(d) for d in all_data.values())

    fig, ax = plt.subplots(figsize=(12,6))
    lines = {}
    for ni, n in enumerate([1,2,4,8]):
        if n not in all_data: continue
        line, = ax.plot([], [], color=WORKER_COLORS[ni], lw=2,
                         marker="o", ms=4, label=f"N={n} spatial workers")
        lines[n] = line
    ax.axhline(cent_rmse, color=GENERAL_COLOR, lw=2.5, ls="--", label="General model")
    all_vals = [r for n,df in all_data.items()
                 for r in df["rmse"].tolist()
                 if not np.isnan(r) and not np.isinf(r)]
    if all_vals: ax.set_ylim(min(all_vals)*0.9, max(all_vals)*1.1)
    ax.set_xlim(0, max_r+1)
    ax.set_xlabel("Global Round", fontsize=12)
    ax.set_ylabel("Wetland Val RMSE (residual)", fontsize=11)
    ax.legend(fontsize=9)

    def update(frame):
        ax.set_title(
            f"RMSE Convergence | {arch} | All N configurations | Round {frame+1}/{max_r}",
            fontsize=12, fontweight="bold")
        for n, line in lines.items():
            df = all_data[n]
            r = df["round"].tolist()[:frame+1]
            v = df["rmse"].tolist()[:frame+1]
            line.set_data(r, v)
        return list(lines.values())

    ani = animation.FuncAnimation(fig, update, frames=max_r, interval=300, blit=True)
    ani.save(FIGS/f"gif_convergence_{arch}.gif",
              writer=PillowWriter(fps=3), dpi=100)
    plt.close()
    print(f"  OK gif_convergence_{arch}.gif")

# ── GIF 3: Worker OOD vs Predicted vs General ─────────────────────────────────
def make_gif_worker_ood(arch, n):
    df = load_rounds(arch, n)
    if df is None: return
    summary = pd.read_csv(RESULTS/"behavior_guided_v3_summary.csv")
    sub = summary[(summary["arch"]==arch)&(summary["n_workers"]==n)]
    cent_rmse = float(sub["cent_rmse"].iloc[0]) if not sub.empty else float("nan")
    rounds = df["round"].tolist(); R = len(rounds)
    rmses  = df["rmse"].tolist()
    wl_all_raw = [parse_list(row.get("worker_losses","[]")) for _,row in df.iterrows()]
    n_show = min(n, 3)
    all_wl = [[wl_all_raw[ri][wi] if wi<len(wl_all_raw[ri]) else float("nan")
                for ri in range(R)] for wi in range(n_show)]

    fig, ax = plt.subplots(figsize=(12,6))
    wlines = [ax.plot([], [], color=WORKER_COLORS[wi], lw=1.5,
                       label=f"{worker_label(wi)} (local)")[0]
               for wi in range(n_show)]
    pred_line, = ax.plot([], [], color=PREDICTED_COLOR, lw=2.5,
                          label="Predicted", zorder=5)
    ax.axhline(cent_rmse, color=GENERAL_COLOR, lw=2, ls="--", label="General")
    all_v = [v for wl in all_wl for v in wl if not np.isnan(v)]
    all_v += [r for r in rmses if not np.isnan(r) and not np.isinf(r)]
    if all_v: ax.set_ylim(0, max(all_v)*1.15)
    ax.set_xlim(0, R+1)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(fontsize=9, ncol=2)

    def update(frame):
        xs = rounds[:frame+1]
        ax.set_title(
            f"Worker OOD vs Predicted vs General | {arch} N={n} | Round {frame+1}/{R}",
            fontsize=12, fontweight="bold")
        for wi, wl in enumerate(wlines):
            wl.set_data(xs, all_wl[wi][:frame+1])
        pred_line.set_data(xs, rmses[:frame+1])
        return wlines + [pred_line]

    ani = animation.FuncAnimation(fig, update, frames=R, interval=300, blit=True)
    ani.save(FIGS/f"gif_worker_ood_{arch}_N{n}.gif",
              writer=PillowWriter(fps=3), dpi=100)
    plt.close()
    print(f"  OK gif_worker_ood_{arch}_N{n}.gif")

# ── GIF 4: Attention evolution ────────────────────────────────────────────────
def make_gif_attention(arch, n):
    df = load_rounds(arch, n)
    if df is None: return
    rounds = df["round"].tolist(); R=len(rounds)
    alps_all = [parse_list(row.get("alphas","[]")) for _,row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(12,6))
    lines = [ax.plot([], [], color=WORKER_COLORS[wi], lw=2,
                      label=worker_label(wi))[0] for wi in range(n)]
    ax.set_ylim(0, 1.05); ax.set_xlim(0, R+1)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Attention (α)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)

    def update(frame):
        xs = rounds[:frame+1]
        ax.set_title(
            f"Attention Evolution | {arch} N={n} | Round {frame+1}/{R}",
            fontsize=12, fontweight="bold")
        for wi, line in enumerate(lines):
            vals = [alps_all[ri][wi] if wi<len(alps_all[ri]) else float("nan")
                     for ri in range(frame+1)]
            line.set_data(xs, vals)
        return lines

    ani = animation.FuncAnimation(fig, update, frames=R, interval=300, blit=True)
    ani.save(FIGS/f"gif_attention_{arch}_N{n}.gif",
              writer=PillowWriter(fps=3), dpi=100)
    plt.close()
    print(f"  OK gif_attention_{arch}_N{n}.gif")

# ── RUN ALL ───────────────────────────────────────────────────────────────────
print("Generating GIFs...")
print()
for arch in ["STGCN", "SpatialMamba"]:
    print(f"=== {arch} ===")
    make_gif_convergence(arch)
    for n in [2, 4, 8]:
        make_gif_cosine_heatmap(arch, n)
        make_gif_worker_ood(arch, n)
        make_gif_attention(arch, n)
    print()

print("All GIFs done:")
gifs = sorted(FIGS.glob("gif_*.gif"))
for f in gifs:
    print(f"  {f.name} ({f.stat().st_size//1024} KB)")
