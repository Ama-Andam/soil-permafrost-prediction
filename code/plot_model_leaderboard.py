"""
Generates publication-quality PNG figures from train.py's model_leaderboard.csv.

Usage:
    python plot_model_leaderboard.py results/training_fast3/model_leaderboard.csv --outdir results/training_fast3/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 100,
    "savefig.bbox": "tight",
})

BAR_COLOR = "#0c447c"
FAIL_COLOR = "#993c1d"


MODEL_COLORS = {
    "LSTM": "#2a78d6", "BiLSTM": "#1baf7a", "GRU": "#eda100", "BiGRU": "#008300",
    "TCN": "#4a3aa7", "PatchTST": "#e34948", "Informer": "#e87ba4",
    "TFT": "#eb6834", "S4D": "#199e70", "Mamba": "#0c447c",
}
FAIL_COLOR = "#888780"


def plot_metric(ax, df: pd.DataFrame, target: str, metric: str, ylabel: str,
                 panel_label: str, higher_is_better: bool, title: str):
    sub = df[df["target"] == target].copy()
    sub = sub.sort_values(metric, ascending=not higher_is_better)

    colors = [MODEL_COLORS.get(m, "#5f5e5a") if s == "ok" else FAIL_COLOR
              for m, s in zip(sub["model"], sub["status"])]
    ax.bar(sub["model"], sub[metric], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="500", pad=8)
    ax.tick_params(axis="x", rotation=40, labelsize=8.5)
    ax.grid(axis="y", linewidth=0.4, color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.1, 1.12, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to model_leaderboard.csv")
    parser.add_argument("--outdir", default="./leaderboard_plots")
    parser.add_argument("--fname", default="fig_model_leaderboard.png")
    parser.add_argument("--dpi", type=int, default=150,
                         help="Output resolution. Use 150 (default) for screen viewing, "
                              "300+ for print/journal submission.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[9, 6.5],
                         help="Figure width height in inches (default: 9 6.5)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    targets = sorted(df["target"].unique(), key=lambda t: "temp" not in t)  # temperature first

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    panel_labels = ["a", "b", "c", "d"]
    label_i = 0

    for row, target in enumerate(targets):
        short_name = "Soil temperature" if "temp" in target else "Soil moisture"
        unit = "(°C)" if "temp" in target else "(m$^3$ m$^{-3}$)"

        plot_metric(axes[row, 0], df, target, "r2", f"{short_name}  $R^2$",
                    panel_labels[label_i], higher_is_better=True,
                    title=f"{short_name} -- model fit ($R^2$)")
        label_i += 1
        plot_metric(axes[row, 1], df, target, "rmse", f"{short_name}  RMSE {unit}",
                    panel_labels[label_i], higher_is_better=False,
                    title=f"{short_name} -- prediction error (RMSE)")
        label_i += 1

    fig.suptitle("Model comparison on reconstructed raw-scale predictions",
                  fontsize=13, fontweight="500", y=1.02)
    fig.tight_layout()
    out_path = outdir / args.fname
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()