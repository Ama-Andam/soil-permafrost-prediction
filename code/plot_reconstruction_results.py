"""
Generates publication-quality PNG figures from the full 256-location
reconstruction_options.py output (reconstruction_rmse_by_location.csv).

Usage:
    python plot_reconstruction_results.py results/reconstruction/reconstruction_rmse_by_location.csv --outdir results/reconstruction/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SITES = ["Bedrock", "Transition", "Upland", "Wetland"]
HARMONIC_COLOR = "#0c447c"
SEASONAL_COLOR = "#993c1d"


def plot_terrain_bars(df: pd.DataFrame, outdir: Path):
    for target, ylabel, fname, panel_label in [
        ("soil_temperature_0_to_7cm", "RMSE (°C)", "fig_temperature_by_terrain.png", "a"),
        ("soil_moisture_0_to_7cm", "RMSE (m$^3$ m$^{-3}$)", "fig_moisture_by_terrain.png", "b"),
    ]:
        sub_all = df[df["target"] == target]
        means = sub_all.groupby("Site")[["seasonal_naive", "harmonic_regression"]].mean().reindex(SITES)
        stds = sub_all.groupby("Site")[["seasonal_naive", "harmonic_regression"]].std().reindex(SITES)
        n = sub_all.groupby("Site").size().reindex(SITES)
        sem = stds.div(np.sqrt(n), axis=0)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        x = np.arange(len(SITES))
        width = 0.32

        ax.bar(x - width / 2, means["seasonal_naive"], width,
               yerr=sem["seasonal_naive"], capsize=3, color=SEASONAL_COLOR,
               label="Seasonal-naive", edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, means["harmonic_regression"], width,
               yerr=sem["harmonic_regression"], capsize=3, color=HARMONIC_COLOR,
               label="Harmonic regression", edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(SITES)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Terrain class")
        ax.legend(frameon=False, loc="upper right", fontsize=9.5)
        ax.grid(axis="y", linewidth=0.4, color="#e1e0d9", zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.12, 1.04, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold")

        fig.tight_layout()
        fig.savefig(outdir / fname)
        plt.close(fig)
        print(f"Saved {outdir / fname}")


def plot_location_maps(df: pd.DataFrame, outdir: Path):
    for target, fname, cbar_label, panel_label in [
        ("soil_temperature_0_to_7cm", "fig_temperature_method_advantage.png",
         "$\\Delta$RMSE = harmonic $-$ seasonal-naive (°C)", "a"),
        ("soil_moisture_0_to_7cm", "fig_moisture_method_advantage.png",
         "$\\Delta$RMSE = harmonic $-$ seasonal-naive (m$^3$ m$^{-3}$)", "b"),
    ]:
        sub = df[df["target"] == target].copy()
        sub["delta"] = sub["harmonic_regression"] - sub["seasonal_naive"]

        vmax = np.abs(sub["delta"]).quantile(0.98)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 5.4))
        sc = ax.scatter(sub["Longitude"], sub["Latitude"], c=sub["delta"],
                         cmap="RdBu_r", norm=norm, s=42, edgecolor="white", linewidth=0.4)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(linewidth=0.4, color="#e1e0d9", zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.12, 1.04, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold")

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9.5)
        cbar.ax.tick_params(labelsize=8.5)

        fig.tight_layout()
        fig.savefig(outdir / fname)
        plt.close(fig)
        print(f"Saved {outdir / fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to reconstruction_rmse_by_location.csv")
    parser.add_argument("--outdir", default="./reconstruction_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    plot_terrain_bars(df, outdir)
    plot_location_maps(df, outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
