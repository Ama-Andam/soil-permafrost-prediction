"""
Compares three practical strategies for estimating the wavelet long-term
component at INFERENCE time, when future data isn't available.

Usage:
    python reconstruction_options.py /path/to/dataset.csv --outdir results/reconstruction --n_sample_locations 256
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

TARGET_COLS = [
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]

WAVELET = "db4"
STEPS_PER_YEAR = int(365.25 * 24 / 3)


def load(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        first_line = f.readline()
        second_line = f.readline()
    header_arg = 0 if "time_utc" in first_line else 1
    df = pd.read_csv(path, header=header_arg)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    return df


def wavelet_long_term(x: np.ndarray, wavelet: str = WAVELET) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    n = len(x)
    level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    long_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    long_term = pywt.waverec(long_coeffs, wavelet, mode="periodization")
    if len(long_term) > n:
        long_term = long_term[:n]
    elif len(long_term) < n:
        long_term = np.pad(long_term, (0, n - len(long_term)), mode="edge")
    return long_term


def option_seasonal_naive(train_long: np.ndarray, n_test: int) -> np.ndarray:
    n_train = len(train_long)
    out = np.empty(n_test)
    for i in range(n_test):
        src_idx = (n_train - STEPS_PER_YEAR) + i
        src_idx = src_idx % n_train
        out[i] = train_long[src_idx]
    return out


def option_harmonic_regression(train_long: np.ndarray, n_test: int) -> np.ndarray:
    n_train = len(train_long)
    t_train = np.arange(n_train)
    omega1 = 2 * np.pi / STEPS_PER_YEAR
    omega2 = 2 * np.pi / (STEPS_PER_YEAR / 2)

    X = np.column_stack([
        np.ones(n_train), t_train,
        np.sin(omega1 * t_train), np.cos(omega1 * t_train),
        np.sin(omega2 * t_train), np.cos(omega2 * t_train),
    ])
    coefs, *_ = np.linalg.lstsq(X, train_long, rcond=None)

    t_test = np.arange(n_train, n_train + n_test)
    X_test = np.column_stack([
        np.ones(n_test), t_test,
        np.sin(omega1 * t_test), np.cos(omega1 * t_test),
        np.sin(omega2 * t_test), np.cos(omega2 * t_test),
    ])
    return X_test @ coefs


def option_last_slope_hold(train_long: np.ndarray, n_test: int, window: int = 30) -> np.ndarray:
    tail = train_long[-window:]
    t_tail = np.arange(window)
    slope, intercept = np.polyfit(t_tail, tail, 1)
    t_test = np.arange(window, window + n_test)
    return intercept + slope * t_test


def evaluate_location(g: pd.DataFrame, target_cols, split_frac=0.8):
    n = len(g)
    n_train = int(n * split_frac)
    n_test = n - n_train

    results = {}
    for col in target_cols:
        full = g[col].values
        oracle_long = wavelet_long_term(full)
        train_series = full[:n_train]
        train_long = wavelet_long_term(train_series)

        oracle_test = oracle_long[n_train:]

        options = {
            "seasonal_naive": option_seasonal_naive(train_long, n_test),
            "harmonic_regression": option_harmonic_regression(train_long, n_test),
            "last_slope_hold": option_last_slope_hold(train_long, n_test),
        }

        rmses = {name: float(np.sqrt(np.mean((pred - oracle_test) ** 2)))
                  for name, pred in options.items()}
        results[col] = {"rmses": rmses, "options": options, "oracle_test": oracle_test,
                         "n_train": n_train, "n_test": n_test}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--outdir", default="./reconstruction_out")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--n_sample_locations", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target_cols = args.targets if args.targets else TARGET_COLS

    print("Loading dataset...")
    df = load(args.csv_path)

    group_keys = ["Site", "Latitude", "Longitude"]
    groups = list(df.groupby(group_keys))
    sample_groups = groups[:args.n_sample_locations]

    all_rmses = []
    plotted_sites = set()
    n_plots_target = 4

    for keys, g in sample_groups:
        g = g.sort_values("time_utc")
        results = evaluate_location(g, target_cols)
        for col, r in results.items():
            row = {"Site": keys[0], "Latitude": keys[1], "Longitude": keys[2], "target": col}
            row.update(r["rmses"])
            all_rmses.append(row)

        site_name = keys[0]
        if site_name not in plotted_sites and len(plotted_sites) < n_plots_target:
            plotted_sites.add(site_name)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(target_cols), 1, figsize=(12, 4 * len(target_cols)))
            if len(target_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, target_cols):
                r = results[col]
                test_time = g["time_utc"].values[r["n_train"]:]
                ax.plot(test_time, r["oracle_test"], label="oracle (uses future info)",
                         linewidth=1.5, color="black")
                for name, pred in r["options"].items():
                    ax.plot(test_time, pred, label=name, linewidth=1.0, alpha=0.8)
                ax.set_title(f"{col} -- {site_name} ({keys[1]:.3f}, {keys[2]:.3f}) -- test period")
                ax.legend(fontsize=8)
            fig.tight_layout()
            fname = outdir / f"reconstruction_options_sample_{site_name}.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"Sample comparison plot saved to {fname}")

    rmse_df = pd.DataFrame(all_rmses)
    rmse_df.to_csv(outdir / "reconstruction_rmse_by_location.csv", index=False)

    print("\nMean RMSE by target and option (lower = closer to oracle = better):")
    summary = rmse_df.groupby("target")[["seasonal_naive", "harmonic_regression", "last_slope_hold"]].mean()
    print(summary.to_string())
    summary.to_csv(outdir / "reconstruction_rmse_summary.csv")
    print(f"\nFull results written to {outdir}/")


if __name__ == "__main__":
    main()
