"""
Wavelet decomposition of soil target series into long-term (trend/seasonal
memory) and short-term (residual) components, per Meisam's suggestion.

Uses a standard DWT (via PyWavelets) rather than plain FFT, since the soil
series are non-stationary (freeze-thaw amplitude/phase shifts across years,
plus trend) -- wavelets localize in both time and frequency, FFT does not.

IMPORTANT: decomposition is done per exact physical location (unique
Site + Latitude + Longitude), NOT per Site label. Each Site label in this
dataset (Bedrock/Transition/Upland/Wetland) bundles up to 64 distinct
physical points -- decomposing at the Site level would mix unrelated time
series together and produce meaningless results.

Output: for each target column, adds two new columns:
    {target}_long   -- reconstructed trend+seasonal component (approximation)
    {target}_short  -- residual (raw - long), this is what gets predicted

To reconstruct a real-valued prediction at inference:
    prediction_raw = predicted_short + long_term_component

Usage:
    python wavelet_decompose.py /path/to/dataset.csv --outdir results/wavelet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

# ---------------------------------------------------------------------------
# CONFIG -- edit TARGET_COLS once the actual 4-depth target set is confirmed.
# Currently only 2 explicit ground-truth columns exist in the merged dataset;
# the other 2 depths (if used) need to come from a separate ground-truth
# source or be explicitly defined against the SMAP layers.
# ---------------------------------------------------------------------------
TARGET_COLS = [
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]

WAVELET = "db4"      # Daubechies-4: smooth, commonly used for geophysical series
LEVEL = None         # None = auto (max level for the series length); set an int to fix it


def load(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        first_line = f.readline()
        second_line = f.readline()
    header_arg = 0 if "time_utc" in first_line else 1
    df = pd.read_csv(path, header=header_arg)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    return df


def decompose_series(x: np.ndarray, wavelet: str = WAVELET, level=LEVEL) -> np.ndarray:
    """
    Returns the long-term (approximation-only) reconstruction of x, same
    length as x. Short-term residual = x - long_term.
    """
    n = len(x)
    x = np.array(x, dtype=np.float64)  # ensure a writable, contiguous copy
    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    level = max(1, level)

    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
    # Zero out all detail coefficients, keep only the approximation -> long-term signal
    long_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    long_term = pywt.waverec(long_coeffs, wavelet, mode="periodization")

    # periodization mode can return a series 1 sample longer/shorter -- trim/pad to match
    if len(long_term) > n:
        long_term = long_term[:n]
    elif len(long_term) < n:
        long_term = np.pad(long_term, (0, n - len(long_term)), mode="edge")

    return long_term


def process(df: pd.DataFrame, target_cols=TARGET_COLS) -> pd.DataFrame:
    group_keys = ["Site", "Latitude", "Longitude"]
    out_frames = []

    n_groups = df.groupby(group_keys).ngroups
    print(f"Decomposing {len(target_cols)} target(s) across {n_groups} physical locations...")

    for i, (keys, g) in enumerate(df.groupby(group_keys)):
        g = g.sort_values("time_utc").copy()
        for col in target_cols:
            long_term = decompose_series(g[col].values)
            g[f"{col}_long"] = long_term
            g[f"{col}_short"] = g[col].values - long_term
        out_frames.append(g)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n_groups} locations done")

    result = pd.concat(out_frames, axis=0).sort_values(group_keys + ["time_utc"]).reset_index(drop=True)
    return result


def sanity_check(df: pd.DataFrame, target_cols, outdir: Path):
    """Quick numeric + visual check that long + short reconstructs the raw signal."""
    report_lines = []
    for col in target_cols:
        raw = df[col].values
        recon = df[f"{col}_long"].values + df[f"{col}_short"].values
        max_err = np.max(np.abs(raw - recon))
        report_lines.append(f"{col}: max reconstruction error = {max_err:.2e} (should be ~0)")
    report_text = "\n".join(report_lines)
    print(report_text)
    (outdir / "wavelet_sanity_check.txt").write_text(report_text + "\n")

    # plot one location per terrain class as a visual check
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = ["Site", "Latitude", "Longitude"]
        unique_locs = df[keys].drop_duplicates()
        sites_seen = []
        rep_locs = []
        for _, row in unique_locs.iterrows():
            if row["Site"] not in sites_seen:
                sites_seen.append(row["Site"])
                rep_locs.append(row)

        for loc in rep_locs:
            mask = (df[keys] == loc.values).all(axis=1)
            g = df[mask].sort_values("time_utc")

            fig, axes = plt.subplots(len(target_cols), 1, figsize=(12, 4 * len(target_cols)))
            if len(target_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, target_cols):
                ax.plot(g["time_utc"], g[col], label="raw", linewidth=0.6, alpha=0.6)
                ax.plot(g["time_utc"], g[f"{col}_long"], label="long-term (trend+seasonal)", linewidth=1.2)
                ax.plot(g["time_utc"], g[f"{col}_short"], label="short-term (residual)", linewidth=0.5, alpha=0.7)
                ax.set_title(f"{col} -- {loc['Site']} ({loc['Latitude']:.3f}, {loc['Longitude']:.3f})")
                ax.legend(fontsize=8)
            fig.tight_layout()
            fname = outdir / f"wavelet_decomposition_sample_{loc['Site']}.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"Sample decomposition plot saved to {fname}")
    except ImportError:
        print("matplotlib not available -- skipping plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the dataset CSV")
    parser.add_argument("--outdir", default="./wavelet_out", help="Output directory")
    parser.add_argument("--targets", nargs="+", default=None,
                         help="Override target columns to decompose (default: TARGET_COLS in script)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_cols = args.targets if args.targets else TARGET_COLS

    print("Loading dataset...")
    df = load(args.csv_path)
    print(f"Loaded {df.shape[0]:,} rows.")

    result = process(df, target_cols)

    out_path = outdir / "dataset_with_wavelet_decomposition.csv"
    result.to_csv(out_path, index=False)
    print(f"\nDecomposed dataset written to {out_path}")

    sanity_check(result, target_cols, outdir)


if __name__ == "__main__":
    main()