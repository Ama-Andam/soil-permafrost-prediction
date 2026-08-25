"""
Ties together the wavelet decomposition + harmonic-regression reconstruction
(validated against the oracle comparison) into a single data-prep pipeline
that any model in the roster -- including S4D and Mamba -- trains against.

Key idea: models NEVER train on raw soil temp/moisture directly. They train
on the short-term (residual) component, with the seasonal/trend cycle
already removed. This is Meisam's suggestion -- avoid burning model capacity
re-learning "it's cold in winter" and focus it on the harder residual
dynamics.

To keep this leakage-free:
  - The long-term component used to build TRAINING targets is computed from
    the TRAIN split only (not the full series).
  - The long-term component used to RECONSTRUCT raw-scale predictions on the
    TEST split is estimated causally via harmonic regression (the option
    that won the earlier oracle-comparison: RMSE 4.59 vs 5.15 for temp,
    0.046 vs 0.056 for moisture, both far better than last-slope-hold).

Output of `prepare_dataset()`: the original dataframe plus, per target:
    {target}_short_target   -- what models train on (the label)
    {target}_long_component -- what gets added back at evaluation time
    split                   -- 'train' or 'test', chronological, per location

Usage as a library:
    from stl_pipeline import prepare_dataset, reconstruct_predictions

    df = prepare_dataset(df, target_cols=[...], split_frac=0.8)
    # ... train models on df[f"{target}_short_target"] using train rows ...
    # ... model predicts short-term values for test rows ...
    raw_preds = reconstruct_predictions(df, target, short_term_preds, test_mask)
"""

import numpy as np
import pandas as pd
import pywt

STEPS_PER_YEAR = int(365.25 * 24 / 3)  # dataset cadence is 3-hourly
WAVELET = "db4"


def _wavelet_long_term(x: np.ndarray, wavelet: str = WAVELET) -> np.ndarray:
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


def _harmonic_extrapolate(train_long: np.ndarray, n_test: int) -> np.ndarray:
    """Fit trend + annual/semi-annual sinusoids on train_long, extrapolate forward."""
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


def prepare_dataset(df: pd.DataFrame, target_cols: list, split_frac: float = 0.8) -> pd.DataFrame:
    """
    Adds {target}_short_target, {target}_long_component, and split columns.
    Must be called per-location internally -- groups by exact (Site, Lat, Lon)
    to avoid mixing distinct physical time series (see earlier EDA note: each
    Site label bundles up to 64 distinct locations).
    """
    group_keys = ["Site", "Latitude", "Longitude"]
    out_frames = []

    n_groups = df.groupby(group_keys).ngroups
    print(f"Preparing short-term targets for {len(target_cols)} target(s) "
          f"across {n_groups} physical locations...")

    for i, (keys, g) in enumerate(df.groupby(group_keys)):
        g = g.sort_values("time_utc").reset_index(drop=True).copy()
        n = len(g)
        n_train = int(n * split_frac)

        split_col = np.array(["train"] * n_train + ["test"] * (n - n_train))
        g["split"] = split_col

        for col in target_cols:
            raw = g[col].values
            train_raw = raw[:n_train]

            train_long = _wavelet_long_term(train_raw)
            test_long_est = _harmonic_extrapolate(train_long, n - n_train)

            long_component = np.concatenate([train_long, test_long_est])
            short_target = raw - long_component  # train: causal residual; test: residual vs. estimate

            g[f"{col}_long_component"] = long_component
            g[f"{col}_short_target"] = short_target

        out_frames.append(g)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n_groups} locations done")

    result = pd.concat(out_frames, axis=0).sort_values(group_keys + ["time_utc"]).reset_index(drop=True)
    return result


def reconstruct_predictions(df: pd.DataFrame, target_col: str,
                             short_term_preds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Converts model's short-term predictions back to raw-scale values for
    evaluation, using the pre-computed long_component column.

    df:               the prepared dataframe (output of prepare_dataset)
    target_col:       e.g. "soil_temperature_0_to_7cm"
    short_term_preds: model's predicted short-term values, aligned to df[mask]
    mask:             boolean array selecting the rows short_term_preds corresponds to
    """
    long_component = df.loc[mask, f"{target_col}_long_component"].values
    return short_term_preds + long_component


def sanity_check(df: pd.DataFrame, target_cols: list):
    """Confirms short_target + long_component == raw on the TRAIN split (should be ~exact),
    and reports typical reconstruction gap on TEST (expected to be nonzero -- that's the
    harmonic-regression estimation error we already measured in the RMSE comparison)."""
    for col in target_cols:
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"

        train_recon = df.loc[train_mask, f"{col}_short_target"] + df.loc[train_mask, f"{col}_long_component"]
        train_err = (train_recon - df.loc[train_mask, col]).abs().max()

        test_recon_err = (df.loc[test_mask, f"{col}_long_component"] - df.loc[test_mask, col]).abs()
        print(f"{col}: train reconstruction max error = {train_err:.2e} (should be ~0)")
        print(f"{col}: test long-term-only gap (RMSE) = {np.sqrt((test_recon_err**2).mean()):.4f} "
              f"(this is the harmonic-regression estimation error, expected nonzero)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--outdir", default="./stl_prepared")
    parser.add_argument("--targets", nargs="+",
                         default=["soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"])
    parser.add_argument("--split_frac", type=float, default=0.8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.csv_path) as f:
        first_line = f.readline()
    header_arg = 0 if "time_utc" in first_line else 1
    df = pd.read_csv(args.csv_path, header=header_arg)
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    prepared = prepare_dataset(df, args.targets, split_frac=args.split_frac)

    out_path = outdir / "dataset_prepared_for_training.csv"
    prepared.to_csv(out_path, index=False)
    print(f"\nPrepared dataset written to {out_path}")

    sanity_check(prepared, args.targets)