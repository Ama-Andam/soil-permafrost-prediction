"""
EDA for the new unified soil dataset (weather + topography + SMAP).
Run on TALON. Produces a text report + a handful of diagnostic plots.

Usage:
    python eda.py /path/to/dataset.csv --outdir ./eda_out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...")

    with open(path, "r") as f:
        first_line = f.readline()
        second_line = f.readline()

    if "time_utc" in first_line:
        header_arg = 0
    elif "time_utc" in second_line:
        header_arg = 1
        print("Detected a two-row header (category row above column-name row) "
              "-- skipping the category row.")
    else:
        print("WARNING: 'time_utc' not found in first two lines. "
              "Loading without parse_dates so you can inspect df.columns.")
        df = pd.read_csv(path)
        print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols. Columns found:")
        print(list(df.columns))
        return df

    df = pd.read_csv(path, header=header_arg)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols "
          f"({df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory)\n")
    return df


def basic_info(df: pd.DataFrame, out):
    print("=" * 80, file=out)
    print("BASIC INFO", file=out)
    print("=" * 80, file=out)
    print(df.dtypes, file=out)
    print(file=out)
    print("Column-level null counts:", file=out)
    print(df.isna().sum().sort_values(ascending=False), file=out)
    print(file=out)


def site_summary(df: pd.DataFrame, out):
    print("=" * 80, file=out)
    print("SITE SUMMARY", file=out)
    print("=" * 80, file=out)
    sites = df["Site"].unique()
    print(f"Number of unique sites: {len(sites)}", file=out)
    print(f"Sites: {sorted(sites)}\n", file=out)

    rows = []
    for site, g in df.groupby("Site"):
        rows.append({
            "Site": site,
            "n_rows": len(g),
            "n_unique_coords": g[["Latitude", "Longitude"]].drop_duplicates().shape[0],
            "n_smap_nodes": g[["smap_node_x", "smap_node_y"]].drop_duplicates().shape[0],
            "date_min": g["time_utc"].min(),
            "date_max": g["time_utc"].max(),
        })
    summary = pd.DataFrame(rows).sort_values("Site")
    print(summary.to_string(index=False), file=out)
    print(file=out)

    multi = summary[summary["n_unique_coords"] > 1]
    if len(multi):
        print("NOTE: these 'Site' labels contain multiple distinct lat/lon pairs "
              "(likely multiple station locations grouped under one name):", file=out)
        print(multi[["Site", "n_unique_coords"]].to_string(index=False), file=out)
        print(file=out)


def time_coverage(df: pd.DataFrame, out):
    print("=" * 80, file=out)
    print("TIME COVERAGE / GAP CHECK (expects hourly cadence)", file=out)
    print("=" * 80, file=out)
    for site, g in df.groupby("Site"):
        g = g.sort_values("time_utc")
        ts = g["time_utc"].drop_duplicates().sort_values()
        diffs = ts.diff().dropna()
        expected = pd.Timedelta(hours=1)
        n_gaps = (diffs > expected).sum()
        biggest_gap = diffs.max() if len(diffs) else pd.Timedelta(0)
        print(f"{site:12s} | rows(ts)={len(ts):6d} | span={ts.min()} -> {ts.max()} "
              f"| gaps>1h: {n_gaps} | largest gap: {biggest_gap}", file=out)
    print(file=out)


def zero_run_check(df: pd.DataFrame, out, cols=("soil_moisture_0_to_7cm", "SM_Surface", "SM_Rootzone")):
    print("=" * 80, file=out)
    print("ZERO-RUN CHECK ON MOISTURE COLUMNS", file=out)
    print("(flags exact-zero streaks -- likely sensor fault / fill value, not physical)", file=out)
    print("=" * 80, file=out)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("No moisture columns found matching expected names.", file=out)
        return

    for site, g in df.groupby("Site"):
        g = g.sort_values("time_utc")
        for col in cols:
            is_zero = (g[col] == 0)
            if not is_zero.any():
                continue
            grp_id = (is_zero != is_zero.shift()).cumsum()
            runs = g.groupby(grp_id).apply(
                lambda x: (x[col].iloc[0] == 0, len(x), x["time_utc"].iloc[0], x["time_utc"].iloc[-1])
                if is_zero.loc[x.index[0]] else None
            )
            zero_runs = [r for r in runs if r is not None and r[0]]
            if zero_runs:
                total_zero_hours = sum(r[1] for r in zero_runs)
                longest = max(zero_runs, key=lambda r: r[1])
                pct = 100 * total_zero_hours / len(g)
                print(f"{site:12s} | {col:22s} | zero-run count: {len(zero_runs):4d} "
                      f"| total zero hours: {total_zero_hours:6d} ({pct:5.1f}% of records) "
                      f"| longest run: {longest[1]} hrs [{longest[2]} -> {longest[3]}]", file=out)
    print(file=out)


def smap_cadence_check(df: pd.DataFrame, out,
                        smap_cols=("Temp_K", "SM_Surface", "SM_Rootzone", "Pressure",
                                   "Greenness", "Snow_Depth_SMAP",
                                   "Soil_Temp_L1", "Soil_Temp_L2", "Soil_Temp_L3", "Soil_Temp_L4")):
    print("=" * 80, file=out)
    print("SMAP CADENCE CHECK", file=out)
    print("(SMAP doesn't revisit hourly -- checks how values repeat, i.e. fill/interp method)", file=out)
    print("=" * 80, file=out)
    smap_cols = [c for c in smap_cols if c in df.columns]
    sample_site = df["Site"].unique()[0]
    g = df[df["Site"] == sample_site].sort_values("time_utc")
    for col in smap_cols:
        change_count = (g[col].diff() != 0).sum()
        print(f"[{sample_site}] {col:18s} | unique consecutive values: {change_count:6d} "
              f"/ {len(g):6d} rows ({100*change_count/len(g):5.1f}% change rate)", file=out)
    print(file=out)


def target_candidate_check(df: pd.DataFrame, out):
    print("=" * 80, file=out)
    print("TARGET-RELATED COLUMNS PRESENT", file=out)
    print("=" * 80, file=out)
    likely_targets = [c for c in df.columns if
                       "soil_temp" in c.lower() or "soil_moisture" in c.lower() or "sm_" in c.lower()]
    print(f"Columns that look like soil temp/moisture targets or SMAP soil layers:", file=out)
    print(likely_targets, file=out)
    print("\nCheck manually: which of these are ground-truth targets vs SMAP input covariates.", file=out)
    print(file=out)


def numeric_describe(df: pd.DataFrame, out):
    print("=" * 80, file=out)
    print("NUMERIC SUMMARY STATS", file=out)
    print("=" * 80, file=out)
    print(df.describe().T.to_string(), file=out)
    print(file=out)


def make_plots(df: pd.DataFrame, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    miss = df.isna().mean().sort_values(ascending=False)
    miss[miss > 0].plot(kind="barh", ax=ax)
    ax.set_title("Fraction missing per column")
    fig.tight_layout()
    fig.savefig(outdir / "missingness.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    df["Site"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Row count per site")
    fig.tight_layout()
    fig.savefig(outdir / "rows_per_site.png", dpi=120)
    plt.close(fig)

    moisture_cols = [c for c in ["soil_moisture_0_to_7cm", "SM_Surface", "SM_Rootzone"] if c in df.columns]
    sites = sorted(df["Site"].unique())[:3]
    if moisture_cols and sites:
        fig, axes = plt.subplots(len(sites), 1, figsize=(12, 3 * len(sites)), sharex=True)
        if len(sites) == 1:
            axes = [axes]
        for ax, site in zip(axes, sites):
            g = df[df["Site"] == site].sort_values("time_utc")
            for col in moisture_cols:
                ax.plot(g["time_utc"], g[col], label=col, linewidth=0.7)
            ax.set_title(site)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "moisture_timeseries_sample.png", dpi=120)
        plt.close(fig)

    print(f"Plots saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the dataset CSV")
    parser.add_argument("--outdir", default="./eda_out", help="Output directory for report + plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "eda_report.txt"

    df = load(args.csv_path)

    with open(report_path, "w") as out:
        basic_info(df, out)
        site_summary(df, out)
        time_coverage(df, out)
        target_candidate_check(df, out)
        zero_run_check(df, out)
        smap_cadence_check(df, out)
        numeric_describe(df, out)

    print(f"\nText report written to {report_path}")

    try:
        make_plots(df, outdir)
    except ImportError:
        print("matplotlib not available -- skipping plots. Text report is still complete.")


if __name__ == "__main__":
    main()
