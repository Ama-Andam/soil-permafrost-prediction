"""
Trains all 10 sequence models on the STL-prepared dataset, using Ray to
parallelize training across models. Writes leaderboard results to disk
incrementally, after each model finishes -- so a time-limit kill doesn't
lose completed results.

Usage:
    python train.py results/stl_prepared/dataset_prepared_for_training.csv \
        --outdir results/training --epochs 15 --n_lookback 24
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn

from dataset import SoilSequenceDataset, compute_feature_scaler, apply_scaler
from models_library import MODEL_REGISTRY
from s4_model import S4DModel
from mamba_model import MambaModel

TARGET_COLS = [
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]

FULL_REGISTRY = dict(MODEL_REGISTRY)
FULL_REGISTRY["S4D"] = S4DModel
FULL_REGISTRY["Mamba"] = MambaModel


def load_prepared(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    return df


def train_one_model(model_cls, train_X, train_y, test_X, n_features, n_targets,
                     epochs, batch_size, lr, device, model_name="model"):
    torch.manual_seed(42)
    model = model_cls(n_features=n_features, n_targets=n_targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_X_t = torch.from_numpy(train_X)
    train_y_t = torch.from_numpy(train_y)
    n_train = len(train_X_t)

    model.train()
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_X_t[idx].to(device)
            yb = train_y_t[idx].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if torch.isnan(loss):
                print(f"[{model_name}] NaN loss at epoch {epoch}, batch {n_batches} -- stopping early", flush=True)
                history.append({"epoch": epoch, "loss": float("nan")})
                return model, history, True, None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        print(f"[{model_name}] epoch {epoch + 1}/{epochs}  loss={avg_loss:.5f}  "
              f"({epoch_time:.1f}s, {n_batches} batches)", flush=True)
        history.append({"epoch": epoch, "loss": avg_loss})

    model.eval()
    with torch.no_grad():
        test_preds = []
        test_X_t = torch.from_numpy(test_X)
        for i in range(0, len(test_X_t), batch_size):
            xb = test_X_t[i:i + batch_size].to(device)
            test_preds.append(model(xb).cpu().numpy())
        test_preds = np.concatenate(test_preds, axis=0)

    return model, history, False, test_preds


def evaluate_reconstructed(test_preds_short, test_long_component, test_raw, target_idx):
    pred_raw = test_preds_short[:, target_idx] + test_long_component
    actual_raw = test_raw

    rmse = float(np.sqrt(np.mean((pred_raw - actual_raw) ** 2)))
    ss_res = np.sum((actual_raw - pred_raw) ** 2)
    ss_tot = np.sum((actual_raw - actual_raw.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


@ray.remote
def train_and_evaluate_remote(model_name, train_X, train_y, test_X, test_y,
                               n_features, n_targets, epochs, batch_size, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = FULL_REGISTRY[model_name]

    print(f"[{model_name}] starting on device={device}", flush=True)
    start = time.time()
    model, history, nan_failure, test_preds = train_one_model(
        model_cls, train_X, train_y, test_X, n_features, n_targets,
        epochs, batch_size, lr, device, model_name=model_name,
    )
    elapsed = time.time() - start

    if nan_failure:
        print(f"[{model_name}] FAILED (NaN) after {elapsed:.1f}s", flush=True)
        return {"model": model_name, "status": "nan_failure", "elapsed_s": elapsed,
                "history": history, "test_preds": None}

    print(f"[{model_name}] DONE in {elapsed:.1f}s", flush=True)
    return {"model": model_name, "status": "ok", "elapsed_s": elapsed,
            "history": history, "test_preds": test_preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_csv", help="Output of stl_pipeline.py")
    parser.add_argument("--outdir", default="./training_out")
    parser.add_argument("--targets", nargs="+", default=TARGET_COLS)
    parser.add_argument("--n_lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models", nargs="+", default=None,
                         help="Subset of models to run (default: all 10)")
    parser.add_argument("--sample_locations", type=int, default=None,
                         help="Randomly sample this many locations instead of all 256")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading prepared dataset...")
    df = load_prepared(args.prepared_csv)

    print(f"Building sliding windows (n_lookback={args.n_lookback})...")
    train_ds = SoilSequenceDataset(df, args.targets, split="train", n_lookback=args.n_lookback,
                                    sample_locations=args.sample_locations)
    test_ds = SoilSequenceDataset(df, args.targets, split="test", n_lookback=args.n_lookback,
                                   sample_locations=args.sample_locations)
    print(f"Train windows: {len(train_ds):,}  |  Test windows: {len(test_ds):,}")

    mean, std = compute_feature_scaler(train_ds)
    apply_scaler(train_ds, mean, std)
    apply_scaler(test_ds, mean, std)

    models_to_run = args.models if args.models else list(FULL_REGISTRY.keys())
    print(f"Training models: {models_to_run}")

    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        print("No Ray cluster found at 'auto' -- falling back to a local Ray instance.")
        ray.init(ignore_reinit_error=True)

    cluster_resources = ray.cluster_resources()
    n_gpus_available = cluster_resources.get("GPU", 0)
    if n_gpus_available > 0:
        gpu_frac = min(1.0, n_gpus_available / max(1, min(len(models_to_run), 4)))
        remote_fn = train_and_evaluate_remote.options(num_gpus=gpu_frac)
        print(f"Ray cluster reports {n_gpus_available} GPU(s) -- "
              f"scheduling each model task with num_gpus={gpu_frac:.2f}")
    else:
        remote_fn = train_and_evaluate_remote
        print("Ray cluster reports 0 GPUs -- running on CPU.")

    futures = [
        remote_fn.remote(
            name, train_ds.X, train_ds.y, test_ds.X, test_ds.y,
            train_ds.n_features, train_ds.n_targets,
            args.epochs, args.batch_size, args.lr,
        )
        for name in models_to_run
    ]

    print(f"\nSubmitted {len(futures)} model training tasks. Waiting for results "
          f"(each will print, and be saved to disk, as it finishes)...\n")

    long_components = {c: [] for c in args.targets}
    raw_actuals = {c: [] for c in args.targets}
    used_keys_set = set(test_ds.used_location_keys)
    group_keys = ["Site", "Latitude", "Longitude"]
    for keys, g in df.groupby(group_keys):
        if keys not in used_keys_set:
            continue
        g = g[g["split"] == "test"].sort_values("time_utc")
        if len(g) <= args.n_lookback:
            continue
        for c in args.targets:
            long_components[c].extend(g[f"{c}_long_component"].values[args.n_lookback:])
            raw_actuals[c].extend(g[c].values[args.n_lookback:])
    for c in args.targets:
        long_components[c] = np.array(long_components[c])
        raw_actuals[c] = np.array(raw_actuals[c])

    leaderboard_path = outdir / "model_leaderboard.csv"
    leaderboard_rows = []
    results = []
    remaining = futures
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        res = ray.get(done[0])
        results.append(res)
        print(f">>> [{len(results)}/{len(futures)}] {res['model']} finished "
              f"(status={res['status']}, {res['elapsed_s']:.1f}s)\n", flush=True)

        if res["status"] == "nan_failure":
            for c in args.targets:
                leaderboard_rows.append({
                    "model": res["model"], "target": c, "status": "nan_failure",
                    "rmse": None, "r2": None, "elapsed_s": res["elapsed_s"],
                })
        else:
            for i, c in enumerate(args.targets):
                rmse, r2 = evaluate_reconstructed(res["test_preds"], long_components[c], raw_actuals[c], i)
                leaderboard_rows.append({
                    "model": res["model"], "target": c, "status": "ok",
                    "rmse": rmse, "r2": r2, "elapsed_s": res["elapsed_s"],
                })

        pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)

    leaderboard = pd.DataFrame(leaderboard_rows)
    print("\n" + "=" * 70)
    print("LEADERBOARD (reconstructed raw-scale metrics)")
    print("=" * 70)
    print(leaderboard.sort_values(["target", "rmse"]).to_string(index=False))
    print(f"\nFull results written to {outdir}/model_leaderboard.csv")


if __name__ == "__main__":
    main()
