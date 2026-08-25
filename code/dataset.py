"""
Builds sliding-window sequences from the STL-prepared dataset (output of
stl_pipeline.py) for training the sequence models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLS = [
    "elevation_m", "elev_roughness_m", "slope_deg",
    "temperature_2m", "precipitation", "snow_depth_weather",
    "Temp_K", "SM_Surface", "SM_Rootzone", "Pressure", "Greenness", "Snow_Depth_SMAP",
    "Soil_Temp_L1", "Soil_Temp_L2", "Soil_Temp_L3", "Soil_Temp_L4",
]


class SoilSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: list, split: str,
                 n_lookback: int = 24, feature_cols: list = None,
                 sample_locations: int = None, seed: int = 42):
        self.target_cols = target_cols
        self.short_cols = [f"{c}_short_target" for c in target_cols]
        self.feature_cols = feature_cols if feature_cols else FEATURE_COLS
        self.n_lookback = n_lookback
        self.sample_locations = sample_locations
        self.seed = seed

        self.X, self.y = self._build_windows(df, split)

    def _build_windows(self, df, split):
        X_list, y_list = [], []
        group_keys = ["Site", "Latitude", "Longitude"]

        groups = list(df.groupby(group_keys))
        if self.sample_locations is not None and self.sample_locations < len(groups):
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(groups), size=self.sample_locations, replace=False)
            groups = [groups[i] for i in sorted(idx)]

        self.used_location_keys = [keys for keys, _ in groups]

        for _, g in groups:
            g = g[g["split"] == split].sort_values("time_utc")
            if len(g) <= self.n_lookback:
                continue

            feats = g[self.feature_cols].values.astype(np.float32)
            targs = g[self.short_cols].values.astype(np.float32)

            n = len(g)
            for i in range(self.n_lookback, n):
                X_list.append(feats[i - self.n_lookback:i])
                y_list.append(targs[i])

        X = np.ascontiguousarray(np.stack(X_list))
        y = np.ascontiguousarray(np.stack(y_list))
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    @property
    def n_features(self):
        return len(self.feature_cols)

    @property
    def n_targets(self):
        return len(self.target_cols)


def compute_feature_scaler(train_ds: SoilSequenceDataset):
    flat = train_ds.X.reshape(-1, train_ds.X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(ds: SoilSequenceDataset, mean, std):
    ds.X = (ds.X - mean) / std
