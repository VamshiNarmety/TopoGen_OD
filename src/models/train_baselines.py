#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from functools import lru_cache
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class ODDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.origin = torch.tensor(df["origin"].values, dtype=torch.long)
        self.destination = torch.tensor(df["destination"].values, dtype=torch.long)
        cont_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.target_log = torch.tensor(np.log1p(df["trip_count"].values), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.origin)

    def __getitem__(self, idx: int):
        return {
            "origin": self.origin[idx],
            "destination": self.destination[idx],
            "cont": self.cont[idx],
            "target_log": self.target_log[idx],
        }


class ODMLP(nn.Module):
    def __init__(self, n_origins: int, n_destinations: int, emb_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.origin_emb = nn.Embedding(n_origins + 1, emb_dim)
        self.destination_emb = nn.Embedding(n_destinations + 1, emb_dim)
        in_dim = emb_dim * 2 + 5
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        o = self.origin_emb(origin)
        d = self.destination_emb(destination)
        x = torch.cat([o, d, cont], dim=1)
        out = self.mlp(x)
        return out.squeeze(1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_od(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input OD file not found: {input_path}")

    df = pd.read_parquet(input_path)
    required = {"pickup_hour", "origin", "destination", "trip_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OD file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce")
    df = df.dropna(subset=["pickup_hour", "origin", "destination", "trip_count"])
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce")
    df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
    df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce")
    df = df.dropna(subset=["origin", "destination", "trip_count"])
    df = df[(df["origin"] > 0) & (df["destination"] > 0) & (df["trip_count"] >= 0)].copy()
    df["origin"] = df["origin"].astype("int32")
    df["destination"] = df["destination"].astype("int32")
    df["trip_count"] = df["trip_count"].astype("float32")
    return df.sort_values("pickup_hour").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["pickup_hour"].dt.hour.astype("int16")
    out["dayofweek"] = out["pickup_hour"].dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("float32")

    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0).astype("float32")
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0).astype("float32")
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dayofweek"] / 7.0).astype("float32")
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dayofweek"] / 7.0).astype("float32")
    return out


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_df = df[df["pickup_hour"] <= train_end_ts].copy()
    val_df = df[(df["pickup_hour"] > train_end_ts) & (df["pickup_hour"] <= val_end_ts)].copy()
    test_df = df[df["pickup_hour"] > val_end_ts].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Temporal split produced an empty set. Check --train-end and --val-end values."
        )

    return train_df, val_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _group_mean_lookup(df: pd.DataFrame, key_cols: list[str], value_col: str = "trip_count") -> pd.DataFrame:
    return df.groupby(key_cols, as_index=False)[value_col].mean().rename(columns={value_col: "value"})


def _infer_network_dir(input_path: Path, network_dir: str | None) -> Path:
    if network_dir is not None:
        return Path(network_dir)
    return input_path.resolve().parent.parent / "network"


def _first_matching_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _discover_zone_shapefile(raw_misc_dir: Path) -> Path:
    zip_candidates = sorted(raw_misc_dir.rglob("*.zip"))
    if zip_candidates:
        return zip_candidates[0]

    candidates = sorted(raw_misc_dir.rglob("*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No shapefile found under: {raw_misc_dir}")
    return candidates[0]


@lru_cache(maxsize=8)
def load_zone_centroids(zone_shapefile: str) -> dict[int, tuple[float, float]]:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyshp is required for gravity/radiation baselines. Install it with `pip install pyshp`."
        ) from exc

    shp_path = Path(zone_shapefile)
    if not shp_path.exists():
        raise FileNotFoundError(f"Taxi zone shapefile not found: {shp_path}")

    if shp_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(shp_path) as zf:
                zf.extractall(tmpdir)
            extracted = sorted(Path(tmpdir).rglob("*.shp"))
            if not extracted:
                raise FileNotFoundError(f"No .shp found inside archive: {shp_path}")
            return load_zone_centroids(str(extracted[0]))

    reader = shapefile.Reader(str(shp_path))
    field_names = [field[0] for field in reader.fields[1:]]
    if "LocationID" not in field_names:
        raise ValueError(f"Shapefile missing LocationID field: {shp_path}")
    location_idx = field_names.index("LocationID")

    centroids: dict[int, tuple[float, float]] = {}
    for record, shp in zip(reader.records(), reader.shapes()):
        location_id = int(record[location_idx])
        points = np.asarray(shp.points, dtype=np.float64)
        if len(points) == 0:
            continue
        # A lightweight centroid approximation based on the average of boundary vertices.
        # This avoids relying on a full GIS stack while still preserving relative distances.
        centroid_x = float(np.mean(points[:, 0]))
        centroid_y = float(np.mean(points[:, 1]))
        centroids[location_id] = (centroid_x, centroid_y)

    if not centroids:
        raise ValueError(f"No zone centroids could be read from: {shp_path}")
    return centroids


def compute_time_multiplier(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train_df["trip_count"].mean()) if len(train_df) else 1.0
    main_lookup = _group_mean_lookup(train_df, ["hour", "dayofweek"])
    hour_lookup = _group_mean_lookup(train_df, ["hour"])
    dow_lookup = _group_mean_lookup(train_df, ["dayofweek"])

    out = pred_df[["hour", "dayofweek"]].copy()
    out = out.merge(main_lookup, on=["hour", "dayofweek"], how="left")
    out = out.rename(columns={"value": "main_value"})
    out = out.merge(hour_lookup, on=["hour"], how="left")
    out = out.rename(columns={"value": "hour_value"})
    out = out.merge(dow_lookup, on=["dayofweek"], how="left")
    out = out.rename(columns={"value": "dow_value"})

    values = out["main_value"].fillna(out["hour_value"]).fillna(out["dow_value"]).fillna(global_mean)
    multiplier = (values / max(global_mean, 1e-6)).astype(np.float32)
    return multiplier.to_numpy()


def _default_zone_xy(centroids: dict[int, tuple[float, float]]) -> tuple[float, float]:
    coords = np.asarray(list(centroids.values()), dtype=np.float64)
    if len(coords) == 0:
        return (0.0, 0.0)
    return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))


def _coords_for_zones(zone_ids: np.ndarray, centroids: dict[int, tuple[float, float]]) -> np.ndarray:
    default_xy = _default_zone_xy(centroids)
    coords = np.empty((len(zone_ids), 2), dtype=np.float64)
    for i, zone_id in enumerate(zone_ids):
        coords[i] = centroids.get(int(zone_id), default_xy)
    return coords


def _pair_distance_lookup(zone_ids: list[int], centroids: dict[int, tuple[float, float]]) -> dict[tuple[int, int], float]:
    zone_arr = np.asarray(zone_ids, dtype=np.int32)
    coords = _coords_for_zones(zone_arr, centroids)
    dist_lookup: dict[tuple[int, int], float] = {}
    for i, origin in enumerate(zone_arr):
        diffs = coords - coords[i]
        dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))
        dists = np.maximum(dists, 1.0)
        for j, dest in enumerate(zone_arr):
            dist_lookup[(int(origin), int(dest))] = float(dists[j])
    return dist_lookup


def _fit_time_adjusted_pair_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    time_factor = compute_time_multiplier(train_df, train_df)
    adjusted = train_df[["origin", "destination", "trip_count"]].copy()
    adjusted["time_multiplier"] = time_factor
    adjusted["adj_trip_count"] = adjusted["trip_count"] / np.maximum(adjusted["time_multiplier"], 1e-6)
    pair_df = (
        adjusted.groupby(["origin", "destination"], as_index=False)
        .agg(adj_trip_count=("adj_trip_count", "mean"), trip_count=("trip_count", "mean"))
        .reset_index(drop=True)
    )
    return pair_df


def _zone_marginals(train_df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float], float]:
    origin_totals = train_df.groupby("origin")["trip_count"].sum().to_dict()
    dest_totals = train_df.groupby("destination")["trip_count"].sum().to_dict()
    global_mean = float(train_df["trip_count"].mean()) if len(train_df) else 1.0
    return (
        {int(k): float(v) for k, v in origin_totals.items()},
        {int(k): float(v) for k, v in dest_totals.items()},
        global_mean,
    )


def predict_od_marginal(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    origin_totals, dest_totals, global_mean = _zone_marginals(train_df)
    origin_vals = np.asarray([origin_totals.get(int(z), global_mean) for z in pred_df["origin"].values], dtype=np.float64)
    dest_vals = np.asarray([dest_totals.get(int(z), global_mean) for z in pred_df["destination"].values], dtype=np.float64)
    numerator = origin_vals * dest_vals
    scale = float(train_df["trip_count"].mean()) / max(float(np.mean(numerator)), 1e-6)
    return (scale * numerator).astype(np.float32)


def predict_gravity(train_df: pd.DataFrame, pred_df: pd.DataFrame, centroids: dict[int, tuple[float, float]]) -> np.ndarray:
    pair_df = _fit_time_adjusted_pair_frame(train_df)
    origin_totals, dest_totals, global_mean = _zone_marginals(train_df)
    zone_ids = sorted(
        set(train_df["origin"].unique())
        .union(set(train_df["destination"].unique()))
        .union(set(pred_df["origin"].unique()))
        .union(set(pred_df["destination"].unique()))
    )
    dist_lookup = _pair_distance_lookup(zone_ids, centroids)

    fit_df = pair_df.copy()
    fit_df["origin_total"] = fit_df["origin"].map(lambda z: origin_totals.get(int(z), global_mean)).astype(float)
    fit_df["dest_total"] = fit_df["destination"].map(lambda z: dest_totals.get(int(z), global_mean)).astype(float)
    fit_df["distance"] = [dist_lookup.get((int(o), int(d)), 1.0) for o, d in zip(fit_df["origin"].values, fit_df["destination"].values)]

    y = np.log1p(fit_df["adj_trip_count"].to_numpy(dtype=np.float64))
    X = np.column_stack(
        [
            np.ones(len(fit_df), dtype=np.float64),
            np.log1p(fit_df["origin_total"].to_numpy(dtype=np.float64)),
            np.log1p(fit_df["dest_total"].to_numpy(dtype=np.float64)),
            -np.log(fit_df["distance"].to_numpy(dtype=np.float64)),
        ]
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    origin_vals = np.asarray([origin_totals.get(int(z), global_mean) for z in pred_df["origin"].values], dtype=np.float64)
    dest_vals = np.asarray([dest_totals.get(int(z), global_mean) for z in pred_df["destination"].values], dtype=np.float64)
    distances = np.asarray(
        [dist_lookup.get((int(o), int(d)), 1.0) for o, d in zip(pred_df["origin"].values, pred_df["destination"].values)],
        dtype=np.float64,
    )
    structural_log = (
        coef[0]
        + coef[1] * np.log1p(origin_vals)
        + coef[2] * np.log1p(dest_vals)
        - coef[3] * np.log(np.maximum(distances, 1.0))
    )
    structural = np.expm1(structural_log)
    time_multiplier = compute_time_multiplier(train_df, pred_df)
    return np.clip(structural * time_multiplier, a_min=0.0, a_max=None).astype(np.float32)


def _radiation_pair_scores(
    zone_ids: list[int],
    centroids: dict[int, tuple[float, float]],
    origin_totals: dict[int, float],
    dest_totals: dict[int, float],
    default_total: float,
) -> dict[tuple[int, int], float]:
    zone_arr = np.asarray(zone_ids, dtype=np.int32)
    coords = _coords_for_zones(zone_arr, centroids)
    dest_vec = np.asarray([dest_totals.get(int(z), default_total) for z in zone_arr], dtype=np.float64)
    origin_vec = np.asarray([origin_totals.get(int(z), default_total) for z in zone_arr], dtype=np.float64)

    scores: dict[tuple[int, int], float] = {}
    for i, origin in enumerate(zone_arr):
        dists = np.sqrt(((coords[:, 0] - coords[i, 0]) ** 2) + ((coords[:, 1] - coords[i, 1]) ** 2))
        order = np.argsort(dists)
        cum = 0.0
        for idx in order:
            dest = int(zone_arr[idx])
            if dest == int(origin):
                continue
            o = float(origin_vec[i])
            d = float(dest_vec[idx])
            s_ij = float(cum)
            denom = max((o + s_ij) * (o + d + s_ij), 1e-6)
            scores[(int(origin), int(dest))] = float((o * d) / denom)
            cum += d
    return scores


def predict_radiation(train_df: pd.DataFrame, pred_df: pd.DataFrame, centroids: dict[int, tuple[float, float]]) -> np.ndarray:
    pair_df = _fit_time_adjusted_pair_frame(train_df)
    origin_totals, dest_totals, global_mean = _zone_marginals(train_df)
    zone_ids = sorted(
        set(train_df["origin"].unique())
        .union(set(train_df["destination"].unique()))
        .union(set(pred_df["origin"].unique()))
        .union(set(pred_df["destination"].unique()))
    )
    scores = _radiation_pair_scores(zone_ids, centroids, origin_totals, dest_totals, global_mean)

    fit_scores = np.asarray([scores.get((int(o), int(d)), 0.0) for o, d in zip(pair_df["origin"].values, pair_df["destination"].values)], dtype=np.float64)
    y = pair_df["adj_trip_count"].to_numpy(dtype=np.float64)
    alpha = float(np.dot(y, fit_scores) / max(np.dot(fit_scores, fit_scores), 1e-6))

    pred_scores = np.asarray([scores.get((int(o), int(d)), 0.0) for o, d in zip(pred_df["origin"].values, pred_df["destination"].values)], dtype=np.float64)
    time_multiplier = compute_time_multiplier(train_df, pred_df)
    return np.clip(alpha * pred_scores * time_multiplier, a_min=0.0, a_max=None).astype(np.float32)


def _community_map_from_file(path: Path) -> dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Community membership file not found: {path}")
    df = pd.read_parquet(path)
    required = {"node", "community_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Community file missing columns: {sorted(missing)}")
    return {int(row.node): int(row.community_id) for row in df.itertuples(index=False)}


def _centrality_map_from_file(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Centrality file not found: {path}")
    df = pd.read_parquet(path)
    node_col = "node" if "node" in df.columns else "node_id" if "node_id" in df.columns else None
    if node_col is None:
        raise ValueError("Centrality file must contain a 'node' or 'node_id' column")
    out: dict[int, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        node = int(row_dict.pop(node_col))
        out[node] = {k: float(v) for k, v in row_dict.items() if pd.notna(v)}
    return out


def predict_community_baseline(train_df: pd.DataFrame, pred_df: pd.DataFrame, community_map: dict[int, int] | None) -> np.ndarray:
    if not community_map:
        return predict_od_marginal(train_df, pred_df)

    pair_df = _fit_time_adjusted_pair_frame(train_df)
    pair_df["origin_comm"] = pair_df["origin"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pair_df["dest_comm"] = pair_df["destination"].map(lambda z: community_map.get(int(z), -1)).astype(int)

    comm_lookup = (
        pair_df.groupby(["origin_comm", "dest_comm"], as_index=False)["adj_trip_count"].mean().rename(columns={"adj_trip_count": "value"})
    )
    od_lookup = pair_df.rename(columns={"adj_trip_count": "od_value"})[["origin", "destination", "od_value"]]
    global_mean = float(pair_df["adj_trip_count"].mean()) if len(pair_df) else 0.0

    pred = pred_df.copy()
    pred["origin_comm"] = pred["origin"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pred["dest_comm"] = pred["destination"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pred = pred.merge(comm_lookup, on=["origin_comm", "dest_comm"], how="left")
    pred = pred.merge(od_lookup, on=["origin", "destination"], how="left")
    structural = pred["value"].fillna(pred["od_value"]).fillna(global_mean).to_numpy(dtype=np.float64)
    time_multiplier = compute_time_multiplier(train_df, pred_df)
    return np.clip(structural * time_multiplier, a_min=0.0, a_max=None).astype(np.float32)


def predict_centrality_baseline(train_df: pd.DataFrame, pred_df: pd.DataFrame, centrality_map: dict[int, dict[str, float]] | None) -> np.ndarray:
    if not centrality_map:
        return predict_od_marginal(train_df, pred_df)

    pair_df = _fit_time_adjusted_pair_frame(train_df)

    origin_score = []
    dest_score = []
    for row in pair_df.itertuples(index=False):
        o = centrality_map.get(int(row.origin), {})
        d = centrality_map.get(int(row.destination), {})
        origin_score.append(float(o.get("out_degree_centrality", o.get("pagerank", 0.0))))
        dest_score.append(float(d.get("pagerank", d.get("in_degree_centrality", 0.0))))

    fit_df = pair_df.copy()
    fit_df["origin_score"] = np.maximum(np.asarray(origin_score, dtype=np.float64), 1e-6)
    fit_df["dest_score"] = np.maximum(np.asarray(dest_score, dtype=np.float64), 1e-6)

    y = np.log1p(fit_df["adj_trip_count"].to_numpy(dtype=np.float64))
    X = np.column_stack(
        [
            np.ones(len(fit_df), dtype=np.float64),
            np.log1p(fit_df["origin_score"].to_numpy(dtype=np.float64)),
            np.log1p(fit_df["dest_score"].to_numpy(dtype=np.float64)),
        ]
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    pred_origin = []
    pred_dest = []
    for row in pred_df.itertuples(index=False):
        o = centrality_map.get(int(row.origin), {})
        d = centrality_map.get(int(row.destination), {})
        pred_origin.append(float(o.get("out_degree_centrality", o.get("pagerank", 0.0))))
        pred_dest.append(float(d.get("pagerank", d.get("in_degree_centrality", 0.0))))
    pred_origin = np.maximum(np.asarray(pred_origin, dtype=np.float64), 1e-6)
    pred_dest = np.maximum(np.asarray(pred_dest, dtype=np.float64), 1e-6)

    structural_log = coef[0] + coef[1] * np.log1p(pred_origin) + coef[2] * np.log1p(pred_dest)
    structural = np.expm1(structural_log)
    time_multiplier = compute_time_multiplier(train_df, pred_df)
    return np.clip(structural * time_multiplier, a_min=0.0, a_max=None).astype(np.float32)


def predict_historical_mean(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    # Primary key: OD + hour-of-day + day-of-week
    key_main = ["origin", "destination", "hour", "dayofweek"]
    main_mean = train_df.groupby(key_main, as_index=False)["trip_count"].mean()
    main_mean = main_mean.rename(columns={"trip_count": "pred_main"})

    # Fallback 1: OD + hour
    key_od_hour = ["origin", "destination", "hour"]
    od_hour_mean = train_df.groupby(key_od_hour, as_index=False)["trip_count"].mean()
    od_hour_mean = od_hour_mean.rename(columns={"trip_count": "pred_od_hour"})

    # Fallback 2: OD only
    key_od = ["origin", "destination"]
    od_mean = train_df.groupby(key_od, as_index=False)["trip_count"].mean()
    od_mean = od_mean.rename(columns={"trip_count": "pred_od"})

    global_mean = float(train_df["trip_count"].mean())

    out = pred_df[key_main].copy()
    out = out.merge(main_mean, on=key_main, how="left")
    out = out.merge(od_hour_mean, on=key_od_hour, how="left")
    out = out.merge(od_mean, on=key_od, how="left")

    pred = out["pred_main"].fillna(out["pred_od_hour"]).fillna(out["pred_od"]).fillna(global_mean)
    return pred.values.astype(np.float32)


def maybe_subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values("pickup_hour").reset_index(drop=True)


def run_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    hidden_dim: int,
    device: str,
) -> tuple[np.ndarray, dict, ODMLP]:
    n_origins = int(max(train_df["origin"].max(), val_df["origin"].max(), test_df["origin"].max()))
    n_destinations = int(max(train_df["destination"].max(), val_df["destination"].max(), test_df["destination"].max()))

    model = ODMLP(n_origins=n_origins, n_destinations=n_destinations, emb_dim=emb_dim, hidden_dim=hidden_dim)
    model.to(device)

    train_ds = ODDataset(train_df)
    val_ds = ODDataset(val_df)
    test_ds = ODDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="MLP epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        batch_bar = tqdm(train_loader, desc=f"MLP train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in batch_bar:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            target_log = batch["target_log"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_log = model(origin, destination, cont)
            loss = loss_fn(pred_log, target_log)
            loss.backward()
            optimizer.step()

            bs = origin.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_n += bs
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                origin = batch["origin"].to(device)
                destination = batch["destination"].to(device)
                cont = batch["cont"].to(device)
                target_log = batch["target_log"].to(device)
                pred_log = model(origin, destination, cont)
                loss = loss_fn(pred_log, target_log)
                bs = origin.shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        val_loss = val_loss_sum / max(1, val_n)
        history.append({"epoch": epoch, "train_mse_log": train_loss, "val_mse_log": val_loss})
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        print(f"Epoch {epoch}/{epochs} | train_mse_log={train_loss:.6f} | val_mse_log={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            pred_log = model(origin, destination, cont)
            pred = torch.expm1(pred_log).clamp(min=0.0)
            preds.append(pred.detach().cpu().numpy())

    pred_test = np.concatenate(preds).astype(np.float32)
    train_info = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "embedding_dim": int(emb_dim),
        "hidden_dim": int(hidden_dim),
        "best_val_mse_log": float(best_val),
        "history": history,
    }
    return pred_test, train_info, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Week-1 baseline models (Historical Mean + MLP)")
    parser.add_argument(
        "--input",
        default="data/processed/od/hourly_od_2023-01_local.parquet",
        help="Input OD parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/baselines",
        help="Output directory for metrics/predictions",
    )
    parser.add_argument(
        "--prefix",
        default="baseline_2023-01",
        help="Output file prefix",
    )
    parser.add_argument(
        "--train-end",
        default="2023-01-23 23:00:00",
        help="Train split end timestamp (inclusive)",
    )
    parser.add_argument(
        "--val-end",
        default="2023-01-27 23:00:00",
        help="Validation split end timestamp (inclusive)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="MLP training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="MLP batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden layer size")
    parser.add_argument(
        "--network-dir",
        default=None,
        help="Optional processed network directory used to auto-discover community/centrality artifacts",
    )
    parser.add_argument(
        "--zone-shapefile",
        default=None,
        help="Optional taxi zone shapefile used for centroid-based baselines (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--community-file",
        default=None,
        help="Optional community membership parquet (node, community_id)",
    )
    parser.add_argument(
        "--centrality-file",
        default=None,
        help="Optional centrality parquet used for the centrality-aware baseline",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows for faster iteration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "models")

    df = load_od(Path(args.input))
    df = add_time_features(df)

    train_df_full, val_df, test_df = temporal_split(df, train_end=args.train_end, val_end=args.val_end)
    train_df_mlp = maybe_subsample(train_df_full, max_rows=args.max_train_rows, seed=args.seed)

    network_dir = _infer_network_dir(Path(args.input), args.network_dir)
    community_file = Path(args.community_file) if args.community_file else _first_matching_file(network_dir / "community", "*_membership.parquet")
    centrality_file = Path(args.centrality_file) if args.centrality_file else _first_matching_file(network_dir / "centrality", "*_centrality.parquet")
    zone_shapefile = Path(args.zone_shapefile) if args.zone_shapefile else _discover_zone_shapefile(Path("data/raw/misc"))
    centroids = load_zone_centroids(str(zone_shapefile))

    community_map = _community_map_from_file(community_file) if community_file is not None else None
    centrality_map = _centrality_map_from_file(centrality_file) if centrality_file is not None else None

    split_summary = {
        "train_rows_structural": int(len(train_df_full)),
        "train_rows_mlp": int(len(train_df_mlp)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_time_min": str(train_df_full["pickup_hour"].min()),
        "train_time_max": str(train_df_full["pickup_hour"].max()),
        "val_time_min": str(val_df["pickup_hour"].min()),
        "val_time_max": str(val_df["pickup_hour"].max()),
        "test_time_min": str(test_df["pickup_hour"].min()),
        "test_time_max": str(test_df["pickup_hour"].max()),
        "network_dir": str(network_dir),
        "zone_shapefile": str(zone_shapefile),
        "community_file": str(community_file) if community_file is not None else None,
        "centrality_file": str(centrality_file) if centrality_file is not None else None,
    }

    y_test = test_df["trip_count"].values.astype(np.float32)

    baseline_predictions: dict[str, np.ndarray] = {}
    baseline_metrics: dict[str, dict] = {}

    traditional_specs = [
        ("historical_mean", lambda: predict_historical_mean(train_df_full, test_df)),
        ("od_marginal", lambda: predict_od_marginal(train_df_full, test_df)),
        ("gravity", lambda: predict_gravity(train_df_full, test_df, centroids)),
        ("radiation", lambda: predict_radiation(train_df_full, test_df, centroids)),
        ("community", lambda: predict_community_baseline(train_df_full, test_df, community_map)),
        ("centrality", lambda: predict_centrality_baseline(train_df_full, test_df, centrality_map)),
    ]
    traditional_bar = tqdm(traditional_specs, desc="Traditional baselines", unit="model")
    for model_name, fn in traditional_bar:
        baseline_predictions[model_name] = fn()
        traditional_bar.set_postfix(model=model_name)

    # MLP baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_mlp, mlp_train_info, mlp_model = run_mlp(
        train_df=train_df_mlp,
        val_df=val_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    baseline_predictions["mlp"] = pred_mlp

    for model_name, pred in baseline_predictions.items():
        baseline_metrics[model_name] = compute_metrics(y_test, pred)

    # Save predictions on test split
    pred_df = test_df[["pickup_hour", "origin", "destination", "trip_count"]].copy()
    pred_df = pred_df.rename(columns={"trip_count": "actual_trip_count"})
    for model_name, pred in baseline_predictions.items():
        pred_df[f"pred_{model_name}"] = pred.astype(np.float32)

    pred_path = output_dir / f"{args.prefix}_test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    # Save metrics and training metadata
    payloads = {
        "historical_mean": {
            "model": "historical_mean",
            "split": "test",
            "metrics": baseline_metrics["historical_mean"],
            "notes": "Keys: origin,destination,hour,dayofweek with hierarchical fallback",
        },
        "od_marginal": {
            "model": "od_marginal",
            "split": "test",
            "metrics": baseline_metrics["od_marginal"],
            "notes": "Origin and destination marginals with global scaling",
        },
        "gravity": {
            "model": "gravity",
            "split": "test",
            "metrics": baseline_metrics["gravity"],
            "notes": "Gravity-style OD model fit with zone centroids and time adjustment",
        },
        "radiation": {
            "model": "radiation",
            "split": "test",
            "metrics": baseline_metrics["radiation"],
            "notes": "Radiation-style OD model fit with cumulative opportunities and time adjustment",
        },
        "community": {
            "model": "community",
            "split": "test",
            "metrics": baseline_metrics["community"],
            "notes": "Community-pair mean baseline derived from network communities",
        },
        "centrality": {
            "model": "centrality",
            "split": "test",
            "metrics": baseline_metrics["centrality"],
            "notes": "Centrality-aware baseline using PageRank/out-degree features",
        },
        "mlp": {
            "model": "mlp_torch",
            "split": "test",
            "metrics": baseline_metrics["mlp"],
            "train": mlp_train_info,
            "device": device,
        },
    }

    comparison = {
        "split": "test",
        "results": baseline_metrics,
        "better_model_by_rmse": min(baseline_metrics, key=lambda k: baseline_metrics[k]["rmse"]),
        "better_model_by_mae": min(baseline_metrics, key=lambda k: baseline_metrics[k]["mae"]),
    }

    split_path = output_dir / f"{args.prefix}_split_summary.json"
    hist_path = output_dir / f"{args.prefix}_historical_metrics.json"
    od_path = output_dir / f"{args.prefix}_od_marginal_metrics.json"
    gravity_path = output_dir / f"{args.prefix}_gravity_metrics.json"
    radiation_path = output_dir / f"{args.prefix}_radiation_metrics.json"
    community_path = output_dir / f"{args.prefix}_community_metrics.json"
    centrality_path = output_dir / f"{args.prefix}_centrality_metrics.json"
    mlp_path = output_dir / f"{args.prefix}_mlp_metrics.json"
    comp_path = output_dir / f"{args.prefix}_comparison.json"
    comp_csv_path = output_dir / f"{args.prefix}_comparison.csv"
    model_path = output_dir / "models" / f"{args.prefix}_mlp.pt"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    for path, key in [
        (hist_path, "historical_mean"),
        (od_path, "od_marginal"),
        (gravity_path, "gravity"),
        (radiation_path, "radiation"),
        (community_path, "community"),
        (centrality_path, "centrality"),
        (mlp_path, "mlp"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payloads[key], f, indent=2)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    comparison_rows = [{"model": name, **metrics} for name, metrics in baseline_metrics.items()]
    pd.DataFrame(comparison_rows).to_csv(comp_csv_path, index=False)

    torch.save(mlp_model.state_dict(), model_path)

    print(f"Wrote: {split_path}")
    print(f"Wrote: {hist_path}")
    print(f"Wrote: {od_path}")
    print(f"Wrote: {gravity_path}")
    print(f"Wrote: {radiation_path}")
    print(f"Wrote: {community_path}")
    print(f"Wrote: {centrality_path}")
    print(f"Wrote: {mlp_path}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {comp_csv_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {model_path}")
    for name, metrics in baseline_metrics.items():
        print(f"Test MAE/RMSE | {name}: {metrics['mae']:.4f}/{metrics['rmse']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
