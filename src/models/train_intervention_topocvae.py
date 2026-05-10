#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


INTERVENTIONS = {
    0: "none",
    1: "edge_removed",
    2: "hub_removed",
    3: "node_added",
}


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
        raise ValueError("Temporal split produced an empty set. Check --train-end and --val-end.")
    return train_df, val_df, test_df


def maybe_subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values("pickup_hour").reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _infer_network_dir(input_path: Path, network_dir: str | None) -> Path:
    if network_dir is not None:
        return Path(network_dir)
    return input_path.resolve().parent.parent / "network"


def _first_matching_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _community_map_from_file(path: Path) -> dict[int, int]:
    df = pd.read_parquet(path)
    required = {"node", "community_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Community file missing columns: {sorted(missing)}")
    return {int(row.node): int(row.community_id) for row in df.itertuples(index=False)}


def _centrality_map_from_file(path: Path) -> dict[int, dict[str, float]]:
    df = pd.read_parquet(path)
    node_col = "node" if "node" in df.columns else "node_id" if "node_id" in df.columns else None
    if node_col is None:
        raise ValueError("Centrality file must contain node or node_id column")
    out: dict[int, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        node = int(row_dict.pop(node_col))
        out[node] = {k: float(v) for k, v in row_dict.items() if pd.notna(v)}
    return out


def compute_time_multiplier(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train_df["trip_count"].mean()) if len(train_df) else 1.0
    main = train_df.groupby(["hour", "dayofweek"], as_index=False)["trip_count"].mean().rename(columns={"trip_count": "main"})
    hour = train_df.groupby(["hour"], as_index=False)["trip_count"].mean().rename(columns={"trip_count": "hour_value"})
    dow = train_df.groupby(["dayofweek"], as_index=False)["trip_count"].mean().rename(columns={"trip_count": "dow_value"})

    out = pred_df[["hour", "dayofweek"]].copy()
    out = out.merge(main, on=["hour", "dayofweek"], how="left")
    out = out.merge(hour, on=["hour"], how="left")
    out = out.merge(dow, on=["dayofweek"], how="left")
    vals = out["main"].fillna(out["hour_value"]).fillna(out["dow_value"]).fillna(global_mean)
    return (vals / max(global_mean, 1e-6)).astype(np.float32).to_numpy()


def _fit_time_adjusted_pair_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    tm = compute_time_multiplier(train_df, train_df)
    tmp = train_df[["origin", "destination", "trip_count"]].copy()
    tmp["adj_trip_count"] = tmp["trip_count"] / np.maximum(tm, 1e-6)
    return (
        tmp.groupby(["origin", "destination"], as_index=False)
        .agg(adj_trip_count=("adj_trip_count", "mean"), trip_count=("trip_count", "mean"))
        .reset_index(drop=True)
    )


def predict_community_baseline(train_df: pd.DataFrame, pred_df: pd.DataFrame, community_map: dict[int, int] | None) -> np.ndarray:
    if not community_map:
        global_mean = float(train_df["trip_count"].mean())
        return np.full(len(pred_df), global_mean, dtype=np.float32)

    pair_df = _fit_time_adjusted_pair_frame(train_df)
    pair_df["origin_comm"] = pair_df["origin"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pair_df["dest_comm"] = pair_df["destination"].map(lambda z: community_map.get(int(z), -1)).astype(int)

    comm_lookup = pair_df.groupby(["origin_comm", "dest_comm"], as_index=False)["adj_trip_count"].mean().rename(columns={"adj_trip_count": "comm_value"})
    od_lookup = pair_df[["origin", "destination", "adj_trip_count"]].rename(columns={"adj_trip_count": "od_value"})
    global_mean = float(pair_df["adj_trip_count"].mean()) if len(pair_df) else 0.0

    pred = pred_df.copy()
    pred["origin_comm"] = pred["origin"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pred["dest_comm"] = pred["destination"].map(lambda z: community_map.get(int(z), -1)).astype(int)
    pred = pred.merge(comm_lookup, on=["origin_comm", "dest_comm"], how="left")
    pred = pred.merge(od_lookup, on=["origin", "destination"], how="left")
    structural = pred["comm_value"].fillna(pred["od_value"]).fillna(global_mean).to_numpy(dtype=np.float64)
    tm = compute_time_multiplier(train_df, pred_df)
    return np.clip(structural * tm, a_min=0.0, a_max=None).astype(np.float32)


def predict_centrality_baseline(train_df: pd.DataFrame, pred_df: pd.DataFrame, centrality_map: dict[int, dict[str, float]] | None) -> np.ndarray:
    if not centrality_map:
        global_mean = float(train_df["trip_count"].mean())
        return np.full(len(pred_df), global_mean, dtype=np.float32)

    pair_df = _fit_time_adjusted_pair_frame(train_df)

    o_score = []
    d_score = []
    for row in pair_df.itertuples(index=False):
        o = centrality_map.get(int(row.origin), {})
        d = centrality_map.get(int(row.destination), {})
        o_score.append(float(o.get("out_degree_centrality", o.get("pagerank", 0.0))))
        d_score.append(float(d.get("pagerank", d.get("in_degree_centrality", 0.0))))

    fit = pair_df.copy()
    fit["o"] = np.maximum(np.asarray(o_score, dtype=np.float64), 1e-6)
    fit["d"] = np.maximum(np.asarray(d_score, dtype=np.float64), 1e-6)
    X = np.column_stack([np.ones(len(fit)), np.log1p(fit["o"].to_numpy()), np.log1p(fit["d"].to_numpy())])
    y = np.log1p(fit["adj_trip_count"].to_numpy())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    po = []
    pd_ = []
    for row in pred_df.itertuples(index=False):
        o = centrality_map.get(int(row.origin), {})
        d = centrality_map.get(int(row.destination), {})
        po.append(float(o.get("out_degree_centrality", o.get("pagerank", 0.0))))
        pd_.append(float(d.get("pagerank", d.get("in_degree_centrality", 0.0))))
    po = np.maximum(np.asarray(po, dtype=np.float64), 1e-6)
    pd_ = np.maximum(np.asarray(pd_, dtype=np.float64), 1e-6)

    structural_log = coef[0] + coef[1] * np.log1p(po) + coef[2] * np.log1p(pd_)
    structural = np.expm1(structural_log)
    tm = compute_time_multiplier(train_df, pred_df)
    return np.clip(structural * tm, a_min=0.0, a_max=None).astype(np.float32)


def load_graph_edges(edge_file: Path, max_node_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_file is None or not edge_file.exists():
        return (
            torch.zeros((1,), dtype=torch.long),
            torch.zeros((1,), dtype=torch.long),
            torch.ones((1,), dtype=torch.float32),
        )
    df = pd.read_parquet(edge_file)
    required = {"origin", "destination"}
    missing = required - set(df.columns)
    if missing:
        return (
            torch.zeros((1,), dtype=torch.long),
            torch.zeros((1,), dtype=torch.long),
            torch.ones((1,), dtype=torch.float32),
        )

    weight_col = "trip_count" if "trip_count" in df.columns else None
    src = np.clip(df["origin"].to_numpy(dtype=np.int64), 0, max_node_id)
    dst = np.clip(df["destination"].to_numpy(dtype=np.int64), 0, max_node_id)
    if weight_col is not None:
        w = df[weight_col].to_numpy(dtype=np.float32)
        w = w / max(float(np.mean(w)), 1e-6)
    else:
        w = np.ones(len(df), dtype=np.float32)

    return torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(w)


def add_topology_features(
    df: pd.DataFrame,
    community_map: dict[int, int] | None,
    centrality_map: dict[int, dict[str, float]] | None,
) -> pd.DataFrame:
    out = df.copy()

    def cval(node: int, k1: str, k2: str) -> float:
        if not centrality_map:
            return 0.0
        row = centrality_map.get(int(node), {})
        return float(row.get(k1, row.get(k2, 0.0)))

    out["origin_pagerank"] = out["origin"].map(lambda z: cval(int(z), "pagerank", "out_degree_centrality")).astype("float32")
    out["dest_pagerank"] = out["destination"].map(lambda z: cval(int(z), "pagerank", "in_degree_centrality")).astype("float32")
    out["origin_outdeg"] = out["origin"].map(lambda z: cval(int(z), "out_degree_centrality", "pagerank")).astype("float32")
    out["dest_indeg"] = out["destination"].map(lambda z: cval(int(z), "in_degree_centrality", "pagerank")).astype("float32")

    if community_map:
        out["origin_comm"] = out["origin"].map(lambda z: community_map.get(int(z), -1)).astype("int32")
        out["dest_comm"] = out["destination"].map(lambda z: community_map.get(int(z), -1)).astype("int32")
        out["same_community"] = (out["origin_comm"] == out["dest_comm"]).astype("float32")
    else:
        out["same_community"] = 0.0

    return out


def prepare_intervention_context(
    train_df: pd.DataFrame,
    community_map: dict[int, int] | None,
    centrality_map: dict[int, dict[str, float]] | None,
    hub_percentile: float,
    critical_edge_percentile: float,
    growth_nodes_count: int,
) -> tuple[set[int], set[tuple[int, int]], set[int], int]:
    if centrality_map:
        scores = []
        for node, vals in centrality_map.items():
            score = float(vals.get("pagerank", vals.get("out_degree_centrality", 0.0)))
            scores.append((int(node), score))
        if scores:
            vals = np.asarray([s for _, s in scores], dtype=np.float64)
            thr = float(np.quantile(vals, hub_percentile))
            hub_nodes = {n for n, s in scores if s >= thr}
            growth_nodes = {n for n, s in sorted(scores, key=lambda x: x[1])[: max(1, growth_nodes_count)]}
        else:
            hub_nodes = set()
            growth_nodes = set()
    else:
        strength = train_df.groupby("origin", as_index=False)["trip_count"].sum().rename(columns={"trip_count": "flow"})
        vals = strength["flow"].to_numpy(dtype=np.float64)
        thr = float(np.quantile(vals, hub_percentile)) if len(vals) else float("inf")
        hub_nodes = {int(n) for n, f in strength[["origin", "flow"]].itertuples(index=False) if f >= thr}
        growth_nodes = {int(n) for n in strength.sort_values("flow").head(max(1, growth_nodes_count))["origin"].tolist()}

    pair_flow = train_df.groupby(["origin", "destination"], as_index=False)["trip_count"].mean()
    if len(pair_flow):
        edge_thr = float(np.quantile(pair_flow["trip_count"].to_numpy(dtype=np.float64), critical_edge_percentile))
        critical_edges = {
            (int(o), int(d))
            for o, d, flow in pair_flow[["origin", "destination", "trip_count"]].itertuples(index=False)
            if float(flow) >= edge_thr
        }
    else:
        critical_edges = set()

    if community_map:
        comm_ids = [int(v) for v in community_map.values()]
        dominant_comm = int(pd.Series(comm_ids).value_counts().index[0]) if len(comm_ids) else -1
    else:
        dominant_comm = -1

    return hub_nodes, critical_edges, growth_nodes, dominant_comm


def add_intervention_flags(
    df: pd.DataFrame,
    hub_nodes: set[int],
    critical_edges: set[tuple[int, int]],
    growth_nodes: set[int],
    dominant_comm: int,
) -> pd.DataFrame:
    out = df.copy()
    out["is_hub_endpoint"] = (
        out["origin"].isin(hub_nodes) | out["destination"].isin(hub_nodes)
    ).astype("float32")

    edge_keys = list(zip(out["origin"].astype(int).tolist(), out["destination"].astype(int).tolist()))
    out["is_critical_edge"] = np.asarray([1.0 if k in critical_edges else 0.0 for k in edge_keys], dtype=np.float32)

    out["is_growth_endpoint"] = (
        out["origin"].isin(growth_nodes) | out["destination"].isin(growth_nodes)
    ).astype("float32")

    if "origin_comm" in out.columns and "dest_comm" in out.columns and dominant_comm >= 0:
        out["touches_dominant_comm"] = (
            (out["origin_comm"] == dominant_comm) | (out["dest_comm"] == dominant_comm)
        ).astype("float32")
    else:
        out["touches_dominant_comm"] = 0.0

    return out


def intervention_multiplier_from_flags(intervention_id: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    # flags: [is_critical_edge, is_hub_endpoint, is_growth_endpoint, same_community, touches_dominant_comm]
    is_critical = flags[:, 0]
    is_hub = flags[:, 1]
    is_growth = flags[:, 2]
    same_comm = flags[:, 3]
    touches_dom = flags[:, 4]

    mult = torch.ones_like(is_critical)

    edge_case = torch.where(is_critical > 0.5, 0.35, 0.95)
    hub_case = torch.where(is_hub > 0.5, 0.45, 0.90)
    node_case = torch.where(
        is_growth > 0.5,
        torch.where(same_comm > 0.5, 1.25, 1.10),
        torch.where(touches_dom > 0.5, 0.90, 1.00),
    )

    mult = torch.where(intervention_id == 1, edge_case, mult)
    mult = torch.where(intervention_id == 2, hub_case, mult)
    mult = torch.where(intervention_id == 3, node_case, mult)
    return mult.clamp(min=0.05, max=2.0)


def apply_intervention_multiplier_np(intervention_name: str, base_pred: np.ndarray, flags_df: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(base_pred, dtype=np.float64).copy()
    is_critical = flags_df["is_critical_edge"].to_numpy(dtype=np.float64)
    is_hub = flags_df["is_hub_endpoint"].to_numpy(dtype=np.float64)
    is_growth = flags_df["is_growth_endpoint"].to_numpy(dtype=np.float64)
    same_comm = flags_df["same_community"].to_numpy(dtype=np.float64)
    touches_dom = flags_df["touches_dominant_comm"].to_numpy(dtype=np.float64)

    if intervention_name == "edge_removed":
        mult = np.where(is_critical > 0.5, 0.35, 0.95)
    elif intervention_name == "hub_removed":
        mult = np.where(is_hub > 0.5, 0.45, 0.90)
    elif intervention_name == "node_added":
        mult = np.where(is_growth > 0.5, np.where(same_comm > 0.5, 1.25, 1.10), np.where(touches_dom > 0.5, 0.90, 1.00))
    else:
        mult = np.ones_like(pred)

    return np.clip(pred * mult, a_min=0.0, a_max=None).astype(np.float32)


class ODDataset(Dataset):
    def __init__(self, df: pd.DataFrame, topo_target_log: np.ndarray) -> None:
        self.origin = torch.tensor(df["origin"].values, dtype=torch.long)
        self.destination = torch.tensor(df["destination"].values, dtype=torch.long)
        cont_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        topo_cols = ["origin_pagerank", "dest_pagerank", "origin_outdeg", "dest_indeg", "same_community"]
        flag_cols = ["is_critical_edge", "is_hub_endpoint", "is_growth_endpoint", "same_community", "touches_dominant_comm"]

        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.topo = torch.tensor(df[topo_cols].values, dtype=torch.float32)
        self.flags = torch.tensor(df[flag_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df["trip_count"].values, dtype=torch.float32)
        self.target_log = torch.tensor(np.log1p(df["trip_count"].values), dtype=torch.float32)
        self.topo_target_log = torch.tensor(topo_target_log, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.origin)

    def __getitem__(self, idx: int):
        return {
            "origin": self.origin[idx],
            "destination": self.destination[idx],
            "cont": self.cont[idx],
            "topo": self.topo[idx],
            "flags": self.flags[idx],
            "target": self.target[idx],
            "target_log": self.target_log[idx],
            "topo_target_log": self.topo_target_log[idx],
        }


class InterventionTopologyCVAE(nn.Module):
    def __init__(
        self,
        n_origins: int,
        n_destinations: int,
        n_interventions: int,
        emb_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.origin_emb = nn.Embedding(n_origins + 1, emb_dim)
        self.destination_emb = nn.Embedding(n_destinations + 1, emb_dim)
        self.intervention_emb = nn.Embedding(n_interventions, emb_dim)
        cond_dim = emb_dim * 3 + 10

        self.encoder = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def condition(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        intervention_id: torch.Tensor,
        cont: torch.Tensor,
        topo: torch.Tensor,
    ) -> torch.Tensor:
        o = self.origin_emb(origin)
        d = self.destination_emb(destination)
        i = self.intervention_emb(intervention_id)
        return torch.cat([o, d, i, cont, topo], dim=1)

    def encode(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        intervention_id: torch.Tensor,
        cont: torch.Tensor,
        topo: torch.Tensor,
        target_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.condition(origin, destination, intervention_id, cont, topo)
        h = self.encoder(torch.cat([cond, target_log.unsqueeze(1)], dim=1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparam(mu, logvar)
        return cond, mu, logvar, z

    def decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([cond, z], dim=1)).squeeze(1)

    def forward(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        intervention_id: torch.Tensor,
        cont: torch.Tensor,
        topo: torch.Tensor,
        target_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond, mu, logvar, z = self.encode(origin, destination, intervention_id, cont, topo, target_log)
        pred = self.decode(cond, z)
        return pred, mu, logvar, z

    def decode_with_samples(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        intervention_id: torch.Tensor,
        cont: torch.Tensor,
        topo: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        cond = self.condition(origin, destination, intervention_id, cont, topo)
        preds = []
        for _ in range(max(1, sample_count)):
            z = torch.randn((origin.shape[0], self.latent_dim), device=origin.device)
            pred = self.decode(cond, z)
            preds.append(torch.expm1(pred).clamp(min=0.0))
        return torch.stack(preds, dim=0).mean(dim=0)


def laplacian_smoothness_loss(
    model: InterventionTopologyCVAE,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_w: torch.Tensor,
    edge_sample_size: int,
) -> torch.Tensor:
    if edge_src.numel() <= 1:
        return torch.tensor(0.0, device=model.origin_emb.weight.device)

    n = edge_src.shape[0]
    k = min(edge_sample_size, n)
    idx = torch.randint(0, n, (k,), device=edge_src.device)
    s = edge_src[idx]
    d = edge_dst[idx]
    w = edge_w[idx]

    o_diff = model.origin_emb(s) - model.origin_emb(d)
    t_diff = model.destination_emb(s) - model.destination_emb(d)
    o_term = (w * torch.sum(o_diff * o_diff, dim=1)).mean()
    t_term = (w * torch.sum(t_diff * t_diff, dim=1)).mean()
    return 0.5 * (o_term + t_term)


def run_intervention_topocvae(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    topo_train_log: np.ndarray,
    topo_val_log: np.ndarray,
    topo_test_log: np.ndarray,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_w: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    hidden_dim: int,
    latent_dim: int,
    beta_kl: float,
    lambda_topo: float,
    lambda_lap: float,
    lambda_counterfactual: float,
    edge_sample_size: int,
    sample_count: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict, InterventionTopologyCVAE]:
    n_origins = int(max(train_df["origin"].max(), val_df["origin"].max(), test_df["origin"].max()))
    n_destinations = int(max(train_df["destination"].max(), val_df["destination"].max(), test_df["destination"].max()))

    model = InterventionTopologyCVAE(
        n_origins=n_origins,
        n_destinations=n_destinations,
        n_interventions=len(INTERVENTIONS),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    train_loader = DataLoader(ODDataset(train_df, topo_train_log), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ODDataset(val_df, topo_val_log), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ODDataset(test_df, topo_test_log), batch_size=batch_size, shuffle=False, num_workers=0)

    edge_src = edge_src.to(device)
    edge_dst = edge_dst.to(device)
    edge_w = edge_w.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Intervention Topo-cVAE epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        batch_bar = tqdm(train_loader, desc=f"Intervention Topo-cVAE train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in batch_bar:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            topo = batch["topo"].to(device)
            flags = batch["flags"].to(device)
            y = batch["target"].to(device)
            y_log = batch["target_log"].to(device)
            topo_log = batch["topo_target_log"].to(device)
            bs = origin.shape[0]

            factual_id = torch.zeros((bs,), dtype=torch.long, device=device)
            cf_id = torch.randint(1, len(INTERVENTIONS), (bs,), dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)

            factual_pred_log, mu, logvar, z = model(origin, destination, factual_id, cont, topo, y_log)
            recon = torch.mean((factual_pred_log - y_log) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            topo_consistency = torch.mean((factual_pred_log - topo_log) ** 2)

            cf_cond = model.condition(origin, destination, cf_id, cont, topo)
            cf_pred_log = model.decode(cf_cond, z)
            cf_mult = intervention_multiplier_from_flags(cf_id, flags)
            cf_target = torch.log1p(torch.clamp(y * cf_mult, min=0.0))
            cf_loss = torch.mean((cf_pred_log - cf_target) ** 2)

            lap = laplacian_smoothness_loss(model, edge_src, edge_dst, edge_w, edge_sample_size)

            loss = recon + beta_kl * kl + lambda_topo * topo_consistency + lambda_counterfactual * cf_loss + lambda_lap * lap
            loss.backward()
            opt.step()

            train_loss_sum += float(loss.item()) * bs
            train_n += bs
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                origin = batch["origin"].to(device)
                destination = batch["destination"].to(device)
                cont = batch["cont"].to(device)
                topo = batch["topo"].to(device)
                flags = batch["flags"].to(device)
                y = batch["target"].to(device)
                y_log = batch["target_log"].to(device)
                topo_log = batch["topo_target_log"].to(device)
                bs = origin.shape[0]

                factual_id = torch.zeros((bs,), dtype=torch.long, device=device)
                cf_id = torch.randint(1, len(INTERVENTIONS), (bs,), dtype=torch.long, device=device)

                factual_pred_log, mu, logvar, z = model(origin, destination, factual_id, cont, topo, y_log)
                recon = torch.mean((factual_pred_log - y_log) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                topo_consistency = torch.mean((factual_pred_log - topo_log) ** 2)

                cf_cond = model.condition(origin, destination, cf_id, cont, topo)
                cf_pred_log = model.decode(cf_cond, z)
                cf_mult = intervention_multiplier_from_flags(cf_id, flags)
                cf_target = torch.log1p(torch.clamp(y * cf_mult, min=0.0))
                cf_loss = torch.mean((cf_pred_log - cf_target) ** 2)

                lap = laplacian_smoothness_loss(model, edge_src, edge_dst, edge_w, edge_sample_size)
                vloss = recon + beta_kl * kl + lambda_topo * topo_consistency + lambda_counterfactual * cf_loss + lambda_lap * lap

                val_sum += float(vloss.item()) * bs
                val_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        val_loss = val_sum / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        print(f"[Intervention Topo-cVAE] Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    scenario_preds: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        all_outputs: dict[str, list[np.ndarray]] = {name: [] for name in INTERVENTIONS.values()}
        for batch in test_loader:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            topo = batch["topo"].to(device)

            for int_id, name in INTERVENTIONS.items():
                intervention_id = torch.full((origin.shape[0],), int_id, dtype=torch.long, device=device)
                pred = model.decode_with_samples(
                    origin=origin,
                    destination=destination,
                    intervention_id=intervention_id,
                    cont=cont,
                    topo=topo,
                    sample_count=sample_count,
                )
                all_outputs[name].append(pred.detach().cpu().numpy())

        for name, chunks in all_outputs.items():
            scenario_preds[name] = np.concatenate(chunks).astype(np.float32)

    train_info = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "embedding_dim": int(emb_dim),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "beta_kl": float(beta_kl),
        "lambda_topology": float(lambda_topo),
        "lambda_laplacian": float(lambda_lap),
        "lambda_counterfactual": float(lambda_counterfactual),
        "edge_sample_size": int(edge_sample_size),
        "sample_count": int(sample_count),
        "best_val_loss": float(best_val),
        "history": history,
    }
    return scenario_preds, train_info, model


def scenario_summary_from_predictions(pred_df: pd.DataFrame) -> dict:
    base = pred_df["pred_none"].to_numpy(dtype=np.float64)
    base_total = float(np.sum(base))
    out = {}
    for name in ["edge_removed", "hub_removed", "node_added"]:
        p = pred_df[f"pred_{name}"].to_numpy(dtype=np.float64)
        total = float(np.sum(p))
        retention = float(total / max(base_total, 1e-6))
        redistribution = float(np.mean(np.abs(p - base)))
        out[name] = {
            "total_predicted_flow": total,
            "flow_retention_vs_none": retention,
            "mean_absolute_redistribution": redistribution,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train intervention-aware topology-constrained cVAE for counterfactual OD redistribution"
    )
    parser.add_argument("--input", default="data/processed/od/hourly_od_2023-01_local.parquet", help="Input OD parquet")
    parser.add_argument("--output-dir", default="data/processed/novelty", help="Output directory")
    parser.add_argument("--prefix", default="novelty_intervention_topocvae_2023-01", help="Output file prefix")
    parser.add_argument("--train-end", default="2023-01-23 23:00:00")
    parser.add_argument("--val-end", default="2023-01-27 23:00:00")
    parser.add_argument("--network-dir", default=None, help="Optional processed network dir")
    parser.add_argument("--community-file", default=None, help="Optional explicit community membership file")
    parser.add_argument("--centrality-file", default=None, help="Optional explicit centrality file")
    parser.add_argument("--edge-file", default=None, help="Optional explicit network edge file")

    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta-kl", type=float, default=0.05)
    parser.add_argument("--lambda-topo", type=float, default=0.2)
    parser.add_argument("--lambda-lap", type=float, default=0.01)
    parser.add_argument("--lambda-counterfactual", type=float, default=0.3)
    parser.add_argument("--edge-sample-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=20)

    parser.add_argument("--hub-percentile", type=float, default=0.90)
    parser.add_argument("--critical-edge-percentile", type=float, default=0.95)
    parser.add_argument("--growth-nodes-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "models")

    df = load_od(Path(args.input))
    df = add_time_features(df)
    train_df_full, val_df, test_df = temporal_split(df, train_end=args.train_end, val_end=args.val_end)
    train_df = maybe_subsample(train_df_full, max_rows=args.max_train_rows, seed=args.seed)

    network_dir = _infer_network_dir(Path(args.input), args.network_dir)
    community_file = Path(args.community_file) if args.community_file else _first_matching_file(network_dir / "community", "*_membership.parquet")
    centrality_file = Path(args.centrality_file) if args.centrality_file else _first_matching_file(network_dir / "centrality", "*_centrality.parquet")
    edge_file = Path(args.edge_file) if args.edge_file else _first_matching_file(network_dir, "*_edges.parquet")

    community_map = _community_map_from_file(community_file) if community_file is not None and community_file.exists() else None
    centrality_map = _centrality_map_from_file(centrality_file) if centrality_file is not None and centrality_file.exists() else None

    train_df = add_topology_features(train_df, community_map, centrality_map)
    val_df = add_topology_features(val_df, community_map, centrality_map)
    test_df = add_topology_features(test_df, community_map, centrality_map)

    hub_nodes, critical_edges, growth_nodes, dominant_comm = prepare_intervention_context(
        train_df=train_df,
        community_map=community_map,
        centrality_map=centrality_map,
        hub_percentile=args.hub_percentile,
        critical_edge_percentile=args.critical_edge_percentile,
        growth_nodes_count=args.growth_nodes_count,
    )

    train_df = add_intervention_flags(train_df, hub_nodes, critical_edges, growth_nodes, dominant_comm)
    val_df = add_intervention_flags(val_df, hub_nodes, critical_edges, growth_nodes, dominant_comm)
    test_df = add_intervention_flags(test_df, hub_nodes, critical_edges, growth_nodes, dominant_comm)

    prior_train_comm = predict_community_baseline(train_df, train_df, community_map)
    prior_val_comm = predict_community_baseline(train_df, val_df, community_map)
    prior_test_comm = predict_community_baseline(train_df, test_df, community_map)
    prior_train_cent = predict_centrality_baseline(train_df, train_df, centrality_map)
    prior_val_cent = predict_centrality_baseline(train_df, val_df, centrality_map)
    prior_test_cent = predict_centrality_baseline(train_df, test_df, centrality_map)

    prior_train = 0.5 * (prior_train_comm + prior_train_cent)
    prior_val = 0.5 * (prior_val_comm + prior_val_cent)
    prior_test = 0.5 * (prior_test_comm + prior_test_cent)

    topo_train_log = np.log1p(prior_train).astype(np.float32)
    topo_val_log = np.log1p(prior_val).astype(np.float32)
    topo_test_log = np.log1p(prior_test).astype(np.float32)

    max_node_id = int(
        max(
            train_df["origin"].max(),
            train_df["destination"].max(),
            val_df["origin"].max(),
            val_df["destination"].max(),
            test_df["origin"].max(),
            test_df["destination"].max(),
        )
    )
    edge_src, edge_dst, edge_w = load_graph_edges(edge_file=edge_file if edge_file is not None else Path(""), max_node_id=max_node_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario_preds, train_info, model = run_intervention_topocvae(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        topo_train_log=topo_train_log,
        topo_val_log=topo_val_log,
        topo_test_log=topo_test_log,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_w=edge_w,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta_kl=args.beta_kl,
        lambda_topo=args.lambda_topo,
        lambda_lap=args.lambda_lap,
        lambda_counterfactual=args.lambda_counterfactual,
        edge_sample_size=args.edge_sample_size,
        sample_count=args.sample_count,
        device=device,
    )

    y_test = test_df["trip_count"].to_numpy(dtype=np.float32)
    metrics_intervention_none = compute_metrics(y_test, scenario_preds["none"])
    metrics_prior = compute_metrics(y_test, prior_test)
    comparison = {
        "split": "test",
        "results": {
            "intervention_topocvae_none": metrics_intervention_none,
            "topology_prior": metrics_prior,
        },
        "better_model_by_rmse": "intervention_topocvae_none"
        if metrics_intervention_none["rmse"] <= metrics_prior["rmse"]
        else "topology_prior",
        "better_model_by_mae": "intervention_topocvae_none"
        if metrics_intervention_none["mae"] <= metrics_prior["mae"]
        else "topology_prior",
    }

    pred_df = test_df[["pickup_hour", "origin", "destination", "trip_count"]].copy().rename(columns={"trip_count": "actual_trip_count"})
    pred_df["pred_none"] = scenario_preds["none"]
    pred_df["pred_edge_removed"] = scenario_preds["edge_removed"]
    pred_df["pred_hub_removed"] = scenario_preds["hub_removed"]
    pred_df["pred_node_added"] = scenario_preds["node_added"]
    pred_df["pred_topology_prior"] = prior_test.astype(np.float32)

    prior_counterfactual = {
        "edge_removed": apply_intervention_multiplier_np("edge_removed", prior_test, test_df),
        "hub_removed": apply_intervention_multiplier_np("hub_removed", prior_test, test_df),
        "node_added": apply_intervention_multiplier_np("node_added", prior_test, test_df),
    }

    scenario_report = {
        "model_scenarios": scenario_summary_from_predictions(pred_df),
        "prior_scenarios": {
            name: {
                "total_predicted_flow": float(np.sum(vals)),
                "flow_retention_vs_none": float(np.sum(vals) / max(float(np.sum(prior_test)), 1e-6)),
                "mean_absolute_redistribution": float(np.mean(np.abs(vals - prior_test))),
            }
            for name, vals in prior_counterfactual.items()
        },
    }

    split_summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "network_dir": str(network_dir),
        "community_file": str(community_file) if community_file is not None else None,
        "centrality_file": str(centrality_file) if centrality_file is not None else None,
        "edge_file": str(edge_file) if edge_file is not None else None,
        "device": device,
        "hub_nodes": int(len(hub_nodes)),
        "critical_edges": int(len(critical_edges)),
        "growth_nodes": int(len(growth_nodes)),
        "dominant_community": int(dominant_comm),
    }

    payload = {
        "model": "intervention_topology_constrained_cvae",
        "split": "test",
        "metrics": metrics_intervention_none,
        "train": train_info,
        "notes": "Intervention-aware novelty: topology-constrained cVAE with explicit scenario conditioning and counterfactual consistency loss",
    }

    split_path = output_dir / f"{args.prefix}_split_summary.json"
    model_path_json = output_dir / f"{args.prefix}_metrics.json"
    comp_path = output_dir / f"{args.prefix}_comparison.json"
    comp_csv_path = output_dir / f"{args.prefix}_comparison.csv"
    scenario_path = output_dir / f"{args.prefix}_scenario_summary.json"
    pred_path = output_dir / f"{args.prefix}_test_predictions.parquet"
    model_path = output_dir / "models" / f"{args.prefix}.pt"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    with open(model_path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(scenario_report, f, indent=2)

    pd.DataFrame([{"model": k, **v} for k, v in comparison["results"].items()]).to_csv(comp_csv_path, index=False)
    pred_df.to_parquet(pred_path, index=False)
    torch.save(model.state_dict(), model_path)

    print(f"Wrote: {split_path}")
    print(f"Wrote: {model_path_json}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {comp_csv_path}")
    print(f"Wrote: {scenario_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {model_path}")
    print(
        f"Test MAE/RMSE | intervention_topocvae_none: {metrics_intervention_none['mae']:.4f}/{metrics_intervention_none['rmse']:.4f}"
    )
    print(f"Test MAE/RMSE | topology_prior: {metrics_prior['mae']:.4f}/{metrics_prior['rmse']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
