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


def add_topology_features(df: pd.DataFrame, community_map: dict[int, int] | None, centrality_map: dict[int, dict[str, float]] | None) -> pd.DataFrame:
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


class ODDataset(Dataset):
    def __init__(self, df: pd.DataFrame, topo_target_log: np.ndarray) -> None:
        self.origin = torch.tensor(df["origin"].values, dtype=torch.long)
        self.destination = torch.tensor(df["destination"].values, dtype=torch.long)
        cont_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        topo_cols = ["origin_pagerank", "dest_pagerank", "origin_outdeg", "dest_indeg", "same_community"]
        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.topo = torch.tensor(df[topo_cols].values, dtype=torch.float32)
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
            "target_log": self.target_log[idx],
            "topo_target_log": self.topo_target_log[idx],
        }


class TopologyCVAE(nn.Module):
    def __init__(self, n_origins: int, n_destinations: int, emb_dim: int = 16, hidden_dim: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.origin_emb = nn.Embedding(n_origins + 1, emb_dim)
        self.destination_emb = nn.Embedding(n_destinations + 1, emb_dim)
        cond_dim = emb_dim * 2 + 10  # time(5) + topo(5)

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

    def condition(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        o = self.origin_emb(origin)
        d = self.destination_emb(destination)
        return torch.cat([o, d, cont, topo], dim=1)

    def forward(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor, topo: torch.Tensor, target_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.condition(origin, destination, cont, topo)
        h = self.encoder(torch.cat([cond, target_log.unsqueeze(1)], dim=1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparam(mu, logvar)
        pred = self.decoder(torch.cat([cond, z], dim=1)).squeeze(1)
        return pred, mu, logvar

    def decode_with_samples(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor, topo: torch.Tensor, latent_dim: int, sample_count: int) -> torch.Tensor:
        cond = self.condition(origin, destination, cont, topo)
        preds = []
        for _ in range(max(1, sample_count)):
            z = torch.randn((origin.shape[0], latent_dim), device=origin.device)
            pred = self.decoder(torch.cat([cond, z], dim=1)).squeeze(1)
            preds.append(torch.expm1(pred).clamp(min=0.0))
        return torch.stack(preds, dim=0).mean(dim=0)


def laplacian_smoothness_loss(model: TopologyCVAE, edge_src: torch.Tensor, edge_dst: torch.Tensor, edge_w: torch.Tensor, edge_sample_size: int) -> torch.Tensor:
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


def run_topology_cvae(
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
    edge_sample_size: int,
    sample_count: int,
    device: str,
) -> tuple[np.ndarray, dict, TopologyCVAE]:
    n_origins = int(max(train_df["origin"].max(), val_df["origin"].max(), test_df["origin"].max()))
    n_destinations = int(max(train_df["destination"].max(), val_df["destination"].max(), test_df["destination"].max()))

    model = TopologyCVAE(
        n_origins=n_origins,
        n_destinations=n_destinations,
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

    epoch_bar = tqdm(range(1, epochs + 1), desc="Topo-cVAE epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        batch_bar = tqdm(train_loader, desc=f"Topo-cVAE train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in batch_bar:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            topo = batch["topo"].to(device)
            y_log = batch["target_log"].to(device)
            topo_log = batch["topo_target_log"].to(device)

            opt.zero_grad(set_to_none=True)
            pred_log, mu, logvar = model(origin, destination, cont, topo, y_log)

            recon = torch.mean((pred_log - y_log) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            topo_consistency = torch.mean((pred_log - topo_log) ** 2)
            lap = laplacian_smoothness_loss(model, edge_src, edge_dst, edge_w, edge_sample_size)

            loss = recon + beta_kl * kl + lambda_topo * topo_consistency + lambda_lap * lap
            loss.backward()
            opt.step()

            bs = origin.shape[0]
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
                y_log = batch["target_log"].to(device)
                topo_log = batch["topo_target_log"].to(device)

                pred_log, mu, logvar = model(origin, destination, cont, topo, y_log)
                recon = torch.mean((pred_log - y_log) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                topo_consistency = torch.mean((pred_log - topo_log) ** 2)
                lap = laplacian_smoothness_loss(model, edge_src, edge_dst, edge_w, edge_sample_size)
                vloss = recon + beta_kl * kl + lambda_topo * topo_consistency + lambda_lap * lap

                bs = origin.shape[0]
                val_sum += float(vloss.item()) * bs
                val_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        val_loss = val_sum / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        print(f"[Topo-cVAE] Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
            topo = batch["topo"].to(device)
            pred = model.decode_with_samples(origin, destination, cont, topo, latent_dim=latent_dim, sample_count=sample_count)
            preds.append(pred.detach().cpu().numpy())

    pred_test = np.concatenate(preds).astype(np.float32)
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
        "edge_sample_size": int(edge_sample_size),
        "sample_count": int(sample_count),
        "best_val_loss": float(best_val),
        "history": history,
    }
    return pred_test, train_info, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train novelty topology-constrained cVAE for OD trip generation")
    parser.add_argument("--input", default="data/processed/od/hourly_od_2023-01_local.parquet", help="Input OD parquet")
    parser.add_argument("--output-dir", default="data/processed/novelty", help="Output directory")
    parser.add_argument("--prefix", default="novelty_topocvae_2023-01", help="Output file prefix")
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
    parser.add_argument("--edge-sample-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=20)
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

    # Topology prior (soft target) from topology-derived baselines
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

    max_node_id = int(max(train_df["origin"].max(), train_df["destination"].max(), val_df["origin"].max(), val_df["destination"].max(), test_df["origin"].max(), test_df["destination"].max()))
    edge_src, edge_dst, edge_w = load_graph_edges(edge_file=edge_file if edge_file is not None else Path(""), max_node_id=max_node_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_topo, train_info, model = run_topology_cvae(
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
        edge_sample_size=args.edge_sample_size,
        sample_count=args.sample_count,
        device=device,
    )

    y_test = test_df["trip_count"].to_numpy(dtype=np.float32)
    metrics_topo = compute_metrics(y_test, pred_topo)
    metrics_prior = compute_metrics(y_test, prior_test)
    comparison = {
        "split": "test",
        "results": {
            "topocvae": metrics_topo,
            "topology_prior": metrics_prior,
        },
        "better_model_by_rmse": "topocvae" if metrics_topo["rmse"] <= metrics_prior["rmse"] else "topology_prior",
        "better_model_by_mae": "topocvae" if metrics_topo["mae"] <= metrics_prior["mae"] else "topology_prior",
    }

    pred_df = test_df[["pickup_hour", "origin", "destination", "trip_count"]].copy().rename(columns={"trip_count": "actual_trip_count"})
    pred_df["pred_topocvae"] = pred_topo.astype(np.float32)
    pred_df["pred_topology_prior"] = prior_test.astype(np.float32)

    split_summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "network_dir": str(network_dir),
        "community_file": str(community_file) if community_file is not None else None,
        "centrality_file": str(centrality_file) if centrality_file is not None else None,
        "edge_file": str(edge_file) if edge_file is not None else None,
        "device": device,
    }

    payload = {
        "model": "topology_constrained_cvae",
        "split": "test",
        "metrics": metrics_topo,
        "train": train_info,
        "notes": "Novelty model: cVAE with topology-prior consistency and graph Laplacian smoothness constraints",
    }

    split_path = output_dir / f"{args.prefix}_split_summary.json"
    topo_path = output_dir / f"{args.prefix}_topocvae_metrics.json"
    comp_path = output_dir / f"{args.prefix}_comparison.json"
    comp_csv_path = output_dir / f"{args.prefix}_comparison.csv"
    pred_path = output_dir / f"{args.prefix}_test_predictions.parquet"
    model_path = output_dir / "models" / f"{args.prefix}_topocvae.pt"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    with open(topo_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    pd.DataFrame([{"model": k, **v} for k, v in comparison["results"].items()]).to_csv(comp_csv_path, index=False)
    pred_df.to_parquet(pred_path, index=False)
    torch.save(model.state_dict(), model_path)

    print(f"Wrote: {split_path}")
    print(f"Wrote: {topo_path}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {comp_csv_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {model_path}")
    print(f"Test MAE/RMSE | topocvae: {metrics_topo['mae']:.4f}/{metrics_topo['rmse']:.4f}")
    print(f"Test MAE/RMSE | topology_prior: {metrics_prior['mae']:.4f}/{metrics_prior['rmse']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
