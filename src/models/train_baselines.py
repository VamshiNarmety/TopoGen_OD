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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
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
    parser.add_argument("--epochs", type=int, default=6, help="MLP training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="MLP batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden layer size")
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

    train_df, val_df, test_df = temporal_split(df, train_end=args.train_end, val_end=args.val_end)
    train_df = maybe_subsample(train_df, max_rows=args.max_train_rows, seed=args.seed)

    split_summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_time_min": str(train_df["pickup_hour"].min()),
        "train_time_max": str(train_df["pickup_hour"].max()),
        "val_time_min": str(val_df["pickup_hour"].min()),
        "val_time_max": str(val_df["pickup_hour"].max()),
        "test_time_min": str(test_df["pickup_hour"].min()),
        "test_time_max": str(test_df["pickup_hour"].max()),
    }

    # Historical mean baseline
    pred_hist = predict_historical_mean(train_df, test_df)
    y_test = test_df["trip_count"].values.astype(np.float32)
    hist_metrics = compute_metrics(y_test, pred_hist)

    # MLP baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_mlp, mlp_train_info, mlp_model = run_mlp(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    mlp_metrics = compute_metrics(y_test, pred_mlp)

    # Save predictions on test split
    pred_df = test_df[["pickup_hour", "origin", "destination", "trip_count"]].copy()
    pred_df = pred_df.rename(columns={"trip_count": "actual_trip_count"})
    pred_df["pred_historical_mean"] = pred_hist.astype(np.float32)
    pred_df["pred_mlp"] = pred_mlp.astype(np.float32)

    pred_path = output_dir / f"{args.prefix}_test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    # Save metrics and training metadata
    hist_payload = {
        "model": "historical_mean",
        "split": "test",
        "metrics": hist_metrics,
        "notes": "Keys: origin,destination,hour,dayofweek with hierarchical fallback",
    }
    mlp_payload = {
        "model": "mlp_torch",
        "split": "test",
        "metrics": mlp_metrics,
        "train": mlp_train_info,
        "device": device,
    }
    comparison = {
        "split": "test",
        "historical_mean": hist_metrics,
        "mlp": mlp_metrics,
        "better_model_by_rmse": "mlp" if mlp_metrics["rmse"] < hist_metrics["rmse"] else "historical_mean",
        "better_model_by_mae": "mlp" if mlp_metrics["mae"] < hist_metrics["mae"] else "historical_mean",
    }

    split_path = output_dir / f"{args.prefix}_split_summary.json"
    hist_path = output_dir / f"{args.prefix}_historical_metrics.json"
    mlp_path = output_dir / f"{args.prefix}_mlp_metrics.json"
    comp_path = output_dir / f"{args.prefix}_comparison.json"
    comp_csv_path = output_dir / f"{args.prefix}_comparison.csv"
    model_path = output_dir / "models" / f"{args.prefix}_mlp.pt"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_payload, f, indent=2)
    with open(mlp_path, "w", encoding="utf-8") as f:
        json.dump(mlp_payload, f, indent=2)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    pd.DataFrame(
        [
            {"model": "historical_mean", **hist_metrics},
            {"model": "mlp_torch", **mlp_metrics},
        ]
    ).to_csv(comp_csv_path, index=False)

    torch.save(mlp_model.state_dict(), model_path)

    print(f"Wrote: {split_path}")
    print(f"Wrote: {hist_path}")
    print(f"Wrote: {mlp_path}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {comp_csv_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {model_path}")
    print(f"Test MAE/RMSE | Historical: {hist_metrics['mae']:.4f}/{hist_metrics['rmse']:.4f}")
    print(f"Test MAE/RMSE | MLP:        {mlp_metrics['mae']:.4f}/{mlp_metrics['rmse']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
