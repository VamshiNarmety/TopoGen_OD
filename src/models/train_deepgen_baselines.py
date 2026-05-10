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
        raise ValueError("Temporal split produced an empty set. Check --train-end and --val-end values.")

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


class ConditionEncoder(nn.Module):
    def __init__(self, n_origins: int, n_destinations: int, emb_dim: int) -> None:
        super().__init__()
        self.origin_emb = nn.Embedding(n_origins + 1, emb_dim)
        self.destination_emb = nn.Embedding(n_destinations + 1, emb_dim)

    def forward(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        o = self.origin_emb(origin)
        d = self.destination_emb(destination)
        return torch.cat([o, d, cont], dim=1)


class PlainCVAE(nn.Module):
    def __init__(
        self,
        n_origins: int,
        n_destinations: int,
        emb_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(n_origins=n_origins, n_destinations=n_destinations, emb_dim=emb_dim)
        cond_dim = emb_dim * 2 + 5

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
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        cont: torch.Tensor,
        target_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.cond_encoder(origin, destination, cont)
        enc_in = torch.cat([cond, target_log.unsqueeze(1)], dim=1)
        h = self.encoder(enc_in)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self._reparam(mu, logvar)
        return cond, mu, logvar, z

    def decode_from_cond_and_z(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dec_in = torch.cat([cond, z], dim=1)
        return self.decoder(dec_in).squeeze(1)

    def forward(
        self,
        origin: torch.Tensor,
        destination: torch.Tensor,
        cont: torch.Tensor,
        target_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond, mu, logvar, z = self.encode(origin, destination, cont, target_log)
        pred_log = self.decode_from_cond_and_z(cond, z)
        return pred_log, mu, logvar


class CGANGenerator(nn.Module):
    def __init__(
        self,
        n_origins: int,
        n_destinations: int,
        noise_dim: int = 16,
        emb_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_encoder = ConditionEncoder(n_origins=n_origins, n_destinations=n_destinations, emb_dim=emb_dim)
        cond_dim = emb_dim * 2 + 5
        self.net = nn.Sequential(
            nn.Linear(cond_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(origin, destination, cont)
        x = torch.cat([cond, noise], dim=1)
        return self.net(x).squeeze(1)


class CGANDiscriminator(nn.Module):
    def __init__(self, n_origins: int, n_destinations: int, emb_dim: int = 16, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(n_origins=n_origins, n_destinations=n_destinations, emb_dim=emb_dim)
        cond_dim = emb_dim * 2 + 5
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, origin: torch.Tensor, destination: torch.Tensor, cont: torch.Tensor, y_log: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(origin, destination, cont)
        x = torch.cat([cond, y_log.unsqueeze(1)], dim=1)
        return self.net(x).squeeze(1)


def run_cvae(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    hidden_dim: int,
    latent_dim: int,
    beta_kl: float,
    sample_count: int,
    device: str,
) -> tuple[np.ndarray, dict, PlainCVAE]:
    n_origins = int(max(train_df["origin"].max(), val_df["origin"].max(), test_df["origin"].max()))
    n_destinations = int(max(train_df["destination"].max(), val_df["destination"].max(), test_df["destination"].max()))

    model = PlainCVAE(
        n_origins=n_origins,
        n_destinations=n_destinations,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    train_loader = DataLoader(ODDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ODDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ODDataset(test_df), batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="cVAE epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_batch_bar = tqdm(train_loader, desc=f"cVAE train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in train_batch_bar:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            target_log = batch["target_log"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_log, mu, logvar = model(origin, destination, cont, target_log)
            recon = torch.mean((pred_log - target_log) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta_kl * kl
            loss.backward()
            optimizer.step()

            bs = origin.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_n += bs
            train_batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                origin = batch["origin"].to(device)
                destination = batch["destination"].to(device)
                cont = batch["cont"].to(device)
                target_log = batch["target_log"].to(device)
                pred_log, mu, logvar = model(origin, destination, cont, target_log)
                recon = torch.mean((pred_log - target_log) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + beta_kl * kl

                bs = origin.shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        val_loss = val_loss_sum / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        print(f"[cVAE] Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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

            cond = model.cond_encoder(origin, destination, cont)
            samples = []
            for _ in range(max(1, sample_count)):
                z = torch.randn((origin.shape[0], latent_dim), device=device)
                pred_log = model.decode_from_cond_and_z(cond, z)
                pred = torch.expm1(pred_log).clamp(min=0.0)
                samples.append(pred)
            pred_mean = torch.stack(samples, dim=0).mean(dim=0)
            preds.append(pred_mean.detach().cpu().numpy())

    pred_test = np.concatenate(preds).astype(np.float32)
    train_info = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "embedding_dim": int(emb_dim),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "beta_kl": float(beta_kl),
        "sample_count": int(sample_count),
        "best_val_loss": float(best_val),
        "history": history,
    }
    return pred_test, train_info, model


def run_cgan(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    hidden_dim: int,
    noise_dim: int,
    adv_weight: float,
    rec_weight: float,
    sample_count: int,
    device: str,
) -> tuple[np.ndarray, dict, CGANGenerator]:
    n_origins = int(max(train_df["origin"].max(), val_df["origin"].max(), test_df["origin"].max()))
    n_destinations = int(max(train_df["destination"].max(), val_df["destination"].max(), test_df["destination"].max()))

    generator = CGANGenerator(
        n_origins=n_origins,
        n_destinations=n_destinations,
        noise_dim=noise_dim,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    discriminator = CGANDiscriminator(
        n_origins=n_origins,
        n_destinations=n_destinations,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    train_loader = DataLoader(ODDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ODDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ODDataset(test_df), batch_size=batch_size, shuffle=False, num_workers=0)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="cGAN epochs", unit="epoch")
    for epoch in epoch_bar:
        generator.train()
        discriminator.train()

        g_loss_sum = 0.0
        d_loss_sum = 0.0
        n_seen = 0

        train_batch_bar = tqdm(train_loader, desc=f"cGAN train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in train_batch_bar:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)
            real_y = batch["target_log"].to(device)
            bs = origin.shape[0]

            # 1) Train discriminator
            d_opt.zero_grad(set_to_none=True)
            noise = torch.randn((bs, noise_dim), device=device)
            fake_y = generator(origin, destination, cont, noise).detach()

            real_logits = discriminator(origin, destination, cont, real_y)
            fake_logits = discriminator(origin, destination, cont, fake_y)

            real_loss = bce(real_logits, torch.ones_like(real_logits))
            fake_loss = bce(fake_logits, torch.zeros_like(fake_logits))
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            d_opt.step()

            # 2) Train generator
            g_opt.zero_grad(set_to_none=True)
            noise = torch.randn((bs, noise_dim), device=device)
            gen_y = generator(origin, destination, cont, noise)
            logits = discriminator(origin, destination, cont, gen_y)

            adv_loss = bce(logits, torch.ones_like(logits))
            rec_loss = mse(gen_y, real_y)
            g_loss = adv_weight * adv_loss + rec_weight * rec_loss
            g_loss.backward()
            g_opt.step()

            d_loss_sum += float(d_loss.item()) * bs
            g_loss_sum += float(g_loss.item()) * bs
            n_seen += bs
            train_batch_bar.set_postfix(d=f"{d_loss.item():.4f}", g=f"{g_loss.item():.4f}")

        generator.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                origin = batch["origin"].to(device)
                destination = batch["destination"].to(device)
                cont = batch["cont"].to(device)
                target_log = batch["target_log"].to(device)

                # Validation with deterministic-ish Monte Carlo mean
                samples = []
                for _ in range(max(1, sample_count)):
                    noise = torch.randn((origin.shape[0], noise_dim), device=device)
                    pred_log = generator(origin, destination, cont, noise)
                    samples.append(pred_log)
                pred_log_mean = torch.stack(samples, dim=0).mean(dim=0)
                val_loss = torch.mean((pred_log_mean - target_log) ** 2)

                bs = origin.shape[0]
                val_loss_sum += float(val_loss.item()) * bs
                val_n += bs

        mean_d = d_loss_sum / max(1, n_seen)
        mean_g = g_loss_sum / max(1, n_seen)
        mean_val = val_loss_sum / max(1, val_n)
        history.append({"epoch": epoch, "d_loss": mean_d, "g_loss": mean_g, "val_mse_log": mean_val})
        epoch_bar.set_postfix(d=f"{mean_d:.4f}", g=f"{mean_g:.4f}", val=f"{mean_val:.4f}")
        print(f"[cGAN] Epoch {epoch}/{epochs} | d_loss={mean_d:.6f} | g_loss={mean_g:.6f} | val_mse_log={mean_val:.6f}")

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}

    if best_state is not None:
        generator.load_state_dict(best_state)

    preds = []
    generator.eval()
    with torch.no_grad():
        for batch in test_loader:
            origin = batch["origin"].to(device)
            destination = batch["destination"].to(device)
            cont = batch["cont"].to(device)

            samples = []
            for _ in range(max(1, sample_count)):
                noise = torch.randn((origin.shape[0], noise_dim), device=device)
                pred_log = generator(origin, destination, cont, noise)
                pred = torch.expm1(pred_log).clamp(min=0.0)
                samples.append(pred)
            pred_mean = torch.stack(samples, dim=0).mean(dim=0)
            preds.append(pred_mean.detach().cpu().numpy())

    pred_test = np.concatenate(preds).astype(np.float32)
    train_info = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "embedding_dim": int(emb_dim),
        "hidden_dim": int(hidden_dim),
        "noise_dim": int(noise_dim),
        "adv_weight": float(adv_weight),
        "rec_weight": float(rec_weight),
        "sample_count": int(sample_count),
        "best_val_mse_log": float(best_val),
        "history": history,
    }
    return pred_test, train_info, generator


def main() -> int:
    parser = argparse.ArgumentParser(description="Train deep generative OD baselines (plain cVAE + cGAN)")
    parser.add_argument("--input", default="data/processed/od/hourly_od_2023-01_local.parquet", help="Input OD parquet")
    parser.add_argument("--output-dir", default="data/processed/deepgen_baselines", help="Output directory")
    parser.add_argument("--prefix", default="deepgen_2023-01", help="Output file prefix")
    parser.add_argument("--train-end", default="2023-01-23 23:00:00", help="Train split end timestamp (inclusive)")
    parser.add_argument("--val-end", default="2023-01-27 23:00:00", help="Validation split end timestamp (inclusive)")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap on training rows")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for each deep model")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width")

    parser.add_argument("--cvae-latent-dim", type=int, default=16, help="Latent dimension for plain cVAE")
    parser.add_argument("--cvae-beta-kl", type=float, default=0.05, help="KL multiplier for cVAE")

    parser.add_argument("--cgan-noise-dim", type=int, default=16, help="Noise dimension for cGAN")
    parser.add_argument("--cgan-adv-weight", type=float, default=0.1, help="Adversarial loss weight for cGAN")
    parser.add_argument("--cgan-rec-weight", type=float, default=1.0, help="Reconstruction loss weight for cGAN")

    parser.add_argument("--sample-count", type=int, default=20, help="MC samples at inference for each generative model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "models")

    df = load_od(Path(args.input))
    df = add_time_features(df)
    train_df_full, val_df, test_df = temporal_split(df, train_end=args.train_end, val_end=args.val_end)
    train_df = maybe_subsample(train_df_full, max_rows=args.max_train_rows, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    y_test = test_df["trip_count"].values.astype(np.float32)

    pred_cvae, cvae_info, cvae_model = run_cvae(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.cvae_latent_dim,
        beta_kl=args.cvae_beta_kl,
        sample_count=args.sample_count,
        device=device,
    )

    pred_cgan, cgan_info, cgan_generator = run_cgan(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        noise_dim=args.cgan_noise_dim,
        adv_weight=args.cgan_adv_weight,
        rec_weight=args.cgan_rec_weight,
        sample_count=args.sample_count,
        device=device,
    )

    metrics = {
        "plain_cvae": compute_metrics(y_test, pred_cvae),
        "cgan": compute_metrics(y_test, pred_cgan),
    }

    pred_df = test_df[["pickup_hour", "origin", "destination", "trip_count"]].copy()
    pred_df = pred_df.rename(columns={"trip_count": "actual_trip_count"})
    pred_df["pred_plain_cvae"] = pred_cvae.astype(np.float32)
    pred_df["pred_cgan"] = pred_cgan.astype(np.float32)

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
        "device": device,
    }

    cvae_payload = {
        "model": "plain_cvae",
        "split": "test",
        "metrics": metrics["plain_cvae"],
        "train": cvae_info,
        "notes": "Plain conditional VAE baseline without topology constraints",
    }
    cgan_payload = {
        "model": "cgan",
        "split": "test",
        "metrics": metrics["cgan"],
        "train": cgan_info,
        "notes": "Conditional GAN baseline with adversarial + reconstruction objective",
    }
    comparison = {
        "split": "test",
        "results": metrics,
        "better_model_by_rmse": min(metrics, key=lambda k: metrics[k]["rmse"]),
        "better_model_by_mae": min(metrics, key=lambda k: metrics[k]["mae"]),
    }

    split_path = output_dir / f"{args.prefix}_split_summary.json"
    cvae_path = output_dir / f"{args.prefix}_plain_cvae_metrics.json"
    cgan_path = output_dir / f"{args.prefix}_cgan_metrics.json"
    comp_path = output_dir / f"{args.prefix}_comparison.json"
    comp_csv_path = output_dir / f"{args.prefix}_comparison.csv"
    pred_path = output_dir / f"{args.prefix}_test_predictions.parquet"
    cvae_model_path = output_dir / "models" / f"{args.prefix}_plain_cvae.pt"
    cgan_model_path = output_dir / "models" / f"{args.prefix}_cgan_generator.pt"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    with open(cvae_path, "w", encoding="utf-8") as f:
        json.dump(cvae_payload, f, indent=2)
    with open(cgan_path, "w", encoding="utf-8") as f:
        json.dump(cgan_payload, f, indent=2)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    pd.DataFrame([{"model": k, **v} for k, v in metrics.items()]).to_csv(comp_csv_path, index=False)
    pred_df.to_parquet(pred_path, index=False)

    torch.save(cvae_model.state_dict(), cvae_model_path)
    torch.save(cgan_generator.state_dict(), cgan_model_path)

    print(f"Wrote: {split_path}")
    print(f"Wrote: {cvae_path}")
    print(f"Wrote: {cgan_path}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {comp_csv_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {cvae_model_path}")
    print(f"Wrote: {cgan_model_path}")
    for name, met in metrics.items():
        print(f"Test MAE/RMSE | {name}: {met['mae']:.4f}/{met['rmse']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
