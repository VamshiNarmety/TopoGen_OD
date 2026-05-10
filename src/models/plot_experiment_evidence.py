#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


sns.set_theme(style="whitegrid")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _discover_zone_shapefile(raw_misc_dir: Path) -> Path | None:
    if not raw_misc_dir.exists():
        return None
    candidates = sorted(raw_misc_dir.rglob("*.shp"))
    if not candidates:
        return None
    return candidates[0]


def _load_zone_geometry(zone_shapefile: Path) -> tuple[dict[int, tuple[float, float]], list[dict[str, object]]]:
    if not zone_shapefile.exists():
        return {}, []

    suffix = zone_shapefile.suffix.lower()

    if suffix in {".geojson", ".json"}:
        try:
            with open(zone_shapefile, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}, []

        features = payload.get("features", []) if isinstance(payload, dict) else []
        centroids: dict[int, tuple[float, float]] = {}
        polygons: list[dict[str, object]] = []

        for feat in features:
            if not isinstance(feat, dict):
                continue
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            if not isinstance(props, dict) or not isinstance(geom, dict):
                continue

            raw_id = props.get("LocationID")
            if raw_id is None:
                raw_id = props.get("area_num_1") or props.get("area_numbe") or props.get("community_area")
            try:
                location_id = int(str(raw_id).strip())
            except Exception:
                continue

            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if not gtype or coords is None:
                continue

            poly_parts: list[np.ndarray] = []
            if gtype == "Polygon":
                for ring in coords:
                    arr = np.asarray(ring, dtype=np.float64)
                    if len(arr):
                        poly_parts.append(arr)
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        arr = np.asarray(ring, dtype=np.float64)
                        if len(arr):
                            poly_parts.append(arr)
            else:
                continue

            if poly_parts:
                all_points = np.vstack(poly_parts)
                centroid_x = float(np.mean(all_points[:, 0]))
                centroid_y = float(np.mean(all_points[:, 1]))
                centroids[location_id] = (centroid_x, centroid_y)
                polygons.append({"id": location_id, "parts": poly_parts})

        return centroids, polygons

    try:
        import shapefile  # type: ignore
    except Exception:
        return {}, []

    reader = shapefile.Reader(str(zone_shapefile))
    fields = [f[0] for f in reader.fields[1:]]
    if "LocationID" not in fields:
        return {}, []

    location_idx = fields.index("LocationID")
    centroids: dict[int, tuple[float, float]] = {}
    polygons: list[dict[str, object]] = []

    for record, shp in zip(reader.records(), reader.shapes()):
        try:
            location_id = int(record[location_idx])
        except Exception:
            continue
        points = np.asarray(shp.points, dtype=np.float64)
        if points.size == 0:
            continue

        parts = list(shp.parts) + [len(points)]
        poly_parts = []
        for start, end in zip(parts[:-1], parts[1:]):
            ring = points[start:end]
            if len(ring):
                poly_parts.append(ring)

        if poly_parts:
            centroid_x = float(np.mean(points[:, 0]))
            centroid_y = float(np.mean(points[:, 1]))
            centroids[location_id] = (centroid_x, centroid_y)
            polygons.append({"id": location_id, "parts": poly_parts})

    return centroids, polygons


def _read_comp(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["model", "mae", "rmse", "source"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    model_col = cols.get("model")
    mae_col = cols.get("mae")
    rmse_col = cols.get("rmse")
    if not (model_col and mae_col and rmse_col):
        return pd.DataFrame(columns=["model", "mae", "rmse", "source"])
    out = df[[model_col, mae_col, rmse_col]].copy()
    out.columns = ["model", "mae", "rmse"]
    out["source"] = source
    return out


def _model_from_pred_col(col: str) -> str:
    if not col.startswith("pred_"):
        return col
    return col.removeprefix("pred_")


def _read_predictions(path: Path, source: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["pickup_hour", "origin", "destination", "actual", "prediction", "model", "source"])

    df = pd.read_parquet(path)
    required = {"pickup_hour", "origin", "destination", "actual_trip_count"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["pickup_hour", "origin", "destination", "actual", "prediction", "model", "source"])

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        return pd.DataFrame(columns=["pickup_hour", "origin", "destination", "actual", "prediction", "model", "source"])

    base = df[["pickup_hour", "origin", "destination", "actual_trip_count"]].copy()
    base = base.rename(columns={"actual_trip_count": "actual"})
    rows = []
    for col in pred_cols:
        part = base.copy()
        part["prediction"] = df[col].astype(float)
        part["model"] = _model_from_pred_col(col)
        part["source"] = source if source else "predictions"
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], errors="coerce")
    out = out.dropna(subset=["pickup_hour", "actual", "prediction"])
    out["actual"] = out["actual"].clip(lower=0.0)
    out["prediction"] = out["prediction"].clip(lower=0.0)
    return out


def _pick_models_for_detail(comp_df: pd.DataFrame, pred_df: pd.DataFrame, max_models: int = 6) -> list[str]:
    """Pick models with consistent ordering across datasets for detail plots (temporal, error, scatter)."""
    available = set(pred_df["model"].unique()) if len(pred_df) else set()
    selected: list[str] = []

    def _add(model: str) -> None:
        if model in available and model not in selected:
            selected.append(model)

    # Priority order ensures same model order across NYC and Chicago (not sorted by RMSE)
    priority = [
        "mlp",
        "plain_cvae",
        "cgan",
        "topocvae",
        "topology_prior",
        "intervention_aware",
        "historical_mean",
        "od_marginal",
        "gravity",
        "radiation",
        "community",
        "centrality",
    ]
    for model in priority:
        _add(model)

    # Fallback: add any remaining models in sorted order
    for model in sorted(list(available)):
        _add(model)

    return selected[:max_models]


def _pick_models_for_graph(
    comp_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    max_models: int = 8,
    include_models: list[str] | None = None,
) -> list[str]:
    available = set(pred_df["model"].unique()) if len(pred_df) else set()
    selected: list[str] = []

    def _add(model: str) -> None:
        if model in available and model not in selected:
            selected.append(model)

    for model in (include_models or []):
        _add(model)

    # Prioritize novelty/intervention variants for topology storytelling
    priority = [
        "topocvae",
        "none",
        "edge_removed",
        "hub_removed",
        "node_added",
        "topology_prior",
        "cgan",
        "plain_cvae",
        "mlp",
        "historical_mean",
        "od_marginal",
    ]
    for model in priority:
        _add(model)

    if len(comp_df) and "rmse" in comp_df.columns:
        for model in comp_df.sort_values("rmse")["model"].astype(str).tolist():
            _add(model)

    for model in sorted(list(available)):
        _add(model)

    return selected[:max_models]


def plot_modelwise_temporal_variation(pred_df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(pred_df) == 0:
        return

    models = _pick_models_for_detail(comp_df, pred_df, max_models=6)
    if not models:
        return

    d = pred_df[pred_df["model"].isin(models)].copy()
    agg = d.groupby(["pickup_hour", "model"], as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))

    # keep first week for readability
    hours = sorted(agg["pickup_hour"].dropna().unique())
    if len(hours) > 168:
        keep = set(hours[:168])
        agg = agg[agg["pickup_hour"].isin(keep)]

    n = len(models)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for i, model in enumerate(models):
        ax = axes[i]
        sub = agg[agg["model"] == model].sort_values("pickup_hour")
        ax.plot(sub["pickup_hour"], sub["actual"], label="actual", linewidth=1.8)
        ax.plot(sub["pickup_hour"], sub["prediction"], label="prediction", linewidth=1.4)
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=30)
        if i == 0:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Model-wise temporal variation (hourly totals, first 168 test hours)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_modelwise_temporal_variation.png", dpi=180)
    plt.close(fig)


def plot_error_distribution(pred_df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(pred_df) == 0:
        return

    models = _pick_models_for_detail(comp_df, pred_df, max_models=8)
    d = pred_df[pred_df["model"].isin(models)].copy()
    if len(d) == 0:
        return

    d["abs_error"] = (d["actual"] - d["prediction"]).abs()
    # cap for readability
    clip_hi = float(d["abs_error"].quantile(0.99)) if len(d) else 0.0
    if clip_hi > 0:
        d["abs_error"] = d["abs_error"].clip(upper=clip_hi)

    plt.figure(figsize=(13, 5.5))
    sns.boxplot(data=d, x="model", y="abs_error")
    plt.xticks(rotation=35, ha="right")
    plt.title("Absolute error distribution by model (capped at 99th percentile)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_error_distribution_by_model.png", dpi=180)
    plt.close()


def plot_prediction_scatter(pred_df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(pred_df) == 0:
        return

    models = _pick_models_for_detail(comp_df, pred_df, max_models=6)
    d = pred_df[pred_df["model"].isin(models)].copy()
    if len(d) == 0:
        return

    # sampling keeps plot manageable
    sample_n = min(12000, len(d))
    d = d.sample(sample_n, random_state=42)

    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lim = float(max(d["actual"].max(), d["prediction"].max()))
    lim = max(lim, 1.0)

    for i, model in enumerate(models):
        ax = axes[i]
        sub = d[d["model"] == model]
        if len(sub) == 0:
            ax.axis("off")
            continue
        ax.hexbin(sub["actual"], sub["prediction"], gridsize=35, cmap="viridis", mincnt=1)
        ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0, color="red")
        ax.set_title(model)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Prediction vs actual density (hexbin)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_prediction_vs_actual_scatter.png", dpi=180)
    plt.close(fig)


def _topk_overlap_ratio(actual_edge: pd.DataFrame, pred_edge: pd.DataFrame, k: int) -> float:
    a = actual_edge.sort_values("actual", ascending=False).head(k)
    p = pred_edge.sort_values("prediction", ascending=False).head(k)
    set_a = set(zip(a["origin"].astype(int), a["destination"].astype(int)))
    set_p = set(zip(p["origin"].astype(int), p["destination"].astype(int)))
    if k <= 0:
        return 0.0
    return float(len(set_a.intersection(set_p)) / k)


def compute_topology_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    if len(pred_df) == 0:
        return pd.DataFrame(columns=["model", "origin_strength_spearman", "destination_strength_spearman", "edge_weight_spearman", "top100_edge_overlap"])

    rows = []
    for model, sub in pred_df.groupby("model"):
        edge = sub.groupby(["origin", "destination"], as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))

        # edge-level rank preservation
        if len(edge) >= 2:
            edge_s = float(edge["actual"].corr(edge["prediction"], method="spearman"))
        else:
            edge_s = float("nan")

        # node-strength preservation (outgoing and incoming)
        out_strength = edge.groupby("origin", as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        in_strength = edge.groupby("destination", as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        out_s = float(out_strength["actual"].corr(out_strength["prediction"], method="spearman")) if len(out_strength) >= 2 else float("nan")
        in_s = float(in_strength["actual"].corr(in_strength["prediction"], method="spearman")) if len(in_strength) >= 2 else float("nan")

        k = min(100, len(edge))
        overlap = 0.0
        if k > 0:
            actual_edge = edge[["origin", "destination", "actual"]]
            pred_edge = edge[["origin", "destination", "prediction"]]
            overlap = _topk_overlap_ratio(actual_edge, pred_edge, k=k)

        rows.append(
            {
                "model": model,
                "origin_strength_spearman": out_s,
                "destination_strength_spearman": in_s,
                "edge_weight_spearman": edge_s,
                "top100_edge_overlap": overlap,
            }
        )

    return pd.DataFrame(rows)


def plot_topology_scores(scores_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(scores_df) == 0:
        return

    score_cols = [
        "origin_strength_spearman",
        "destination_strength_spearman",
        "edge_weight_spearman",
        "top100_edge_overlap",
    ]

    # heatmap for compact comparison
    heat = scores_df.set_index("model")[score_cols].copy()
    plt.figure(figsize=(10, max(4, 0.6 * len(heat))))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title("Topology preservation scorecard by model")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_topology_preservation_heatmap.png", dpi=180)
    plt.close()

    # bar chart for easier reading in slides
    melt = scores_df.melt(id_vars=["model"], value_vars=score_cols, var_name="metric", value_name="score")
    plt.figure(figsize=(13, 5.5))
    sns.barplot(data=melt, x="model", y="score", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Topology preservation metrics by model")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_topology_preservation_bars.png", dpi=180)
    plt.close()


def _aggregate_zone_strength(sub: pd.DataFrame) -> pd.Series:
    origin = sub.groupby("origin", as_index=True)["actual"].sum()
    dest = sub.groupby("destination", as_index=True)["actual"].sum()
    return origin.add(dest, fill_value=0.0)


def _aggregate_zone_prediction_strength(sub: pd.DataFrame) -> pd.Series:
    origin = sub.groupby("origin", as_index=True)["prediction"].sum()
    dest = sub.groupby("destination", as_index=True)["prediction"].sum()
    return origin.add(dest, fill_value=0.0)


def _draw_zone_boundaries(ax, polygons: list[dict[str, object]], boundary_color: str = "#d1d5db") -> None:
    for poly in polygons:
        parts = poly.get("parts", [])
        if not isinstance(parts, list):
            continue
        for ring in parts:
            arr = np.asarray(ring, dtype=float)
            if len(arr) >= 2:
                ax.plot(arr[:, 0], arr[:, 1], color=boundary_color, linewidth=0.3, alpha=0.7)


def _plot_hotspot_panel(
    ax,
    polygons: list[dict[str, object]],
    centroids: dict[int, tuple[float, float]],
    values: pd.Series,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    use_diverging: bool = False,
) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _draw_zone_boundaries(ax, polygons)

    if len(values) == 0 or not centroids:
        return

    zone_ids = [int(z) for z in values.index.tolist() if int(z) in centroids]
    if not zone_ids:
        return

    xs = np.asarray([centroids[z][0] for z in zone_ids], dtype=float)
    ys = np.asarray([centroids[z][1] for z in zone_ids], dtype=float)
    vals = np.asarray([float(values.loc[z]) for z in zone_ids], dtype=float)

    if use_diverging:
        if center is None:
            center = 0.0
        norm = TwoSlopeNorm(vmin=vmin if vmin is not None else float(np.nanmin(vals)), vcenter=center, vmax=vmax if vmax is not None else float(np.nanmax(vals)))
        sc = ax.scatter(xs, ys, c=vals, cmap=cmap, norm=norm, s=np.clip(np.abs(vals) * 0.0025, 8, 120), alpha=0.9, linewidths=0)
    else:
        sc = ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=np.clip(vals * 0.0015, 8, 120), alpha=0.9, linewidths=0)
    return sc


def plot_spatial_hotspot_grid(
    pred_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    zone_shapefile: Path | None = None,
    top_k_zones: int = 60,
    max_models: int = 6,
    include_models: list[str] | None = None,
) -> str | None:
    if len(pred_df) == 0:
        return None

    if zone_shapefile is None:
        zone_shapefile = _discover_zone_shapefile(Path("data/raw/misc"))
    if zone_shapefile is None:
        return None

    centroids, polygons = _load_zone_geometry(zone_shapefile)
    if not centroids or not polygons:
        return None

    models = _pick_models_for_graph(comp_df, pred_df, max_models=max_models, include_models=include_models)
    if not models:
        return None

    actual_zone = pred_df.groupby(["origin", "destination"], as_index=False).agg(actual=("actual", "sum"))
    actual_strength = _aggregate_zone_strength(actual_zone).sort_values(ascending=False)
    actual_strength = actual_strength.head(top_k_zones)

    model_strengths: dict[str, pd.Series] = {}
    model_errors: dict[str, pd.Series] = {}
    for model in models:
        sub = pred_df[pred_df["model"] == model]
        if len(sub) == 0:
            continue
        edge = sub.groupby(["origin", "destination"], as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        pred_strength = _aggregate_zone_prediction_strength(edge)
        pred_strength = pred_strength.sort_values(ascending=False).head(top_k_zones)
        actual_by_zone = _aggregate_zone_strength(edge)
        common = actual_by_zone.index.union(pred_strength.index)
        actual_by_zone = actual_by_zone.reindex(common, fill_value=0.0)
        pred_strength = pred_strength.reindex(common, fill_value=0.0)
        model_strengths[model] = pred_strength
        model_errors[model] = (pred_strength - actual_by_zone).sort_values(ascending=False)

    if not model_strengths:
        return None

    actual_max = float(actual_strength.max()) if len(actual_strength) else 1.0
    pred_max = max((float(s.max()) for s in model_strengths.values() if len(s)), default=1.0)
    error_abs_max = max((float(np.abs(s).max()) for s in model_errors.values() if len(s)), default=1.0)

    nrows = len(model_strengths)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, max(4.2 * nrows, 5.0)), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    score_lookup = comp_df.groupby("model", as_index=True).first() if len(comp_df) else pd.DataFrame()
    for i, model in enumerate(model_strengths.keys()):
        row_axes = axes[i]
        _plot_hotspot_panel(row_axes[0], polygons, centroids, actual_strength, "Actual hotspot", cmap="viridis", vmin=0.0, vmax=actual_max)
        pred_panel = _plot_hotspot_panel(row_axes[1], polygons, centroids, model_strengths[model], f"Predicted: {model}", cmap="viridis", vmin=0.0, vmax=pred_max)
        err_title = f"Error (pred-actual): {model}"
        if len(score_lookup) and model in score_lookup.index:
            row = score_lookup.loc[model]
            err_title += f" | MAE={row.get('mae', np.nan):.3f}, RMSE={row.get('rmse', np.nan):.3f}"
        _plot_hotspot_panel(row_axes[2], polygons, centroids, model_errors[model], err_title, cmap="coolwarm", vmin=-error_abs_max, vmax=error_abs_max, center=0.0, use_diverging=True)
        if i == 0:
            row_axes[0].set_ylabel("Actual")
        row_axes[0].set_xlabel("")
        row_axes[1].set_xlabel("")
        row_axes[2].set_xlabel("")

    fig.suptitle("Spatial hotspot comparison: actual vs predicted vs error", fontsize=15)
    out_path = out_dir / f"{prefix}_spatial_hotspot_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name)).strip("_")


def _norm_width(values: np.ndarray, min_w: float = 0.4, max_w: float = 3.0) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    v = np.asarray(values, dtype=float)
    v = np.clip(v, a_min=0.0, a_max=None)
    vmax = float(v.max())
    if vmax <= 0:
        return np.full_like(v, min_w, dtype=float)
    return min_w + (max_w - min_w) * (v / vmax)


def _build_graph_from_edges(edge_df: pd.DataFrame, weight_col: str) -> nx.Graph:
    g = nx.Graph()
    for row in edge_df.itertuples(index=False):
        o = int(row.origin)
        d = int(row.destination)
        w = float(getattr(row, weight_col))
        if o == d:
            continue
        if g.has_edge(o, d):
            g[o][d]["weight"] += w
        else:
            g.add_edge(o, d, weight=w)
    return g


def _draw_graph_panel(ax, graph: nx.Graph, pos: dict[int, np.ndarray], title: str, edge_color: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=18,
        node_color="#2f2f2f",
        alpha=0.9,
        linewidths=0,
    )
    edge_list = list(graph.edges(data=True))
    widths = _norm_width(np.array([d.get("weight", 0.0) for _, _, d in edge_list], dtype=float))
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_color,
        width=widths,
        alpha=0.7,
    )


def plot_actual_vs_pred_graphs(
    pred_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    topology_scores: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    top_k_edges: int = 150,
    max_models: int = 6,
) -> list[str]:
    if len(pred_df) == 0:
        return []

    models = _pick_models_for_detail(comp_df, pred_df, max_models=max_models)
    if not models:
        return []

    written = []
    score_lookup = topology_scores.set_index("model") if len(topology_scores) else pd.DataFrame()

    for model in models:
        sub = pred_df[pred_df["model"] == model]
        if len(sub) == 0:
            continue

        edge = sub.groupby(["origin", "destination"], as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        if len(edge) == 0:
            continue

        actual_top = edge.sort_values("actual", ascending=False).head(top_k_edges)
        pred_top = edge.sort_values("prediction", ascending=False).head(top_k_edges)
        keep_pairs = set(zip(actual_top["origin"].astype(int), actual_top["destination"].astype(int)))
        keep_pairs.update(zip(pred_top["origin"].astype(int), pred_top["destination"].astype(int)))
        view = edge[[ (int(o), int(d)) in keep_pairs for o, d in zip(edge["origin"], edge["destination"]) ]].copy()
        if len(view) == 0:
            continue

        # Same layout for both panels for direct topology comparison
        layout_graph = _build_graph_from_edges(view.assign(weight=view["actual"] + view["prediction"]), "weight")
        if layout_graph.number_of_nodes() == 0:
            continue
        pos = nx.spring_layout(layout_graph, seed=42, weight="weight")

        g_actual = _build_graph_from_edges(view, "actual")
        g_pred = _build_graph_from_edges(view, "prediction")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _draw_graph_panel(axes[0], g_actual, pos, f"Actual graph (top-{top_k_edges} edges)", edge_color="#6b7280")
        _draw_graph_panel(axes[1], g_pred, pos, f"Predicted graph: {model}", edge_color="#2563eb")

        extra = ""
        if len(score_lookup) and model in score_lookup.index:
            row = score_lookup.loc[model]
            edge_s = row.get("edge_weight_spearman", np.nan)
            overlap = row.get("top100_edge_overlap", np.nan)
            extra = f" | edge_spearman={edge_s:.3f}, top100_overlap={overlap:.2f}"
        fig.suptitle(f"City topology: actual vs predicted ({model}){extra}")
        fig.tight_layout()

        out_path = out_dir / f"{prefix}_graph_actual_vs_pred_{_safe_name(model)}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(str(out_path.name))

    return written


def plot_actual_vs_pred_graph_grid(
    pred_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    topology_scores: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    top_k_edges: int = 150,
    max_models: int = 5,
    max_nodes: int = 60,
    include_models: list[str] | None = None,
) -> str | None:
    if len(pred_df) == 0:
        return None

    models = _pick_models_for_graph(comp_df, pred_df, max_models=max_models, include_models=include_models)
    if not models:
        return None

    # Build per-panel edge tables
    panel_edges: dict[str, pd.DataFrame] = {}
    union_rows = []

    # actual panel from all rows
    actual_edge_all = pred_df.groupby(["origin", "destination"], as_index=False).agg(actual=("actual", "sum"))
    actual_edge = actual_edge_all.sort_values("actual", ascending=False).head(top_k_edges).copy()
    actual_edge = actual_edge.rename(columns={"actual": "weight"})
    panel_edges["actual"] = actual_edge
    union_rows.append(actual_edge.assign(src="actual"))

    for model in models:
        sub = pred_df[pred_df["model"] == model]
        edge = sub.groupby(["origin", "destination"], as_index=False).agg(prediction=("prediction", "sum"))
        edge = edge.sort_values("prediction", ascending=False).head(top_k_edges).copy()
        edge = edge.rename(columns={"prediction": "weight"})
        panel_edges[model] = edge
        union_rows.append(edge.assign(src=model))

    union_edge = pd.concat(union_rows, ignore_index=True) if union_rows else pd.DataFrame()
    if len(union_edge) == 0:
        return None

    # Keep strongest nodes to reduce clutter
    node_strength = (
        union_edge.groupby("origin", as_index=False)["weight"].sum().rename(columns={"origin": "node", "weight": "s_out"})
        .merge(
            union_edge.groupby("destination", as_index=False)["weight"].sum().rename(columns={"destination": "node", "weight": "s_in"}),
            on="node",
            how="outer",
        )
        .fillna(0.0)
    )
    node_strength["strength"] = node_strength["s_out"] + node_strength["s_in"]
    top_nodes = set(node_strength.sort_values("strength", ascending=False).head(max_nodes)["node"].astype(int).tolist())
    if not top_nodes:
        return None

    for key, edge in list(panel_edges.items()):
        panel_edges[key] = edge[
            edge["origin"].astype(int).isin(top_nodes) & edge["destination"].astype(int).isin(top_nodes)
        ].copy()

    # Shared layout graph
    layout_base = []
    for key, edge in panel_edges.items():
        if len(edge):
            layout_base.append(edge.assign(panel=key))
    if not layout_base:
        return None
    layout_df = pd.concat(layout_base, ignore_index=True)
    layout_agg = layout_df.groupby(["origin", "destination"], as_index=False)["weight"].mean()
    layout_graph = _build_graph_from_edges(layout_agg, "weight")
    if layout_graph.number_of_nodes() == 0:
        return None

    # Stronger repulsion for clearer spacing
    k = 2.5 / np.sqrt(max(layout_graph.number_of_nodes(), 1))
    pos = nx.spring_layout(layout_graph, seed=42, weight="weight", k=k, iterations=400)

    panel_names = ["actual"] + models
    n_panels = len(panel_names)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    score_lookup = topology_scores.set_index("model") if len(topology_scores) else pd.DataFrame()

    for i, name in enumerate(panel_names):
        ax = axes[i]
        edge = panel_edges.get(name, pd.DataFrame(columns=["origin", "destination", "weight"]))
        graph = _build_graph_from_edges(edge, "weight") if len(edge) else nx.Graph()
        if name == "actual":
            title = f"Actual (top-{top_k_edges}, nodes≤{max_nodes})"
            color = "#6b7280"
        else:
            title = name
            if len(score_lookup) and name in score_lookup.index:
                row = score_lookup.loc[name]
                title += f"\nedge_s={row.get('edge_weight_spearman', np.nan):.3f}, ovl={row.get('top100_edge_overlap', np.nan):.2f}"
            color = "#2563eb"
        _draw_graph_panel(ax, graph, pos, title, edge_color=color)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("City graph topology comparison: Actual vs model predictions", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_graph_actual_vs_pred_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def plot_model_comparison(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(df) == 0:
        return

    # MAE
    plt.figure(figsize=(12, 5))
    d = df.sort_values("mae")
    sns.barplot(data=d, x="model", y="mae", hue="source")
    plt.xticks(rotation=35, ha="right")
    plt.title("Model comparison by MAE")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_mae_comparison.png", dpi=180)
    plt.close()

    # RMSE
    plt.figure(figsize=(12, 5))
    d = df.sort_values("rmse")
    sns.barplot(data=d, x="model", y="rmse", hue="source")
    plt.xticks(rotation=35, ha="right")
    plt.title("Model comparison by RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_rmse_comparison.png", dpi=180)
    plt.close()

    # Top-5 focus table image
    top = d.nsmallest(5, "rmse")[ ["model", "mae", "rmse", "source"] ].copy()
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=top.round(4).values, colLabels=top.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_top5_table.png", dpi=180)
    plt.close()


def plot_disruption(disruption_csv: Path, out_dir: Path, prefix: str) -> None:
    if not disruption_csv.exists():
        return
    df = pd.read_csv(disruption_csv)
    required = {"scenario_id", "flow_retention_ratio", "global_efficiency_undirected"}
    if not required.issubset(df.columns):
        return

    d = df.copy()
    d = d.sort_values("flow_retention_ratio", ascending=True)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=d, x="scenario_id", y="flow_retention_ratio")
    plt.xticks(rotation=40, ha="right")
    plt.title("Disruption scenarios: flow retention ratio")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_disruption_flow_retention.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=d, x="scenario_id", y="global_efficiency_undirected")
    plt.xticks(rotation=40, ha="right")
    plt.title("Disruption scenarios: global efficiency")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_disruption_efficiency.png", dpi=180)
    plt.close()


def _read_history(path: Path, model_name: str, key_map: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["epoch", "value", "series", "model"])
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hist = None
    if isinstance(payload, dict):
        if "train" in payload and isinstance(payload["train"], dict) and "history" in payload["train"]:
            hist = payload["train"]["history"]
        elif "history" in payload:
            hist = payload["history"]
    if not isinstance(hist, list):
        return pd.DataFrame(columns=["epoch", "value", "series", "model"])

    rows = []
    for row in hist:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        for src_key, series_name in key_map.items():
            if src_key in row:
                rows.append({
                    "epoch": int(epoch),
                    "value": float(row[src_key]),
                    "series": series_name,
                    "model": model_name,
                })
    return pd.DataFrame(rows)


def plot_training_curves(histories: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(histories) == 0:
        return
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=histories, x="epoch", y="value", hue="model", style="series")
    plt.title("Training curves (validation + training losses)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_training_curves.png", dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visual evidence pack for TopoGen-OD experiments")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)

    parser.add_argument("--baseline-comp", default=None)
    parser.add_argument("--deepgen-comp", default=None)
    parser.add_argument("--novelty-comp", default=None)
    parser.add_argument("--intervention-comp", default=None)
    parser.add_argument("--disruption-comp", default=None)

    parser.add_argument("--mlp-metrics", default=None)
    parser.add_argument("--deepgen-cvae-metrics", default=None)
    parser.add_argument("--deepgen-cgan-metrics", default=None)
    parser.add_argument("--novelty-metrics", default=None)
    parser.add_argument("--intervention-metrics", default=None)

    parser.add_argument("--baseline-preds", default=None)
    parser.add_argument("--deepgen-preds", default=None)
    parser.add_argument("--novelty-preds", default=None)
    parser.add_argument("--intervention-preds", default=None)
    parser.add_argument("--graph-topk", type=int, default=150, help="Top-K OD edges to display in graph comparison plots")
    parser.add_argument("--graph-max-models", type=int, default=6, help="Maximum number of models for graph comparison plots")
    parser.add_argument("--graph-max-nodes", type=int, default=60, help="Maximum number of nodes in graph comparison plots for readability")
    parser.add_argument(
        "--graph-include-models",
        default="topocvae,none,edge_removed,hub_removed,node_added,topology_prior",
        help="Comma-separated model names to force-include in graph comparison grid",
    )
    parser.add_argument("--graph-separate", action="store_true", help="Also emit separate per-model actual-vs-predicted graph figures")
    parser.add_argument("--spatial-zone-shapefile", default=None, help="Optional zone shapefile for spatial hotspot maps")
    parser.add_argument("--spatial-top-zones", type=int, default=60, help="Top zones to show in spatial hotspot plots")
    parser.add_argument("--spatial-max-models", type=int, default=6, help="Maximum models for spatial hotspot plots")
    parser.add_argument(
        "--spatial-include-models",
        default="topocvae,none,edge_removed,hub_removed,node_added,topology_prior",
        help="Comma-separated model names to force-include in spatial hotspot grid",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    comp_frames = []
    if args.baseline_comp:
        comp_frames.append(_read_comp(Path(args.baseline_comp), "baseline"))
    if args.deepgen_comp:
        comp_frames.append(_read_comp(Path(args.deepgen_comp), "deepgen"))
    if args.novelty_comp:
        comp_frames.append(_read_comp(Path(args.novelty_comp), "novelty_v1"))
    if args.intervention_comp:
        comp_frames.append(_read_comp(Path(args.intervention_comp), "novelty_v2"))

    comp_df = pd.concat(comp_frames, ignore_index=True) if comp_frames else pd.DataFrame()
    plot_model_comparison(comp_df, out_dir, args.prefix)

    if args.disruption_comp:
        plot_disruption(Path(args.disruption_comp), out_dir, args.prefix)

    histories = []
    if args.mlp_metrics:
        histories.append(_read_history(Path(args.mlp_metrics), "mlp", {"train_mse_log": "train", "val_mse_log": "val"}))
    if args.deepgen_cvae_metrics:
        histories.append(_read_history(Path(args.deepgen_cvae_metrics), "plain_cvae", {"train_loss": "train", "val_loss": "val"}))
    if args.deepgen_cgan_metrics:
        histories.append(_read_history(Path(args.deepgen_cgan_metrics), "cgan", {"d_loss": "disc", "g_loss": "gen", "val_mse_log": "val"}))
    if args.novelty_metrics:
        histories.append(_read_history(Path(args.novelty_metrics), "topocvae", {"train_loss": "train", "val_loss": "val"}))
    if args.intervention_metrics:
        histories.append(_read_history(Path(args.intervention_metrics), "intervention_topocvae", {"train_loss": "train", "val_loss": "val"}))

    hist_df = pd.concat([h for h in histories if len(h)], ignore_index=True) if histories else pd.DataFrame()
    plot_training_curves(hist_df, out_dir, args.prefix)

    pred_frames = []
    if args.baseline_preds:
        pred_frames.append(_read_predictions(Path(args.baseline_preds), "baseline"))
    if args.deepgen_preds:
        pred_frames.append(_read_predictions(Path(args.deepgen_preds), "deepgen"))
    if args.novelty_preds:
        pred_frames.append(_read_predictions(Path(args.novelty_preds), "novelty_v1"))
    if args.intervention_preds:
        pred_frames.append(_read_predictions(Path(args.intervention_preds), "novelty_v2"))

    pred_df = pd.concat([p for p in pred_frames if len(p)], ignore_index=True) if pred_frames else pd.DataFrame()
    plot_modelwise_temporal_variation(pred_df, comp_df, out_dir, args.prefix)
    plot_error_distribution(pred_df, comp_df, out_dir, args.prefix)
    plot_prediction_scatter(pred_df, comp_df, out_dir, args.prefix)

    topology_scores = compute_topology_scores(pred_df)
    plot_topology_scores(topology_scores, out_dir, args.prefix)
    graph_include_models = [m.strip() for m in str(args.graph_include_models).split(",") if m.strip()]
    spatial_include_models = [m.strip() for m in str(args.spatial_include_models).split(",") if m.strip()]

    graph_grid_file = plot_actual_vs_pred_graph_grid(
        pred_df,
        comp_df,
        topology_scores,
        out_dir,
        args.prefix,
        top_k_edges=max(10, int(args.graph_topk)),
        max_models=max(1, int(args.graph_max_models)),
        max_nodes=max(20, int(args.graph_max_nodes)),
        include_models=graph_include_models,
    )
    graph_files = []
    if args.graph_separate:
        graph_files = plot_actual_vs_pred_graphs(
            pred_df,
            comp_df,
            topology_scores,
            out_dir,
            args.prefix,
            top_k_edges=max(10, int(args.graph_topk)),
            max_models=max(1, int(args.graph_max_models)),
        )

    spatial_grid_file = plot_spatial_hotspot_grid(
        pred_df,
        comp_df,
        out_dir,
        args.prefix,
        zone_shapefile=Path(args.spatial_zone_shapefile) if args.spatial_zone_shapefile else None,
        top_k_zones=max(10, int(args.spatial_top_zones)),
        max_models=max(1, int(args.spatial_max_models)),
        include_models=spatial_include_models,
    )

    if len(topology_scores):
        topology_scores.sort_values("edge_weight_spearman", ascending=False).to_csv(
            out_dir / f"{args.prefix}_topology_scores.csv", index=False
        )

    summary = {
        "plots_generated_in": str(out_dir),
        "prefix": args.prefix,
        "comparison_rows": int(len(comp_df)),
        "history_rows": int(len(hist_df)),
        "prediction_rows": int(len(pred_df)),
        "models_in_prediction_plots": sorted(pred_df["model"].unique().tolist()) if len(pred_df) else [],
        "graph_topology_grid_file": graph_grid_file,
        "graph_include_models": graph_include_models,
        "graph_topology_files": graph_files,
        "spatial_hotspot_grid_file": spatial_grid_file,
        "spatial_include_models": spatial_include_models,
    }
    with open(out_dir / f"{args.prefix}_evidence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_dir / (args.prefix + '_evidence_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
