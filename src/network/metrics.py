#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_edges(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Edge file not found: {input_path}")

    df = pd.read_parquet(input_path)
    required = {"origin", "destination", "trip_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Edge file is missing columns: {sorted(missing)}")

    df = df.copy()
    df = df.dropna(subset=["origin", "destination", "trip_count"])
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce")
    df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
    df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce")
    df = df.dropna(subset=["origin", "destination", "trip_count"]).copy()
    df = df[(df["origin"] > 0) & (df["destination"] > 0) & (df["trip_count"] > 0)].copy()
    df["origin"] = df["origin"].astype("int32")
    df["destination"] = df["destination"].astype("int32")
    df["trip_count"] = df["trip_count"].astype("int64")
    return df


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in edges.itertuples(index=False):
        g.add_edge(int(row.origin), int(row.destination), weight=int(row.trip_count))
    return g


def top_k(mapping: dict[int, float], k: int = 10) -> list[dict[str, int | float]]:
    return [
        {"node": int(node), "value": float(value)}
        for node, value in sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


def compute_metrics(g: nx.DiGraph, edges: pd.DataFrame) -> dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = nx.density(g) if n_nodes > 1 else 0.0

    in_degree = dict(g.in_degree())
    out_degree = dict(g.out_degree())
    in_strength = dict(g.in_degree(weight="weight"))
    out_strength = dict(g.out_degree(weight="weight"))

    weak_components = list(nx.weakly_connected_components(g)) if n_nodes else []
    largest_component = max((len(c) for c in weak_components), default=0)

    metrics = {
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "density": float(density),
        "total_trip_count": int(edges["trip_count"].sum()) if len(edges) else 0,
        "unique_origins": int(edges["origin"].nunique()) if len(edges) else 0,
        "unique_destinations": int(edges["destination"].nunique()) if len(edges) else 0,
        "largest_weakly_connected_component": int(largest_component),
        "avg_in_degree": float(sum(in_degree.values()) / n_nodes) if n_nodes else 0.0,
        "avg_out_degree": float(sum(out_degree.values()) / n_nodes) if n_nodes else 0.0,
        "avg_in_strength": float(sum(in_strength.values()) / n_nodes) if n_nodes else 0.0,
        "avg_out_strength": float(sum(out_strength.values()) / n_nodes) if n_nodes else 0.0,
        "degree_summary": {
            "in_degree_top10": top_k({int(k): float(v) for k, v in in_degree.items()}),
            "out_degree_top10": top_k({int(k): float(v) for k, v in out_degree.items()}),
            "in_strength_top10": top_k({int(k): float(v) for k, v in in_strength.items()}),
            "out_strength_top10": top_k({int(k): float(v) for k, v in out_strength.items()}),
        },
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute baseline graph metrics from aggregated OD edges")
    parser.add_argument(
        "--input",
        default="data/processed/network/network_2023-01_edges.parquet",
        help="Input edge parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/network",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        default="network_2023-01_metrics",
        help="Output file prefix",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    edges = load_edges(input_path)
    g = build_graph(edges)
    metrics = compute_metrics(g, edges)

    metrics_path = output_dir / f"{args.prefix}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote: {metrics_path}")
    print(
        f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}, Density: {metrics['density']:.6f}, "
        f"Largest WCC: {metrics['largest_weakly_connected_component']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
