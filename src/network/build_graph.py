#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_od(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"OD file not found: {input_path}")

    df = pd.read_parquet(input_path)
    required = {"pickup_hour", "origin", "destination", "trip_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OD file is missing columns: {sorted(missing)}")

    df = df.copy()
    df = df.dropna(subset=["pickup_hour", "origin", "destination", "trip_count"])
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce")
    df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
    df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce")
    df = df.dropna(subset=["origin", "destination", "trip_count"]).copy()
    df = df[(df["origin"] > 0) & (df["destination"] > 0) & (df["trip_count"] > 0)].copy()
    df["origin"] = df["origin"].astype("int32")
    df["destination"] = df["destination"].astype("int32")
    df["trip_count"] = df["trip_count"].astype("int64")
    return df


def build_edge_list(df: pd.DataFrame, by_hour: bool = False) -> pd.DataFrame:
    group_cols = ["origin", "destination"]
    if by_hour:
        group_cols = ["pickup_hour"] + group_cols

    edges = (
        df.groupby(group_cols, as_index=False)["trip_count"]
        .sum()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return edges


def build_graph(edges: pd.DataFrame, by_hour: bool = False) -> nx.DiGraph:
    g = nx.DiGraph()
    if by_hour:
        # Aggregate graph structure across time; hourly details remain in the edge table.
        edge_df = edges.groupby(["origin", "destination"], as_index=False)["trip_count"].sum()
    else:
        edge_df = edges

    for row in edge_df.itertuples(index=False):
        g.add_edge(int(row.origin), int(row.destination), weight=int(row.trip_count))
    return g


def graph_summary(g: nx.DiGraph, edges: pd.DataFrame, by_hour: bool = False) -> dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = nx.density(g) if n_nodes > 1 else 0.0

    in_degrees = dict(g.in_degree())
    out_degrees = dict(g.out_degree())
    strengths_in = dict(g.in_degree(weight="weight"))
    strengths_out = dict(g.out_degree(weight="weight"))

    weak_components = list(nx.weakly_connected_components(g)) if n_nodes else []
    largest_weak_component = max((len(c) for c in weak_components), default=0)

    top_out = sorted(strengths_out.items(), key=lambda x: x[1], reverse=True)[:10]
    top_in = sorted(strengths_in.items(), key=lambda x: x[1], reverse=True)[:10]

    summary = {
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "density": float(density),
        "by_hour": bool(by_hour),
        "total_trip_count": int(edges["trip_count"].sum()) if len(edges) else 0,
        "unique_origins": int(edges["origin"].nunique()) if len(edges) else 0,
        "unique_destinations": int(edges["destination"].nunique()) if len(edges) else 0,
        "average_in_degree": float(sum(in_degrees.values()) / n_nodes) if n_nodes else 0.0,
        "average_out_degree": float(sum(out_degrees.values()) / n_nodes) if n_nodes else 0.0,
        "average_in_strength": float(sum(strengths_in.values()) / n_nodes) if n_nodes else 0.0,
        "average_out_strength": float(sum(strengths_out.values()) / n_nodes) if n_nodes else 0.0,
        "largest_weakly_connected_component": int(largest_weak_component),
        "top_10_out_strength": [{"node": int(node), "strength": int(val)} for node, val in top_out],
        "top_10_in_strength": [{"node": int(node), "strength": int(val)} for node, val in top_in],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a directed transport graph from OD data")
    parser.add_argument(
        "--input",
        default="data/processed/od/hourly_od_2023-01_local.parquet",
        help="Input OD parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/network",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        default="network_2023-01",
        help="Output file prefix",
    )
    parser.add_argument(
        "--by-hour",
        action="store_true",
        help="Keep hourly rows in the edge table and aggregate graph across all hours",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = load_od(input_path)
    edges = build_edge_list(df, by_hour=args.by_hour)
    g = build_graph(edges, by_hour=args.by_hour)
    summary = graph_summary(g, edges, by_hour=args.by_hour)

    edges_path = output_dir / f"{args.prefix}_edges.parquet"
    summary_path = output_dir / f"{args.prefix}_summary.json"
    graphml_path = output_dir / f"{args.prefix}.graphml"

    edges.to_parquet(edges_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    nx.write_graphml(g, graphml_path)

    print(f"Wrote: {edges_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {graphml_path}")
    print(
        f"Nodes: {summary['nodes']}, Edges: {summary['edges']}, "
        f"Trip count: {summary['total_trip_count']}, Density: {summary['density']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
