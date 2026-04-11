#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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


def build_undirected_graph(edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for row in edges.itertuples(index=False):
        u = int(row.origin)
        v = int(row.destination)
        w = int(row.trip_count)
        if g.has_edge(u, v):
            g[u][v]["weight"] += w
        else:
            g.add_edge(u, v, weight=w)
    return g


def detect_communities(g: nx.Graph):
    """Try Louvain first; fallback to greedy modularity communities."""
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(g, weight="weight", seed=42)
        method = "louvain"
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(g, weight="weight"))
        method = "greedy_modularity"
    return communities, method


def membership_table(communities: Iterable[set[int]]) -> pd.DataFrame:
    rows = []
    for cid, nodes in enumerate(communities):
        for node in nodes:
            rows.append({"node": int(node), "community_id": int(cid)})
    return pd.DataFrame(rows).sort_values(["community_id", "node"]).reset_index(drop=True)


def summarize_communities(g: nx.Graph, communities: list[set[int]], method: str) -> dict:
    if len(communities) == 0:
        return {
            "method": method,
            "num_communities": 0,
            "modularity": None,
            "largest_community_size": 0,
            "smallest_community_size": 0,
            "community_sizes": [],
        }

    try:
        modularity = nx.algorithms.community.modularity(g, communities, weight="weight")
    except Exception:
        modularity = None

    sizes = sorted([len(c) for c in communities], reverse=True)
    return {
        "method": method,
        "num_communities": int(len(communities)),
        "modularity": float(modularity) if modularity is not None else None,
        "largest_community_size": int(max(sizes)),
        "smallest_community_size": int(min(sizes)),
        "community_sizes": [int(x) for x in sizes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect communities in the OD transport network")
    parser.add_argument(
        "--input",
        default="data/processed/network/network_2023-01_clean_edges.parquet",
        help="Input edge parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/network/community",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        default="community_2023-01",
        help="Output file prefix",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    edges = load_edges(input_path)
    g = build_undirected_graph(edges)
    communities, method = detect_communities(g)
    membership = membership_table(communities)
    summary = summarize_communities(g, communities, method)
    summary["nodes"] = int(g.number_of_nodes())
    summary["edges"] = int(g.number_of_edges())

    membership_path = output_dir / f"{args.prefix}_membership.parquet"
    summary_path = output_dir / f"{args.prefix}_summary.json"

    membership.to_parquet(membership_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {membership_path}")
    print(f"Wrote: {summary_path}")
    print(
        f"Method: {summary['method']}, Communities: {summary['num_communities']}, "
        f"Modularity: {summary['modularity']}, Largest size: {summary['largest_community_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
