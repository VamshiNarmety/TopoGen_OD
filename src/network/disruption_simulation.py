#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    items = [x.strip() for x in value.split(",") if x.strip()]
    out = [int(x) for x in items]
    if any(x <= 0 for x in out):
        raise ValueError("All k values must be positive integers")
    return sorted(set(out))


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
        u = int(row.origin)
        v = int(row.destination)
        w = int(row.trip_count)
        if g.has_edge(u, v):
            g[u][v]["weight"] += w
        else:
            g.add_edge(u, v, weight=w)
    return g


def largest_weak_component_subgraph(g: nx.DiGraph) -> nx.DiGraph:
    if g.number_of_nodes() == 0:
        return g.copy()
    wccs = list(nx.weakly_connected_components(g))
    if not wccs:
        return g.copy()
    largest_nodes = max(wccs, key=len)
    return g.subgraph(largest_nodes).copy()


def graph_metrics(g: nx.DiGraph, baseline_trip_count: int, scenario_id: str, scenario_type: str) -> dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    total_trip_count = int(sum(data.get("weight", 0) for _, _, data in g.edges(data=True)))

    if n_nodes > 1:
        density = float(nx.density(g))
    else:
        density = 0.0

    wccs = list(nx.weakly_connected_components(g)) if n_nodes else []
    largest_wcc = max((len(c) for c in wccs), default=0)
    num_wcc = len(wccs)

    undirected = g.to_undirected()
    if undirected.number_of_nodes() > 1 and undirected.number_of_edges() > 0:
        global_eff = float(nx.global_efficiency(undirected))
    else:
        global_eff = 0.0

    lwc_sub = largest_weak_component_subgraph(g)
    lwc_undirected = lwc_sub.to_undirected()
    if lwc_undirected.number_of_nodes() > 1 and lwc_undirected.number_of_edges() > 0:
        try:
            avg_shortest_path_lwc = float(nx.average_shortest_path_length(lwc_undirected))
        except Exception:
            avg_shortest_path_lwc = None
    else:
        avg_shortest_path_lwc = None

    flow_retention_ratio = (
        float(total_trip_count / baseline_trip_count)
        if baseline_trip_count > 0
        else 0.0
    )

    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "density": density,
        "total_trip_count": int(total_trip_count),
        "flow_retention_ratio": flow_retention_ratio,
        "num_weakly_connected_components": int(num_wcc),
        "largest_weakly_connected_component": int(largest_wcc),
        "global_efficiency_undirected": global_eff,
        "avg_shortest_path_lwc_undirected": avg_shortest_path_lwc,
    }


def load_target_order_from_centrality(path: Path, measure: str) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Centrality file not found: {path}")

    df = pd.read_parquet(path)
    required = {"node_id", measure}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Centrality file missing columns for measure '{measure}': {sorted(missing)}"
        )

    ranked = (
        df[["node_id", measure]]
        .dropna()
        .sort_values(measure, ascending=False)
        .drop_duplicates(subset=["node_id"])
    )
    return [int(x) for x in ranked["node_id"].tolist()]


def targeted_node_removal(g: nx.DiGraph, target_nodes: list[int]) -> nx.DiGraph:
    g2 = g.copy()
    g2.remove_nodes_from([n for n in target_nodes if n in g2])
    return g2


def random_node_removal(g: nx.DiGraph, k: int, seed: int) -> tuple[nx.DiGraph, list[int]]:
    nodes = sorted(g.nodes())
    if not nodes:
        return g.copy(), []

    k_eff = min(k, len(nodes))
    sampled = (
        pd.Series(nodes)
        .sample(n=k_eff, random_state=seed, replace=False)
        .astype(int)
        .tolist()
    )
    g2 = g.copy()
    g2.remove_nodes_from(sampled)
    return g2, sampled


def targeted_edge_removal(g: nx.DiGraph, k: int) -> tuple[nx.DiGraph, list[tuple[int, int]]]:
    g2 = g.copy()
    weighted_edges = sorted(
        g2.edges(data=True),
        key=lambda x: x[2].get("weight", 0),
        reverse=True,
    )
    selected = [(int(u), int(v)) for u, v, _ in weighted_edges[: min(k, len(weighted_edges))]]
    g2.remove_edges_from(selected)
    return g2, selected


def save_scenario_outputs(
    output_dir: Path,
    prefix: str,
    scenario_id: str,
    graph_after: nx.DiGraph,
    metrics: dict,
) -> None:
    edges_after = []
    for u, v, data in graph_after.edges(data=True):
        edges_after.append(
            {
                "origin": int(u),
                "destination": int(v),
                "trip_count": int(data.get("weight", 0)),
            }
        )

    edges_df = pd.DataFrame(edges_after, columns=["origin", "destination", "trip_count"])
    edges_path = output_dir / f"{prefix}_{scenario_id}_edges.parquet"
    metrics_path = output_dir / f"{prefix}_{scenario_id}_metrics.json"

    edges_df.to_parquet(edges_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    k_target_nodes = parse_int_list(args.targeted_node_k)
    k_random_nodes = parse_int_list(args.random_node_k)
    k_target_edges = parse_int_list(args.targeted_edge_k)

    edges = load_edges(input_path)
    g_base = build_graph(edges)

    baseline_trip_count = int(edges["trip_count"].sum())
    baseline_metrics = graph_metrics(
        g=g_base,
        baseline_trip_count=baseline_trip_count,
        scenario_id="baseline",
        scenario_type="baseline",
    )
    baseline_metrics["removed_nodes"] = 0
    baseline_metrics["removed_edges"] = 0
    baseline_metrics["notes"] = "Reference graph before disruption"

    all_metrics: list[dict] = [baseline_metrics]
    save_scenario_outputs(
        output_dir=output_dir,
        prefix=args.prefix,
        scenario_id="baseline",
        graph_after=g_base,
        metrics=baseline_metrics,
    )

    # 1) Targeted node removal (centrality-based)
    if k_target_nodes:
        centrality_path = Path(args.centrality)
        target_order = load_target_order_from_centrality(centrality_path, args.centrality_measure)

        for k in k_target_nodes:
            targets = target_order[: min(k, len(target_order))]
            scenario_id = f"targeted_node_{args.centrality_measure}_k{k}"
            g_after = targeted_node_removal(g_base, targets)
            m = graph_metrics(
                g=g_after,
                baseline_trip_count=baseline_trip_count,
                scenario_id=scenario_id,
                scenario_type="targeted_node_removal",
            )
            m["removed_nodes"] = int(len(targets))
            m["removed_edges"] = int(g_base.number_of_edges() - g_after.number_of_edges())
            m["centrality_measure"] = args.centrality_measure
            m["removed_node_ids"] = [int(x) for x in targets]
            all_metrics.append(m)
            save_scenario_outputs(output_dir, args.prefix, scenario_id, g_after, m)

    # 2) Random node removal (repeatable by seed)
    if k_random_nodes and args.random_repeats > 0:
        for k in k_random_nodes:
            for rep in range(args.random_repeats):
                seed = args.seed + rep
                scenario_id = f"random_node_k{k}_seed{seed}"
                g_after, removed_nodes = random_node_removal(g_base, k=k, seed=seed)
                m = graph_metrics(
                    g=g_after,
                    baseline_trip_count=baseline_trip_count,
                    scenario_id=scenario_id,
                    scenario_type="random_node_removal",
                )
                m["removed_nodes"] = int(len(removed_nodes))
                m["removed_edges"] = int(g_base.number_of_edges() - g_after.number_of_edges())
                m["seed"] = int(seed)
                m["removed_node_ids"] = [int(x) for x in removed_nodes]
                all_metrics.append(m)
                save_scenario_outputs(output_dir, args.prefix, scenario_id, g_after, m)

    # 3) Targeted edge removal (highest weighted edges)
    if k_target_edges:
        for k in k_target_edges:
            scenario_id = f"targeted_edge_weight_k{k}"
            g_after, removed_edges = targeted_edge_removal(g_base, k=k)
            m = graph_metrics(
                g=g_after,
                baseline_trip_count=baseline_trip_count,
                scenario_id=scenario_id,
                scenario_type="targeted_edge_removal",
            )
            m["removed_nodes"] = int(g_base.number_of_nodes() - g_after.number_of_nodes())
            m["removed_edges"] = int(len(removed_edges))
            m["removed_edge_pairs"] = [[int(u), int(v)] for u, v in removed_edges]
            all_metrics.append(m)
            save_scenario_outputs(output_dir, args.prefix, scenario_id, g_after, m)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values(["scenario_type", "scenario_id"]).reset_index(drop=True)

    comparison_parquet = output_dir / f"{args.prefix}_comparison.parquet"
    comparison_csv = output_dir / f"{args.prefix}_comparison.csv"
    comparison_json = output_dir / f"{args.prefix}_comparison.json"

    comparison_df.to_parquet(comparison_parquet, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    run_meta = {
        "input_edges": str(input_path),
        "output_dir": str(output_dir),
        "prefix": args.prefix,
        "centrality_file": str(args.centrality),
        "centrality_measure": args.centrality_measure,
        "targeted_node_k": k_target_nodes,
        "random_node_k": k_random_nodes,
        "random_repeats": int(args.random_repeats),
        "targeted_edge_k": k_target_edges,
        "base_seed": int(args.seed),
        "num_scenarios": int(len(all_metrics)),
    }
    with open(output_dir / f"{args.prefix}_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Wrote: {comparison_parquet}")
    print(f"Wrote: {comparison_csv}")
    print(f"Wrote: {comparison_json}")
    print(f"Wrote: {output_dir / f'{args.prefix}_run_config.json'}")
    print(f"Scenarios generated: {len(all_metrics)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate network disruptions and save paper-ready scenario comparisons"
    )
    parser.add_argument(
        "--input",
        default="data/processed/network/network_2023-01_clean_edges.parquet",
        help="Input edge parquet (origin, destination, trip_count)",
    )
    parser.add_argument(
        "--centrality",
        default="data/processed/network/centrality/centrality_2023-01_centrality.parquet",
        help="Centrality parquet used for targeted node removal",
    )
    parser.add_argument(
        "--centrality-measure",
        default="betweenness_centrality",
        help="Column in centrality parquet to rank targeted node removals",
    )
    parser.add_argument(
        "--targeted-node-k",
        default="5,10,20",
        help="Comma-separated node counts for targeted node removal",
    )
    parser.add_argument(
        "--random-node-k",
        default="5,10,20",
        help="Comma-separated node counts for random node removal",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=3,
        help="Number of random repeats per k",
    )
    parser.add_argument(
        "--targeted-edge-k",
        default="100,500",
        help="Comma-separated edge counts for top-weight edge removal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible random removals",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/network/disruption",
        help="Output directory for scenario files",
    )
    parser.add_argument(
        "--prefix",
        default="disruption_2023-01_clean",
        help="Prefix used in all generated outputs for easy comparison",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
