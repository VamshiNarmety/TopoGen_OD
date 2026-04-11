#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_k(scenario_id: str) -> int | None:
    m = re.search(r"_k(\d+)", scenario_id)
    return int(m.group(1)) if m else None


def load_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Comparison file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {
        "scenario_id",
        "scenario_type",
        "flow_retention_ratio",
        "largest_weakly_connected_component",
        "global_efficiency_undirected",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Comparison file missing columns: {sorted(missing)}")

    df = df.copy()
    df["k"] = df["scenario_id"].astype(str).apply(extract_k)

    def category(row: pd.Series) -> str:
        sid = str(row["scenario_id"])
        stype = str(row["scenario_type"])
        if stype == "baseline":
            return "baseline"
        if stype == "targeted_node_removal":
            return "targeted_node"
        if stype == "random_node_removal":
            return "random_node"
        if stype == "targeted_edge_removal":
            return "targeted_edge"
        if "random_node" in sid:
            return "random_node"
        if "targeted_node" in sid:
            return "targeted_node"
        if "targeted_edge" in sid:
            return "targeted_edge"
        return "other"

    df["scenario_category"] = df.apply(category, axis=1)
    return df


def aggregate_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "flow_retention_ratio",
        "largest_weakly_connected_component",
        "global_efficiency_undirected",
    ]

    grouped = (
        df[df["scenario_category"].isin(["targeted_node", "random_node", "targeted_edge"])]
        .groupby(["scenario_category", "k"], dropna=True)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def save_line_plot(
    ax: plt.Axes,
    baseline_value: float,
    random_df: pd.DataFrame,
    targeted_df: pd.DataFrame,
    x_col: str,
    random_y_col: str,
    targeted_y_col: str,
    y_label: str,
    title: str,
) -> None:
    if baseline_value is not None:
        ax.axhline(baseline_value, linestyle="--", linewidth=1.2, label="baseline")

    if len(random_df):
        random_df = random_df.sort_values(x_col)
        ax.plot(random_df[x_col], random_df[random_y_col], marker="o", label="random node (mean)")
        std_col = f"{random_y_col}_std"
        if std_col in random_df.columns:
            y = random_df[random_y_col]
            s = random_df[std_col].fillna(0)
            ax.fill_between(random_df[x_col], y - s, y + s, alpha=0.2)

    if len(targeted_df):
        targeted_df = targeted_df.sort_values(x_col)
        ax.plot(targeted_df[x_col], targeted_df[targeted_y_col], marker="s", label="targeted node")

    ax.set_xlabel("k removed")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()


def plot_all(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)

    baseline_df = df[df["scenario_category"] == "baseline"]
    baseline_row = baseline_df.iloc[0] if len(baseline_df) else None

    random_node = df[df["scenario_category"] == "random_node"].copy()
    targeted_node = df[df["scenario_category"] == "targeted_node"].copy()
    targeted_edge = df[df["scenario_category"] == "targeted_edge"].copy()

    random_mean = (
        random_node.groupby("k", dropna=True)[
            ["flow_retention_ratio", "largest_weakly_connected_component", "global_efficiency_undirected"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    random_mean.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_")
        for col in random_mean.columns.to_flat_index()
    ]

    targeted_node_mean = (
        targeted_node.groupby("k", dropna=True)[
            ["flow_retention_ratio", "largest_weakly_connected_component", "global_efficiency_undirected"]
        ]
        .mean()
        .reset_index()
    )

    # 1) Node-removal comparison (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    save_line_plot(
        ax=axes[0],
        baseline_value=float(baseline_row["flow_retention_ratio"]) if baseline_row is not None else None,
        random_df=random_mean,
        targeted_df=targeted_node_mean,
        x_col="k",
        random_y_col="flow_retention_ratio_mean",
        targeted_y_col="flow_retention_ratio",
        y_label="Flow retention ratio",
        title="Flow Retention vs Node Removal",
    )

    save_line_plot(
        ax=axes[1],
        baseline_value=float(baseline_row["largest_weakly_connected_component"]) if baseline_row is not None else None,
        random_df=random_mean,
        targeted_df=targeted_node_mean,
        x_col="k",
        random_y_col="largest_weakly_connected_component_mean",
        targeted_y_col="largest_weakly_connected_component",
        y_label="Largest WCC size",
        title="Connectivity vs Node Removal",
    )

    save_line_plot(
        ax=axes[2],
        baseline_value=float(baseline_row["global_efficiency_undirected"]) if baseline_row is not None else None,
        random_df=random_mean,
        targeted_df=targeted_node_mean,
        x_col="k",
        random_y_col="global_efficiency_undirected_mean",
        targeted_y_col="global_efficiency_undirected",
        y_label="Global efficiency",
        title="Efficiency vs Node Removal",
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_node_removal_comparison.png", dpi=200)
    plt.close(fig)

    # 2) Targeted edge removal chart (if present)
    if len(targeted_edge):
        edge_sorted = targeted_edge.sort_values("k")
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
        ax2.plot(
            edge_sorted["k"],
            edge_sorted["flow_retention_ratio"],
            marker="o",
            label="targeted edge removal",
        )
        if baseline_row is not None:
            ax2.axhline(
                float(baseline_row["flow_retention_ratio"]),
                linestyle="--",
                linewidth=1.2,
                label="baseline",
            )
        ax2.set_xlabel("k edges removed")
        ax2.set_ylabel("Flow retention ratio")
        ax2.set_title("Flow Retention vs Targeted Edge Removal")
        ax2.grid(alpha=0.25)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{prefix}_edge_removal_flow.png", dpi=200)
        plt.close(fig2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready plots from disruption comparison table")
    parser.add_argument(
        "--input",
        default="data/processed/network/disruption/disruption_2023-01_clean_comparison.csv",
        help="Input comparison file (.csv or .parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/network/disruption/visualizations",
        help="Output directory for figures and aggregated table",
    )
    parser.add_argument(
        "--prefix",
        default="disruption_2023-01_clean",
        help="Prefix for generated figure/table files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    df = load_comparison(input_path)
    agg = aggregate_for_paper(df)

    agg_csv = out_dir / f"{args.prefix}_paper_aggregate.csv"
    agg_parquet = out_dir / f"{args.prefix}_paper_aggregate.parquet"
    agg.to_csv(agg_csv, index=False)
    agg.to_parquet(agg_parquet, index=False)

    plot_all(df, out_dir, args.prefix)

    print(f"Wrote: {agg_csv}")
    print(f"Wrote: {agg_parquet}")
    print(f"Wrote: {out_dir / f'{args.prefix}_node_removal_comparison.png'}")
    edge_fig = out_dir / f"{args.prefix}_edge_removal_flow.png"
    if edge_fig.exists():
        print(f"Wrote: {edge_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
