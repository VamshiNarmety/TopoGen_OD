#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hourly OD table from NYC yellow trip parquet")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output-dir", default="data/processed/od", help="Output directory")
    parser.add_argument("--prefix", default="hourly_od_2023-01", help="Output file prefix")
    parser.add_argument(
        "--start-date",
        default="2023-01-01 00:00:00",
        help="Inclusive start datetime in YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--end-date",
        default="2023-01-31 23:59:59",
        help="Inclusive end datetime in YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    needed = ["tpep_pickup_datetime", "PULocationID", "DOLocationID"]
    df = pd.read_parquet(input_path, columns=needed)

    # Remove rows with missing required fields
    df = df.dropna(subset=needed).copy()

    # Keep only valid numeric positive zone IDs
    df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
    df["DOLocationID"] = pd.to_numeric(df["DOLocationID"], errors="coerce")
    df = df.dropna(subset=["PULocationID", "DOLocationID"]).copy()
    df = df[(df["PULocationID"] > 0) & (df["DOLocationID"] > 0)].copy()

    # Hourly bucket from pickup timestamp
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime"]).copy()

    start_dt = pd.Timestamp(dt.datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S"))
    end_dt = pd.Timestamp(dt.datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S"))
    df = df[(df["tpep_pickup_datetime"] >= start_dt) & (df["tpep_pickup_datetime"] <= end_dt)].copy()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")

    # Final OD columns
    df["origin"] = df["PULocationID"].astype("int32")
    df["destination"] = df["DOLocationID"].astype("int32")

    # Aggregate to hourly OD counts
    od = (
        df.groupby(["pickup_hour", "origin", "destination"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
        .sort_values(["pickup_hour", "origin", "destination"])
        .reset_index(drop=True)
    )

    parquet_path = output_dir / f"{args.prefix}.parquet"
    csv_path = output_dir / f"{args.prefix}.csv"

    suffix = args.prefix.replace("hourly_od_", "", 1)
    summary_path = output_dir / f"od_build_summary_{suffix}.json"

    od.to_parquet(parquet_path, index=False)
    if args.write_csv:
        od.to_csv(csv_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_parquet": str(parquet_path),
        "output_csv": str(csv_path) if args.write_csv else None,
        "rows": int(len(od)),
        "unique_origins": int(od["origin"].nunique()) if len(od) else 0,
        "unique_destinations": int(od["destination"].nunique()) if len(od) else 0,
        "unique_od_pairs": int(od[["origin", "destination"]].drop_duplicates().shape[0]) if len(od) else 0,
        "trip_count_sum": int(od["trip_count"].sum()) if len(od) else 0,
        "pickup_hour_min": str(od["pickup_hour"].min()) if len(od) else None,
        "pickup_hour_max": str(od["pickup_hour"].max()) if len(od) else None,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {parquet_path}")
    if args.write_csv:
        print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(
        f"Rows: {summary['rows']}, "
        f"Unique OD pairs: {summary['unique_od_pairs']}, "
        f"Trips: {summary['trip_count_sum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())