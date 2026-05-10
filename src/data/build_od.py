#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_inputs(single_input: str | None, input_glob: str | None) -> list[Path]:
    paths: list[Path] = []
    if single_input:
        p = Path(single_input)
        if p.is_dir():
            paths.extend(sorted([x for x in p.iterdir() if x.suffix.lower() in {".parquet", ".csv"}]))
        else:
            paths.append(p)
    if input_glob:
        paths.extend(sorted(Path().glob(input_glob)))

    uniq = []
    seen = set()
    for p in paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def _provider_columns(provider: str) -> tuple[str, str, str]:
    if provider == "nyc_yellow":
        return "tpep_pickup_datetime", "PULocationID", "DOLocationID"
    if provider == "chicago_taxi":
        return "pickup_datetime", "origin", "destination"
    raise ValueError(f"Unknown provider: {provider}")


def _read_source(path: Path, pickup_col: str, origin_col: str, destination_col: str) -> pd.DataFrame:
    needed = [pickup_col, origin_col, destination_col]
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=needed)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=needed)
    raise ValueError(f"Unsupported input format for {path}. Use parquet/csv.")


def _iter_filtered_frames(
    paths: Iterable[Path],
    pickup_col: str,
    origin_col: str,
    destination_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> Iterable[pd.DataFrame]:
    for path in paths:
        df = _read_source(path, pickup_col, origin_col, destination_col)
        df = df.dropna(subset=[pickup_col, origin_col, destination_col]).copy()

        df[origin_col] = pd.to_numeric(df[origin_col], errors="coerce")
        df[destination_col] = pd.to_numeric(df[destination_col], errors="coerce")
        df = df.dropna(subset=[origin_col, destination_col]).copy()
        df = df[(df[origin_col] > 0) & (df[destination_col] > 0)].copy()

        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df = df.dropna(subset=[pickup_col]).copy()

        if start_dt is not None:
            df = df[df[pickup_col] >= start_dt].copy()
        if end_dt is not None:
            df = df[df[pickup_col] <= end_dt].copy()

        if len(df) == 0:
            continue

        df["pickup_hour"] = df[pickup_col].dt.floor("h")
        df["origin"] = df[origin_col].astype("int32")
        df["destination"] = df[destination_col].astype("int32")

        yield (
            df.groupby(["pickup_hour", "origin", "destination"], as_index=False)
            .size()
            .rename(columns={"size": "trip_count"})
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hourly OD table from NYC or Chicago trip data")
    parser.add_argument("--input", default=None, help="Single input parquet/csv path or directory")
    parser.add_argument("--input-glob", default=None, help="Glob pattern for many inputs, e.g. data/raw/trip_data/chicago/*.parquet")
    parser.add_argument("--provider", choices=["nyc_yellow", "chicago_taxi"], default="nyc_yellow", help="Input schema provider")
    parser.add_argument("--output-dir", default="data/processed/od", help="Output directory")
    parser.add_argument("--prefix", default="hourly_od", help="Output file prefix")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start datetime in YYYY-MM-DD HH:MM:SS (optional; default: full available range)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end datetime in YYYY-MM-DD HH:MM:SS (optional; default: full available range)",
    )
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV")
    args = parser.parse_args()

    input_paths = _resolve_inputs(args.input, args.input_glob)
    if not input_paths:
        raise ValueError("No input files resolved. Use --input or --input-glob.")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    pickup_col, origin_col, destination_col = _provider_columns(args.provider)

    start_dt = pd.Timestamp(dt.datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")) if args.start_date else None
    end_dt = pd.Timestamp(dt.datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S")) if args.end_date else None

    agg_chunks = list(
        _iter_filtered_frames(
            paths=input_paths,
            pickup_col=pickup_col,
            origin_col=origin_col,
            destination_col=destination_col,
            start_dt=start_dt,
            end_dt=end_dt,
        )
    )

    if not agg_chunks:
        raise ValueError("No rows left after parsing/filtering/date constraints.")

    od = pd.concat(agg_chunks, ignore_index=True)
    od = (
        od.groupby(["pickup_hour", "origin", "destination"], as_index=False)["trip_count"]
        .sum()
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
        "provider": args.provider,
        "input_count": int(len(input_paths)),
        "input_examples": [str(p) for p in input_paths[:10]],
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