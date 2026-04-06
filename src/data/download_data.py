#!/usr/bin/env python3
"""
Full data downloader for TopoGen-OD (NYC-focused).

What it downloads:
1) TLC trip data (yellow/green/fhv) for a month range
2) NYC taxi zones (zip + optional auto-unzip)
3) Taxi zone lookup CSV
4) Optional OSM road network (NYC) as GraphML

Examples:
  python src/data/download_data.py --dataset yellow --start 2023-01 --end 2023-03
  python src/data/download_data.py --dataset all --start 2023-01 --end 2023-01 --with-osm
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from tqdm import tqdm

BASE = "https://d37ci6vzurychx.cloudfront.net"
TRIP_BASE = f"{BASE}/trip-data"
MISC_BASE = f"{BASE}/misc"


def month_range(start_ym: str, end_ym: str) -> List[str]:
    start = dt.datetime.strptime(start_ym, "%Y-%m")
    end = dt.datetime.strptime(end_ym, "%Y-%m")
    if end < start:
        raise ValueError("end month must be >= start month")

    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        year = cur.year + (cur.month // 12)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = dt.datetime(year, month, 1)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stream_download(url: str, out_path: Path, timeout: int = 120, retries: int = 3) -> Tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "already_exists"

    ensure_dir(out_path.parent)

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    return False, f"http_{r.status_code}"

                total = int(r.headers.get("content-length", 0))
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")

                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=out_path.name,
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                tmp_path.replace(out_path)
                return True, "downloaded"

        except Exception as e:
            if attempt == retries:
                return False, f"error_{type(e).__name__}:{e}"
            sleep_s = 2 ** attempt
            time.sleep(sleep_s)

    return False, "unknown_error"


def unzip_file(zip_path: Path, out_dir: Path) -> Tuple[bool, str]:
    try:
        ensure_dir(out_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        return True, "unzipped"
    except Exception as e:
        return False, f"unzip_error_{type(e).__name__}:{e}"


def build_trip_urls(dataset: str, months: Iterable[str]) -> Dict[str, str]:
    urls = {}
    if dataset == "all":
        datasets = ["yellow", "green", "fhv"]
    else:
        datasets = [dataset]

    for ds in datasets:
        for ym in months:
            fname = f"{ds}_tripdata_{ym}.parquet"
            urls[f"trip_data/{ds}/{fname}"] = f"{TRIP_BASE}/{fname}"
    return urls


def build_misc_urls() -> Dict[str, str]:
    return {
        "misc/taxi_zones.zip": f"{MISC_BASE}/taxi_zones.zip",
        "misc/taxi_zone_lookup.csv": f"{MISC_BASE}/taxi_zone_lookup.csv",
    }


def try_download_osm(output_root: Path, place: str) -> Tuple[bool, str]:
    """
    Optional OSM download via osmnx.
    Saved to: data/raw/osm/nyc_drive.graphml
    """
    try:
        import osmnx as ox  # optional dependency

        osm_dir = output_root / "osm"
        ensure_dir(osm_dir)
        graph_path = osm_dir / "nyc_drive.graphml"

        if graph_path.exists() and graph_path.stat().st_size > 0:
            return True, "already_exists"

        G = ox.graph_from_place(place, network_type="drive")
        ox.save_graphml(G, graph_path)
        return True, "downloaded"
    except Exception as e:
        return False, f"osm_error_{type(e).__name__}:{e}"


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TopoGen-OD data resources")
    parser.add_argument("--output-dir", default="data/raw", help="Root output folder")
    parser.add_argument("--dataset", choices=["yellow", "green", "fhv", "all"], default="yellow")
    parser.add_argument("--start", default="2023-01", help="Start month YYYY-MM")
    parser.add_argument("--end", default="2023-03", help="End month YYYY-MM")
    parser.add_argument("--with-osm", action="store_true", help="Also download NYC OSM graph")
    parser.add_argument("--osm-place", default="New York City, New York, USA", help="OSM place name")
    parser.add_argument("--unzip", action="store_true", help="Unzip taxi_zones.zip after download")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    months = month_range(args.start, args.end)
    trip_urls = build_trip_urls(args.dataset, months)
    misc_urls = build_misc_urls()

    all_urls = {}
    all_urls.update(trip_urls)
    all_urls.update(misc_urls)

    manifest = {
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "config": {
            "dataset": args.dataset,
            "start": args.start,
            "end": args.end,
            "with_osm": args.with_osm,
            "osm_place": args.osm_place,
            "unzip": args.unzip,
        },
        "results": {},
    }

    print(f"\nDownloading {len(all_urls)} files into: {output_root}\n")

    for rel_path, url in all_urls.items():
        out_path = output_root / rel_path
        ok, status = stream_download(url, out_path)
        manifest["results"][rel_path] = {"url": url, "ok": ok, "status": status, "path": str(out_path)}
        print(f"[{'OK' if ok else 'FAIL'}] {rel_path} -> {status}")

        if ok and args.unzip and rel_path.endswith(".zip"):
            unzip_dir = out_path.parent / out_path.stem
            z_ok, z_status = unzip_file(out_path, unzip_dir)
            manifest["results"][f"{rel_path}#unzip"] = {
                "ok": z_ok,
                "status": z_status,
                "path": str(unzip_dir),
            }
            print(f"   [{'OK' if z_ok else 'FAIL'}] unzip -> {z_status}")

    if args.with_osm:
        print("\nDownloading OSM road network (optional)...")
        ok, status = try_download_osm(output_root, args.osm_place)
        manifest["results"]["osm/nyc_drive.graphml"] = {
            "ok": ok,
            "status": status,
            "path": str(output_root / "osm" / "nyc_drive.graphml"),
        }
        print(f"[{'OK' if ok else 'FAIL'}] osm/nyc_drive.graphml -> {status}")

    manifest_path = output_root / "download_manifest.json"
    write_manifest(manifest_path, manifest)

    failed = [k for k, v in manifest["results"].items() if not v.get("ok", False)]
    print(f"\nManifest written: {manifest_path}")
    print(f"Done. Failed items: {len(failed)}")
    if failed:
        print("Failed keys:")
        for k in failed:
            print(f"  - {k}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())