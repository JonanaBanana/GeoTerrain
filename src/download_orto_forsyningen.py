"""
Download orthophoto imagery from dataforsyningen.dk WMS API.

Fetches a 10×10 km tile as 100 × 1×1 km² GeoTIFF subtiles at native 12.5 cm/px
resolution.  Each subtile is saved as a georeferenced GeoTIFF (EPSG:25832)
so it can be used directly with stitch_texture.py.

Config
------
Add  forsyningen_token: YOUR_TOKEN  to  config/api_key.yaml.
Get a free token at https://dataforsyningen.dk.

Folder layout
-------------
inputs/tile_<N>_<E>/
    satellite_forsyningen/
        <year>/
            orto_<year>_1km_<northing_km>_<easting_km>.tif
            ...
            metadata.csv

Each year folder is a drop-in for stitch_texture.py:

    python3 src/stitch_texture.py \\
        inputs/tile_623_57/satellite_forsyningen/2024 --out 623_57_2024

Usage
-----
python3 src/download_orto_forsyningen.py --tile 623,57
python3 src/download_orto_forsyningen.py --tile 623,57 --years 2024 2023
python3 src/download_orto_forsyningen.py --tile 623,57 --workers 8
python3 src/download_orto_forsyningen.py --tile 623,57 --force
"""

import argparse
import concurrent.futures
import csv
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import requests
import yaml
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


ROOT            = Path(__file__).parent.parent
CONF            = ROOT / "config" / "api_key.yaml"
INPUTS          = ROOT / "inputs"

WMS_ENDPOINT    = "https://api.dataforsyningen.dk/orto_foraar_DAF"
TILE_SIZE_M     = 1_000          # metres per subtile side
PIXELS_PER_TILE = 4_000          # → 0.125 m/px  (12.5 cm native resolution)
GSD_M           = TILE_SIZE_M / PIXELS_PER_TILE   # 0.125
SUBTILE_GRID    = 10             # 10×10 subtiles per 10 km tile

# Each 1 km subtile is split into CHUNK_GRID×CHUNK_GRID WMS requests to keep
# individual image dimensions small enough to avoid server-side gateway timeouts.
# 4×4 → sixteen 2000×2000 px requests of 250 m × 250 m each.
CHUNK_GRID      = 4
CHUNK_SIZE_M    = TILE_SIZE_M    // CHUNK_GRID   # 250 m per chunk
CHUNK_PX        = PIXELS_PER_TILE // CHUNK_GRID  # 2000 px per chunk

# Available orthophoto years on dataforsyningen.dk (geodanmark series).
# Extend this list as new years become available.
AVAILABLE_YEARS = list(range(2021, 2026))   # 2021–2025


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_token() -> str:
    """Read forsyningen_token from config/api_key.yaml, exit with hint on failure."""
    if not CONF.exists():
        print(f"Config not found: {CONF}")
        print("  cp config/example_api_key.yaml config/api_key.yaml")
        print("  Then add:  forsyningen_token: YOUR_TOKEN")
        print("  Get a token at: https://dataforsyningen.dk")
        sys.exit(1)
    with open(CONF) as f:
        cfg = yaml.safe_load(f) or {}
    token = cfg.get("forsyningen_token", "").strip()
    if not token or token.lower() in ("your_token_here", "xxxx", ""):
        print(f"Missing or placeholder forsyningen_token in {CONF}")
        print("  Add:  forsyningen_token: YOUR_TOKEN")
        print("  Get a token at: https://dataforsyningen.dk")
        sys.exit(1)
    return token


# ══════════════════════════════════════════════════════════════════════════════
# WMS URL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def layer_name(year: int) -> str:
    return f"geodanmark_{year}_12_5cm"


def build_wms_url(
    west: int, south: int, east: int, north: int,
    width_px: int, height_px: int,
    year: int, token: str,
) -> str:
    """
    Build a WMS 1.1.1 GetMap URL for an arbitrary BBOX.

    BBOX order for EPSG:25832 is west,south,east,north (easting,northing).
    TRANSPARENT=FALSE ensures a 3-band RGB response with no alpha channel.
    """
    bbox = f"{west},{south},{east},{north}"
    return (
        f"{WMS_ENDPOINT}"
        f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&FORMAT=image%2Fpng"
        f"&TRANSPARENT=FALSE"
        f"&LAYERS={layer_name(year)}"
        f"&STYLES="
        f"&WIDTH={width_px}&HEIGHT={height_px}"
        f"&SRS=EPSG%3A25832"
        f"&BBOX={bbox}"
        f"&token={token}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD + GEOTIFF CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_with_retry(url: str, retries: int = 3) -> bytes:
    """GET url, return raw bytes. Retries on transient network errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "image" not in ct:
                # WMS returns an XML service exception on bad requests.
                raise ValueError(
                    f"WMS returned non-image content ({ct}):\n"
                    f"{resp.text[:400]}"
                )
            return resp.content
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def download_subtile(
    northing_km: int,
    easting_km: int,
    year: int,
    token: str,
    out_path: Path,
) -> dict:
    """
    Download one 1 km² subtile from the WMS and write it as a GeoTIFF.

    The 1 km tile is fetched as a CHUNK_GRID×CHUNK_GRID grid of smaller WMS
    requests (currently 4×4 = sixteen 250 m × 2000 px chunks) and stitched
    into the final PIXELS_PER_TILE×PIXELS_PER_TILE array before writing.
    This keeps individual requests small enough to avoid server-side gateway
    timeouts that occur with a single large 8000×8000 px request.

    The GeoTIFF embeds:
    - CRS: EPSG:25832 (UTM zone 32N)
    - Affine transform derived directly from the tile BBOX, guaranteeing
      sub-pixel spatial accuracy.
    - GDAL metadata tags: YEAR, LAYER, GSD_M, SOURCE, DOWNLOADED_UTC.

    Returns a metadata dict suitable for the CSV row.
    """
    west  = easting_km  * TILE_SIZE_M
    south = northing_km * TILE_SIZE_M
    east  = west  + TILE_SIZE_M
    north = south + TILE_SIZE_M

    # Fetch CHUNK_GRID×CHUNK_GRID chunks and stitch into one array.
    # Row 0 in the image is the northernmost strip; ri=0 → top of tile.
    full = np.zeros((PIXELS_PER_TILE, PIXELS_PER_TILE, 3), dtype=np.uint8)
    for ri in range(CHUNK_GRID):
        for ci in range(CHUNK_GRID):
            c_west  = west  + ci * CHUNK_SIZE_M
            c_east  = c_west + CHUNK_SIZE_M
            c_north = north - ri * CHUNK_SIZE_M
            c_south = c_north - CHUNK_SIZE_M
            url = build_wms_url(
                c_west, c_south, c_east, c_north,
                CHUNK_PX, CHUNK_PX,
                year, token,
            )
            png_bytes = _fetch_with_retry(url)
            chunk = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            if chunk.shape[:2] != (CHUNK_PX, CHUNK_PX):
                raise ValueError(
                    f"Chunk ({ri},{ci}): WMS returned {chunk.shape[1]}×{chunk.shape[0]} px, "
                    f"expected {CHUNK_PX}×{CHUNK_PX}"
                )
            r0 = ri * CHUNK_PX
            c0 = ci * CHUNK_PX
            full[r0:r0 + CHUNK_PX, c0:c0 + CHUNK_PX] = chunk

    H, W = full.shape[:2]   # PIXELS_PER_TILE × PIXELS_PER_TILE

    # from_bounds builds a north-up transform: origin at (west, north),
    # pixel_width=+GSD_M, pixel_height=-GSD_M.
    transform = from_bounds(west, south, east, north, W, H)
    crs       = CRS.from_epsg(25832)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    try:
        with rasterio.open(
            tmp_path, "w",
            driver   = "GTiff",
            height   = H,
            width    = W,
            count    = 3,
            dtype    = "uint8",
            crs      = crs,
            transform= transform,
            compress = "deflate",
            predictor= 2,
            zlevel   = 6,
        ) as dst:
            dst.write(full[:, :, 0], 1)   # R
            dst.write(full[:, :, 1], 2)   # G
            dst.write(full[:, :, 2], 3)   # B
            dst.update_tags(
                YEAR           = str(year),
                LAYER          = layer_name(year),
                GSD_M          = str(GSD_M),
                SOURCE         = "dataforsyningen.dk",
                DOWNLOADED_UTC = datetime.now(timezone.utc).isoformat(),
            )
        tmp_path.replace(out_path)   # atomic rename — only reaches here on success
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    size_mb = out_path.stat().st_size / 1_048_576
    return {
        "filename":       out_path.name,
        "year":           year,
        "northing_km":    northing_km,
        "easting_km":     easting_km,
        "bbox_west_m":    west,
        "bbox_south_m":   south,
        "bbox_east_m":    east,
        "bbox_north_m":   north,
        "crs_epsg":       25832,
        "width_px":       W,
        "height_px":      H,
        "gsd_m":          GSD_M,
        "layer":          layer_name(year),
        "size_mb":        round(size_mb, 2),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METADATA CSV
# ══════════════════════════════════════════════════════════════════════════════

def update_metadata_csv(year_dir: Path, new_rows: list[dict]):
    """
    Merge new_rows into the existing metadata.csv (if any), then write it back.
    Existing rows for the same (northing_km, easting_km) are replaced.
    """
    csv_path = year_dir / "metadata.csv"

    existing: dict[tuple, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["northing_km"], row["easting_km"])
                existing[key] = row

    for row in new_rows:
        key = (str(row["northing_km"]), str(row["easting_km"]))
        existing[key] = {k: str(v) for k, v in row.items()}

    if not existing:
        return

    all_rows = sorted(
        existing.values(),
        key=lambda r: (int(r["northing_km"]), int(r["easting_km"])),
    )
    fieldnames = list(next(iter(existing.values())).keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Metadata → {csv_path}  ({len(all_rows)} tiles)")


# ══════════════════════════════════════════════════════════════════════════════
# YEAR ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def download_year(
    n: int,
    e: int,
    year: int,
    token: str,
    force: bool,
    n_workers: int,
) -> tuple[int, int, int]:
    """
    Download all 100 subtiles for one tile+year.
    Returns (downloaded, skipped, failed) counts.
    """
    year_dir = INPUTS / f"tile_{n}_{e}" / "satellite_forsyningen" / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    subtiles = [
        (n * SUBTILE_GRID + ni, e * SUBTILE_GRID + ei)
        for ni in range(SUBTILE_GRID)
        for ei in range(SUBTILE_GRID)
    ]

    to_download, n_skip = [], 0
    for northing_km, easting_km in subtiles:
        fname = f"orto_{year}_1km_{northing_km}_{easting_km}.tif"
        path  = year_dir / fname
        path.with_suffix(".tmp").unlink(missing_ok=True)   # clean up any interrupted write
        if not force and path.exists() and path.stat().st_size > 100_000:
            n_skip += 1
        else:
            to_download.append((northing_km, easting_km, path))

    n_dl = len(to_download)
    print(f"\n  [{year}]  {n_dl} to download"
          + (f", {n_skip} already present" if n_skip else ""))
    print(f"         Output → {year_dir}")

    if not to_download:
        return 0, n_skip, 0

    lock    = Lock()
    counter = [0]
    done:   list[dict]           = []
    failed: list[tuple[int, int, str]] = []

    def _worker(args: tuple):
        northing_km, easting_km, path = args
        try:
            meta = download_subtile(northing_km, easting_km, year, token, path)
            with lock:
                counter[0] += 1
                done.append(meta)
                print(
                    f"  [{counter[0]:3d}/{n_dl}] "
                    f"orto_{year}_1km_{northing_km}_{easting_km}.tif  "
                    f"({meta['size_mb']:.1f} MB)"
                )
        except Exception as exc:
            with lock:
                failed.append((northing_km, easting_km, str(exc)))
                print(
                    f"  [FAIL] orto_{year}_1km_{northing_km}_{easting_km}.tif  {exc}",
                    file=sys.stderr,
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_worker, to_download))

    if failed:
        print(f"\n  {len(failed)} failure(s) for year {year}:", file=sys.stderr)
        for nk, ek, msg in failed:
            print(f"    N={nk} E={ek}: {msg}", file=sys.stderr)

    update_metadata_csv(year_dir, done)
    return len(done), n_skip, len(failed)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_tile(raw: str) -> tuple[int, int]:
    parts = re.split(r"[,\s]+", raw.strip())
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse tile '{raw}'. Use '623,57' or '6230000 570000'"
        )
    a, b = int(parts[0]), int(parts[1])
    return (a // 10_000 if a >= 10_000 else a,
            b // 10_000 if b >= 10_000 else b)


def parse_args():
    p = argparse.ArgumentParser(
        description="Download orthophoto tiles (12.5 cm/px) from dataforsyningen.dk WMS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tile", required=True,
        help="Tile index N,E (e.g. 623,57) or UTM lower-left (e.g. 6230000 570000)",
    )
    p.add_argument(
        "--years", type=int, nargs="+",
        default=AVAILABLE_YEARS,
        metavar="YEAR",
        help=f"Year(s) to download. Available: {AVAILABLE_YEARS[0]}–{AVAILABLE_YEARS[-1]}",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent download threads",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download tiles that already exist on disk",
    )
    args = p.parse_args()

    try:
        args.n, args.e = parse_tile(args.tile)
    except ValueError as exc:
        p.error(str(exc))

    for yr in args.years:
        if yr not in AVAILABLE_YEARS:
            p.error(
                f"Year {yr} not available. "
                f"Choose from: {AVAILABLE_YEARS[0]}–{AVAILABLE_YEARS[-1]}"
            )

    if args.workers < 1:
        p.error("--workers must be >= 1")

    return args


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args  = parse_args()
    token = load_token()

    e_min = args.e * 10_000
    n_min = args.n * 10_000
    n_tiles_total = len(args.years) * SUBTILE_GRID ** 2

    print(f"\nOrthophoto downloader — dataforsyningen.dk")
    print(f"  Tile     : ({args.n}, {args.e})")
    print(f"  Northing : {n_min:,}–{n_min + 10_000:,} m  (UTM 32N)")
    print(f"  Easting  : {e_min:,}–{e_min + 10_000:,} m  (UTM 32N)")
    print(f"  Years    : {args.years}")
    print(f"  Workers  : {args.workers}")
    print(f"  GSD      : {GSD_M} m/px  "
          f"({PIXELS_PER_TILE}×{PIXELS_PER_TILE} px per 1 km tile)")
    print(f"  Max tiles: {n_tiles_total}  "
          f"(≈{n_tiles_total * PIXELS_PER_TILE ** 2 * 3 / 1e9:.0f} GB uncompressed if all new)")

    grand_dl = grand_skip = grand_fail = 0
    for year in sorted(args.years):
        dl, sk, fa = download_year(
            args.n, args.e, year, token,
            force=args.force,
            n_workers=args.workers,
        )
        grand_dl   += dl
        grand_skip += sk
        grand_fail += fa

    print(f"\nSummary: {grand_dl} downloaded, {grand_skip} skipped, {grand_fail} failed")
    if grand_fail:
        print(f"Completed with {grand_fail} failure(s) — re-run to retry failed tiles.",
              file=sys.stderr)
    else:
        print("All done.")


if __name__ == "__main__":
    main()
