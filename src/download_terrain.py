"""
Interactively download terrain and satellite data from datafordeler.dk.

Reads API key from  config/api_key.yaml
Downloads to        inputs/             (DSM / DTM  .tif files)
                    inputs/satellite/   (100× 1×1 km satellite GeoTIFFs)

Usage
-----
python3 src/download_terrain.py
"""

import re
import sys
import tempfile
import zipfile
from pathlib import Path

import rasterio
import requests
import yaml


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT     = Path(__file__).parent.parent
CONF     = ROOT / "config" / "api_key.yaml"
INPUTS   = ROOT / "inputs"
BASE_URL = "https://api.datafordeler.dk/FileDownloads"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_api_key() -> str:
    if not CONF.exists():
        sys.exit(
            f"Config file not found: {CONF}\n"
            f"Create it with:\n  api_key: YOUR_KEY_HERE"
        )
    with open(CONF) as f:
        cfg = yaml.safe_load(f)
    key = (cfg or {}).get("api_key", "").strip()
    if not key or key == "YOUR_API_KEY_HERE":
        sys.exit(f"Set a real api_key value in {CONF}")
    return key


# ══════════════════════════════════════════════════════════════════════════════
# TILE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def tile_bounds(n: int, e: int) -> dict:
    """UTM 32N bounds for a 10×10 km tile identified by index (n, e)."""
    e_min = e * 10_000
    n_min = n * 10_000
    return dict(e_min=e_min, n_min=n_min,
                e_max=e_min + 10_000, n_max=n_min + 10_000)


# ══════════════════════════════════════════════════════════════════════════════
# URL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _raster_file_url(filename: str, key: str) -> str:
    return f"{BASE_URL}/GetRasterFile?FileName={filename}&apiKey={key}"


def dsm_url(n: int, e: int, key: str) -> str:
    return _raster_file_url(f"DSM_10km_{n}_{e}.tif", key)


def dtm_url(n: int, e: int, key: str) -> str:
    return _raster_file_url(f"DTM_10km_{n}_{e}.tif", key)


def _sat_bbox(n: int, e: int, inset: int = 1000) -> str:
    b = tile_bounds(n, e)
    return (f"minx={b['e_min']+inset}&miny={b['n_min']+inset}"
            f"&maxx={b['e_max']-inset}&maxy={b['n_max']-inset}")


def sat_url(n: int, e: int, key: str) -> str:
    return (f"{BASE_URL}/GetRasterMultipleFiles?{_sat_bbox(n, e)}"
            f"&register=GeoDKO&datasetname=GEODKO12,5cm&version=1&apiKey={key}")


def sat_size_url(n: int, e: int, key: str) -> str:
    return (f"{BASE_URL}/GetRasterMultipleSize?{_sat_bbox(n, e)}"
            f"&register=GEODKO&version=1&datasetname=GEODKO12,5cm&apiKey={key}")


# ══════════════════════════════════════════════════════════════════════════════
# EXISTENCE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _tif_info(path: Path) -> str:
    return f"{path.stat().st_size / 1e9:.2f} GB" if path.exists() else ""


def tile_dir(n: int, e: int) -> Path:
    return INPUTS / f"tile_{n}_{e}"


def dsm_path(n: int, e: int) -> Path:
    return tile_dir(n, e) / "surface_model" / f"DSM_10km_{n}_{e}.tif"


def dtm_path(n: int, e: int) -> Path:
    return tile_dir(n, e) / "terrain_model" / f"DTM_10km_{n}_{e}.tif"


def sat_dir(n: int, e: int) -> Path:
    return tile_dir(n, e) / "satellite_images"


def count_sat_tiles(n: int, e: int) -> int:
    """Count already-downloaded 1×1 km tiles in the tile-specific subdirectory."""
    d = sat_dir(n, e)
    if not d.exists():
        return 0
    pat = re.compile(r"_(\d+)_(\d+)\.tif$", re.IGNORECASE)
    n_range = range(n * 10, n * 10 + 10)
    e_range = range(e * 10, e * 10 + 10)
    return sum(
        1 for f in d.glob("*.tif")
        if (m := pat.search(f.name))
        and int(m.group(1)) in n_range
        and int(m.group(2)) in e_range
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _progress(done: int, total: int):
    if total:
        print(f"  {done/1e6:6.0f} / {total/1e6:.0f} MB  ({done/total*100:.0f}%)",
              end="\r", flush=True)
    else:
        print(f"  {done/1e6:6.0f} MB received", end="\r", flush=True)


def download_tif(url: str, dest: Path, label: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {label} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    print(f"  {done/1e6:.1f} MB  →  {dest}" + " " * 20)


def download_satellite(n: int, e: int, key: str):
    tile_sat_dir = sat_dir(n, e)
    tile_sat_dir.mkdir(parents=True, exist_ok=True)

    print("\nQuerying satellite download size …")
    try:
        sr = requests.get(sat_size_url(n, e, key), timeout=30)
        sr.raise_for_status()
        print(f"  {sr.text.strip()}")
    except requests.RequestException as exc:
        print(f"  (size query failed: {exc})")

    print("\nDownloading satellite tiles …")
    tmp_path = Path(tempfile.mktemp(suffix=".zip"))
    try:
        with requests.get(sat_url(n, e, key), stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
        print(f"  {done/1e6:.1f} MB received" + " " * 20)

        with zipfile.ZipFile(tmp_path) as zf:
            tifs = [name for name in zf.namelist()
                    if name.lower().endswith(".tif")]
            total_t = len(tifs)

            if total_t < 100:
                print(f"\n  Warning: the archive contains only {total_t}/100 tiles.")
                print("  This is normal if part of the tile covers sea or is outside")
                print("  the dataset boundary, but may indicate a partial download.")
                if not _confirm("  Continue and extract anyway? [y/N]: "):
                    print("  Extraction cancelled — temp file discarded.")
                    return

            print(f"  Extracting {total_t} files …")
            for i, name in enumerate(tifs, 1):
                out = tile_sat_dir / Path(name).name
                out.write_bytes(zf.read(name))
                print(f"  [{i:3d}/{total_t}] {Path(name).name}", end="\r", flush=True)
        print(f"  Extracted {total_t} files  →  {tile_sat_dir}" + " " * 30)

    finally:
        tmp_path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_raster(path: Path,
                  expected_bands: int | None = None,
                  expected_epsg: int | None = None,
                  expected_res_m: float | None = None,
                  show_block_progress: bool = False) -> tuple[bool, str]:
    """
    Open a GeoTIFF, validate its metadata, and read every block to detect
    tile-level corruption (e.g. TIFFReadEncodedTile failures).
    Returns (ok, human-readable message).
    """
    try:
        with rasterio.open(path) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            res  = abs(src.transform.a)
            w, h, bands = src.width, src.height, src.count

            if expected_bands is not None and bands != expected_bands:
                return False, f"expected {expected_bands} band(s), got {bands}"
            if expected_epsg is not None and epsg != expected_epsg:
                return False, f"expected EPSG:{expected_epsg}, got EPSG:{epsg}"
            if expected_res_m is not None:
                if not (expected_res_m * 0.95 <= res <= expected_res_m * 1.05):
                    return False, f"expected ~{expected_res_m} m/px, got {res:.4f} m/px"

            windows = list(src.block_windows(1))
            n_blocks = len(windows)
            for i, (_, win) in enumerate(windows, 1):
                if show_block_progress:
                    print(f"    block {i}/{n_blocks}", end="\r", flush=True)
                src.read(1, window=win)

        return True, f"{w}×{h} px  EPSG:{epsg}  {res:.4f} m/px  {bands} band(s)"

    except rasterio.errors.RasterioIOError as exc:
        return False, f"read error — {exc}"
    except Exception as exc:
        return False, f"error — {exc}"


def verify_dsm_dtm(path: Path, label: str) -> bool:
    print(f"\nVerifying {label} …")
    ok, msg = verify_raster(path,
                            expected_bands=1,
                            expected_epsg=25832,
                            expected_res_m=0.4,
                            show_block_progress=True)
    if ok:
        print(f"  OK  {msg}" + " " * 30)
    else:
        print(f"\n  FAILED: {msg}")
    return ok


def verify_satellite_tiles(n: int, e: int) -> bool:
    tifs = sorted(sat_dir(n, e).glob("*.tif"))
    total = len(tifs)
    if not total:
        print("  No satellite tiles found to verify.")
        return False

    print(f"\nVerifying {total} satellite tiles …")
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(tifs, 1):
        print(f"  [{i:3d}/{total}] {path.name}", end="\r", flush=True)
        ok, msg = verify_raster(path, expected_epsg=25832, expected_res_m=0.125)
        if not ok:
            failed.append((path.name, msg))

    if failed:
        print(f"\n  {len(failed)}/{total} tiles FAILED:" + " " * 30)
        for name, msg in failed:
            print(f"    {name}: {msg}")
        return False

    print(f"  All {total} tiles OK" + " " * 40)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_MENU = """\
What would you like to download?

  1) DSM only          (surface model — buildings & vegetation included, ~2.5 GB)
  2) DTM only          (terrain model — bare ground only,                ~2.5 GB)
  3) Satellite only    (100 × 1×1 km GeoTIFF tiles at 12.5 cm            ~6.3 GB)
  4) DSM + DTM
  5) DSM + Satellite
  6) DTM + Satellite
  7) DSM + DTM + Satellite  (everything)

For more information about the datadistributer dowload web-api, visit: 
https://confluence.sdfi.dk/pages/viewpage.action?pageId=151999753
https://datafordeler.dk/dataoversigt/danmarks-hoejdemodel-dhm/dhm-fildownload-raster/
https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/geodanmark-ortofoto-fildownload-raster/

To visualize the data on a map before downloading, visit: 
https://dataforsyningen.dk/map
"""

_CHOICES: dict[str, tuple[bool, bool, bool]] = {
    "1": (True,  False, False),
    "2": (False, True,  False),
    "3": (False, False, True),
    "4": (True,  True,  False),
    "5": (True,  False, True),
    "6": (False, True,  True),
    "7": (True,  True,  True),
}


def prompt_choice() -> tuple[bool, bool, bool]:
    print(_MENU)
    while True:
        c = input("Enter choice [1–7]: ").strip()
        if c in _CHOICES:
            return _CHOICES[c]
        print("  Please enter a number from 1 to 7.")


def prompt_tile() -> tuple[int, int]:
    print("\nEnter tile as  N,E  (e.g. 623,57)")
    print("  or UTM 32N lower-left corner  northing easting  (e.g. 6230000 570000)")
    print("\nTo check the tiles overlaid on a map, visit: https://earthmaps.dataforsyningen.dk/dfTiles")
    while True:
        raw = input("Tile: ").strip()

        # "N,E" or "N, E"
        m = re.fullmatch(r"(\d+)\s*,\s*(\d+)", raw)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            # If values look like raw UTM metres, convert to tile index
            n = a // 10_000 if a >= 10_000 else a
            e = b // 10_000 if b >= 10_000 else b
            return n, e

        # "northing easting" (space-separated)
        parts = raw.split()
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                return a // 10_000, b // 10_000
            except ValueError:
                pass

        print("  Could not parse.  Try  623,57  or  6230000 570000")


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key = load_api_key()

    want_dsm, want_dtm, want_sat = prompt_choice()
    n, e = prompt_tile()

    b = tile_bounds(n, e)
    print(f"\nTile ({n}, {e})")
    print(f"  Northing : {b['n_min']:,} – {b['n_max']:,} m  (UTM 32N)")
    print(f"  Easting  : {b['e_min']:,} – {b['e_max']:,} m  (UTM 32N)")

    # ── Skip checks ───────────────────────────────────────────────────────────
    skip_dsm = skip_dtm = skip_sat = False

    if want_dsm:
        p = dsm_path(n, e)
        if p.exists():
            print(f"\n  DSM already present  ({_tif_info(p)})  →  {p}")
            skip_dsm = not _confirm("  Re-download? [y/N]: ")

    if want_dtm:
        p = dtm_path(n, e)
        if p.exists():
            print(f"\n  DTM already present  ({_tif_info(p)})  →  {p}")
            skip_dtm = not _confirm("  Re-download? [y/N]: ")

    if want_sat:
        found = count_sat_tiles(n, e)
        if found >= 100:
            print(f"\n  Satellite tiles already present  (100/100)  →  {sat_dir(n, e)}")
            skip_sat = not _confirm("  Re-download? [y/N]: ")
        elif found > 0:
            print(f"\n  Warning: only {found}/100 satellite tiles found in {sat_dir(n, e)}")
            skip_sat = not _confirm("  Re-download all? [y/N]: ")

    # ── Downloads + verification ──────────────────────────────────────────────
    anything = False

    if want_dsm and not skip_dsm:
        dest, label = dsm_path(n, e), f"DSM_10km_{n}_{e}.tif"
        download_tif(dsm_url(n, e, api_key), dest, label)
        anything = True
        if not verify_dsm_dtm(dest, label):
            if _confirm("  Re-download? [y/N]: "):
                dest.unlink(missing_ok=True)
                download_tif(dsm_url(n, e, api_key), dest, label)
                verify_dsm_dtm(dest, label)

    if want_dtm and not skip_dtm:
        dest, label = dtm_path(n, e), f"DTM_10km_{n}_{e}.tif"
        download_tif(dtm_url(n, e, api_key), dest, label)
        anything = True
        if not verify_dsm_dtm(dest, label):
            if _confirm("  Re-download? [y/N]: "):
                dest.unlink(missing_ok=True)
                download_tif(dtm_url(n, e, api_key), dest, label)
                verify_dsm_dtm(dest, label)

    if want_sat and not skip_sat:
        download_satellite(n, e, api_key)
        anything = True
        if not verify_satellite_tiles(n, e):
            if _confirm("  Re-download satellite tiles? [y/N]: "):
                import shutil
                shutil.rmtree(sat_dir(n, e), ignore_errors=True)
                download_satellite(n, e, api_key)
                verify_satellite_tiles(n, e)

    if not anything:
        print("\nNothing to download.")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
