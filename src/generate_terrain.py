"""
Generate terrain mesh (OBJ) and satellite texture (PNG) in one command.

Point at a tile folder produced by download_terrain.py.  The script
auto-discovers the elevation TIF from  surface_model/  or  terrain_model/
and the satellite images from  satellite_images/.

Shared parameters (--out, --crop, --invert-crop, --tiles) are forwarded to
both tif_to_obj.py and stitch_texture.py automatically.

Outputs are written under  outputs/<name>/
    outputs/<name>/meshes/    ← from tif_to_obj
    outputs/<name>/textures/  ← from stitch_texture

Usage
-----
python3 src/generate_terrain.py inputs/tile_623_57 --out 623_57
python3 src/generate_terrain.py inputs/tile_623_57 --out 623_57 --resolution 2 --size 16000
python3 src/generate_terrain.py inputs/tile_623_57 --out 623_57_crop --crop 0.3 0.7 0.3 0.7
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rasterio

SCRIPT_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate terrain OBJ mesh and satellite texture PNG together",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tile",  help="Tile folder (e.g. inputs/tile_623_57) containing "
                                 "surface_model/ or terrain_model/ and satellite_images/")
    p.add_argument("--out", required=True,
                   help="Output folder name under outputs/")

    # ── Shared ────────────────────────────────────────────────────────────────
    p.add_argument("--tiles", type=int, default=1,
                   help="Split into N×N mesh tiles and N×N texture tiles")
    p.add_argument("--crop", type=float, nargs=4,
                   metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
                   help="Crop bounding box as fractions 0–1 "
                        "(X: 0=west 1=east, Y: 0=south 1=north). "
                        "Example: --crop 0.3 0.7 0.3 0.7")
    p.add_argument("--invert-crop", action="store_true",
                   help="Export the 4 outer strips surrounding --crop instead of the "
                        "inner region (requires --crop)")

    # ── Mesh only (tif_to_obj) ─────────────────────────────────────────────
    p.add_argument("--resolution", type=float, default=10.0,
                   help="Mesh vertex spacing in metres")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers for OBJ export (default: all CPU cores)")

    # ── Texture only (stitch_texture) ──────────────────────────────────────
    p.add_argument("--size", type=int, default=4096,
                   help="Target texture long-edge size in pixels")

    args = p.parse_args()
    _validate(args, p)
    return args


def _resolve_tile_folder(tile_dir: Path, p):
    """
    Discover the elevation TIF and satellite folder inside a tile directory.
    Sets args.tif and args.satellite, or calls p.error() with a clear message.
    """
    if not tile_dir.exists():
        p.error(f"Tile folder not found: {tile_dir}")
    if not tile_dir.is_dir():
        p.error(f"Tile path is not a directory: {tile_dir}")

    sat = tile_dir / "satellite_images"
    if not sat.is_dir():
        p.error(f"satellite_images/ not found in {tile_dir}")

    tifs = list((tile_dir / "surface_model").glob("*.tif"))

    if not tifs:
        p.error(f"No .tif file found in surface_model/ under {tile_dir}")
    if len(tifs) > 1:
        names = ", ".join(t.name for t in sorted(tifs))
        p.error(f"Multiple TIF files found in {tile_dir / 'surface_model'}: {names}\n"
                f"  Remove or move the unwanted file so only one remains.")

    return str(tifs[0]), str(sat)


def _validate(args, p):
    # ── Resolve tile folder into tif + satellite paths ────────────────────────
    args.tif, args.satellite = _resolve_tile_folder(Path(args.tile), p)

    # ── Numeric ranges ────────────────────────────────────────────────────────
    if args.resolution <= 0:
        p.error(f"--resolution must be > 0, got {args.resolution}")
    try:
        with rasterio.open(args.tif) as src:
            native_res = min(src.res)
        if args.resolution < native_res:
            p.error(
                f"--resolution {args.resolution} m is finer than the TIF's native "
                f"resolution ({native_res:.4g} m/px) — upsampling adds no detail. "
                f"Use --resolution >= {native_res:.4g}"
            )
    except rasterio.errors.RasterioIOError:
        pass  # invalid TIF — the sub-script will report a clearer error
    if args.size < 2:
        p.error(f"--size must be >= 2, got {args.size}")
    if args.tiles < 1:
        p.error(f"--tiles must be >= 1, got {args.tiles}")
    if args.workers is not None and args.workers < 1:
        p.error(f"--workers must be >= 1, got {args.workers}")

    # ── Crop bounding box ─────────────────────────────────────────────────────
    if args.invert_crop and not args.crop:
        p.error("--invert-crop requires --crop")

    if args.crop:
        mn_x, mx_x, mn_y, mx_y = args.crop
        for name, val in [("MIN_X", mn_x), ("MAX_X", mx_x),
                          ("MIN_Y", mn_y), ("MAX_Y", mx_y)]:
            if not (0.0 <= val <= 1.0):
                p.error(f"--crop {name} must be in [0, 1], got {val}")
        if mn_x >= mx_x:
            p.error(f"--crop MIN_X ({mn_x}) must be less than MAX_X ({mx_x})")
        if mn_y >= mx_y:
            p.error(f"--crop MIN_Y ({mn_y}) must be less than MAX_Y ({mx_y})")

    # ── Warn about ignored combinations ───────────────────────────────────────
    if args.invert_crop and args.tiles > 1:
        print("Warning: --tiles is ignored when --invert-crop is set "
              "(outer strips are always exported as 4 individual files)",
              file=sys.stderr)


def build_mesh_cmd(args) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "tif_to_obj.py"),
        args.tif,
        "--resolution", str(args.resolution),
        "--tiles",      str(args.tiles),
        "--out",        args.out,
    ]
    if args.crop:
        cmd += ["--crop"] + [str(v) for v in args.crop]
    if args.invert_crop:
        cmd.append("--invert-crop")
    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]
    return cmd


def build_texture_cmd(args) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "stitch_texture.py"),
        args.satellite,
        "--size",  str(args.size),
        "--tiles", str(args.tiles),
        "--out",   args.out,
    ]
    if args.crop:
        cmd += ["--crop"] + [str(v) for v in args.crop]
    if args.invert_crop:
        cmd.append("--invert-crop")
    return cmd


def run_step(cmd: list[str], label: str):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {label} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    args = parse_args()

    print(f"\nTerrain generator")
    print(f"  Tile      : {args.tile}")
    print(f"  Elevation : {args.tif}  ({args.resolution} m/px)")
    print(f"  Texture   : {args.satellite}  ({args.size} px max)")
    print(f"  Output    : outputs/{args.out}/")
    print(f"  Tiles     : {args.tiles}×{args.tiles}")
    if args.crop:
        mn_x, mx_x, mn_y, mx_y = args.crop
        print(f"  Crop      : X {mn_x*100:.1f}%–{mx_x*100:.1f}%  "
              f"Y {mn_y*100:.1f}%–{mx_y*100:.1f}%"
              + ("  [inverted]" if args.invert_crop else ""))

    run_step(build_mesh_cmd(args),    "Step 1 / 2 — Mesh   (tif_to_obj.py)")
    run_step(build_texture_cmd(args), "Step 2 / 2 — Texture (stitch_texture.py)")

    print(f"\nAll done → outputs/{args.out}/")


if __name__ == "__main__":
    main()
