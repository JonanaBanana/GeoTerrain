"""
Stitch 1 km × 1 km satellite tile TIFs into a texture PNG (or NxN tiles).

Tile filenames must follow the pattern  <prefix>_{northing_km}_{easting_km}.tif
(e.g. 2025_1km_6230_570.tif).  The script auto-detects grid extents.

Outputs are written under  outputs/<name>/
    outputs/<name>/textures/texture.png                (single)
    outputs/<name>/textures/texture_tile_00_00.png …   (tiled, --tiles N)
    outputs/<name>/texture_metadata.csv                (one row per satellite tile)

Usage
-----
python3 src/stitch_texture.py inputs/<tile_folder> --out myterrain
python3 src/stitch_texture.py inputs/<tile_folder> --out myterrain --size 8192
python3 src/stitch_texture.py inputs/<tile_folder> --out myterrain --size 8192 --tiles 4
python3 src/stitch_texture.py inputs/<tile_folder> --out myterrain_center --crop 0.3 0.7 0.3 0.7
python3 src/stitch_texture.py inputs/<tile_folder> --out myterrain_outer --crop 0.3 0.7 0.3 0.7 --invert-crop
"""

import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


# ══════════════════════════════════════════════════════════════════════════════
# TILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def parse_tile_files(folder: Path):
    """
    Scan folder for tile TIFs and return:
        tiles     : dict  (northing_km, easting_km) -> Path
        northings : sorted list of unique northing km indices
        eastings  : sorted list of unique easting  km indices
    """
    pattern = re.compile(r"_(\d+)_(\d+)\.tif$", re.IGNORECASE)
    tiles = {}
    for f in sorted(folder.glob("*.tif")):
        m = pattern.search(f.name)
        if m:
            n_km, e_km = int(m.group(1)), int(m.group(2))
            tiles[(n_km, e_km)] = f

    if not tiles:
        raise FileNotFoundError(f"No tile TIFs matching *_<N>_<E>.tif found in {folder}")

    northings = sorted(set(k[0] for k in tiles))
    eastings  = sorted(set(k[1] for k in tiles))
    return tiles, northings, eastings


# ══════════════════════════════════════════════════════════════════════════════
# MOSAIC BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _load_region(tiles: dict, northings: list, eastings: list,
                 tile_px_x: int, tile_px_y: int,
                 img_r0: int, img_c0: int, img_r1: int, img_c1: int,
                 show_progress: bool = True) -> np.ndarray:
    """
    Load only the satellite tiles that overlap [img_r0:img_r1, img_c0:img_c1]
    in full-mosaic pixel coordinates and return the exact cropped pixel region
    as an RGB uint8 array.  This is the single source of truth for tile
    selection and pixel arithmetic used by both build_mosaic and
    save_tiled_png_lazy.
    """
    n_rows = len(northings)
    n_cols = len(eastings)
    full_row_of = {n: i for i, n in enumerate(sorted(northings, reverse=True))}
    full_col_of = {e: i for i, e in enumerate(sorted(eastings))}

    col_t0 = img_c0 // tile_px_x;  col_t1 = min((img_c1 - 1) // tile_px_x, n_cols - 1)
    row_t0 = img_r0 // tile_px_y;  row_t1 = min((img_r1 - 1) // tile_px_y, n_rows - 1)

    needed = {k: v for k, v in tiles.items()
              if row_t0 <= full_row_of[k[0]] <= row_t1
              and col_t0 <= full_col_of[k[1]] <= col_t1}

    sub_northings = sorted(set(k[0] for k in needed))
    sub_eastings  = sorted(set(k[1] for k in needed))
    row_of = {n: i for i, n in enumerate(sorted(sub_northings, reverse=True))}
    col_of = {e: i for i, e in enumerate(sorted(sub_eastings))}

    sub_h = tile_px_y * len(sub_northings)
    sub_w = tile_px_x * len(sub_eastings)
    sub   = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
    total = len(needed)

    for idx, ((n_km, e_km), path) in enumerate(sorted(needed.items()), 1):
        if show_progress:
            print(f"  [{idx:3d}/{total}]  {path.name}", end="\r", flush=True)
        row = row_of[n_km];  col = col_of[e_km]
        rt0, rt1 = row * tile_px_y, (row + 1) * tile_px_y
        ct0, ct1 = col * tile_px_x, (col + 1) * tile_px_x
        with rasterio.open(path) as src:
            buf = np.empty((3, tile_px_y, tile_px_x), dtype=np.uint8)
            src.read([1, 2, 3], out=buf, resampling=Resampling.lanczos)
        sub[rt0:rt1, ct0:ct1] = buf.transpose(1, 2, 0)

    if show_progress:
        print()

    sc0 = img_c0 - col_t0 * tile_px_x;  sc1 = img_c1 - col_t0 * tile_px_x
    sr0 = img_r0 - row_t0 * tile_px_y;  sr1 = img_r1 - row_t0 * tile_px_y
    return sub[sr0:sr1, sc0:sc1]


def build_mosaic(tiles: dict, northings: list, eastings: list,
                 tile_px_x: int, tile_px_y: int,
                 crop: tuple = None) -> np.ndarray:
    """
    Assemble satellite tiles into a single RGB uint8 array.
    When crop is given, only the overlapping tiles are loaded and the result
    is already cropped — pixel-identical to crop_mosaic(full_mosaic, crop).
    Use save_tiled_png_lazy instead when --tiles > 1 to avoid a large
    intermediate array.
    """
    n_rows = len(northings)
    n_cols = len(eastings)
    full_h = tile_px_y * n_rows
    full_w = tile_px_x * n_cols

    if crop is not None:
        min_x, max_x, min_y, max_y = crop
        c0 = int(round(min_x * full_w));         c1 = int(round(max_x * full_w))
        r0 = int(round((1.0 - max_y) * full_h)); r1 = int(round((1.0 - min_y) * full_h))
        c0, c1 = max(0, c0), min(full_w, c1)
        r0, r1 = max(0, r0), min(full_h, r1)
        return _load_region(tiles, northings, eastings, tile_px_x, tile_px_y,
                            r0, c0, r1, c1)

    return _load_region(tiles, northings, eastings, tile_px_x, tile_px_y,
                        0, 0, full_h, full_w)


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════

def save_texture_metadata_csv(tiles: dict, out_dir: Path):
    """
    Write one CSV row per satellite tile with its rasterio metadata.

    Columns: northing_km, easting_km, filename, width_px, height_px,
             band_count, dtype, nodata, compression, crs_epsg,
             pixel_size_x_m, pixel_size_y_m,
             bounds_left, bounds_bottom, bounds_right, bounds_top,
             band1_colorinterp … band<N>_colorinterp
    """
    import csv

    rows = []
    for (n_km, e_km), path in sorted(tiles.items()):
        row = {"northing_km": n_km, "easting_km": e_km, "filename": path.name}
        with rasterio.open(path) as src:
            row["width_px"]       = src.width
            row["height_px"]      = src.height
            row["band_count"]     = src.count
            row["dtype"]          = src.dtypes[0]
            row["nodata"]         = src.nodata
            row["compression"]    = src.compression.value if src.compression else "none"
            row["crs_epsg"]       = src.crs.to_epsg() if src.crs else ""
            row["pixel_size_x_m"] = src.res[0]
            row["pixel_size_y_m"] = src.res[1]
            b = src.bounds
            row["bounds_left"]    = b.left
            row["bounds_bottom"]  = b.bottom
            row["bounds_right"]   = b.right
            row["bounds_top"]     = b.top
            for i, ci in enumerate(src.colorinterp, 1):
                row[f"band{i}_colorinterp"] = ci.name
        rows.append(row)

    if not rows:
        return

    csv_path = out_dir / "texture_metadata.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Metadata → {csv_path}  ({len(rows)} tiles)")


# ══════════════════════════════════════════════════════════════════════════════
# CROP
# ══════════════════════════════════════════════════════════════════════════════

def crop_mosaic_invert(mosaic: np.ndarray, crop: tuple) -> dict:
    """
    Return the 4 outer texture strips that surround the crop bounding box,
    matching the geometry produced by tif_to_obj export_inverted_crop.

    Returns a dict with keys 'top', 'bottom', 'left', 'right' (strips that
    touch a terrain edge are omitted).
    """
    min_x, max_x, min_y, max_y = crop
    H, W = mosaic.shape[:2]

    c0 = int(round(min_x * W))
    c1 = int(round(max_x * W))
    r0 = int(round((1.0 - max_y) * H))
    r1 = int(round((1.0 - min_y) * H))

    c0, c1 = max(0, c0), min(W, c1)
    r0, r1 = max(0, r0), min(H, r1)

    strips = {}
    if r0 > 0:
        strips["top"]    = mosaic[0:r0, :]
    if r1 < H:
        strips["bottom"] = mosaic[r1:H, :]
    if c0 > 0:
        strips["left"]   = mosaic[r0:r1, 0:c0]
    if c1 < W:
        strips["right"]  = mosaic[r0:r1, c1:W]
    return strips


def crop_mosaic(mosaic: np.ndarray, crop: tuple) -> np.ndarray:
    """
    Crop the mosaic to the same percentage bounding box used by tif_to_obj --crop.

    crop = (min_x, max_x, min_y, max_y) as fractions 0..1:
        X: 0 = west edge,  1 = east edge  (columns)
        Y: 0 = south edge, 1 = north edge (rows, inverted vs image axis)
    """
    min_x, max_x, min_y, max_y = crop
    H, W = mosaic.shape[:2]

    c0 = int(round(min_x * W))
    c1 = int(round(max_x * W))
    # Image row 0 = north → invert Y
    r0 = int(round((1.0 - max_y) * H))
    r1 = int(round((1.0 - min_y) * H))

    c0, c1 = max(0, c0), min(W, c1)
    r0, r1 = max(0, r0), min(H, r1)
    return mosaic[r0:r1, c0:c1]


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_png(mosaic: np.ndarray, out_path: Path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    Image.fromarray(mosaic).save(out_path)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  Texture  → {out_path}  "
          f"({mosaic.shape[1]}×{mosaic.shape[0]} px, {size_mb:.1f} MB)")


def save_tiled_png(mosaic: np.ndarray, tex_dir: Path, n_tiles: int):
    """
    Split the mosaic into an n_tiles × n_tiles grid of PNG files whose naming
    matches the OBJ tiles produced by tif_to_obj.py --tiles N:
        tile_00_00 = top-left  (north-west)
        row index increases downward (southward)
        col index increases rightward (eastward)
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    img    = Image.fromarray(mosaic)
    img_h, img_w = mosaic.shape[:2]

    row_bounds = np.round(np.linspace(0, img_h, n_tiles + 1)).astype(int)
    col_bounds = np.round(np.linspace(0, img_w, n_tiles + 1)).astype(int)

    total = n_tiles * n_tiles
    for ri in range(n_tiles):
        for ci in range(n_tiles):
            r0, r1 = row_bounds[ri], row_bounds[ri + 1]
            c0, c1 = col_bounds[ci], col_bounds[ci + 1]
            tile_img  = img.crop((c0, r0, c1, r1))
            tile_path = tex_dir / f"texture_tile_{ri:02d}_{ci:02d}.png"
            tile_img.save(tile_path)
            idx      = ri * n_tiles + ci + 1
            size_mb  = tile_path.stat().st_size / 1_048_576
            print(f"  [{idx:3d}/{total}] {tile_path.name}  "
                  f"({c1-c0}×{r1-r0} px, {size_mb:.1f} MB)")


def save_tiled_png_lazy(tiles: dict, northings: list, eastings: list,
                        tile_px: int, n_tiles: int, tex_dir: Path,
                        crop: tuple = None):
    """
    Write n_tiles×n_tiles PNG files without ever building a full mosaic array.
    Each output tile is loaded on demand via _load_region(), keeping peak memory
    proportional to one output tile rather than the entire mosaic.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    n_rows  = len(northings)
    n_cols  = len(eastings)
    full_h  = tile_px * n_rows
    full_w  = tile_px * n_cols

    if crop is not None:
        min_x, max_x, min_y, max_y = crop
        fc0 = int(round(min_x * full_w));         fc1 = int(round(max_x * full_w))
        fr0 = int(round((1.0 - max_y) * full_h)); fr1 = int(round((1.0 - min_y) * full_h))
        fc0, fc1 = max(0, fc0), min(full_w, fc1)
        fr0, fr1 = max(0, fr0), min(full_h, fr1)
    else:
        fc0, fr0, fc1, fr1 = 0, 0, full_w, full_h

    crop_w = fc1 - fc0
    crop_h = fr1 - fr0

    row_bounds = np.round(np.linspace(0, crop_h, n_tiles + 1)).astype(int)
    col_bounds = np.round(np.linspace(0, crop_w, n_tiles + 1)).astype(int)

    total = n_tiles * n_tiles
    for ri in range(n_tiles):
        for ci in range(n_tiles):
            out_r0, out_r1 = row_bounds[ri], row_bounds[ri + 1]
            out_c0, out_c1 = col_bounds[ci], col_bounds[ci + 1]
            w = out_c1 - out_c0
            h = out_r1 - out_r0
            idx = ri * n_tiles + ci + 1
            print(f"  [{idx:3d}/{total}] texture_tile_{ri:02d}_{ci:02d}.png  "
                  f"({w}×{h} px) …", end="\r", flush=True)
            region = _load_region(
                tiles, northings, eastings, tile_px, tile_px,
                fr0 + out_r0, fc0 + out_c0,
                fr0 + out_r1, fc0 + out_c1,
                show_progress=False,
            )
            tile_path = tex_dir / f"texture_tile_{ri:02d}_{ci:02d}.png"
            Image.fromarray(region).save(tile_path)
            size_mb = tile_path.stat().st_size / 1_048_576
            print(f"  [{idx:3d}/{total}] texture_tile_{ri:02d}_{ci:02d}.png  "
                  f"({w}×{h} px, {size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def resolve_dirs(tile_folder: Path, custom_out: str | None) -> tuple[Path, Path]:
    name    = custom_out or tile_folder.name
    out_dir = Path(__file__).parent.parent / "outputs" / name
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, tex_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Stitch satellite tile TIFs into a single texture PNG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="Folder containing 1 km×1 km tile TIFs")
    p.add_argument("--out",  default=None,
                   help="Output name under outputs/ (default: satellite folder name)")
    p.add_argument("--size",  type=int, default=4096,
                   help="Target texture width in pixels (height scaled to match grid aspect)")
    p.add_argument("--tiles", type=int, default=1,
                   help="Split output into N×N texture tiles matching tif_to_obj --tiles N "
                        "(1 = single texture)")
    p.add_argument("--crop",  type=float, nargs=4,
                   metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
                   help="Crop to a sub-region given as fractions 0–1 "
                        "(X: 0=west 1=east, Y: 0=south 1=north). "
                        "Must match --crop used in tif_to_obj. "
                        "Example: --crop 0.3 0.7 0.3 0.7")
    p.add_argument("--invert-crop", action="store_true",
                   help="Export the 4 outer texture strips surrounding --crop instead "
                        "of the inner region (requires --crop)")
    args = p.parse_args()
    if args.invert_crop and not args.crop:
        p.error("--invert-crop requires --crop")
    return args


def main():
    args                 = parse_args()
    tile_folder          = Path(args.folder)
    out_dir, tex_dir     = resolve_dirs(tile_folder, args.out)

    print(f"\nStitching: {tile_folder}")

    tiles, northings, eastings = parse_tile_files(tile_folder)
    n_rows = len(northings)
    n_cols = len(eastings)
    print(f"  Grid     : {n_cols} × {n_rows}  "
          f"(E {eastings[0]}–{eastings[-1]}, N {northings[0]}–{northings[-1]})")
    print(f"  Tiles    : {len(tiles)}")

    tile_px = max(1, args.size // max(n_rows, n_cols))
    img_w   = tile_px * n_cols
    img_h   = tile_px * n_rows
    print(f"  Texture  : {img_w}×{img_h} px  ({tile_px} px per satellite tile)")
    print(f"  Output   → {out_dir}")

    if args.crop:
        mn_x, mx_x, mn_y, mx_y = args.crop
        print(f"  Crop     : X {mn_x*100:.1f}%–{mx_x*100:.1f}%  "
              f"Y {mn_y*100:.1f}%–{mx_y*100:.1f}%")

    print("  Saving metadata …")
    save_texture_metadata_csv(tiles, out_dir)

    if args.invert_crop:
        print("  Loading and downsampling tiles …")
        mosaic = build_mosaic(tiles, northings, eastings, tile_px, tile_px)
        strips = crop_mosaic_invert(mosaic, args.crop)
        print(f"  Outer    : {len(strips)} strip(s)")
        for label, strip in strips.items():
            save_png(strip, tex_dir / f"texture_outer_{label}.png")
    elif args.tiles == 1:
        print("  Loading and downsampling tiles …")
        mosaic = build_mosaic(tiles, northings, eastings, tile_px, tile_px,
                              crop=args.crop or None)
        if args.crop:
            print(f"  Cropped  : {mosaic.shape[1]}×{mosaic.shape[0]} px")
        save_png(mosaic, tex_dir / "texture.png")
    else:
        print(f"  Processing {args.tiles}×{args.tiles} texture tiles …")
        save_tiled_png_lazy(tiles, northings, eastings, tile_px,
                            args.tiles, tex_dir, crop=args.crop or None)

    print("\nDone.")


if __name__ == "__main__":
    main()
