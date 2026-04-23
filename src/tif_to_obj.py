"""
Convert a GeoTIFF DSM/DEM to an OBJ mesh (or NxN tiled OBJs) for IsaacSim.

UV coordinates (planar X/Y projection) are written into the OBJ so a
satellite texture can be assigned later in IsaacSim without re-exporting.
When tiling, all tiles share the same UV space (0..1 over the full terrain),
so a single stitched texture PNG can be applied to every tile.

Outputs are written under  outputs/<name>/
    outputs/<name>/meshes/mesh.obj              (single mesh)
    outputs/<name>/meshes/mesh_tile_00_00.obj … (tiled)
    outputs/<name>/metadata.csv

Usage
-----
python3 src/tif_to_obj.py inputs/DSM_10km_623_57.tif --out myterrain
python3 src/tif_to_obj.py inputs/DSM_10km_623_57.tif --resolution 5 --out myterrain
python3 src/tif_to_obj.py inputs/DSM_10km_623_57.tif --resolution 2 --tiles 4 --out myterrain
python3 src/tif_to_obj.py inputs/DSM_10km_623_57.tif --resolution 1 --tiles 10 --out myterrain

Requires: rasterio  (pip install rasterio)
"""

import argparse
import io
import os
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", message="A NumPy version")
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt


# ══════════════════════════════════════════════════════════════════════════════
# HEIGHTMAP LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_heightmap(path: str, target_res_m: float):
    """
    Read and resample the GeoTIFF to ~target_res_m metres per pixel.
    Returns (heights_2d, pixel_size_x, pixel_size_y) in metres.
    No-data holes are filled by nearest-valid-neighbour.
    """
    with rasterio.open(path) as src:
        native_res_x, native_res_y = src.res
        orig_w, orig_h = src.width, src.height
        nodata = src.nodata

        scale = native_res_x / target_res_m
        out_w = max(2, int(round(orig_w * scale)))
        out_h = max(2, int(round(orig_h * scale)))

        print(f"  Source   : {orig_w}×{orig_h} px  @ {native_res_x:.2f} m/px")
        print(f"  Resampled: {out_w}×{out_h} px  @ {target_res_m:.2f} m/px  "
              f"({out_w * target_res_m / 1000:.1f} km × "
              f"{out_h * target_res_m / 1000:.1f} km)")

        data = src.read(
            1,
            out=np.empty((out_h, out_w), dtype=np.float32),
            resampling=Resampling.bilinear,
        )

    if nodata is not None:
        mask = data == nodata
    else:
        mask = ~np.isfinite(data)

    if mask.any():
        n_bad = mask.sum()
        _, nearest = distance_transform_edt(mask, return_indices=True)
        data[mask] = data[nearest[0][mask], nearest[1][mask]]
        print(f"  Filled {n_bad:,} no-data pixels via nearest-neighbour")

    actual_res_x = (orig_w / out_w) * native_res_x
    actual_res_y = (orig_h / out_h) * native_res_y
    return data, actual_res_x, actual_res_y


# ══════════════════════════════════════════════════════════════════════════════
# METADATA CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_metadata_csv(path: str, out_dir: Path):
    """
    Extract non-elevation metadata from the GeoTIFF and write it to a CSV.
    Covers: file properties, CRS, affine transform, band descriptors, tags,
    and per-band statistics for any extra bands beyond band 1.
    """
    import csv

    rows = []

    with rasterio.open(path) as src:
        rows.append(("file", path))
        rows.append(("driver", src.driver))
        rows.append(("width_px", src.width))
        rows.append(("height_px", src.height))
        rows.append(("band_count", src.count))
        rows.append(("dtype", src.dtypes[0]))
        rows.append(("nodata", src.nodata))
        rows.append(("compression", src.compression.value if src.compression else "none"))

        if src.crs:
            rows.append(("crs_wkt",   src.crs.to_wkt()))
            rows.append(("crs_epsg",  src.crs.to_epsg()))
            rows.append(("crs_proj4", src.crs.to_proj4()))
        else:
            rows.append(("crs_wkt", "undefined"))

        t = src.transform
        rows.append(("transform_pixel_size_x_m", t.a))
        rows.append(("transform_pixel_size_y_m", t.e))
        rows.append(("transform_origin_x",        t.c))
        rows.append(("transform_origin_y",        t.f))

        b = src.bounds
        rows.append(("bounds_left",   b.left))
        rows.append(("bounds_bottom", b.bottom))
        rows.append(("bounds_right",  b.right))
        rows.append(("bounds_top",    b.top))

        for k, v in src.tags().items():
            rows.append((f"tag:{k}", v))

        for band_idx in range(1, src.count + 1):
            prefix = f"band{band_idx}"
            rows.append((f"{prefix}:description", src.descriptions[band_idx - 1] or ""))
            rows.append((f"{prefix}:units", src.units[band_idx - 1] if src.units else ""))
            rows.append((f"{prefix}:colorinterp", src.colorinterp[band_idx - 1].name))
            for k, v in src.tags(band_idx).items():
                rows.append((f"{prefix}:tag:{k}", v))
            if band_idx > 1:
                try:
                    stats = src.statistics(band_idx)
                    rows.append((f"{prefix}:min",  stats.min))
                    rows.append((f"{prefix}:max",  stats.max))
                    rows.append((f"{prefix}:mean", stats.mean))
                    rows.append((f"{prefix}:std",  stats.std))
                except Exception:
                    d = src.read(band_idx).astype(np.float32)
                    nd = src.nodata
                    valid = d[d != nd] if nd is not None else d[np.isfinite(d)]
                    if valid.size:
                        rows.append((f"{prefix}:min",  float(valid.min())))
                        rows.append((f"{prefix}:max",  float(valid.max())))
                        rows.append((f"{prefix}:mean", float(valid.mean())))
                        rows.append((f"{prefix}:std",  float(valid.std())))

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows)
    print(f"  Metadata → {csv_path}  ({len(rows)} entries)")


# ══════════════════════════════════════════════════════════════════════════════
# MESH BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_mesh(heights: np.ndarray, res_x: float, res_y: float,
               full_size_x: float = None, full_size_y: float = None,
               x_origin: float = None, y_origin: float = None):
    """
    Build (vertices, faces, uvs) for a height array.

    In single-tile mode (full_size_x/y omitted) the mesh is centred at the
    origin and UVs span 0..1 over the tile extents.

    In tiled mode the caller passes the full terrain dimensions and the
    lower-left corner of this tile in the full terrain's coordinate frame so
    that UVs are consistent across all tiles (all reference 0..1 over the
    whole terrain, matching the stitched satellite texture).
    """
    H, W = heights.shape

    tile_size_x = (W - 1) * res_x
    tile_size_y = (H - 1) * res_y

    if full_size_x is None:
        full_size_x = tile_size_x
        full_size_y = tile_size_y
        x_origin    = -tile_size_x / 2
        y_origin    = -tile_size_y / 2

    xs = np.linspace(x_origin, x_origin + tile_size_x, W)
    ys = np.linspace(y_origin, y_origin + tile_size_y, H)
    xx, yy = np.meshgrid(xs, ys)

    vertices = np.stack([xx.ravel(), yy.ravel(), heights.ravel()], axis=1)

    # UVs: 0..1 over the FULL terrain extent (so tiled meshes share one texture)
    # V is flipped: OBJ v=0 is at image bottom, but PIL row 0 is at image top.
    uvs = np.stack([
        (xx.ravel() - (-full_size_x / 2)) / full_size_x,
        1.0 - (yy.ravel() - (-full_size_y / 2)) / full_size_y,
    ], axis=1).astype(np.float32)

    ii, jj = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    a = ii * W + jj;        b = ii * W + (jj + 1)
    c = (ii + 1) * W + jj;  d = (ii + 1) * W + (jj + 1)
    faces = np.concatenate([
        np.stack([a, b, d], axis=1),
        np.stack([a, d, c], axis=1),
    ])

    return vertices, faces, uvs


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT — parallel chunk formatters (must be module-level for pickling)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_vertices(chunk: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, chunk, fmt="v %.3f %.3f %.3f")
    return buf.getvalue()

def _fmt_uvs(chunk: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, chunk, fmt="vt %.6f %.6f")
    return buf.getvalue()

def _fmt_faces(chunk: np.ndarray) -> bytes:
    fi6 = chunk[:, [0, 0, 1, 1, 2, 2]]
    buf = io.BytesIO()
    np.savetxt(buf, fi6, fmt="f %d/%d %d/%d %d/%d")
    return buf.getvalue()


def _stream_write(file_buf, data: np.ndarray, fmt_fn, executor: ProcessPoolExecutor,
                  n_workers: int, chunk_size: int):
    """Submit chunks to the pool, write results to file in order.
    Keeps at most n_workers formatted chunks in memory at a time."""
    in_flight: deque = deque()
    for start in range(0, len(data), chunk_size):
        if len(in_flight) >= n_workers:
            file_buf.write(in_flight.popleft().result())
        in_flight.append(executor.submit(fmt_fn, data[start:start + chunk_size]))
    while in_flight:
        file_buf.write(in_flight.popleft().result())


def export(vertices, faces, uvs, obj_path: Path, n_workers: int):
    chunk_size = max(500_000, len(vertices) // (n_workers * 4))
    with open(obj_path, "wb") as f, \
         ProcessPoolExecutor(max_workers=n_workers) as pool:
        _stream_write(f, vertices,  _fmt_vertices, pool, n_workers, chunk_size)
        f.write(b"\n")
        _stream_write(f, uvs,       _fmt_uvs,      pool, n_workers, chunk_size)
        f.write(b"\n")
        _stream_write(f, faces + 1, _fmt_faces,    pool, n_workers, chunk_size)
    print(f"  Mesh     → {obj_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TILED EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_tiled(heights: np.ndarray, res_x: float, res_y: float,
                 mesh_dir: Path, n_tiles: int, n_workers: int):
    """
    Split the heightmap into an n_tiles × n_tiles grid of OBJ files.

    Adjacent tiles share their boundary vertices (1-vertex overlap), so
    there are no gaps when all tiles are loaded together in IsaacSim.

    Each tile uses LOCAL UV coordinates (0..1 within the tile), so each tile
    pairs with its own texture tile from stitch_texture.py --tiles N.

    Output files: <mesh_dir>/mesh_tile_<row>_<col>.obj
    tile_00_00 = north-west corner; row index increases southward,
    column index increases eastward — matching stitch_texture tile naming.
    """
    H, W = heights.shape
    full_size_x = (W - 1) * res_x
    full_size_y = (H - 1) * res_y

    row_bounds = np.round(np.linspace(0, H - 1, n_tiles + 1)).astype(int)
    col_bounds = np.round(np.linspace(0, W - 1, n_tiles + 1)).astype(int)

    total = n_tiles * n_tiles
    print(f"  Tiling   : {n_tiles}×{n_tiles} = {total} tiles")

    for ri in range(n_tiles):
        for ci in range(n_tiles):
            r0, r1 = row_bounds[ri], min(row_bounds[ri + 1] + 1, H)
            c0, c1 = col_bounds[ci], min(col_bounds[ci + 1] + 1, W)

            tile_h = heights[r0:r1, c0:c1]
            th, tw = tile_h.shape

            idx = ri * n_tiles + ci + 1
            print(f"  [{idx:3d}/{total}] mesh_tile_{ri:02d}_{ci:02d}  "
                  f"{tw}×{th} = {th*tw:,} verts, {2*(th-1)*(tw-1):,} tris …")

            # Build with local UVs (0..1 per tile); vertices are centred at origin.
            vertices, faces, uvs = build_mesh(tile_h, res_x, res_y)

            # Shift vertices into global terrain coordinates so tiles fit together.
            x_origin = -full_size_x / 2 + c0 * res_x
            y_origin = -full_size_y / 2 + r0 * res_y
            vertices[:, 0] += x_origin + (tw - 1) * res_x / 2
            vertices[:, 1] += y_origin + (th - 1) * res_y / 2

            obj_path = mesh_dir / f"mesh_tile_{ri:02d}_{ci:02d}.obj"
            export(vertices, faces, uvs, obj_path, n_workers)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def resolve_dirs(tif_path: str, resolution: float,
                 custom_name: str | None) -> tuple[Path, Path]:
    name      = custom_name or (Path(tif_path).stem + f"_{int(resolution)}m")
    out_dir   = Path(__file__).parent.parent / "outputs" / name
    mesh_dir  = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, mesh_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert GeoTIFF DSM/DEM to OBJ mesh for IsaacSim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tif",           help="Input GeoTIFF file")
    p.add_argument("--resolution",  type=float, default=10.0,
                   help="Target vertex spacing in metres")
    p.add_argument("--tiles",       type=int,   default=1,
                   help="Split into N×N tiles (1 = single mesh)")
    p.add_argument("--out",         default=None,
                   help="Output folder name (default: <tif_stem>_<resolution>m)")
    p.add_argument("--workers",     type=int, default=os.cpu_count(),
                   help="Parallel workers for OBJ export (default: all CPU cores)")
    return p.parse_args()


def main():
    args             = parse_args()
    out_dir, mesh_dir = resolve_dirs(args.tif, args.resolution, args.out)

    print(f"\nConverting: {args.tif}")
    print(f"  resolution={args.resolution} m  tiles={args.tiles}×{args.tiles}"
          f"  workers={args.workers}")
    print(f"  Output   → {out_dir}")

    heights, res_x, res_y = load_heightmap(args.tif, args.resolution)

    H, W = heights.shape
    print(f"  Heights  : min={heights.min():.1f} m  max={heights.max():.1f} m  "
          f"span={heights.max()-heights.min():.1f} m")
    print(f"  Total    : {W}×{H} = {H*W:,} vertices, {2*(H-1)*(W-1):,} triangles")

    print("  Saving metadata …")
    save_metadata_csv(args.tif, out_dir)

    if args.tiles == 1:
        print("  Building mesh …")
        vertices, faces, uvs = build_mesh(heights, res_x, res_y)
        print("  Exporting …")
        export(vertices, faces, uvs, mesh_dir / "mesh.obj", args.workers)
    else:
        per_tile = (H // args.tiles) * (W // args.tiles)
        print(f"  ~{per_tile:,} vertices / tile  (~{2*per_tile:,} triangles / tile)")
        export_tiled(heights, res_x, res_y, mesh_dir, args.tiles, args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
