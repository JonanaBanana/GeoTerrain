# GeoTerrain

Pipeline for generating 3D terrain meshes (OBJ) and satellite textures (PNG) from Danish geodata, intended for import into NVIDIA Isaac Sim.

The pipeline covers three stages:

1. **Download** — fetch elevation and satellite data from datafordeler.dk
2. **Mesh** — convert a DSM GeoTIFF to an OBJ mesh with UV coordinates
3. **Texture** — stitch 1×1 km satellite GeoTIFFs into a texture PNG

A single convenience script (`generate_terrain.py`) runs stages 2 and 3 together.

---

## Requirements

```
pip install rasterio numpy scipy pillow requests pyyaml
```

---

## Data source

Data is served by [datafordeler.dk](https://datafordeler.dk). You need a free account and an API key to download files. To visualise tile indices on a map before downloading, visit [dataforsyningen.dk/map](https://dataforsyningen.dk/map) or the tile overlay at [earthmaps.dataforsyningen.dk/dfTiles](https://earthmaps.dataforsyningen.dk/dfTiles).

### API key setup

Copy the example config and fill in your key:

```bash
cp config/example_api_key.yaml config/api_key.yaml
# then edit config/api_key.yaml and replace xxxx with your real key
```

---

## Coordinate system and tile indexing

All elevation and satellite data is in **UTM zone 32N (EPSG:25832)**. Tiles are indexed as `(N, E)` where:

```
northing_min = N × 10,000 m
easting_min  = E × 10,000 m
```

Each tile covers a **10 × 10 km** area. For example, tile `(623, 57)` covers northing 6,230,000–6,240,000 m and easting 570,000–580,000 m (central Jutland / Aarhus region).

---

## Folder structure

```
inputs/
  tile_623_57/
    surface_model/          ← DSM (Digital Surface Model — buildings & vegetation included)
      DSM_10km_623_57.tif   ← used by this pipeline for mesh generation
    terrain_model/          ← DTM (Digital Terrain Model — bare ground, no buildings)
      DTM_10km_623_57.tif   ← NOT used by this pipeline; download only if needed elsewhere
    satellite_images/       ← 100× 1km×1km GeoTIFF tiles at 12.5 cm/px
      *.tif

outputs/
  <name>/
    meshes/
      mesh.obj              ← single mesh
      mesh_tile_00_00.obj … ← tiled meshes (--tiles N)
      mesh_outer_top.obj …  ← outer strips (--invert-crop)
    textures/
      texture.png           ← single texture
      texture_tile_00_00.png … ← tiled textures
      texture_outer_top.png …  ← outer strip textures
    surface_metadata.csv
    texture_metadata.csv

config/
  api_key.yaml
  example_api_key.yaml
```

---

## Scripts

### `src/download_terrain.py` — Download data interactively

Interactive CLI. Prompts for which datasets to download and which tile to fetch.

```bash
python3 src/download_terrain.py
```

**What it does:**

1. Reads the API key from `config/api_key.yaml`
2. Asks what to download: DSM, DTM, satellite imagery, or any combination
3. Asks for the tile index (`N,E` or full UTM coordinates, e.g. `623,57` or `6230000 570000`)
4. Downloads files into `inputs/tile_N_E/` with the correct subfolder layout
5. Verifies each downloaded file by checking CRS, pixel resolution, and reading every compressed block — this catches corrupt tiles before they cause cryptic errors later
6. If verification fails, offers to re-download automatically

**Data sizes (per 10×10 km tile):**

| Dataset | Format | Size | Required for this pipeline |
|---|---|---|---|
| DSM | Single GeoTIFF, ~0.4 m/px | ~2.5 GB | Yes — used for mesh generation |
| Satellite | 100 × 1km GeoTIFFs, 12.5 cm/px | ~6.3 GB | Yes — used for texture |
| DTM | Single GeoTIFF, ~0.4 m/px | ~2.5 GB | No — for external use only |

> **For terrain + texture generation, download DSM and Satellite only (choice 5).** The DTM (bare-ground model) is not used anywhere in this pipeline — download it only if another application requires it.

> **Sea tiles:** If the tile boundary overlaps the sea or the dataset edge, fewer than 100 satellite tiles are returned. The script warns you and asks whether to extract anyway — this is expected behaviour for coastal tiles.

---

### `src/generate_terrain.py` — Generate mesh + texture together

The main entry point. Points at a tile folder and runs both `tif_to_obj.py` and `stitch_texture.py` in sequence, forwarding shared parameters automatically.

```bash
python3 src/generate_terrain.py inputs/tile_N_E --out N_E
```

**Recommended two-tier workflow for Isaac Sim:**

The typical setup is a high-resolution inner mission area surrounded by a lower-resolution outer context ring. Run `generate_terrain.py` twice with the same `--crop` bounds:

```bash
# Step 1 — high-resolution mission area (native satellite resolution, 2 m mesh)
python3 src/generate_terrain.py inputs/tile_N_E --out N_E_inner \
    --crop 0.3 0.7 0.3 0.7 \
    --resolution 2 --size 80000 --tiles 4

# Step 2 — lower-resolution outer context ring (1/4 resolution, 5 m mesh)
python3 src/generate_terrain.py inputs/tile_N_E --out N_E_outer \
    --crop 0.3 0.7 0.3 0.7 --invert-crop \
    --resolution 5 --size 20000
```

**Other common usage:**

```bash
# Quick preview of a full tile at default settings (2 m mesh, 16000 px texture)
python3 src/generate_terrain.py inputs/tile_N_E --out N_E

# Split the inner mesh into a 4×4 tile grid (reduces per-file size for Isaac Sim)
python3 src/generate_terrain.py inputs/tile_N_E --out N_E_inner \
    --crop 0.3 0.7 0.3 0.7 --resolution 2 --size 80000 --tiles 4
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `tile` | required | Path to tile folder, e.g. `inputs/tile_N_E` |
| `--out` | required | Output folder name under `outputs/` |
| `--resolution` | `10.0` | Mesh vertex spacing in metres |
| `--size` | `4096` | Texture long-edge size in pixels |
| `--tiles` | `1` | Split into N×N mesh tiles and N×N texture tiles |
| `--crop MIN_X MAX_X MIN_Y MAX_Y` | — | Crop bounding box as fractions 0–1 (X: 0=west, Y: 0=south) |
| `--invert-crop` | — | Export the 4 outer strips around `--crop` instead of the inner region |
| `--workers` | all cores | Parallel workers for OBJ export |

The script auto-discovers the DSM from `surface_model/` inside the tile folder. It always uses the **surface model** (DSM) for mesh generation — the terrain model (DTM) folder is not used here.

---

### `src/tif_to_obj.py` — Convert GeoTIFF to OBJ mesh

Converts a DSM GeoTIFF to an OBJ mesh with UV coordinates.

```bash
python3 src/tif_to_obj.py inputs/tile_N_E/surface_model/DSM_10km_623_57.tif \
    --out 623_57
```

**Key behaviours:**

- Resamples the raster to the target `--resolution` (GDAL uses internal overviews for large downsampling factors — fast and memory-efficient)
- Fills no-data holes by nearest-valid-neighbour before meshing
- UV coordinates are planar (X/Y projection) and span 0–1 over the **full** terrain extent, so all tiles in a tiled export share one texture
- With `--tiles N`, produces `mesh_tile_00_00.obj` … `mesh_tile_NN_NN.obj`; adjacent tiles share boundary vertices (no seams)
- With `--invert-crop`, produces four outer strip OBJs: `mesh_outer_top/bottom/left/right.obj`

**Arguments:** same as `generate_terrain.py` mesh arguments, plus a direct `tif` positional path.

---

### `src/stitch_texture.py` — Stitch satellite tiles into a texture PNG

Assembles 1×1 km satellite GeoTIFFs into one or more texture PNGs.

```bash
python3 src/stitch_texture.py inputs/tile_N_E/satellite_images \
    --out 623_57 --size 8192
```

**Key behaviours:**

- The `--size` argument sets the long-edge pixel count of the output texture; the other axis is scaled to match the grid aspect ratio
- With `--crop`, only the overlapping satellite tiles are loaded — the full 100-tile mosaic is never assembled in memory
- With `--tiles N`, each output tile is loaded and saved independently (one at a time), keeping peak memory proportional to a single tile rather than the entire mosaic
- Tile naming (`texture_tile_00_00.png` …) matches the OBJ tile naming convention, so Isaac Sim assignments are straightforward

**Arguments:** same as `generate_terrain.py` texture arguments, plus a direct `folder` positional path.

---

## UV alignment guarantee

Mesh UVs and texture pixels are computed from the same crop arithmetic, guaranteeing alignment when imported into Isaac Sim:

- The OBJ UV for each vertex is derived from its position in the **full** (uncropped) raster
- The texture crop uses the identical percentage bounds, rounded to the same integer pixel positions
- Tiling boundaries are computed with `np.linspace` in both scripts using the same grid dimensions

As long as `--crop` and `--tiles` values are identical between `tif_to_obj.py` and `stitch_texture.py`, the outputs will align. `generate_terrain.py` forwards both flags automatically.

---

## Use cases

| Goal | Key arguments |
|---|---|
| High-res mission area (inner crop) | `--crop … --resolution 2 --size 80000 --tiles 4` |
| Low-res outer context ring | `--crop … --invert-crop --resolution 5 --size 20000` |
| Quick full-tile preview | `--resolution 2 --size 16000` (defaults) |
| Texture only (mesh already done) | `stitch_texture.py … --size 80000 --tiles 4` |

The `--size 80000` value corresponds to native satellite resolution (8000 px per 1 km tile × 10 tiles). The `--size 20000` outer ring is ¼ of that in each axis, which keeps Isaac Sim within comfortable import limits.

---

## Limitations

- **Danish data only.** The API endpoints (`datafordeler.dk`) serve Danish national datasets. Tile indices are specific to Denmark's coordinate grid.
- **DTM not used here.** The terrain model (DTM, bare ground) is not consumed by any script in this pipeline. Download it only if another application requires it — you can safely skip it when choosing what to download.
- **No reprojection.** All outputs stay in EPSG:25832 (UTM 32N). Isaac Sim import treats the OBJ as a flat metric space.
- **Sea/coastal tiles.** Satellite coverage is incomplete over the sea. The download script warns about partial archives; the mesh pipeline still works on land portions.
- **Large textures require disk space.** A full 10×10 km tile at full satellite resolution (12.5 cm/px) would be 80,000×80,000 px. The `--size` argument downsamples to a manageable target; `--tiles N` avoids holding the full mosaic in RAM.
