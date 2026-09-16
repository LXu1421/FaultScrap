#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06a-ScarpTraceCleaning.py -- Pre-process raw scarp trace point clouds.

Read ``data/HighCon/Trace*.xlsx`` point clouds and produce cleaned, ordered
polylines suitable for the parameter-profiling pipeline in 06b.

Stages
------
0.  Diagnose  --  density-coloured scatter on hillshade; report scatter width,
    gap locations, branching patterns.
1.  Outlier rejection  --  kNN density filter + DEM terrain (slope) filter.
2.  Segment  --  DBSCAN spatial clustering replaces sequential-gap splitting.
    Each cluster's guide centerline length is the recovery target; extraction
    radius and fragment merging are relaxed until >=80% of that length is
    recovered (or relaxation is exhausted, in which case it is flagged).
    Recovery is measured as projected span along the cluster's principal
    direction, not raw polyline length, so ratios stay in [0, 1].
    Resulting fragments are then sorted north->south, grouped by break type,
    and merged where strike agrees within a tolerance, capping each file at
    fewer than 16 final segments.
3.  Centerline  --  rasterise -> skeletonize -> graph longest-path (guide), then
    snap to ordered original measured points near guide centerline.
4.  Smooth & simplify  --  light Savitzky-Golay smoothing of measured points,
    preserving geometry.
5.  QC  --  overlay cleaned lines on hillshade/slope; flag low-confidence
    segments; report per-segment statistics.

Outputs are written to ``data/HighCon/cleaned_traces/``:

  trace_XX.csv          Cleaned ordered polyline (columns: x, y)
  manifest.csv          Per-trace metadata for 06b consumption
  qc/                   Diagnostic plots per stage

Usage
-----
    .venv/Scripts/python.exe 06a-ScarpTraceCleaning.py
    .venv/Scripts/python.exe 06a-ScarpTraceCleaning.py --dem-source path.tif
    .venv/Scripts/python.exe 06a-ScarpTraceCleaning.py --skip-diagnose
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter, find_peaks
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
import networkx as nx
from shapely.geometry import LineString, Point
from shapely import simplify

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from derivative_common import DATA_DIR, OUT_DIR, safe_makedirs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# Constants
# ===================================================================
HIGHCON_DIR = DATA_DIR / "HighCon"
CLEANED_DIR = HIGHCON_DIR / "cleaned_traces"
QC_DIR = CLEANED_DIR / "qc"

KNOWN_TRACE_VALIDATION_DIR = DATA_DIR / "derivatives" / "known_trace_scarp_validation"
DEFAULT_ALL_INFERRED_FILES = [
    KNOWN_TRACE_VALIDATION_DIR / "lidar_0_5m" / "dem_wide_inferred_points.csv",
    KNOWN_TRACE_VALIDATION_DIR / "dem10m" / "all_inferred_points.csv",
]
TRACE_INPUT_FILES = ["Trace1.xlsx", "Trace2.xlsx", "Trace3.xlsx"]

DEM_PATH = OUT_DIR / "lidar_0_5m_epsg28355_10m_smoke.tif"

# Stage 1 - outlier rejection
KNN_K = 8                   # k-th nearest neighbour for density
DENSITY_MAD_FACTOR = 5.0    # reject if kNN dist > median + factor * MAD
TERRAIN_SLOPE_MIN_DEG = 0.1   # keep gentle slopes; scarps can be low-angle in DEM
TERRAIN_SLOPE_MAX_DEG = 85.0  # keep valid steep scarps and terrain artifacts only at extreme angles
TERRAIN_WINDOW_PX = 3        # half-window for slope sampling (px on each side)
TERRAIN_SLOPE_SMOOTH_PX = 3  # Gaussian smoothing of slope grid before sampling

# Stage 2 - DBSCAN
DBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_POINTS = 10
MIN_CLUSTER_EXTENT_M = 50.0
PARALLEL_SEPARATION_MIN_GAP_M = 15.0
PARALLEL_DIRECTION_TOLERANCE_DEG = 20.0
BUFFER_WIDTH_FOR_SEPARATION = 10.0
MAX_POINTS_PER_TRACE_SECTION = 50

# Stage 3 - centerline extraction
RASTER_CELL_SIZE_M = 5.0    # grid cell for rasterisation
SKELETON_SIGMA = 0.8        # Gaussian pre-smooth of density mask
SPUR_LENGTH_THRESHOLD = 4   # prune skeleton branches shorter than this (pixels)
MIN_CENTERLINE_POINTS = 5   # minimum vertices in a valid centerline
POINT_SPACING_MIN_M = 0.5  # minimum spacing between selected output points
POINT_SPACING_MAX_M = 30.0  # maximum allowed gap before interpolation
CENTERLINE_SEARCH_RADIUS_M = 30.0  # snap distance to original points

# Break type / tier handling
BREAK_TYPE_COLUMN = "break_type"
VALID_BREAK_TYPES = {"p", "i", "s", "unknown"}
BREAK_TYPE_WEIGHTS = {
    "p": 1.0,
    "i": 0.8,
    "s": 0.6,
    "unknown": 0.5,
}
BREAK_TYPE_CONSISTENCY_THRESHOLD = 0.7
BREAK_TYPE_MATCH_ROUND_DECIMALS = 3
BREAK_TYPE_MATCH_MAX_DISTANCE_M = 2.0

# Strict quality control
MIN_TRACE_LENGTH_M = 100.0
MIN_POINTS_PER_TRACE = 10
MAX_TRACES_PER_FILE = 5
MIN_BREAK_TYPE_CONSISTENCY = 0.8
MIN_POINT_SPACING_M = 10.0
MAX_POINT_SPACING_M = 30.0
TARGET_POINT_SPACING_M = 10.0
POINTS_PER_KM = 100

# Segment merge controls
MERGE_MAX_GAP_M = 50.0
MERGE_DIRECTION_TOLERANCE_DEG = 30.0
MERGE_BREAK_TYPE_MUST_MATCH = True
MERGE_CLUSTER_GAP_M = 200.0
GREEDY_MERGE_MAX_DIST_M = 1000.0

# Per-cluster length recovery target: final trace(s) from a cluster must
# retain at least this fraction of that cluster's principal-axis extent
# (projected span along the cluster's main direction).  Both the original
# and recovered lengths are measured as projected spans along the same
# direction, so the ratio is a true spatial-coverage fraction in [0, 1].
RECOVERY_TARGET_FRACTION = 0.8
RECOVERY_RADIUS_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]   # progressive snap-radius relaxation
RECOVERY_MERGE_GAP_MULTIPLIERS = [1.0, 2.0, 3.0]     # progressive fragment-merge relaxation

# Final segment consolidation: sort north->south, group by break type, then
# merge segments whose strike agrees within a tolerance, until under the cap.
MAX_SEGMENTS_PER_FILE = 15
SEGMENT_MERGE_STRIKE_TOL_DEG = 30.0
SEGMENT_MERGE_MAX_TOL_DEG = 90.0
SEGMENT_MERGE_TOL_STEP_DEG = 15.0

# Final global merge parameters
GLOBAL_MERGE_MAX_GAP_M = 300.0
GLOBAL_MERGE_DIRECTION_TOL_DEG = 30.0
FINAL_MAX_TRACES_PER_FILE = 25

# Short fragment preservation: traces too short to be полноценными segments
# are still saved as "fragment" type for visualisation in 06b.
FRAGMENT_MIN_POINTS = 3        # absolute minimum points to keep a fragment
FRAGMENT_MIN_LENGTH_M = 20.0   # minimum length (m) to save as fragment

# Stage 4 - smoothing
SAVGOL_WINDOW = 11          # must be odd; use smaller window for measured points
SAVGOL_POLYORDER = 3
DP_TOLERANCE_M = 2.0        # Douglas-Peucker tolerance

PLOT_DPI = 200


# ===================================================================
# Helpers
# ===================================================================

def _compute_hillshade(dem, transform):
    """Compute hillshade from a 2-D elevation array (azimuth=315, alt=8)."""
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dzdx = np.gradient(dem, axis=1) / res_x
    dzdy = np.gradient(dem, axis=0) / res_y
    az_rad = np.radians(315.0 - 90.0)
    alt_rad = np.radians(8.0)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(dzdy, -dzdx)
    hs = (np.sin(alt_rad) * np.sin(slope)
          + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect))
    return np.clip(hs * 254.0 / np.sin(np.radians(8.0)) + 254.0 / 2, 0, 255)


def _dem_window_for_bounds(src, x0, x1, y0, y1):
    """Return (window, transform) covering the given bounds."""
    x0 = max(x0, src.bounds.left)
    x1 = min(x1, src.bounds.right)
    y0 = max(y0, src.bounds.bottom)
    y1 = min(y1, src.bounds.top)
    win = from_bounds(x0, y0, x1, y1, src.transform)
    win_transform = src.window_transform(win)
    return win, win_transform


def normalize_break_types(break_types):
    """Normalize break type labels to p/i/s/unknown."""
    normalized = []
    for bt in break_types:
        if pd.isna(bt):
            normalized.append("unknown")
            continue
        bt_str = str(bt).strip().lower()
        if bt_str in {"p", "primary", "1", "main"}:
            normalized.append("p")
        elif bt_str in {"i", "intermediate", "2", "mid"}:
            normalized.append("i")
        elif bt_str in {"s", "secondary", "3", "minor"}:
            normalized.append("s")
        else:
            normalized.append("unknown")
    return np.array(normalized, dtype=object)


def load_points_with_attributes(fpath):
    """Load point coordinates plus break_type from CSV/XLSX."""
    suffix = fpath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(fpath)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(fpath)
    else:
        raise ValueError(f"Unsupported input format: {fpath}")

    column_mapping = {}
    for col in df.columns:
        col_key = str(col).strip().lower().replace(" ", "_")
        if col_key in {"x", "xcoord", "easting", "east", "lon", "longitude"}:
            column_mapping[col] = "x"
        elif col_key in {"y", "ycoord", "northing", "north", "lat", "latitude"}:
            column_mapping[col] = "y"
        elif col_key in {"break_type", "type", "tier", "break", "scarp_type"}:
            # Keep true break_type if present; otherwise tier acts as fallback source.
            if col_key == "tier" and "break_type" not in [
                str(c).strip().lower().replace(" ", "_") for c in df.columns
            ]:
                column_mapping[col] = "tier"
            elif col_key == "break_type":
                column_mapping[col] = "break_type"

    df = df.rename(columns=column_mapping)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Could not find X/Y columns in {fpath}. Available: {df.columns.tolist()}"
        )

    coords = df[["x", "y"]].values.astype(np.float64)

    if "break_type" in df.columns:
        raw_break_types = df["break_type"].to_numpy()
    elif "tier" in df.columns:
        raw_break_types = df["tier"].to_numpy()
    else:
        raw_break_types = np.array(["unknown"] * len(df), dtype=object)

    break_types = normalize_break_types(raw_break_types)
    attributes = {
        "break_type": break_types,
        "original_indices": np.arange(len(df)),
        "source_file": fpath.name,
    }

    unique_types, counts = np.unique(break_types, return_counts=True)
    type_info = ", ".join([f"{t}: {c}" for t, c in zip(unique_types, counts)])
    print(f"  Loaded {len(coords)} points with break types [{type_info}]")
    return coords, attributes


def _read_xy_table(fpath):
    """Read a table and standardize coordinate columns to x/y."""
    suffix = fpath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(fpath)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(fpath)
    else:
        raise ValueError(f"Unsupported input format: {fpath}")

    column_mapping = {}
    for col in df.columns:
        col_key = str(col).strip().lower().replace(" ", "_")
        if col_key in {"x", "xcoord", "easting", "east", "lon", "longitude"}:
            column_mapping[col] = "x"
        elif col_key in {"y", "ycoord", "northing", "north", "lat", "latitude"}:
            column_mapping[col] = "y"
        elif col_key in {"break_type", "type", "tier", "break", "scarp_type"}:
            if col_key == "tier" and "break_type" not in [
                str(c).strip().lower().replace(" ", "_") for c in df.columns
            ]:
                column_mapping[col] = "tier"
            elif col_key == "break_type":
                column_mapping[col] = "break_type"

    df = df.rename(columns=column_mapping)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Could not find X/Y columns in {fpath}. Available: {df.columns.tolist()}"
        )
    return df


def build_break_type_lookup(all_inferred_paths):
    """Build coordinate->dominant break_type lookup from inferred-point CSVs."""
    rows = []
    for fpath in all_inferred_paths:
        if not fpath.exists():
            print(f"  [Warning] Missing all_inferred file: {fpath}")
            continue
        df = _read_xy_table(fpath)
        if "break_type" in df.columns:
            raw_bt = df["break_type"].to_numpy()
        elif "tier" in df.columns:
            raw_bt = df["tier"].to_numpy()
        else:
            raw_bt = np.array(["unknown"] * len(df), dtype=object)
        bt = normalize_break_types(raw_bt)
        part = pd.DataFrame(
            {
                "x": df["x"].astype(np.float64),
                "y": df["y"].astype(np.float64),
                "break_type": bt,
            }
        )
        rows.append(part)

    if not rows:
        return {
            "coord_to_type": {},
            "tree": None,
            "tree_points": np.empty((0, 2), dtype=np.float64),
            "tree_types": np.array([], dtype=object),
        }

    combined = pd.concat(rows, ignore_index=True)
    combined["rx"] = combined["x"].round(BREAK_TYPE_MATCH_ROUND_DECIMALS)
    combined["ry"] = combined["y"].round(BREAK_TYPE_MATCH_ROUND_DECIMALS)

    coord_to_type = {}
    grouped = combined.groupby(["rx", "ry"]) ["break_type"]
    for (rx, ry), series in grouped:
        bt_values = series.to_numpy(dtype=object)
        dom, _ = get_dominant_break_type(bt_values)
        coord_to_type[(float(rx), float(ry))] = dom

    tree_points = combined[["x", "y"]].to_numpy(dtype=np.float64)
    tree_types = combined["break_type"].to_numpy(dtype=object)
    tree = cKDTree(tree_points) if len(tree_points) else None

    unique_types, counts = np.unique(tree_types, return_counts=True)
    type_info = ", ".join([f"{t}: {c}" for t, c in zip(unique_types, counts)])
    print(
        f"[Break Type Lookup] {len(tree_points)} points, "
        f"{len(coord_to_type)} unique rounded coords [{type_info}]"
    )

    return {
        "coord_to_type": coord_to_type,
        "tree": tree,
        "tree_points": tree_points,
        "tree_types": tree_types,
    }


def assign_break_types_by_lookup(coords, lookup):
    """Assign break types to coords using rounded-key match then nearest fallback."""
    coord_to_type = lookup["coord_to_type"]
    tree = lookup["tree"]
    tree_types = lookup["tree_types"]

    assigned = np.array(["unknown"] * len(coords), dtype=object)
    exact_hits = 0
    nearest_hits = 0

    for i, (x, y) in enumerate(coords):
        key = (
            float(np.round(x, BREAK_TYPE_MATCH_ROUND_DECIMALS)),
            float(np.round(y, BREAK_TYPE_MATCH_ROUND_DECIMALS)),
        )
        if key in coord_to_type:
            assigned[i] = coord_to_type[key]
            exact_hits += 1
            continue

        if tree is not None and len(tree_types) > 0:
            dist, idx = tree.query(np.array([x, y], dtype=np.float64), k=1)
            if np.isfinite(dist) and float(dist) <= BREAK_TYPE_MATCH_MAX_DISTANCE_M:
                assigned[i] = str(tree_types[int(idx)])
                nearest_hits += 1

    print(
        f"  Break-type assignment: exact={exact_hits}, nearest={nearest_hits}, "
        f"unknown={int(np.sum(assigned == 'unknown'))}"
    )
    return assigned


def load_trace_points_with_break_types(fpath, lookup):
    """Load Trace*.xlsx coords and assign break types via inferred-point lookup."""
    df = _read_xy_table(fpath)
    coords = df[["x", "y"]].to_numpy(dtype=np.float64)
    break_types = assign_break_types_by_lookup(coords, lookup)
    attrs = {
        "break_type": break_types,
        "original_indices": np.arange(len(df)),
        "source_file": fpath.name,
    }
    dom, cons = get_dominant_break_type(break_types)
    print(f"  Loaded {len(coords)} trace points; dominant type={dom} ({cons:.1%})")
    return coords, attrs


def get_dominant_break_type(points_break_types):
    """Return (dominant_type, consistency_fraction)."""
    if len(points_break_types) == 0:
        return "unknown", 0.0
    unique_types, counts = np.unique(points_break_types, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    dominant_type = str(unique_types[dominant_idx])
    consistency = float(counts[dominant_idx] / len(points_break_types))
    return dominant_type, consistency


# ===================================================================
# Stage 0 -- Diagnose before you clean
# ===================================================================

def stage0_diagnose(coords, file_name, dem_hs, dem_transform, qc_dir):
    """Plot raw point cloud coloured by kNN density on hillshade."""
    print(f"\n  [Stage 0] Diagnosing {file_name} ({len(coords)} pts)")

    tree = cKDTree(coords)
    k = min(KNN_K, len(coords) - 1)
    dists_knn, _ = tree.query(coords, k=k + 1)
    nn_dist = dists_knn[:, -1]  # distance to k-th NN

    median_d = np.median(nn_dist)
    mad_d = np.median(np.abs(nn_dist - median_d))
    p95_d = np.percentile(nn_dist, 95)

    # Gap analysis: sort by x, find gaps in sequential order
    diffs = np.diff(coords, axis=0)
    seq_dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    gap_threshold = median_d * 10 if median_d > 0 else 50.0
    gap_mask = seq_dists > gap_threshold
    n_gaps = gap_mask.sum()
    gap_distances = seq_dists[gap_mask] if n_gaps > 0 else np.array([])

    print(f"    kNN distance:  median={median_d:.1f} m, MAD={mad_d:.1f} m, "
          f"p95={p95_d:.1f} m")
    print(f"    Sequential gaps (> {gap_threshold:.0f} m): {n_gaps}", end="")
    if n_gaps > 0:
        print(f" at distances {gap_distances[:10]}")
    else:
        print()

    # Plot: hillshade + density-coloured scatter
    fig, ax = plt.subplots(figsize=(14, 10))

    x0, x1 = coords[:, 0].min() - 200, coords[:, 0].max() + 200
    y0, y1 = coords[:, 1].min() - 200, coords[:, 1].max() + 200

    if dem_hs is not None:
        inv = ~dem_transform
        c0, r0 = inv * (x0, y1)
        c1, r1 = inv * (x1, y0)
        r0i, r1i = int(max(0, min(r0, r1))), int(min(dem_hs.shape[0], max(r0, r1)))
        c0i, c1i = int(max(0, min(c0, c1))), int(min(dem_hs.shape[1], max(c0, c1)))
        if r1i > r0i and c1i > c0i:
            hs_crop = dem_hs[r0i:r1i, c0i:c1i]
            extent = [
                dem_transform.c + c0i * dem_transform.a,
                dem_transform.c + c1i * dem_transform.a,
                dem_transform.f + r1i * dem_transform.e,
                dem_transform.f + r0i * dem_transform.e,
            ]
            ax.imshow(hs_crop, extent=extent, cmap="gray", aspect="equal",
                      alpha=0.5)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=nn_dist, s=1,
                    cmap="YlOrRd", alpha=0.7)
    plt.colorbar(sc, ax=ax, label=f"{k}-NN distance (m)")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(f"Stage 0 -- {file_name}\n"
                 f"n={len(coords)}, median kNN={median_d:.1f} m, "
                 f"gaps={n_gaps}", fontsize=11)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(qc_dir / f"stage0_{Path(file_name).stem}.png", dpi=PLOT_DPI)
    plt.close(fig)

    return {
        "median_knn_m": float(median_d),
        "mad_knn_m": float(mad_d),
        "p95_knn_m": float(p95_d),
        "n_gaps": int(n_gaps),
        "gap_threshold_m": float(gap_threshold),
    }


# ===================================================================
# Stage 1 -- Outlier rejection
# ===================================================================

def stage1_outlier_rejection(coords, dem_src, diagnostics):
    """Apply density filter + terrain filter.

    Steps:
      0. Remove exact duplicates (rounded to 0.1 m) -- the raw Excel files
         contain many identical coordinate pairs.
      (a) Density filter: kNN distance with median+MAD threshold.
      (b) Terrain filter: sample DEM slope at each surviving point using
          vectorized map_coordinates (fast); drop near-flat points.
    """
    n_orig = len(coords)

    # --- (0) Deduplicate ---
    rounded = np.round(coords, 1)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx.sort()
    coords = coords[unique_idx]
    n_dedup = len(coords)
    print(f"  [Stage 0] Deduplicated: {n_orig} -> {n_dedup} unique points")
    n_orig = n_dedup

    # --- (a) Density filter ---
    tree = cKDTree(coords)
    k = min(KNN_K, len(coords) - 1)
    dists_knn, _ = tree.query(coords, k=k + 1)
    nn_dist = dists_knn[:, -1]

    median_d = np.median(nn_dist)
    mad_d = np.median(np.abs(nn_dist - median_d))
    threshold = median_d + DENSITY_MAD_FACTOR * max(mad_d, 0.1)

    density_mask = nn_dist <= threshold
    dropped_density = (~density_mask).sum()
    print(f"  [Stage 1a] Density filter: dropped {dropped_density} / {n_orig} "
          f"(threshold {threshold:.1f} m)")

    coords_density = coords[density_mask]
    mask = density_mask.copy()

    # --- (b) Terrain filter ---
    if dem_src is None:
        print(f"  [Stage 1b] No DEM available; skipping terrain filter")
        kept = coords_density
        return kept, {
            "n_original": n_orig,
            "n_after_density": int(density_mask.sum()),
            "n_after_terrain": int(density_mask.sum()),
            "density_threshold_m": float(threshold),
        }

    # Compute full DEM slope grid once (vectorized)
    print(f"  [Stage 1b] Computing DEM slope grid...")
    dem_data = dem_src.read(1).astype(np.float64)
    nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
    dem_data[dem_data == nodata] = np.nan

    res_x = abs(dem_src.transform.a)
    res_y = abs(dem_src.transform.e)
    dzdx = np.gradient(dem_data, axis=1) / res_x
    dzdy = np.gradient(dem_data, axis=0) / res_y
    slope_grid = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Sample slope at each surviving point via map_coordinates
    inv = ~dem_src.transform
    cols, rows = inv * (coords_density[:, 0], coords_density[:, 1])

    valid_dem = (
        (cols >= TERRAIN_WINDOW_PX) & (cols < dem_src.width - TERRAIN_WINDOW_PX) &
        (rows >= TERRAIN_WINDOW_PX) & (rows < dem_src.height - TERRAIN_WINDOW_PX)
    )

    slopes = np.full(len(coords_density), np.nan)
    if np.any(valid_dem):
        valid_idx = np.where(valid_dem)[0]
        coords_rc = np.array([rows[valid_idx], cols[valid_idx]])
        slopes[valid_idx] = map_coordinates(
            slope_grid, coords_rc, order=1, mode="constant", cval=np.nan
        )

    terrain_mask = (
        (slopes >= TERRAIN_SLOPE_MIN_DEG) &
        (slopes <= TERRAIN_SLOPE_MAX_DEG) |
        ~valid_dem  # keep points outside DEM extent
    )
    dropped_terrain = (~terrain_mask).sum()
    print(f"  [Stage 1b] Terrain filter: dropped {dropped_terrain} "
          f"(slope range {TERRAIN_SLOPE_MIN_DEG}-{TERRAIN_SLOPE_MAX_DEG} deg)")
    valid_slopes = slopes[valid_dem & terrain_mask]
    if len(valid_slopes) > 0:
        med_slope = np.median(valid_slopes)
        print(f"          Median slope of kept points: {med_slope:.1f} deg")

    kept = coords_density[terrain_mask]
    print(f"  [Stage 1] Total: {len(kept)}/{n_orig} points retained")
    return kept, {
        "n_original": n_orig,
        "n_after_density": int(density_mask.sum()),
        "n_after_terrain": int(terrain_mask.sum()),
        "density_threshold_m": float(threshold),
    }


def stage1_outlier_rejection_with_types(coords, break_types, dem_src, diagnostics):
    """Stage 1 filtering that keeps break_type aligned with coordinates."""
    if len(coords) != len(break_types):
        raise ValueError("coords and break_types length mismatch")

    n_orig = len(coords)
    rounded = np.round(coords, 1)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx.sort()

    coords = coords[unique_idx]
    break_types = break_types[unique_idx]
    n_dedup = len(coords)
    print(f"  [Stage 0] Deduplicated: {n_orig} -> {n_dedup} unique points")
    n_orig = n_dedup

    if len(coords) < 2:
        return coords, break_types, {
            "n_original": n_orig,
            "n_after_density": int(len(coords)),
            "n_after_terrain": int(len(coords)),
            "density_threshold_m": 0.0,
        }

    tree = cKDTree(coords)
    k = min(KNN_K, len(coords) - 1)
    dists_knn, _ = tree.query(coords, k=k + 1)
    nn_dist = dists_knn[:, -1]

    median_d = np.median(nn_dist)
    mad_d = np.median(np.abs(nn_dist - median_d))
    threshold = median_d + DENSITY_MAD_FACTOR * max(mad_d, 0.1)

    density_mask = nn_dist <= threshold
    coords_density = coords[density_mask]
    break_types_density = break_types[density_mask]
    print(
        f"  [Stage 1a] Density filter: dropped {(~density_mask).sum()} / {n_orig} "
        f"(threshold {threshold:.1f} m)"
    )

    if dem_src is None:
        print("  [Stage 1b] No DEM available; skipping terrain filter")
        return coords_density, break_types_density, {
            "n_original": n_orig,
            "n_after_density": int(density_mask.sum()),
            "n_after_terrain": int(density_mask.sum()),
            "density_threshold_m": float(threshold),
        }

    print("  [Stage 1b] Computing DEM slope grid...")
    dem_data = dem_src.read(1).astype(np.float64)
    nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
    dem_data[dem_data == nodata] = np.nan

    res_x = abs(dem_src.transform.a)
    res_y = abs(dem_src.transform.e)
    dzdx = np.gradient(dem_data, axis=1) / res_x
    dzdy = np.gradient(dem_data, axis=0) / res_y
    slope_grid = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    inv = ~dem_src.transform
    cols, rows = inv * (coords_density[:, 0], coords_density[:, 1])
    valid_dem = (
        (cols >= TERRAIN_WINDOW_PX) & (cols < dem_src.width - TERRAIN_WINDOW_PX) &
        (rows >= TERRAIN_WINDOW_PX) & (rows < dem_src.height - TERRAIN_WINDOW_PX)
    )

    slopes = np.full(len(coords_density), np.nan)
    if np.any(valid_dem):
        valid_idx = np.where(valid_dem)[0]
        coords_rc = np.array([rows[valid_idx], cols[valid_idx]])
        slopes[valid_idx] = map_coordinates(
            slope_grid, coords_rc, order=1, mode="constant", cval=np.nan
        )

    terrain_mask = (
        (slopes >= TERRAIN_SLOPE_MIN_DEG) &
        (slopes <= TERRAIN_SLOPE_MAX_DEG) |
        ~valid_dem
    )
    print(
        f"  [Stage 1b] Terrain filter: dropped {(~terrain_mask).sum()} "
        f"(slope range {TERRAIN_SLOPE_MIN_DEG}-{TERRAIN_SLOPE_MAX_DEG} deg)"
    )

    coords_kept = coords_density[terrain_mask]
    break_types_kept = break_types_density[terrain_mask]
    print(f"  [Stage 1] Total: {len(coords_kept)}/{n_orig} points retained")
    return coords_kept, break_types_kept, {
        "n_original": n_orig,
        "n_after_density": int(density_mask.sum()),
        "n_after_terrain": int(terrain_mask.sum()),
        "density_threshold_m": float(threshold),
    }


# ===================================================================
# Stage 2 -- Segment into real sub-traces (DBSCAN)
# ===================================================================

def stage2_segment(coords):
    """Segment the cleaned cloud using DBSCAN spatial clustering."""
    if len(coords) < MIN_CLUSTER_POINTS:
        return []

    tree = cKDTree(coords)
    k = min(5, len(coords) - 1)
    dists, _ = tree.query(coords, k=k + 1)
    nn_dist = dists[:, 1:].mean(axis=1)
    median_spacing = np.median(nn_dist)

    eps = max(median_spacing * 2.0, 5.0)
    print(f"  [Stage 2] DBSCAN: eps={eps:.1f} m, min_samples={DBSCAN_MIN_SAMPLES} "
          f"(median spacing={median_spacing:.1f} m)")

    labels = DBSCAN(
        eps=eps, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1
    ).fit_predict(coords)

    clusters = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_pts = coords[labels == label]
        if len(cluster_pts) < MIN_CLUSTER_POINTS:
            continue
        extent = np.sqrt(
            np.ptp(cluster_pts[:, 0])**2 + np.ptp(cluster_pts[:, 1])**2
        )
        if extent < MIN_CLUSTER_EXTENT_M:
            continue
        clusters.append(cluster_pts)

    n_noise = (labels == -1).sum()
    print(f"  [Stage 2] Found {len(clusters)} clusters "
          f"({n_noise} noise points)")
    for i, c in enumerate(clusters):
        ext = np.sqrt(np.ptp(c[:, 0])**2 + np.ptp(c[:, 1])**2)
        print(f"    Cluster {i}: {len(c)} pts, extent={ext:.0f} m")

    return clusters


def detect_and_separate_parallel_traces(
    cluster_pts,
    min_gap=PARALLEL_SEPARATION_MIN_GAP_M,
    direction_tol=PARALLEL_DIRECTION_TOLERANCE_DEG,
):
    """Detect and separate likely parallel traces within one point cluster."""
    if len(cluster_pts) < MIN_CLUSTER_POINTS * 2:
        return [cluster_pts]

    centered = cluster_pts - np.mean(cluster_pts, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if np.any(~np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0:
        return [cluster_pts]

    main_direction = eigenvectors[:, -1]
    perpendicular_direction = eigenvectors[:, 0]
    extent_ratio = (
        np.sqrt(eigenvalues[-1] / eigenvalues[0])
        if eigenvalues[0] > 0 else 1.0
    )
    main_azimuth = np.degrees(np.arctan2(main_direction[0], main_direction[1])) % 360

    print(
        f"    Parallel trace check: extent_ratio={extent_ratio:.1f}, "
        f"main_dir={main_azimuth:.0f} deg, dir_tol={direction_tol:.0f} deg"
    )

    if extent_ratio < 2.0:
        return [cluster_pts]

    projections = np.dot(centered, perpendicular_direction)
    n_bins = max(8, min(50, len(cluster_pts) // 10))
    hist, bin_edges = np.histogram(projections, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    hist_smooth = gaussian_filter(hist.astype(float), sigma=1.0)
    if hist_smooth.max() <= 0:
        return [cluster_pts]

    peaks, _ = find_peaks(
        hist_smooth,
        height=hist_smooth.max() * 0.1,
        distance=3,
        prominence=hist_smooth.max() * 0.05,
    )
    print(f"    Found {len(peaks)} modes in perpendicular direction")
    if len(peaks) < 2:
        return [cluster_pts]

    peak_positions = bin_centers[peaks]
    peak_gaps = np.diff(peak_positions)
    if not np.any(peak_gaps >= float(min_gap)):
        print(
            "    Peaks too close together "
            f"(max gap={peak_gaps.max():.1f} m); treating as single trace"
        )
        return [cluster_pts]

    boundaries = []
    for i in range(len(peaks) - 1):
        if peak_gaps[i] < float(min_gap):
            continue
        valley_start = peaks[i]
        valley_end = peaks[i + 1]
        if valley_end <= valley_start + 1:
            continue
        valley_local = np.argmin(hist_smooth[valley_start:valley_end])
        valley_idx = valley_start + valley_local
        boundaries.append(bin_centers[valley_idx])

    if not boundaries:
        return [cluster_pts]

    boundaries = [-np.inf] + sorted(boundaries) + [np.inf]
    separated_traces = []
    for i in range(len(boundaries) - 1):
        mask = (projections >= boundaries[i]) & (projections < boundaries[i + 1])
        trace_pts = cluster_pts[mask]
        if len(trace_pts) < MIN_CLUSTER_POINTS:
            continue

        trace_centered = trace_pts - np.mean(trace_pts, axis=0)
        trace_cov = np.cov(trace_centered.T)
        trace_eigvals, _ = np.linalg.eigh(trace_cov)
        if np.any(~np.isfinite(trace_eigvals)):
            continue

        trace_extent = float(np.sqrt(max(trace_eigvals[-1], 0.0)))
        if trace_extent >= MIN_CLUSTER_EXTENT_M:
            separated_traces.append(trace_pts)
            print(
                f"    Separated trace {i}: {len(trace_pts)} points, "
                f"extent={trace_extent:.0f} m"
            )

    if len(separated_traces) >= 2:
        print(f"    Successfully separated into {len(separated_traces)} parallel traces")
        return separated_traces

    print("    Could not reliably separate parallel traces, keeping single cluster")
    return [cluster_pts]


def point_to_line_distance_sampling(
    centerline_guide,
    candidate_points,
    max_distance=PARALLEL_SEPARATION_MIN_GAP_M / 2.0,
):
    """Sample points along a guide line with a strict distance cap."""
    if len(centerline_guide) < 2 or len(candidate_points) < 2:
        return np.empty((0, 2), dtype=np.float64)

    guide_line = LineString(centerline_guide)
    total_length = float(guide_line.length)
    sample_spacing = max(POINT_SPACING_MIN_M, 1.0)
    if total_length < sample_spacing:
        return np.empty((0, 2), dtype=np.float64)

    tree = cKDTree(candidate_points)
    selected_indices = set()
    selected_points = []

    k = min(10, len(candidate_points))
    num_samples = max(2, int(total_length / sample_spacing) + 1)
    for dist in np.linspace(0.0, total_length, num_samples):
        point_on_line = guide_line.interpolate(float(dist))
        pt_array = np.array([point_on_line.x, point_on_line.y], dtype=np.float64)
        distances, indices = tree.query(
            pt_array,
            k=k,
            distance_upper_bound=float(max_distance),
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        valid_mask = np.isfinite(distances) & (distances < float(max_distance))
        if not np.any(valid_mask):
            continue

        valid_distances = distances[valid_mask]
        valid_indices = indices[valid_mask]
        order = np.argsort(valid_distances)
        for pos in order:
            idx = int(valid_indices[pos])
            if 0 <= idx < len(candidate_points) and idx not in selected_indices:
                selected_indices.add(idx)
                selected_points.append(candidate_points[idx])
                break

    if not selected_points:
        return np.empty((0, 2), dtype=np.float64)

    selected_points = np.array(selected_points, dtype=np.float64)
    guide_line = LineString(centerline_guide)
    proj = np.array([guide_line.project(Point(p)) for p in selected_points])
    order = np.argsort(proj)
    selected_points = selected_points[order]
    return selected_points


def tier_weighted_point_selection(
    guide_point,
    candidate_coords,
    candidate_break_types,
    target_break_type,
    max_distance,
):
    """Select candidate index using combined break-type and distance score."""
    if len(candidate_coords) == 0:
        return None, 0.0

    distances = np.sqrt(np.sum((candidate_coords - guide_point) ** 2, axis=1))
    within = distances <= float(max_distance)
    if not np.any(within):
        return None, 0.0

    scores = np.full(len(candidate_coords), -1.0, dtype=np.float64)
    for i in range(len(candidate_coords)):
        if not within[i]:
            continue

        distance_score = 1.0 / (1.0 + distances[i] / 5.0)
        candidate_type = str(candidate_break_types[i])
        if candidate_type == target_break_type:
            type_score = 1.0
        elif {candidate_type, target_break_type} == {"p", "i"}:
            type_score = 0.3
        else:
            type_score = BREAK_TYPE_WEIGHTS.get(candidate_type, 0.5)

        scores[i] = 0.6 * type_score + 0.4 * distance_score

    best_idx = int(np.argmax(scores))
    if scores[best_idx] <= 0:
        return None, 0.0
    return best_idx, float(scores[best_idx])


def extract_trace_with_break_type_consistency(
    cluster_pts,
    cluster_break_types,
    centerline_guide,
    max_distance=15.0,
):
    """Extract ordered points along a guide while favoring one break type."""
    if len(centerline_guide) < 2 or len(cluster_pts) < 2:
        return np.empty((0, 2), dtype=np.float64), np.array([], dtype=object)

    dominant_type, consistency = get_dominant_break_type(cluster_break_types)
    print(f"    Trace break type target: {dominant_type} ({consistency:.1%})")
    if consistency < BREAK_TYPE_CONSISTENCY_THRESHOLD:
        print(f"    WARNING: mixed types in cluster ({consistency:.1%} dominant)")

    tree = cKDTree(cluster_pts)
    guide_line = LineString(centerline_guide)
    total_length = float(guide_line.length)
    if total_length < POINT_SPACING_MIN_M:
        return np.empty((0, 2), dtype=np.float64), np.array([], dtype=object)

    num_samples = max(2, int(total_length / POINT_SPACING_MIN_M) + 1)
    selected_indices = set()
    selected_coords = []
    selected_types = []

    for dist in np.linspace(0.0, total_length, num_samples):
        p = guide_line.interpolate(float(dist))
        guide_pt = np.array([p.x, p.y], dtype=np.float64)
        dists, inds = tree.query(
            guide_pt,
            k=min(20, len(cluster_pts)),
            distance_upper_bound=float(max_distance) * 2.0,
        )

        dists = np.atleast_1d(dists)
        inds = np.atleast_1d(inds)
        valid = np.isfinite(dists) & (dists < float(max_distance) * 2.0)
        if not np.any(valid):
            continue

        valid_inds = inds[valid].astype(int)
        candidate_coords = cluster_pts[valid_inds]
        candidate_types = cluster_break_types[valid_inds]

        best_local_idx, score = tier_weighted_point_selection(
            guide_pt,
            candidate_coords,
            candidate_types,
            target_break_type=dominant_type,
            max_distance=max_distance,
        )
        if best_local_idx is None or score <= 0.3:
            continue

        global_idx = int(valid_inds[best_local_idx])
        if global_idx in selected_indices:
            continue
        selected_indices.add(global_idx)
        selected_coords.append(cluster_pts[global_idx])
        selected_types.append(cluster_break_types[global_idx])

    if len(selected_coords) < 3:
        return np.empty((0, 2), dtype=np.float64), np.array([], dtype=object)

    selected_coords = np.array(selected_coords, dtype=np.float64)
    selected_types = np.array(selected_types, dtype=object)
    final_type, final_consistency = get_dominant_break_type(selected_types)
    print(
        f"    Selected trace: {len(selected_coords)} pts, "
        f"type={final_type} ({final_consistency:.1%})"
    )
    return selected_coords, selected_types


def separate_parallel_traces_by_break_type(cluster_pts, cluster_break_types):
    """Split cluster into trace groups with break-type preference."""
    if len(cluster_pts) < MIN_CLUSTER_POINTS:
        return [cluster_pts], [cluster_break_types]

    unique_types = np.unique(cluster_break_types)
    if len(unique_types) <= 1:
        spatial_groups = detect_and_separate_parallel_traces(cluster_pts)
        grouped_types = []
        for pts in spatial_groups:
            tree = cKDTree(cluster_pts)
            _, idx = tree.query(pts, k=1)
            grouped_types.append(cluster_break_types[np.atleast_1d(idx).astype(int)])
        return spatial_groups, grouped_types

    print(f"    Found multiple break types: {list(unique_types)}")

    grouped_pts = []
    grouped_types = []
    for bt in unique_types:
        mask = cluster_break_types == bt
        type_pts = cluster_pts[mask]
        type_bts = cluster_break_types[mask]
        if len(type_pts) < MIN_CLUSTER_POINTS:
            continue
        extent = np.sqrt(np.ptp(type_pts[:, 0]) ** 2 + np.ptp(type_pts[:, 1]) ** 2)
        if extent < MIN_CLUSTER_EXTENT_M:
            continue
        grouped_pts.append(type_pts)
        grouped_types.append(type_bts)
        print(f"    Type-separated {bt}: {len(type_pts)} pts, extent={extent:.0f} m")

    if len(grouped_pts) >= 2:
        return grouped_pts, grouped_types

    # Fallback to spatial separation if type-only split is not coherent.
    spatial_groups = detect_and_separate_parallel_traces(cluster_pts)
    spatial_types = []
    tree = cKDTree(cluster_pts)
    for pts in spatial_groups:
        _, idx = tree.query(pts, k=1)
        spatial_types.append(cluster_break_types[np.atleast_1d(idx).astype(int)])
    return spatial_groups, spatial_types


def stage2_segment_with_break_types(coords, break_types):
    """Stage 2 segmentation with break-type aware pre-separation."""
    if len(coords) < MIN_CLUSTER_POINTS:
        return [], []

    trace_groups, trace_group_types = separate_parallel_traces_by_break_type(
        coords, break_types
    )

    all_clusters = []
    all_cluster_types = []
    total_noise = 0

    for trace_pts, trace_bts in zip(trace_groups, trace_group_types):
        if len(trace_pts) < MIN_CLUSTER_POINTS:
            continue

        tree = cKDTree(trace_pts)
        k = min(5, len(trace_pts) - 1)
        dists, _ = tree.query(trace_pts, k=k + 1)
        nn_dist = dists[:, 1:].mean(axis=1)
        median_spacing = np.median(nn_dist)
        eps = max(median_spacing * 4.0, 10.0)

        labels = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit_predict(trace_pts)

        total_noise += int((labels == -1).sum())
        for label in sorted(set(labels)):
            if label == -1:
                continue
            mask = labels == label
            cluster_pts = trace_pts[mask]
            cluster_bts = trace_bts[mask]
            if len(cluster_pts) < MIN_CLUSTER_POINTS:
                continue
            extent = np.sqrt(np.ptp(cluster_pts[:, 0]) ** 2 + np.ptp(cluster_pts[:, 1]) ** 2)
            if extent < MIN_CLUSTER_EXTENT_M:
                continue

            all_clusters.append(cluster_pts)
            all_cluster_types.append(cluster_bts)

    print(
        f"  [Stage 2 with break types] Found {len(all_clusters)} clusters "
        f"({total_noise} noise points)"
    )
    for i, (cluster, bts) in enumerate(zip(all_clusters, all_cluster_types)):
        ext = np.sqrt(np.ptp(cluster[:, 0]) ** 2 + np.ptp(cluster[:, 1]) ** 2)
        dom, cons = get_dominant_break_type(bts)
        print(
            f"    Cluster {i}: {len(cluster)} pts, extent={ext:.0f} m, "
            f"type={dom} ({cons:.1%})"
        )

    return all_clusters, all_cluster_types


def stage2_segment_enhanced(coords):
    """Stage 2 with pre-separation of potential parallel traces."""
    if len(coords) < MIN_CLUSTER_POINTS:
        return []

    parallel_traces = detect_and_separate_parallel_traces(coords)
    all_clusters = []
    total_noise = 0

    for trace_pts in parallel_traces:
        if len(trace_pts) < MIN_CLUSTER_POINTS:
            continue

        tree = cKDTree(trace_pts)
        k = min(5, len(trace_pts) - 1)
        dists, _ = tree.query(trace_pts, k=k + 1)
        nn_dist = dists[:, 1:].mean(axis=1)
        median_spacing = np.median(nn_dist)
        eps = max(median_spacing * 4.0, 10.0)

        labels = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit_predict(trace_pts)

        total_noise += int((labels == -1).sum())
        for label in sorted(set(labels)):
            if label == -1:
                continue
            cluster_pts = trace_pts[labels == label]
            if len(cluster_pts) < MIN_CLUSTER_POINTS:
                continue
            extent = np.sqrt(
                np.ptp(cluster_pts[:, 0])**2 + np.ptp(cluster_pts[:, 1])**2
            )
            if extent < MIN_CLUSTER_EXTENT_M:
                continue
            all_clusters.append(cluster_pts)

    print(
        f"  [Stage 2 Enhanced] Found {len(all_clusters)} clusters "
        f"from {len(parallel_traces)} trace groups ({total_noise} noise points)"
    )
    for i, c in enumerate(all_clusters):
        ext = np.sqrt(np.ptp(c[:, 0])**2 + np.ptp(c[:, 1])**2)
        print(f"    Cluster {i}: {len(c)} pts, extent={ext:.0f} m")

    return all_clusters


# ===================================================================
# Stage 3 -- Extract a centerline per cluster
# ===================================================================

def _rasterize_points(pts, cell_size):
    """Rasterise points onto a fine grid as a density mask.

    Each point splats a small Gaussian kernel onto the grid to avoid
    sparse, disconnected pixels when point spacing >> cell size.
    """
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    margin = cell_size * 5
    x0 -= margin
    x1 += margin
    y0 -= margin
    y1 += margin

    n_cols = int(np.ceil((x1 - x0) / cell_size)) + 1
    n_rows = int(np.ceil((y1 - y0) / cell_size)) + 1

    mask = np.zeros((n_rows, n_cols), dtype=np.float32)

    # Splat each point as a small kernel (3x3) to bridge gaps
    for px, py in pts:
        col_f = (px - x0) / cell_size
        row_f = (y1 - py) / cell_size
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = int(row_f) + dr
                c = int(col_f) + dc
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    dist = np.sqrt(dr*dr + dc*dc)
                    mask[r, c] += np.exp(-dist * dist / 2.0)

    return mask, (x0, y0, y1, cell_size)


def _skeleton_to_graph(skel):
    """Build a networkx graph from skeleton pixels (8-connectivity)."""
    G = nx.Graph()
    ys, xs = np.where(skel)
    pixel_set = set(zip(ys, xs))

    for r, c in pixel_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixel_set:
                    dist = np.sqrt(dr*dr + dc*dc)
                    G.add_edge((r, c), (nr, nc), weight=dist)

    return G


def _prune_spurs(G, threshold):
    """Remove short dead-end branches from the skeleton graph."""
    if G.number_of_nodes() == 0:
        return G

    endpoints = [n for n in G.nodes() if G.degree(n) == 1]

    pruned = G.copy()
    removed = set()
    for ep in endpoints:
        if ep not in pruned:
            continue
        path = [ep]
        current = ep
        while pruned.degree(current) == 1 or (current == ep and len(path) == 1):
            neighbours = list(pruned.neighbors(current))
            if not neighbours:
                break
            nxt = neighbours[0]
            if nxt in path:
                break
            path.append(nxt)
            current = nxt
            if pruned.degree(current) >= 3:
                break

        spur_length = len(path) - 1
        if spur_length < threshold and len(path) > 1:
            junction = path[-1]
            for node in path[:-1]:
                if node in pruned:
                    pruned.remove_node(node)
                    removed.add(node)

    if removed:
        # Remove isolated nodes
        isolated = [n for n in pruned.nodes() if pruned.degree(n) == 0]
        pruned.remove_nodes_from(isolated)

    return pruned


def _longest_path_in_tree(G):
    """Find the longest path in a tree via double BFS."""
    if G.number_of_nodes() == 0:
        return []

    nodes = list(G.nodes())
    if len(nodes) == 1:
        return [nodes[0]]

    # First BFS from arbitrary node
    dists1 = nx.single_source_dijkstra_path_length(G, nodes[0], weight="weight")
    far1 = max(dists1.items(), key=lambda x: x[1])[0]

    # Second BFS from far1
    dists2 = nx.single_source_dijkstra_path_length(G, far1, weight="weight")
    far2 = max(dists2.items(), key=lambda x: x[1])[0]

    path = nx.shortest_path(G, far1, far2, weight="weight")
    return path


def _longest_path_general(G):
    """Find approximate longest path for small general graphs.

    Tries endpoints (degree-1 nodes) pairwise; falls back to diameter.
    """
    if G.number_of_nodes() == 0:
        return []

    if G.number_of_nodes() <= 2:
        return list(G.nodes())

    # For trees, use the fast double-BFS
    if nx.is_tree(G):
        return _longest_path_in_tree(G)

    # General graph: try all pairs of endpoints
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    if len(endpoints) >= 2:
        best_path = []
        best_len = 0.0
        for i in range(len(endpoints)):
            try:
                dists = nx.single_source_dijkstra_path_length(
                    G, endpoints[i], weight="weight")
                for j in range(i + 1, len(endpoints)):
                    d = dists.get(endpoints[j], 0)
                    if d > best_len:
                        best_len = d
                        best_path = nx.shortest_path(
                            G, endpoints[i], endpoints[j], weight="weight")
            except nx.NetworkXNoPath:
                continue
        if best_path:
            return best_path

    # Fallback: take largest connected component, brute-force longest path
    largest_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest_cc)
    if sub.number_of_nodes() <= 200:
        # Brute force: all pairs shortest path
        all_pairs = dict(nx.all_pairs_dijkstra_path_length(sub, weight="weight"))
        best_path = []
        best_len = 0.0
        for src, dists in all_pairs.items():
            for dst, d in dists.items():
                if d > best_len:
                    best_len = d
                    best_path = nx.shortest_path(sub, src, dst, weight="weight")
        return best_path

    # Last resort: double BFS
    return _longest_path_in_tree(sub)


def _pixels_to_coords(path, origin_info):
    """Convert pixel coordinates back to real-world coordinates."""
    x0, y0, y1, cell_size = origin_info
    coords = np.zeros((len(path), 2))
    for i, (r, c) in enumerate(path):
        coords[i, 0] = x0 + (c + 0.5) * cell_size
        coords[i, 1] = y1 - (r + 0.5) * cell_size
    return coords


def select_original_points_along_centerline(
    centerline_coords, original_pts, search_radius=CENTERLINE_SEARCH_RADIUS_M
):
    """Select ordered original points that lie close to a guide centerline."""
    if len(centerline_coords) < 2 or len(original_pts) < 2:
        return np.empty((0, 2), dtype=np.float64)

    centerline_line = LineString(centerline_coords)
    total_length = float(centerline_line.length)
    if total_length < POINT_SPACING_MIN_M:
        return np.empty((0, 2), dtype=np.float64)

    tree = cKDTree(original_pts)
    k = min(5, len(original_pts))
    num_samples = max(2, int(total_length / POINT_SPACING_MIN_M) + 1)
    sample_distances = np.linspace(0.0, total_length, num_samples)

    selected_indices = set()
    selected_points = []

    for dist in sample_distances:
        p = centerline_line.interpolate(float(dist))
        q_pt = np.array([p.x, p.y], dtype=np.float64)
        dists, inds = tree.query(
            q_pt, k=k, distance_upper_bound=float(search_radius)
        )

        dists = np.atleast_1d(dists)
        inds = np.atleast_1d(inds)
        valid_mask = np.isfinite(dists) & (dists < float(search_radius))
        if not np.any(valid_mask):
            continue

        valid_inds = inds[valid_mask]
        for idx in valid_inds:
            idx = int(idx)
            if 0 <= idx < len(original_pts) and idx not in selected_indices:
                selected_indices.add(idx)
                selected_points.append(original_pts[idx])
                break

    if not selected_points:
        return np.empty((0, 2), dtype=np.float64)

    return np.array(selected_points, dtype=np.float64)


def densify_point_sequence(
    points,
    min_spacing=POINT_SPACING_MIN_M,
    max_spacing=POINT_SPACING_MAX_M,
):
    """Constrain spacing by thinning too-close points and filling large gaps."""
    if len(points) < 2:
        return points

    # Step 1: keep points at least min_spacing apart in sequence order.
    kept = [points[0]]
    for i in range(1, len(points)):
        dist = float(np.linalg.norm(points[i] - kept[-1]))
        if dist >= float(min_spacing):
            kept.append(points[i])

    kept_points = np.array(kept, dtype=np.float64)
    if len(kept_points) < 2:
        return kept_points

    # Step 2: interpolate only where gaps exceed max_spacing.
    densified = [kept_points[0]]
    for i in range(1, len(kept_points)):
        p0 = densified[-1]
        p1 = kept_points[i]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len > float(max_spacing):
            n_segments = int(np.ceil(seg_len / float(max_spacing)))
            for j in range(1, n_segments):
                t = j / n_segments
                densified.append((1.0 - t) * p0 + t * p1)
        densified.append(p1)

    densified = np.array(densified, dtype=np.float64)
    if len(densified) < 2:
        return densified

    # Final pass: enforce min spacing after interpolation.
    final_points = [densified[0]]
    for i in range(1, len(densified)):
        dist = float(np.linalg.norm(densified[i] - final_points[-1]))
        if dist >= float(min_spacing):
            final_points.append(densified[i])

    return np.array(final_points, dtype=np.float64)


def check_spacing(coords):
    """Report spacing statistics against configured spacing bounds."""
    if len(coords) < 2:
        print("    Spacing check: insufficient points")
        return np.array([], dtype=np.float64)

    diffs = np.diff(coords, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    too_close = int(np.sum(dists < POINT_SPACING_MIN_M))
    too_far = int(np.sum(dists > POINT_SPACING_MAX_M))

    print(
        f"    Spacing check: min={dists.min():.1f}m, max={dists.max():.1f}m, "
        f"median={np.median(dists):.1f}m"
    )
    print(f"    Points too close (<{POINT_SPACING_MIN_M}m): {too_close}")
    print(f"    Gaps too large (>{POINT_SPACING_MAX_M}m): {too_far}")
    return dists


def _trace_length_m(coords):
    """Return cumulative polyline length in meters."""
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def validate_trace_quality(trace_coords, trace_break_types=None):
    """Comprehensive quality check on extracted traces."""
    issues = []
    n_pts = len(trace_coords)

    if n_pts < MIN_POINTS_PER_TRACE:
        issues.append(f"Too few points: {n_pts} < {MIN_POINTS_PER_TRACE}")
        return False, 0.0, issues

    diffs = np.diff(trace_coords, axis=0)
    seg_lens = np.sqrt(np.sum(diffs**2, axis=1)) if len(diffs) else np.array([])
    total_length = float(seg_lens.sum())

    if total_length < MIN_TRACE_LENGTH_M:
        issues.append(f"Too short: {total_length:.0f}m < {MIN_TRACE_LENGTH_M:.0f}m")
        return False, 0.0, issues

    if len(seg_lens) > 0:
        too_close = int(np.sum(seg_lens < MIN_POINT_SPACING_M * 0.5))
        too_far = int(np.sum(seg_lens > MAX_POINT_SPACING_M * 1.5))
        if too_close > len(seg_lens) * 0.5:
            issues.append(f"Too many close points: {too_close}/{len(seg_lens)}")
        if too_far > len(seg_lens) * 0.3:
            issues.append(f"Too many large gaps: {too_far}/{len(seg_lens)}")

    if trace_break_types is not None and len(trace_break_types) > 0:
        _, consistency = get_dominant_break_type(np.array(trace_break_types, dtype=object))
        if consistency < MIN_BREAK_TYPE_CONSISTENCY:
            issues.append(
                f"Low break type consistency: {consistency:.1%} < "
                f"{MIN_BREAK_TYPE_CONSISTENCY:.1%}"
            )

    quality_score = 100.0
    if n_pts < MIN_POINTS_PER_TRACE * 2:
        quality_score -= 20
    if total_length < MIN_TRACE_LENGTH_M * 2:
        quality_score -= 20
    if issues:
        quality_score -= len(issues) * 15
    quality_score = float(max(0.0, min(100.0, quality_score)))

    is_valid = len(issues) <= 2 and quality_score >= 40.0
    return is_valid, quality_score, issues


def merge_nearby_segments(
    segments,
    break_types_list=None,
    max_gap=MERGE_MAX_GAP_M,
    direction_tol=MERGE_DIRECTION_TOLERANCE_DEG,
    match_break_type=MERGE_BREAK_TYPE_MUST_MATCH,
):
    """Merge neighboring segments when geometry and break type agree."""
    if len(segments) <= 1:
        if break_types_list is None:
            return segments, None
        return segments, break_types_list

    merged_segments = []
    merged_break_types = [] if break_types_list is not None else None

    current_segment = segments[0]
    current_bts = break_types_list[0] if break_types_list is not None else None

    for i in range(1, len(segments)):
        next_segment = segments[i]
        next_bts = break_types_list[i] if break_types_list is not None else None
        should_merge = True

        end_of_current = current_segment[-1]
        start_of_next = next_segment[0]
        gap_distance = float(np.linalg.norm(start_of_next - end_of_current))
        if gap_distance > float(max_gap):
            should_merge = False

        if should_merge and len(current_segment) >= 2 and len(next_segment) >= 2:
            current_dir = current_segment[-1] - current_segment[-2]
            next_dir = next_segment[1] - next_segment[0]
            current_angle = float(np.arctan2(current_dir[0], current_dir[1]))
            next_angle = float(np.arctan2(next_dir[0], next_dir[1]))
            angle_diff = abs((next_angle - current_angle + np.pi) % (2.0 * np.pi) - np.pi)
            angle_diff_deg = float(np.degrees(angle_diff))
            if angle_diff_deg > float(direction_tol):
                should_merge = False

        if (
            should_merge
            and match_break_type
            and current_bts is not None
            and next_bts is not None
            and len(current_bts) > 0
            and len(next_bts) > 0
        ):
            current_dom, _ = get_dominant_break_type(np.array(current_bts, dtype=object))
            next_dom, _ = get_dominant_break_type(np.array(next_bts, dtype=object))
            if current_dom != next_dom:
                should_merge = False

        if should_merge:
            current_segment = np.vstack([current_segment, next_segment])
            if current_bts is not None and next_bts is not None:
                current_bts = np.concatenate([current_bts, next_bts])
        else:
            if len(current_segment) >= MIN_POINTS_PER_TRACE:
                merged_segments.append(current_segment)
                if merged_break_types is not None:
                    merged_break_types.append(current_bts)
            current_segment = next_segment
            current_bts = next_bts

    if len(current_segment) >= MIN_POINTS_PER_TRACE:
        merged_segments.append(current_segment)
        if merged_break_types is not None:
            merged_break_types.append(current_bts)

    print(f"  Merged {len(segments)} segments into {len(merged_segments)} traces")
    return merged_segments, merged_break_types


def extract_high_confidence_traces(
    all_candidate_traces,
    all_break_types=None,
    max_traces=MAX_TRACES_PER_FILE,
):
    """Filter, rank, and keep top high-confidence traces."""
    if not all_candidate_traces:
        return [], [] if all_break_types is not None else None

    scored = []
    for i, trace in enumerate(all_candidate_traces):
        bts = all_break_types[i] if all_break_types is not None else None
        is_valid, quality_score, issues = validate_trace_quality(trace, bts)
        if not is_valid:
            print(f"    Trace {i} rejected: {', '.join(issues)}")
            continue

        length = _trace_length_m(trace)
        ranking_score = float(quality_score)
        ranking_score += min(length / 100.0, 30.0)
        if bts is not None and len(bts) > 0:
            _, consistency = get_dominant_break_type(np.array(bts, dtype=object))
            ranking_score += consistency * 20.0
        if len(trace) < MIN_POINTS_PER_TRACE * 2:
            ranking_score -= 30.0

        scored.append(
            {
                "trace": trace,
                "break_types": bts,
                "score": ranking_score,
                "length": length,
                "n_points": len(trace),
                "quality_score": quality_score,
            }
        )

    if not scored:
        return [], [] if all_break_types is not None else None

    scored.sort(key=lambda item: item["score"], reverse=True)
    kept = scored[: int(max_traces)]
    dropped = scored[int(max_traces):]

    kept_traces = [item["trace"] for item in kept]
    kept_break_types = [item["break_types"] for item in kept] if all_break_types is not None else None

    for i, item in enumerate(kept, start=1):
        print(
            f"    Kept trace {i}: {item['n_points']} pts, "
            f"{item['length']:.0f}m, score={item['score']:.0f}"
        )
    for item in dropped:
        print(
            f"    Rejected trace: {item['n_points']} pts, "
            f"{item['length']:.0f}m, score={item['score']:.0f} "
            f"(exceeds max {max_traces} traces)"
        )

    return kept_traces, kept_break_types


def merge_traces_strict(
    traces,
    break_types_list,
    max_gap=100.0,
    direction_tol=25.0,
):
    """Merge traces from different clusters when they are nearly continuous."""
    if len(traces) <= 1:
        return traces, break_types_list

    entries = sorted(
        zip(traces, break_types_list),
        key=lambda item: -np.mean(item[0][:, 1]),
    )

    merged_traces = []
    merged_bts = []
    current = entries[0][0]
    current_bts = entries[0][1]

    for next_trace, next_bts in entries[1:]:
        gap = float(np.linalg.norm(current[-1] - next_trace[0]))

        angle_diff = 0.0
        lateral_offset = 0.0
        if len(current) >= 2 and len(next_trace) >= 2:
            dir_cur = current[-1] - current[-2]
            dir_next = next_trace[1] - next_trace[0]
            angle_cur = float(np.degrees(np.arctan2(dir_cur[0], dir_cur[1])) % 180.0)
            angle_next = float(np.degrees(np.arctan2(dir_next[0], dir_next[1])) % 180.0)
            angle_diff = min(
                abs(angle_cur - angle_next),
                180.0 - abs(angle_cur - angle_next),
            )

            dir_cur_norm = dir_cur / (np.linalg.norm(dir_cur) + 1e-10)
            lateral_offset = abs(
                (next_trace[0][0] - current[-1][0]) * dir_cur_norm[1]
                - (next_trace[0][1] - current[-1][1]) * dir_cur_norm[0]
            )

        dom_cur, _ = get_dominant_break_type(np.asarray(current_bts, dtype=object))
        dom_next, _ = get_dominant_break_type(np.asarray(next_bts, dtype=object))

        if (
            gap <= max_gap
            and angle_diff <= direction_tol
            and dom_cur == dom_next
            and lateral_offset <= 20.0
        ):
            combined = np.vstack([current, next_trace])
            combined_bts = np.concatenate([current_bts, next_bts])
            if len(combined) > 2:
                centered = combined - combined.mean(axis=0)
                cov = np.cov(centered.T)
                _, eigvecs = np.linalg.eigh(cov)
                direction = eigvecs[:, -1]
                proj = np.dot(combined, direction)
                order = np.argsort(proj)
                current = combined[order]
                current_bts = combined_bts[order]
            else:
                current = combined
                current_bts = combined_bts
        else:
            merged_traces.append(current)
            merged_bts.append(current_bts)
            current = next_trace
            current_bts = next_bts

    merged_traces.append(current)
    merged_bts.append(current_bts)
    return merged_traces, merged_bts


def merge_all_fragments_aggressive(
    traces,
    break_types_list,
    max_gap=GLOBAL_MERGE_MAX_GAP_M,
    direction_tol=GLOBAL_MERGE_DIRECTION_TOL_DEG,
):
    """
    Aggressively merge all trace fragments into long continuous traces.
    Fragments are merged if they are close and have similar strike.
    Returns (merged_traces, merged_break_types, short_fragments, short_fragments_bts)
    sorted by length (desc).  short_fragments are traces that survived the
    merge but are below the segment threshold -- they are returned separately
    so the caller can save them as "fragment" type.
    """
    if len(traces) <= 1:
        # Even a single trace might be too short for a segment; return it
        # as a short fragment if applicable.
        short_traces = []
        short_bts = []
        for coords, bts in zip(traces, break_types_list):
            if len(coords) < MIN_POINTS_PER_TRACE:
                short_traces.append(coords)
                short_bts.append(bts)
        return traces, break_types_list, short_traces, short_bts

    items = []
    for coords, bts in zip(traces, break_types_list):
        items.append(
            {
                "coords": coords,
                "bts": bts,
                "alive": True,
                "strike": _trace_strike_deg(coords),
                "centroid_y": float(np.mean(coords[:, 1])),
                "length": _trace_length_m(coords),
            }
        )

    items.sort(key=lambda e: -e["centroid_y"])

    while True:
        best_pair = None
        best_gap = float("inf")

        n = len(items)
        for i in range(n):
            if not items[i]["alive"]:
                continue
            for j in range(i + 1, n):
                if not items[j]["alive"]:
                    continue

                pi = items[i]["coords"]
                pj = items[j]["coords"]
                dists = [
                    np.linalg.norm(pi[-1] - pj[0]),
                    np.linalg.norm(pi[-1] - pj[-1]),
                    np.linalg.norm(pi[0] - pj[0]),
                    np.linalg.norm(pi[0] - pj[-1]),
                ]
                min_dist = min(dists)
                if min_dist > max_gap:
                    continue

                strike_diff = _strike_diff_deg(items[i]["strike"], items[j]["strike"])
                if strike_diff > direction_tol:
                    continue

                idx_min = int(np.argmin(dists))
                if idx_min == 0:
                    p1, p2 = pi[-1], pj[0]
                elif idx_min == 1:
                    p1, p2 = pi[-1], pj[-1]
                elif idx_min == 2:
                    p1, p2 = pi[0], pj[0]
                else:
                    p1, p2 = pi[0], pj[-1]

                connecting_vec = p2 - p1
                length = np.linalg.norm(connecting_vec)
                if length < 1e-6:
                    alignment = 1.0
                else:
                    connecting_dir = connecting_vec / length
                    strike_rad = np.radians(items[i]["strike"])
                    strike_vec = np.array([np.sin(strike_rad), np.cos(strike_rad)])
                    alignment = abs(np.dot(connecting_dir, strike_vec))
                if alignment < 0.5:
                    continue

                if min_dist < best_gap:
                    best_gap = min_dist
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        combined_coords = np.vstack([items[i]["coords"], items[j]["coords"]])
        combined_bts = np.concatenate([items[i]["bts"], items[j]["bts"]])

        if len(combined_coords) > 2:
            centered = combined_coords - combined_coords.mean(axis=0)
            cov = np.cov(centered.T)
            _, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, -1]
            proj = np.dot(combined_coords, direction)
            order = np.argsort(proj)
            combined_coords = combined_coords[order]
            combined_bts = combined_bts[order]
        else:
            order = np.argsort(combined_coords[:, 1])[::-1]
            combined_coords = combined_coords[order]
            combined_bts = combined_bts[order]

        items[i]["coords"] = combined_coords
        items[i]["bts"] = combined_bts
        items[i]["strike"] = _trace_strike_deg(combined_coords)
        items[i]["centroid_y"] = float(np.mean(combined_coords[:, 1]))
        items[i]["length"] = _trace_length_m(combined_coords)
        items[j]["alive"] = False

        print(
            f"    [Global merge] merged fragments {i}+{j} -> "
            f"length={items[i]['length']:.0f}m, gap={best_gap:.0f}m"
        )

    merged_traces = []
    merged_bts = []
    short_fragments = []
    short_fragments_bts = []
    for item in items:
        if not item["alive"]:
            continue
        if len(item["coords"]) >= MIN_POINTS_PER_TRACE:
            merged_traces.append(item["coords"])
            merged_bts.append(item["bts"])
        else:
            # Too short for a segment, but keep as a fragment
            short_fragments.append(item["coords"])
            short_fragments_bts.append(item["bts"])

    order = np.argsort([_trace_length_m(t) for t in merged_traces])[::-1]
    merged_traces = [merged_traces[i] for i in order]
    merged_bts = [merged_bts[i] for i in order]
    return merged_traces, merged_bts, short_fragments, short_fragments_bts


def check_and_split_direction_less_aggressive(
    coords,
    angle_threshold=60.0,
    min_segment_length=MIN_TRACE_LENGTH_M,
):
    """Split only on major direction changes and keep long-enough segments."""
    if len(coords) < MIN_POINTS_PER_TRACE:
        return [coords]

    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt(np.sum(diffs**2, axis=1)) if len(diffs) else np.array([])
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_length = float(arc[-1])
    if total_length < float(min_segment_length):
        return [coords]

    window_size = max(5, len(coords) // 4)
    directions = []
    dir_anchor = []
    for i in range(len(coords) - window_size):
        vec = coords[i + window_size] - coords[i]
        length = float(np.linalg.norm(vec))
        if length > 0:
            directions.append(vec / length)
            dir_anchor.append(i + window_size // 2)

    if not directions:
        return [coords]

    split_indices = [0]
    reference_dir = directions[0]
    for i in range(1, len(directions)):
        dot_product = float(np.clip(np.dot(reference_dir, directions[i]), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot_product)))
        if angle > float(angle_threshold):
            split_idx = int(dir_anchor[i])
            split_point_arc = float(arc[min(split_idx, len(arc) - 1)])
            if (
                split_point_arc > float(min_segment_length)
                and (total_length - split_point_arc) > float(min_segment_length)
            ):
                split_indices.append(split_idx)
                reference_dir = directions[i]

    split_indices.append(len(coords))
    split_indices = sorted(set(split_indices))

    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        seg = coords[start_idx:end_idx]
        if len(seg) < MIN_POINTS_PER_TRACE:
            continue
        seg_length = _trace_length_m(seg)
        if seg_length >= float(min_segment_length):
            segments.append(seg)

    return segments if segments else [coords]


def adaptive_trace_extraction(cluster_pts, cluster_break_types, centerline_guide):
    """Extract traces with adaptive spacing and conservative splitting."""
    if len(centerline_guide) < 2 or len(cluster_pts) < 2:
        return [], []

    guide_line = LineString(centerline_guide)
    total_length = float(guide_line.length)
    if total_length < MIN_TRACE_LENGTH_M:
        return [], []

    sample_spacing = max(MIN_POINT_SPACING_M, total_length / (MIN_POINTS_PER_TRACE * 2.0))
    num_samples = max(MIN_POINTS_PER_TRACE, int(total_length / sample_spacing))
    print(
        f"    Adaptive sampling: {num_samples} samples over {total_length:.0f}m "
        f"(spacing={sample_spacing:.1f}m)"
    )

    tree = cKDTree(cluster_pts)
    selected_coords = []
    selected_bts = []
    selected_indices = set()

    for dist in np.linspace(0.0, total_length, num_samples):
        point_on_line = guide_line.interpolate(float(dist))
        guide_pt = np.array([point_on_line.x, point_on_line.y], dtype=np.float64)
        distance, index = tree.query(guide_pt, k=1)
        if np.isfinite(distance) and float(distance) < sample_spacing * 1.5:
            idx = int(index)
            if idx not in selected_indices:
                selected_indices.add(idx)
                selected_coords.append(cluster_pts[idx])
                selected_bts.append(cluster_break_types[idx])

    if len(selected_coords) < MIN_POINTS_PER_TRACE:
        return [], []

    selected_coords = np.array(selected_coords, dtype=np.float64)
    selected_bts = np.array(selected_bts, dtype=object)

    direction_segments = check_and_split_direction_less_aggressive(selected_coords)
    segment_bts = []
    if direction_segments:
        # Re-map each split segment to break types by nearest selected point.
        sel_tree = cKDTree(selected_coords)
        for seg in direction_segments:
            _, idx = sel_tree.query(seg, k=1)
            segment_bts.append(selected_bts[np.atleast_1d(idx).astype(int)])

    return direction_segments, segment_bts


def merge_continuous_clusters(
    clusters,
    cluster_break_types,
    max_gap=MERGE_CLUSTER_GAP_M,
    direction_tol=MERGE_DIRECTION_TOLERANCE_DEG,
):
    """Merge DBSCAN clusters that likely form one continuous scarp."""
    if len(clusters) <= 1:
        return clusters, cluster_break_types

    print(f"  Attempting to merge {len(clusters)} clusters into continuous traces...")

    cluster_props = []
    for i, (pts, bts) in enumerate(zip(clusters, cluster_break_types)):
        dom_type, _ = get_dominant_break_type(bts)

        centered = pts - np.mean(pts, axis=0)
        if len(pts) > 2:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_direction = eigvecs[:, -1]
        else:
            main_direction = pts[-1] - pts[0]
            main_direction = main_direction / (np.linalg.norm(main_direction) + 1e-10)

        main_angle = np.degrees(np.arctan2(main_direction[0], main_direction[1])) % 180.0

        if len(pts) > 2:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_matrix = np.sum(diff * diff, axis=2)
            a, b = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            endpoint1 = pts[a]
            endpoint2 = pts[b]
        else:
            endpoint1 = pts[0]
            endpoint2 = pts[-1]

        cluster_props.append(
            {
                "index": i,
                "points": pts,
                "break_types": bts,
                "break_type": dom_type,
                "direction": main_direction,
                "angle": main_angle,
                "endpoint1": endpoint1,
                "endpoint2": endpoint2,
            }
        )

    merge_graph = nx.Graph()
    for i in range(len(cluster_props)):
        merge_graph.add_node(i)

    for i in range(len(cluster_props)):
        for j in range(i + 1, len(cluster_props)):
            pi = cluster_props[i]
            pj = cluster_props[j]

            if pi["break_type"] != pj["break_type"]:
                continue

            angle_diff = abs(pi["angle"] - pj["angle"])
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if angle_diff > float(direction_tol):
                continue

            dists = [
                np.linalg.norm(pi["endpoint1"] - pj["endpoint1"]),
                np.linalg.norm(pi["endpoint1"] - pj["endpoint2"]),
                np.linalg.norm(pi["endpoint2"] - pj["endpoint1"]),
                np.linalg.norm(pi["endpoint2"] - pj["endpoint2"]),
            ]
            min_dist = float(min(dists))
            if min_dist > float(max_gap):
                continue

            d_idx = int(np.argmin(dists))
            if d_idx == 0:
                vec_between = pj["endpoint1"] - pi["endpoint1"]
            elif d_idx == 1:
                vec_between = pj["endpoint2"] - pi["endpoint1"]
            elif d_idx == 2:
                vec_between = pj["endpoint1"] - pi["endpoint2"]
            else:
                vec_between = pj["endpoint2"] - pi["endpoint2"]

            vec_between = vec_between / (np.linalg.norm(vec_between) + 1e-10)
            alignment = abs(float(np.dot(vec_between, pi["direction"])))
            if alignment > 0.5:
                merge_graph.add_edge(i, j, weight=min_dist)

    merged_clusters = []
    merged_break_types = []
    for component in nx.connected_components(merge_graph):
        idxs = sorted(list(component))
        if len(idxs) == 1:
            idx = idxs[0]
            merged_clusters.append(cluster_props[idx]["points"])
            merged_break_types.append(cluster_props[idx]["break_types"])
            continue

        merged_pts = np.vstack([cluster_props[idx]["points"] for idx in idxs])
        merged_bts = np.concatenate([cluster_props[idx]["break_types"] for idx in idxs])

        main_dir = cluster_props[idxs[0]]["direction"]
        projections = np.dot(merged_pts, main_dir)
        sort_idx = np.argsort(projections)
        merged_pts = merged_pts[sort_idx]
        merged_bts = merged_bts[sort_idx]

        merged_clusters.append(merged_pts)
        merged_break_types.append(merged_bts)

        extent = np.sqrt(np.ptp(merged_pts[:, 0]) ** 2 + np.ptp(merged_pts[:, 1]) ** 2)
        print(
            f"    Merged clusters {idxs}: {len(merged_pts)} pts, "
            f"extent={extent:.0f}m, type={cluster_props[idxs[0]]['break_type']}"
        )

    print(
        f"  After merging: {len(merged_clusters)} continuous traces "
        f"(from {len(clusters)} clusters)"
    )
    return merged_clusters, merged_break_types


def merge_clusters_greedy_nearest(
    clusters,
    cluster_break_types,
    max_dist=GREEDY_MERGE_MAX_DIST_M,
):
    """Greedily merge clusters by nearest-neighbour proximity.

    At each iteration, the pair of clusters (i, j) with the **same dominant
    break type** and the **smallest inter-cluster point distance** is merged,
    provided that distance is <= ``max_dist``.  This repeats until no more
    pairs qualify.  Merging is purely distance- and break-type-driven; no
    direction/alignment gate is applied.

    Parameters
    ----------
    clusters : list of (N_i, 2) arrays
        Cluster point arrays (one per cluster).
    cluster_break_types : list of arrays
        Break-type arrays parallel to *clusters*.
    max_dist : float
        Maximum inter-cluster point distance for merging (metres).

    Returns
    -------
    (merged_clusters, merged_break_types) : list pair
    """
    n = len(clusters)
    if n <= 1:
        return list(clusters), list(cluster_break_types)

    print(f"  Greedy merge: {n} clusters, max_dist={max_dist}m")

    # Work on mutable copies
    items = [
        {
            "pts": np.asarray(clusters[i], dtype=np.float64),
            "bts": np.asarray(cluster_break_types[i]),
            "btype": get_dominant_break_type(cluster_break_types[i])[0],
            "alive": True,
        }
        for i in range(n)
    ]

    while True:
        best_dist = float("inf")
        best_pair = None  # (i, j)

        for i in range(n):
            if not items[i]["alive"]:
                continue
            for j in range(i + 1, n):
                if not items[j]["alive"]:
                    continue
                if items[i]["btype"] != items[j]["btype"]:
                    continue
                # Min point-to-point distance between the two clusters
                pi = items[i]["pts"]
                pj = items[j]["pts"]
                diff = pi[:, np.newaxis, :] - pj[np.newaxis, :, :]
                d2 = np.sum(diff * diff, axis=2)
                min_d = float(np.sqrt(np.min(d2)))
                if min_d < best_dist and min_d <= max_dist:
                    best_dist = min_d
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        # Merge j into i
        merged_pts = np.vstack([items[i]["pts"], items[j]["pts"]])
        merged_bts = np.concatenate([items[i]["bts"], items[j]["bts"]])

        # Sort merged points along principal direction for spatial ordering
        centered = merged_pts - np.mean(merged_pts, axis=0)
        if len(merged_pts) > 2:
            cov = np.cov(centered.T)
            _, eigvecs = np.linalg.eigh(cov)
            main_dir = eigvecs[:, -1]
        else:
            main_dir = np.array([1.0, 0.0])
        proj = np.dot(merged_pts, main_dir)
        sort_idx = np.argsort(proj)
        merged_pts = merged_pts[sort_idx]
        merged_bts = merged_bts[sort_idx]

        items[i]["pts"] = merged_pts
        items[i]["bts"] = merged_bts
        items[j]["alive"] = False

        btype = items[i]["btype"]
        print(
            f"    Merged clusters {i}+{j}: {len(merged_pts)} pts, "
            f"gap={best_dist:.0f}m, type={btype}"
        )

    merged_clusters = [items[i]["pts"] for i in range(n) if items[i]["alive"]]
    merged_break_types = [items[i]["bts"] for i in range(n) if items[i]["alive"]]

    print(
        f"  Greedy merge done: {len(merged_clusters)} clusters "
        f"(from {n})"
    )
    return merged_clusters, merged_break_types


def extract_dense_trace_from_cluster(cluster_pts, cluster_break_types):
    """Use many original points, thinned to near-target spacing."""
    if len(cluster_pts) < MIN_POINTS_PER_TRACE:
        return np.empty((0, 2), dtype=np.float64), np.array([], dtype=object)

    print(f"    Extracting trace from {len(cluster_pts)} points...")

    centered = cluster_pts - np.mean(cluster_pts, axis=0)
    if len(cluster_pts) > 2:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        main_direction = eigvecs[:, -1]
    else:
        main_direction = cluster_pts[-1] - cluster_pts[0]
        main_direction = main_direction / (np.linalg.norm(main_direction) + 1e-10)

    projections = np.dot(cluster_pts, main_direction)
    sort_idx = np.argsort(projections)
    sorted_pts = cluster_pts[sort_idx]
    sorted_bts = cluster_break_types[sort_idx]

    selected_indices = [0]
    cumulative_dist = 0.0
    last_kept = sorted_pts[0]
    for i in range(1, len(sorted_pts)):
        dist = float(np.linalg.norm(sorted_pts[i] - last_kept))
        if dist >= TARGET_POINT_SPACING_M:
            selected_indices.append(i)
            last_kept = sorted_pts[i]
            cumulative_dist = 0.0
        else:
            cumulative_dist += dist
            if cumulative_dist >= MAX_POINT_SPACING_M:
                selected_indices.append(i)
                last_kept = sorted_pts[i]
                cumulative_dist = 0.0

    if selected_indices[-1] != len(sorted_pts) - 1:
        selected_indices.append(len(sorted_pts) - 1)

    selected_pts = sorted_pts[selected_indices]
    selected_bts = sorted_bts[selected_indices]

    print(
        f"    Selected {len(selected_pts)}/{len(cluster_pts)} points "
        f"(target spacing={TARGET_POINT_SPACING_M:.1f}m)"
    )

    if len(selected_pts) > 1:
        d = np.sqrt(np.sum(np.diff(selected_pts, axis=0) ** 2, axis=1))
        print(
            f"    Trace length: {float(d.sum()):.0f}m, spacing: "
            f"min={float(d.min()):.1f}m, median={float(np.median(d)):.1f}m, "
            f"max={float(d.max()):.1f}m"
        )

    return selected_pts, selected_bts


def _guide_length_m(guide_lines):
    """Sum polyline length across one or more guide centerline segments."""
    return float(sum(_trace_length_m(g) for g in guide_lines))


def _cluster_principal_extent_m(cluster_pts):
    """Projected span of cluster points along their principal direction.

    This is a stable geometric measure of the cluster's true along-strike
    length, unlike the skeleton centerline which can under-estimate due to
    rasterisation gaps or over-estimate when the skeleton meanders.
    """
    if len(cluster_pts) < 2:
        return 0.0
    centered = cluster_pts - cluster_pts.mean(axis=0)
    if len(cluster_pts) > 2:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]
    else:
        direction = cluster_pts[-1] - cluster_pts[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return 0.0
    direction = direction / norm
    projections = np.dot(cluster_pts, direction)
    return float(projections.max() - projections.min())


def _projected_span_m(coords, direction):
    """Projected span of coords along a given unit direction vector."""
    if len(coords) < 2:
        return 0.0
    projections = np.dot(coords, direction)
    return float(projections.max() - projections.min())


def _cluster_principal_direction(cluster_pts):
    """Return the unit principal direction of a cluster's point cloud."""
    if len(cluster_pts) < 2:
        return np.array([1.0, 0.0])
    centered = cluster_pts - cluster_pts.mean(axis=0)
    if len(cluster_pts) > 2:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]
    else:
        direction = cluster_pts[-1] - cluster_pts[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return direction / norm


def extract_cluster_trace_with_recovery(cluster_pts, cluster_bts, cluster_id):
    """Extract trace fragment(s) for a cluster, relaxing until >=80% of the
    cluster's principal-axis extent ("original trace length") is recovered.

    The recovery target is the projected span of the cluster along its
    principal direction -- a stable geometric measure that does not suffer
    from skeleton rasterisation artefacts.  Both the original length and the
    recovered length are measured as projected spans along the *same*
    principal direction, so the ratio is a true spatial-coverage fraction
    (0-100 %) rather than a sinuosity-inflated polyline-length ratio.

    Returns (fragments, fragment_break_types, recovery_ratio, original_length_m,
    status) where status is one of OK / BELOW_TARGET / NO_GUIDE / NO_TRACE.
    """
    original_length = _cluster_principal_extent_m(cluster_pts)
    if original_length <= 0:
        return [], [], 0.0, original_length, "NO_GUIDE"

    principal_dir = _cluster_principal_direction(cluster_pts)

    guide_lines = _extract_centerline_guides_parallel_aware(cluster_pts)
    if not guide_lines:
        return [], [], 0.0, original_length, "NO_GUIDE"

    base_radius = min(
        CENTERLINE_SEARCH_RADIUS_M,
        max(BUFFER_WIDTH_FOR_SEPARATION, PARALLEL_SEPARATION_MIN_GAP_M / 2.0),
    )
    tree = cKDTree(cluster_pts)

    best = None  # (fragments, frag_bts, ratio, recovered_length, radius_mult, gap_mult)
    for radius_mult in RECOVERY_RADIUS_MULTIPLIERS:
        radius = base_radius * radius_mult
        raw_frags = []
        raw_frag_bts = []
        for guide in guide_lines:
            selected = point_to_line_distance_sampling(guide, cluster_pts, max_distance=radius)
            if len(selected) < MIN_CENTERLINE_POINTS:
                continue
            _, idx = tree.query(selected, k=1)
            selected_bts = cluster_bts[np.atleast_1d(idx).astype(int)]
            densified = densify_point_sequence(
                selected, min_spacing=POINT_SPACING_MIN_M, max_spacing=POINT_SPACING_MAX_M
            )
            if len(densified) < MIN_CENTERLINE_POINTS:
                continue
            sel_tree = cKDTree(selected)
            _, idx2 = sel_tree.query(densified, k=1)
            raw_frags.append(densified)
            raw_frag_bts.append(selected_bts[np.atleast_1d(idx2).astype(int)])

        if not raw_frags:
            continue

        for gap_mult in RECOVERY_MERGE_GAP_MULTIPLIERS:
            merged_frags, merged_bts = merge_nearby_segments(
                raw_frags,
                raw_frag_bts,
                max_gap=100.0,
                direction_tol=30.0,
                match_break_type=False,
            )
            # Measure recovered length as projected span along the cluster's
            # principal direction -- consistent with how original_length is
            # measured, so the ratio is a true spatial-coverage fraction.
            if merged_frags:
                all_frag_pts = np.vstack(merged_frags)
                recovered_length = _projected_span_m(all_frag_pts, principal_dir)
            else:
                recovered_length = 0.0
            ratio = min(recovered_length / original_length, 1.0)

            if best is None or recovered_length > best[3]:
                best = (merged_frags, merged_bts, ratio, recovered_length, radius_mult, gap_mult)

            if ratio >= RECOVERY_TARGET_FRACTION:
                print(
                    f"    [Recovery] cluster {cluster_id}: {ratio:.0%} recovered "
                    f"(radius x{radius_mult}, merge-gap x{gap_mult}) "
                    f"[{recovered_length:.0f}/{original_length:.0f} m]"
                )
                return merged_frags, merged_bts, ratio, original_length, "OK"

    if best is None:
        return [], [], 0.0, original_length, "NO_TRACE"

    merged_frags, merged_bts, ratio, recovered_length, radius_mult, gap_mult = best
    print(
        f"    [Recovery][WARNING] cluster {cluster_id}: only {ratio:.0%} recovered "
        f"after max relaxation (radius x{radius_mult}, merge-gap x{gap_mult}) "
        f"[{recovered_length:.0f}/{original_length:.0f} m]"
    )
    return merged_frags, merged_bts, ratio, original_length, "BELOW_TARGET"


def _trace_strike_deg(coords):
    """Axial (mod-180) orientation of a polyline's principal direction."""
    if len(coords) < 2:
        return 0.0
    if len(coords) > 2:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]
    else:
        direction = coords[-1] - coords[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return 0.0
    direction = direction / norm
    return float(np.degrees(np.arctan2(direction[0], direction[1])) % 180.0)


def _strike_diff_deg(a, b):
    """Smallest angular difference between two axial (mod-180) strikes."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _order_by_projection(coords, break_types):
    """Re-order points along the combined set's principal direction."""
    centered = coords - coords.mean(axis=0)
    if len(coords) > 2:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]
    else:
        direction = coords[-1] - coords[0]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
    projections = np.dot(coords, direction)
    order = np.argsort(projections)
    return coords[order], break_types[order]


def merge_segments_by_orientation(
    traces,
    break_types_list,
    cluster_ids,
    max_segments=MAX_SEGMENTS_PER_FILE,
    strike_tol_deg=SEGMENT_MERGE_STRIKE_TOL_DEG,
):
    """Sort segments north->south (then by break type) and merge adjacent
    segments whose strike agrees within *strike_tol_deg*, escalating the
    tolerance until the file has <= *max_segments* segments or relaxation
    is exhausted. Returns a list of dicts with coords/break_types/cluster_ids.
    """
    entries = []
    for coords, bts, cid in zip(traces, break_types_list, cluster_ids):
        dom_type, _ = get_dominant_break_type(bts)
        entries.append(
            {
                "coords": coords,
                "break_types": bts,
                "cluster_ids": {cid},
                "break_type": dom_type,
                "centroid_y": float(np.mean(coords[:, 1])),
                "strike": _trace_strike_deg(coords),
            }
        )

    tol = float(strike_tol_deg)
    while True:
        entries.sort(key=lambda e: (-e["centroid_y"], e["break_type"]))

        merged_any = True
        while merged_any and len(entries) > 1:
            merged_any = False
            new_entries = []
            i = 0
            while i < len(entries):
                cur = entries[i]
                if i + 1 < len(entries):
                    nxt = entries[i + 1]
                    if (
                        cur["break_type"] == nxt["break_type"]
                        and _strike_diff_deg(cur["strike"], nxt["strike"]) <= tol
                    ):
                        combined_coords = np.vstack([cur["coords"], nxt["coords"]])
                        combined_bts = np.concatenate([cur["break_types"], nxt["break_types"]])
                        ordered_coords, ordered_bts = _order_by_projection(combined_coords, combined_bts)
                        dom_type, _ = get_dominant_break_type(ordered_bts)
                        new_entries.append(
                            {
                                "coords": ordered_coords,
                                "break_types": ordered_bts,
                                "cluster_ids": cur["cluster_ids"] | nxt["cluster_ids"],
                                "break_type": dom_type,
                                "centroid_y": float(np.mean(ordered_coords[:, 1])),
                                "strike": _trace_strike_deg(ordered_coords),
                            }
                        )
                        i += 2
                        merged_any = True
                        continue
                new_entries.append(cur)
                i += 1
            entries = new_entries

        if len(entries) <= max_segments or tol >= SEGMENT_MERGE_MAX_TOL_DEG:
            break
        tol = min(tol + SEGMENT_MERGE_TOL_STEP_DEG, SEGMENT_MERGE_MAX_TOL_DEG)
        print(
            f"    [Merge] {len(entries)} segments remain > {max_segments}; "
            f"relaxing strike tolerance to {tol:.0f} deg"
        )

    if len(entries) > max_segments:
        print(
            f"    [Merge][WARNING] Still {len(entries)} segments after max "
            f"strike relaxation ({tol:.0f} deg); keeping longest {max_segments}"
        )
        entries.sort(key=lambda e: _trace_length_m(e["coords"]), reverse=True)
        entries = entries[:max_segments]

    return entries


def simplified_pipeline_for_trace(
    fpath,
    break_type_lookup,
    dem_src,
    dem_hs,
    dem_transform,
    qc_dir,
    trace_id_start,
    skip_diagnose=False,
    skip_terrain=False,
):
    """Simplified dense pipeline: clean -> cluster -> merge -> dense extract."""
    coords, attributes = load_trace_points_with_break_types(fpath, break_type_lookup)
    break_types = attributes["break_type"]
    n_original = len(coords)

    if len(coords) < MIN_CLUSTER_POINTS:
        return [], trace_id_start

    print(f"\n{'='*60}")
    print(f"Processing {fpath.name}: {n_original} points")
    print(f"{'='*60}")

    diagnostics = {}
    if not skip_diagnose:
        diagnostics = stage0_diagnose(coords, fpath.name, dem_hs, dem_transform, qc_dir)

    if skip_terrain or dem_src is None:
        cleaned_coords, cleaned_break_types, stage1_info = stage1_outlier_rejection_with_types(
            coords, break_types, None, diagnostics
        )
    else:
        cleaned_coords, cleaned_break_types, stage1_info = stage1_outlier_rejection_with_types(
            coords, break_types, dem_src, diagnostics
        )

    if len(cleaned_coords) < MIN_CLUSTER_POINTS:
        return [], trace_id_start

    clusters, cluster_break_types = stage2_segment_with_break_types(
        cleaned_coords, cleaned_break_types
    )
    if not clusters:
        return [], trace_id_start

    # ------------------------------------------------------------------
    # Conservative continuous-cluster merge: only join end-to-end traces that
    # are close in gap and direction, avoiding parallel offset clusters.
    # ------------------------------------------------------------------
    merged_clusters, merged_break_types = merge_continuous_clusters(
        clusters,
        cluster_break_types,
        max_gap=300.0,
        direction_tol=25.0,
    )

    all_traces = []
    all_break_types = []
    all_cluster_ids = []
    cluster_recovery = {}
    cluster_principal_dirs = {}
    for cluster_id, (cluster_pts, cluster_bts) in enumerate(zip(merged_clusters, merged_break_types)):
        if len(cluster_pts) < FRAGMENT_MIN_POINTS:
            continue

        frags, frag_bts, ratio, original_length, status = extract_cluster_trace_with_recovery(
            cluster_pts, cluster_bts, cluster_id
        )
        cluster_recovery[cluster_id] = {
            "original_length_m": original_length,
            "recovery_ratio": ratio,
            "recovery_status": status,
        }
        cluster_principal_dirs[cluster_id] = _cluster_principal_direction(cluster_pts)
        for frag, frag_bt in zip(frags, frag_bts):
            if len(frag) < FRAGMENT_MIN_POINTS:
                continue
            all_traces.append(frag)
            all_break_types.append(frag_bt)
            all_cluster_ids.append(cluster_id)

    print(f"\n  Extracted {len(all_traces)} traces from merged clusters")
    if all_traces:
        print(f"\n  Global merging of {len(all_traces)} fragments...")
        merged_global_traces, merged_global_bts, short_frags, short_frag_bts = merge_all_fragments_aggressive(
            all_traces,
            all_break_types,
        )
        print(f"  After global merge: {len(merged_global_traces)} traces, "
              f"{len(short_frags)} short fragments kept")

        final_traces = merged_global_traces[:FINAL_MAX_TRACES_PER_FILE]
        final_bts = merged_global_bts[:FINAL_MAX_TRACES_PER_FILE]
        print(f"  Keeping top {len(final_traces)} longest traces")
    else:
        final_traces = []
        final_bts = []
        short_frags = []
        short_frag_bts = []

    if not final_traces and not short_frags:
        return [], trace_id_start

    manifest_rows = []
    trace_id = trace_id_start
    for rank, (trace, bts) in enumerate(zip(final_traces, final_bts)):
        smoothed = stage4_smooth_simplify(trace)
        final_length = _trace_length_m(smoothed)
        if final_length < MIN_TRACE_LENGTH_M * 0.5 or len(smoothed) < MIN_POINTS_PER_TRACE:
            # Too short for a segment -- redirect to fragments instead of
            # discarding entirely.
            if len(smoothed) >= FRAGMENT_MIN_POINTS and final_length >= FRAGMENT_MIN_LENGTH_M:
                short_frags.append(smoothed)
                short_frag_bts.append(bts)
            continue

        dom_type, consistency = get_dominant_break_type(bts)
        qc = stage5_qc(
            smoothed,
            cleaned_coords,
            dem_hs,
            dem_transform,
            trace_id,
            qc_dir,
            stage1_info,
        )

        csv_path = CLEANED_DIR / f"trace_{trace_id:02d}_{fpath.stem}.csv"
        cl_df = pd.DataFrame(
            {
                "x": smoothed[:, 0],
                "y": smoothed[:, 1],
                "break_type": [dom_type] * len(smoothed),
            }
        )
        cl_df.to_csv(csv_path, index=False)

        manifest_rows.append(
            {
                "trace_id": trace_id,
                "source_file": fpath.name,
                "n_points": len(smoothed),
                "length_m": final_length,
                "break_type": dom_type,
                "break_type_consistency": consistency,
                "qc_flag": qc["qc_flag"],
                "trace_type": "segment",
                "csv_path": csv_path.name,
            }
        )

        print(f"  Trace {trace_id}: {len(smoothed)} pts, {final_length:.0f}m, type={dom_type}")
        trace_id += 1

    # ------------------------------------------------------------------
    # Save short fragments that survived the global merge but are below
    # the segment threshold.  These are stored as "fragment" type and
    # are visualised in 06b's per-original-trace overviews.
    # ------------------------------------------------------------------
    for frag_coords, frag_bts in zip(short_frags, short_frag_bts):
        # Fragments redirected from the segment loop are already smoothed;
        # those coming from the global merge are raw.  Detect by checking
        # if the coords look like they came from stage4 (few points, evenly
        # spaced) vs raw cluster extraction.  Safe to just always smooth --
        # stage4 is idempotent on already-smoothed data.
        frag_smoothed = stage4_smooth_simplify(frag_coords)
        frag_length = _trace_length_m(frag_smoothed)
        if len(frag_smoothed) < FRAGMENT_MIN_POINTS or frag_length < FRAGMENT_MIN_LENGTH_M:
            continue

        dom_type, consistency = get_dominant_break_type(frag_bts)

        csv_path = CLEANED_DIR / f"trace_{trace_id:02d}_{fpath.stem}.csv"
        cl_df = pd.DataFrame(
            {
                "x": frag_smoothed[:, 0],
                "y": frag_smoothed[:, 1],
                "break_type": [dom_type] * len(frag_smoothed),
            }
        )
        cl_df.to_csv(csv_path, index=False)

        manifest_rows.append(
            {
                "trace_id": trace_id,
                "source_file": fpath.name,
                "n_points": len(frag_smoothed),
                "length_m": frag_length,
                "break_type": dom_type,
                "break_type_consistency": consistency,
                "qc_flag": "fragment",
                "trace_type": "fragment",
                "csv_path": csv_path.name,
            }
        )

        print(f"  Fragment {trace_id}: {len(frag_smoothed)} pts, {frag_length:.0f}m, type={dom_type}")
        trace_id += 1

    return manifest_rows, trace_id


def stage3_centerline(cluster_pts):
    """Extract guide centerline(s), then select ordered original points.

    Returns a list of (N, 2) arrays -- one per centerline segment
    (multiple if Y-branches are split).
    """
    mask, origin = _rasterize_points(cluster_pts, RASTER_CELL_SIZE_M)

    thresh = 0.0
    if SKELETON_SIGMA > 0:
        smooth = gaussian_filter(mask, sigma=SKELETON_SIGMA)
        # Use relative threshold for density mask
        thresh = max(0.1, smooth.max() * 0.15)
        skel = skeletonize(smooth > thresh)
    else:
        skel = skeletonize(mask > 0)

    n_skel_pixels = int(skel.sum())
    print(f"    Skeleton: {n_skel_pixels} pixels from "
          f"{mask.shape[0]}x{mask.shape[1]} grid (thresh={thresh:.2f})")
    if n_skel_pixels < MIN_CENTERLINE_POINTS:
        return []

    G = _skeleton_to_graph(skel)
    if G.number_of_nodes() < MIN_CENTERLINE_POINTS:
        return []

    G = _prune_spurs(G, SPUR_LENGTH_THRESHOLD)
    if G.number_of_nodes() < MIN_CENTERLINE_POINTS:
        return []

    # Check for Y-branches (branch points with degree >= 3)
    branch_nodes = [n for n in G.nodes() if G.degree(n) >= 3]

    guide_lines = []
    if len(branch_nodes) >= 1:
        # Try splitting at branch points
        segments = _split_at_branches(G, branch_nodes)
        if segments:
            guide_lines = [
                _pixels_to_coords(p, origin)
                for p in segments
                if len(p) >= MIN_CENTERLINE_POINTS
            ]

    if not guide_lines:
        # Single centerline
        path = _longest_path_general(G)
        if len(path) < MIN_CENTERLINE_POINTS:
            return []
        guide_lines = [_pixels_to_coords(path, origin)]

    result_lines = []
    for guide in guide_lines:
        selected = select_original_points_along_centerline(guide, cluster_pts)
        if len(selected) < MIN_CENTERLINE_POINTS:
            continue

        densified = densify_point_sequence(
            selected,
            min_spacing=POINT_SPACING_MIN_M,
            max_spacing=POINT_SPACING_MAX_M,
        )
        if len(densified) >= MIN_CENTERLINE_POINTS:
            print(
                f"    Selected {len(selected)} original points, "
                f"densified to {len(densified)} points"
            )
            result_lines.append(densified)

    return result_lines


def _extract_centerline_guides_parallel_aware(cluster_pts):
    """Extract guide centerline geometries from a cluster (before snapping)."""
    mask, origin = _rasterize_points(cluster_pts, RASTER_CELL_SIZE_M)

    if SKELETON_SIGMA > 0:
        smooth = gaussian_filter(mask, sigma=SKELETON_SIGMA)
        thresh = max(0.1, smooth.max() * 0.15)
        skel = skeletonize(smooth > thresh)
    else:
        skel = skeletonize(mask > 0)

    n_skel_pixels = int(skel.sum())
    if n_skel_pixels < MIN_CENTERLINE_POINTS:
        return []

    G = _skeleton_to_graph(skel)
    if G.number_of_nodes() < MIN_CENTERLINE_POINTS:
        return []

    G = _prune_spurs(G, SPUR_LENGTH_THRESHOLD)
    if G.number_of_nodes() < MIN_CENTERLINE_POINTS:
        return []

    branch_nodes = [n for n in G.nodes() if G.degree(n) >= 3]
    guide_lines = []
    if branch_nodes:
        segments = _split_at_branches(G, branch_nodes)
        guide_lines = [
            _pixels_to_coords(p, origin)
            for p in segments
            if len(p) >= MIN_CENTERLINE_POINTS
        ]
    else:
        path = _longest_path_general(G)
        if len(path) >= MIN_CENTERLINE_POINTS:
            guide_lines = [_pixels_to_coords(path, origin)]

    return guide_lines


def stage3_centerline_parallel_aware(cluster_pts):
    """Parallel-aware Stage 3 that constrains point selection distance."""
    guide_lines = _extract_centerline_guides_parallel_aware(cluster_pts)
    if not guide_lines:
        return []

    result_lines = []
    tight_radius = min(
        CENTERLINE_SEARCH_RADIUS_M,
        max(BUFFER_WIDTH_FOR_SEPARATION, PARALLEL_SEPARATION_MIN_GAP_M / 2.0),
    )

    for guide in guide_lines:
        selected = point_to_line_distance_sampling(
            guide,
            cluster_pts,
            max_distance=tight_radius,
        )
        if len(selected) < MIN_CENTERLINE_POINTS:
            continue

        densified = densify_point_sequence(
            selected,
            min_spacing=POINT_SPACING_MIN_M,
            max_spacing=POINT_SPACING_MAX_M,
        )
        if len(densified) >= MIN_CENTERLINE_POINTS:
            print(
                f"    Extracted trace: {len(densified)} points, "
                f"search_radius={tight_radius:.1f} m"
            )
            result_lines.append(densified)

    return result_lines


def _split_at_branches(G, branch_nodes):
    """Split a skeleton graph at branch points into separate segments."""
    segments = []
    G_work = G.copy()

    for bn in branch_nodes:
        if bn not in G_work:
            continue
        neighbours = list(G_work.neighbors(bn))
        if len(neighbours) < 3:
            continue

        # Remove branch node and find connected components
        G_work.remove_node(bn)
        components = list(nx.connected_components(G_work))

        if len(components) >= 2:
            # Find endpoints in each component and extract longest path
            for comp in components:
                sub = G_work.subgraph(comp)
                endpoints = [n for n in sub.nodes() if sub.degree(n) == 1]
                if len(endpoints) >= 2:
                    best_path = []
                    best_len = 0
                    for i in range(len(endpoints)):
                        try:
                            dists = nx.single_source_dijkstra_path_length(
                                sub, endpoints[i], weight="weight")
                            for j in range(i + 1, len(endpoints)):
                                d = dists.get(endpoints[j], 0)
                                if d > best_len:
                                    best_len = d
                                    best_path = nx.shortest_path(
                                        sub, endpoints[i], endpoints[j],
                                        weight="weight")
                        except nx.NetworkXNoPath:
                            continue
                    if len(best_path) >= MIN_CENTERLINE_POINTS:
                        segments.append(best_path)
                elif sub.number_of_nodes() >= MIN_CENTERLINE_POINTS:
                    path = _longest_path_in_tree(sub) if nx.is_tree(sub) \
                        else _longest_path_general(sub)
                    if len(path) >= MIN_CENTERLINE_POINTS:
                        segments.append(path)
            return segments

    return []


# ===================================================================
# Stage 4 -- Smooth and simplify
# ===================================================================

def stage4_smooth_simplify(coords):
    """Light smoothing of selected measured points with optional simplification."""
    n = len(coords)
    if n < 5:
        return coords

    # Use a smaller SG window to preserve measured geometry.
    win = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if win < 5:
        win = 5
    if win % 2 == 0:
        win += 1
    poly = min(SAVGOL_POLYORDER, win - 2)

    if win <= n and poly >= 1:
        x_smooth = savgol_filter(coords[:, 0], win, poly)
        y_smooth = savgol_filter(coords[:, 1], win, poly)
    else:
        x_smooth, y_smooth = coords[:, 0], coords[:, 1]

    smoothed = np.column_stack([x_smooth, y_smooth])

    # Douglas-Peucker simplification via shapely
    if len(smoothed) >= 2:
        try:
            line = LineString(smoothed)
            simplified = simplify(line, DP_TOLERANCE_M, preserve_topology=False)
            if simplified.geom_type == "LineString" and len(simplified.coords) >= 2:
                return np.array(simplified.coords)
        except Exception:
            pass

    return smoothed


# ===================================================================
# Stage 5 -- QC pass
# ===================================================================

def stage5_qc(centerline, original_pts, dem_hs, dem_transform, trace_id,
              qc_dir, stage1_info):
    """Overlay cleaned centerline on hillshade; compute QC metrics."""
    n_orig = len(original_pts)
    n_clean = len(centerline)

    # Length and sinuosity
    diffs = np.diff(centerline, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    length_m = float(seg_lens.sum())

    straight_dist = float(np.sqrt(
        (centerline[-1, 0] - centerline[0, 0])**2 +
        (centerline[-1, 1] - centerline[0, 1])**2
    ))
    sinuosity = length_m / straight_dist if straight_dist > 0 else 1.0

    # Fraction of original points close to centerline
    if n_orig > 0 and n_clean >= 2:
        centerline_line = LineString(centerline)
        distances = np.array([
            centerline_line.distance(Point(p))
            for p in original_pts
        ])
        close_frac = float((distances < RASTER_CELL_SIZE_M * 2).mean())
        median_dist = float(np.median(distances))
    else:
        close_frac = 0.0
        median_dist = float('nan')

    qc_flag = "OK" if close_frac > 0.6 else "REVIEW"

    print(f"  [Stage 5] Trace {trace_id}: length={length_m:.0f} m, "
          f"sinuosity={sinuosity:.2f}, close_frac={close_frac:.2f} "
          f"[{qc_flag}]")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    x0 = min(original_pts[:, 0].min(), centerline[:, 0].min()) - 200
    x1 = max(original_pts[:, 0].max(), centerline[:, 0].max()) + 200
    y0 = min(original_pts[:, 1].min(), centerline[:, 1].min()) - 200
    y1 = max(original_pts[:, 1].max(), centerline[:, 1].max()) + 200

    if dem_hs is not None:
        inv = ~dem_transform
        c0, r0 = inv * (x0, y1)
        c1, r1 = inv * (x1, y0)
        r0i = int(max(0, min(r0, r1)))
        r1i = int(min(dem_hs.shape[0], max(r0, r1)))
        c0i = int(max(0, min(c0, c1)))
        c1i = int(min(dem_hs.shape[1], max(c0, c1)))
        if r1i > r0i and c1i > c0i:
            hs_crop = dem_hs[r0i:r1i, c0i:c1i]
            extent = [
                dem_transform.c + c0i * dem_transform.a,
                dem_transform.c + c1i * dem_transform.a,
                dem_transform.f + r1i * dem_transform.e,
                dem_transform.f + r0i * dem_transform.e,
            ]
            ax.imshow(hs_crop, extent=extent, cmap="gray", aspect="equal",
                      alpha=0.5)

    ax.scatter(original_pts[:, 0], original_pts[:, 1], s=1, c="blue",
               alpha=0.3, label=f"Original ({n_orig})")
    ax.plot(centerline[:, 0], centerline[:, 1], "r-", linewidth=2,
            label=f"Cleaned ({n_clean})")
    ax.scatter(centerline[:, 0], centerline[:, 1], c="red", s=10, zorder=5)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(f"Stage 5 QC -- Trace {trace_id}\n"
                 f"length={length_m:.0f} m, sinuosity={sinuosity:.2f}, "
                 f"close_frac={close_frac:.2f} [{qc_flag}]", fontsize=11)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(qc_dir / f"stage5_trace_{trace_id:02d}.png", dpi=PLOT_DPI)
    plt.close(fig)

    return {
        "length_m": length_m,
        "sinuosity": sinuosity,
        "close_frac": close_frac,
        "median_dist_m": median_dist,
        "qc_flag": qc_flag,
    }


def process_trace_file_with_limits(
    fpath,
    break_type_lookup,
    dem_src,
    dem_hs,
    dem_transform,
    qc_dir,
    trace_id_start,
    skip_diagnose=False,
    skip_terrain=False,
    max_traces_per_file=MAX_TRACES_PER_FILE,
):
    """Process one Trace*.xlsx using strict filters and output limits."""
    coords, attributes = load_trace_points_with_break_types(fpath, break_type_lookup)
    break_types = attributes["break_type"]
    n_original = len(coords)

    if n_original < MIN_CLUSTER_POINTS:
        print(f"  Too few points ({n_original}), skipping")
        return [], trace_id_start

    diagnostics = {}
    if not skip_diagnose:
        diagnostics = stage0_diagnose(coords, fpath.name, dem_hs, dem_transform, qc_dir)

    if skip_terrain or dem_src is None:
        cleaned_coords, cleaned_break_types, stage1_info = stage1_outlier_rejection_with_types(
            coords, break_types, None, diagnostics
        )
    else:
        cleaned_coords, cleaned_break_types, stage1_info = stage1_outlier_rejection_with_types(
            coords, break_types, dem_src, diagnostics
        )

    if len(cleaned_coords) < MIN_CLUSTER_POINTS:
        print(f"  Too few points after cleaning ({len(cleaned_coords)}), skipping")
        return [], trace_id_start

    clusters, cluster_break_types = stage2_segment_with_break_types(
        cleaned_coords, cleaned_break_types
    )
    if not clusters:
        return [], trace_id_start

    all_candidate_traces = []
    all_candidate_break_types = []
    for ci, (cluster_pts, cluster_bts) in enumerate(zip(clusters, cluster_break_types)):
        print(f"\n  [Cluster {ci}] {len(cluster_pts)} points")
        guide_lines = _extract_centerline_guides_parallel_aware(cluster_pts)
        if not guide_lines:
            continue

        for guide in guide_lines:
            traces, trace_bts_list = adaptive_trace_extraction(cluster_pts, cluster_bts, guide)
            for trace, trace_bts in zip(traces, trace_bts_list):
                if len(trace) >= MIN_POINTS_PER_TRACE:
                    all_candidate_traces.append(trace)
                    all_candidate_break_types.append(trace_bts)

    print(f"\n  Found {len(all_candidate_traces)} candidate traces")
    if not all_candidate_traces:
        return [], trace_id_start

    if len(all_candidate_traces) > 1:
        merged_traces, merged_bts = merge_nearby_segments(
            all_candidate_traces,
            all_candidate_break_types,
        )
    else:
        merged_traces = all_candidate_traces
        merged_bts = all_candidate_break_types

    final_traces, final_bts = extract_high_confidence_traces(
        merged_traces,
        merged_bts,
        max_traces=max_traces_per_file,
    )
    print(f"  Final high-confidence traces: {len(final_traces)}")

    manifest_rows = []
    trace_id = trace_id_start
    for trace, trace_bts in zip(final_traces, final_bts):
        densified = densify_point_sequence(
            trace,
            min_spacing=POINT_SPACING_MIN_M,
            max_spacing=POINT_SPACING_MAX_M,
        )
        smoothed = stage4_smooth_simplify(densified)

        if trace_bts is not None and len(trace_bts) > 0:
            bt_tree = cKDTree(trace)
            _, bt_idx = bt_tree.query(smoothed, k=1)
            smoothed_bts = np.array(trace_bts, dtype=object)[np.atleast_1d(bt_idx).astype(int)]
        else:
            smoothed_bts = np.array(["unknown"] * len(smoothed), dtype=object)

        is_valid, quality_score, issues = validate_trace_quality(smoothed, smoothed_bts)
        if not is_valid:
            print(f"  Trace rejected in final validation: {', '.join(issues)}")
            continue

        dominant_type, consistency = get_dominant_break_type(smoothed_bts)

        dx = smoothed[-1, 0] - smoothed[0, 0]
        dy = smoothed[-1, 1] - smoothed[0, 1]
        azimuth = np.degrees(np.arctan2(dx, dy)) % 360

        qc = stage5_qc(
            smoothed,
            cleaned_coords,
            dem_hs,
            dem_transform,
            trace_id,
            qc_dir,
            stage1_info,
        )

        csv_path = CLEANED_DIR / f"trace_{trace_id:02d}.csv"
        cl_df = pd.DataFrame(
            {
                "x": smoothed[:, 0],
                "y": smoothed[:, 1],
                "break_type": [dominant_type] * len(smoothed),
            }
        )
        cl_df.to_csv(csv_path, index=False)
        print(f"    -> {csv_path.name} ({len(smoothed)} vertices)")

        manifest_rows.append(
            {
                "trace_id": trace_id,
                "source_file": fpath.name,
                "n_points_original": n_original,
                "n_points_after_stage1": stage1_info["n_after_terrain"],
                "n_centerline_vertices": len(smoothed),
                "length_m": qc["length_m"],
                "azimuth_deg": azimuth,
                "sinuosity": qc["sinuosity"],
                "close_frac": qc["close_frac"],
                "median_dist_m": qc["median_dist_m"],
                "qc_flag": qc["qc_flag"],
                "break_type": dominant_type,
                "break_type_consistency": consistency,
                "quality_score": quality_score,
                "density_threshold_m": stage1_info.get("density_threshold_m", 0),
                "csv_path": csv_path.name,
            }
        )
        trace_id += 1

    return manifest_rows, trace_id


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-source", default=str(DEM_PATH),
                        help="Path to DEM GeoTIFF (default: lidar_0_5m_10m_smoke)")
    parser.add_argument("--skip-diagnose", action="store_true",
                        help="Skip Stage 0 diagnostic plots")
    parser.add_argument("--skip-terrain", action="store_true",
                        help="Skip Stage 1b terrain filter")
    args = parser.parse_args()

    print("=" * 70)
    print("06a-ScarpTraceCleaning.py")
    print("Pre-process raw scarp trace point clouds -> cleaned polylines")
    print("=" * 70)

    safe_makedirs(CLEANED_DIR)
    safe_makedirs(QC_DIR)

    # Remove old outputs
    for old in CLEANED_DIR.glob("trace_*.csv"):
        old.unlink()
    manifest_path = CLEANED_DIR / "manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()

    # Load DEM
    dem_path = Path(args.dem_source)
    if not dem_path.exists():
        print(f"[Warning] DEM not found: {dem_path}")
        print("          Terrain filter will be skipped; Stage 0/5 will omit hillshade.")
        dem_src = None
        dem_hs = None
        dem_transform = None
    else:
        print(f"\n[DEM] Loading {dem_path.name}...")
        dem_src = rasterio.open(dem_path)
        dem_data = dem_src.read(1).astype(np.float32)
        nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
        dem_data[dem_data == nodata] = np.nan
        dem_hs = _compute_hillshade(dem_data, dem_src.transform)
        dem_transform = dem_src.transform
        print(f"  Shape: {dem_src.shape}, CRS: {dem_src.crs}, "
              f"res: {dem_src.res}")

    # Build break-type lookup from inferred points (reference only, not processing input).
    lookup_paths = [p for p in DEFAULT_ALL_INFERRED_FILES if p.exists()]
    if not lookup_paths:
        raise FileNotFoundError(
            f"No all_inferred_points.csv files found under {KNOWN_TRACE_VALIDATION_DIR}"
        )
    break_type_lookup = build_break_type_lookup(lookup_paths)

    # Strictly use Trace1.xlsx and Trace2.xlsx as processing input.
    trace_files = [HIGHCON_DIR / name for name in TRACE_INPUT_FILES]
    missing_trace_inputs = [p for p in trace_files if not p.exists()]
    if missing_trace_inputs:
        raise FileNotFoundError(
            f"Missing required trace input files: {[p.name for p in missing_trace_inputs]}"
        )

    print(f"\n[Traces] Using fixed inputs: {[p.name for p in trace_files]}")

    manifest_rows = []
    trace_id = 0

    for fi, fpath in enumerate(trace_files):
        print(f"\n{'='*50}")
        print(f"Processing {fpath.name}")
        print(f"{'='*50}")

        file_rows, trace_id = simplified_pipeline_for_trace(
            fpath,
            break_type_lookup,
            dem_src,
            dem_hs,
            dem_transform,
            QC_DIR,
            trace_id_start=trace_id,
            skip_diagnose=args.skip_diagnose,
            skip_terrain=args.skip_terrain,
        )
        manifest_rows.extend(file_rows)

    # Write manifest
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\n[Manifest] {len(manifest_df)} traces -> {manifest_path}")
    else:
        print(f"\n[Warning] No valid traces produced")

    # Summary
    print(f"\n{'='*70}")
    print("06a-ScarpTraceCleaning.py -- DONE")
    print(f"  Output:   {CLEANED_DIR}")
    print(f"  Traces:   {len(manifest_rows)}")
    print(f"  QC plots: {QC_DIR}")
    print(f"{'='*70}")

    if dem_src is not None:
        dem_src.close()


if __name__ == "__main__":
    main()
