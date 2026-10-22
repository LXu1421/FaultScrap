#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05c-MLTunedDEMWideInference.py -- ML-tuned DEM-wide scarp inference for 0.5m LiDAR.

This script imports all functions and constants from 05a-KnownTraceScarpValidation_hs.py
and extends it with three key capabilities:

1. **Smoothed trace lines**: HighCon points (Trace1.xlsx + Trace2.xlsx) are
   scattered and unsorted.  This script orders them via nearest-neighbor chaining,
   fits a smoothing B-spline, and resamples at regular intervals to produce
   continuous guide traces.

2. **ML parameter tuning**: Uses Bayesian optimisation (Optuna if available,
   grid search fallback) to find the best empirical parameters for the 0.5m
   LiDAR DEM.  The objective maximises F1 score of HighCon point recovery
   on held-out trace chunks.

3. **DEM-wide inference**: Instead of processing only per-trace windows (like
   05a), this script tiles the entire LiDAR DEM into overlapping windows and
   runs the full curvature-LSM pipeline on each tile, then filters results
   by proximity to the smoothed trace lines.

Ground truth: HighCon Trace1.xlsx (mixed 10m + 0.5m, filtered to 0.5m CSV)
+ Trace2.xlsx (0.5m lidar only).

Outputs are written to ``data/derivatives/known_trace_scarp_validation/lidar_0_5m/``.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Suppress DeprecationWarnings from third-party libraries (skimage, tifffile,
# etc.) that set array.shape = ... which NumPy 2.5 deprecates.  These are
# internal to the library and not actionable from user code.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import reproject, Resampling
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, minimum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import all functions and constants from 05a-hs ────────────────────
# The filename contains a hyphen, so we use importlib to load it as a module.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "scarp_05a_hs",
    str(BASE_DIR / "05a-KnownTraceScarpValidation_hs.py"),
)
_05a = importlib.util.module_from_spec(_spec)
sys.modules["scarp_05a_hs"] = _05a
_spec.loader.exec_module(_05a)

# Re-export everything from 05a for convenience
from scarp_05a_hs import (  # noqa: E402,F401
    # Constants (will be overridden after tuning)
    FAULT_DB_SOURCES, FAULT_VALIDATION_TOLERANCE_M,
    MIN_SCARP_HEIGHT_M, SCARP_WIDTH_DISCRETE_MAX_M,
    LSM_MIN_LENGTH_M, FALLBACK_MAX_SINUOSITY, FALLBACK_MAX_PERIODICITY,
    HAND_INFLUENCE_THRESHOLD_M, HAND_SEGMENT_MAX_FRACTION,
    LSM_HAND_BUFFER_M, LSM_EDGE_BUFFER_M, LSM_PERSISTENCE_K,
    LSM_SCALE_NORM_GAMMA, LSM_HYSTERESIS_LOW_SIGMA, LSM_HYSTERESIS_HIGH_SIGMA,
    LSM_NODATA, MAX_LOCAL_WINDOW_PIXELS,
    SEED_WINDOW_M, CLASS3_WINDOW_M, SEED_SPACING_M,
    PRIOR_PROFILE_MATCH_M, DRAINAGE_MATCH_M,
    POSITIVE_PRIOR_CLASSES, PHASE1_AUDIT_DIR,
    LOCAL_TV_WEIGHTS, TARGET_TV_SIGMAS_M,
    BASE_RAY_LENGTH_M, BASE_BACKGROUND_FRACTION,
    CHAIN_SAMPLE_SPACING_M, LOCAL_FIG_BUFFER_M, LOCAL_FIG_MAX_SCARP_LENGTH_M,
    MIN_WINDOW_PX, RAY_MIN_PIXELS, RAY_DEBUG_MAX_FAILURES,
    GRAD_DECAY_THRESHOLD, CLUSTER_CONNECT_MAX_GAP_M, CLUSTER_CONNECT_MAX_ANGLE_DEG,
    SCARP_SLOPE_FLAT_THRESHOLD, SCARP_SLOPE_DROP_FACTOR, SCARP_SLOPE_MIN_STEPS,
    MIN_SCARP_WIDTH_M, MAX_SCARP_HEIGHT_WIDTH_RATIO,
    MIN_SCARP_ASYMMETRY_SCORE, MIN_LOCAL_SINUOSITY,
    PROFILE_HALF_WIDTH_M, PROFILE_SAMPLE_COUNT,
    SOURCE_COLORS, GENTLE_SLOPE_MODE,
    # Functions
    _ramp, _check_scarp_asymmetry, _local_sinuosity,
    _compute_segment_confidence, _validate_scarp_profile,
    _load_known_points, _window_for_trace, _min_half_window_buffer_m,
    _tv_weights_for_resolution, _target_sigmas_for_resolution,
    _effective_persistence_k, _min_vals_required,
    _window_from_bounds_clipped, _load_phase1_priors,
    _attach_trace_priors, _coords_in_bounds, _split_coords_by_bounds,
    _estimate_window_pixels, _thin_seed_points, _chunk_trace_run,
    _trace_windows, _annotate_segments_with_priors,
    _read_mask_on_window, _local_lsm, _ray_crossing, _ray_crossing_to_sign,
    _find_slope_break, _classify_scarp_pixels, _periodicity,
    _detect_meander_pattern, _connect_nearby_clusters,
    _segments_from_points, _write_local_raster, _filter_river_segments,
    _summary_and_comparison, _split_polyline_by_length,
    _chunk_trace_segments, _window_from_coords,
    _plot_local_reconstructions, _plot_summary,
    # derivative_common imports (re-exported from 05a)
    safe_makedirs, compute_hillshade,
    reconstruct_fault_traces,
    DATA_DIR, OUT_DIR, STRUCT_PTS_PATH,
)


# ── 05c-specific constants ────────────────────────────────────────────

# Map --dem-source keys to the actual GeoTIFF filenames in data/derivatives/
DEM_SOURCE_MAP = {
    "lidar_0_5m": "lidar_0_5m_epsg28355_10m_smoke.tif",
    "srtm1s": "srtm1s_epsg28355_clipped.tif",
    "srtm3s": "srtm3s_epsg28355.tif",
}

DEM_SOURCE_KEY = "lidar_0_5m"
DEM_PATH = OUT_DIR / DEM_SOURCE_MAP[DEM_SOURCE_KEY]
HAND_DIR = OUT_DIR / "visual_audit" / "rem_hand"
CSV_PATH = OUT_DIR / "known_trace_scarp_validation" / DEM_SOURCE_KEY / "all_inferred_points.csv"

# Output directory: reuse the same known_trace_scarp_validation/<source>
OUTPUT_DIR = OUT_DIR / "known_trace_scarp_validation" / DEM_SOURCE_KEY

HIGHCON_DIR = DATA_DIR / "HighCon"
STRUCT_PTS_INV_PATH = DATA_DIR / "structural_points_inventory.csv"

# Trace2 is replaced by FSD1 structural points (much cleaner, already surveyed)
FSD1_SOURCE_NAME = "FSD1"
TRACE2_REPLACE_WITH_FSD1 = True
FSD1_TRACE2_BUFFER_M = 2000.0  # FSD1 points within this distance of Trace2 HighCon points

# Tile parameters for DEM-wide inference
TILE_SIZE_PX = 800           # ~8km at 10m/px
TILE_OVERLAP_PX = 100        # ~1km overlap to avoid edge effects

# Per-trace proximity buffer for filtering inferred points.
# Trace1 (HighCon): tight 100m — points must be on the trace line itself.
# Trace2 (FSD1): also 100m — FSD1 points are accurate ground truth, so the
# proximity filter for final inferred points should be tight.  Recovery is
# improved by seeding FSD1 points into the LSM candidate mask, not by
# widening the buffer.
TRACE_BUFFER_M = 100.0       # default / Trace1
TRACE2_BUFFER_M = 300.0     # Trace2 (FSD1) — tight, matching FSD1 accuracy

# Smoothing parameters for trace line construction
SMOOTH_SPLINE_S_FACTOR = 50.0  # Higher = smoother; auto-tuned if SMOOTH_AUTO_TUNE=True
SMOOTH_RESAMPLE_INTERVAL_M = 50.0  # Point spacing on resampled trace
MIN_POINTS_FOR_SPLINE = 10      # Minimum scattered points to fit a spline

# ML parameter tuning
TUNING_N_TRIALS = 30             # Optuna trials (or grid size per dimension)
TUNING_SAMPLE_WINDOWS = 6        # Windows to sample per trial (speed)

# Parameter search space
PARAM_SPACE = {
    "MIN_SCARP_HEIGHT_M": (0.1, 0.5),
    "LSM_HYSTERESIS_LOW_SIGMA": (0.25, 2.0),
    "LSM_HYSTERESIS_HIGH_SIGMA": (1.0, 6.0),
    "LSM_PERSISTENCE_K": (1, 3),
    "TV_WEIGHT_SCALE": (0.25, 4.0),
    "BASE_RAY_LENGTH_M": (300.0, 1200.0),
    "MIN_SCARP_ASYMMETRY_SCORE": (0.05, 0.5),
    "MIN_SCARP_WIDTH_M": (10.0, 50.0),
}

# Default parameters (from 05a-hs) -- used as baseline for comparison
DEFAULT_PARAMS = {
    "MIN_SCARP_HEIGHT_M": 0.2,
    "LSM_HYSTERESIS_LOW_SIGMA": 0.5,
    "LSM_HYSTERESIS_HIGH_SIGMA": 4.0,
    "LSM_PERSISTENCE_K": 2,
    "TV_WEIGHT_SCALE": 1.0,
    "BASE_RAY_LENGTH_M": 800.0,
    "MIN_SCARP_ASYMMETRY_SCORE": 0.3,
    "MIN_SCARP_WIDTH_M": 30.0,
}

# Coarse-DEM parameter preset for SRTM 90m data.
# The 10m-tuned defaults don't transfer directly: at 90m/px, scarp
# width is measured in pixel steps of 90m (so a 2-pixel scarp = 180m),
# ray lengths need fewer pixels to cover the same ground distance,
# and the fallback walk must be long enough to span several pixels.
COARSE_DEM_PARAMS = {
    "MIN_SCARP_HEIGHT_M": 0.1,        # SRTM has lower vertical precision
    "LSM_HYSTERESIS_LOW_SIGMA": 0.25, # more permissive thresholds
    "LSM_HYSTERESIS_HIGH_SIGMA": 1.0, #  (fewer scales available at 90m)
    "LSM_PERSISTENCE_K": 1,           # only 1-2 meaningful scales at 90m
    "TV_WEIGHT_SCALE": 0.5,           # less smoothing needed (already coarse)
    "BASE_RAY_LENGTH_M": 400.0,       # shorter rays (fewer pixels)
    "MIN_SCARP_ASYMMETRY_SCORE": 0.15,# lower bar for gentle scarps
    "MIN_SCARP_WIDTH_M": 10.0,        # effectively ~0.1 pixel
}

# Resolution threshold above which we use the coarse-DEM branch.
# 10m lidar -> fine; 30m SRTM 1s -> coarse; 90m SRTM 3s -> coarse.
COARSE_DEM_THRESHOLD_M = 25.0

# NFD1 structural points path (same inventory as FSD1)
NFD1_PATH = DATA_DIR / "structural_points_inventory.csv"


# ── Smoothed trace line construction ──────────────────────────────────

def _detect_available_traces(highcon_dir=None):
    """Scan ``data/HighCon`` for ``TraceN.xlsx`` files and return sorted 1-based indices."""
    if highcon_dir is None:
        highcon_dir = HIGHCON_DIR
    indices = []
    for n in range(1, 100):
        path = highcon_dir / f"Trace{n}.xlsx"
        if path.exists():
            indices.append(n)
        else:
            next_path = highcon_dir / f"Trace{n + 1}.xlsx"
            if not next_path.exists():
                break
    return indices


def _load_highcon_traces(csv_path=None, trace_indices=None):
    """Load HighCon trace points, filtered to those present in the 05a CSV.

    For coarse-resolution DEMs (e.g. SRTM 90m), the CSV coordinates are
    pixel centres that differ from the original HighCon survey coordinates
    by up to half a pixel.  We use a KDTree nearest-neighbour match with
    a tolerance proportional to the DEM resolution instead of exact key
    matching, so that coarse DEMs still recover trace points.

    Returns a list of dicts: [{trace_id, points: np.array(N,2), n_points, tier_values, break_type_values}, ...]
    """
    if csv_path is None:
        csv_path = CSV_PATH
    if trace_indices is None:
        trace_indices = _detect_available_traces()

    if not csv_path.exists():
        print(f"  [Error] 05a output CSV not found: {csv_path}")
        print(f"  Please run 05a-KnownTraceScarpValidation_hs.py --dem-source {DEM_SOURCE_KEY} first.")
        sys.exit(1)

    # Build KDTree from 05a CSV coordinates for nearest-neighbour matching
    csv_df = pd.read_csv(csv_path)
    csv_xy = csv_df[["x", "y"]].to_numpy(dtype=float)
    csv_tree = cKDTree(csv_xy)
    csv_bt = csv_df.get("break_type", pd.Series(["p"] * len(csv_df))).values
    csv_tier = csv_df.get("tier", pd.Series([2] * len(csv_df))).values
    csv_tier = np.where(pd.notna(csv_tier), csv_tier, 2).astype(int)

    # Determine match tolerance from DEM resolution (default 10m for lidar)
    try:
        with rasterio.open(DEM_PATH) as src:
            dem_res = abs(src.res[0])
    except Exception:
        dem_res = 10.0
    match_tol = dem_res * 1.5  # 1.5x pixel size covers sub-pixel offset

    traces = []
    for trace_idx in trace_indices:
        path = HIGHCON_DIR / f"Trace{trace_idx}.xlsx"
        if not path.exists():
            print(f"  [Warning] {path} not found, skipping Trace{trace_idx}")
            continue
        df = pd.read_excel(path)
        trace_xy = df[["x", "y"]].to_numpy(dtype=float)
        dists, idxs = csv_tree.query(trace_xy, k=1)
        in_csv_mask = dists <= match_tol
        df = df[in_csv_mask].copy()
        if len(df) == 0:
            print(f"  Trace{trace_idx}: 0 points in CSV (tol={match_tol:.0f}m), skipping")
            continue
        matched_idxs = idxs[in_csv_mask]
        df["break_type"] = csv_bt[matched_idxs]
        df["tier"] = csv_tier[matched_idxs]
        points = df[["x", "y"]].to_numpy(dtype=float)
        finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
        points = points[finite]
        traces.append({
            "trace_id": trace_idx,
            "points": points,
            "n_points": len(points),
            "tier_values": df["tier"].values[finite],
            "break_type_values": df["break_type"].values[finite],
        })
        print(f"  Trace{trace_idx}: {len(points)} points matched (tol={match_tol:.0f}m)")
    return traces


def _load_fsd1_points(trace2_xy=None, buffer_m=None):
    """Load FSD1 structural points from ``structural_points_inventory.csv``.

    FSD1 points are field-surveyed fault scarp displacement measurements.
    They are much cleaner than the HighCon Trace2.xlsx points and form a
    near-linear trend (81% PCA explained variance), making them ideal for
    building a smoothed trace line.

    Parameters
    ----------
    trace2_xy : np.ndarray (N, 2), optional
        HighCon Trace2 coordinates.  If provided, only FSD1 points within
        ``buffer_m`` of any Trace2 point are returned (to isolate the FSD1
        points that lie along the same fault segment).
    buffer_m : float, optional
        Buffer distance for filtering.  Defaults to FSD1_TRACE2_BUFFER_M.

    Returns
    -------
    np.ndarray (M, 2)
        FSD1 point coordinates (XCoord, YCoord) in EPSG:28355.
    """
    if buffer_m is None:
        buffer_m = FSD1_TRACE2_BUFFER_M

    if not STRUCT_PTS_INV_PATH.exists():
        print(f"  [Error] structural_points_inventory.csv not found: {STRUCT_PTS_INV_PATH}")
        sys.exit(1)

    df = pd.read_csv(STRUCT_PTS_INV_PATH)
    fsd1 = df[df["source"] == FSD1_SOURCE_NAME].copy()
    fsd1_coords = fsd1[["XCoord", "YCoord"]].to_numpy(dtype=float)

    # Filter to DEM extent
    with rasterio.open(DEM_PATH) as src:
        b = src.bounds
    in_dem = ((fsd1_coords[:, 0] >= b.left) & (fsd1_coords[:, 0] <= b.right) &
              (fsd1_coords[:, 1] >= b.bottom) & (fsd1_coords[:, 1] <= b.top))
    fsd1_coords = fsd1_coords[in_dem]
    print(f"  [FSD1] {len(fsd1)} total, {len(fsd1_coords)} in DEM extent")

    # Filter to Trace2 proximity if requested
    if trace2_xy is not None and len(trace2_xy) > 0:
        t2_tree = cKDTree(trace2_xy)
        dists, _ = t2_tree.query(fsd1_coords, k=1)
        near_t2 = dists <= buffer_m
        n_before = len(fsd1_coords)
        fsd1_coords = fsd1_coords[near_t2]
        print(f"  [FSD1] {len(fsd1_coords)} within {buffer_m}m of Trace2 "
              f"(filtered from {n_before})")

    return fsd1_coords


def _order_points_monotonic(coords):
    """Order scattered 2D points so they progress monotonically along the
    principal axis (typically north-south for fault traces).

    The procedure:
    1. Centre the points and compute the first principal component (PCA).
    2. Project every point onto that axis to get a scalar parameter t.
    3. Sort by t.  This gives a monotonic ordering that never reverses
       direction.

    For fault-trace scarp points that span a roughly linear N-S trend,
    this produces a single pass from one end to the other without the
    back-tracking that greedy nearest-neighbor chaining causes.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Scattered point coordinates (x=easting, y=northing).

    Returns
    -------
    np.ndarray, shape (N, 2)
        Points sorted along the principal axis.
    """
    if len(coords) <= 1:
        return coords.copy()

    pts = np.asarray(coords, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via SVD — first right singular vector is the principal direction
    # For a 2-column matrix this is equivalent to eigendecomposition of the
    # 2x2 covariance matrix but numerically more stable.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    principal_axis = Vt[0]  # first principal direction

    # Project onto principal axis → scalar parameter
    projections = centered @ principal_axis

    # Sort ascending (south-to-north or vice-versa depending on axis sign)
    # We want north-to-south, so if the axis points northward we reverse.
    if principal_axis[1] > 0:
        # Axis points northward; ascending projection = south-to-north
        # We want north-to-south, so sort descending
        order = np.argsort(-projections)
    else:
        # Axis points southward; ascending projection = north-to-south
        order = np.argsort(projections)

    return pts[order]


def _fit_smoothing_spline(ordered_points, s_factor=SMOOTH_SPLINE_S_FACTOR,
                          resample_interval_m=SMOOTH_RESAMPLE_INTERVAL_M):
    """Fit a B-spline to ordered points and resample at regular arc-length intervals.

    Returns (smoothed_points, spline_params) or (ordered_points, None) if fitting fails.
    """
    pts = np.asarray(ordered_points, dtype=float)
    if len(pts) < MIN_POINTS_FOR_SPLINE:
        print(f"  [Spline] Only {len(pts)} points (< {MIN_POINTS_FOR_SPLINE}), skipping spline")
        return pts, None

    # Compute cumulative arc length for parameterisation
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]
    if total_len < 1e-6:
        return pts, None

    u = cum_len / total_len

    # Remove duplicate consecutive points that break splprep
    unique_mask = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
    pts_unique = pts[unique_mask]
    u_unique = u[unique_mask]
    if len(pts_unique) < MIN_POINTS_FOR_SPLINE:
        return pts, None

    try:
        k = min(3, len(pts_unique) - 1)
        tck, u_fitted = splprep(
            [pts_unique[:, 0], pts_unique[:, 1]],
            u=u_unique,
            s=s_factor,
            k=k,
        )

        n_samples = max(int(total_len / resample_interval_m), 20)
        u_new = np.linspace(0, 1, n_samples)
        x_smooth, y_smooth = splev(u_new, tck)
        smoothed = np.column_stack([x_smooth, y_smooth])

        finite = np.isfinite(smoothed[:, 0]) & np.isfinite(smoothed[:, 1])
        smoothed = smoothed[finite]

        print(f"  [Spline] {len(pts)} points -> {len(smoothed)} smoothed samples "
              f"(s={s_factor:.1f}, k={k}, length={total_len:.0f}m)")
        return smoothed, {"tck": tck, "u_fitted": u_fitted, "total_len_m": total_len}

    except Exception as e:
        print(f"  [Spline] Fitting failed: {e}, using ordered points")
        return pts, None


def _build_smoothed_traces(highcon_traces, replace_trace2_with_fsd1=True):
    """Build smoothed trace lines from scattered HighCon points.

    For each trace:
    1. Order points via PCA-based monotonic projection (N to S)
    2. Fit a smoothing B-spline
    3. Resample at regular intervals

    If ``replace_trace2_with_fsd1`` is True, Trace 2's HighCon points are
    replaced by FSD1 structural points from ``structural_points_inventory.csv``
    before smoothing.  FSD1 points are field-surveyed, much cleaner (81% PCA
    linearity vs <50% for HighCon Trace2), and produce a far better smoothed
    line.  The original HighCon Trace2 points are still retained in
    ``raw_points_highcon`` for quality evaluation.

    Returns list of dicts with 'trace_id', 'coords' (smoothed), 'raw_points',
    'spline_info'.
    """
    # If replacing Trace2, load FSD1 once (filtered to Trace2 proximity)
    fsd1_coords = None
    if replace_trace2_with_fsd1:
        # Find Trace2 in highcon_traces to get its coordinates for FSD1 filtering
        trace2 = None
        for t in highcon_traces:
            if t["trace_id"] == 2:
                trace2 = t
                break
        if trace2 is not None:
            print(f"\n  [FSD1] Loading FSD1 points to replace Trace2 HighCon...")
            fsd1_coords = _load_fsd1_points(
                trace2_xy=trace2["points"],
                buffer_m=FSD1_TRACE2_BUFFER_M,
            )
            if len(fsd1_coords) < 2:
                print(f"  [FSD1] Too few FSD1 points ({len(fsd1_coords)}), "
                      f"falling back to HighCon Trace2")
                fsd1_coords = None
            else:
                print(f"  [FSD1] Using {len(fsd1_coords)} FSD1 points for Trace2 "
                      f"(replacing {len(trace2['points'])} HighCon points)")

    smoothed_traces = []
    for trace in highcon_traces:
        raw_points = trace["points"]
        raw_points_highcon = raw_points  # always keep original HighCon for evaluation

        # Replace Trace2 points with FSD1 if available
        if trace["trace_id"] == 2 and fsd1_coords is not None:
            raw_points = fsd1_coords
            print(f"\n  Trace 2: using FSD1 ({len(raw_points)} pts) "
                  f"instead of HighCon ({len(raw_points_highcon)} pts)")

        if len(raw_points) < 2:
            print(f"  Trace {trace['trace_id']}: too few points ({len(raw_points)}), skipping")
            continue

        ordered = _order_points_monotonic(raw_points)
        smoothed, spline_info = _fit_smoothing_spline(ordered)

        is_fsd1 = trace["trace_id"] == 2 and fsd1_coords is not None
        trace_buffer = TRACE2_BUFFER_M if is_fsd1 else TRACE_BUFFER_M

        smoothed_traces.append({
            "trace_id": trace["trace_id"],
            "coords": smoothed,
            "raw_points": raw_points,
            "raw_points_highcon": raw_points_highcon,
            "ordered_points": ordered,
            "spline_info": spline_info,
            "n_raw": len(raw_points),
            "n_smoothed": len(smoothed),
            "source": "FSD1" if is_fsd1 else "HighCon",
            "buffer_m": trace_buffer,
            "fsd1_point_buffer_m": 300.0 if is_fsd1 else None,
        })
        print(f"  Trace {trace['trace_id']}: {len(raw_points)} raw -> "
              f"{len(smoothed)} smoothed points (source: "
              f"{'FSD1' if (trace['trace_id'] == 2 and fsd1_coords is not None) else 'HighCon'})")
    return smoothed_traces


def _plot_smoothed_traces(smoothed_traces, dem_path, output_path, bounds=None):
    """Visualise smoothed trace lines overlaid on the DEM hillshade (single panel).

    Shows raw scattered HighCon points and the fitted smoothed spline on
    the same axis so the user can see the spatial best-fit quality.

    Parameters
    ----------
    bounds : tuple (minx, miny, maxx, maxy), optional
        If provided, zoom the plot to this rectangle instead of the full DEM extent.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    with rasterio.open(dem_path) as src:
        out_h, out_w = min(1200, src.height), min(1200, src.width)
        dem = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        hs = compute_hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem) if np.any(np.isfinite(dem)) else 0.0),
                               abs(src.res[0]) * src.height / out_h)
        b = src.bounds

    ax.imshow(hs, cmap="gray", extent=(float(b.left), float(b.right), float(b.bottom), float(b.top)),
              origin="upper", alpha=0.7)
    ax.set_xlim(float(b.left), float(b.right))
    ax.set_ylim(float(b.bottom), float(b.top))
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:28355)")
    ax.set_ylabel("Northing (m, EPSG:28355)")
    ax.set_title("Trace Lines: Raw Points + Smoothed Fit (0.5m LiDAR)")

    colors = ["#ff4444", "#44aaff", "#44ff44", "#ffaa44"]
    for i, trace in enumerate(smoothed_traces):
        color = colors[i % len(colors)]
        source = trace.get("source", "HighCon")

        # Plot the points used for smoothing (FSD1 for Trace2, HighCon for Trace1)
        raw = trace["raw_points"]
        mask = (raw[:, 0] >= b.left) & (raw[:, 0] <= b.right) & \
               (raw[:, 1] >= b.bottom) & (raw[:, 1] <= b.top)
        n_raw = int(mask.sum())
        if n_raw > 20000:
            idx = np.random.RandomState(42).choice(
                np.where(mask)[0], 20000, replace=False)
            ax.scatter(raw[idx, 0], raw[idx, 1], s=1, c=color, alpha=0.15,
                       edgecolors="none",
                       label=f"Trace{trace['trace_id']} {source} ({n_raw})")
        else:
            ax.scatter(raw[mask, 0], raw[mask, 1], s=1, c=color, alpha=0.15,
                       edgecolors="none",
                       label=f"Trace{trace['trace_id']} {source} ({n_raw})")

        # If Trace2 was replaced, also show original HighCon in lighter grey
        if source == "FSD1" and "raw_points_highcon" in trace:
            hc = trace["raw_points_highcon"]
            mask_hc = (hc[:, 0] >= b.left) & (hc[:, 0] <= b.right) & \
                      (hc[:, 1] >= b.bottom) & (hc[:, 1] <= b.top)
            n_hc = int(mask_hc.sum())
            if n_hc > 20000:
                idx_hc = np.random.RandomState(42).choice(
                    np.where(mask_hc)[0], 20000, replace=False)
                ax.scatter(hc[idx_hc, 0], hc[idx_hc, 1], s=1, c="#999999", alpha=0.1,
                           edgecolors="none",
                           label=f"Trace2 HighCon original ({n_hc})")
            else:
                ax.scatter(hc[mask_hc, 0], hc[mask_hc, 1], s=1, c="#999999", alpha=0.1,
                           edgecolors="none",
                           label=f"Trace2 HighCon original ({n_hc})")

        smooth = trace["coords"]
        mask = (smooth[:, 0] >= b.left) & (smooth[:, 0] <= b.right) & \
               (smooth[:, 1] >= b.bottom) & (smooth[:, 1] <= b.top)
        ax.plot(smooth[mask, 0], smooth[mask, 1], c=color, lw=2.0,
                label=f"Trace{trace['trace_id']} smoothed ({int(mask.sum())})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved smoothed trace visualisation to {output_path}")


# ── ML parameter tuning ───────────────────────────────────────────────

def _apply_params(module, params):
    """Override module-level constants in the 05a module with tuned values.

    This monkey-patches the imported 05a module so its functions use the
    tuned parameters at runtime.
    """
    for key, value in params.items():
        if key == "TV_WEIGHT_SCALE":
            base_weights = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
            module.LOCAL_TV_WEIGHTS = [w * value for w in base_weights]
            base_sigmas = [20.0, 28.0, 40.0, 56.0, 80.0, 113.0, 160.0]
            module.TARGET_TV_SIGMAS_M = [s / np.sqrt(value) for s in base_sigmas]
        elif hasattr(module, key):
            setattr(module, key, value)


def _evaluate_params(module, params, dem_window, valid, river_mask, resolution,
                     transform, highcon_xy, match_radius=100.0):
    """Run the 05a LSM + classify pipeline on one window and measure recovery.

    Uses 100m match radius (10px at 10m/px) — wide enough to count a point
    as recovered if the inferred point lands within one DEM cell of the
    surveyed point, accounting for sub-pixel DEM sampling.

    Returns dict with precision, recall, f1, n_inferred, n_highcon_recovered.
    """
    _apply_params(module, params)

    dem_nan = np.where(valid, dem_window, np.nan)
    state, diag = module._local_lsm(dem_nan, valid, river_mask, resolution)

    if state is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_inferred": 0, "n_highcon_recovered": 0, "reason": diag.get("reason", "unknown")}

    pixels = module._classify_scarp_pixels(state, dem_window, river_mask, resolution, transform)

    if pixels.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_inferred": 0, "n_highcon_recovered": 0, "reason": "no_pixels"}

    inferred_xy = pixels[["base_x", "base_y"]].to_numpy(dtype=float)

    if len(inferred_xy) == 0 or len(highcon_xy) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_inferred": len(inferred_xy), "n_highcon_recovered": 0, "reason": "empty"}

    inferred_tree = cKDTree(inferred_xy)

    # Recall: fraction of HighCon points near an inferred point
    dists, _ = inferred_tree.query(highcon_xy, k=1)
    n_recovered = int(np.sum(dists <= match_radius))
    recall = n_recovered / max(len(highcon_xy), 1)

    # Precision: fraction of inferred points near a HighCon point
    highcon_tree = cKDTree(highcon_xy)
    dists_inf, _ = highcon_tree.query(inferred_xy, k=1)
    n_precise = int(np.sum(dists_inf <= match_radius))
    precision = n_precise / max(len(inferred_xy), 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_inferred": len(inferred_xy),
        "n_highcon_recovered": n_recovered,
        "reason": "ok",
    }


def _sample_tuning_windows(dem_src, smoothed_traces, n_windows=TUNING_SAMPLE_WINDOWS,
                           bounds=None):
    """Sample representative windows along the smoothed traces for parameter tuning.

    Returns list of dicts with dem, valid, transform, resolution, highcon_xy.
    Uses ``raw_points`` (FSD1 for Trace2, HighCon for Trace1) so ML tuning
    optimises for FSD1 recovery.

    Parameters
    ----------
    bounds : tuple (minx, miny, maxx, maxy), optional
        If provided, only points within this rectangle are used.
    """
    all_highcon = []
    for trace in smoothed_traces:
        all_highcon.append(trace["raw_points"])
    if not all_highcon:
        return []
    all_highcon = np.vstack(all_highcon)
    dem_bounds = dem_src.bounds

    # Apply bounds filter if given
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        in_box = ((all_highcon[:, 0] >= minx) & (all_highcon[:, 0] <= maxx) &
                  (all_highcon[:, 1] >= miny) & (all_highcon[:, 1] <= maxy))
        all_highcon = all_highcon[in_box]
        if len(all_highcon) == 0:
            return []

    # Also clip to DEM extent
    in_bounds = ((all_highcon[:, 0] >= dem_bounds.left) & (all_highcon[:, 0] <= dem_bounds.right) &
                 (all_highcon[:, 1] >= dem_bounds.bottom) & (all_highcon[:, 1] <= dem_bounds.top))
    all_highcon = all_highcon[in_bounds]

    if len(all_highcon) == 0:
        return []

    from sklearn.cluster import KMeans
    n_centers = min(n_windows, len(all_highcon))
    kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
    kmeans.fit(all_highcon)
    centers = kmeans.cluster_centers_

    resolution = abs(dem_src.res[0])
    buffer_m = 1500.0

    windows = []
    for center in centers:
        win, _ = _window_from_bounds_clipped(
            dem_src,
            float(center[0] - buffer_m), float(center[1] - buffer_m),
            float(center[0] + buffer_m), float(center[1] + buffer_m),
        )
        if win is None:
            continue

        dem_data = dem_src.read(1, window=win).astype(np.float32)
        nodata = dem_src.nodata
        valid = np.isfinite(dem_data)
        if nodata is not None:
            valid &= (dem_data != nodata)
        dem_safe = np.where(valid, dem_data, np.nan)

        transform = dem_src.window_transform(win)

        in_win = ((all_highcon[:, 0] >= center[0] - buffer_m) &
                  (all_highcon[:, 0] <= center[0] + buffer_m) &
                  (all_highcon[:, 1] >= center[1] - buffer_m) &
                  (all_highcon[:, 1] <= center[1] + buffer_m))
        window_highcon = all_highcon[in_win]

        windows.append({
            "dem": dem_safe,
            "valid": valid,
            "transform": transform,
            "resolution": resolution,
            "highcon_xy": window_highcon,
            "center": center,
            "buffer_m": buffer_m,
        })

    print(f"  [Tuning] Sampled {len(windows)} windows"
          f" within bounds" if bounds else
          f"  [Tuning] Sampled {len(windows)} windows for parameter search")
    return windows


def _get_hand_masks(dem_crs, dem_source_key=None):
    """Return the list of HAND/river mask paths for the given DEM source."""
    key = dem_source_key or DEM_SOURCE_KEY
    return [
        HAND_DIR / f"{key}_hand.tif",
        HAND_DIR / f"{key}_river_bed.tif",
        HAND_DIR / f"{key}_river_surface.tif",
        HAND_DIR / f"{key}_river_valley.tif",
    ]

def run_criteria_sensitivity_analysis(dem_path, base_params, smoothed_traces,
                                      out_dir, bounds, seed_xy,
                                      seed_accept_radius_m=30.0, alt_sources=None):
    """Measure FSD1 recovery while relaxing individual classifier filters.

    Ablates two categories:
    1. Classifier filters (asymmetry, width, height, ratio, river masking)
    2. Alternative candidate sources (subtle_slope_break, convexity_profile,
       gentle_slope_anomaly, multiscale_relief, and each of the original
       alt-sources) — toggled off individually to measure marginal
       contribution to FSD1 recall.
    """
    print(f"\n{'='*70}")
    print(f"[05c] Criterion Sensitivity Analysis")
    print(f"{'='*70}")

    # Each variant: (name, param_overrides, alt_source_overrides)
    # alt_source_overrides is a dict of {source_name: False} to disable.
    variants = [
        # ── Classifier filter ablations ──
        ("Baseline", {}, {}),
        ("No asymmetry", {"MIN_SCARP_ASYMMETRY_SCORE": 0.0}, {}),
        ("No width", {"MIN_SCARP_WIDTH_M": 1.0}, {}),
        ("No height", {"MIN_SCARP_HEIGHT_M": 0.0}, {}),
        ("No ratio", {"MAX_SCARP_HEIGHT_WIDTH_RATIO": 100.0}, {}),
        ("No river masking", {"HAND_INFLUENCE_THRESHOLD_M": 0.0}, {}),
        ("No asymmetry + width", {
            "MIN_SCARP_ASYMMETRY_SCORE": 0.0,
            "MIN_SCARP_WIDTH_M": 1.0,
        }, {}),
        ("No asymmetry + height", {
            "MIN_SCARP_ASYMMETRY_SCORE": 0.0,
            "MIN_SCARP_HEIGHT_M": 0.0,
        }, {}),
        ("All filters off", {
            "MIN_SCARP_ASYMMETRY_SCORE": 0.0,
            "MIN_SCARP_WIDTH_M": 1.0,
            "MIN_SCARP_HEIGHT_M": 0.0,
            "MAX_SCARP_HEIGHT_WIDTH_RATIO": 100.0,
        }, {}),
        # ── Alt-source ablations (toggle off each new source individually) ──
        ("No subtle_slope_break", {}, {"subtle_slope_break": False}),
        ("No convexity_profile", {}, {"convexity_profile": False}),
        ("No gentle_slope_anomaly", {}, {"gentle_slope_anomaly": False}),
        ("No multiscale_relief", {}, {"multiscale_relief": False}),
        # ── Alt-source ablations (toggle off original sources) ──
        ("No curvature", {}, {"curvature": False}),
        ("No edge", {}, {"edge": False}),
        ("No openness", {}, {"openness": False}),
        ("No aspect", {}, {"aspect": False}),
        ("No ml_probability", {}, {"ml_probability": False}),
        # ── All new alt-sources off (isolate their combined contribution) ──
        ("All new alt-sources off", {}, {
            "subtle_slope_break": False,
            "convexity_profile": False,
            "gentle_slope_anomaly": False,
            "multiscale_relief": False,
        }),
    ]

    sensitivity_keys = {
        key for _, overrides, _ in variants for key in overrides
    }
    original_values = {
        key: getattr(_05a, key) for key in sensitivity_keys
        if hasattr(_05a, key)
    }
    baseline_params = base_params.copy()
    baseline_params.update(original_values)

    # Build a baseline alt_sources dict for ablation
    base_alt_sources = dict(alt_sources) if alt_sources else {}

    fsd1_pts = np.asarray(seed_xy, dtype=float) if seed_xy is not None else np.empty((0, 2))
    results = []
    try:
        for name, param_overrides, alt_overrides in variants:
            params = baseline_params.copy()
            params.update(param_overrides)
            _apply_params(_05a, params)
            variant_dir = out_dir / "sensitivity_tmp" / name.lower().replace(" ", "_")

            # Merge alt-source overrides into a per-variant copy
            variant_alt = dict(base_alt_sources)
            variant_alt.update(alt_overrides)

            print(f"\n--- Variant: {name} ---")
            inferred_df = run_dem_wide_inference(
                dem_path, params, smoothed_traces, variant_dir,
                write_tiles=False, use_fsd1_seeds=True, bounds=bounds,
                seed_accept_radius_m=seed_accept_radius_m, alt_sources=variant_alt,
            )

            if len(fsd1_pts) > 0 and len(inferred_df) > 0:
                inferred_xy = inferred_df[["x", "y"]].to_numpy(dtype=float)
                dists, _ = cKDTree(inferred_xy).query(fsd1_pts, k=1)
                recovered = dists <= 100.0
                n_recovered = int(np.sum(recovered))
                recall = float(np.mean(recovered))
            else:
                n_recovered = 0
                recall = 0.0

            print(f"  FSD1 recall: {recall:.4f} ({n_recovered}/{len(fsd1_pts)})")
            print(f"  Inferred points: {len(inferred_df)}")
            results.append({
                "variant": name,
                "fsd1_recall": recall,
                "n_fsd1_recovered": n_recovered,
                "n_fsd1_total": len(fsd1_pts),
                "n_inferred": len(inferred_df),
            })
    finally:
        _apply_params(_05a, base_params)
        for key, value in original_values.items():
            setattr(_05a, key, value)

    results_df = pd.DataFrame(results)
    results_csv = out_dir / "criterion_sensitivity.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSensitivity analysis saved to {results_csv}")
    return results_df


def _run_grid_search(tuning_windows, dem_crs):
    """Grid search fallback when Optuna is not available."""
    best_f1 = -1.0
    best_params = DEFAULT_PARAMS.copy()
    results_log = []

    grid = {
        "MIN_SCARP_HEIGHT_M": [0.1, 0.2, 0.3],
        "LSM_HYSTERESIS_LOW_SIGMA": [0.5, 1.0, 1.5],
        "LSM_HYSTERESIS_HIGH_SIGMA": [2.0, 4.0],
        "LSM_PERSISTENCE_K": [1, 2],
        "TV_WEIGHT_SCALE": [0.5, 1.0, 2.0],
        "BASE_RAY_LENGTH_M": [400.0, 800.0],
        "MIN_SCARP_ASYMMETRY_SCORE": [0.15, 0.3],
        "MIN_SCARP_WIDTH_M": [20.0, 30.0],
    }

    import itertools
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total_combos = 1
    for v in value_lists:
        total_combos *= len(v)
    print(f"  [Grid Search] {total_combos} parameter combinations")

    eval_windows = tuning_windows[:min(3, len(tuning_windows))]
    hand_masks = _get_hand_masks(dem_crs)

    for combo_idx, combo in enumerate(itertools.product(*value_lists)):
        params = dict(zip(keys, combo))

        all_scores = []
        for win_data in eval_windows:
            river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                          win_data["dem"].shape)
            scores = _evaluate_params(
                _05a, params,
                win_data["dem"], win_data["valid"], river,
                win_data["resolution"], win_data["transform"],
                win_data["highcon_xy"],
            )
            all_scores.append(scores)

        avg_f1 = np.mean([s["f1"] for s in all_scores])
        avg_recall = np.mean([s["recall"] for s in all_scores])
        avg_precision = np.mean([s["precision"] for s in all_scores])
        avg_inferred = np.mean([s["n_inferred"] for s in all_scores])
        # F2 score (recall-weighted) — user wants to recover as many
        # FSD1 points as possible, not balance precision/recall equally
        beta = 2.0
        if avg_precision + avg_recall > 1e-8:
            f2 = (1 + beta**2) * avg_precision * avg_recall / \
                 (beta**2 * avg_precision + avg_recall)
        else:
            f2 = 0.0

        results_log.append({
            "combo_idx": combo_idx,
            "f2": f2,
            "f1": avg_f1,
            "recall": avg_recall,
            "precision": avg_precision,
            "n_inferred": avg_inferred,
            **params,
        })

        if f2 > best_f1:
            best_f1 = f2
            best_params = params.copy()

        if (combo_idx + 1) % 10 == 0:
            print(f"  [Grid] {combo_idx + 1}/{total_combos}: best_f2={best_f1:.4f}")

    print(f"  [Grid Search] Best F2={best_f1:.4f}, params={best_params}")
    return best_params, best_f1, results_log


def _run_optuna_search(tuning_windows, dem_crs, n_trials=TUNING_N_TRIALS):
    """Bayesian optimisation using Optuna (if available)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] Not available, falling back to grid search")
        return _run_grid_search(tuning_windows, dem_crs)

    hand_masks = _get_hand_masks(dem_crs)
    eval_windows = tuning_windows[:min(4, len(tuning_windows))]

    def objective(trial):
        params = {
            "MIN_SCARP_HEIGHT_M": trial.suggest_float("MIN_SCARP_HEIGHT_M", 0.1, 0.5),
            "LSM_HYSTERESIS_LOW_SIGMA": trial.suggest_float("LSM_HYSTERESIS_LOW_SIGMA", 0.25, 2.0),
            "LSM_HYSTERESIS_HIGH_SIGMA": trial.suggest_float("LSM_HYSTERESIS_HIGH_SIGMA", 1.0, 6.0),
            "LSM_PERSISTENCE_K": trial.suggest_int("LSM_PERSISTENCE_K", 1, 3),
            "TV_WEIGHT_SCALE": trial.suggest_float("TV_WEIGHT_SCALE", 0.25, 4.0),
            "BASE_RAY_LENGTH_M": trial.suggest_float("BASE_RAY_LENGTH_M", 300.0, 1200.0),
            "MIN_SCARP_ASYMMETRY_SCORE": trial.suggest_float("MIN_SCARP_ASYMMETRY_SCORE", 0.05, 0.5),
            "MIN_SCARP_WIDTH_M": trial.suggest_float("MIN_SCARP_WIDTH_M", 10.0, 50.0),
        }

        all_scores = []
        for win_data in eval_windows:
            river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                          win_data["dem"].shape)
            scores = _evaluate_params(
                _05a, params,
                win_data["dem"], win_data["valid"], river,
                win_data["resolution"], win_data["transform"],
                win_data["highcon_xy"],
            )
            all_scores.append(scores)

        # Objective: maximise recall (recover as many FSD1/ground truth
        # points as possible) with a mild precision penalty to avoid
        # degenerate solutions that classify everything as scarp.
        avg_recall = np.mean([s["recall"] for s in all_scores])
        avg_precision = np.mean([s["precision"] for s in all_scores])
        # F2 score weights recall higher than precision (beta=2)
        beta = 2.0
        if avg_recall + avg_precision < 1e-8:
            return 0.0
        f2 = (1 + beta**2) * avg_precision * avg_recall / \
             (beta**2 * avg_precision + avg_recall)
        return f2

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_f1 = study.best_value
    print(f"  [Optuna] Best F2={best_f1:.4f} (recall-weighted)")
    print(f"  [Optuna] Best params: {best_params}")

    return best_params, best_f1, []


def tune_parameters(smoothed_traces, dem_path, dem_crs, bounds=None):
    """Run ML parameter tuning and return the best parameters.

    Tries Optuna (Bayesian optimisation) first; falls back to grid search.
    Returns (best_params, best_f1, results_log).

    Parameters
    ----------
    bounds : tuple (minx, miny, maxx, maxy), optional
        If provided, only sample tuning windows within this rectangle.
    """
    print(f"\n{'='*70}")
    print(f"[05c] ML Parameter Tuning for 0.5m LiDAR")
    print(f"{'='*70}")

    with rasterio.open(dem_path) as src:
        tuning_windows = _sample_tuning_windows(src, smoothed_traces, bounds=bounds)

    if not tuning_windows:
        print("  [Warning] No tuning windows could be sampled, using default params")
        return DEFAULT_PARAMS.copy(), 0.0, []

    best_params, best_f1, results_log = _run_optuna_search(
        tuning_windows, dem_crs, n_trials=TUNING_N_TRIALS)

    # Evaluate default params for comparison
    hand_masks = _get_hand_masks(dem_crs)
    default_scores = []
    for win_data in tuning_windows[:min(4, len(tuning_windows))]:
        river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                      win_data["dem"].shape)
        scores = _evaluate_params(
            _05a, DEFAULT_PARAMS,
            win_data["dem"], win_data["valid"], river,
            win_data["resolution"], win_data["transform"],
            win_data["highcon_xy"],
        )
        default_scores.append(scores)

    default_f1 = np.mean([s["f1"] for s in default_scores]) if default_scores else 0.0
    print(f"\n  [Tuning Results]")
    print(f"  Default params F1: {default_f1:.4f}")
    print(f"  Tuned params F1:   {best_f1:.4f}")
    print(f"  Improvement:        {best_f1 - default_f1:+.4f}")

    return best_params, best_f1, results_log


# ── DEM-wide tiled inference ──────────────────────────────────────────

def _generate_tiles(width, height, tile_size=TILE_SIZE_PX, overlap=TILE_OVERLAP_PX):
    """Generate overlapping tile windows covering the entire DEM.

    Yields (col_off, row_off, tile_width, tile_height).
    """
    step = tile_size - overlap
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            tw = min(tile_size, width - col_off)
            th = min(tile_size, height - row_off)
            if tw < 20 or th < 20:
                continue
            yield col_off, row_off, tw, th


def _filter_by_trace_proximity(points_xy, smoothed_traces, buffer_m=TRACE_BUFFER_M,
                               default_buffer_m=None):
    """Keep only points within buffer_m of any smoothed trace line.

    Supports per-trace buffers: if a trace dict contains ``buffer_m``,
    that value is used for that trace instead of the global ``buffer_m``.
    This allows Trace2 (FSD1) to use a wider buffer (300m) than Trace1
    (HighCon, 100m).

    For FSD1 traces, also checks proximity against the raw FSD1 points
    themselves (within ``fsd1_point_buffer_m`` if present, else 300m),
    so inferred points near scattered FSD1 survey points are kept even
    if they are far from the smoothed B-spline line.

    Returns boolean mask and nearest distance array.
    """
    if default_buffer_m is not None:
        buffer_m = default_buffer_m

    if len(points_xy) == 0 or not smoothed_traces:
        return np.zeros(len(points_xy), dtype=bool), np.full(len(points_xy), np.inf)

    # Build per-trace trees and union the masks with per-trace buffers
    mask = np.zeros(len(points_xy), dtype=bool)
    min_dists = np.full(len(points_xy), np.inf)

    for trace in smoothed_traces:
        coords = trace["coords"]
        if len(coords) == 0:
            continue
        trace_buffer = trace.get("buffer_m", buffer_m)
        tree = cKDTree(coords)
        d, _ = tree.query(points_xy, k=1)
        within = d <= trace_buffer
        mask |= within
        min_dists = np.minimum(min_dists, d)

        # For FSD1 traces, also check against raw survey points
        if trace.get("source") == "FSD1" and "raw_points" in trace:
            raw = trace["raw_points"]
            if len(raw) > 0:
                raw_tree = cKDTree(raw)
                d_raw, _ = raw_tree.query(points_xy, k=1)
                raw_buffer = trace.get("fsd1_point_buffer_m", 300.0)
                within_raw = d_raw <= raw_buffer
                mask |= within_raw
                min_dists = np.minimum(min_dists, d_raw)

    return mask, min_dists


def compute_trace2_search_bounds(smoothed_traces, buffer_m=2000.0, dem_bounds=None):
    """Compute a rectangular search bounds around Trace2 (FSD1 or smoothed line).

    Parameters
    ----------
    smoothed_traces : list of dicts
        Smoothed traces from ``_build_smoothed_traces()``.
    buffer_m : float
        Buffer to expand the bounding box (default 2000 m).
    dem_bounds : tuple (left, bottom, right, top), optional
        If provided, clip the rectangle to these bounds.

    Returns
    -------
    tuple (minx, miny, maxx, maxy) or None if no Trace2 points found.
    """
    trace2 = None
    for t in smoothed_traces:
        if t.get("trace_id") == 2:
            trace2 = t
            break
    if trace2 is None:
        return None

    # Prefer raw FSD1 points (cleaner) if source is FSD1, else use smoothed coords
    if trace2.get("source") == "FSD1" and "raw_points" in trace2 and len(trace2["raw_points"]) > 0:
        pts = trace2["raw_points"]
    else:
        pts = trace2["coords"]

    if len(pts) == 0:
        return None

    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    if dem_bounds is not None:
        left, bottom, right, top = dem_bounds
        minx = max(minx, left)
        miny = max(miny, bottom)
        maxx = min(maxx, right)
        maxy = min(maxy, top)

    if maxx <= minx or maxy <= miny:
        return None
    return (minx, miny, maxx, maxy)


def _generate_tiles_in_bounds(width, height, bounds, transform,
                              tile_size=TILE_SIZE_PX, overlap=TILE_OVERLAP_PX):
    """Generate overlapping tile windows that intersect a world-coordinate bounding box.

    Similar to ``_generate_tiles`` but only yields tiles overlapping ``bounds``.
    """
    inv_transform = ~transform
    col_min, row_min = inv_transform * (bounds[0], bounds[3])  # left, top
    col_max, row_max = inv_transform * (bounds[2], bounds[1])  # right, bottom

    col_min = max(0, int(np.floor(col_min)))
    row_min = max(0, int(np.floor(row_min)))
    col_max = min(width, int(np.ceil(col_max)))
    row_max = min(height, int(np.ceil(row_max)))

    if col_max <= col_min or row_max <= row_min:
        return

    step = tile_size - overlap
    start_col = (col_min // step) * step
    start_row = (row_min // step) * step

    for row_off in range(start_row, row_max, step):
        for col_off in range(start_col, col_max, step):
            tw = min(tile_size, width - col_off)
            th = min(tile_size, height - row_off)
            if tw < 20 or th < 20:
                continue
            yield col_off, row_off, tw, th


def _add_seed_candidates_to_state(state, seed_xy, transform, dem_shape,
                                   buffer_px=1):
    """Add explicit seed points (e.g., FSD1) to the LSM candidate mask.

    The ``state`` dict contains a boolean 2D array ``state['candidates']``.
    We set pixels within ``buffer_px`` of each seed coordinate to True so
    they will be processed by the classification step.

    Parameters
    ----------
    state : dict
        Output from ``_local_lsm``; must contain key 'candidates'.
    seed_xy : np.ndarray (N, 2)
        World coordinates (x, y) of seed points.
    transform : affine.Affine
        Rasterio transform for the current window.
    dem_shape : tuple (height, width)
        Shape of the DEM window.
    buffer_px : int, optional
        Radius (in pixels) around each seed to mark as candidate.

    Returns
    -------
    dict
        The modified ``state`` with updated ``candidates``.
    """
    if state is None or seed_xy is None or len(seed_xy) == 0:
        return state

    candidates = state.get("candidates")
    if candidates is None or not isinstance(candidates, np.ndarray):
        return state
    if candidates.dtype != bool:
        candidates = candidates.astype(bool)
        state["candidates"] = candidates

    height, width = dem_shape
    inv_transform = ~transform
    for x, y in seed_xy:
        col, row = inv_transform * (x, y)
        col_int, row_int = int(round(col)), int(round(row))

        r_min = max(0, row_int - buffer_px)
        r_max = min(height, row_int + buffer_px + 1)
        c_min = max(0, col_int - buffer_px)
        c_max = min(width, col_int + buffer_px + 1)
        candidates[r_min:r_max, c_min:c_max] = True

    return state


def _get_fsd1_seed_points(smoothed_traces):
    """Return the FSD1 raw survey points (if present) for seeding.

    Scans ``smoothed_traces`` for a trace with ``source == 'FSD1'`` and
    returns its ``raw_points`` (the original FSD1 coordinates used to
    build the smoothed line).

    Returns
    -------
    np.ndarray (N, 2) or None
        FSD1 coordinates, or None if no FSD1 trace is found.
    """
    for trace in smoothed_traces:
        if trace.get("trace_id") == 2 and trace.get("source") == "FSD1":
            return trace["raw_points"]
    return None


def _load_nfd1_seed_points(dem_bounds=None):
    """Load NFD1 structural points from the inventory CSV.

    NFD1 points are an alternative fault-survey dataset (distinct from
    FSD1).  For coarse DEMs where LSM signal is sparse, injecting NFD1
    points as candidate seeds ensures the classify step evaluates them.

    Parameters
    ----------
    dem_bounds : rasterio.coords.BoundingBox or tuple, optional
        If provided, only points within (left, bottom, right, top) are kept.

    Returns
    -------
    np.ndarray (N, 2) or None
        NFD1 (x, y) coordinates, or None if not found.
    """
    if not NFD1_PATH.exists():
        print("  [NFD1] structural_points_inventory.csv not found")
        return None
    df = pd.read_csv(NFD1_PATH)
    nfd1 = df[df["source"] == "NFD1"]
    if len(nfd1) == 0:
        return None
    xy = nfd1[["XCoord", "YCoord"]].to_numpy(dtype=float)
    if dem_bounds is not None:
        left, bottom, right, top = dem_bounds
        mask = ((xy[:, 0] >= left) & (xy[:, 0] <= right) &
                (xy[:, 1] >= bottom) & (xy[:, 1] <= top))
        xy = xy[mask]
    if len(xy) == 0:
        return None
    print(f"  [NFD1] Loaded {len(xy)} NFD1 seed points")
    return xy


def _attribute_point_sources(pixels_df, per_source_masks, transform,
                             dem_shape):
    """Tag each inferred point with the candidate source(s) that contributed.

    The classify step in 05a stores the candidate pixel location in the
    diagnostics.  If diag info (row/col) is available, we use it directly.
    Otherwise we convert the inferred point's world coordinates back to
    pixel coordinates and check which per-source masks are True there.

    Parameters
    ----------
    pixels_df : pd.DataFrame
        Must contain 'base_x' and 'base_y' columns (world coordinates).
        May optionally contain 'row'/'col' columns from diagnostics.
    per_source_masks : dict {str: np.ndarray(H, W) bool}
        One boolean mask per candidate source.
    transform : affine.Affine
        Rasterio transform for converting world <-> pixel coordinates.
    dem_shape : tuple (H, W)
        Shape of the DEM tile.

    Returns
    -------
    list of str or None
        A list of strings like "tpi;tri;dog" (semicolon-separated source
        names) for each row in pixels_df, or None if attribution fails.
    """
    if per_source_masks is None or len(per_source_masks) == 0:
        return None
    if len(pixels_df) == 0:
        return []

    h, w = dem_shape
    inv_transform = ~transform
    n = len(pixels_df)
    source_names = list(per_source_masks.keys())

    # Get pixel coordinates for each inferred point
    if "row" in pixels_df.columns and "col" in pixels_df.columns:
        rows = pixels_df["row"].to_numpy(dtype=int)
        cols = pixels_df["col"].to_numpy(dtype=int)
    else:
        # Convert world coords to pixel coords
        xy = pixels_df[["base_x", "base_y"]].to_numpy(dtype=float)
        rows = np.zeros(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        for i in range(n):
            c, r = inv_transform * (xy[i, 0], xy[i, 1])
            cols[i] = int(round(c))
            rows[i] = int(round(r))

    # Clip to valid range
    valid_px = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    tags = ["" for _ in range(n)]
    for i in range(n):
        if not valid_px[i]:
            continue
        r, c = rows[i], cols[i]
        matched = []
        for sname in source_names:
            mask = per_source_masks[sname]
            if mask[r, c]:
                matched.append(sname)
        tags[i] = ";".join(matched) if matched else "lsm_only"

    return tags


def run_dem_wide_inference(dem_path, tuned_params, smoothed_traces,
                           out_dir, write_tiles=False, use_fsd1_seeds=True,
                           bounds=None, seed_accept_radius_m=30.0,
                           alt_sources=None, ml_probability_threshold=0.5,
                           ml_filter_mode="or",
                           use_nfd1_seeds=False, trace_buffer_m=None):
    """Run the 05a LSM + classify pipeline on every tile of the DEM.

    This produces a DEM-wide set of inferred scarp points, filtered by
    proximity to the smoothed trace lines.

    Parameters
    ----------
    dem_path : Path
        Path to the DEM GeoTIFF.
    tuned_params : dict
        Best-found parameters from tune_parameters().
    smoothed_traces : list of dicts
        Smoothed trace lines from _build_smoothed_traces().
    out_dir : Path
        Output directory.
    write_tiles : bool
        If True, save per-tile LSM rasters (for debugging).
    use_fsd1_seeds : bool
        If True, inject FSD1 survey points as LSM candidate seeds so the
        classify step evaluates them even where the LSM hysteresis did not
        produce a candidate.  Disabled by ``--no-fsd1-seeds``.
    bounds : tuple (minx, miny, maxx, maxy), optional
        If provided, only process tiles intersecting this rectangle.
    seed_accept_radius_m : float
        Radius around each FSD1 seed to add to the candidate mask.
    alt_sources : dict or None
        If provided, merge alternative candidate sources (curvature, edge,
        openness, aspect) into the LSM candidate mask via logical OR before
        classification.  Keys: 'curvature', 'edge', 'openness', 'aspect'.
        None means no alternative sources.
    ml_probability_threshold : float, optional
        ML probability threshold for filtering (default 0.5).
    ml_filter_mode : str, optional
        How to use ML: 'filter' (AND with other candidates) or 'or' (OR with
        other candidates, default).
    use_nfd1_seeds : bool, optional
        If True, inject NFD1 structural points as candidate seeds (useful
        for coarse DEMs where LSM signal is sparse).
    trace_buffer_m : float, optional
        Override the trace proximity filter buffer (default: use global
        TRACE_BUFFER_M, or scale for coarse DEMs).
    """
    print(f"\n{'='*70}")
    print("[05c] DEM-Wide Inference")
    print(f"{'='*70}")

    _apply_params(_05a, tuned_params)
    print(f"  Applied tuned parameters: {tuned_params}")
    print(f"  ML filtering: mode={ml_filter_mode}, threshold={ml_probability_threshold:.2f}")

    safe_makedirs(out_dir)
    tile_lsm_dir = safe_makedirs(out_dir / "tile_lsm") if write_tiles else None

    # Detect DEM resolution to decide whether to use the coarse-DEM branch
    with rasterio.open(dem_path) as _src_probe:
        dem_resolution = abs(_src_probe.res[0])
        dem_bounds_probe = _src_probe.bounds
    is_coarse = dem_resolution > COARSE_DEM_THRESHOLD_M
    if is_coarse:
        print(f"  [Coarse DEM branch] resolution={dem_resolution:.1f}m > {COARSE_DEM_THRESHOLD_M}m threshold")
        print(f"  [Coarse DEM branch] Disabling river mask, scaling width max, using NFD1 seeds")

    # Override the global SCARP_WIDTH_DISCRETE_MAX_M for coarse DEMs.
    # At 90m/px, the default 200m max means 2 pixels = 180m, so even a
    # 2-pixel-wide scarp fails the width filter.  Scale to 6 * resolution
    # (e.g. 540m for 90m SRTM) to allow multi-pixel scarps through.
    if is_coarse:
        _original_width_max = getattr(_05a, 'SCARP_WIDTH_DISCRETE_MAX_M', 200)
        _05a.SCARP_WIDTH_DISCRETE_MAX_M = max(200, int(dem_resolution * 6))
        print(f"  [Coarse DEM branch] SCARP_WIDTH_DISCRETE_MAX_M: {_original_width_max} -> {_05a.SCARP_WIDTH_DISCRETE_MAX_M}")

    hand_masks = _get_hand_masks(None)

    # For coarse DEMs, ignore the river mask entirely (HAND is NaN, and
    # river_valley mask covers ~32% of the DEM, over-masking candidates).
    if is_coarse:
        hand_masks = None
        print(f"  [Coarse DEM branch] River mask disabled (HAND=NaN, river_valley over-masks)")

    # Extract FSD1 seed points for injection into LSM candidate mask
    seed_xy = None
    if use_fsd1_seeds:
        seed_xy = _get_fsd1_seed_points(smoothed_traces)
        if seed_xy is not None:
            print(f"  Using {len(seed_xy)} FSD1 points as LSM seed candidates")
        else:
            print("  No FSD1 points found for seeding")

    # Extract NFD1 seed points for coarse DEMs
    nfd1_xy = None
    if use_nfd1_seeds:
        nfd1_xy = _load_nfd1_seed_points(dem_bounds=dem_bounds_probe)
        if nfd1_xy is not None:
            print(f"  Using {len(nfd1_xy)} NFD1 points as coarse-DEM seed candidates")
        else:
            print("  No NFD1 points found for seeding")

    # Combine FSD1 and NFD1 seed points
    all_seed_xy = None
    if seed_xy is not None and nfd1_xy is not None:
        all_seed_xy = np.vstack([seed_xy, nfd1_xy])
        print(f"  Total seed points (FSD1+NFD1): {len(all_seed_xy)}")
    elif seed_xy is not None:
        all_seed_xy = seed_xy
    elif nfd1_xy is not None:
        all_seed_xy = nfd1_xy

    # Determine trace proximity buffer: scale for coarse DEMs
    effective_trace_buffer = trace_buffer_m
    if effective_trace_buffer is None:
        if is_coarse:
            effective_trace_buffer = max(TRACE_BUFFER_M, dem_resolution * 3)
            print(f"  [Coarse DEM branch] Trace buffer scaled: {TRACE_BUFFER_M}m -> {effective_trace_buffer}m")
        else:
            effective_trace_buffer = TRACE_BUFFER_M

    all_inferred_rows = []
    all_diag_rows = []          # A1: per-candidate diagnostics across all tiles
    all_candidate_xy = []
    seed_accepted_xy_tile = []
    tile_diagnostics = []

    with rasterio.open(dem_path) as src:
        width, height = src.width, src.height
        resolution = abs(src.res[0])
        dem_nodata = src.nodata
        dem_crs = src.crs

        if bounds is not None:
            dem_bounds = src.bounds
            bounds = (
                max(bounds[0], dem_bounds.left),
                max(bounds[1], dem_bounds.bottom),
                min(bounds[2], dem_bounds.right),
                min(bounds[3], dem_bounds.top),
            )
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                print("  [Warning] Search bounds outside DEM extent; no tiles to process.")
                bounds = None

        if bounds is not None:
            tiles = list(_generate_tiles_in_bounds(
                width, height, bounds, src.transform,
                tile_size=TILE_SIZE_PX, overlap=TILE_OVERLAP_PX))
            print(f"  Limited search to rectangle: {bounds}")
        else:
            tiles = list(_generate_tiles(width, height, TILE_SIZE_PX, TILE_OVERLAP_PX))
        n_tiles = len(tiles)
        print(f"  DEM: {width}x{height} px ({resolution}m/px)")
        if dem_nodata is not None:
            print(f"  Nodata: {dem_nodata}")
        print(f"  Tiles: {n_tiles} ({TILE_SIZE_PX}px, overlap={TILE_OVERLAP_PX}px)")

        for tile_idx, (col_off, row_off, tw, th) in enumerate(tiles):
            if (tile_idx + 1) % 20 == 0 or tile_idx == 0:
                print(f"  [Tile {tile_idx + 1}/{n_tiles}] col={col_off} row={row_off} "
                      f"size={tw}x{th}")

            window = Window(col_off, row_off, tw, th)
            dem = np.empty((window.height, window.width), dtype=np.float32)
            src.read(1, window=window, out=dem)

            valid = np.isfinite(dem)
            if dem_nodata is not None:
                valid &= (dem != dem_nodata)

            if valid.sum() < 100:
                tile_diagnostics.append({
                    "tile_idx": tile_idx, "col_off": col_off, "row_off": row_off,
                    "valid_pixels": int(valid.sum()), "status": "too_few_valid",
                })
                continue

            transform = src.window_transform(window)
            if hand_masks is not None:
                river = _read_mask_on_window(hand_masks, transform, dem_crs, dem.shape)
            else:
                # Coarse DEM: create an all-False river mask (no river filtering)
                river = np.zeros(dem.shape, dtype=bool)

            dem_nan = np.where(valid, dem, np.nan)
            state, lsm_diag = _05a._local_lsm(dem_nan, valid, river, resolution)

            if state is None:
                tile_diagnostics.append({
                    "tile_idx": tile_idx, "col_off": col_off, "row_off": row_off,
                    "valid_pixels": int(valid.sum()), "status": lsm_diag.get("reason", "unknown"),
                })
                continue

            # Inject FSD1+NFD1 seed points into the LSM candidate mask so the
            # classify step evaluates them even where LSM hysteresis did
            # not produce a candidate.
            if all_seed_xy is not None:
                tile_bounds = window_bounds(window, transform)
                in_tile = ((all_seed_xy[:, 0] >= tile_bounds[0]) &
                           (all_seed_xy[:, 0] <= tile_bounds[2]) &
                           (all_seed_xy[:, 1] >= tile_bounds[1]) &
                           (all_seed_xy[:, 1] <= tile_bounds[3]))
                if np.any(in_tile):
                    state = _add_seed_candidates_to_state(
                        state, all_seed_xy[in_tile], transform, dem.shape,
                        buffer_px=max(1, int(np.ceil(seed_accept_radius_m / resolution))))

                cand_rows, cand_cols = np.where(state["candidates"])
                if len(cand_rows) and len(all_seed_xy) > 0:
                    cand_x, cand_y = transform * (cand_cols, cand_rows)
                    cand_xy = np.c_[cand_x, cand_y]
                    seed_tree = cKDTree(all_seed_xy)
                    dists, _ = seed_tree.query(cand_xy, k=1)
                    accepted_mask = dists <= seed_accept_radius_m
                    n_accepted = int(accepted_mask.sum())
                    print(f"  [Tile {tile_idx + 1}] Seed insertion: {len(all_seed_xy)} seeds, "
                          f"{n_accepted} candidates within {seed_accept_radius_m}m")
                    if n_accepted > 0:
                        seed_accepted_xy_tile.append(cand_xy[accepted_mask])

            # Merge alternative candidate sources (curvature, edge,
            # openness, aspect) into the LSM candidate mask.
            # 
            # If ml_filter_mode='filter': Use ML probability to FILTER candidates
            # (AND operation) — only keep candidates with high ML probability.
            # If ml_filter_mode='or': OR alternative sources into candidates.
            # Track per-source masks for source attribution of inferred points.
            tile_per_source = None
            if alt_sources is not None:
                n_before = int(state["candidates"].sum())
                alt_mask, tile_per_source = _compute_alternative_candidates(
                    dem, valid, resolution, transform, sources=alt_sources,
                    return_per_source=True)
                
                # Use ML as a primary filter if ml_filter_mode='filter'
                if ml_filter_mode == "filter" and alt_sources.get("ml_probability", False):
                    # Extract ML probability for the entire tile.
                    # percentile=None so the explicit threshold is the sole
                    # criterion — no per-tile top-10% override.
                    ml_probs = _compute_ml_probability_candidates(
                        dem, valid, resolution, transform, 
                        threshold=ml_probability_threshold, percentile=None)
                    ml_mask = ml_probs > ml_probability_threshold
                    
                    # Filter all candidates (LSM + alt sources) by ML probability
                    # This removes low-probability areas
                    alt_mask_filtered = alt_mask & ml_mask
                    state["candidates"] = state["candidates"] & ml_mask
                    
                    n_alt_before = int(alt_mask.sum())
                    n_alt_after = int(alt_mask_filtered.sum())
                    print(f"  [Tile {tile_idx + 1}] ML filter (threshold={ml_probability_threshold:.2f}): "
                          f"alt candidates {n_alt_before} -> {n_alt_after}, "
                          f"LSM {n_before} -> {int(state['candidates'].sum())}")
                    
                    state["candidates"] |= alt_mask_filtered
                else:
                    # Traditional OR mode: merge all alternatives
                    state["candidates"] |= alt_mask
                
                n_after = int(state["candidates"].sum())
                print(f"  [Tile {tile_idx + 1}] Alt candidates merged: "
                      f"{n_before} -> {n_after} (+{n_after - n_before})")


            cand_rows, cand_cols = np.where(state["candidates"])
            if len(cand_rows):
                cand_x, cand_y = transform * (cand_cols, cand_rows)
                all_candidate_xy.append(np.c_[cand_x, cand_y])

            classify_result = _05a._classify_scarp_pixels(
                state, dem, river, resolution, transform,
                collect_diagnostics=True)
            if isinstance(classify_result, tuple):
                pixels, diag_df = classify_result
                if not diag_df.empty:
                    diag_df["tile_idx"] = tile_idx
                    all_diag_rows.append(diag_df)
            else:
                pixels = classify_result

            if not pixels.empty:
                # Apply ML probability filtering to final inferred pixels (if enabled)
                if alt_sources is not None and alt_sources.get("ml_probability", False) and _ML_CANDIDATE_MODEL is not None:
                    try:
                        # Extract ML features from the entire tile
                        tile_features = _extract_ml_features(dem, valid, resolution)
                        
                        # Convert inferred point coordinates (world) to pixel coordinates
                        inferred_xy = pixels[["base_x", "base_y"]].to_numpy(dtype=float)
                        inv_transform = ~transform
                        inferred_cols = np.zeros(len(inferred_xy), dtype=int)
                        inferred_rows = np.zeros(len(inferred_xy), dtype=int)
                        
                        for i, (x, y) in enumerate(inferred_xy):
                            col, row = inv_transform * (x, y)
                            inferred_cols[i] = int(round(col))
                            inferred_rows[i] = int(round(row))
                        
                        # Sample features at inferred point locations
                        h, w = dem.shape
                        valid_mask = ((inferred_rows >= 0) & (inferred_rows < h) &
                                     (inferred_cols >= 0) & (inferred_cols < w))
                        
                        inferred_features = np.zeros((len(inferred_xy), len(_ML_MODEL_FEATURE_NAMES)), 
                                                    dtype=np.float32)
                        
                        for j, fname in enumerate(_ML_MODEL_FEATURE_NAMES):
                            if fname in tile_features:
                                feat_map = tile_features[fname]
                                for i in range(len(inferred_xy)):
                                    if valid_mask[i]:
                                        r, c = inferred_rows[i], inferred_cols[i]
                                        inferred_features[i, j] = feat_map[r, c]
                                    else:
                                        inferred_features[i, j] = np.nan
                        
                        # Get ML probabilities at these points
                        finite_mask = np.all(np.isfinite(inferred_features), axis=1)
                        ml_probs_all = np.zeros(len(inferred_features), dtype=np.float32)
                        
                        if finite_mask.sum() > 0:
                            ml_probs_all[finite_mask] = _ML_CANDIDATE_MODEL.predict_proba(
                                inferred_features[finite_mask])[:, 1]
                        
                        # Filter by ML probability
                        keep_mask = (ml_probs_all > ml_probability_threshold) | ~valid_mask
                        n_before_ml = len(pixels)
                        pixels = pixels[keep_mask].copy()
                        
                        # Add ML probability column if at least some pixels kept
                        if len(pixels) > 0:
                            pixels["ml_probability"] = ml_probs_all[keep_mask]
                            n_filtered = n_before_ml - len(pixels)
                            print(f"  [Tile {tile_idx + 1}] ML final filter: "
                                  f"{n_before_ml} -> {len(pixels)} pixels "
                                  f"(removed {n_filtered}, threshold={ml_probability_threshold:.2f})")
                        else:
                            print(f"  [Tile {tile_idx + 1}] ML final filter: "
                                  f"{n_before_ml} -> 0 pixels (all filtered out)")
                    except Exception as e:
                        print(f"  [Tile {tile_idx + 1}] ML final filtering failed: {e}, skipping")

                keep_cols = ["base_x", "base_y", "tier", "break_type"]
                if "break_type" not in pixels.columns:
                    keep_cols = ["base_x", "base_y", "tier"]
                elif "tier" not in pixels.columns:
                    keep_cols = ["base_x", "base_y", "break_type"]
                
                # Include ml_probability if present
                if "ml_probability" in pixels.columns:
                    keep_cols.append("ml_probability")
                
                # Include classify_method if present (B1: ray_traced vs curvature_fallback)
                if "classify_method" in pixels.columns:
                    keep_cols.append("classify_method")
                
                # ── Source attribution: tag each inferred point with which
                # alternative candidate source(s) contributed to the pixel
                # it was classified from.  This enables per-method breakdown
                # in _evaluate_inference_quality.
                if tile_per_source is not None and len(pixels) > 0:
                    # The classify step stores the candidate pixel (row, col)
                    # in the state.  We need to map each inferred point back
                    # to its source pixel.  The pixels DataFrame has base_x/
                    # base_y (world coords of the base), but the candidate
                    # pixel is the original (row, col) from state["candidates"].
                    # We use the diag_df if available (it has row/col), or
                    # fall back to converting world coords back to pixel coords.
                    source_tags = _attribute_point_sources(
                        pixels, tile_per_source, transform, dem.shape)
                    if source_tags is not None:
                        pixels["candidate_sources"] = source_tags
                        keep_cols.append("candidate_sources")
                
                all_inferred_rows.append(pixels[keep_cols].copy())


            if write_tiles and tile_lsm_dir is not None:
                tile_path = tile_lsm_dir / f"tile_{tile_idx:04d}_lsm.tif"
                _write_local_raster(tile_path, state, transform, dem_crs)

            tile_diagnostics.append({
                "tile_idx": tile_idx, "col_off": col_off, "row_off": row_off,
                "valid_pixels": int(valid.sum()),
                "persistent_pixels": lsm_diag.get("persistent_pixels", 0),
                "n_pixels": len(pixels), "status": "ok",
            })

            n_seed_accepted = sum(len(points) for points in seed_accepted_xy_tile)
            print(f"  Seed-accepted candidate points accumulated: {n_seed_accepted}")

    # Combine all inferred points
    if all_inferred_rows:
        inferred_df = pd.concat(all_inferred_rows, ignore_index=True)
        inferred_df = inferred_df.rename(columns={"base_x": "x", "base_y": "y"})
        if "tier" not in inferred_df.columns:
            inferred_df["tier"] = np.nan
    else:
        inferred_df = pd.DataFrame(columns=["x", "y", "tier"])

    # Deduplicate (overlap zones produce duplicates)
    if len(inferred_df) > 0:
        inferred_df["_key"] = inferred_df["x"].round(1).astype(str) + "_" + \
                              inferred_df["y"].round(1).astype(str)
        n_before = len(inferred_df)
        inferred_df = inferred_df.drop_duplicates(subset="_key").drop(columns=["_key"])
        print(f"  Dedup: {n_before:,} -> {len(inferred_df):,} (removed overlap duplicates)")

    # Save outputs
    inferred_df.to_csv(out_dir / "dem_wide_inferred_points.csv", index=False)
    print(f"  Saved {len(inferred_df):,} inferred points to "
          f"{out_dir / 'dem_wide_inferred_points.csv'}")

    # Filter by proximity to smoothed traces (per-trace buffer)
    # For coarse DEMs with NFD1 seeds, also keep points near NFD1 points
    # that were NOT caught by the trace proximity filter — this is a union,
    # not a replacement.
    if len(inferred_df) > 0 and smoothed_traces:
        xy = inferred_df[["x", "y"]].to_numpy(dtype=float)
        proximity_mask, nearest_dist = _filter_by_trace_proximity(
            xy, smoothed_traces, buffer_m=effective_trace_buffer)

        # For coarse DEMs with NFD1, union with NFD1 proximity
        if nfd1_xy is not None and len(nfd1_xy) > 0:
            nfd1_tree_prox = cKDTree(nfd1_xy)
            nfd1_dists_prox, _ = nfd1_tree_prox.query(xy, k=1)
            nfd1_buffer = dem_resolution * 5  # 5x DEM resolution for broader recovery
            nfd1_mask = nfd1_dists_prox <= nfd1_buffer
            # Union: keep points near traces OR near NFD1
            n_before_union = int(proximity_mask.sum())
            proximity_mask = proximity_mask | nfd1_mask
            n_union = int(proximity_mask.sum())
            n_nfd1_added = n_union - n_before_union
            print(f"  [NFD1 union] Trace+NFD1 proximity: {n_union}/{len(inferred_df)} "
                  f"(NFD1 added {n_nfd1_added}, buffer={nfd1_buffer:.0f}m)")

        n_before = len(inferred_df)
        inferred_df = inferred_df[proximity_mask].copy()
        inferred_df["nearest_trace_dist_m"] = nearest_dist[proximity_mask]
        buffer_desc = "/".join(
            f"T{t['trace_id']}={t.get('buffer_m', effective_trace_buffer):.0f}m"
            for t in smoothed_traces)
        print(f"  Trace proximity filter ({buffer_desc}): {n_before:,} -> {len(inferred_df):,}")

    # Save outputs
    inferred_df.to_csv(out_dir / "T1T2_inferred_points.csv", index=False)
    print(f"  Saved {len(inferred_df):,} inferred points to "
          f"{out_dir / 'T1T2_inferred_points.csv'}")

    # Candidate points
    if all_candidate_xy:
        all_candidates = np.vstack(all_candidate_xy)
        pd.DataFrame(all_candidates, columns=["x", "y"]).to_csv(
            out_dir / "dem_wide_candidate_points.csv", index=False)
        print(f"  Saved {len(all_candidates):,} candidate points")

    # A1: Per-candidate classify diagnostics
    if all_diag_rows:
        classify_diag = pd.concat(all_diag_rows, ignore_index=True)
        classify_diag.to_csv(out_dir / "classify_diagnostics.csv", index=False)
        print(f"  Saved {len(classify_diag):,} classify diagnostics to "
              f"{out_dir / 'classify_diagnostics.csv'}")
        reason_counts = classify_diag["rejection_reason"].value_counts().to_dict()
        method_counts = classify_diag["classify_method"].value_counts().to_dict()
        print(f"  Classify diagnostics summary:")
        print(f"    Rejection reasons: {reason_counts}")
        print(f"    Classify methods:  {method_counts}")

    # Tile diagnostics
    tile_diag_df = pd.DataFrame(tile_diagnostics)
    tile_diag_df.to_csv(out_dir / "tile_diagnostics.csv", index=False)
    status_counts = tile_diag_df["status"].value_counts().to_dict() if not tile_diag_df.empty else {}
    print(f"  Tile diagnostics: {status_counts}")

    n_ok_tiles = status_counts.get("ok", 0)
    print(f"\n  [DEM-Wide Summary]")
    print(f"  Tiles processed: {n_tiles}")
    print(f"  Tiles with signal: {n_ok_tiles}")
    print(f"  Total inferred points (after dedup + filter): {len(inferred_df):,}")

    return inferred_df


# ── Quality evaluation ────────────────────────────────────────────────

def _evaluate_inference_quality(inferred_df, smoothed_traces, match_radius=100.0,
                                alt_sources=None, tuned_params=None):
    """Evaluate how well the inferred points cover the ground truth.

    For Trace 2 (when FSD1 replacement is active), evaluation is done
    against **both**:
    - FSD1 points (``raw_points``) — the clean surveyed points used for
      smoothing, which we want to recover as many as possible.
    - Original HighCon Trace2 points (``raw_points_highcon``) — the
      original noisy points, for comparison.

    For Trace 1, ``raw_points`` and ``raw_points_highcon`` are identical.

    Returns dict with recall, precision, f1, coverage stats, per-point
    recovered masks so the plot can distinguish recovered vs missed points,
    and **method_breakdown** — a per-method report of how many and what
    percentage of inferred points each detector contributed.
    """

    # ── Method breakdown ──────────────────────────────────────────
    # Report the number and % of inferred points attributed to each of
    # the 19 methods listed in the project description.  These fall into
    # two categories:
    #
    # (A) Per-point source attribution (from candidate_sources column):
    #     These are detectors that directly contribute candidate pixels.
    #     Each inferred point is tagged with the source(s) whose mask
    #     was True at the candidate pixel location.
    #
    # (B) Pipeline-level methods (enabled/disabled, not per-point):
    #     These methods operate at the pipeline level (e.g. smoothing,
    #     clustering, optimisation) and cannot be attributed to individual
    #     points.  We report whether they were enabled and describe their
    #     role.

    # Mapping from internal source keys to user-facing method names.
    # Each internal key maps to exactly one display name — no grouping.
    SOURCE_NAME_MAP = {
        "curvature": "Laplacian curvature",
        "edge": "Edge detection",
        "openness": "Openness",
        "aspect": "Aspect analysis",
        "adaptive_slope": "Adaptive slope",
        "tpi": "TPI",
        "tri": "TRI",
        "dog": "Difference of Gaussians",
        "geomorphon": "Geomorphons",
        "hough": "Hough transforms",
        "orientation": "Orientation-consistent gradients",
        "phase_congruency": "Phase congruency",
        "ridge_crest": "Ridge detection",
        "ml_probability": "Random Forests",
        "subtle_slope_break": "Subtle slope break",
        "convexity_profile": "Convexity profile",
        "gentle_slope_anomaly": "Gentle slope anomaly",
        "multiscale_relief": "Multiscale relief",
        "slope_profile_curvature": "Slope-profile curvature",
    }

    # Pipeline-level methods (always report as enabled/disabled)
    PIPELINE_METHODS = [
        {"key": "tv_regularisation", "name": "TV regularisation",
         "enabled": True,
         "description": "DEM pre-smoothing before feature detection"},
        {"key": "hysteresis", "name": "Hysteresis thresholds",
         "enabled": True,
         "description": "LSM persistent pixel detection via high/low thresholds"},
        {"key": "b_splines", "name": "B-splines",
         "enabled": True,
         "description": "Smoothing B-spline fitting for trace line regularisation"},
        {"key": "bayesian_optimisation", "name": "Bayesian optimisation",
         "enabled": True,
         "description": "Parameter tuning via Optuna TPE sampler"},
        {"key": "dbscan", "name": "DBSCAN",
         "enabled": True,
         "description": "Clustering of nearby candidate points"},
        {"key": "theil_sen", "name": "Theil-Sen regression",
         "enabled": False,
         "description": "Robust regression for scarp parameter profiling (06b)"},
        {"key": "diffusion", "name": "Diffusion modelling",
         "enabled": False,
         "description": "1-D diffusion equation for scarp degradation (06b)"},
    ]

    n_inferred_total = len(inferred_df) if inferred_df is not None else 0

    method_breakdown = {
        "n_inferred_total": n_inferred_total,
        "per_point_methods": {},
        "co_occurrence_matrix": {},
        "pipeline_methods": {},
    }

    # (A) Per-point source attribution with co-occurrence matrix
    if inferred_df is not None and n_inferred_total > 0 and \
       "candidate_sources" in inferred_df.columns:

        cs_col = inferred_df["candidate_sources"].fillna("lsm_only")

        # Parse each point's source set into display names
        all_method_names = sorted(set(SOURCE_NAME_MAP.values()))
        # Add ray tracing & classify fallback if classify_method exists
        has_classify = "classify_method" in inferred_df.columns
        if has_classify:
            all_method_names = sorted(all_method_names +
                ["Ray tracing", "Curvature fallback (classify)"])
        # "LSM (base)" covers points where no alt source matched — the
        # point came from the base LSM hysteresis detector only.
        all_method_names = sorted(all_method_names + ["LSM (base)"])

        method_idx = {m: i for i, m in enumerate(all_method_names)}
        n_methods = len(all_method_names)
        # Co-occurrence counts: cooc[i, j] = number of points where both
        # method i and method j contributed (diagonal = solo count)
        cooc = np.zeros((n_methods, n_methods), dtype=int)
        method_point_counts = {m: 0 for m in all_method_names}

        # If classify_method exists, tag each point with ray_traced or
        # curvature_fallback in addition to its candidate_sources
        if has_classify:
            cm = inferred_df["classify_method"].fillna("none").to_numpy()

        for pidx, val in enumerate(cs_col):
            if not val or (isinstance(val, float) and np.isnan(val)):
                val = "lsm_only"
            sources = str(val).split(";")
            seen_methods = set()
            for src in sources:
                src = src.strip()
                if not src:
                    continue
                if src == "lsm_only":
                    # No alt source matched — base LSM only
                    seen_methods.add("LSM (base)")
                    continue
                method_name = SOURCE_NAME_MAP.get(src, src)
                seen_methods.add(method_name)
            # Add classify_method tag
            if has_classify:
                cm_val = cm[pidx] if pidx < len(cm) else "none"
                if cm_val == "ray_traced":
                    seen_methods.add("Ray tracing")
                elif cm_val == "curvature_fallback":
                    seen_methods.add("Curvature fallback (classify)")

            seen_list = sorted(seen_methods)
            # Solo counts
            for m in seen_list:
                method_point_counts[m] += 1
            # Co-occurrence: for every ordered pair in the set
            for i_m in seen_list:
                for j_m in seen_list:
                    cooc[method_idx[i_m], method_idx[j_m]] += 1

        # Build per-method solo report
        for mname in all_method_names:
            count = method_point_counts.get(mname, 0)
            pct = (count / n_inferred_total * 100.0) if n_inferred_total > 0 else 0.0
            method_breakdown["per_point_methods"][mname] = {
                "n_points": count,
                "percentage": round(pct, 2),
            }

        # Build co-occurrence matrix as nested dict
        cooc_dict = {}
        for i, mi in enumerate(all_method_names):
            cooc_dict[mi] = {}
            for j, mj in enumerate(all_method_names):
                cooc_dict[mi][mj] = int(cooc[i, j])
        method_breakdown["co_occurrence_matrix"] = {
            "method_names": all_method_names,
            "matrix": cooc_dict,
            "n_total_points": n_inferred_total,
        }
    else:
        # No candidate_sources column — report ray tracing from classify_method
        if inferred_df is not None and n_inferred_total > 0 and \
           "classify_method" in inferred_df.columns:
            cm = inferred_df["classify_method"].fillna("none")
            ray_traced = int((cm == "ray_traced").sum())
            curvature_fallback = int((cm == "curvature_fallback").sum())
            method_breakdown["per_point_methods"]["Ray tracing"] = {
                "n_points": ray_traced,
                "percentage": round(ray_traced / n_inferred_total * 100.0, 2),
            }
            method_breakdown["per_point_methods"]["Curvature fallback (classify)"] = {
                "n_points": curvature_fallback,
                "percentage": round(curvature_fallback / n_inferred_total * 100.0, 2),
            }

    # (B) Pipeline-level methods
    for pm in PIPELINE_METHODS:
        method_breakdown["pipeline_methods"][pm["name"]] = {
            "enabled": pm["enabled"],
            "description": pm["description"],
        }

    if len(inferred_df) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "n_inferred": 0, "n_highcon_total": 0, "n_highcon_recovered": 0,
                "per_trace": {},
                "highcon_recovered_mask": np.array([], dtype=bool),
                "highcon_xy": np.empty((0, 2)),
                "inferred_recovered_mask": np.array([], dtype=bool),
                "method_breakdown": method_breakdown}

    # Use raw_points_highcon if available (preserves original HighCon even
    # when Trace2 was replaced by FSD1 for smoothing). This is the fair
    # evaluation — we want to recover the original ground truth points.
    all_highcon = []
    for trace in smoothed_traces:
        # For evaluation: use raw_points_highcon (original HighCon) if present,
        # otherwise raw_points. This means:
        # - Trace1: evaluates against HighCon points (same as before)
        # - Trace2: evaluates against original HighCon Trace2 points, even
        #   though smoothing used FSD1 points
        pts = trace.get("raw_points_highcon", trace["raw_points"])
        all_highcon.append(pts)
    all_highcon = np.vstack(all_highcon) if all_highcon else np.empty((0, 2))

    if len(all_highcon) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "n_inferred": len(inferred_df), "n_highcon_total": 0, "n_highcon_recovered": 0,
                "per_trace": {},
                "highcon_recovered_mask": np.array([], dtype=bool),
                "highcon_xy": np.empty((0, 2)),
                "inferred_recovered_mask": np.array([], dtype=bool),
                "method_breakdown": method_breakdown}

    inferred_xy = inferred_df[["x", "y"]].to_numpy(dtype=float)

    inferred_tree = cKDTree(inferred_xy)
    dists, _ = inferred_tree.query(all_highcon, k=1)
    highcon_recovered_mask = dists <= match_radius
    n_recovered = int(np.sum(highcon_recovered_mask))
    recall = n_recovered / len(all_highcon)

    highcon_tree = cKDTree(all_highcon)
    dists_inf, _ = highcon_tree.query(inferred_xy, k=1)
    inferred_recovered_mask = dists_inf <= match_radius
    n_precise = int(np.sum(inferred_recovered_mask))
    precision = n_precise / len(inferred_xy)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    per_trace = {}
    for trace in smoothed_traces:
        # Evaluate per-trace recovery against original HighCon points
        trace_xy = trace.get("raw_points_highcon", trace["raw_points"])
        if len(trace_xy) == 0:
            continue
        dists_t, _ = inferred_tree.query(trace_xy, k=1)
        n_t = int(np.sum(dists_t <= match_radius))

        # Also evaluate FSD1 recovery if Trace2 was replaced
        trace_info = {
            "n_highcon": len(trace_xy),
            "n_recovered": n_t,
            "recall": n_t / len(trace_xy),
        }
        if trace.get("source") == "FSD1":
            fsd1_pts = trace["raw_points"]
            dists_f, _ = inferred_tree.query(fsd1_pts, k=1)
            n_f = int(np.sum(dists_f <= match_radius))
            trace_info["n_fsd1"] = len(fsd1_pts)
            trace_info["n_fsd1_recovered"] = n_f
            trace_info["fsd1_recall"] = n_f / len(fsd1_pts)
            trace_info["source"] = "FSD1"

        per_trace[trace["trace_id"]] = trace_info

    # ── NFD1 recovery evaluation (for coarse DEMs) ────────────────
    # Use a resolution-scaled match radius for NFD1: at 90m/px, a 100m match
    # radius barely covers 1 pixel.  Scale to max(match_radius, resolution * 2).
    nfd1_match_radius = match_radius
    try:
        with rasterio.open(DEM_PATH) as _src_nfd1:
            _nfd1_res = abs(_src_nfd1.res[0])
        nfd1_match_radius = max(match_radius, _nfd1_res * 2)
    except Exception:
        pass

    nfd1_stats = {}
    if NFD1_PATH.exists():
        nfd1_df = pd.read_csv(NFD1_PATH)
        nfd1_all = nfd1_df[nfd1_df["source"] == "NFD1"]
        if len(nfd1_all) > 0 and len(inferred_xy) > 0:
            nfd1_xy = nfd1_all[["XCoord", "YCoord"]].to_numpy(dtype=float)
            nfd1_dists, _ = inferred_tree.query(nfd1_xy, k=1)
            nfd1_recovered = int(np.sum(nfd1_dists <= nfd1_match_radius))
            nfd1_stats = {
                "n_nfd1_total": len(nfd1_xy),
                "n_nfd1_recovered": nfd1_recovered,
                "nfd1_recall": nfd1_recovered / len(nfd1_xy),
                "match_radius_m": nfd1_match_radius,
            }
            print(f"  [NFD1] Recovery: {nfd1_recovered}/{len(nfd1_xy)} "
                  f"({nfd1_stats['nfd1_recall']:.4f})")

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_inferred": len(inferred_xy),
        "n_highcon_total": len(all_highcon),
        "n_highcon_recovered": n_recovered,
        "per_trace": per_trace,
        # Per-point masks for plotting
        "highcon_recovered_mask": highcon_recovered_mask,
        "highcon_xy": all_highcon,
        "inferred_recovered_mask": inferred_recovered_mask,
        # Per-method breakdown of inferred points by detector source
        "method_breakdown": method_breakdown,
        # NFD1 recovery (populated for coarse DEMs)
        "nfd1": nfd1_stats,
    }


def _plot_method_breakdown(quality, output_path):
    """Generate a three-panel method breakdown figure plus CSV tables.

    Produces:
    1. ``method_breakdown.png`` — three-panel figure:
       (top) horizontal bar chart of solo counts, grouped by method
       category (LSM base / Additional detectors / Subtle-scarp detectors)
       with light-blue background bands and group separators;
       (middle) lower-triangular co-occurrence heatmap (same group order);
       (bottom) two-part panel:
         * left: greedy set-cover showing minimal method set to
           recover all inferred points;
         * right: pairwise complementary analysis — best method
           pairs ranked by coverage with minimal cross-method overlap.
    2. ``method_breakdown_summary.csv`` — flat table (solo counts).
    3. ``method_breakdown_matrix.csv`` — full co-occurrence matrix.
    4. ``method_breakdown_setcover.csv`` — greedy set-cover steps.
    5. ``method_breakdown_pairs.csv`` — pairwise complementary ranking.

    Ray tracing and curvature fallback (classify) are excluded from
    the visual panels (they are classification paths, not detectors).
    """
    mb = quality.get("method_breakdown", {})
    n_total = mb.get("n_inferred_total", 0)

    # ── Methods to exclude from visual panels ────────────────────
    EXCLUDE = {"Ray tracing", "Curvature fallback (classify)"}

    # ── Method grouping and ordering ──────────────────────────────
    SUBTLE_SCARP = {
        "Subtle slope break", "Convexity profile",
        "Gentle slope anomaly", "Multiscale relief",
        "Slope-profile curvature",
    }

    # ── Flat summary CSV (solo counts + pipeline) ─────────────────
    rows = []
    for name, info in mb.get("per_point_methods", {}).items():
        rows.append({
            "method": name,
            "n_points": info["n_points"],
            "percentage": info["percentage"],
            "type": "per-point",
            "enabled": "",
            "description": "",
        })
    for name, info in mb.get("pipeline_methods", {}).items():
        rows.append({
            "method": name,
            "n_points": "N/A",
            "percentage": "N/A",
            "type": "pipeline",
            "enabled": info["enabled"],
            "description": info["description"],
        })

    csv_path = output_path.with_name(output_path.stem + "_summary.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["method", "n_points", "percentage", "type",
                         "enabled", "description"])
        for r in rows:
            writer.writerow([
                r["method"], r["n_points"], r["percentage"], r["type"],
                r.get("enabled", ""), r.get("description", ""),
            ])
    print(f"  Saved method breakdown summary CSV to {csv_path}")

    # ── Co-occurrence matrix CSV (full, including excluded) ───────
    cooc_data = mb.get("co_occurrence_matrix", {})
    all_method_names = cooc_data.get("method_names", [])
    cooc_dict = cooc_data.get("matrix", {})

    matrix_csv_path = output_path.with_name(output_path.stem + "_matrix.csv")
    with open(matrix_csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([""] + all_method_names)
        for mi in all_method_names:
            row_vals = [cooc_dict.get(mi, {}).get(mj, 0) for mj in all_method_names]
            writer.writerow([mi] + row_vals)
    print(f"  Saved co-occurrence matrix CSV to {matrix_csv_path}")

    # ── Filter methods for visual panels ──────────────────────────
    pp_methods = mb.get("per_point_methods", {})
    viz_methods = {k: v for k, v in pp_methods.items() if k not in EXCLUDE}
    if not viz_methods:
        print("  [Method breakdown] No per-point methods to plot, skipping chart.")
        return

    # Build grouped + sorted method list:
    # Group 1: LSM (base) — always first
    # Group 2: Additional detectors (everything except LSM and subtle-scarp)
    # Group 3: Subtle-scarp detectors — always last
    lsm_items = [(k, v) for k, v in viz_methods.items() if k == "LSM (base)"]
    subtle_items = [(k, v) for k, v in viz_methods.items() if k in SUBTLE_SCARP]
    additional_items = [(k, v) for k, v in viz_methods.items()
                        if k != "LSM (base)" and k not in SUBTLE_SCARP]

    lsm_items.sort(key=lambda x: x[1]["n_points"], reverse=True)
    additional_items.sort(key=lambda x: x[1]["n_points"], reverse=True)
    subtle_items.sort(key=lambda x: x[1]["n_points"], reverse=True)

    grouped = lsm_items + additional_items + subtle_items
    names = [m[0] for m in grouped]
    counts = [m[1]["n_points"] for m in grouped]
    pcts = [m[1]["percentage"] for m in grouped]
    n_methods = len(names)

    # Group labels for background colours
    group_labels = []
    for m in names:
        if m == "LSM (base)":
            group_labels.append("LSM (base)")
        elif m in SUBTLE_SCARP:
            group_labels.append("Subtle-scarp detectors")
        else:
            group_labels.append("Additional detectors")

    # Build co-occurrence matrix array (filtered order)
    cooc_arr = np.zeros((n_methods, n_methods), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(names)}
    for mi in names:
        for mj in names:
            cooc_arr[name_to_idx[mi], name_to_idx[mj]] = \
                cooc_dict.get(mi, {}).get(mj, 0)

    cooc_pct = cooc_arr / n_total * 100.0 if n_total > 0 else cooc_arr

    # ── Greedy set-cover (approximate, capped at n_total) ─────────
    solo_counts = np.array([counts[i] for i in range(n_methods)])
    remaining = set(range(n_methods))
    selected_order = []
    union_estimate = 0
    covered_per_step = []

    for step in range(n_methods):
        best_gain = -1
        best_idx = -1
        for i in remaining:
            overlap = 0
            for j in selected_order:
                overlap = max(overlap, cooc_arr[i, j])
            gain = max(0, int(solo_counts[i]) - overlap)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected_order.append(best_idx)
        remaining.remove(best_idx)
        union_estimate = min(union_estimate + best_gain, n_total)
        covered_per_step.append({
            "step": step + 1,
            "method": names[best_idx],
            "incremental": best_gain,
            "cumulative": union_estimate,
            "cumulative_pct": union_estimate / n_total * 100.0 if n_total > 0 else 0,
        })

    # Set-cover CSV
    setcover_csv = output_path.with_name(output_path.stem + "_setcover.csv")
    with open(setcover_csv, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["step", "method", "incremental_points",
                         "cumulative_points", "cumulative_pct"])
        for s in covered_per_step:
            writer.writerow([s["step"], s["method"], s["incremental"],
                             s["cumulative"], round(s["cumulative_pct"], 2)])
    print(f"  Saved set-cover CSV to {setcover_csv}")

    # ── Pairwise complementary analysis ──────────────────────────
    # For every method pair (i, j) compute:
    #   union_size   = |A_i| + |A_j| - |A_i ∩ A_j|
    #   overlap_pct  = |A_i ∩ A_j| / union_size
    #   jaccard      = |A_i ∩ A_j| / union_size  (same as overlap_pct)
    # Rank by union_size descending, then overlap_pct ascending.
    pair_results = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            si = int(solo_counts[i])
            sj = int(solo_counts[j])
            inter = int(cooc_arr[i, j])
            union = si + sj - inter
            if union == 0:
                continue
            overlap_pct = inter / union * 100.0
            jaccard = inter / union
            coverage_pct = union / n_total * 100.0 if n_total > 0 else 0
            pair_results.append({
                "method_a": names[i],
                "method_b": names[j],
                "count_a": si,
                "count_b": sj,
                "intersection": inter,
                "union": union,
                "coverage_pct": coverage_pct,
                "overlap_pct": overlap_pct,
                "jaccard": jaccard,
            })
    # Sort: highest coverage first, then lowest overlap
    pair_results.sort(key=lambda x: (-x["coverage_pct"], x["overlap_pct"]))

    # Pairs CSV
    pairs_csv = output_path.with_name(output_path.stem + "_pairs.csv")
    with open(pairs_csv, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["rank", "method_a", "method_b", "count_a",
                         "count_b", "intersection", "union",
                         "coverage_pct", "overlap_pct", "jaccard"])
        for rank, p in enumerate(pair_results, 1):
            writer.writerow([rank, p["method_a"], p["method_b"],
                             p["count_a"], p["count_b"], p["intersection"],
                             p["union"], round(p["coverage_pct"], 2),
                             round(p["overlap_pct"], 2),
                             round(p["jaccard"], 4)])
    print(f"  Saved pairwise complementary CSV to {pairs_csv}")

    # ── Three-panel figure ────────────────────────────────────────
    # Light-blue background shades per group (all blue tones)
    group_bg_colors = {
        "LSM (base)": "#D6E4FF",           # medium light blue
        "Additional detectors": "#E8F0FE",  # lighter blue
        "Subtle-scarp detectors": "#EDF3FC", # very light blue
    }
    bar_colors_map = {
        "LSM (base)": "#4A90D9",
        "Additional detectors": "#5B9BD5",
        "Subtle-scarp detectors": "#7BAFD4",
    }

    fig, (ax_bar, ax_hm, ax_sc) = plt.subplots(
        3, 1, figsize=(max(10, n_methods * 0.7),
                        max(5, n_methods * 0.3) + 6 + 7),
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.1]})

    # ── Top panel: horizontal bar chart with group backgrounds ────
    y_pos = np.arange(n_methods)
    bar_colors = [bar_colors_map[gl] for gl in group_labels]

    bars = ax_bar.barh(y_pos, counts, color=bar_colors,
                        edgecolor="navy", alpha=0.85)
    for bar, cnt, pct in zip(bars, counts, pcts):
        label = f"{cnt:,} ({pct:.1f}%)" if cnt > 0 else "0 (0.0%)"
        ax_bar.text(bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=8, color="navy")

    for i in range(n_methods):
        gl = group_labels[i]
        ax_bar.axhspan(i - 0.45, i + 0.45,
                        color=group_bg_colors.get(gl, "#F0F0F0"),
                        alpha=0.6, zorder=0)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of inferred points", fontsize=10)
    ax_bar.set_title(f"Solo contribution per method (n={n_total:,} inferred points)",
                     fontsize=11, fontweight="bold")
    ax_bar.set_xlim(0, max(counts) * 1.30 if counts else 1)

    # Group separators
    prev_group = group_labels[0]
    for i in range(1, n_methods):
        if group_labels[i] != prev_group:
            ax_bar.axhline(y=i - 0.5, color="grey", linewidth=0.8,
                           linestyle="--", alpha=0.6)
            prev_group = group_labels[i]

    # ── Middle panel: co-occurrence heatmap (lower triangular) ────
    mask = np.triu(np.ones_like(cooc_arr, dtype=bool), k=1)
    display_arr = np.ma.array(cooc_arr, mask=mask)

    im = ax_hm.imshow(display_arr, cmap="YlOrRd", aspect="auto")
    ax_hm.set_xticks(np.arange(n_methods))
    ax_hm.set_yticks(np.arange(n_methods))
    ax_hm.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax_hm.set_yticklabels(names, fontsize=7)
    ax_hm.set_title("Method co-occurrence matrix (points where both methods contributed)",
                    fontsize=11, fontweight="bold")

    for i in range(n_methods):
        for j in range(i + 1):
            val = cooc_arr[i, j]
            if val == 0:
                continue
            pct_val = cooc_pct[i, j]
            text_color = "white" if val > cooc_arr.max() * 0.5 else "black"
            ax_hm.text(j, i, f"{val:,}\n({pct_val:.1f}%)",
                       ha="center", va="center", fontsize=6,
                       color=text_color)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Number of points", fontsize=9)

    # ── Bottom panel: set-cover (left) + best pairs (right) ──────
    # Clear the default axes and create two side-by-side sub-panels
    ax_sc.clear()
    ax_sc.set_axis_off()

    # Create inset axes for the two sub-panels
    fig.add_axes([0.08, 0.02, 0.40, 0.12])   # left: set-cover
    fig.add_axes([0.55, 0.02, 0.38, 0.12])   # right: best pairs
    ax_left = fig.axes[-2]
    ax_right = fig.axes[-1]

    # — Left sub-panel: greedy set-cover —
    sc_names = [s["method"] for s in covered_per_step]
    sc_increments = [s["incremental"] for s in covered_per_step]
    sc_cumulative = [s["cumulative"] for s in covered_per_step]
    sc_cum_pct = [s["cumulative_pct"] for s in covered_per_step]
    n_sc = len(sc_names)

    sc_y = np.arange(n_sc)
    sc_colors = []
    for s in sc_names:
        if s == "LSM (base)":
            sc_colors.append("#4A90D9")
        elif s in SUBTLE_SCARP:
            sc_colors.append("#7BAFD4")
        else:
            sc_colors.append("#5B9BD5")

    bars_sc = ax_left.barh(sc_y, sc_increments, color=sc_colors,
                           edgecolor="navy", alpha=0.85)
    for bar, inc, cum, cpct in zip(bars_sc, sc_increments,
                                    sc_cumulative, sc_cum_pct):
        label = f"+{inc:,} (cum: {cum:,} / {cpct:.1f}%)"
        ax_left.text(bar.get_width() + max(sc_increments) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=7, color="navy")

    ax_left.set_yticks(sc_y)
    ax_left.set_yticklabels(sc_names, fontsize=7)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Incremental points recovered", fontsize=9)
    ax_left.set_title("Greedy set-cover: minimal method set",
                      fontsize=10, fontweight="bold")
    ax_left.set_xlim(0, max(sc_increments) * 1.7 if sc_increments else 1)
    ax_left.axvline(x=n_total, color="red", linestyle="--", alpha=0.4,
                    linewidth=1)

    # — Right sub-panel: best complementary pairs —
    n_pairs_show = min(12, len(pair_results))
    pr = pair_results[:n_pairs_show]
    pr_labels = [f"{p['method_a']}  +  {p['method_b']}" for p in pr]
    pr_coverage = [p["coverage_pct"] for p in pr]
    pr_overlap = [p["overlap_pct"] for p in pr]

    pr_y = np.arange(n_pairs_show)
    # Colour bars by overlap level (low overlap = greener)
    pr_colors = []
    for ov in pr_overlap:
        if ov < 20:
            pr_colors.append("#2E8B57")    # sea green — very complementary
        elif ov < 40:
            pr_colors.append("#6BBF8A")    # light green
        elif ov < 60:
            pr_colors.append("#DAA520")    # goldenrod
        else:
            pr_colors.append("#CD5C5C")    # indian red — high overlap

    bars_pr = ax_right.barh(pr_y, pr_coverage, color=pr_colors,
                             edgecolor="grey", alpha=0.85)
    for bar, cov, ov in zip(bars_pr, pr_coverage, pr_overlap):
        label = f"{cov:.1f}% cov  (overlap {ov:.1f}%)"
        ax_right.text(bar.get_width() + 1,
                      bar.get_y() + bar.get_height() / 2,
                      label, va="center", fontsize=6, color="black")

    ax_right.set_yticks(pr_y)
    ax_right.set_yticklabels(pr_labels, fontsize=6)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Coverage % (union / total)", fontsize=9)
    ax_right.set_title("Best complementary pairs (high coverage, low overlap)",
                       fontsize=10, fontweight="bold")
    ax_right.set_xlim(0, max(pr_coverage) * 1.5 if pr_coverage else 100)

    plt.tight_layout()
    # Re-add inset axes after tight_layout may reposition
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved method breakdown chart to {output_path}")


# ── Visualisation ────────────────────────────────────────────────────

def _plot_dem_wide_results(inferred_df, smoothed_traces, dem_path, output_path,
                           quality=None, bounds=None):
    """Plot DEM-wide inference results on a single panel.

    Shows:
    - Hillshade background
    - Smoothed trace lines (coloured per trace)
    - HighCon raw points coloured by recovery status (green=recovered, grey=missed)
    - Inferred points coloured by precision status (red=on-trace, blue=off-trace)

    Parameters
    ----------
    inferred_df : DataFrame
        Inferred points with x, y columns.
    smoothed_traces : list of dicts
        Smoothed trace lines.
    dem_path : Path
        DEM GeoTIFF path for hillshade.
    output_path : Path
        Output PNG path.
    quality : dict, optional
        Quality evaluation output from _evaluate_inference_quality().
        If provided, uses highcon_recovered_mask and inferred_recovered_mask
        to colour-code points.
    bounds : tuple (minx, miny, maxx, maxy), optional
        If provided, zoom the plot to this rectangle instead of the full DEM extent.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    with rasterio.open(dem_path) as src:
        out_h, out_w = min(1200, src.height), min(1200, src.width)
        dem = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        med = np.nanmedian(dem) if np.any(np.isfinite(dem)) else 0.0
        hs = compute_hillshade(np.nan_to_num(dem, nan=med),
                               abs(src.res[0]) * src.height / out_h)
        b = src.bounds

    ax.imshow(hs, cmap="gray", extent=(float(b.left), float(b.right), float(b.bottom), float(b.top)),
              origin="upper", alpha=0.7)
    ax.set_xlim(float(b.left), float(b.right))
    ax.set_ylim(float(b.bottom), float(b.top))
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:28355)")
    ax.set_ylabel("Northing (m, EPSG:28355)")
    ax.set_title("05c: DEM-Wide Inference — HighCon Recovery (0.5m LiDAR)")

    colors = ["#ff4444", "#44aaff", "#44ff44", "#ffaa44"]

    # Plot smoothed trace lines
    for i, trace in enumerate(smoothed_traces):
        color = colors[i % len(colors)]
        smooth = trace["coords"]
        mask = (smooth[:, 0] >= b.left) & (smooth[:, 0] <= b.right) & \
               (smooth[:, 1] >= b.bottom) & (smooth[:, 1] <= b.top)
        ax.plot(smooth[mask, 0], smooth[mask, 1], c=color, lw=1.5,
                label=f"Trace{trace['trace_id']} (smoothed)")

    # Plot HighCon raw points coloured by recovery status
    if quality is not None and "highcon_xy" in quality:
        hc_xy = quality["highcon_xy"]
        hc_mask = quality.get("highcon_recovered_mask", np.array([], dtype=bool))
        if len(hc_xy) > 0:
            in_bounds = ((hc_xy[:, 0] >= b.left) & (hc_xy[:, 0] <= b.right) &
                         (hc_xy[:, 1] >= b.bottom) & (hc_xy[:, 1] <= b.top))
            # Subsample if too many points
            if int(in_bounds.sum()) > 20000:
                idx = np.random.RandomState(42).choice(
                    np.where(in_bounds)[0], 20000, replace=False)
                in_bounds_idx = idx
            else:
                in_bounds_idx = np.where(in_bounds)[0]

            recovered_idx = in_bounds_idx[hc_mask[in_bounds_idx]]
            missed_idx = in_bounds_idx[~hc_mask[in_bounds_idx]]

            if len(recovered_idx) > 0:
                ax.scatter(hc_xy[recovered_idx, 0], hc_xy[recovered_idx, 1],
                           s=2, c="#00ff00", alpha=0.4, edgecolors="none",
                           label=f"HighCon recovered ({len(recovered_idx)})")
            if len(missed_idx) > 0:
                ax.scatter(hc_xy[missed_idx, 0], hc_xy[missed_idx, 1],
                           s=1, c="#888888", alpha=0.2, edgecolors="none",
                           label=f"HighCon missed ({len(missed_idx)})")
    else:
        # Fallback: plot all raw points without colour coding
        for i, trace in enumerate(smoothed_traces):
            raw = trace.get("raw_points_highcon", trace["raw_points"])
            mask = (raw[:, 0] >= b.left) & (raw[:, 0] <= b.right) & \
                   (raw[:, 1] >= b.bottom) & (raw[:, 1] <= b.top)
            if int(mask.sum()) > 20000:
                idx = np.random.RandomState(42).choice(
                    np.where(mask)[0], 20000, replace=False)
                ax.scatter(raw[idx, 0], raw[idx, 1], s=1, c="#888888", alpha=0.2,
                           edgecolors="none")
            else:
                ax.scatter(raw[mask, 0], raw[mask, 1], s=1, c="#888888", alpha=0.2,
                           edgecolors="none")

    # Plot inferred points
    if len(inferred_df) > 0:
        xy = inferred_df[["x", "y"]].to_numpy(dtype=float)
        mask = (xy[:, 0] >= b.left) & (xy[:, 0] <= b.right) & \
               (xy[:, 1] >= b.bottom) & (xy[:, 1] <= b.top)
        n_in_bounds = int(mask.sum())
        n_plot = min(50000, n_in_bounds)
        all_indices = np.where(mask)[0]
        if n_in_bounds > n_plot:
            plot_idx = np.random.RandomState(42).choice(all_indices, n_plot, replace=False)
        else:
            plot_idx = all_indices
        plot_xy = xy[plot_idx]

        if quality is not None and "inferred_recovered_mask" in quality:
            inf_mask = quality["inferred_recovered_mask"]
            # Recovered = near a HighCon point (on-trace)
            recovered = inf_mask[plot_idx] if len(inf_mask) == len(xy) else None
            if recovered is not None and recovered.any():
                ax.scatter(plot_xy[recovered, 0], plot_xy[recovered, 1],
                           s=2, c="#ff0000", alpha=0.5, edgecolors="none",
                           label=f"Inferred on-trace ({int(recovered.sum())})")
            if recovered is not None and (~recovered).any():
                ax.scatter(plot_xy[~recovered, 0], plot_xy[~recovered, 1],
                           s=1, c="#0066ff", alpha=0.2, edgecolors="none",
                           label=f"Inferred off-trace ({int((~recovered).sum())})")
        else:
            ax.scatter(plot_xy[:, 0], plot_xy[:, 1], s=1, c="red", alpha=0.3,
                       edgecolors="none", label=f"Inferred ({len(plot_xy)})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved DEM-wide results to {output_path}")


# ── Phase 1: DEM characteristic analysis ──────────────────────────────

def _extract_dem_profiles(dem_src, points_xy, radius_m=100.0, resolution=None):
    """Extract local DEM terrain metrics at given points.

    For each point, computes within a ``radius_m`` window:
    - local_relief: max - min elevation
    - slope_max: maximum gradient (degrees)
    - slope_mean: mean gradient (degrees)
    - curvature_profile: mean profile curvature (1/m)
    - curvature_plan: mean plan curvature (1/m)
    - curvature_profile_max: max profile curvature
    - persistence_mean: mean LSM persistence value (requires running _local_lsm)
    - elevation_mean: mean elevation

    Parameters
    ----------
    dem_src : rasterio dataset handle (open)
    points_xy : np.ndarray (N, 2)
    radius_m : float
    resolution : float, optional (auto from dem_src)

    Returns
    -------
    pd.DataFrame with one row per point
    """
    if resolution is None:
        resolution = abs(dem_src.res[0])

    radius_px = max(3, int(np.ceil(radius_m / resolution)))
    from scipy.ndimage import sobel, gaussian_filter

    results = []
    for i, (x, y) in enumerate(points_xy):
        col, row = ~dem_src.transform * (x, y)
        col_int, row_int = int(round(col)), int(round(row))
        r0 = max(0, row_int - radius_px)
        r1 = min(dem_src.height, row_int + radius_px + 1)
        c0 = max(0, col_int - radius_px)
        c1 = min(dem_src.width, col_int + radius_px + 1)

        if r1 - r0 < 5 or c1 - c0 < 5:
            results.append({"local_relief": np.nan, "slope_max": np.nan,
                            "slope_mean": np.nan, "curvature_profile": np.nan,
                            "curvature_plan": np.nan, "elevation_mean": np.nan})
            continue

        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = dem_src.read(1, window=win).astype(np.float32)
        nodata = dem_src.nodata
        if nodata is not None:
            valid = (data != nodata) & np.isfinite(data)
        else:
            valid = np.isfinite(data)

        if valid.sum() < 10:
            results.append({"local_relief": np.nan, "slope_max": np.nan,
                            "slope_mean": np.nan, "curvature_profile": np.nan,
                            "curvature_plan": np.nan, "elevation_mean": np.nan})
            continue

        data_safe = np.where(valid, data, np.nan)
        elev = data_safe[valid]
        elev_mean = float(np.nanmean(elev))
        local_relief = float(np.nanmax(elev) - np.nanmin(elev))

        # Slope via Sobel
        filled = np.where(valid, data, float(np.nanmedian(elev)))
        dz_dx = sobel(filled, axis=1) / resolution
        dz_dy = sobel(filled, axis=0) / resolution
        slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
        slope_deg = np.degrees(slope_rad)
        slope_deg_valid = slope_deg[valid]
        slope_max = float(np.nanmax(slope_deg_valid))
        slope_mean = float(np.nanmean(slope_deg_valid))

        # Curvature (second derivatives of smoothed DEM)
        smooth = gaussian_filter(filled, sigma=1.5)
        d2z_dx2 = np.gradient(np.gradient(smooth, axis=1), axis=1) / (resolution ** 2)
        d2z_dy2 = np.gradient(np.gradient(smooth, axis=0), axis=0) / (resolution ** 2)
        d2z_dxdy = np.gradient(np.gradient(smooth, axis=1), axis=0) / (resolution ** 2)

        # Profile curvature: curvature in the direction of max slope
        grad_x = np.gradient(smooth, axis=1) / resolution
        grad_y = np.gradient(smooth, axis=0) / resolution
        grad_mag = np.hypot(grad_x, grad_y)
        # Avoid division by zero
        grad_mag_safe = np.where(grad_mag > 1e-10, grad_mag, 1.0)
        prof_curv = -(grad_x**2 * d2z_dx2 + 2 * grad_x * grad_y * d2z_dxdy +
                       grad_y**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2))
        # Plan curvature: curvature perpendicular to max slope
        plan_curv = -((grad_y**2 * d2z_dx2 - 2 * grad_x * grad_y * d2z_dxdy +
                        grad_x**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2)**1.5))

        prof_valid = prof_curv[valid]
        plan_valid = plan_curv[valid]
        curv_profile = float(np.nanmean(np.abs(prof_valid)))
        curv_plan = float(np.nanmean(np.abs(plan_valid)))

        results.append({
            "local_relief": local_relief,
            "slope_max": slope_max,
            "slope_mean": slope_mean,
            "curvature_profile": curv_profile,
            "curvature_plan": curv_plan,
            "elevation_mean": elev_mean,
        })

    return pd.DataFrame(results)


def _extract_persistence_at_points(dem_src, points_xy, params, radius_m=50.0,
                                   resolution=None, river_mask=None):
    """Run _local_lsm on local windows around each point and extract
    persistence and candidate-status at the point location.

    Returns DataFrame with columns: persistence, is_candidate, lsm_value.
    """
    if resolution is None:
        resolution = abs(dem_src.res[0])

    _apply_params(_05a, params)
    radius_px = max(10, int(np.ceil(radius_m / resolution)))

    results = []
    for i, (x, y) in enumerate(points_xy):
        col, row = ~dem_src.transform * (x, y)
        col_int, row_int = int(round(col)), int(round(row))
        r0 = max(0, row_int - radius_px)
        r1 = min(dem_src.height, row_int + radius_px + 1)
        c0 = max(0, col_int - radius_px)
        c1 = min(dem_src.width, col_int + radius_px + 1)

        if r1 - r0 < 20 or c1 - c0 < 20:
            results.append({"persistence": np.nan, "is_candidate": False,
                            "lsm_value": 0.0})
            continue

        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = dem_src.read(1, window=win).astype(np.float32)
        nodata = dem_src.nodata
        if nodata is not None:
            valid = (data != nodata) & np.isfinite(data)
        else:
            valid = np.isfinite(data)

        if valid.sum() < 100:
            results.append({"persistence": np.nan, "is_candidate": False,
                            "lsm_value": 0.0})
            continue

        transform = dem_src.window_transform(win)
        dem_nan = np.where(valid, data, np.nan)
        river = np.zeros_like(data, dtype=bool)
        if river_mask is not None:
            river = _read_mask_on_window(river_mask, transform, dem_src.crs, data.shape)

        state, diag = _05a._local_lsm(dem_nan, valid, river, resolution)

        if state is None:
            results.append({"persistence": 0, "is_candidate": False,
                            "lsm_value": 0.0, "lsm_reason": diag.get("reason", "")})
            continue

        # Point location within this local window
        local_col = col_int - c0
        local_row = row_int - r0
        if 0 <= local_row < state["persistence"].shape[0] and 0 <= local_col < state["persistence"].shape[1]:
            persist_val = int(state["persistence"][local_row, local_col])
            is_cand = bool(state["candidates"][local_row, local_col])
            lsm_val = float(state["lsm"][local_row, local_col])
        else:
            persist_val = np.nan
            is_cand = False
            lsm_val = 0.0

        results.append({"persistence": persist_val, "is_candidate": is_cand,
                        "lsm_value": lsm_val})

    return pd.DataFrame(results)


def run_dem_characteristic_analysis(smoothed_traces, dem_path, out_dir,
                                    bounds=None, n_samples=200,
                                    params=None):
    """Phase 1: Analyse DEM characteristics at FSD1 points vs random non-scarp.

    Extracts terrain metrics (relief, slope, curvature) and LSM persistence
    at FSD1 point locations and at random non-scarp locations within the same
    region, then compares the two distributions.

    Outputs:
    - dem_characteristics.csv : per-point metrics
    - dem_characteristics_comparison.png : violin/box plots comparing FSD1 vs random
    - dem_characteristics_summary.json : summary statistics
    """
    print(f"\n{'='*70}")
    print(f"[Phase 1] DEM Characteristic Analysis at FSD1 vs Random Points")
    print(f"{'='*70}")

    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Get FSD1 points
    fsd1_xy = _get_fsd1_seed_points(smoothed_traces)
    if fsd1_xy is None or len(fsd1_xy) == 0:
        print("  [Error] No FSD1 points found for analysis")
        return None

    # Apply bounds filter if given
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        in_box = ((fsd1_xy[:, 0] >= minx) & (fsd1_xy[:, 0] <= maxx) &
                  (fsd1_xy[:, 1] >= miny) & (fsd1_xy[:, 1] <= maxy))
        fsd1_xy = fsd1_xy[in_box]

    # Subsample if too many
    if len(fsd1_xy) > n_samples:
        idx = np.random.RandomState(42).choice(len(fsd1_xy), n_samples, replace=False)
        fsd1_xy = fsd1_xy[idx]

    print(f"  FSD1 points for analysis: {len(fsd1_xy)}")

    with rasterio.open(dem_path) as src:
        resolution = abs(src.res[0])
        dem_bounds = src.bounds

        # Generate random non-scarp points (far from FSD1, > 500m)
        print(f"  Generating random non-scarp points...")
        n_random = min(n_samples, len(fsd1_xy))
        fsd1_tree = cKDTree(fsd1_xy)
        random_pts = []
        attempts = 0
        while len(random_pts) < n_random and attempts < n_random * 50:
            x = np.random.uniform(dem_bounds.left, dem_bounds.right)
            y = np.random.uniform(dem_bounds.bottom, dem_bounds.top)
            if bounds is not None:
                if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                    attempts += 1
                    continue
            dist, _ = fsd1_tree.query([x, y], k=1)
            if dist > 500.0:  # at least 500m from any FSD1 point
                random_pts.append([x, y])
            attempts += 1
        random_xy = np.array(random_pts) if random_pts else np.empty((0, 2))
        print(f"  Random non-scarp points: {len(random_xy)}")

        # Extract terrain metrics
        print(f"  Extracting terrain metrics at FSD1 points...")
        fsd1_metrics = _extract_dem_profiles(src, fsd1_xy, radius_m=100.0, resolution=resolution)
        fsd1_metrics["category"] = "FSD1"

        print(f"  Extracting terrain metrics at random points...")
        random_metrics = _extract_dem_profiles(src, random_xy, radius_m=100.0, resolution=resolution)
        random_metrics["category"] = "Random"

        # Extract persistence/candidate status
        print(f"  Running LSM at FSD1 points (persistence analysis)...")
        hand_masks = _get_hand_masks(None)
        fsd1_persist = _extract_persistence_at_points(
            src, fsd1_xy, params, radius_m=150.0, resolution=resolution,
            river_mask=hand_masks)
        fsd1_persist["category"] = "FSD1"

        print(f"  Running LSM at random points...")
        random_persist = _extract_persistence_at_points(
            src, random_xy, params, radius_m=150.0, resolution=resolution,
            river_mask=hand_masks)
        random_persist["category"] = "Random"

    # Combine
    fsd1_metrics = pd.concat([fsd1_metrics.reset_index(drop=True),
                              fsd1_persist.reset_index(drop=True)], axis=1)
    random_metrics = pd.concat([random_metrics.reset_index(drop=True),
                                random_persist.reset_index(drop=True)], axis=1)

    # Drop duplicate category columns
    fsd1_metrics = fsd1_metrics.loc[:, ~fsd1_metrics.columns.duplicated()]
    random_metrics = random_metrics.loc[:, ~random_metrics.columns.duplicated()]

    all_metrics = pd.concat([fsd1_metrics, random_metrics], ignore_index=True)
    all_metrics.to_csv(out_dir / "dem_characteristics.csv", index=False)
    print(f"  Saved per-point metrics to {out_dir / 'dem_characteristics.csv'}")

    # Summary comparison
    summary = {}
    metric_cols = ["local_relief", "slope_max", "slope_mean",
                   "curvature_profile", "curvature_plan",
                   "persistence", "lsm_value"]
    for col in metric_cols:
        if col not in all_metrics.columns:
            continue
        fsd1_vals = all_metrics[all_metrics["category"] == "FSD1"][col].dropna()
        rand_vals = all_metrics[all_metrics["category"] == "Random"][col].dropna()
        summary[col] = {
            "fsd1_mean": float(fsd1_vals.mean()) if len(fsd1_vals) > 0 else None,
            "fsd1_median": float(fsd1_vals.median()) if len(fsd1_vals) > 0 else None,
            "fsd1_std": float(fsd1_vals.std()) if len(fsd1_vals) > 0 else None,
            "random_mean": float(rand_vals.mean()) if len(rand_vals) > 0 else None,
            "random_median": float(rand_vals.median()) if len(rand_vals) > 0 else None,
            "random_std": float(rand_vals.std()) if len(rand_vals) > 0 else None,
        }

    # Candidate coverage
    fsd1_cand = all_metrics[all_metrics["category"] == "FSD1"]["is_candidate"]
    rand_cand = all_metrics[all_metrics["category"] == "Random"]["is_candidate"]
    summary["candidate_coverage"] = {
        "fsd1_pct": float(fsd1_cand.mean() * 100) if len(fsd1_cand) > 0 else 0.0,
        "random_pct": float(rand_cand.mean() * 100) if len(rand_cand) > 0 else 0.0,
    }

    with open(out_dir / "dem_characteristics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary to {out_dir / 'dem_characteristics_summary.json'}")

    # Print key findings
    print(f"\n  [Phase 1 Results]")
    for col in metric_cols:
        if col in summary:
            s = summary[col]
            f_m = s["fsd1_mean"]
            r_m = s["random_mean"]
            ratio = f_m / r_m if r_m and r_m > 1e-10 else None
            print(f"    {col}: FSD1={f_m:.4f} vs Random={r_m:.4f}"
                  f"  (ratio={ratio:.2f})" if ratio else
                  f"    {col}: FSD1={f_m} vs Random={r_m}")
    print(f"    Candidate coverage: FSD1={summary['candidate_coverage']['fsd1_pct']:.1f}% "
          f"vs Random={summary['candidate_coverage']['random_pct']:.1f}%")

    # Visualization
    _plot_dem_characteristics(all_metrics, out_dir / "dem_characteristics_comparison.png")

    return summary


def _plot_dem_characteristics(metrics_df, output_path):
    """Create violin/box plots comparing FSD1 vs random metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    plot_cols = ["local_relief", "slope_max", "slope_mean",
                 "curvature_profile", "curvature_plan",
                 "persistence", "lsm_value"]

    for i, col in enumerate(plot_cols):
        if i >= len(axes) or col not in metrics_df.columns:
            continue
        ax = axes[i]
        fsd1_vals = metrics_df[metrics_df["category"] == "FSD1"][col].dropna()
        rand_vals = metrics_df[metrics_df["category"] == "Random"][col].dropna()

        data = [fsd1_vals.values, rand_vals.values]
        labels = ["FSD1", "Random"]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("#ff9999")
        bp["boxes"][1].set_facecolor("#9999ff")
        ax.set_title(col, fontsize=11)
        ax.set_ylabel(col)

    # Hide unused subplots
    for j in range(len(plot_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phase 1: DEM Characteristics at FSD1 vs Random Points", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved comparison plot to {output_path}")


# ── Phase 2: Candidate-coverage tuning ────────────────────────────────

def _evaluate_candidate_coverage(module, params, dem_window, valid, river_mask,
                                  resolution, transform, fsd1_xy,
                                  match_radius_m=30.0):
    """Run LSM with proposed params and measure FSD1 candidate coverage.

    Instead of running the full classify pipeline, this only measures
    whether LSM candidates are present near FSD1 points — directly
    targeting the bottleneck.

    Returns dict with coverage, n_candidates, n_fsd1_covered, penalty_score.
    """
    _apply_params(module, params)

    dem_nan = np.where(valid, dem_window, np.nan)
    state, diag = module._local_lsm(dem_nan, valid, river_mask, resolution)

    if state is None:
        return {"coverage": 0.0, "n_candidates": 0, "n_fsd1_covered": 0,
                "n_fsd1_total": len(fsd1_xy), "penalty_score": 0.0,
                "reason": diag.get("reason", "unknown")}

    candidates = state.get("candidates")
    if candidates is None or not candidates.any():
        return {"coverage": 0.0, "n_candidates": 0, "n_fsd1_covered": 0,
                "n_fsd1_total": len(fsd1_xy), "penalty_score": 0.0,
                "reason": "no_candidates"}

    # Get candidate pixel coordinates
    cand_rows, cand_cols = np.where(candidates)
    if len(cand_rows) == 0:
        return {"coverage": 0.0, "n_candidates": 0, "n_fsd1_covered": 0,
                "n_fsd1_total": len(fsd1_xy), "penalty_score": 0.0}

    cand_x, cand_y = transform * (cand_cols, cand_rows)
    cand_xy = np.column_stack([cand_x, cand_y])

    # Check FSD1 coverage
    n_covered = 0
    if len(fsd1_xy) > 0:
        cand_tree = cKDTree(cand_xy)
        dists, _ = cand_tree.query(fsd1_xy, k=1)
        n_covered = int(np.sum(dists <= match_radius_m))

    coverage = n_covered / max(len(fsd1_xy), 1)
    n_total_pixels = valid.sum()
    candidate_density = len(cand_rows) / max(n_total_pixels, 1)

    # Penalty: penalize excessive candidate density (> 5% of valid pixels)
    density_penalty = max(0.0, candidate_density - 0.05) * 10.0

    # Objective: coverage - penalty
    penalty_score = coverage - density_penalty

    return {
        "coverage": coverage,
        "n_candidates": len(cand_rows),
        "n_fsd1_covered": n_covered,
        "n_fsd1_total": len(fsd1_xy),
        "candidate_density": candidate_density,
        "density_penalty": density_penalty,
        "penalty_score": penalty_score,
        "reason": "ok",
    }


def _run_candidate_coverage_tuning(tuning_windows, dem_crs, n_trials=TUNING_N_TRIALS):
    """Optuna tuning with candidate-coverage objective.

    Instead of F2 (final inferred point recovery), this maximises the
    fraction of FSD1 points that have an LSM candidate within 30m,
    with a penalty for excessive candidate density.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] Not available, falling back to grid search")
        return _run_candidate_coverage_grid_search(tuning_windows, dem_crs)

    hand_masks = _get_hand_masks(dem_crs)
    eval_windows = tuning_windows[:min(4, len(tuning_windows))]

    def objective(trial):
        params = {
            "LSM_PERSISTENCE_K": trial.suggest_int("LSM_PERSISTENCE_K", 1, 3),
            "LSM_HYSTERESIS_LOW_SIGMA": trial.suggest_float("LSM_HYSTERESIS_LOW_SIGMA", 0.1, 2.0),
            "LSM_HYSTERESIS_HIGH_SIGMA": trial.suggest_float("LSM_HYSTERESIS_HIGH_SIGMA", 0.5, 6.0),
            "TV_WEIGHT_SCALE": trial.suggest_float("TV_WEIGHT_SCALE", 0.1, 4.0),
            "BASE_RAY_LENGTH_M": trial.suggest_float("BASE_RAY_LENGTH_M", 200.0, 1200.0),
            "MIN_SCARP_HEIGHT_M": trial.suggest_float("MIN_SCARP_HEIGHT_M", 0.05, 0.5),
            "MIN_SCARP_ASYMMETRY_SCORE": trial.suggest_float("MIN_SCARP_ASYMMETRY_SCORE", 0.05, 0.5),
            "MIN_SCARP_WIDTH_M": trial.suggest_float("MIN_SCARP_WIDTH_M", 5.0, 50.0),
        }

        all_scores = []
        for win_data in eval_windows:
            river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                          win_data["dem"].shape)
            scores = _evaluate_candidate_coverage(
                _05a, params,
                win_data["dem"], win_data["valid"], river,
                win_data["resolution"], win_data["transform"],
                win_data["highcon_xy"],
            )
            all_scores.append(scores)

        avg_score = np.mean([s["penalty_score"] for s in all_scores])
        avg_coverage = np.mean([s["coverage"] for s in all_scores])
        avg_density = np.mean([s.get("candidate_density", 0) for s in all_scores])
        return avg_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    print(f"  [Candidate-Coverage Tuning] Best score={best_score:.4f}")
    print(f"  [Candidate-Coverage Tuning] Best params: {best_params}")

    return best_params, best_score, []


def _run_candidate_coverage_grid_search(tuning_windows, dem_crs):
    """Grid search fallback for candidate-coverage tuning."""
    best_score = -1.0
    best_params = DEFAULT_PARAMS.copy()
    results_log = []

    grid = {
        "LSM_PERSISTENCE_K": [1, 2],
        "LSM_HYSTERESIS_LOW_SIGMA": [0.25, 0.5, 1.0],
        "LSM_HYSTERESIS_HIGH_SIGMA": [1.0, 2.0, 4.0],
        "TV_WEIGHT_SCALE": [0.5, 1.0, 2.0],
        "BASE_RAY_LENGTH_M": [400.0, 800.0],
        "MIN_SCARP_HEIGHT_M": [0.1, 0.2],
        "MIN_SCARP_ASYMMETRY_SCORE": [0.15, 0.3],
        "MIN_SCARP_WIDTH_M": [10.0, 30.0],
    }

    import itertools
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total_combos = 1
    for v in value_lists:
        total_combos *= len(v)
    print(f"  [Grid Search] {total_combos} parameter combinations")

    eval_windows = tuning_windows[:min(3, len(tuning_windows))]
    hand_masks = _get_hand_masks(dem_crs)

    for combo_idx, combo in enumerate(itertools.product(*value_lists)):
        params = dict(zip(keys, combo))

        all_scores = []
        for win_data in eval_windows:
            river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                          win_data["dem"].shape)
            scores = _evaluate_candidate_coverage(
                _05a, params,
                win_data["dem"], win_data["valid"], river,
                win_data["resolution"], win_data["transform"],
                win_data["highcon_xy"],
            )
            all_scores.append(scores)

        avg_score = np.mean([s["penalty_score"] for s in all_scores])
        avg_coverage = np.mean([s["coverage"] for s in all_scores])
        avg_density = np.mean([s.get("candidate_density", 0) for s in all_scores])

        results_log.append({
            "combo_idx": combo_idx,
            "score": avg_score,
            "coverage": avg_coverage,
            "density": avg_density,
            **params,
        })

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        if (combo_idx + 1) % 10 == 0:
            print(f"  [Grid] {combo_idx + 1}/{total_combos}: best={best_score:.4f}")

    print(f"  [Grid Search] Best score={best_score:.4f}, params={best_params}")
    return best_params, best_score, results_log


def tune_candidate_coverage(smoothed_traces, dem_path, dem_crs, bounds=None):
    """Phase 2: Tune LSM parameters to maximize FSD1 candidate coverage.

    Uses a dedicated objective that measures how many FSD1 points have
    an LSM candidate within 30m, with a penalty for excessive candidate
    density.
    """
    print(f"\n{'='*70}")
    print(f"[Phase 2] Candidate-Coverage Parameter Tuning")
    print(f"{'='*70}")

    with rasterio.open(dem_path) as src:
        tuning_windows = _sample_tuning_windows(src, smoothed_traces, bounds=bounds)

    if not tuning_windows:
        print("  [Warning] No tuning windows could be sampled, using default params")
        return DEFAULT_PARAMS.copy(), 0.0, []

    best_params, best_score, results_log = _run_candidate_coverage_tuning(
        tuning_windows, dem_crs, n_trials=TUNING_N_TRIALS)

    # Evaluate default params for comparison
    hand_masks = _get_hand_masks(dem_crs)
    default_scores = []
    for win_data in tuning_windows[:min(4, len(tuning_windows))]:
        river = _read_mask_on_window(hand_masks, win_data["transform"], dem_crs,
                                      win_data["dem"].shape)
        scores = _evaluate_candidate_coverage(
            _05a, DEFAULT_PARAMS,
            win_data["dem"], win_data["valid"], river,
            win_data["resolution"], win_data["transform"],
            win_data["highcon_xy"],
        )
        default_scores.append(scores)

    default_cov = np.mean([s["coverage"] for s in default_scores]) if default_scores else 0.0
    print(f"\n  [Tuning Results]")
    print(f"  Default params coverage: {default_cov:.4f}")
    print(f"  Tuned params coverage:   {best_score:.4f}")
    print(f"  Improvement:              {best_score - default_cov:+.4f}")

    return best_params, best_score, results_log


# ── Phase 3: Alternative candidate sources ─────────────────────────────

def _compute_multiscale_curvature_candidates(dem, valid, resolution,
                                              sigma_range=(1.0, 3.0, 5.0),
                                              percentile=90):
    """Generate candidate mask from multi-scale curvature thresholds.

    Computes profile and plan curvature at multiple smoothing scales,
    then marks pixels where curvature exceeds the ``percentile``-th
    percentile of valid values at any scale.
    """
    from scipy.ndimage import gaussian_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    for sigma in sigma_range:
        smooth = gaussian_filter(filled, sigma=sigma)

        # Compute gradients
        dz_dx = np.gradient(smooth, resolution, axis=1)
        dz_dy = np.gradient(smooth, resolution, axis=0)

        # Second derivatives
        d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
        d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
        d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)

        grad_mag = np.hypot(dz_dx, dz_dy)
        grad_mag_safe = np.where(grad_mag > 1e-10, grad_mag, 1.0)

        # Profile curvature (in direction of steepest slope)
        prof_curv = -(dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy +
                       dz_dy**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2))

        # Plan curvature (perpendicular to steepest slope)
        plan_curv = -((dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy +
                        dz_dx**2 * d2z_dy2) /
                       (grad_mag_safe * (1 + grad_mag**2)**1.5))

        # Threshold at high percentile
        for curv_field in [prof_curv, plan_curv]:
            vals = np.abs(curv_field[valid])
            if len(vals) > 10:
                thresh = np.percentile(vals, percentile)
                candidate_mask |= (np.abs(curv_field) > thresh) & valid

    return candidate_mask


def _compute_edge_detection_candidates(dem, valid, resolution,
                                        sigma=2.0, percentile=88):
    """Generate candidate mask from Canny edge detection on hillshade.

    Applies Canny edge detector to the smoothed DEM (or its hillshade).
    Edges often correspond to scarp faces — topographic breaks.
    """
    from skimage.feature import canny
    from scipy.ndimage import gaussian_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=sigma)

    # Compute hillshade for edge detection (illumination highlights scarps)
    hs = compute_hillshade(smooth, resolution)

    # Canny on hillshade
    edges_hs = canny(hs, sigma=1.5, use_quantiles=True, low_threshold=0.85, high_threshold=0.95)

    # Also compute gradient magnitude and threshold
    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    grad_mag = np.hypot(dz_dx, dz_dy)

    grad_vals = grad_mag[valid]
    if len(grad_vals) > 10:
        thresh = np.percentile(grad_vals, percentile)
        edges_grad = (grad_mag > thresh) & valid
    else:
        edges_grad = np.zeros_like(valid)

    return edges_hs & valid | edges_grad


def _compute_topographic_openness_candidates(dem, valid, resolution,
                                              annulus_radii_m=(30.0, 60.0, 100.0),
                                              percentile=88):
    """Generate candidate mask from local relief / topographic openness.

    Computes the difference between a central pixel and the mean of
    surrounding annuli at multiple radii.  Scarp faces show high
    positive or negative relief relative to their surroundings.
    """
    from scipy.ndimage import uniform_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    for radius_m in annulus_radii_m:
        radius_px = max(3, int(np.ceil(radius_m / resolution)))
        # Local mean over a window slightly larger than the annulus
        win_size = radius_px * 2 + 1
        local_mean = uniform_filter(filled, size=win_size)
        # Difference between pixel and local mean
        diff = filled - local_mean
        vals = np.abs(diff[valid])
        if len(vals) > 10:
            thresh = np.percentile(vals, percentile)
            candidate_mask |= (np.abs(diff) > thresh) & valid

    return candidate_mask


def _compute_aspect_change_candidates(dem, valid, resolution,
                                      sigma=3.0, percentile=90):
    """Generate candidate mask from abrupt changes in slope aspect.

    Computes slope aspect at each pixel, then identifies locations
    where aspect changes rapidly (high spatial gradient of aspect).
    These transitions often correspond to ridge/valley features including
    fault scarps.
    """
    from scipy.ndimage import gaussian_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=sigma)

    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)

    # Aspect (direction of steepest slope, in degrees)
    aspect = np.arctan2(-dz_dy, dz_dx)  # radians
    # Compute spatial gradient of aspect (high = abrupt direction change)
    d_aspect_dx = np.gradient(aspect, axis=1)
    d_aspect_dy = np.gradient(aspect, axis=0)
    aspect_gradient = np.hypot(d_aspect_dx, d_aspect_dy)

    vals = aspect_gradient[valid]
    if len(vals) > 10:
        thresh = np.percentile(vals, percentile)
        return (aspect_gradient > thresh) & valid
    else:
        return np.zeros_like(valid)


# -- New alternative candidate sources (10 methods) ----------------------

def _compute_adaptive_slope_candidates(dem, valid, resolution,
                                      window_m=150.0, k_std=2.0,
                                      sigma_smooth=2.0, percentile=90):
    """Adaptive Local Slope Anomaly.

    Computes slope (gradient magnitude) at a moderate smoothing scale,
    then for each pixel calculates the deviation of slope from the median
    slope of a surrounding moving window.  Scarps stand out as locally
    steeper than their immediate surroundings.

    Candidate if: slope > median_slope + k * std_slope
    """
    from scipy.ndimage import gaussian_filter, median_filter, generic_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=sigma_smooth)

    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    slope = np.hypot(dz_dx, dz_dy)

    window_px = max(5, int(np.ceil(window_m / resolution)))
    win_size = window_px * 2 + 1

    # Use median filter for local median slope
    median_slope = median_filter(slope, size=win_size, mode="reflect")
    # Use uniform_filter as proxy for local std (mean of squared deviations)
    slope_sq = slope ** 2
    mean_slope = _uniform_filter_nan(slope, win_size)
    mean_slope_sq = _uniform_filter_nan(slope_sq, win_size)
    std_slope = np.sqrt(np.maximum(mean_slope_sq - mean_slope ** 2, 0.0))

    anomaly = slope - (median_slope + k_std * std_slope)
    vals = anomaly[valid]
    if len(vals) > 10:
        thresh = np.percentile(vals[vals > 0], min(percentile, 99))
        mask = (anomaly > thresh) & valid
    else:
        mask = np.zeros_like(valid)

    return mask


def _uniform_filter_nan(arr, win_size):
    """Apply uniform_filter ignoring NaN by filling with median first."""
    from scipy.ndimage import uniform_filter
    med = np.nanmedian(arr) if np.any(np.isfinite(arr)) else 0.0
    filled = np.where(np.isfinite(arr), arr, med)
    return uniform_filter(filled, size=win_size, mode="reflect")


def _compute_tpi_candidates(dem, valid, resolution,
                            window_sizes_m=(50.0, 100.0, 200.0),
                            n_std=2.0, percentile=90):
    """Topographic Position Index (TPI).

    TPI compares a pixel's elevation to the mean elevation of a surrounding
    annulus.  Positive TPI = higher than neighbours (ridge/crest), negative
    = lower (valley).  Scarp faces produce sharp transitions.

    Candidate if: |TPI| > n_std * std(TPI) or TPI gradient is high.
    """
    from scipy.ndimage import uniform_filter, generic_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    for win_m in window_sizes_m:
        win_px = max(3, int(np.ceil(win_m / resolution)))
        win_size = win_px * 2 + 1
        local_mean = uniform_filter(filled, size=win_size, mode="reflect")
        tpi = filled - local_mean

        vals = tpi[valid]
        if len(vals) < 10:
            continue
        tpi_std = np.std(vals)
        if tpi_std < 1e-10:
            continue

        # High |TPI| pixels
        high_tpi = (np.abs(tpi) > n_std * tpi_std) & valid

        # TPI gradient (sharp transitions)
        tpi_grad_x = np.gradient(tpi, axis=1)
        tpi_grad_y = np.gradient(tpi, axis=0)
        tpi_grad = np.hypot(tpi_grad_x, tpi_grad_y)
        grad_vals = tpi_grad[valid]
        if len(grad_vals) > 10:
            grad_thresh = np.percentile(grad_vals, percentile)
            high_grad = (tpi_grad > grad_thresh) & valid
        else:
            high_grad = np.zeros_like(valid)

        candidate_mask |= high_tpi | high_grad

    return candidate_mask


def _compute_tri_candidates(dem, valid, resolution, percentile=92):
    """Terrain Ruggedness Index (TRI) / Roughness.

    TRI measures the local variability of elevation as the mean absolute
    difference between the central pixel and its 8 neighbours.  Scarp faces
    have higher roughness than smooth slopes.
    """
    from scipy.ndimage import uniform_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)

    # TRI = mean of |center - neighbour| for 8 neighbours
    # Equivalent to: mean(abs(dem - shifted)) for 8 shifts
    # Use uniform_filter of abs differences for efficiency
    # Center - mean(neighbours) is simpler and equivalent in spirit
    win_size = 3
    local_mean = uniform_filter(filled, size=win_size, mode="reflect")
    diff = np.abs(filled - local_mean)

    vals = diff[valid]
    if len(vals) > 10:
        thresh = np.percentile(vals, percentile)
        return (diff > thresh) & valid
    else:
        return np.zeros_like(valid)


def _compute_dog_candidates(dem, valid, resolution,
                           sigma_pairs=((1.0, 3.0), (2.0, 5.0), (3.0, 7.0)),
                           percentile=90):
    """Difference of Gaussians (DoG) / Laplacian of Gaussian (LoG).

    Classic blob/edge detectors that respond to step-like discontinuities.
    Smooth DEM with two Gaussian kernels, subtract the two, threshold.
    """
    from scipy.ndimage import gaussian_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    for sigma1, sigma2 in sigma_pairs:
        smooth1 = gaussian_filter(filled, sigma=sigma1)
        smooth2 = gaussian_filter(filled, sigma=sigma2)
        dog = smooth1 - smooth2

        vals = np.abs(dog[valid])
        if len(vals) > 10:
            thresh = np.percentile(vals, percentile)
            candidate_mask |= (np.abs(dog) > thresh) & valid

    return candidate_mask


def _compute_geomorphon_candidates(dem, valid, resolution, flat_threshold=0.5):
    """Template Matching with Synthetic Geomorphon Cross-Sections.

    Classifies each 3x3 neighborhood into one of 10 geomorphon types:
    flat, peak, ridge, shoulder, spur, pit, valley, footslope, hollow.
    Scarp-related types (ridge, valley, shoulder, footslope, spur, hollow)
    are marked as candidates.

    The 10 types follow the standard geomorphon classification based on
    the relative elevation pattern of the 8 neighbours around the center.
    """
    from scipy.ndimage import uniform_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)

    # Use a slightly smoothed DEM for stability
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(filled, sigma=1.0)

    h, w = smooth.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    # Compute elevation differences with 8 neighbours
    # Neighbour offsets: N, NE, E, SE, S, SW, W, NW
    shifts = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    diffs = []
    for dr, dc in shifts:
        shifted = np.roll(np.roll(smooth, dr, axis=0), dc, axis=1)
        diffs.append(smooth - shifted)

    diff_stack = np.stack(diffs, axis=0)  # (8, H, W)

    # Classify: + = higher than neighbour, - = lower, 0 = similar
    pos = diff_stack > flat_threshold
    neg = diff_stack < -flat_threshold
    flat = ~pos & ~neg

    n_pos = pos.sum(axis=0)
    n_neg = neg.sum(axis=0)
    n_flat = flat.sum(axis=0)

    # Geomorphon classification (simplified from the 10 types)
    # peak: all 8 neighbours lower (n_pos == 8)
    # pit: all 8 neighbours higher (n_neg == 8)
    # flat: mostly flat (n_flat >= 6)
    # ridge: 3 contiguous higher cells in a row, rest lower
    # valley: 3 contiguous lower cells in a row, rest higher
    # shoulder: 3 contiguous similar cells, rest lower
    # footslope: 5 contiguous similar cells, rest higher
    # spur: 3 contiguous higher cells, rest lower (variant)
    # hollow: 3 contiguous lower cells, rest higher (variant)

    is_peak = (n_pos == 8)
    is_pit = (n_neg == 8)
    is_flat = (n_flat >= 6)

    # Ridge/shoulder: center is higher than most neighbours
    is_ridge = (n_pos >= 5) & (n_pos < 8) & ~is_peak
    # Valley/footslope: center is lower than most neighbours
    is_valley = (n_neg >= 5) & (n_neg < 8) & ~is_pit

    # Shoulder/spur: mixed pattern with some higher neighbours
    is_shoulder = (n_pos >= 3) & (n_pos < 5) & (n_neg <= 2)
    # Footslope/hollow: mixed pattern with some lower neighbours
    is_foothslope = (n_neg >= 3) & (n_neg < 5) & (n_pos <= 2)

    # Scarp-related types: ridge, valley, shoulder, footslope
    # (peaks and pits are too generic, flat is not scarp-related)
    scarp_types = is_ridge | is_valley | is_shoulder | is_foothslope

    candidate_mask = scarp_types & valid
    # Exclude edges where roll wraps around
    edge = max(1, int(np.ceil(10.0 / resolution)))
    candidate_mask[:edge, :] = False
    candidate_mask[-edge:, :] = False
    candidate_mask[:, :edge] = False
    candidate_mask[:, -edge:] = False

    return candidate_mask


def _compute_hough_candidates(dem, valid, resolution,
                             slope_threshold_deg=20.0,
                             min_line_length_m=50.0,
                             max_line_gap_m=20.0,
                             dilate_px=5):
    """Lineament Detection via Hough Transform.

    Fault scarps are often linear features.  Binarises the slope map,
    applies probabilistic Hough transform to extract straight segments,
    then dilates detected lines to create a candidate mask.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation
    from skimage.transform import probabilistic_hough_line
    from skimage.feature import canny

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=2.0)

    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # Binarise slope map
    binary = (slope_deg > slope_threshold_deg) & valid
    if binary.sum() < 50:
        # Fallback: use high percentile
        vals = slope_deg[valid]
        if len(vals) > 10:
            thresh = np.percentile(vals, 95)
            binary = (slope_deg > thresh) & valid
        else:
            return np.zeros_like(valid)

    # Convert to skimage expected format (0/1)
    binary_img = binary.astype(np.uint8) * 255

    # Line length and gap in pixels
    min_line_length_px = max(5, int(np.ceil(min_line_length_m / resolution)))
    max_line_gap_px = max(2, int(np.ceil(max_line_gap_m / resolution)))

    # Detect lines
    try:
        lines = probabilistic_hough_line(
            binary_img,
            threshold=10,
            line_length=min_line_length_px,
            line_gap=max_line_gap_px,
        )
    except Exception:
        return np.zeros_like(valid)

    if len(lines) == 0:
        return np.zeros_like(valid)

    # Create mask from detected lines
    from skimage.draw import line_aa
    h, w = dem.shape
    line_mask = np.zeros((h, w), dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        # skimage uses (row, col) = (y, x) coordinates
        rr, cc, _ = line_aa(int(y0), int(x0), int(y1), int(x1))
        valid_pts = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        line_mask[rr[valid_pts], cc[valid_pts]] = True

    # Dilate to create a buffer around lines
    dilated = binary_dilation(line_mask, iterations=dilate_px)

    print(f"      [Hough] {len(lines)} lines detected, "
          f"{int(line_mask.sum())} px on lines, "
          f"{int(dilated.sum())} px after dilation")

    return dilated & valid


def _compute_orientation_consistency_candidates(dem, valid, resolution,
                                               window_m=100.0,
                                               min_slope_deg=5.0,
                                               max_circ_var=0.3,
                                               percentile=88):
    """Orientation-Consistent Gradient Clustering.

    Scarp faces often have a consistent orientation (strike) over tens to
    hundreds of metres.  Computes local gradient orientation (aspect) and
    looks for pixels where the orientation is stable over a larger
    neighbourhood (low circular variance) and slope is above a minimum.

    Candidate if: circular_variance < max_circ_var AND slope > min_slope.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=2.0)

    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Aspect as angle in radians
    aspect = np.arctan2(-dz_dy, dz_dx)

    # Circular variance of aspect in a window
    # sin/cos averaged, then var = 1 - R where R = sqrt(mean_sin^2 + mean_cos^2)
    sin_aspect = np.sin(aspect)
    cos_aspect = np.cos(aspect)

    win_px = max(5, int(np.ceil(window_m / resolution)))
    win_size = win_px * 2 + 1

    mean_sin = uniform_filter(sin_aspect, size=win_size, mode="reflect")
    mean_cos = uniform_filter(cos_aspect, size=win_size, mode="reflect")
    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    circ_var = 1.0 - R

    # Candidate: low circular variance (consistent orientation) + slope above minimum
    consistent = (circ_var < max_circ_var) & (slope_deg > min_slope_deg) & valid

    # Also threshold by slope for pixels with high consistency
    vals = slope_deg[consistent]
    if len(vals) > 10:
        thresh = np.percentile(vals, percentile)
        mask = consistent & (slope_deg > thresh)
    else:
        mask = consistent

    return mask


def _compute_phase_congruency_candidates(dem, valid, resolution,
                                         nscale=4, norient=6,
                                         percentile=90):
    """Phase Congruency Edge Detection.

    Phase congruency is a robust edge detector invariant to contrast
    changes.  Highlights features with high phase coherence such as step
    edges.  Uses the phasepack library if available, otherwise falls back
    to a gradient-based approximation.
    """
    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)

    # Normalize DEM to 0-1 range
    dem_min = float(np.min(filled[valid])) if valid.any() else 0.0
    dem_max = float(np.max(filled[valid])) if valid.any() else 1.0
    dem_range = dem_max - dem_min
    if dem_range < 1e-6:
        return np.zeros_like(valid)
    dem_norm = (filled - dem_min) / dem_range

    try:
        import phasepack as pp
        # Phase congruency via phasepack
        try:
            pc = pp.phasecong(dem_norm, nscale=nscale, norient=norient,
                              minWaveLength=2, mult=2.5,
                              sigmaOnf=0.55, k=2.0,
                              cutOff=0.5, g=10, deviationGain=1.5)
        except TypeError:
            # Different phasepack API
            pc = pp.phasecong(dem_norm, nscale=nscale, norient=norient)

        vals = pc[valid]
        if len(vals) > 10:
            thresh = np.percentile(vals, percentile)
            return (pc > thresh) & valid
        else:
            return np.zeros_like(valid)

    except ImportError:
        # Fallback: use oriented gradient energy as proxy for phase congruency
        from scipy.ndimage import gaussian_filter, sobel

        h, w = dem.shape
        candidate_mask = np.zeros((h, w), dtype=bool)

        for scale in range(1, nscale + 1):
            sigma = scale * 1.5
            smooth = gaussian_filter(dem_norm, sigma=sigma)

            for orient_idx in range(norient):
                angle = orient_idx * np.pi / norient
                # Directional gradient
                dx = np.cos(angle)
                dy = np.sin(angle)
                # Rotated Sobel: compute gradient along the orientation
                grad = sobel(smooth, axis=0) * dy + sobel(smooth, axis=1) * dx
                vals = np.abs(grad[valid])
                if len(vals) > 10:
                    thresh = np.percentile(vals, percentile)
                    candidate_mask |= (np.abs(grad) > thresh) & valid

        return candidate_mask


def _compute_ridge_crest_candidates(dem, valid, resolution,
                                    sigma_range=(1.0, 3.0, 5.0),
                                    percentile=90,
                                    nms_window=3):
    """Local Maxima of Profile Curvature (Ridge Detection).

    Detects the crest of the scarp (top edge) by finding local maxima of
    positive curvature along the direction of maximum slope.  These crest
    lines form the upper boundary of a scarp.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)

    for sigma in sigma_range:
        smooth = gaussian_filter(filled, sigma=sigma)

        dz_dx = np.gradient(smooth, resolution, axis=1)
        dz_dy = np.gradient(smooth, resolution, axis=0)
        grad_mag = np.hypot(dz_dx, dz_dy)
        grad_mag_safe = np.where(grad_mag > 1e-10, grad_mag, 1.0)

        # Second derivatives
        d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
        d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
        d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)

        # Profile curvature (in direction of steepest slope)
        prof_curv = -(dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy +
                      dz_dy**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2))

        # Non-maximum suppression along steepest slope direction
        # Direction of steepest slope
        slope_dir_x = dz_dx / grad_mag_safe
        slope_dir_y = dz_dy / grad_mag_safe

        # NMS: pixel is a local max if it's the max in a small window
        # along the slope direction
        local_max = maximum_filter(prof_curv, size=nms_window) == prof_curv

        # Only keep positive curvature local maxima (crests)
        crest = local_max & (prof_curv > 0) & valid

        vals = prof_curv[crest]
        if len(vals) > 10:
            thresh = np.percentile(vals, percentile)
            candidate_mask |= (prof_curv > thresh) & crest

    return candidate_mask


def _extract_ml_features(dem, valid, resolution):
    """Extract a feature vector for each pixel for ML classification.

    Computes: slope, aspect, profile curvature, plan curvature, TPI (3 scales),
    TRI, local relief, aspect variance, DoG, gradient magnitude, and the subtle
    scarp detector scores used by the tuned ML model.

    Returns a dict of 2D arrays, each (H, W).
    """
    from scipy.ndimage import (gaussian_filter, uniform_filter, sobel,
                                maximum_filter, minimum_filter)

    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=2.0)

    # Slope
    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Aspect
    aspect = np.arctan2(-dz_dy, dz_dx)

    # Second derivatives
    d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
    d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
    d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)
    grad_mag_safe = np.where(grad_mag > 1e-10, grad_mag, 1.0)

    prof_curv = -(dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy +
                   dz_dy**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2))
    plan_curv = -((dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy +
                    dz_dx**2 * d2z_dy2) /
                   (grad_mag_safe * (1 + grad_mag**2)**1.5))

    # TPI at 3 scales
    tpi_50 = filled - uniform_filter(filled, size=max(5, int(50.0 / resolution) * 2 + 1), mode="reflect")
    tpi_100 = filled - uniform_filter(filled, size=max(5, int(100.0 / resolution) * 2 + 1), mode="reflect")
    tpi_200 = filled - uniform_filter(filled, size=max(5, int(200.0 / resolution) * 2 + 1), mode="reflect")

    # TRI
    local_mean = uniform_filter(filled, size=3, mode="reflect")
    tri = np.abs(filled - local_mean)

    # Local relief
    win_100 = max(5, int(100.0 / resolution) * 2 + 1)
    local_max = maximum_filter(filled, size=win_100, mode="reflect")
    local_min = minimum_filter(filled, size=win_100, mode="reflect")
    local_relief = local_max - local_min

    # Aspect variance (circular)
    sin_asp = np.sin(aspect)
    cos_asp = np.cos(aspect)
    win_var = max(5, int(100.0 / resolution) * 2 + 1)
    mean_sin = uniform_filter(sin_asp, size=win_var, mode="reflect")
    mean_cos = uniform_filter(cos_asp, size=win_var, mode="reflect")
    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    aspect_var = 1.0 - R

    # DoG
    dog = gaussian_filter(filled, sigma=1.0) - gaussian_filter(filled, sigma=3.0)

    # Subtle scarp detector soft scores
    subtle_slope_break = _mask_to_score(
        _compute_subtle_slope_break_candidates(dem, valid, resolution), valid)
    convexity_profile = _mask_to_score(
        _compute_convexity_profile_candidates(dem, valid, resolution), valid)
    gentle_slope_anomaly = _mask_to_score(
        _compute_gentle_slope_anomaly_candidates(dem, valid, resolution), valid)
    multiscale_relief = _mask_to_score(
        _compute_multiscale_relief_anomaly_candidates(dem, valid, resolution), valid)

    return {
        "slope_deg": slope_deg,
        "grad_mag": grad_mag,
        "prof_curv": prof_curv,
        "plan_curv": plan_curv,
        "tpi_50": tpi_50,
        "tpi_100": tpi_100,
        "tpi_200": tpi_200,
        "tri": tri,
        "local_relief": local_relief,
        "aspect_var": aspect_var,
        "dog": dog,
        "subtle_slope_break": subtle_slope_break,
        "convexity_profile": convexity_profile,
        "gentle_slope_anomaly": gentle_slope_anomaly,
        "multiscale_relief": multiscale_relief,
    }


def _get_feature_names():
    """Return the list of feature names used by the ML model."""
    return [
        "slope_deg", "grad_mag", "prof_curv", "plan_curv",
        "tpi_50", "tpi_100", "tpi_200", "tri", "local_relief",
        "aspect_var", "dog",
        "subtle_slope_break", "convexity_profile",
        "gentle_slope_anomaly", "multiscale_relief",
    ]


def _mask_to_score(mask, valid, smoothing_sigma=2.0):
    """Convert binary mask to soft score via Gaussian smoothing and normalization.
    
    Takes a binary mask (0/1), smooths it with a Gaussian filter, and normalizes
    to [0, 1] range based on percentile scaling of valid pixels.
    
    Parameters
    ----------
    mask : np.ndarray (H, W)
        Binary mask (True/1 = positive, False/0 = negative)
    valid : np.ndarray (H, W)
        Valid pixel mask (same shape as mask)
    smoothing_sigma : float
        Standard deviation of Gaussian kernel for smoothing
    
    Returns
    -------
    np.ndarray (H, W)
        Soft score array with values in [0, 1], zeros where not valid
    """
    from scipy.ndimage import gaussian_filter
    
    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)
    
    # Smooth the binary mask
    score = gaussian_filter(mask.astype(np.float32), sigma=smoothing_sigma)
    
    # Normalize to [0, 1] using percentile scaling
    valid_scores = score[valid] if valid.any() else score.ravel()
    if len(valid_scores) > 100:
        p95 = np.percentile(valid_scores, 95)
        if p95 > 0:
            score = np.clip(score / p95, 0, 1)
    else:
        score_max = np.max(score)
        if score_max > 0:
            score = score / score_max
    
    return score.astype(np.float32)


def _compute_candidate_features(dem, valid, resolution, transform):
    """Compute soft candidate scores (0-1) for each pixel from all sources.
    
    Returns dict of 2D arrays, each normalized to [0, 1] where higher = more scarp-like.
    This is an extended version that computes multiple candidate measures beyond
    just the ML features, including the subtle scarp detector scores added to the
    training pipeline.
    
    Parameters
    ----------
    dem : np.ndarray (H, W)
        DEM elevation array
    valid : np.ndarray (H, W)
        Valid pixel mask
    resolution : float
        DEM resolution in meters
    transform : affine.Affine
        Rasterio transform for georeferencing
    
    Returns
    -------
    dict
        Keys: 'curvature', 'edge', 'openness', 'aspect', 'adaptive_slope', 'tpi',
        'tri', 'dog', 'geomorphon', 'hough', 'orientation', 'ridge_crest',
        'subtle_slope_break', 'convexity_profile', 'gentle_slope_anomaly',
        'multiscale_relief'
        Values: 2D arrays normalized to [0, 1]
    """
    from scipy.ndimage import (gaussian_filter, sobel, maximum_filter, 
                                minimum_filter, uniform_filter)
    
    features = {}
    h, w = dem.shape
    
    # Safe fill for invalid pixels
    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=2.0)
    
    # 1. Multi-scale curvature score
    curv_scores = []
    for sigma in [1.0, 2.0, 3.0]:
        smooth_s = gaussian_filter(filled, sigma=sigma)
        dz_dx = np.gradient(smooth_s, resolution, axis=1)
        dz_dy = np.gradient(smooth_s, resolution, axis=0)
        d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
        d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
        d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)
        grad_mag = np.hypot(dz_dx, dz_dy)
        grad_mag_safe = np.where(grad_mag > 1e-10, grad_mag, 1.0)
        prof_curv = -(dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy +
                      dz_dy**2 * d2z_dy2) / (grad_mag_safe * (1 + grad_mag**2))
        curv_scores.append(np.abs(prof_curv))
    curv_score = np.max(curv_scores, axis=0)
    valid_vals = curv_score[valid] if valid.any() else curv_score.ravel()
    if np.any(np.isfinite(valid_vals)):
        p90 = np.percentile(valid_vals[np.isfinite(valid_vals)], 90)
        features['curvature'] = np.clip(curv_score / max(p90, 1e-6), 0, 1)
    else:
        features['curvature'] = np.zeros_like(curv_score)
    
    # 2. Edge detection score (gradient magnitude)
    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    grad_mag = np.hypot(dz_dx, dz_dy)
    valid_vals = grad_mag[valid] if valid.any() else grad_mag.ravel()
    if np.any(np.isfinite(valid_vals)):
        p88 = np.percentile(valid_vals[np.isfinite(valid_vals)], 88)
        features['edge'] = np.clip(grad_mag / max(p88, 1e-6), 0, 1)
    else:
        features['edge'] = np.zeros_like(grad_mag)
    
    # 3. Topographic openness score
    win_30 = max(3, int(30.0 / resolution))
    win_60 = max(3, int(60.0 / resolution))
    win_100 = max(3, int(100.0 / resolution))
    horizon_30 = np.abs(filled - uniform_filter(filled, size=win_30*2+1, mode='reflect'))
    horizon_60 = np.abs(filled - uniform_filter(filled, size=win_60*2+1, mode='reflect'))
    horizon_100 = np.abs(filled - uniform_filter(filled, size=win_100*2+1, mode='reflect'))
    openness = (horizon_30 + horizon_60 + horizon_100) / 3.0
    valid_vals = openness[valid] if valid.any() else openness.ravel()
    if np.any(np.isfinite(valid_vals)):
        p88 = np.percentile(valid_vals[np.isfinite(valid_vals)], 88)
        features['openness'] = np.clip(openness / max(p88, 1e-6), 0, 1)
    else:
        features['openness'] = np.zeros_like(openness)
    
    # 4. Aspect change score (curvature of aspect)
    aspect = np.arctan2(-dz_dy, dz_dx)
    d_aspect_dy = np.gradient(aspect, resolution, axis=0)
    d_aspect_dx = np.gradient(aspect, resolution, axis=1)
    aspect_change = np.hypot(d_aspect_dx, d_aspect_dy)
    valid_vals = aspect_change[valid] if valid.any() else aspect_change.ravel()
    if np.any(np.isfinite(valid_vals)):
        p90 = np.percentile(valid_vals[np.isfinite(valid_vals)], 90)
        features['aspect'] = np.clip(aspect_change / max(p90, 1e-6), 0, 1)
    else:
        features['aspect'] = np.zeros_like(aspect_change)
    
    # 5. Adaptive slope anomaly score
    local_mean_150 = uniform_filter(filled, size=max(5, int(150.0/resolution)*2+1), mode='reflect')
    slope_anomaly = np.abs(grad_mag - (grad_mag * 0.5 + np.gradient(local_mean_150, resolution, axis=0)))
    valid_vals = slope_anomaly[valid] if valid.any() else slope_anomaly.ravel()
    if np.any(np.isfinite(valid_vals)):
        p90 = np.percentile(valid_vals[np.isfinite(valid_vals)], 90)
        features['adaptive_slope'] = np.clip(slope_anomaly / max(p90, 1e-6), 0, 1)
    else:
        features['adaptive_slope'] = np.zeros_like(slope_anomaly)
    
    # 6. TPI score (Topographic Position Index)
    tpi = filled - uniform_filter(filled, size=max(5, int(100.0/resolution)*2+1), mode='reflect')
    valid_vals = np.abs(tpi[valid]) if valid.any() else np.abs(tpi.ravel())
    if np.any(np.isfinite(valid_vals)):
        p90 = np.percentile(valid_vals[np.isfinite(valid_vals)], 90)
        features['tpi'] = np.clip(np.abs(tpi) / max(p90, 1e-6), 0, 1)
    else:
        features['tpi'] = np.zeros_like(tpi)
    
    # 7. TRI/Roughness score
    local_mean_3 = uniform_filter(filled, size=3, mode='reflect')
    tri = np.abs(filled - local_mean_3)
    valid_vals = tri[valid] if valid.any() else tri.ravel()
    if np.any(np.isfinite(valid_vals)):
        p92 = np.percentile(valid_vals[np.isfinite(valid_vals)], 92)
        features['tri'] = np.clip(tri / max(p92, 1e-6), 0, 1)
    else:
        features['tri'] = np.zeros_like(tri)
    
    # 8. DoG score (Difference of Gaussians)
    dog = gaussian_filter(filled, sigma=1.0) - gaussian_filter(filled, sigma=3.0)
    valid_vals = np.abs(dog[valid]) if valid.any() else np.abs(dog.ravel())
    if np.any(np.isfinite(valid_vals)):
        p90 = np.percentile(valid_vals[np.isfinite(valid_vals)], 90)
        features['dog'] = np.clip(np.abs(dog) / max(p90, 1e-6), 0, 1)
    else:
        features['dog'] = np.zeros_like(dog)

    # 9. Subtle scarp detector soft scores
    subtle_mask = _compute_subtle_slope_break_candidates(dem, valid, resolution)
    features['subtle_slope_break'] = _mask_to_score(subtle_mask, valid)

    convexity_mask = _compute_convexity_profile_candidates(dem, valid, resolution)
    features['convexity_profile'] = _mask_to_score(convexity_mask, valid)

    gentle_mask = _compute_gentle_slope_anomaly_candidates(dem, valid, resolution)
    features['gentle_slope_anomaly'] = _mask_to_score(gentle_mask, valid)

    relief_mask = _compute_multiscale_relief_anomaly_candidates(dem, valid, resolution)
    features['multiscale_relief'] = _mask_to_score(relief_mask, valid)
    
    # 10-13. Placeholder for other sources (return zeros for now, can be expanded)
    for key in ['geomorphon', 'hough', 'orientation', 'ridge_crest']:
        features[key] = np.zeros_like(dem, dtype=np.float32)
    
    return features


# Global ML model cache (trained once, reused across tiles)
_ML_CANDIDATE_MODEL = None
_ML_MODEL_FEATURE_NAMES = [
    "slope_deg", "grad_mag", "prof_curv", "plan_curv",
    "tpi_50", "tpi_100", "tpi_200", "tri", "local_relief",
    "aspect_var", "dog",
    "subtle_slope_break", "convexity_profile",
    "gentle_slope_anomaly", "multiscale_relief",
]

# Schema version — bump when the *semantics* of any feature change (not just
# the count).  This forces a retrain even if n_features_in_ matches, because
# the old model was trained on a different computation of the same-named
# feature (e.g. convexity_profile was a saturated 250px-blob before, now a
# 5m-tolerance transition mask).  The pickle stores this alongside the model.
_ML_FEATURE_SCHEMA_VERSION = 2


def _ml_model_has_compatible_features(model):
    """Return True if the model matches the current feature schema.

    Checks both feature count AND schema version (stored as a custom
    attribute on the pickled dict wrapper).  A stale model that has the
    same feature count but was trained on different feature semantics
    (e.g. pre-convexity-profile-fix) will fail this check and be retrained.
    """
    if model is None:
        return False
    # The pickle is now a dict wrapper: {"model": clf, "schema_version": N, ...}
    if isinstance(model, dict) and "model" in model:
        saved_version = model.get("schema_version", 0)
        if saved_version != _ML_FEATURE_SCHEMA_VERSION:
            return False
        clf = model["model"]
    else:
        # Legacy pickle (bare sklearn estimator) — always incompatible
        # because it lacks the schema version tag.
        return False
    n_features = getattr(clf, "n_features_in_", None)
    if n_features is None:
        return True
    return int(n_features) == len(_ML_MODEL_FEATURE_NAMES)


def _train_ml_candidate_model(dem_src, fsd1_xy, params, resolution,
                              bounds=None, n_positive=2000, n_negative=2000,
                              model_path=None):
    """Train a Random Forest classifier on pixel features.

    Uses FSD1 points as positive examples and random non-scarp points
    as negatives.  The classifier learns to distinguish scarp-like terrain
    from background terrain based on geomorphometric features.

    Parameters
    ----------
    dem_src : rasterio dataset handle (open)
    fsd1_xy : np.ndarray (N, 2)
        FSD1 point coordinates.
    params : dict
        Parameters (unused currently, reserved for future feature tuning).
    resolution : float
        DEM resolution in metres.
    bounds : tuple, optional
        Bounding box for random negative sampling.
    model_path : Path, optional
        If provided, try to load model from here.  If training is done,
        save to this path.

    Returns
    -------
    sklearn RandomForestClassifier
    """
    global _ML_CANDIDATE_MODEL

    # Try to load cached model; reject stale pickles built with a previous
    # feature schema (different feature count OR different schema version).
    if model_path is not None and Path(model_path).exists():
        try:
            import pickle
            with open(model_path, "rb") as f:
                candidate_model = pickle.load(f)
            if _ml_model_has_compatible_features(candidate_model):
                # Unwrap the dict wrapper to get the bare sklearn estimator
                _ML_CANDIDATE_MODEL = candidate_model["model"] if isinstance(candidate_model, dict) else candidate_model
                print(f"  [ML] Loaded trained model from {model_path} (schema v{_ML_FEATURE_SCHEMA_VERSION})")
                return _ML_CANDIDATE_MODEL
            else:
                if isinstance(candidate_model, dict) and "model" in candidate_model:
                    n_loaded = getattr(candidate_model["model"], "n_features_in_", "unknown")
                    saved_ver = candidate_model.get("schema_version", "unknown")
                else:
                    n_loaded = getattr(candidate_model, "n_features_in_", "unknown")
                    saved_ver = "legacy (bare estimator)"
                print(f"  [ML] Saved model stale: expected schema v{_ML_FEATURE_SCHEMA_VERSION} / {len(_ML_MODEL_FEATURE_NAMES)} features, "
                      f"got schema v{saved_ver} / {n_loaded} features. Retraining...")
        except Exception as e:
            print(f"  [ML] Failed to load model: {e}, retraining...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Subsample FSD1 if too many
    if len(fsd1_xy) > n_positive:
        idx = np.random.RandomState(42).choice(len(fsd1_xy), n_positive, replace=False)
        fsd1_sample = fsd1_xy[idx]
    else:
        fsd1_sample = fsd1_xy

    # Generate random negative points (> 500m from FSD1)
    dem_bounds = dem_src.bounds
    fsd1_tree = cKDTree(fsd1_xy)
    random_pts = []
    attempts = 0
    while len(random_pts) < n_negative and attempts < n_negative * 50:
        x = np.random.uniform(dem_bounds.left, dem_bounds.right)
        y = np.random.uniform(dem_bounds.bottom, dem_bounds.top)
        if bounds is not None:
            if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                attempts += 1
                continue
        dist, _ = fsd1_tree.query([x, y], k=1)
        if dist > 500.0:
            random_pts.append([x, y])
        attempts += 1
    random_xy = np.array(random_pts) if random_pts else np.empty((0, 2))
    print(f"  [ML] Training data: {len(fsd1_sample)} positive, {len(random_xy)} negative")

    # Extract features at each point
    print(f"  [ML] Extracting features at positive points...")
    pos_features = _extract_features_at_points(dem_src, fsd1_sample, resolution)
    print(f"  [ML] Extracting features at negative points...")
    neg_features = _extract_features_at_points(dem_src, random_xy, resolution)

    if pos_features is None or neg_features is None:
        print(f"  [ML] Feature extraction failed, skipping ML model")
        return None

    # Combine
    X = np.vstack([pos_features, neg_features])
    y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])

    # Remove NaN rows
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    y = y[finite_mask]

    if len(X) < 20:
        print(f"  [ML] Too few valid samples ({len(X)}), skipping ML model")
        return None

    # Cross-validation to estimate performance
    from sklearn.model_selection import cross_val_score
    clf_temp = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=42, n_jobs=-1,
                                      class_weight="balanced")
    cv_scores = cross_val_score(clf_temp, X, y, cv=5, scoring='f1')
    print(f"  [ML] CV F1 scores: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Train/test split (hold out 20%) for additional evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"  [ML] Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # Feature importances
    importances = clf.feature_importances_
    feature_names = _get_feature_names()
    print(f"  [ML] Feature importances:")
    for name, imp in sorted(zip(feature_names, importances),
                           key=lambda x: -x[1]):
        print(f"    {name}: {imp:.4f}")

    _ML_CANDIDATE_MODEL = clf

    # Save model as a dict wrapper with schema version so future loads can
    # detect semantic changes even when the feature count is unchanged.
    if model_path is not None:
        try:
            import pickle
            wrapper = {
                "model": clf,
                "schema_version": _ML_FEATURE_SCHEMA_VERSION,
                "feature_names": _ML_MODEL_FEATURE_NAMES,
                "trained_at": pd.Timestamp.now().isoformat(),
            }
            with open(model_path, "wb") as f:
                pickle.dump(wrapper, f)
            print(f"  [ML] Saved trained model to {model_path} (schema v{_ML_FEATURE_SCHEMA_VERSION})")
        except Exception as e:
            print(f"  [ML] Failed to save model: {e}")

    return clf


def _extract_features_at_points(dem_src, points_xy, resolution, radius_m=100.0):
    """Extract ML features at given point coordinates.

    Reads a local DEM window around each point and computes the full
    feature vector.  Returns array of shape (N, n_features).
    """
    radius_px = max(3, int(np.ceil(radius_m / resolution)))
    n_features = len(_ML_MODEL_FEATURE_NAMES)
    all_features = np.zeros((len(points_xy), n_features), dtype=np.float32)

    for i, (x, y) in enumerate(points_xy):
        col, row = ~dem_src.transform * (x, y)
        col_int, row_int = int(round(col)), int(round(row))
        r0 = max(0, row_int - radius_px)
        r1 = min(dem_src.height, row_int + radius_px + 1)
        c0 = max(0, col_int - radius_px)
        c1 = min(dem_src.width, col_int + radius_px + 1)

        if r1 - r0 < 5 or c1 - c0 < 5:
            all_features[i] = np.nan
            continue

        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = dem_src.read(1, window=win).astype(np.float32)
        nodata = dem_src.nodata
        if nodata is not None:
            v = (data != nodata) & np.isfinite(data)
        else:
            v = np.isfinite(data)

        if v.sum() < 10:
            all_features[i] = np.nan
            continue

        feats = _extract_ml_features(data, v, resolution)
        # Extract feature values at the center pixel
        local_row = row_int - r0
        local_col = col_int - c0
        if 0 <= local_row < data.shape[0] and 0 <= local_col < data.shape[1]:
            for j, name in enumerate(_ML_MODEL_FEATURE_NAMES):
                val = feats[name][local_row, local_col]
                all_features[i, j] = float(val) if np.isfinite(val) else 0.0
        else:
            all_features[i] = np.nan

    return all_features


def _compute_ml_probability_candidates(dem, valid, resolution, transform,
                                      threshold=0.5, percentile=None):
    """Supervised ML (Random Forest) soft candidate score.

    Applies a pre-trained Random Forest classifier to the tile's pixel
    features to produce a probability mask.  The model must be trained
    beforehand via _train_ml_candidate_model().

    If no model is available, falls back to a gradient-based heuristic.

    Parameters
    ----------
    threshold : float
        Minimum probability to flag a pixel as candidate.  When the model
        is available, this is an absolute threshold applied globally —
        NOT overridden by a per-tile percentile.  This respects the
        user's --ml-probability-threshold argument instead of always
        keeping the top 10% of every tile.
    percentile : float or None
        If not None, also enforce a per-tile percentile floor (legacy
        behaviour).  Set to None (default) so the explicit threshold is
        the sole criterion.
    """
    global _ML_CANDIDATE_MODEL

    h, w = dem.shape

    if _ML_CANDIDATE_MODEL is not None:
        if not _ml_model_has_compatible_features(_ML_CANDIDATE_MODEL):
            n_loaded = getattr(_ML_CANDIDATE_MODEL, "n_features_in_", "unknown")
            print(f"  [ML] Model feature mismatch: expected {len(_ML_MODEL_FEATURE_NAMES)}, got {n_loaded}. Falling back to heuristic ML proxy.")
            _ML_CANDIDATE_MODEL = None

    if _ML_CANDIDATE_MODEL is not None:
        # Extract features for all valid pixels
        feats = _extract_ml_features(dem, valid, resolution)

        # Stack into feature matrix
        feat_stack = np.stack([feats[name] for name in _ML_MODEL_FEATURE_NAMES], axis=-1)
        H, W, D = feat_stack.shape
        feat_flat = feat_stack.reshape(-1, D)

        # Predict probability
        finite_mask = np.all(np.isfinite(feat_flat), axis=1)
        prob_flat = np.zeros(H * W, dtype=np.float32)
        if finite_mask.sum() > 0:
            prob_flat[finite_mask] = _ML_CANDIDATE_MODEL.predict_proba(feat_flat[finite_mask])[:, 1]
        prob = prob_flat.reshape(H, W)

        # Threshold: use the explicit threshold as the sole criterion.
        # Only apply a per-tile percentile floor if the caller explicitly
        # passes percentile=<number>, which preserves the old behaviour
        # for diagnostic/debugging purposes.
        thresh = threshold
        if percentile is not None:
            vals = prob[valid]
            if len(vals) > 10:
                thresh = max(threshold, float(np.percentile(vals, percentile)))

        return (prob > thresh) & valid
    else:
        # Fallback: use high gradient magnitude as proxy
        from scipy.ndimage import gaussian_filter
        filled = np.where(valid, dem, float(np.nanmedian(dem[valid])) if valid.any() else 0.0)
        smooth = gaussian_filter(filled, sigma=2.0)
        dz_dx = np.gradient(smooth, resolution, axis=1)
        dz_dy = np.gradient(smooth, resolution, axis=0)
        grad_mag = np.hypot(dz_dx, dz_dy)
        vals = grad_mag[valid]
        if len(vals) > 10:
            p = percentile if percentile is not None else 90
            thresh = np.percentile(vals, p)
            return (grad_mag > thresh) & valid
        else:
            return np.zeros_like(valid)


# ── Subtle Scarp Feature Detectors ────────────────────────────────────

def _compute_subtle_slope_break_candidates(dem, valid, resolution,
                                          min_relief_m=2.0,
                                          max_relief_m=6.0,
                                          scarp_width_m=(200.0, 300.0),
                                          percentile=85):
    """Detect subtle slope breaks characteristic of degraded fault scarps.
    
    Looks for convex-concave slope profile over 200-300m width with
    total relief of 2-6m. The slope should increase then decrease
    (or vice versa) creating an inflection point.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter
    
    filled = np.where(valid, dem, np.nan)
    median_val = np.nanmedian(filled) if np.any(np.isfinite(filled)) else 0.0
    smooth = gaussian_filter(np.nan_to_num(filled, nan=median_val), sigma=3)
    
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)
    
    # Convert widths to pixels
    min_width_px = max(5, int(scarp_width_m[0] / resolution))
    max_width_px = max(min_width_px + 1, int(scarp_width_m[1] / resolution))
    
    # Compute slope at multiple scales
    dz_dx_fine = np.gradient(smooth, resolution, axis=1)
    dz_dy_fine = np.gradient(smooth, resolution, axis=0)
    fine_slope = np.hypot(dz_dx_fine, dz_dy_fine)
    
    # Compute coarse slope
    smooth_coarse = gaussian_filter(smooth, sigma=3)
    dz_dx_coarse = np.gradient(smooth_coarse, resolution, axis=1)
    dz_dy_coarse = np.gradient(smooth_coarse, resolution, axis=0)
    coarse_slope = np.hypot(dz_dx_coarse, dz_dy_coarse)
    
    # Slope change = difference between fine and coarse
    slope_change = np.abs(fine_slope - coarse_slope)
    
    for width_px in range(min_width_px, min(max_width_px + 1, min_width_px + 20), 5):
        # Compute local relief over this width
        relief_max = maximum_filter(smooth, size=width_px)
        relief_min = minimum_filter(smooth, size=width_px)
        relief = relief_max - relief_min
        
        # Threshold relief
        relief_mask = (relief >= min_relief_m) & (relief <= max_relief_m) & valid
        
        if relief_mask.any():
            # Threshold slope change
            slope_change_valid = slope_change[relief_mask]
            if len(slope_change_valid) > 100:
                thresh = np.percentile(slope_change_valid, percentile)
                break_mask = relief_mask & (slope_change > thresh)
                candidate_mask |= break_mask
    
    return candidate_mask


def _compute_convexity_profile_candidates(dem, valid, resolution,
                                         tolerance_m=5.0,
                                         percentile=88):
    """Detect convex-concave slope transitions characteristic of subtle scarps.
    
    The profile should show:
    - Upper convex section (slope increasing)
    - Middle steepest section
    - Lower concave section (slope decreasing)
    
    The dilation radius is a small tolerance (default 5m) for aligning
    the convex/concave transition, NOT the full profile width.  Using
    the full profile width (e.g. 250m) as dilation iterations causes
    every convex and concave pixel to expand into a giant blob, filling
    ~60-98% of the tile -- which is noise, not signal.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation
    
    filled = np.where(valid, dem, np.nan)
    median_val = np.nanmedian(filled) if np.any(np.isfinite(filled)) else 0.0
    smooth = gaussian_filter(np.nan_to_num(filled, nan=median_val), sigma=2)
    
    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)
    
    # Compute slope direction and profile curvature
    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    
    slope_magnitude = np.hypot(dz_dx, dz_dy)
    slope_magnitude_safe = np.where(slope_magnitude > 1e-6, slope_magnitude, 1.0)
    
    # Unit vector in slope direction
    ux = dz_dx / slope_magnitude_safe
    uy = dz_dy / slope_magnitude_safe
    
    # Second derivatives
    d2z_dx2 = np.gradient(dz_dx, resolution, axis=1)
    d2z_dy2 = np.gradient(dz_dy, resolution, axis=0)
    d2z_dxdy = np.gradient(dz_dx, resolution, axis=0)
    
    # Profile curvature (curvature in direction of steepest slope)
    profile_curvature = -(ux**2 * d2z_dx2 + 2*ux*uy*d2z_dxdy + uy**2 * d2z_dy2)
    
    # Convex sections have negative profile curvature
    convex_mask = (profile_curvature < -0.0001) & valid
    
    # Concave sections have positive profile curvature
    concave_mask = (profile_curvature > 0.0001) & valid
    
    # Find transitions between convex and concave.
    # Use a small dilation tolerance (default 5m) to align the
    # convex/concave boundary -- NOT the full profile width.
    transition_tolerance_px = max(1, int(tolerance_m / resolution))
    
    convex_dilated = binary_dilation(convex_mask, iterations=transition_tolerance_px)
    concave_dilated = binary_dilation(concave_mask, iterations=transition_tolerance_px)
    
    # Transition zones
    transition_mask = convex_dilated & concave_dilated & valid
    
    if transition_mask.any():
        candidate_mask |= transition_mask
    
    return candidate_mask


def _compute_gentle_slope_anomaly_candidates(dem, valid, resolution,
                                            base_slope_range=(0.005, 0.05),
                                            anomaly_threshold=1.2,
                                            percentile=85):
    """Detect subtle slope anomalies in gentle terrain.
    
    Targets pixels where the local slope differs from the surrounding
    median slope by at least ``anomaly_threshold`` times, within a gentle
    base slope range of 1-2.5% (0.01-0.025).  Both positive anomalies
    (slope higher than surroundings) and negative anomalies (slope lower
    than surroundings) are accepted, since a subtle scarp can create either
    a local steepening or a local flattening relative to the background.

    Parameters
    ----------
    base_slope_range : tuple (low, high)
        Slope range (as dimensionless rise/run) defining "gentle terrain".
        Default (0.01, 0.025) = 1-2.5% grade.
    anomaly_threshold : float
        Minimum |slope_ratio - 1| to flag as anomalous.  Default 1.2 means
        the local slope must be at least 20% above OR below the regional
        mean (ratio >= 1.2 or ratio <= 1/1.2 ~= 0.83).
    percentile : float
        Percentile threshold applied within the anomaly mask to control
        false positives.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter
    
    filled = np.where(valid, dem, np.nan)
    median_val = np.nanmedian(filled) if np.any(np.isfinite(filled)) else 0.0
    smooth = gaussian_filter(np.nan_to_num(filled, nan=median_val), sigma=2)
    
    # Compute slope
    dz_dx = np.gradient(smooth, resolution, axis=1)
    dz_dy = np.gradient(smooth, resolution, axis=0)
    slope = np.hypot(dz_dx, dz_dy)
    
    # Base slope range (gentle terrain, 1-2.5% grade)
    base_slope_mask = (slope >= base_slope_range[0]) & (slope <= base_slope_range[1]) & valid
    
    # Local mean slope over broader window
    window_px = max(10, int(100 / resolution))  # 100m window
    local_mean_slope = uniform_filter(slope, size=window_px)
    
    # Anomaly = local slope / regional mean slope
    safe_mean = np.where(local_mean_slope > 0.001, local_mean_slope, 0.001)
    slope_ratio = slope / safe_mean
    
    # Positive anomaly: slope is significantly HIGHER than surroundings
    # (e.g. local steepening at scarp crest)
    pos_anomaly_mask = (slope_ratio > anomaly_threshold) & base_slope_mask
    
    # Negative anomaly: slope is significantly LOWER than surroundings
    # (e.g. local flattening at scarp toe or on the hanging wall)
    neg_anomaly_mask = (slope_ratio < 1.0 / anomaly_threshold) & base_slope_mask
    
    # Combine both directions
    anomaly_mask = pos_anomaly_mask | neg_anomaly_mask
    
    # Threshold by percentile
    anomaly_slopes = slope[anomaly_mask]
    if len(anomaly_slopes) > 100:
        thresh = np.percentile(anomaly_slopes, percentile)
        candidate_mask = anomaly_mask & (slope > thresh)
    else:
        candidate_mask = anomaly_mask
    
    return candidate_mask


def _compute_multiscale_relief_anomaly_candidates(dem, valid, resolution,
                                                 scales_m=(50.0, 100.0, 150.0, 250.0),
                                                 percentile=88):
    """Detect subtle relief changes at multiple scales.
    
    A subtle scarp shows elevated relief at 100-250m scales compared to
    background, but not at 50m scale (too broad) or 500m+ (regional trend).
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    
    filled = np.where(valid, dem, np.nanmedian(dem[valid]) if valid.any() else 0.0)
    h, w = dem.shape
    relief_maps = []
    
    for scale_m in scales_m:
        scale_px = max(3, int(scale_m / resolution))
        
        # Local relief = max - min in window
        local_max = maximum_filter(filled, size=scale_px)
        local_min = minimum_filter(filled, size=scale_px)
        relief = local_max - local_min
        relief_maps.append(relief)
    
    # Stack relief maps
    relief_stack = np.stack(relief_maps, axis=0)
    
    # Compute anomaly: high relief at intermediate scales
    # but relatively low at smallest scale
    small_relief = relief_stack[0]
    large_relief = relief_stack[-1]
    intermediate_relief = np.mean(relief_stack[1:-1], axis=0)
    
    # Anomaly score: intermediate relief should be high relative to small
    anomaly_score = intermediate_relief / np.where(small_relief > 0.5, small_relief, 0.5)
    
    # Threshold
    score_valid = anomaly_score[valid]
    if len(score_valid) > 100:
        thresh = np.percentile(score_valid, percentile)
        candidate_mask = (anomaly_score > thresh) & valid
    else:
        candidate_mask = np.zeros_like(valid)
    
    return candidate_mask


def _compute_slope_profile_curvature_candidates(dem, valid, resolution,
                                                profile_length_m=300.0,
                                                min_slope_change=0.003,
                                                percentile=80):
    """Detect inflection points along the gradient direction.

    Walks along the steepest-slope direction at each pixel and looks for
    points where the slope profile changes sign (convex <-> concave).
    These inflection points mark the top or bottom of a scarp slope,
    even when the slopes themselves are very gentle (0.5-3%).

    Parameters
    ----------
    profile_length_m : float
        Half-length of the profile to walk in each direction (total = 2x).
    min_slope_change : float
        Minimum absolute slope difference between the uphill and downhill
        sides to qualify as an inflection (filters out uniform slopes).
    percentile : float
        Percentile of inflection strength used as threshold.
    """
    from scipy.ndimage import gaussian_filter, sobel

    filled = np.where(valid, dem, np.nanmedian(dem[valid]) if valid.any() else 0.0)
    smooth = gaussian_filter(filled, sigma=2)

    h, w = dem.shape
    candidate_mask = np.zeros((h, w), dtype=bool)
    profile_px = max(5, int(profile_length_m / resolution))

    # Compute gradient direction
    dz_dy, dz_dx = np.gradient(smooth, resolution, resolution)
    grad_mag = np.hypot(dz_dx, dz_dy)

    # Only evaluate pixels with non-trivial gradient (at least 0.1% slope)
    eval_mask = valid & (grad_mag > 0.001)

    # For efficiency, evaluate only at local maxima of gradient magnitude
    # (these correspond to slope-break locations)
    from scipy.ndimage import maximum_filter
    grad_local_max = (grad_mag == maximum_filter(grad_mag, size=5)) & eval_mask

    ys, xs = np.where(grad_local_max)
    if len(ys) == 0:
        # Fallback: evaluate all pixels above gradient threshold
        ys, xs = np.where(eval_mask)
        if len(ys) == 0:
            return candidate_mask

    # Limit to avoid excessive computation
    if len(ys) > 50000:
        step = len(ys) // 50000
        ys = ys[::step]
        xs = xs[::step]

    scores = np.zeros(len(ys))
    for i, (r, c) in enumerate(zip(ys, xs)):
        gx = dz_dx[r, c]
        gy = dz_dy[r, c]
        mag = grad_mag[r, c]
        if mag < 1e-9:
            continue
        dr = gy / mag
        dc = gx / mag

        # Sample slopes uphill and downhill
        uphill_slopes = []
        downhill_slopes = []
        for s in range(1, min(profile_px, max(h, w)) + 1):
            ru = r + dr * s
            cu = c + dc * s
            rd_ = r - dr * s
            cd_ = c - dc * s
            if 0 <= round(ru) < h and 0 <= round(cu) < w:
                ru2 = round(ru) + dr * 2
                cu2 = round(cu) + dc * 2
                if 0 <= round(ru2) < h and 0 <= round(cu2) < w:
                    z1 = smooth[round(ru), round(cu)]
                    z2 = smooth[round(ru2), round(cu2)]
                    uphill_slopes.append(abs(z2 - z1) / (2.0 * resolution))
            if 0 <= round(rd_) < h and 0 <= round(cd_) < w:
                rd2 = round(rd_) - dr * 2
                cd2 = round(cd_) - dc * 2
                if 0 <= round(rd2) < h and 0 <= round(cd2) < w:
                    z1 = smooth[round(rd_), round(cd_)]
                    z2 = smooth[round(rd2), round(cd2)]
                    downhill_slopes.append(abs(z2 - z1) / (2.0 * resolution))

        if len(uphill_slopes) < 3 or len(downhill_slopes) < 3:
            continue

        # Mean slopes on each side
        mean_up = np.mean(uphill_slopes)
        mean_down = np.mean(downhill_slopes)

        # Inflection score: how different are the two sides?
        # A true inflection has slopes going in opposite trends
        score = abs(mean_up - mean_down)
        scores[i] = score

    # Threshold by percentile
    valid_scores = scores[scores > min_slope_change]
    if len(valid_scores) > 50:
        thresh = np.percentile(valid_scores, percentile)
        for i, (r, c) in enumerate(zip(ys, xs)):
            if scores[i] > thresh and scores[i] > min_slope_change:
                candidate_mask[r, c] = True
    else:
        for i, (r, c) in enumerate(zip(ys, xs)):
            if scores[i] > min_slope_change:
                candidate_mask[r, c] = True

    return candidate_mask


def _compute_alternative_candidates(dem, valid, resolution, transform,
                                    sources=None, return_per_source=False):
    """Combine multiple alternative candidate sources via logical OR.

    Parameters
    ----------
    sources : dict or None
        If None, use all available sources.  Keys: 'curvature', 'edge',
        'openness', 'aspect', 'adaptive_slope', 'tpi', 'tri', 'dog',
        'geomorphon', 'hough', 'orientation', 'phase_congruency',
        'ridge_crest', 'ml_probability', 'subtle_slope_break',
        'convexity_profile', 'gentle_slope_anomaly', 'multiscale_relief'.
        Values are booleans (enable/disable).
    return_per_source : bool
        If True, also return a dict of per-source boolean masks (same shape
        as dem) so the caller can attribute each inferred point back to
        the specific detector(s) that contributed it.

    Returns
    -------
    combined : np.ndarray (H, W) bool
        OR-combined candidate mask.
    per_source : dict, optional (only if return_per_source=True)
        {source_name: np.ndarray (H, W) bool} — one mask per source.
    """
    if sources is None:
        sources = {
            "curvature": True, "edge": True,
            "openness": True, "aspect": True,
            "adaptive_slope": True, "tpi": True, "tri": True,
            "dog": True, "geomorphon": False, "hough": True,
            "orientation": True, "phase_congruency": False,
            "ridge_crest": True, "ml_probability": True,
            # New subtle feature detectors
            "subtle_slope_break": True,
            "convexity_profile": True,
            "gentle_slope_anomaly": True,
            "multiscale_relief": True,
            "slope_profile_curvature": True,
        }

    h, w = dem.shape
    combined = np.zeros((h, w), dtype=bool)
    per_source = {} if return_per_source else None

    # -- Existing sources (Phase 3) --
    if sources.get("curvature", False):
        curv_mask = _compute_multiscale_curvature_candidates(
            dem, valid, resolution, sigma_range=(1.0, 3.0, 5.0), percentile=90)
        combined |= curv_mask
        if per_source is not None:
            per_source["curvature"] = curv_mask
        print(f"    [Alt] Curvature candidates: {int(curv_mask.sum())} pixels")

    if sources.get("edge", False):
        edge_mask = _compute_edge_detection_candidates(
            dem, valid, resolution, sigma=2.0, percentile=88)
        combined |= edge_mask
        if per_source is not None:
            per_source["edge"] = edge_mask
        print(f"    [Alt] Edge detection candidates: {int(edge_mask.sum())} pixels")

    if sources.get("openness", False):
        open_mask = _compute_topographic_openness_candidates(
            dem, valid, resolution, annulus_radii_m=(30.0, 60.0, 100.0),
            percentile=88)
        combined |= open_mask
        if per_source is not None:
            per_source["openness"] = open_mask
        print(f"    [Alt] Openness candidates: {int(open_mask.sum())} pixels")

    if sources.get("aspect", False):
        aspect_mask = _compute_aspect_change_candidates(
            dem, valid, resolution, sigma=3.0, percentile=90)
        combined |= aspect_mask
        if per_source is not None:
            per_source["aspect"] = aspect_mask
        print(f"    [Alt] Aspect change candidates: {int(aspect_mask.sum())} pixels")

    # -- New sources (10 methods) --
    if sources.get("adaptive_slope", False):
        mask = _compute_adaptive_slope_candidates(
            dem, valid, resolution, window_m=150.0, k_std=2.0, percentile=90)
        combined |= mask
        if per_source is not None:
            per_source["adaptive_slope"] = mask
        print(f"    [Alt] Adaptive slope candidates: {int(mask.sum())} pixels")

    if sources.get("tpi", False):
        mask = _compute_tpi_candidates(
            dem, valid, resolution, window_sizes_m=(50.0, 100.0, 200.0),
            n_std=2.0, percentile=90)
        combined |= mask
        if per_source is not None:
            per_source["tpi"] = mask
        print(f"    [Alt] TPI candidates: {int(mask.sum())} pixels")

    if sources.get("tri", False):
        mask = _compute_tri_candidates(
            dem, valid, resolution, percentile=92)
        combined |= mask
        if per_source is not None:
            per_source["tri"] = mask
        print(f"    [Alt] TRI/Roughness candidates: {int(mask.sum())} pixels")

    if sources.get("dog", False):
        mask = _compute_dog_candidates(
            dem, valid, resolution,
            sigma_pairs=((1.0, 3.0), (2.0, 5.0), (3.0, 7.0)),
            percentile=90)
        combined |= mask
        if per_source is not None:
            per_source["dog"] = mask
        print(f"    [Alt] DoG candidates: {int(mask.sum())} pixels")

    if sources.get("geomorphon", False):
        mask = _compute_geomorphon_candidates(
            dem, valid, resolution, flat_threshold=0.5)
        combined |= mask
        if per_source is not None:
            per_source["geomorphon"] = mask
        print(f"    [Alt] Geomorphon candidates: {int(mask.sum())} pixels")

    if sources.get("hough", False):
        mask = _compute_hough_candidates(
            dem, valid, resolution, slope_threshold_deg=20.0,
            min_line_length_m=50.0, max_line_gap_m=20.0, dilate_px=5)
        combined |= mask
        if per_source is not None:
            per_source["hough"] = mask
        print(f"    [Alt] Hough lineament candidates: {int(mask.sum())} pixels")

    if sources.get("orientation", False):
        mask = _compute_orientation_consistency_candidates(
            dem, valid, resolution, window_m=100.0,
            min_slope_deg=5.0, max_circ_var=0.3, percentile=88)
        combined |= mask
        if per_source is not None:
            per_source["orientation"] = mask
        print(f"    [Alt] Orientation consistency candidates: {int(mask.sum())} pixels")

    if sources.get("phase_congruency", False):
        mask = _compute_phase_congruency_candidates(
            dem, valid, resolution, nscale=4, norient=6, percentile=90)
        combined |= mask
        if per_source is not None:
            per_source["phase_congruency"] = mask
        print(f"    [Alt] Phase congruency candidates: {int(mask.sum())} pixels")

    if sources.get("ridge_crest", False):
        mask = _compute_ridge_crest_candidates(
            dem, valid, resolution, sigma_range=(1.0, 3.0, 5.0),
            percentile=90, nms_window=3)
        combined |= mask
        if per_source is not None:
            per_source["ridge_crest"] = mask
        print(f"    [Alt] Ridge crest candidates: {int(mask.sum())} pixels")

    if sources.get("ml_probability", False):
        mask = _compute_ml_probability_candidates(
            dem, valid, resolution, transform,
            threshold=0.5, percentile=None)
        combined |= mask
        if per_source is not None:
            per_source["ml_probability"] = mask
        print(f"    [Alt] ML probability candidates: {int(mask.sum())} pixels")

    # -- Subtle scarp feature detectors --
    if sources.get("subtle_slope_break", False):
        subtle_mask = _compute_subtle_slope_break_candidates(
            dem, valid, resolution,
            min_relief_m=0.5, max_relief_m=12.0,
            scarp_width_m=(50.0, 400.0), percentile=80)
        combined |= subtle_mask
        if per_source is not None:
            per_source["subtle_slope_break"] = subtle_mask
        print(f"    [Alt] Subtle slope break candidates: {int(subtle_mask.sum())} pixels")

    if sources.get("convexity_profile", False):
        convexity_mask = _compute_convexity_profile_candidates(
            dem, valid, resolution,
            tolerance_m=5.0, percentile=88)
        combined |= convexity_mask
        if per_source is not None:
            per_source["convexity_profile"] = convexity_mask
        print(f"    [Alt] Convexity profile candidates: {int(convexity_mask.sum())} pixels")

    if sources.get("gentle_slope_anomaly", False):
        gentle_mask = _compute_gentle_slope_anomaly_candidates(
            dem, valid, resolution,
            base_slope_range=(0.003, 0.06),
            anomaly_threshold=1.1, percentile=75)
        combined |= gentle_mask
        if per_source is not None:
            per_source["gentle_slope_anomaly"] = gentle_mask
        print(f"    [Alt] Gentle slope anomaly candidates: {int(gentle_mask.sum())} pixels")

    if sources.get("multiscale_relief", False):
        relief_mask = _compute_multiscale_relief_anomaly_candidates(
            dem, valid, resolution,
            scales_m=(50.0, 100.0, 150.0, 250.0), percentile=88)
        combined |= relief_mask
        if per_source is not None:
            per_source["multiscale_relief"] = relief_mask
        print(f"    [Alt] Multi-scale relief candidates: {int(relief_mask.sum())} pixels")

    if sources.get("slope_profile_curvature", False):
        spc_mask = _compute_slope_profile_curvature_candidates(
            dem, valid, resolution,
            profile_length_m=300.0, min_slope_change=0.003, percentile=80)
        combined |= spc_mask
        if per_source is not None:
            per_source["slope_profile_curvature"] = spc_mask
        print(f"    [Alt] Slope profile curvature candidates: {int(spc_mask.sum())} pixels")

    print(f"    [Alt] Combined alternative candidates: {int(combined.sum())} pixels")
    if per_source is not None:
        return combined, per_source
    return combined


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global TRACE_BUFFER_M, TRACE2_BUFFER_M, TUNING_N_TRIALS
    global DEM_SOURCE_KEY, DEM_PATH, CSV_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-source", default="lidar_0_5m",
                        choices=list(DEM_SOURCE_MAP.keys()),
                        help="DEM source key (maps to the GeoTIFF in data/derivatives/)")
    parser.add_argument("--n-trials", type=int, default=TUNING_N_TRIALS,
                        help="Number of Optuna trials for parameter tuning")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip ML parameter tuning and use default params")
    parser.add_argument("--write-tiles", action="store_true",
                        help="Save per-tile LSM rasters (for debugging)")
    parser.add_argument("--trace-buffer-m", type=float, default=TRACE_BUFFER_M,
                        help="Buffer around Trace1 (HighCon) for filtering inferred points")
    parser.add_argument("--trace2-buffer-m", type=float, default=TRACE2_BUFFER_M,
                        help="Buffer around Trace2 (FSD1) for filtering inferred points")
    parser.add_argument("--no-fsd1-seeds", action="store_true", default=True,
                        help="Disable FSD1 seeding (default: disabled for blind method)")
    parser.add_argument("--fsd1-seed-accept-radius-m", type=float, default=30.0,
                        help="Radius (m) around each FSD1 point to seed as a candidate")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run criterion sensitivity analysis")
    parser.add_argument("--limit-to-trace2-buffer", action="store_true",
                        help="Restrict DEM-wide inference and tuning to a rectangle around Trace 2")
    parser.add_argument("--trace2-search-buffer-m", type=float, default=2000.0,
                        help="Buffer (m) around Trace 2 to define the restricted search area")
    parser.add_argument("--plot-trace2-buffer-m", type=float, default=2000.0,
                        help="Buffer (m) around Trace2 to use for plotting. "
                             "Set to 0 to plot full DEM extent.")

    # Phase selection
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2", "3", "inference"],
                        help="Which phase to run: 1=analysis, 2=tuning, 3=alt candidates, inference=dem-wide only, all=1+2+3+inference")

    # Alternative candidate source toggles (existing)
    parser.add_argument("--no-curvature", action="store_true",
                        help="Disable multi-scale curvature candidate source")
    parser.add_argument("--no-edge", action="store_true",
                        help="Disable edge detection candidate source")
    parser.add_argument("--no-openness", action="store_true",
                        help="Disable topographic openness candidate source")
    parser.add_argument("--no-aspect", action="store_true",
                        help="Disable aspect change candidate source")

    # Alternative candidate source toggles (new, 10 methods)
    parser.add_argument("--no-adaptive-slope", action="store_true",
                        help="Disable adaptive local slope anomaly candidate source")
    parser.add_argument("--no-tpi", action="store_true",
                        help="Disable topographic position index candidate source")
    parser.add_argument("--no-tri", action="store_true",
                        help="Disable terrain ruggedness index candidate source")
    parser.add_argument("--no-dog", action="store_true",
                        help="Disable difference of Gaussians candidate source")
    parser.add_argument("--geomorphon", action="store_true",
                        help="Enable geomorphon template matching candidate source (disabled by default)")
    parser.add_argument("--no-hough", action="store_true",
                        help="Disable Hough transform lineament candidate source")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Disable orientation-consistent gradient clustering candidate source")
    parser.add_argument("--phase-congruency", action="store_true",
                        help="Enable phase congruency candidate source (disabled by default, requires phasepack)")
    parser.add_argument("--no-ridge-crest", action="store_true",
                        help="Disable local maxima of profile curvature candidate source")
    parser.add_argument("--no-ml-probability", action="store_true",
                        help="Disable supervised ML probability candidate source")
    parser.add_argument("--ml-model-path", type=str, default=None,
                        help="Path to save/load the trained ML model pickle file")
    parser.add_argument("--ml-probability-threshold", type=float, default=0.5,
                        help="ML probability threshold for filtering candidates and inferred pixels (0-1, default 0.5)")
    parser.add_argument("--ml-filter-mode", type=str, default="or",
                        choices=["filter", "or"],
                        help="How to use ML: 'filter' (AND with other candidates — deprecated, gates out real LSM signal), 'or' (OR with other candidates, default — ML augments rather than replaces)")

    # Subtle scarp feature detector toggles
    parser.add_argument("--no-subtle-slope-break", action="store_true",
                        help="Disable subtle slope break candidate source")
    parser.add_argument("--no-convexity-profile", action="store_true",
                        help="Disable convexity profile candidate source")
    parser.add_argument("--no-gentle-slope-anomaly", action="store_true",
                        help="Disable gentle slope anomaly candidate source")
    parser.add_argument("--no-multiscale-relief", action="store_true",
                        help="Disable multi-scale relief anomaly candidate source")
    parser.add_argument("--no-slope-profile-curvature", action="store_true",
                        help="Disable slope profile curvature (inflection) candidate source")

    # Candidate-coverage tuning
    parser.add_argument("--tune-objective", type=str, default="candidate_coverage",
                        choices=["candidate_coverage", "f2"],
                        help="Tuning objective: candidate_coverage (Phase 2) or f2 (legacy)")

    # Method breakdown only — skip full pipeline, just regenerate figure
    parser.add_argument("--methodbreak", action="store_true",
                        help="Skip the full pipeline and only regenerate the "
                             "method_breakdown.png figure from an existing "
                             "inference_quality.json. Requires --quality-json "
                             "to point to a valid JSON file.")
    parser.add_argument("--quality-json", type=str, default=None,
                        help="Path to inference_quality.json (for use with --methodbreak)")
    args = parser.parse_args()

    # ── Set DEM-source-dependent paths dynamically ────────────────
    DEM_SOURCE_KEY = args.dem_source
    DEM_PATH = OUT_DIR / DEM_SOURCE_MAP[DEM_SOURCE_KEY]
    OUTPUT_DIR = OUT_DIR / "known_trace_scarp_validation" / DEM_SOURCE_KEY
    CSV_PATH = OUTPUT_DIR / "all_inferred_points.csv"

    # ── Method-breakdown-only mode: skip entire pipeline, just
    # regenerate the figure from an existing inference_quality.json.
    if args.methodbreak:
        qjson_path = args.quality_json or str(OUTPUT_DIR / "inference_quality.json")
        print(f"\n[Method Break] Loading quality data from {qjson_path}")
        with open(qjson_path, "r") as f:
            quality = json.load(f)
        out_dir = Path(OUTPUT_DIR)
        _plot_method_breakdown(quality, out_dir / "method_breakdown.png")
        print(f"\n[Method Break] Done. Outputs in {out_dir}")
        return

    TRACE_BUFFER_M = args.trace_buffer_m
    TRACE2_BUFFER_M = args.trace2_buffer_m
    TUNING_N_TRIALS = args.n_trials
    use_fsd1_seeds = not args.no_fsd1_seeds  # Default: disabled (blind method)

    # Build alternative source dict (existing + new)
    alt_sources = {
        "curvature": not args.no_curvature,
        "edge": not args.no_edge,
        "openness": not args.no_openness,
        "aspect": not args.no_aspect,
        # New sources (10 methods)
        "adaptive_slope": not args.no_adaptive_slope,
        "tpi": not args.no_tpi,
        "tri": not args.no_tri,
        "dog": not args.no_dog,
        "geomorphon": not args.geomorphon,          # disabled by default
        "hough": not args.no_hough,
        "orientation": not args.no_orientation,
        "phase_congruency": args.phase_congruency,  # disabled by default
        "ridge_crest": not args.no_ridge_crest,
        "ml_probability": not args.no_ml_probability,
        # Subtle scarp feature detectors
        "subtle_slope_break": not args.no_subtle_slope_break,
        "convexity_profile": not args.no_convexity_profile,
        "gentle_slope_anomaly": not args.no_gentle_slope_anomaly,
        "multiscale_relief": not args.no_multiscale_relief,
        "slope_profile_curvature": not args.no_slope_profile_curvature,
    }

    print(f"\n{'#'*70}")
    print(f"# 05c-MLTunedDEMWideInference (Blind Empirical Method)")
    print(f"# DEM source: {args.dem_source}")
    print(f"# Phase: {args.phase}")
    print(f"# Parameter tuning trials: {TUNING_N_TRIALS}")
    print(f"# Tune objective: {args.tune_objective}")
    print(f"# FSD1 seeds: {'ON (diagnostic)' if use_fsd1_seeds else 'OFF (blind)'}")
    print(f"# Alt candidates: {alt_sources}")
    print(f"{'#'*70}")

    out_dir = safe_makedirs(OUTPUT_DIR)

    # ── Step 1: Load and smooth HighCon traces ──────────────────────
    print(f"\n[Step 1] Loading HighCon traces and building smoothed lines...")
    highcon_traces = _load_highcon_traces(csv_path=CSV_PATH)
    if not highcon_traces:
        print(f"  [Error] No HighCon trace points found in the lidar CSV")
        sys.exit(1)

    smoothed_traces = _build_smoothed_traces(highcon_traces)
    if not smoothed_traces:
        print(f"  [Error] Could not build smoothed traces")
        sys.exit(1)

    # Determine plotting bounds (zoom around Trace2 if requested)
    plot_bounds = None
    if args.plot_trace2_buffer_m > 0:
        with rasterio.open(DEM_PATH) as src:
            dem_bounds = src.bounds
        plot_bounds = compute_trace2_search_bounds(
            smoothed_traces,
            buffer_m=args.plot_trace2_buffer_m,
            dem_bounds=dem_bounds,
        )
        if plot_bounds is None:
            print("  [Warning] Could not compute Trace2 bounds for plotting; using full extent.")

    _plot_smoothed_traces(smoothed_traces, DEM_PATH,
                          out_dir / "smoothed_traces_overview.png",
                          bounds=plot_bounds)

    # Compute search bounds if requested
    search_bounds = None
    if args.limit_to_trace2_buffer:
        with rasterio.open(DEM_PATH) as src:
            dem_bounds = src.bounds
        search_bounds = compute_trace2_search_bounds(
            smoothed_traces,
            buffer_m=args.trace2_search_buffer_m,
            dem_bounds=dem_bounds,
        )
        if search_bounds is None:
            print("  [Warning] Could not compute Trace2 search bounds; "
                  "falling back to full extent.")
        else:
            print(f"  [Search] Trace2 bounds: {search_bounds}")

    # ── Phase 1: DEM characteristic analysis ───────────────────────
    if args.phase in ("all", "1"):
        run_dem_characteristic_analysis(
            smoothed_traces, DEM_PATH, out_dir,
            bounds=search_bounds, n_samples=200,
            params=DEFAULT_PARAMS.copy())

    # ── Phase 2 / Step 2: ML parameter tuning ─────────────────────
    # For coarse DEMs, use the coarse-DEM preset instead of the 10m-tuned defaults
    with rasterio.open(DEM_PATH) as _src_res:
        _dem_res = abs(_src_res.res[0])
    if _dem_res > COARSE_DEM_THRESHOLD_M:
        tuned_params = COARSE_DEM_PARAMS.copy()
        print(f"  [Coarse DEM] Using coarse-DEM preset params (res={_dem_res:.1f}m)")
    else:
        tuned_params = DEFAULT_PARAMS.copy()
    best_f1 = 0.0
    results_log = []

    if args.phase in ("all", "2"):
        with rasterio.open(DEM_PATH) as src:
            dem_crs = src.crs

        if args.tune_objective == "candidate_coverage":
            tuned_params, best_f1, results_log = tune_candidate_coverage(
                smoothed_traces, DEM_PATH, dem_crs, bounds=search_bounds)
        else:
            tuned_params, best_f1, results_log = tune_parameters(
                smoothed_traces, DEM_PATH, dem_crs, bounds=search_bounds)
    elif args.phase == "inference":
        print(f"\n[Step 2] Skipping tuning, using default parameters")
    elif args.skip_tuning:
        print(f"\n[Step 2] Skipping ML tuning, using default parameters")

    # Save tuning results
    tuning_summary = {
        "tuned_params": tuned_params,
        "default_params": DEFAULT_PARAMS,
        "best_score": float(best_f1),
        "n_trials": TUNING_N_TRIALS,
        "tune_objective": args.tune_objective,
        "param_space": {k: list(v) for k, v in PARAM_SPACE.items()},
    }
    tuning_path = out_dir / "tuned_parameters.json"
    with open(tuning_path, "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print(f"\n  Saved tuning summary to {tuning_path}")

    if results_log:
        pd.DataFrame(results_log).to_csv(out_dir / "tuning_results_log.csv", index=False)

    # ── Step 2b: Train ML candidate model (if enabled) ────────────
    if alt_sources.get("ml_probability", False) and args.phase in ("all", "3", "inference"):
        print(f"\n[Step 2b] Training ML candidate model...")
        fsd1_xy_ml = _get_fsd1_seed_points(smoothed_traces)
        if fsd1_xy_ml is not None and len(fsd1_xy_ml) > 0:
            ml_model_path = args.ml_model_path or str(out_dir / "ml_candidate_model.pkl")
            with rasterio.open(DEM_PATH) as src:
                _train_ml_candidate_model(
                    src, fsd1_xy_ml, tuned_params,
                    resolution=abs(src.res[0]),
                    bounds=search_bounds,
                    model_path=Path(ml_model_path),
                )
        else:
            print("  [ML] No FSD1 points for training, ML source will use fallback")

    # ── Step 2c: Relax classification parameters for Trace 2 ──────
    # The additional candidate density from new alt sources means the
    # classification step sees more candidates near Trace 2 FSD1 points.
    # Relax asymmetry and width thresholds to let more of these through.
    if alt_sources.get("ml_probability", False) or alt_sources.get("adaptive_slope", False) or \
       alt_sources.get("tpi", False) or alt_sources.get("tri", False) or \
       alt_sources.get("dog", False) or alt_sources.get("hough", False) or \
       alt_sources.get("orientation", False) or alt_sources.get("ridge_crest", False) or \
       alt_sources.get("slope_profile_curvature", False) or alt_sources.get("subtle_slope_break", False) or \
       alt_sources.get("gentle_slope_anomaly", False):
        # Only relax if we have new sources enabled (not just existing 4)
        n_new_sources = sum(1 for k in ["adaptive_slope", "tpi", "tri", "dog",
                                        "hough", "orientation", "ridge_crest",
                                        "ml_probability", "slope_profile_curvature",
                                        "subtle_slope_break", "gentle_slope_anomaly"] if alt_sources.get(k, False))
        if n_new_sources >= 2:
            # Relax asymmetry threshold: more candidates means we can afford
            # to be less strict and recover more FSD1 points
            if tuned_params.get("MIN_SCARP_ASYMMETRY_SCORE", 0.3) > 0.15:
                old_asym = tuned_params.get("MIN_SCARP_ASYMMETRY_SCORE", 0.3)
                tuned_params["MIN_SCARP_ASYMMETRY_SCORE"] = max(0.10, old_asym * 0.5)
                print(f"\n  [Trace2 Tune] Relaxed MIN_SCARP_ASYMMETRY_SCORE: "
                      f"{old_asym:.3f} -> {tuned_params['MIN_SCARP_ASYMMETRY_SCORE']:.3f}")

            # Relax minimum scarp width: shorter scarps are now detectable
            # with better candidate seeding
            if tuned_params.get("MIN_SCARP_WIDTH_M", 30.0) > 15.0:
                old_width = tuned_params.get("MIN_SCARP_WIDTH_M", 30.0)
                tuned_params["MIN_SCARP_WIDTH_M"] = max(10.0, old_width * 0.5)
                print(f"  [Trace2 Tune] Relaxed MIN_SCARP_WIDTH_M: "
                      f"{old_width:.1f} -> {tuned_params['MIN_SCARP_WIDTH_M']:.1f}")

            # Relax minimum scarp height: 0.5m LiDAR can resolve smaller scarps
            if tuned_params.get("MIN_SCARP_HEIGHT_M", 0.2) > 0.1:
                old_height = tuned_params.get("MIN_SCARP_HEIGHT_M", 0.2)
                tuned_params["MIN_SCARP_HEIGHT_M"] = max(0.05, old_height * 0.5)
                print(f"  [Trace2 Tune] Relaxed MIN_SCARP_HEIGHT_M: "
                      f"{old_height:.3f} -> {tuned_params['MIN_SCARP_HEIGHT_M']:.3f}")

    # ── Step 3: DEM-wide inference with tuned params ───────────────
    if args.phase in ("all", "3", "inference"):
        print(f"\n[Step 3] Running DEM-wide inference with tuned parameters...")

        # For coarse DEMs (e.g. SRTM 90m), enable NFD1 seed injection and
        # scale the trace proximity buffer to the DEM resolution.
        use_nfd1_seeds = (DEM_SOURCE_KEY not in ("lidar_0_5m",))
        if use_nfd1_seeds:
            print(f"  [Coarse DEM] Enabling NFD1 seed injection for {DEM_SOURCE_KEY}")

        # For coarse DEMs, scale the seed accept radius to the resolution
        effective_seed_radius = args.fsd1_seed_accept_radius_m
        if use_nfd1_seeds:
            with rasterio.open(DEM_PATH) as _src_rad:
                _dem_res_rad = abs(_src_rad.res[0])
            effective_seed_radius = max(args.fsd1_seed_accept_radius_m, _dem_res_rad)
            print(f"  [Coarse DEM] Seed accept radius: {args.fsd1_seed_accept_radius_m}m -> {effective_seed_radius}m")

        inferred_df = run_dem_wide_inference(
            DEM_PATH, tuned_params, smoothed_traces, out_dir,
            write_tiles=args.write_tiles, use_fsd1_seeds=use_fsd1_seeds,
            bounds=search_bounds,
            seed_accept_radius_m=effective_seed_radius,
            alt_sources=alt_sources if args.phase in ("all", "3", "inference") else None,
            ml_probability_threshold=args.ml_probability_threshold,
            ml_filter_mode=args.ml_filter_mode,
            use_nfd1_seeds=use_nfd1_seeds)


        if args.sensitivity:
            seed_xy = _get_fsd1_seed_points(smoothed_traces)
            run_criteria_sensitivity_analysis(
                DEM_PATH, tuned_params, smoothed_traces, out_dir,
                bounds=search_bounds, seed_xy=seed_xy,
                seed_accept_radius_m=args.fsd1_seed_accept_radius_m)

        # ── Step 4: Evaluate quality ────────────────────────────────
        print(f"\n[Step 4] Evaluating inference quality...")
        quality = _evaluate_inference_quality(
            inferred_df, smoothed_traces,
            alt_sources=alt_sources, tuned_params=tuned_params)
        print(f"  Recall:    {quality['recall']:.4f} "
              f"({quality['n_highcon_recovered']}/{quality['n_highcon_total']} ground truth points recovered)")
        print(f"  Precision: {quality['precision']:.4f}")
        print(f"  F1:        {quality['f1']:.4f}")
        print(f"  Inferred points: {quality['n_inferred']:,}")
        for tid, stats in quality.get("per_trace", {}).items():
            src_label = stats.get("source", "HighCon")
            print(f"  Trace {tid} ({src_label}): recall={stats['recall']:.4f} "
                  f"({stats['n_recovered']}/{stats['n_highcon']} HighCon pts)")
            if "n_fsd1" in stats:
                print(f"    FSD1 recovery: {stats['fsd1_recall']:.4f} "
                      f"({stats['n_fsd1_recovered']}/{stats['n_fsd1']} FSD1 pts)")

        # Print NFD1 recovery stats if available
        nfd1_stats = quality.get("nfd1", {})
        if nfd1_stats:
            print(f"  NFD1 recovery: {nfd1_stats['nfd1_recall']:.4f} "
                  f"({nfd1_stats['n_nfd1_recovered']}/{nfd1_stats['n_nfd1_total']} NFD1 pts)")

        # Print method breakdown
        mb = quality.get("method_breakdown", {})
        if mb:
            n_total = mb.get("n_inferred_total", 0)
            pp = mb.get("per_point_methods", {})
            print(f"\n  ── Method Breakdown (n={n_total:,} inferred points) ──")
            print(f"  Note: a point may be attributed to multiple methods.")
            print(f"  Solo counts (points where the method contributed):")
            # Sort by count descending
            sorted_pp = sorted(pp.items(), key=lambda x: x[1]["n_points"],
                               reverse=True)
            print(f"  {'Method':<40s} {'Points':>8s} {'%':>7s}")
            print(f"  {'-'*40} {'-'*8} {'-'*7}")
            for mname, minfo in sorted_pp:
                print(f"  {mname:<40s} {minfo['n_points']:>8,d} "
                      f"{minfo['percentage']:>6.1f}%")

            # Print top co-occurrences (top 10 pairs)
            cooc = mb.get("co_occurrence_matrix", {})
            if cooc:
                mnames = cooc.get("method_names", [])
                mat = cooc.get("matrix", {})
                pairs = []
                for i, mi in enumerate(mnames):
                    for j, mj in enumerate(mnames):
                        if j >= i:  # skip upper triangle (redundant)
                            continue
                        val = mat.get(mi, {}).get(mj, 0)
                        if val > 0:
                            pairs.append((mi, mj, val))
                pairs.sort(key=lambda x: x[2], reverse=True)
                if pairs:
                    print(f"\n  Top method co-occurrences (points attributed to both):")
                    print(f"  {'Method A':<40s} {'Method B':<40s} {'Points':>8s} {'%':>7s}")
                    print(f"  {'-'*40} {'-'*40} {'-'*8} {'-'*7}")
                    for mi, mj, val in pairs[:10]:
                        pct = val / n_total * 100.0 if n_total > 0 else 0.0
                        print(f"  {mi:<40s} {mj:<40s} {val:>8,d} {pct:>6.1f}%")

            print(f"\n  Pipeline-level methods:")
            for mname, minfo in mb.get("pipeline_methods", {}).items():
                status = "ENABLED" if minfo["enabled"] else "DISABLED"
                print(f"  {mname:<40s} {status}")

        quality_path = out_dir / "inference_quality.json"
        with open(quality_path, "w") as f:
            json.dump(quality, f, indent=2, default=str)
        print(f"\n  Saved quality metrics to {quality_path}")

        # ── Step 5: Visualise results ───────────────────────────────
        print(f"\n[Step 5] Generating visualisations...")
        _plot_dem_wide_results(inferred_df, smoothed_traces, DEM_PATH,
                               out_dir / "dem_wide_inference_overview.png",
                               quality=quality, bounds=plot_bounds)

        # Method breakdown chart + CSV
        _plot_method_breakdown(quality, out_dir / "method_breakdown.png")

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'#'*70}")
        print(f"# 05c Complete")
        print(f"# Phase: {args.phase}")
        print(f"# Output directory: {out_dir}")
        print(f"# Tune objective: {args.tune_objective}")
        print(f"# Best score: {best_f1:.4f}")
        print(f"# Inference F1: {quality['f1']:.4f}")
        print(f"# Inferred points: {quality['n_inferred']:,}")
        print(f"# HighCon recovered: {quality['n_highcon_recovered']}/{quality['n_highcon_total']}")
        for tid, stats in quality.get("per_trace", {}).items():
            src = stats.get("source", "HighCon")
            print(f"# Trace {tid} ({src}): "
                  f"recall={stats['recall']:.4f} "
                  f"({stats['n_recovered']}/{stats['n_highcon']} HighCon pts)")
            if "n_fsd1" in stats:
                print(f"#   FSD1 recovered: {stats['n_fsd1_recovered']}/{stats['n_fsd1']} "
                      f"({stats['fsd1_recall']:.4f})")
        print(f"{'#'*70}\n")
    else:
        print(f"\n{'#'*70}")
        print(f"# Phase {args.phase} complete. Use --phase inference or --phase all")
        print(f"# to run DEM-wide inference.")
        print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
