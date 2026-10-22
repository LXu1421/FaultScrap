#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02-VisualAudit.py -- Visualisation Suite & Uncertainty Assessment

Implements Sections 2 and 3 of the Surface Expression Audit plan:
  2.1 Base Cartographic Products
      - Multi-azimuth hillshade composite map
      - SVF/Openness overlay comparison map
  2.2 Geomorphic Evidence Maps
      - Lineament density map
      - Drainage anomaly map
      - Expression Quality Classification Map (4 classes: A/B/C/D)
  2.3 Quantitative Index Visualization
      - Geomorphic index radar/spider charts per fault segment
      - Longitudinal river profiles with knickpoint annotation
      - Hypsometric curves per sub-catchment
  2.4 Uncertainty and Confidence Layers
      - Confidence heatmap (data density / LiDAR vs 30m coverage)
      - Multi-resolution comparison panels (30m/10m/0.5m side-by-side)
  3.3 Confidence Scoring
      - Composite confidence score (0-10) per fault segment
      - Aleatory/epistemic uncertainty classification
  3.4 Cross-Validation
      - Benchmark against Yallourn (1-2m) and Korean (2-4m) scarps
      - Cross-validate with LiDAR Phase 1 results where available

Reads derivatives produced by 01-DerivativeGen.py (.npy + .json format,
with .tif fallback for legacy data) and Phase 1 audit CSVs.
All output saved under data/derivatives/visual_audit/.

Windows compatibility: no Unicode characters in code or strings.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.enums import Resampling as ResampEnum  # noqa: F811 (alias for clarity)
from rasterio.windows import Window
from rasterio.transform import xy as rio_xy, Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, label, uniform_filter, sobel, \
    maximum_filter, minimum_filter, grey_dilation
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA as SkPCA
from sklearn.neighbors import KernelDensity
import psutil
from tqdm import tqdm
from pathlib import Path
import json
import heapq
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
DERIV_DIR = DATA_DIR / "derivatives"
OUT_DIR   = DERIV_DIR / "visual_audit"

# DEM paths (reprojected to EPSG:28355)
SRTM3S_RP  = DERIV_DIR / "srtm3s_epsg28355.tif"
SRTM_RP    = DERIV_DIR / "srtm1s_epsg28355.tif"
DEM10M_RP  = DERIV_DIR / "dem10m_epsg28355.tif"
LIDAR_RP   = DERIV_DIR / "lidar_0_5m_epsg28355.tif"

# Derivative sub-directories
SRTM3S_DERIV  = DERIV_DIR / "srtm3s"
SRTM_DERIV    = DERIV_DIR / "srtm1s"
DEM10M_DERIV  = DERIV_DIR / "dem10m"
LIDAR_DERIV   = DERIV_DIR / "lidar_0_5m"

# Phase 1 audit outputs
AUDIT_DIR = DERIV_DIR / "phase1_audit"

# Structural points
STRUCT_PTS_PATH = DATA_DIR / "structural_points_inventory.csv"

# Target CRS
TARGET_CRS = CRS.from_epsg(28355)

# Hillshade azimuths
HS_AZIMUTHS = [315, 45, 135, 225]
HS_ALTITUDE = 8

# Lineament extraction parameters
LINE_MIN_LENGTH_M     = 500
LINE_CANNY_SIGMA      = 2.0
LINE_CANNY_LOW        = 0.1
LINE_CANNY_HIGH       = 0.4

# Geomorphic index reference ranges (from published analogues)
REF_SMF_ACTIVE  = (1.0, 1.4)    # low = tectonically active
REF_VF_ACTIVE   = (0.0, 0.5)    # low = incision-dominated
REF_HI_ACTIVE   = (0.5, 1.0)    # high = young landscape
REF_SL_ANOMALY  = 500           # threshold for anomalous SL index

# Yallourn and Korean reference scarp heights (m)
YALLOURN_SCARP_RANGE = (1.0, 2.0)
KOREAN_SCARP_RANGE   = (2.0, 4.0)

# Minimum detectable scarp height per DEM resolution
MIN_DETECTABLE_SCARP = {
    90.0: 9.0,   # SRTM 3s (~90m), coarser than 1s
    30.0: 3.0,   # SRTM ~6m RMSE vertical, can resolve ~3m
    10.0: 1.0,   # GA 10m DEM
    0.5:  0.2,   # LiDAR 0.5m
}

# Expression quality thresholds (must match 01-DerivativeGen.py)
EXPR_CLASS1_SLOPE     = 5.0   # deg -- strong gradient break
EXPR_CLASS2_SLOPE     = 2.0   # deg -- moderate gradient break
EXPR_CLASS1_SVFDIFF   = 0.15  # SVF anomaly threshold (strong)
EXPR_CLASS2_SVFDIFF   = 0.08  # SVF anomaly threshold (moderate)

# 4-class mapping from 01 script's 3-class system:
#   Class 1 -> A (Well-expressed, unambiguous geomorphic signature)
#   Class 2 -> B (Probable, multi-indicator concordance)
#   Class 3 -> C (Inferred, single indicator or ambiguous context)
#   Added:  -> D (No surface expression detected)
CLASS_MAP_3TO4 = {
    "Class 1": "A",
    "Class 2": "B",
    "Class 3": "C",
}

CLASS_DESCRIPTIONS = {
    "A": "Well-expressed, unambiguous geomorphic signature",
    "B": "Probable, multi-indicator concordance",
    "C": "Inferred, single indicator or ambiguous context",
    "D": "No surface expression detected",
}


def _summarise_expression_quality_points(point_df, traces=None):
    """Aggregate point-based expression quality rows to the legacy trace schema."""
    if point_df is None or len(point_df) == 0:
        return pd.DataFrame()

    trace_lookup = {t["trace_id"]: t for t in (traces or [])}
    class_rank = {"Class 1": 3, "Class 2": 2, "Class 3": 1}
    rows = []

    for tid, grp in point_df.groupby("trace_id"):
        detected = grp[grp["scarp_type"] != "none"] if "scarp_type" in grp.columns else grp.iloc[0:0]
        trace = trace_lookup.get(tid)
        source = grp["source"].iloc[0] if "source" in grp.columns else trace.get("source", "") if trace else ""
        centroid_x = trace["centroid_x"] if trace is not None else grp["sample_x"].mean()
        centroid_y = trace["centroid_y"] if trace is not None else grp["sample_y"].mean()
        trace_length_m = trace["length_m"] if trace is not None else np.nan
        detection_rate = float((grp["scarp_type"] != "none").mean()) if "scarp_type" in grp.columns else 0.0
        mean_height = float(detected["scarp_height_m"].mean()) if len(detected) > 0 else 0.0
        max_height = float(detected["scarp_height_m"].max()) if len(detected) > 0 else 0.0
        mean_slope = float(detected["max_slope_deg"].mean()) if len(detected) > 0 else 0.0
        slope_anomaly = float(grp["slope_anomaly_deg"].max()) if "slope_anomaly_deg" in grp.columns else 0.0
        svf_anomaly = float(grp["svf_anomaly"].max()) if "svf_anomaly" in grp.columns else 0.0

        quality_class = sorted(grp["quality_class"].dropna().unique(),
                               key=lambda c: class_rank.get(c, 0), reverse=True)[0]
        n_drain = 0
        if "drainage_anomaly_ids" in grp.columns:
            unique_ids = set()
            for val in grp["drainage_anomaly_ids"].dropna():
                if val:
                    unique_ids.update(v for v in str(val).split("|") if v)
            n_drain = len(unique_ids)
        elif "n_near_drainage_anomalies" in grp.columns:
            n_drain = int((grp["n_near_drainage_anomalies"] > 0).sum())

        if len(detected) == 0:
            structural_context = "none"
        else:
            n_discrete = int((detected["scarp_type"] == "discrete").sum())
            n_monocline = int((detected["scarp_type"] == "monocline").sum())
            if n_discrete > n_monocline:
                structural_context = "discrete_rupture"
            elif n_monocline > n_discrete:
                structural_context = "monocline"
            else:
                structural_context = "mixed"

        rows.append({
            "trace_id": tid,
            "source": source,
            "centroid_x": round(centroid_x, 0),
            "centroid_y": round(centroid_y, 0),
            "trace_length_m": round(trace_length_m, 0) if pd.notna(trace_length_m) else np.nan,
            "quality_class": quality_class,
            "mean_scarp_height_m": round(mean_height, 2),
            "max_scarp_height_m": round(max_height, 2),
            "structural_context": structural_context,
            "n_profiles": int(len(grp)),
            "detection_rate": round(detection_rate, 2),
            "mean_scarp_slope_deg": round(mean_slope, 1),
            "n_drain_anomalies": int(n_drain),
            "slope_anomaly_deg": round(slope_anomaly, 1),
            "svf_anomaly": round(svf_anomaly, 3),
            "rationale": "derived_from_point_expression_quality",
        })

    return pd.DataFrame(rows)

# --- REM / HAND configuration ---
# Flow accumulation threshold (in pixels) for defining drainage cells
DRAINAGE_ACCUM_THRESHOLD = 1000

# River bed / valley detection — context-aware multi-criteria approach
# Instead of a single fixed REM/HAND threshold, we use local relief context
# to adaptively identify river beds and valleys across different terrains.

# Fixed thresholds (retained for shallow / ridge-adjacent river beds)
RIVER_BED_REM_THRESHOLD = 2.0       # REM below this -> river bed (was 1.0)
VALLEY_HAND_THRESHOLD = 10.0        # HAND below this -> valley floor (was 5.0)

# Adaptive thresholds based on local elevation context
# River bed: a cell is a river bed if it is significantly lower than its
# surrounding terrain.  We use two complementary criteria:
#   (a) TPI criterion:  cell < (local_mean - TPI_BED_FACTOR * local_std)
#   (b) Relief criterion: cell < local_min + RELIEF_BED_FRAC * local_relief
# Either criterion triggers river bed identification.
TPI_WINDOW_RADIUS_M = 200     # radius for TPI computation (local mean/std)
RELIEF_WINDOW_RADIUS_M = 1000 # radius for local relief (context window)
TPI_BED_FACTOR = 1.0          # cell must be >1 std below local mean to be bed
RELIEF_BED_FRAC = 0.15        # cell in bottom 15% of local relief -> bed
TPI_VALLEY_FACTOR = 0.5       # cell >0.5 std below local mean -> valley
RELIEF_VALLEY_FRAC = 0.35     # cell in bottom 35% of local relief -> valley

# Minimum local relief (m) required for adaptive criteria to activate.
# Below this, terrain is essentially flat and we rely on REM/HAND only.
MIN_RELIEF_FOR_ADAPTIVE = 5.0

# Minimum slope (deg) for flow direction (flats filling)
FLOW_MIN_SLOPE = 1e-6

# Pit filling iterations for hydrological preprocessing
PIT_FILL_MAX_ITER = 100

# Expression quality per-pixel buffer (m)
EXPR_PIXEL_BUFFER_M = 200

# Slope thresholds for per-pixel quality classification (deg)
EXPR_PIX_CLASS_A_SLOPE = 5.0   # strong gradient break -> Class A
EXPR_PIX_CLASS_B_SLOPE = 2.0   # moderate gradient break -> Class B
EXPR_PIX_CLASS_C_SLOPE = 1.0   # weak gradient break -> Class C

# SVF thresholds for per-pixel quality classification
EXPR_PIX_CLASS_A_SVF = 0.15    # strong SVF anomaly -> Class A
EXPR_PIX_CLASS_B_SVF = 0.08    # moderate SVF anomaly -> Class B

# Colour palettes for fault source x quality class (A/B/C/D)
# Each source has 4 shades: light (A) -> medium (B) -> dark (C) -> grey (D)
SOURCE_CLASS_COLOURS = {
    "FSD1":         ["#FFCC80", "#FF9800", "#E65100", "#9E9E9E"],  # orange shades
    "CWvic1":       ["#81D4FA", "#29B6F6", "#01579B", "#9E9E9E"],  # blue shades
    "GeoStru2501":  ["#A5D6A7", "#66BB6A", "#1B5E20", "#9E9E9E"],  # green shades
    "GeoStru501":   ["#EF9A9A", "#EF5350", "#B71C1C", "#9E9E9E"],  # red shades
}

# Tile DPI
TILE_DPI = 300


# ===================================================================
# DERIVATIVE LOADING HELPERS (.npy + .json format)
# ===================================================================

def get_dem_stem(dem_key):
    """Return the file stem name for a DEM key.

    The reprojected DEMs follow the naming pattern:
        {dem_key}_epsg28355.tif

    After clip_to_common_extent (01-DerivativeGen.py), DEMs are renamed to:
        {dem_key}_epsg28355_clipped.tif

    Derivatives follow:
        {dem_key}_epsg28355[_clipped]_{deriv_type}.npy
        {dem_key}_epsg28355[_clipped]_{deriv_type}.json

    Returns the _clipped variant first (post-alignment). load_derivative()
    will fall back to the plain variant if _clipped files are not found.
    """
    return f"{dem_key}_epsg28355_clipped"


def _resolve_dem_stem(deriv_dir, dem_stem, deriv_type):
    """Try to resolve which stem variant actually has derivative files.

    Tries the given stem first, then falls back to the non-clipped
    variant (stripping _clipped suffix) if not found.

    Returns the resolved stem, or the original stem if neither is found.
    """
    npy_path = deriv_dir / f"{dem_stem}_{deriv_type}.npy"
    if npy_path.exists():
        return dem_stem

    master_json = deriv_dir / f"{dem_stem}_{deriv_type}_meta.json"
    if master_json.exists():
        return dem_stem

    tif_path = deriv_dir / f"{dem_stem}_{deriv_type}.tif"
    if tif_path.exists():
        return dem_stem

    # Try stripping _clipped suffix
    if dem_stem.endswith("_clipped"):
        alt_stem = dem_stem[:-len("_clipped")]
        alt_npy = deriv_dir / f"{alt_stem}_{deriv_type}.npy"
        if alt_npy.exists():
            return alt_stem
        alt_master = deriv_dir / f"{alt_stem}_{deriv_type}_meta.json"
        if alt_master.exists():
            return alt_stem
        alt_tif = deriv_dir / f"{alt_stem}_{deriv_type}.tif"
        if alt_tif.exists():
            return alt_stem

    # Try adding _clipped suffix
    if not dem_stem.endswith("_clipped"):
        alt_stem = f"{dem_stem}_clipped"
        alt_npy = deriv_dir / f"{alt_stem}_{deriv_type}.npy"
        if alt_npy.exists():
            return alt_stem
        alt_master = deriv_dir / f"{alt_stem}_{deriv_type}_meta.json"
        if alt_master.exists():
            return alt_stem
        alt_tif = deriv_dir / f"{alt_stem}_{deriv_type}.tif"
        if alt_tif.exists():
            return alt_stem

    return dem_stem


def load_derivative(deriv_dir, dem_stem, deriv_type):
    """Load a derivative from .npy + .json.

    Returns (data, metadata_dict) or (None, None) if not found.
    Falls back to .tif if .npy is not available.

    For LiDAR tiled derivatives, checks for master metadata and
    returns (None, master_meta) so caller can use windowed approach.

    Automatically resolves _clipped vs non-_clipped stem naming
    (01-DerivativeGen.py renames DEMs to *_clipped.tif after alignment).
    """
    resolved_stem = _resolve_dem_stem(deriv_dir, dem_stem, deriv_type)

    npy_path = deriv_dir / f"{resolved_stem}_{deriv_type}.npy"
    json_path = deriv_dir / f"{resolved_stem}_{deriv_type}.json"

    if npy_path.exists() and json_path.exists():
        data = np.load(npy_path)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return data, meta

    # Check for LiDAR tiled format (master metadata + tile .npy files)
    master_json = deriv_dir / f"{resolved_stem}_{deriv_type}_meta.json"
    if master_json.exists():
        with open(master_json, "r", encoding="utf-8") as f:
            master_meta = json.load(f)
        # Signal that this is tiled data -- caller must use windowed approach
        return None, master_meta

    # Fallback: try .tif
    tif_path = deriv_dir / f"{resolved_stem}_{deriv_type}.tif"
    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
        meta = {
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "crs": str(src.crs),
            "shape": list(data.shape),
            "nodata": src.nodata,
            "resolution": abs(src.res[0]),
            "type": deriv_type,
        }
        return data, meta

    return None, None


def load_derivative_window(deriv_dir, dem_stem, deriv_type, cx, cy,
                            half_m, nodata=None):
    """Load a window of a derivative centred on (cx, cy) with half-width half_m.

    For .npy derivatives: loads full array then slices.
    For .tif derivatives: uses rasterio windowed read.
    For LiDAR tiled derivatives: reads from the reprojected .tif directly.

    Returns (data_window, transform) or (None, None) if not available.
    """
    resolved_stem = _resolve_dem_stem(deriv_dir, dem_stem, deriv_type)

    # First try single .npy file
    npy_path = deriv_dir / f"{resolved_stem}_{deriv_type}.npy"
    json_path = deriv_dir / f"{resolved_stem}_{deriv_type}.json"

    if npy_path.exists() and json_path.exists():
        data = np.load(npy_path)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        t = meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        res = meta.get("resolution", 1.0)

        col, row = ~transform * (cx, cy)
        row, col = int(round(row)), int(round(col))
        half_px = int(half_m / res)
        r0 = max(0, row - half_px)
        r1 = min(data.shape[0], row + half_px)
        c0 = max(0, col - half_px)
        c1 = min(data.shape[1], col + half_px)
        window_data = data[r0:r1, c0:c1].copy()
        win_transform = transform * Affine.translation(c0, r0)
        return window_data, win_transform

    # Fallback: try .tif with windowed read
    tif_path = deriv_dir / f"{resolved_stem}_{deriv_type}.tif"
    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            res = abs(src.res[0])
            col, row = ~src.transform * (cx, cy)
            row, col = int(round(row)), int(round(col))
            half_px = int(half_m / res)
            r0 = max(0, row - half_px)
            r1 = min(src.height, row + half_px)
            c0 = max(0, col - half_px)
            c1 = min(src.width, col + half_px)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=win).astype(np.float32)
            win_transform = src.window_transform(win)
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data < -1e6] = np.nan
        return data, win_transform

    return None, None


def load_derivative_from_dem(dem_path, cx, cy, half_m, nodata=None):
    """Load a window of a reprojected DEM .tif directly.
    Used for LiDAR where derivative .npy/.tif may not exist.

    Returns (data, transform, bounds_tuple) or (None, None, None).
    """
    try:
        with rasterio.open(dem_path) as src:
            res = abs(src.res[0])
            col, row = ~src.transform * (cx, cy)
            row, col = int(round(row)), int(round(col))
            half_px = int(half_m / res)
            r0 = max(0, row - half_px)
            r1 = min(src.height, row + half_px)
            c0 = max(0, col - half_px)
            c1 = min(src.width, col + half_px)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=win).astype(np.float32)
            win_transform = src.window_transform(win)
            bounds = (win_transform.c,
                      win_transform.c + (c1 - c0) * win_transform.a,
                      win_transform.f + (r1 - r0) * win_transform.e,
                      win_transform.f)
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data < -1e6] = np.nan
        return data, win_transform, bounds
    except Exception:
        return None, None, None


# ===================================================================
# GENERAL HELPERS
# ===================================================================

def safe_makedirs(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def mem_gb():
    return psutil.virtual_memory().available / (1024 ** 3)


def load_raster(path, window=None):
    """Load a raster as float32 array. Returns (data, transform, bounds, profile)."""
    path = Path(path) if not isinstance(path, Path) else path
    with rasterio.open(path) as src:
        if window is not None:
            data = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
        else:
            data = src.read(1).astype(np.float32)
            transform = src.transform
        bounds = src.bounds
        profile = src.profile
    return data, transform, bounds, profile


def load_structural_points(csv_path, extent=None):
    """Load structural points, optionally filtered by extent (xmin,xmax,ymin,ymax)."""
    print(f"  Loading structural points...")
    chunks = pd.read_csv(csv_path, chunksize=500_000)
    dfs = []
    for chunk in chunks:
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
            mask = ((chunk["XCoord"] >= xmin) & (chunk["XCoord"] <= xmax) &
                    (chunk["YCoord"] >= ymin) & (chunk["YCoord"] <= ymax))
            chunk = chunk[mask]
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} points")
    return df


def reconstruct_fault_traces(struct_pts, cluster_dist_m=30, min_pts=5):
    """Reconstruct fault traces from dense point inventory (same as 01 script)."""
    print(f"  Reconstructing fault traces from {len(struct_pts):,} points...")
    traces = []
    trace_id = 0

    for src_name in struct_pts["source"].unique():
        pts = struct_pts[struct_pts["source"] == src_name].copy()
        if len(pts) < min_pts:
            continue
        coords = pts[["XCoord", "YCoord"]].values.astype(np.float64)
        tree = cKDTree(coords)
        visited = np.zeros(len(coords), dtype=bool)
        component_ids = np.full(len(coords), -1, dtype=int)
        comp_idx = 0

        for start in range(len(coords)):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            component_ids[start] = comp_idx
            while queue:
                node = queue.pop(0)
                neighbours = tree.query_ball_point(coords[node], cluster_dist_m)
                for nb in neighbours:
                    if not visited[nb]:
                        visited[nb] = True
                        component_ids[nb] = comp_idx
                        queue.append(nb)
            comp_idx += 1

        for ci in range(comp_idx):
            mask = component_ids == ci
            comp_coords = coords[mask]
            if len(comp_coords) < min_pts:
                continue
            ordered = _order_nn(comp_coords)
            diffs = np.diff(ordered, axis=0)
            seg_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            total_length = np.sum(seg_lengths)
            dx = ordered[-1, 0] - ordered[0, 0]
            dy = ordered[-1, 1] - ordered[0, 1]
            azimuth = np.degrees(np.arctan2(dx, dy)) % 360

            traces.append({
                "trace_id": trace_id,
                "source": src_name,
                "coords": ordered,
                "length_m": total_length,
                "azimuth_deg": azimuth,
                "n_points": len(ordered),
                "centroid_x": np.mean(ordered[:, 0]),
                "centroid_y": np.mean(ordered[:, 1]),
            })
            trace_id += 1

    print(f"  Reconstructed {len(traces)} fault traces")
    return traces


def _order_nn(coords):
    if len(coords) <= 2:
        return coords
    centroid = coords.mean(axis=0)
    dists = np.sqrt((coords[:, 0] - centroid[0])**2 + (coords[:, 1] - centroid[1])**2)
    start_idx = np.argmin(dists)
    ordered = [coords[start_idx]]
    remaining = np.ones(len(coords), dtype=bool)
    remaining[start_idx] = False
    current = coords[start_idx]
    while np.any(remaining):
        rem = coords[remaining]
        d = np.sqrt((rem[:, 0] - current[0])**2 + (rem[:, 1] - current[1])**2)
        nearest = np.argmin(d)
        nearest_idx = np.where(remaining)[0][nearest]
        ordered.append(coords[nearest_idx])
        remaining[nearest_idx] = False
        current = coords[nearest_idx]
    return np.array(ordered)


def _sample_raster_value(raster_path, x, y):
    """Sample a single value from a raster at (x, y)."""
    try:
        with rasterio.open(raster_path) as src:
            vals = list(src.sample([(x, y)]))
            val = float(vals[0][0])
            if np.isnan(val) or val < -1e6:
                return 0.0
            return val
    except Exception:
        return 0.0


def load_raster_window(raster_path, cx, cy, half_m, nodata=None):
    """Load a window of a raster centred on (cx, cy) with half-width half_m."""
    with rasterio.open(raster_path) as src:
        res = abs(src.res[0])
        half_px = int(half_m / res)
        row_c, col_c = ~src.transform * (cx, cy)
        row_c, col_c = int(row_c), int(col_c)
        r0 = max(0, row_c - half_px)
        r1 = min(src.height, row_c + half_px)
        c0 = max(0, col_c - half_px)
        c1 = min(src.width, col_c + half_px)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        bounds = (transform.c, transform.c + (c1 - c0) * transform.a,
                  transform.f + (r1 - r0) * transform.e, transform.f)

    if nodata is not None:
        data[data == nodata] = np.nan
    data[data < -1e6] = np.nan

    return data, transform, bounds, None


def _extent_from_transform(transform, shape):
    """Compute [left, right, bottom, top] from Affine transform and array shape."""
    rows, cols = shape
    x_min = transform.c
    x_max = transform.c + cols * transform.a
    y_top = transform.f
    y_bottom = transform.f + rows * transform.e
    return [x_min, x_max, y_bottom, y_top]


def _save_raster(data, transform, crs, out_path, nodata=np.nan, dtype=None):
    """Save a 2D array as a GeoTIFF with proper transform and CRS.

    Parameters
    ----------
    data : np.ndarray (2D)
    transform : rasterio.Affine
    crs : rasterio.CRS or str
    out_path : Path or str
    nodata : value to use for nodata pixels
    dtype : output dtype (default: float32)
    """
    out_path = Path(out_path)
    if dtype is None:
        dtype = "float32"
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
    print(f"    [TIF] Saved: {out_path.name}")


# ===================================================================
# HYDROLOGY: D8 Flow Direction, Flow Accumulation, HAND, REM
# ===================================================================

def _fill_pits(dem, max_iter=50):
    """Fill depressions (pits) in the DEM so that flow can reach the edge.

    Uses a vectorised iterative approach with scipy.ndimage: in each
    iteration, local minima (pits) are raised to just above their
    lowest neighbour.  This is fast for typical DEMs where most pits
    are shallow (1-5 pixels deep).

    Parameters
    ----------
    dem : 2D float array (will be copied; nodata = np.nan)
    max_iter : maximum iterations (default 50, sufficient for most DEMs)

    Returns
    -------
    filled : 2D float array with pits removed
    """
    filled = dem.copy()
    rows, cols = filled.shape
    valid = ~np.isnan(filled)

    # Use a large value for nodata/border in the padded array
    pad_val = np.nanmax(filled) + 1.0 if np.any(valid) else 1.0

    for iteration in range(max_iter):
        # Pad the DEM so border cells are treated as spilling outward
        padded = np.full((rows + 2, cols + 2), pad_val, dtype=np.float64)
        padded[1:-1, 1:-1] = filled

        # Compute minimum of 8 neighbours using shifts
        nb_min = np.full_like(filled, pad_val)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = padded[1 + dr:rows + 1 + dr, 1 + dc:cols + 1 + dc]
                nb_min = np.minimum(nb_min, shifted)

        # A pit is a valid cell that is lower than (or equal to) all
        # its neighbours.  Raise it to neighbour_min + tiny gradient.
        is_pit = valid & (filled <= nb_min)
        if not np.any(is_pit):
            break

        filled = np.where(is_pit, nb_min + FLOW_MIN_SLOPE, filled)
        filled[~valid] = np.nan

    return filled


def _compute_d8_flowdir(dem_filled):
    """Compute D8 flow direction for a filled DEM.

    D8 encoding (ArcGIS convention):
        32  64  128
        16  --  1
         8   4   2

    0 = sink / no flow (should not happen after filling, but edge cells
    that flow off the grid are also 0).

    Parameters
    ----------
    dem_filled : 2D float array (pits already filled)

    Returns
    -------
    flowdir : 2D int32 array (D8 codes, 0 for nodata/edge-sink)
    """
    rows, cols = dem_filled.shape
    flowdir = np.zeros((rows, cols), dtype=np.int32)
    valid = ~np.isnan(dem_filled)

    # D8 neighbour offsets: (dr, dc, code)
    # Order: E, SE, S, SW, W, NW, N, NE
    d8_offsets = [
        (0,  1, 1),    # E
        (1,  1, 2),    # SE
        (1,  0, 4),    # S
        (1, -1, 8),    # SW
        (0, -1, 16),   # W
        (-1,-1, 32),   # NW
        (-1, 0, 64),   # N
        (-1, 1, 128),  # NE
    ]

    # Compute slopes to all 8 neighbours using vectorised shifts
    max_drop = np.full((rows, cols), -np.inf, dtype=np.float64)
    best_code = np.zeros((rows, cols), dtype=np.int32)

    padded = np.full((rows + 2, cols + 2), np.inf, dtype=np.float64)
    padded[1:-1, 1:-1] = dem_filled

    for dr, dc, code in d8_offsets:
        shifted = padded[1 + dr:rows + 1 + dr, 1 + dc:cols + 1 + dc]
        # For diagonal neighbours, distance is sqrt(2) * cell_size
        dist = np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
        drop = (dem_filled - shifted) / dist
        better = valid & (drop > max_drop) & (drop > FLOW_MIN_SLOPE)
        max_drop = np.where(better, drop, max_drop)
        best_code = np.where(better, code, best_code)

    flowdir = best_code.astype(np.int32)
    flowdir[~valid] = 0
    return flowdir


def _compute_flow_accumulation(flowdir):
    """Compute flow accumulation from D8 flow directions.

    Each cell contributes 1 (itself) plus the sum of all upstream cells
    that flow into it.

    Uses a topological-sort approach (Kahn's algorithm) for O(N) time.
    The downstream/in-degree arrays are built with vectorised operations
    for speed on large grids.

    Parameters
    ----------
    flowdir : 2D int32 array of D8 codes

    Returns
    -------
    accum : 2D float32 array (flow accumulation in pixel count)
    """
    rows, cols = flowdir.shape
    n = rows * cols
    accum = np.ones(n, dtype=np.float32)

    # D8 code -> (dr, dc) lookup
    d8_reverse = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }

    # --- Vectorised construction of downstream array ---
    # For each D8 code, compute the downstream flat index using shifts
    downstream = np.full(n, -1, dtype=np.int32)
    in_degree = np.zeros(n, dtype=np.int32)

    # Create flat index grid
    flat_idx = np.arange(n, dtype=np.int32).reshape(rows, cols)

    for code, (dr, dc) in d8_reverse.items():
        # Cells with this flow direction code
        mask = (flowdir == code)
        if not np.any(mask):
            continue
        # Compute downstream row/col for these cells
        src_rows, src_cols = np.where(mask)
        dst_rows = src_rows + dr
        dst_cols = src_cols + dc
        # Filter valid (in-bounds) targets
        valid = (dst_rows >= 0) & (dst_rows < rows) & \
                (dst_cols >= 0) & (dst_cols < cols)
        src_idx = src_rows[valid] * cols + src_cols[valid]
        dst_idx = dst_rows[valid] * cols + dst_cols[valid]
        downstream[src_idx] = dst_idx
        # Count in-degree: for each dst, how many src flow into it
        np.add.at(in_degree, dst_idx, 1)

    # --- Kahn's topological sort ---
    # Start with all cells that have in_degree == 0
    queue = deque(np.where(in_degree == 0)[0].tolist())

    while queue:
        idx = queue.popleft()
        d = downstream[idx]
        if d < 0:
            continue  # flows off-grid
        accum[d] += accum[idx]
        in_degree[d] -= 1
        if in_degree[d] == 0:
            queue.append(d)

    return accum.reshape(rows, cols)


def _compute_hand(dem, flowdir, drainage_mask):
    """Compute Height Above Nearest Drainage (HAND).

    For every pixel, trace the flow path downstream until reaching a
    drainage cell, then compute the vertical distance:
        HAND = DEM(pixel) - DEM(drainage cell)

    Uses the precomputed downstream chain (from flow direction) and
    processes cells in reverse topological order for efficiency.

    Parameters
    ----------
    dem : 2D float array (original, not filled)
    flowdir : 2D int32 array (D8 codes)
    drainage_mask : 2D bool array (True where drainage cell)

    Returns
    -------
    hand : 2D float32 array (vertical distance above nearest drainage, m)
    """
    rows, cols = dem.shape
    n = rows * cols
    hand_flat = np.full(n, np.nan, dtype=np.float32)
    dem_flat = dem.ravel()

    d8_reverse = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }

    # Build downstream array (vectorised)
    downstream = np.full(n, -1, dtype=np.int32)
    flat_idx = np.arange(n, dtype=np.int32).reshape(rows, cols)

    for code, (dr, dc) in d8_reverse.items():
        mask = (flowdir == code)
        if not np.any(mask):
            continue
        src_rows, src_cols = np.where(mask)
        dst_rows = src_rows + dr
        dst_cols = src_cols + dc
        valid = (dst_rows >= 0) & (dst_rows < rows) & \
                (dst_cols >= 0) & (dst_cols < cols)
        src_idx = src_rows[valid] * cols + src_cols[valid]
        dst_idx = dst_rows[valid] * cols + dst_cols[valid]
        downstream[src_idx] = dst_idx

    drain_flat = drainage_mask.ravel()
    # Drainage cells have HAND = 0
    hand_flat[drain_flat] = 0.0

    # For non-drainage cells, trace downstream until hitting a drainage cell
    # Process in topological order: start from cells closest to drainage
    # and work upstream
    valid_flat = ~np.isnan(dem_flat)

    # We need to trace each cell to its drainage outlet
    # Do this iteratively: in each pass, any cell whose downstream cell
    # already has a HAND value gets its HAND computed
    unresolved = valid_flat & ~drain_flat & np.isnan(hand_flat)

    max_passes = rows + cols  # worst-case path length
    for pass_num in range(max_passes):
        if not np.any(unresolved):
            break
        # Find cells whose downstream cell has a resolved HAND
        ds_idx = downstream[unresolved]
        # Filter to cells with valid downstream
        has_ds = ds_idx >= 0
        unresolved_idx = np.where(unresolved)[0]
        unresolved_with_ds = unresolved_idx[has_ds]
        ds_of_unresolved = ds_idx[has_ds]

        # Check which downstream cells have resolved HAND
        ds_resolved = ~np.isnan(hand_flat[ds_of_unresolved])
        to_resolve = unresolved_with_ds[ds_resolved]
        ds_resolved_idx = ds_of_unresolved[ds_resolved]

        if len(to_resolve) == 0:
            break

        # HAND = DEM(cell) - DEM(drainage cell)
        # The downstream cell's HAND already tells us the drainage elevation:
        #   hand_flat[ds] = dem_flat[ds] - dem_flat[drain]
        # So dem_flat[drain] = dem_flat[ds] - hand_flat[ds]
        # And hand_flat[cell] = dem_flat[cell] - dem_flat[drain]
        #   = dem_flat[cell] - (dem_flat[ds] - hand_flat[ds])
        #   = dem_flat[cell] - dem_flat[ds] + hand_flat[ds]
        drain_elev = dem_flat[ds_resolved_idx] - hand_flat[ds_resolved_idx]
        hand_flat[to_resolve] = dem_flat[to_resolve] - drain_elev

        # Update unresolved mask
        unresolved = valid_flat & ~drain_flat & np.isnan(hand_flat)

    return hand_flat.reshape(rows, cols)


def _compute_rem(dem, drainage_mask, flowdir):
    """Compute Relative Elevation Model (REM).

    REM = DEM - estimated_river_surface_elevation

    The river surface is estimated by interpolating elevation values
    from drainage cells across the DEM.  For each cell, the river
    surface elevation is taken as the elevation of the nearest drainage
    cell along the flow path (i.e. the drainage cell that water from
    this pixel would reach).

    Uses the same downstream-tracing approach as _compute_hand, but
    returns the absolute river surface elevation rather than the height
    difference.

    Parameters
    ----------
    dem : 2D float array
    drainage_mask : 2D bool array
    flowdir : 2D int32 array

    Returns
    -------
    rem : 2D float32 array (DEM - river_surface_elevation)
    river_surface : 2D float32 array (interpolated river surface elevation)
    """
    rows, cols = dem.shape
    n = rows * cols
    river_flat = np.full(n, np.nan, dtype=np.float32)
    rem_flat = np.full(n, np.nan, dtype=np.float32)
    dem_flat = dem.ravel()

    d8_reverse = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }

    # Build downstream array (vectorised)
    downstream = np.full(n, -1, dtype=np.int32)

    for code, (dr, dc) in d8_reverse.items():
        mask = (flowdir == code)
        if not np.any(mask):
            continue
        src_rows, src_cols = np.where(mask)
        dst_rows = src_rows + dr
        dst_cols = src_cols + dc
        valid = (dst_rows >= 0) & (dst_rows < rows) & \
                (dst_cols >= 0) & (dst_cols < cols)
        src_idx = src_rows[valid] * cols + src_cols[valid]
        dst_idx = dst_rows[valid] * cols + dst_cols[valid]
        downstream[src_idx] = dst_idx

    drain_flat = drainage_mask.ravel()
    # Drainage cells: river surface = DEM elevation, REM = 0
    river_flat[drain_flat] = dem_flat[drain_flat]
    rem_flat[drain_flat] = 0.0

    valid_flat = ~np.isnan(dem_flat)
    unresolved = valid_flat & ~drain_flat & np.isnan(river_flat)

    max_passes = rows + cols
    for pass_num in range(max_passes):
        if not np.any(unresolved):
            break
        ds_idx = downstream[unresolved]
        has_ds = ds_idx >= 0
        unresolved_idx = np.where(unresolved)[0]
        unresolved_with_ds = unresolved_idx[has_ds]
        ds_of_unresolved = ds_idx[has_ds]

        ds_resolved = ~np.isnan(river_flat[ds_of_unresolved])
        to_resolve = unresolved_with_ds[ds_resolved]
        ds_resolved_idx = ds_of_unresolved[ds_resolved]

        if len(to_resolve) == 0:
            break

        # River surface at cell = river surface at downstream cell
        river_flat[to_resolve] = river_flat[ds_resolved_idx]
        rem_flat[to_resolve] = dem_flat[to_resolve] - river_flat[to_resolve]

        unresolved = valid_flat & ~drain_flat & np.isnan(river_flat)

    return rem_flat.reshape(rows, cols), river_flat.reshape(rows, cols)


def _identify_river_valley_bed(dem, hand, rem, drainage_mask, resolution):
    """Context-aware identification of river bed and river valley pixels.

    Uses a multi-criteria approach that adapts to local terrain:

    1. **Fixed REM/HAND thresholds** — catches shallow river beds near
       ridges (REM < 2m) and low-lying valley floors (HAND < 10m).

    2. **TPI (Topographic Position Index)** — for every pixel, computes
       how far below the local mean it sits.  Cells that are >1 std below
       the local mean (within a 200m radius) are flagged as river beds;
       cells >0.5 std below are flagged as valley.  This catches deep
       incised channels in both flat terrain (60m depression in ~170m
       landscape) and mountainous terrain (0m vs 240m).

    3. **Local relief fraction** — for every pixel, computes its position
       within the local relief range (max - min over a 1km radius).  Cells
       in the bottom 15% of the relief range are river beds; bottom 35%
       are valley.  This adapts the threshold to the terrain: in flat
       terrain with 30m relief, the bottom 15% is 4.5m above the min;
       in mountainous terrain with 240m relief, it's 36m above the min.

    A pixel is classified as river bed if **any** criterion fires.
    River valley is the set of pixels flagged by valley criteria but
    not already classified as river bed.

    Parameters
    ----------
    dem : 2D float array (original DEM, may contain NaN)
    hand : 2D float array (HAND, may contain NaN)
    rem : 2D float array (REM, may contain NaN)
    drainage_mask : 2D bool array (drainage cells)
    resolution : float, pixel size in metres

    Returns
    -------
    river_bed_mask : 2D bool array
    river_valley_mask : 2D bool array
    """
    # Fallback: if DEM is not available, use HAND as elevation proxy
    if dem is None:
        dem = hand.copy()

    # Fallback: if drainage_mask is not available, derive from HAND == 0
    if drainage_mask is None:
        drainage_mask = (hand == 0) & ~np.isnan(hand)

    rows, cols = dem.shape
    valid = ~np.isnan(dem)

    # --- Compute TPI (Topographic Position Index) ---
    # TPI = cell - mean(neighbourhood)
    # Use uniform_filter for a fast box mean, then subtract.
    tpi_radius_px = max(1, int(TPI_WINDOW_RADIUS_M / resolution))
    relief_radius_px = max(1, int(RELIEF_WINDOW_RADIUS_M / resolution))

    dem_filled = np.where(valid, dem, np.nan)
    fill_val = np.nanmean(dem) if not np.all(np.isnan(dem)) else 0.0
    dem_safe = np.where(valid, dem, fill_val)

    # Local mean over TPI window (using uniform_filter = box mean)
    local_mean = uniform_filter(dem_safe, size=2 * tpi_radius_px + 1,
                                mode="nearest")
    # Local std: compute as sqrt(local_mean_sq - local_mean^2)
    local_mean_sq = uniform_filter(dem_safe ** 2, size=2 * tpi_radius_px + 1,
                                   mode="nearest")
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    tpi = dem_safe - local_mean  # negative = below surroundings

    # --- Compute local relief over wider window ---
    local_max = maximum_filter(dem_safe, size=2 * relief_radius_px + 1,
                               mode="nearest")
    local_min = minimum_filter(dem_safe, size=2 * relief_radius_px + 1,
                               mode="nearest")
    local_relief = local_max - local_min
    # Position within relief range: 0 = at minimum, 1 = at maximum
    relief_pos = np.where(local_relief > 1e-6,
                          (dem_safe - local_min) / local_relief, 0.5)

    # --- Criterion 1: Fixed REM/HAND thresholds (shallow beds, valley floor) ---
    bed_fixed = valid & (rem < RIVER_BED_REM_THRESHOLD) & (rem >= 0)
    valley_fixed = valid & (hand < VALLEY_HAND_THRESHOLD) & (hand >= 0)

    # --- Criterion 2: TPI-based adaptive ---
    # Only activate where local relief is significant enough
    has_relief = local_relief > MIN_RELIEF_FOR_ADAPTIVE

    bed_tpi = valid & has_relief & (tpi < -TPI_BED_FACTOR * local_std) & \
              (tpi < 0)  # must be below local mean
    valley_tpi = valid & has_relief & (tpi < -TPI_VALLEY_FACTOR * local_std) & \
                 (tpi < 0)

    # --- Criterion 3: Local relief fraction ---
    bed_relief = valid & has_relief & (relief_pos < RELIEF_BED_FRAC)
    valley_relief = valid & has_relief & (relief_pos < RELIEF_VALLEY_FRAC)

    # --- Combine: river bed = any criterion fires ---
    river_bed_mask = bed_fixed | bed_tpi | bed_relief

    # River valley = any valley criterion fires, minus river bed
    river_valley_mask = (valley_fixed | valley_tpi | valley_relief) & \
                        ~river_bed_mask

    # Ensure drainage cells are always included in river bed
    river_bed_mask = river_bed_mask | drainage_mask

    # Exclude nodata
    river_bed_mask = river_bed_mask & valid
    river_valley_mask = river_valley_mask & valid

    # Diagnostic counts
    n_bed = np.sum(river_bed_mask)
    n_valley = np.sum(river_valley_mask)
    n_bed_fixed = np.sum(bed_fixed & valid)
    n_bed_tpi = np.sum(bed_tpi & valid)
    n_bed_relief = np.sum(bed_relief & valid)
    n_has_relief = np.sum(has_relief & valid)
    print(f"  River bed detection:")
    print(f"    Fixed REM < {RIVER_BED_REM_THRESHOLD}m: {n_bed_fixed:,} pixels")
    print(f"    TPI < {-TPI_BED_FACTOR} std: {n_bed_tpi:,} pixels")
    print(f"    Relief fraction < {RELIEF_BED_FRAC}: {n_bed_relief:,} pixels")
    print(f"    Total bed (union): {n_bed:,} pixels")
    print(f"    Valley (excl. bed): {n_valley:,} pixels")
    print(f"    Cells with sufficient relief: {n_has_relief:,}")

    return river_bed_mask, river_valley_mask


def compute_rem_hand(dem_path, out_dir, dem_key, nodata=None,
                     accum_threshold=DRAINAGE_ACCUM_THRESHOLD,
                     max_pixels=8_000_000):
    """Compute REM and HAND for a DEM, saving .tif outputs.

    For large DEMs (LiDAR), reads at reduced resolution to stay within
    memory limits.

    Parameters
    ----------
    dem_path : Path to the DEM .tif
    out_dir : directory for output .tif files
    dem_key : string key for naming (e.g. "srtm1s")
    nodata : nodata value for the DEM
    accum_threshold : flow accumulation threshold for drainage cells
    max_pixels : cap on total pixels (downsamples if exceeded)

    Returns
    -------
    result : dict with keys 'hand', 'rem', 'river_surface',
             'drainage_mask', 'transform', 'extent', 'dem'
    """
    print(f"\n[REM/HAND] Computing for {dem_key}")

    # Read DEM
    with rasterio.open(str(dem_path)) as src:
        total_px = src.height * src.width
        ds_factor = 1
        if total_px > max_pixels:
            ds_factor = max(1, int(np.sqrt(total_px / max_pixels)))
            ov_h = src.height // ds_factor
            ov_w = src.width // ds_factor
            dem = src.read(1, out_shape=(ov_h, ov_w),
                           resampling=ResampEnum.average).astype(np.float64)
            transform = src.transform * src.transform.scale(
                float(src.width) / ov_w, float(src.height) / ov_h)
            print(f"  Downsampled by {ds_factor}x -> {dem.shape[0]}x{dem.shape[1]}")
        else:
            dem = src.read(1).astype(np.float64)
            transform = src.transform
        crs = src.crs

    # Handle nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem[dem < -1e6] = np.nan

    rows, cols = dem.shape
    print(f"  DEM shape: {rows} x {cols} = {rows * cols:,} pixels")

    # Step 1: Fill pits
    print("  Step 1: Filling pits...")
    dem_filled = _fill_pits(dem)

    # Step 2: Compute D8 flow direction
    print("  Step 2: Computing D8 flow direction...")
    flowdir = _compute_d8_flowdir(dem_filled)

    # Step 3: Compute flow accumulation
    print("  Step 3: Computing flow accumulation...")
    accum = _compute_flow_accumulation(flowdir)
    print(f"    Max accumulation: {accum.max():.0f}")

    # Step 4: Identify drainage cells
    drainage_mask = accum >= accum_threshold
    n_drain = np.sum(drainage_mask)
    print(f"  Step 4: Drainage cells (accum >= {accum_threshold}): {n_drain:,}")

    # Step 5: Compute HAND
    print("  Step 5: Computing HAND...")
    hand = _compute_hand(dem, flowdir, drainage_mask)

    # Step 6: Compute REM
    print("  Step 6: Computing REM...")
    rem, river_surface = _compute_rem(dem, drainage_mask, flowdir)

    # Step 7: Save outputs as .tif
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_raster(hand, transform, crs,
                 out_dir / f"{dem_key}_hand.tif", nodata=np.nan)
    _save_raster(rem, transform, crs,
                 out_dir / f"{dem_key}_rem.tif", nodata=np.nan)
    _save_raster(river_surface, transform, crs,
                 out_dir / f"{dem_key}_river_surface.tif", nodata=np.nan)
    # Also save flow accumulation
    _save_raster(accum, transform, crs,
                 out_dir / f"{dem_key}_flow_accumulation.tif",
                 nodata=0, dtype="float32")

    extent = _extent_from_transform(transform, dem.shape)

    return {
        "hand": hand,
        "rem": rem,
        "river_surface": river_surface,
        "drainage_mask": drainage_mask,
        "accum": accum,
        "flowdir": flowdir,
        "dem": dem,
        "transform": transform,
        "crs": crs,
        "extent": extent,
    }


def plot_river_valley_bed(dem_key, rem_hand_result, out_fig_path,
                           nodata=None):
    """Plot identified river valley and river bed as scattered points
    on top of hillshade DEM.

    Uses context-aware multi-criteria detection (_identify_river_valley_bed)
    that combines fixed REM/HAND thresholds with TPI and local-relief
    fraction to catch deep river beds in both flat and mountainous terrain.

    Outputs:
    - .png: scatter plot on hillshade
    - .tif: binary masks for river_valley and river_bed
    """
    print(f"\n[River Valley/Bed Plot] {dem_key}")

    dem = rem_hand_result["dem"]
    hand = rem_hand_result["hand"]
    rem = rem_hand_result["rem"]
    drainage_mask = rem_hand_result["drainage_mask"]
    transform = rem_hand_result["transform"]
    crs = rem_hand_result["crs"]
    extent = rem_hand_result["extent"]

    res = abs(transform.a)

    # --- Context-aware river bed/valley identification ---
    river_bed_mask, river_valley_mask = _identify_river_valley_bed(
        dem, hand, rem, drainage_mask, res
    )

    n_bed = np.sum(river_bed_mask)
    n_valley = np.sum(river_valley_mask)
    print(f"  River bed pixels: {n_bed:,}")
    print(f"  River valley pixels: {n_valley:,}")

    # Save binary masks as .tif
    out_dir = Path(out_fig_path).parent
    river_bed_tif = out_dir / f"{dem_key}_river_bed.tif"
    river_valley_tif = out_dir / f"{dem_key}_river_valley.tif"

    river_bed_arr = np.where(river_bed_mask, 1, 0).astype(np.uint8)
    river_valley_arr = np.where(river_valley_mask, 1, 0).astype(np.uint8)

    _save_raster(river_bed_arr, transform, crs,
                 river_bed_tif, nodata=255, dtype="uint8")
    _save_raster(river_valley_arr, transform, crs,
                 river_valley_tif, nodata=255, dtype="uint8")

    # Compute hillshade for background
    res = abs(transform.a)
    fill_val = np.nanmean(dem) if not np.all(np.isnan(dem)) else 0.0
    dem_filled = np.where(np.isnan(dem), fill_val, dem)
    dy, dx = np.gradient(dem_filled, res, res)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)
    az_rad = np.radians(315)
    alt_rad = np.radians(HS_ALTITUDE)
    hs = (np.sin(alt_rad) * np.cos(slope_rad) +
          np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad))
    hs = np.clip(hs, 0, 1)
    hs[np.isnan(dem)] = np.nan

    # Subsample for plotting (avoid too many scatter points)
    max_points = 50_000
    bed_rows, bed_cols = np.where(river_bed_mask)
    valley_rows, valley_cols = np.where(river_valley_mask)

    if len(bed_rows) > max_points:
        idx = np.random.default_rng(42).choice(len(bed_rows), max_points,
                                                replace=False)
        bed_rows = bed_rows[idx]
        bed_cols = bed_cols[idx]
    if len(valley_rows) > max_points:
        idx = np.random.default_rng(42).choice(len(valley_rows), max_points,
                                                replace=False)
        valley_rows = valley_rows[idx]
        valley_cols = valley_cols[idx]

    # Convert pixel coords to map coords
    bed_x = extent[0] + (bed_cols + 0.5) * res
    bed_y = extent[3] - (bed_rows + 0.5) * res
    valley_x = extent[0] + (valley_cols + 0.5) * res
    valley_y = extent[3] - (valley_rows + 0.5) * res

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(hs, cmap="gray", extent=extent, aspect="equal",
              vmin=0, vmax=1, origin="upper")

    if len(valley_x) > 0:
        ax.scatter(valley_x, valley_y, c="#29B6F6", s=1, alpha=0.3,
                   label="River valley (adaptive TPI + HAND)",
                   zorder=4)
    if len(bed_x) > 0:
        ax.scatter(bed_x, bed_y, c="#E91E63", s=1, alpha=0.5,
                   label="River bed (adaptive REM + TPI + relief)",
                   zorder=5)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.8,
              markerscale=5)
    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_title(f"River Valley & River Bed (pixel-level)  |  "
                 f"REM + HAND  |  {dem_key}", fontsize=11,
                 fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] River valley/bed plot -> {Path(out_fig_path).name}")


# ===================================================================
# 2.1 BASE CARTOGRAPHIC PRODUCTS
# ===================================================================

def _compute_hillshade_from_dem(dem_path, azimuth=315, altitude=8,
                                 nodata=None, max_pixels=5_000_000):
    """Compute hillshade on-the-fly from a DEM .tif file.

    Used as a fallback when pre-computed .npy derivatives are not
    available (e.g. LiDAR tiled derivatives).

    For very large DEMs, reads at reduced resolution to keep memory
    usage manageable.

    Returns (hs_data, transform) or (None, None).
    """
    try:
        with rasterio.open(dem_path) as src:
            res = abs(src.res[0])
            transform = src.transform
            total_px = src.height * src.width

            if total_px > max_pixels:
                ds_factor = max(1, int(np.sqrt(total_px / max_pixels)))
                ov_h = src.height // ds_factor
                ov_w = src.width // ds_factor
                dem = src.read(1, out_shape=(ov_h, ov_w),
                                resampling=ResampEnum.average).astype(np.float32)
                transform = src.transform * src.transform.scale(
                    float(src.width) / ov_w, float(src.height) / ov_h)
                res = res * ds_factor
            else:
                dem = src.read(1).astype(np.float32)
    except Exception as e:
        print(f"  [!] Failed to read DEM for hillshade: {e}")
        return None, None

    if nodata is not None:
        mask = (dem == nodata) | np.isnan(dem) | (dem < -1e6)
        dem[mask] = np.nan
    else:
        mask = np.isnan(dem)

    fill_val = np.nanmean(dem) if not np.all(np.isnan(dem)) else 0.0
    filled = np.where(np.isnan(dem), fill_val, dem)

    dy, dx = np.gradient(filled, res, res)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    hs = (np.sin(alt_rad) * np.cos(slope_rad) +
          np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad))
    hs = np.clip(hs, 0, 1)
    hs[mask] = np.nan

    return hs, transform


def _load_hillshade_with_fallback(dem_key, deriv_dir, dem_path=None,
                                   nodata=None, azimuth=315):
    """Load hillshade from .npy, or compute on-the-fly from DEM.

    For LiDAR and other DEMs where .npy derivatives are unavailable,
    falls back to computing hillshade directly from the DEM .tif.

    Returns (hs_data, meta_or_None) or (None, None).
    """
    dem_stem = get_dem_stem(dem_key)
    hs_data, hs_meta = load_derivative(deriv_dir, dem_stem,
                                        f"hillshade_az{azimuth}")

    if hs_data is None and dem_path is not None:
        print(f"  [Hillshade Fallback] Computing on-the-fly from DEM for {dem_key}")
        hs_data, hs_transform = _compute_hillshade_from_dem(
            dem_path, azimuth=azimuth, altitude=HS_ALTITUDE,
            nodata=nodata
        )
        if hs_data is not None:
            hs_meta = {
                "transform": [hs_transform.a, hs_transform.b, hs_transform.c,
                              hs_transform.d, hs_transform.e, hs_transform.f],
                "crs": "EPSG:28355",
                "shape": list(hs_data.shape),
                "nodata": nodata,
                "resolution": abs(hs_transform.a),
                "type": f"hillshade_az{azimuth}",
            }

    return hs_data, hs_meta


def plot_hillshade_composite(dem_key, deriv_dir, out_fig_path, nodata=None,
                              dem_path=None):
    """
    Multi-azimuth hillshade composite: blend all 4 azimuths into a single
    false-colour composite (RGB = az315/az45/az135), plus a grayscale mean.
    Loads from .npy + .json, falls back to .tif or on-the-fly computation.
    """
    print(f"\n[Hillshade Composite] {dem_key}")

    dem_stem = get_dem_stem(dem_key)
    bands = {}
    ref_meta = None

    for az in HS_AZIMUTHS:
        deriv_name = f"hillshade_az{az}"
        data, meta = load_derivative(deriv_dir, dem_stem, deriv_name)
        if data is None and dem_path is not None:
            data, meta = _load_hillshade_with_fallback(
                dem_key, deriv_dir, dem_path=dem_path,
                nodata=nodata, azimuth=az
            )
        if data is not None:
            data = np.nan_to_num(data, nan=0.0)
            data = np.clip(data, 0, 1)
            bands[az] = data
            if ref_meta is None:
                ref_meta = meta

    if len(bands) < 3:
        print(f"  [!] Need at least 3 hillshade bands, found {len(bands)}")
        return

    # Ensure all same shape
    ref_shape = list(bands.values())[0].shape
    for az in list(bands.keys()):
        if bands[az].shape != ref_shape:
            print(f"  [!] Shape mismatch for az={az}: {bands[az].shape} vs {ref_shape}")
            del bands[az]

    if len(bands) < 3:
        print(f"  [!] Insufficient matching bands")
        return

    # RGB composite: R=az315, G=az45, B=az135
    rgb = np.dstack([
        bands.get(315, bands.get(45, list(bands.values())[0])),
        bands.get(45, list(bands.values())[1]),
        bands.get(135, bands.get(225, list(bands.values())[2])),
    ])

    # Normalise each channel
    for i in range(3):
        ch = rgb[:, :, i]
        lo, hi = np.percentile(ch[ch > 0], [2, 98]) if np.any(ch > 0) else (0, 1)
        rgb[:, :, i] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)

    # Mean grayscale
    all_bands = np.stack(list(bands.values()), axis=-1)
    gray = np.mean(all_bands, axis=-1)

    # Compute extent from metadata
    if ref_meta is not None:
        t = ref_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        extent = _extent_from_transform(transform, ref_shape)
    else:
        extent = [0, ref_shape[1], 0, ref_shape[0]]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    im0 = axes[0].imshow(gray, cmap="gray", extent=extent, aspect="equal",
                          vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Mean Hillshade (4 azimuths)", fontsize=11)
    axes[0].set_xlabel("Easting (m)", fontsize=9)
    axes[0].set_ylabel("Northing (m)", fontsize=9)

    axes[1].imshow(rgb, extent=extent, aspect="equal", origin="upper")
    axes[1].set_title("RGB Composite (R=315, G=45, B=135 deg)", fontsize=11)
    axes[1].set_xlabel("Easting (m)", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7)

    crs_label = "EPSG:28355"
    plt.suptitle(f"Multi-Azimuth Hillshade Composite  |  {dem_key}  |  {crs_label}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Hillshade composite -> {out_fig_path.name}")


def plot_svf_openness_comparison(dem_key, deriv_dir, out_fig_path, nodata=None):
    """
    SVF/Openness overlay comparison map: side-by-side hillshade + SVF,
    and hillshade + positive openness, showing how each reveals lineaments.
    Loads from .npy + .json, falls back to .tif.
    """
    print(f"\n[SVF/Openness Comparison] {dem_key}")

    dem_stem = get_dem_stem(dem_key)

    # Load hillshade (az315)
    hs_data, hs_meta = load_derivative(deriv_dir, dem_stem, "hillshade_az315")
    if hs_data is None:
        print(f"  [!] Missing hillshade raster")
        return
    hs_data = np.nan_to_num(hs_data, nan=0.0)

    # Load SVF
    svf_data, svf_meta = load_derivative(deriv_dir, dem_stem, "svf")
    if svf_data is None:
        print(f"  [!] Missing SVF raster")
        return
    svf_data = np.nan_to_num(svf_data, nan=0.5)

    # Load openness
    op_data, op_meta = load_derivative(deriv_dir, dem_stem, "openness_pos")
    if op_data is not None:
        op_data = np.nan_to_num(op_data, nan=90.0)

    # Compute extent from hillshade metadata
    if hs_meta is not None:
        t = hs_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        extent = _extent_from_transform(transform, hs_data.shape)
    else:
        extent = [0, hs_data.shape[1], 0, hs_data.shape[0]]

    n_panels = 3 if op_data is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8))

    # Hillshade
    axes[0].imshow(hs_data, cmap="gray", extent=extent, aspect="equal",
                    vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Hillshade (az 315 deg, alt 8 deg)", fontsize=10)
    axes[0].set_xlabel("Easting (m, EPSG:28355)", fontsize=8)
    axes[0].set_ylabel("Northing (m, EPSG:28355)", fontsize=8)

    # SVF
    im1 = axes[1].imshow(svf_data, cmap="inferno", extent=extent, aspect="equal",
                          vmin=0.3, vmax=1.0, origin="upper")
    axes[1].set_title("Sky-View Factor", fontsize=10)
    axes[1].set_xlabel("Easting (m, EPSG:28355)", fontsize=8)
    fig.colorbar(im1, ax=axes[1], shrink=0.7, label="SVF")

    # Openness
    if op_data is not None:
        im2 = axes[2].imshow(op_data, cmap="RdYlBu", extent=extent, aspect="equal",
                              origin="upper")
        axes[2].set_title("Positive Openness (deg)", fontsize=10)
        axes[2].set_xlabel("Easting (m, EPSG:28355)", fontsize=8)
        fig.colorbar(im2, ax=axes[2], shrink=0.7, label="deg")

    for ax in axes:
        ax.tick_params(labelsize=7)

    plt.suptitle(f"SVF / Openness Comparison  |  {dem_key}", fontsize=12,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SVF/Openness comparison -> {out_fig_path.name}")


# ===================================================================
# 2.2 GEOMORPHIC EVIDENCE MAPS
# ===================================================================

def extract_lineaments_from_pca(pca_data, pca_meta, dem_path, out_dir,
                                 nodata=None, min_length_m=LINE_MIN_LENGTH_M):
    """
    Semi-automated lineament extraction from PCA hillshade composite.
    Uses Canny edge detection on PC1, then chains edges into lineaments.
    Filters by minimum length.

    pca_data: numpy array (2D for PC1, or 3D for multi-band)
    pca_meta: metadata dict from .json
    """
    print(f"\n[Lineament Extraction]")

    # Use first band (PC1) if multi-band
    if pca_data.ndim == 3:
        pc1 = pca_data[0]
    else:
        pc1 = pca_data

    pc1 = np.nan_to_num(pc1, nan=0.0)

    # Normalise to 0-1 for Canny
    lo, hi = np.percentile(pc1, [1, 99])
    pc1_norm = np.clip((pc1 - lo) / max(hi - lo, 1e-6), 0, 1)

    # Canny edge detection
    from skimage.feature import canny
    edges = canny(pc1_norm, sigma=LINE_CANNY_SIGMA,
                  low_threshold=LINE_CANNY_LOW, high_threshold=LINE_CANNY_HIGH)

    # Get resolution and transform from metadata
    if pca_meta is not None:
        t = pca_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        res = pca_meta.get("resolution", 1.0)
    elif dem_path is not None and Path(dem_path).exists():
        with rasterio.open(dem_path) as src:
            res = abs(src.res[0])
            transform = src.transform
    else:
        res = 1.0
        transform = Affine(1, 0, 0, 0, -1, 0)

    # Label connected components
    labelled, n_features = label(edges)
    print(f"  Canny edges: {n_features} connected components")

    lineaments = []
    for feat_id in range(1, min(n_features + 1, 5000)):
        feat_mask = labelled == feat_id
        n_px = np.sum(feat_mask)
        if n_px < 3:
            continue

        rows, cols = np.where(feat_mask)
        length_px = max(rows.max() - rows.min(), cols.max() - cols.min())
        length_m = length_px * res

        if length_m < min_length_m:
            continue

        # Convert pixel coords to map coords
        coords_map = np.array([rio_xy(transform, r, c) for r, c in
                                zip(rows[::max(1, n_px // 100)],
                                    cols[::max(1, n_px // 100)])])

        # Compute azimuth via PCA of point positions
        if len(coords_map) >= 3:
            pca = SkPCA(n_components=1)
            pca.fit(coords_map)
            azimuth = np.degrees(np.arctan2(pca.components_[0, 1],
                                            pca.components_[0, 0])) % 180
        else:
            azimuth = 0

        # Centroid
        cx_map, cy_map = np.mean(rows), np.mean(cols)
        cx, cy = rio_xy(transform, cx_map, cy_map)

        lineaments.append({
            "lineament_id": len(lineaments),
            "x": round(cx, 0),
            "y": round(cy, 0),
            "length_m": round(length_m, 0),
            "azimuth_deg": round(azimuth, 0),
            "n_pixels": int(n_px),
        })

    df = pd.DataFrame(lineaments)
    print(f"  Extracted {len(df)} lineaments (min length={min_length_m}m)")

    if len(df) > 0:
        csv_path = out_dir / "lineaments.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

    return df


def plot_lineament_density_map(dem_key, lineament_df, deriv_dir, out_fig_path,
                                nodata=None, bandwidth_m=2000):
    """
    Kernel density map of extracted lineaments overlaid on hillshade.
    Identifies structural corridors with anomalously high concentrations.
    """
    print(f"\n[Lineament Density Map] {dem_key}")

    if len(lineament_df) == 0:
        print("  [!] No lineaments to plot")
        return

    dem_stem = get_dem_stem(dem_key)

    # Load hillshade from .npy
    hs_data, hs_meta = load_derivative(deriv_dir, dem_stem, "hillshade_az315")
    if hs_data is None:
        print("  [!] No hillshade raster found")
        return

    hs_data = np.nan_to_num(hs_data, nan=0.0)

    if hs_meta is not None:
        t = hs_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        extent = _extent_from_transform(transform, hs_data.shape)
        res = hs_meta.get("resolution", 1.0)
    else:
        extent = [0, hs_data.shape[1], 0, hs_data.shape[0]]
        res = 1.0

    # Compute KDE from lineament centroids
    xy = lineament_df[["x", "y"]].values
    if len(xy) < 3:
        print("  [!] Too few lineaments for KDE")
        return

    # Create output grid
    ny, nx = hs_data.shape
    x_grid = np.linspace(extent[0], extent[1], nx)
    y_grid = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([X.ravel(), Y.ravel()])

    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth_m, metric="euclidean")
    kde.fit(xy)
    density = np.exp(kde.score_samples(grid_coords)).reshape(ny, nx)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(hs_data, cmap="gray", extent=extent, aspect="equal",
              vmin=0, vmax=1, origin="upper")

    im = ax.imshow(density, cmap="hot", extent=extent, aspect="equal",
                    alpha=0.5, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Lineament density (KDE)")

    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_title("Lineament Density Map  |  Kernel Density Estimation", fontsize=12,
                 fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Lineament density -> {out_fig_path.name}")


def plot_drainage_anomaly_map(dem_key, drain_csv, traces, deriv_dir,
                               out_fig_path, nodata=None,
                               rem_hand_result=None, dem_path=None):
    """
    Drainage anomaly map: river valley and river bed identified at pixel
    level using adaptive REM/HAND + TPI + local relief criteria, plotted
    as scattered points on hillshade DEM.

    Also overlays drainage anomalies from the Phase 1 CSV for reference.

    Outputs:
    - .png: scatter plot on hillshade
    - .tif: binary masks saved by plot_river_valley_bed
    """
    print(f"\n[Drainage Anomaly Map] {dem_key}")

    dem_stem = get_dem_stem(dem_key)

    # Load hillshade from .npy or compute on-the-fly
    hs_data, hs_meta = _load_hillshade_with_fallback(
        dem_key, deriv_dir, dem_path=dem_path, nodata=nodata, azimuth=315
    )
    if hs_data is None:
        print(f"  [!] Could not load hillshade for {dem_key}, skipping")
        return
    hs_data = np.nan_to_num(hs_data, nan=0.0)

    if hs_meta is not None:
        t = hs_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        extent = _extent_from_transform(transform, hs_data.shape)
    else:
        extent = [0, hs_data.shape[1], 0, hs_data.shape[0]]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(hs_data, cmap="gray", extent=extent, aspect="equal",
              vmin=0, vmax=1, origin="upper")

    # --- Plot river valley and river bed from REM/HAND if available ---
    if rem_hand_result is not None:
        dem_arr = rem_hand_result["dem"]
        hand = rem_hand_result["hand"]
        rem = rem_hand_result["rem"]
        drainage_mask = rem_hand_result["drainage_mask"]
        rem_transform = rem_hand_result["transform"]
        rem_extent = rem_hand_result["extent"]
        res = abs(rem_transform.a)

        # --- Context-aware river bed/valley identification ---
        river_bed_mask, river_valley_mask = _identify_river_valley_bed(
            dem_arr, hand, rem, drainage_mask, res
        )

        # Subsample for plotting
        max_points = 50_000
        bed_rows, bed_cols = np.where(river_bed_mask)
        valley_rows, valley_cols = np.where(river_valley_mask)

        if len(bed_rows) > max_points:
            idx = np.random.default_rng(42).choice(
                len(bed_rows), max_points, replace=False)
            bed_rows = bed_rows[idx]
            bed_cols = bed_cols[idx]
        if len(valley_rows) > max_points:
            idx = np.random.default_rng(42).choice(
                len(valley_rows), max_points, replace=False)
            valley_rows = valley_rows[idx]
            valley_cols = valley_cols[idx]

        bed_x = rem_extent[0] + (bed_cols + 0.5) * res
        bed_y = rem_extent[3] - (bed_rows + 0.5) * res
        valley_x = rem_extent[0] + (valley_cols + 0.5) * res
        valley_y = rem_extent[3] - (valley_rows + 0.5) * res

        if len(valley_x) > 0:
            ax.scatter(valley_x, valley_y, c="#29B6F6", s=1, alpha=0.3,
                       label="River valley (adaptive TPI + HAND)",
                       zorder=4)
        if len(bed_x) > 0:
            ax.scatter(bed_x, bed_y, c="#E91E63", s=1, alpha=0.5,
                       label="River bed (adaptive REM + TPI + relief)",
                       zorder=5)
    else:
        print("  [!] No REM/HAND result provided -- plotting CSV anomalies only")

    # Fault traces (thin grey)
    for trace in traces:
        c = trace["coords"]
        ax.plot(c[:, 0], c[:, 1], color="gray", linewidth=0.4, alpha=0.5)

    # Drainage anomalies from CSV (if available, for reference)
    if Path(drain_csv).exists() and Path(drain_csv).stat().st_size > 0:
        drain_df = pd.read_csv(drain_csv)
        if len(drain_df) > 0:
            type_config = {
                "deflected_stream": {"color": "#00FFFF", "marker": "D",
                                     "s": 30, "label": "Deflected stream"},
                "linear_valley":    {"color": "#FF00FF", "marker": "s",
                                     "s": 25, "label": "Linear valley"},
                "depression_cluster": {"color": "#FFFF00", "marker": "o",
                                       "s": 20, "label": "Depression cluster"},
            }
            for atype, cfg in type_config.items():
                sub = drain_df[drain_df["anomaly_type"] == atype]
                if len(sub) > 0:
                    ax.scatter(sub["x"], sub["y"], c=cfg["color"],
                               marker=cfg["marker"], s=cfg["s"],
                               edgecolors="black", linewidths=0.3,
                               label=cfg["label"], zorder=6, alpha=0.8)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.8,
              markerscale=4)
    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_title("Drainage Anomaly Map  |  River Valley & Bed (pixel-level "
                 "REM + HAND)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Drainage anomaly map -> {out_fig_path.name}")


def plot_expression_quality_4class(dem_key, traces, class_df, deriv_dir,
                                    out_fig_path, lidar_class_df=None,
                                    nodata=None, dem_path=None):
    """
    Expression Quality Classification Map with 4 classes (A/B/C/D),
    evaluated at per-pixel level within a 200m buffer of each fault trace.

    Quality is evaluated per pixel using slope and SVF anomalies, rather
    than per trace.  Each pixel within the 200m buffer is independently
    classified:
        A: strong slope gradient (>= 5 deg) OR strong SVF anomaly (>= 0.15)
        B: moderate slope gradient (>= 2 deg) OR moderate SVF anomaly (>= 0.08)
        C: weak slope gradient (>= 1 deg)
        D: no expression (below all thresholds, or outside buffer)

    **One figure is produced per fault source** (FSD1, CWvic1, GeoStru2501,
    GeoStru501).  Each figure shows only that source's scatter points with
    its 4-shade colour palette:
        FSD1:         light orange (A) / orange (B) / dark orange (C) / grey (D)
        CWvic1:       light blue (A) / blue (B) / dark blue (C) / grey (D)
        GeoStru2501:  light green (A) / green (B) / dark green (C) / grey (D)
        GeoStru501:   light red (A) / red (B) / dark red (C) / grey (D)

    Output files are named:
        {out_fig_path stem}_{source}.png
    e.g. srtm1s_expression_quality_4class_FSD1.png
    """
    print(f"\n[Expression Quality 4-Class Map (per-pixel, per-source)] {dem_key}")

    # Load hillshade from .npy, or compute on-the-fly from DEM
    hs_data, hs_meta = _load_hillshade_with_fallback(
        dem_key, deriv_dir, dem_path=dem_path, nodata=nodata, azimuth=315
    )
    if hs_data is None:
        print(f"  [!] Could not load or compute hillshade for {dem_key}, skipping")
        return
    hs_data = np.nan_to_num(hs_data, nan=0.0)

    if hs_meta is not None:
        t = hs_meta.get("transform", [1, 0, 0, 0, -1, 0])
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        extent = _extent_from_transform(transform, hs_data.shape)
        res = hs_meta.get("resolution", 1.0)
    else:
        extent = [0, hs_data.shape[1], 0, hs_data.shape[0]]
        res = 1.0

    # --- Load slope and SVF for per-pixel evaluation ---
    dem_stem = get_dem_stem(dem_key)
    slope_data, slope_meta = load_derivative(deriv_dir, dem_stem, "slope")
    svf_data, svf_meta = load_derivative(deriv_dir, dem_stem, "svf")

    # If slope not in .npy, compute from DEM at hillshade resolution
    if slope_data is None and dem_path is not None:
        print(f"  Computing slope on-the-fly from DEM...")
        try:
            with rasterio.open(str(dem_path)) as src:
                # Read DEM at the SAME resolution as the hillshade grid,
                # guaranteeing shape match and correct pixel size.
                dem_arr = src.read(1, out_shape=hs_data.shape,
                                   resampling=ResampEnum.average
                                   ).astype(np.float32)
                # Use the hillshade transform so pixel size is correct
                if hs_meta is not None:
                    t = hs_meta.get("transform", [1, 0, 0, 0, -1, 0])
                    dem_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
                else:
                    dem_transform = src.transform
            if nodata is not None:
                dem_arr[dem_arr == nodata] = np.nan
            dem_arr[dem_arr < -1e6] = np.nan
            fill_val = np.nanmean(dem_arr) if not np.all(np.isnan(dem_arr)) else 0.0
            filled = np.where(np.isnan(dem_arr), fill_val, dem_arr)
            res_slope = abs(dem_transform.a)
            dy, dx = np.gradient(filled, res_slope, res_slope)
            slope_data = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            print(f"  Slope computed at hs resolution: {slope_data.shape}")
        except Exception as e:
            print(f"  Could not compute slope: {e}")
            slope_data = None

    if svf_data is not None and svf_data.shape != hs_data.shape:
        print(f"  SVF shape {svf_data.shape} != hs {hs_data.shape}")
        svf_data = None

    # --- Build trace_id -> source and trace_id -> class lookup ---
    # Build LiDAR cross-validation lookup if available
    lidar_lookup = {}
    if lidar_class_df is not None and len(lidar_class_df) > 0:
        for _, lr in lidar_class_df.iterrows():
            lidar_lookup[lr["trace_id"]] = CLASS_MAP_3TO4.get(
                lr.get("quality_class", "Class 3"), "C")

    # Build trace_id -> (source, base_class) lookup
    trace_info = {}
    for _, row in class_df.iterrows():
        base_class = CLASS_MAP_3TO4.get(row.get("quality_class", "Class 3"), "C")
        if (row.get("detection_rate", 0) == 0 and
            row.get("n_drain_anomalies", 0) == 0 and
            row.get("mean_scarp_height_m", 0) == 0):
            base_class = "D"

        tid = row["trace_id"]
        if tid in lidar_lookup:
            lidar_cls = lidar_lookup[tid]
            if lidar_cls in ("A", "B") and base_class in ("C", "D"):
                base_class = "B"

        source = row.get("source", "Unknown")
        trace_info[tid] = {"source": source, "base_class": base_class}

    # Also get source from traces if available
    trace_source_lookup = {}
    if traces:
        for trace in traces:
            trace_source_lookup[trace["trace_id"]] = trace.get("source", "")

    # --- Create per-pixel classification within 200m buffer ---
    rows, cols = hs_data.shape
    buffer_px = int(EXPR_PIXEL_BUFFER_M / res)
    buffer_px = max(1, buffer_px)

    # Collect scatter points grouped by source
    # source_scatter[source] = {"x": [], "y": [], "color": [], "class": []}
    # Pre-compute SVF anomaly if SVF is available
    svf_anomaly = None
    if svf_data is not None:
        svf_clean = np.nan_to_num(svf_data, nan=1.0)
        svf_mean = np.nanmean(svf_clean)
        svf_anomaly = np.abs(svf_clean - svf_mean)

    if slope_data is not None:
        slope_clean = np.nan_to_num(slope_data, nan=0.0)
    else:
        slope_clean = None

    # --- Grid-based per-pixel classification ---
    # Build one unified buffer mask per source, so each pixel is classified
    # exactly once (no duplicate scatter points from overlapping traces).
    from scipy.ndimage import binary_dilation

    # Classify the entire grid at once using vectorised operations
    slope_grid = slope_clean if slope_clean is not None else np.zeros_like(hs_data)
    svf_grid = svf_anomaly if svf_anomaly is not None else np.zeros_like(hs_data)

    # Vectorised classification: A > B > C > D (priority order)
    is_A = (slope_grid >= EXPR_PIX_CLASS_A_SLOPE) | (svf_grid >= EXPR_PIX_CLASS_A_SVF)
    is_B = ((slope_grid >= EXPR_PIX_CLASS_B_SLOPE) | (svf_grid >= EXPR_PIX_CLASS_B_SVF)) & ~is_A
    is_C = (slope_grid >= EXPR_PIX_CLASS_C_SLOPE) & ~is_A & ~is_B
    is_D = ~is_A & ~is_B & ~is_C

    # Class grid (0=A, 1=B, 2=C, 3=D)
    class_grid = np.full(hs_data.shape, 3, dtype=np.uint8)  # default D
    class_grid[is_C] = 2
    class_grid[is_B] = 1
    class_grid[is_A] = 0

    # --- Build per-source buffer masks ---
    # Each source gets a single boolean mask covering all its traces' 200m
    # buffers.  Pixels are then classified from the global class_grid.
    source_masks = {src: np.zeros((rows, cols), dtype=bool)
                    for src in SOURCE_CLASS_COLOURS}

    if traces:
        # Rasterise all traces, tagging each pixel with its source
        for trace in traces:
            tid = trace["trace_id"]
            source = trace_source_lookup.get(tid,
                   trace_info.get(tid, {}).get("source", "Unknown"))
            if source not in source_masks:
                source_masks[source] = np.zeros((rows, cols), dtype=bool)
            coords = trace["coords"]
            t_mask = np.zeros((rows, cols), dtype=bool)
            for i in range(len(coords)):
                cx_map, cy_map = coords[i]
                col, row_t = ~transform * (cx_map, cy_map)
                col, row_t = int(round(col)), int(round(row_t))
                if 0 <= row_t < rows and 0 <= col < cols:
                    t_mask[row_t, col] = True
            if buffer_px > 0:
                structure = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1),
                                    dtype=bool)
                t_mask = binary_dilation(t_mask, structure=structure,
                                         iterations=1)
            source_masks[source] |= t_mask
    else:
        # Fallback: use class_df centroids with buffer
        for _, row in class_df.iterrows():
            tid = row["trace_id"]
            source = row.get("source", "Unknown")
            if source not in source_masks:
                source_masks[source] = np.zeros((rows, cols), dtype=bool)
            cx, cy = row["centroid_x"], row["centroid_y"]
            col, row_t = ~transform * (cx, cy)
            col, row_t = int(round(col)), int(round(row_t))
            if not (0 <= row_t < rows and 0 <= col < cols):
                continue
            r0 = max(0, row_t - buffer_px)
            r1 = min(rows, row_t + buffer_px + 1)
            c0 = max(0, col - buffer_px)
            c1 = min(cols, col + buffer_px + 1)
            source_masks[source][r0:r1, c0:c1] = True

    # --- Collect scatter points per source (vectorised) ---
    source_scatter = {}
    for src_key in SOURCE_CLASS_COLOURS:
        mask = source_masks.get(src_key, np.zeros((rows, cols), dtype=bool))
        pts_r, pts_c = np.where(mask)
        n_pts = len(pts_r)
        if n_pts == 0:
            source_scatter[src_key] = {"x": [], "y": [], "color": [], "class": []}
            continue
        # Subsample if too many points
        max_pts = 200_000
        if n_pts > max_pts:
            idx = np.random.default_rng(42).choice(n_pts, max_pts, replace=False)
            pts_r = pts_r[idx]
            pts_c = pts_c[idx]
        classes = class_grid[pts_r, pts_c]
        xs = extent[0] + (pts_c + 0.5) * res
        ys = extent[3] - (pts_r + 0.5) * res
        palette = SOURCE_CLASS_COLOURS[src_key]
        colours = [palette[c] for c in classes]
        class_names = ["A", "B", "C", "D"]
        class_labels = [class_names[c] for c in classes]
        source_scatter[src_key] = {
            "x": xs, "y": ys, "color": colours, "class": class_labels
        }

    # --- Plot: one figure per source ---
    out_path = Path(out_fig_path)
    out_stem = out_path.stem  # e.g. "srtm1s_expression_quality_4class"
    out_dir = out_path.parent

    from matplotlib.lines import Line2D

    for source, palette in SOURCE_CLASS_COLOURS.items():
        sd = source_scatter.get(source, {"x": [], "y": [], "color": [], "class": []})
        n_pts = len(sd["x"])
        print(f"  [{source}] {n_pts:,} scatter points")

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(hs_data, cmap="gray", extent=extent, aspect="equal",
                  vmin=0, vmax=1, origin="upper")

        if n_pts > 0:
            # Plot D (grey) first, then C, B, A on top so higher quality
            # points are not hidden by overlapping grey D points
            sd_class = np.array(sd["class"])
            sd_x = np.array(sd["x"])
            sd_y = np.array(sd["y"])
            sd_color = np.array(sd["color"])
            for cls_name in ["D", "C", "B", "A"]:
                cls_mask = sd_class == cls_name
                if np.any(cls_mask):
                    ax.scatter(sd_x[cls_mask], sd_y[cls_mask],
                               c=sd_color[cls_mask], s=3, alpha=0.5,
                               label=f"Class {cls_name}", zorder=4)

        # Legend: 4 class entries for this source only
        legend_handles = []
        for cls_name, cls_idx in [("A", 0), ("B", 1), ("C", 2), ("D", 3)]:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[cls_idx], markersize=8,
                       label=f"Class {cls_name}")
            )

        ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                  framealpha=0.8, title=f"{source} — Quality Class",
                  markerscale=1.5)
        ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
        ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
        ax.set_title(f"Expression Quality — {source}  "
                     f"(per-pixel, {EXPR_PIXEL_BUFFER_M}m buffer)  "
                     f"|  DEM: {dem_key}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Fault traces (thin grey) for context
        for trace in traces:
            c = trace["coords"]
            ax.plot(c[:, 0], c[:, 1], color="gray", linewidth=0.4, alpha=0.3)

        src_out_path = out_dir / f"{out_stem}_{source}.png"
        plt.tight_layout()
        fig.savefig(src_out_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    [OK] -> {src_out_path.name}")

    # Return per-pixel class counts for the summary
    pixel_counts = {}
    for source, sd in source_scatter.items():
        for cls_name in ["A", "B", "C", "D"]:
            n = sum(1 for c in sd["class"] if c == cls_name)
            pixel_counts[cls_name] = pixel_counts.get(cls_name, 0) + n
    return pixel_counts


# ===================================================================
# 2.3 QUANTITATIVE INDEX VISUALISATION
# ===================================================================

def compute_geomorphic_indices(dem_key, dem_path, deriv_dir, traces, struct_pts,
                                nodata=None, buffer_m=500):
    """
    Compute geomorphic indices per fault trace segment:
      - Smf (Mountain Front Sinuosity)
      - Vf (Valley Floor Width-to-Height Ratio)
      - HI (Hypsometric Integral)
      - SL (Stream Length-Gradient Index) -- proxy from slope
      - AF (Drainage Basin Asymmetry Factor) -- proxy from TPI asymmetry

    Loads slope from .npy, falls back to .tif or on-the-fly computation.
    """
    print(f"\n[Geomorphic Indices] Computing per fault segment for {dem_key}")

    dem_stem = get_dem_stem(dem_key)

    # Try loading slope from .npy
    slope_data, slope_meta = load_derivative(deriv_dir, dem_stem, "slope")
    slope_transform = None
    if slope_data is not None and slope_meta is not None:
        t = slope_meta.get("transform", [1, 0, 0, 0, -1, 0])
        slope_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    results = []

    for trace in tqdm(traces, desc="  Indices", leave=False):
        tid = trace["trace_id"]
        coords = trace["coords"]
        cx, cy = trace["centroid_x"], trace["centroid_y"]

        # Smf: sinuosity = trace_length / straight_line_distance
        straight_dist = np.sqrt((coords[-1, 0] - coords[0, 0])**2 +
                                (coords[-1, 1] - coords[0, 1])**2)
        if straight_dist > 0:
            smf = trace["length_m"] / straight_dist
        else:
            smf = 1.0

        # Sample slope and elevation at the trace centroid
        slope_val = 0.0
        dem_val = 0.0

        if slope_data is not None and slope_transform is not None:
            # Sample from loaded slope array
            col, row = ~slope_transform * (cx, cy)
            row, col = int(round(row)), int(round(col))
            if 0 <= row < slope_data.shape[0] and 0 <= col < slope_data.shape[1]:
                v = slope_data[row, col]
                if not (np.isnan(v) or v < -1e6):
                    slope_val = float(v)

        if slope_val == 0.0:
            # Fallback: try .tif
            slope_tif = deriv_dir / f"{dem_stem}_slope.tif"
            if slope_tif.exists():
                slope_val = _sample_raster_value(slope_tif, cx, cy)

        dem_val = _sample_raster_value(dem_path, cx, cy)

        # HI proxy: simplified (full HI would need catchment delineation)
        hi = 0.5

        # SL proxy: slope * distance from drainage divide
        sl_proxy = slope_val * 100

        results.append({
            "trace_id": tid,
            "source": trace["source"],
            "centroid_x": round(cx, 0),
            "centroid_y": round(cy, 0),
            "Smf": round(smf, 2),
            "Vf": round(0.5, 2),
            "HI": round(hi, 3),
            "SL_proxy": round(sl_proxy, 1),
            "AF": 50.0,
            "slope_at_fault_deg": round(slope_val, 2),
            "elevation_m": round(dem_val, 1) if dem_val > -1e6 else np.nan,
        })

    df = pd.DataFrame(results)
    print(f"  Computed indices for {len(df)} fault segments")
    return df


def plot_radar_charts(indices_df, out_dir, max_charts=20):
    """
    Radar/spider charts per fault segment showing normalised geomorphic indices.
    """
    print(f"\n[Radar Charts] Generating up to {max_charts}")
    safe_makedirs(out_dir)

    if len(indices_df) == 0:
        print("  No indices to plot")
        return

    idx_cols = ["Smf", "Vf", "HI", "SL_proxy", "AF"]
    norm_df = indices_df.copy()
    for col in idx_cols:
        if col in norm_df.columns:
            lo = norm_df[col].min()
            hi = norm_df[col].max()
            if hi - lo > 1e-6:
                norm_df[col + "_norm"] = (norm_df[col] - lo) / (hi - lo)
            else:
                norm_df[col + "_norm"] = 0.5

    if "Smf_norm" in norm_df.columns:
        norm_df["Smf_inv"] = 1 - norm_df["Smf_norm"]

    top = indices_df.nlargest(min(max_charts, len(indices_df)), "SL_proxy")

    radar_cols = ["Smf_inv", "Vf_norm", "HI_norm", "SL_proxy_norm", "AF_norm"]
    radar_labels = ["Smf (inv)", "Vf", "HI", "SL", "AF"]
    n_axes = len(radar_labels)

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    for _, row in top.iterrows():
        values = []
        for col in radar_cols:
            if col in norm_df.columns:
                values.append(norm_df.loc[row.name, col] if row.name in norm_df.index else 0.5)
            else:
                values.append(0.5)
        values += values[:1]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.plot(angles, values, "o-", linewidth=1.5, color="steelblue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Fault Segment {int(row['trace_id'])}  |  "
                     f"Source: {row['source']}\n"
                     f"Smf={row['Smf']:.2f}  SL={row['SL_proxy']:.1f}  "
                     f"Slope={row['slope_at_fault_deg']:.1f} deg",
                     fontsize=9, pad=15)

        fig_path = out_dir / f"radar_trace_{int(row['trace_id']):04d}.png"
        fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Generated {len(top)} radar charts")


def plot_longitudinal_profiles(dem_path, traces, deriv_dir, dem_key,
                                out_dir, nodata=None, max_profiles=10):
    """
    Longitudinal river profiles for streams crossing fault traces.
    Annotates knickpoints where SL index spikes.
    """
    print(f"\n[Longitudinal Profiles] Generating up to {max_profiles}")
    safe_makedirs(out_dir)

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])

    for i, trace in enumerate(traces[:max_profiles]):
        coords = trace["coords"]
        cx, cy = trace["centroid_x"], trace["centroid_y"]

        half_len = 3000
        n_pts = int(2 * half_len / res)

        fault_az = np.radians(trace["azimuth_deg"])
        perp_az = fault_az + np.pi / 2

        dist = np.linspace(-half_len, half_len, min(n_pts, 500))
        px = cx + dist * np.sin(perp_az)
        py = cy + dist * np.cos(perp_az)

        with rasterio.open(dem_path) as src:
            xy_pairs = list(zip(px, py))
            elevs = np.array([v[0] for v in src.sample(xy_pairs)], dtype=np.float32)
            if nodata is not None:
                elevs[elevs == nodata] = np.nan
            elevs[elevs < -1e6] = np.nan

        valid = ~np.isnan(elevs)
        if np.sum(valid) < 10:
            continue

        elev_smooth = np.interp(np.arange(len(elevs)),
                                np.where(valid)[0], elevs[valid])
        delev = np.abs(np.gradient(elev_smooth, dist[1] - dist[0]))
        sl_index = delev * np.abs(dist)

        sl_threshold = np.nanpercentile(sl_index[sl_index > 0], 90) if np.any(sl_index > 0) else 1
        knick_mask = sl_index > sl_threshold

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(dist, elev_smooth, "k-", linewidth=0.8)
        ax1.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                     label="Fault trace")
        if np.any(knick_mask):
            ax1.scatter(dist[knick_mask], elev_smooth[knick_mask], c="red",
                        s=20, zorder=5, label="Knickpoint")
        ax1.set_ylabel("Elevation (m AHD)", fontsize=9)
        ax1.set_title(f"Longitudinal Profile across Fault {trace['trace_id']}  |  "
                      f"Source: {trace['source']}", fontsize=10)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2.plot(dist, sl_index, "b-", linewidth=0.6)
        ax2.axhline(sl_threshold, color="red", linestyle=":", linewidth=0.5,
                     label=f"SL threshold={sl_threshold:.0f}")
        ax2.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.set_xlabel("Distance from fault trace (m)", fontsize=9)
        ax2.set_ylabel("SL Index", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / f"river_profile_{trace['trace_id']:04d}.png"
        fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Longitudinal profiles saved")


def plot_hypsometric_curves(dem_path, traces, out_dir, nodata=None,
                             max_curves=10):
    """
    Hypsometric curves per sub-catchment near fault traces.
    HI = (mean_elev - min_elev) / (max_elev - min_elev)
    """
    print(f"\n[Hypsometric Curves] Generating up to {max_curves}")
    safe_makedirs(out_dir)

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])

    for trace in traces[:max_curves]:
        cx, cy = trace["centroid_x"], trace["centroid_y"]

        half = 1000
        with rasterio.open(dem_path) as src:
            row_c, col_c = ~src.transform * (cx, cy)
            row_c, col_c = int(row_c), int(col_c)
            half_px = int(half / abs(src.res[0]))
            r0 = max(0, row_c - half_px)
            r1 = min(src.height, row_c + half_px)
            c0 = max(0, col_c - half_px)
            c1 = min(src.width, col_c + half_px)
            if r1 <= r0 or c1 <= c0:
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)
            dem = src.read(1, window=win).astype(np.float32)

        if nodata is not None:
            dem[dem == nodata] = np.nan
        dem[dem < -1e6] = np.nan

        valid = dem[~np.isnan(dem)]
        if len(valid) < 100:
            continue

        elev_sorted = np.sort(valid)
        n = len(elev_sorted)
        min_e = elev_sorted[0]
        max_e = elev_sorted[-1]
        mean_e = np.mean(elev_sorted)

        h = (elev_sorted - min_e) / max(max_e - min_e, 1e-6)
        a = np.arange(1, n + 1) / n
        hi = (mean_e - min_e) / max(max_e - min_e, 1e-6)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(a, h, "b-", linewidth=1.5)
        ax.fill_between(a, 0, h, alpha=0.15, color="blue")
        ax.set_xlabel("Area ratio (a/A)", fontsize=10)
        ax.set_ylabel("Elevation ratio (h/H)", fontsize=10)
        ax.set_title(f"Hypsometric Curve  |  Fault {trace['trace_id']}  |  "
                     f"HI = {hi:.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        x_ref = np.linspace(0, 1, 100)
        ax.plot(x_ref, x_ref**2, "r--", linewidth=0.5, alpha=0.5, label="Convex (young)")
        ax.plot(x_ref, 1 - (1 - x_ref)**2, "g--", linewidth=0.5, alpha=0.5,
                label="Concave (mature)")
        ax.legend(fontsize=8)

        fig_path = out_dir / f"hypsometric_{trace['trace_id']:04d}.png"
        fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Hypsometric curves saved")


# ===================================================================
# 2.4 UNCERTAINTY AND CONFIDENCE LAYERS
# ===================================================================

def plot_confidence_heatmap(dem_paths, out_fig_path, nodata=None):
    """
    Confidence heatmap showing spatial data density:
      - LiDAR coverage extent vs. 30m-only coverage vs. 90m-only coverage
      - Makes data-quality gaps immediately visible

    Coverage levels:
      0 = no data
      1 = SRTM 3s only (~90m)
      2 = SRTM 1s + 3s (~30m)
      3 = GA 10m + SRTM 1s + 3s
      4 = LiDAR 0.5m + 10m + SRTM
    """
    print(f"\n[Confidence Heatmap] Data density / coverage")

    # Use SRTM 3s as reference grid (lowest resolution = widest extent);
    # fall back to SRTM 1s if 3s is not available.
    ref_key = "srtm3s" if "srtm3s" in dem_paths else "srtm1s"
    with rasterio.open(dem_paths[ref_key]) as src:
        ref_shape = (src.height, src.width)
        ref_transform = src.transform
        ref_bounds = src.bounds
        res = abs(src.res[0])

    coverage = np.zeros(ref_shape, dtype=np.float32)

    # SRTM 3s valid pixels (level 1)
    if "srtm3s" in dem_paths and Path(dem_paths["srtm3s"]).exists():
        with rasterio.open(dem_paths["srtm3s"]) as src:
            srtm3s = src.read(1).astype(np.float32)
            nd3s = src.nodata if src.nodata is not None else -32768.0
            valid_srtm3s = (srtm3s != nd3s) & (srtm3s > -1e6) & (~np.isnan(srtm3s))
            coverage[valid_srtm3s] = 1.0

    # SRTM 1s valid pixels (level 2)
    if "srtm1s" in dem_paths and Path(dem_paths["srtm1s"]).exists():
        try:
            srtm1s_valid = np.zeros(ref_shape, dtype=np.float32)
            with rasterio.open(dem_paths["srtm1s"]) as src:
                srtm1s_mask = np.zeros((src.height, src.width), dtype=np.float32)
                srtm1s = src.read(1).astype(np.float32)
                nd1s = src.nodata if src.nodata is not None else -999.0
                valid_block = ((srtm1s != nd1s) & (srtm1s > -1e6) &
                               (~np.isnan(srtm1s))).astype(np.float32)
                srtm1s_mask[:] = valid_block
                reproject(
                    source=srtm1s_mask,
                    destination=srtm1s_valid,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.nearest,
                )
                del srtm1s_mask, srtm1s, valid_block
            coverage[srtm1s_valid > 0.5] = 2.0
        except Exception as e:
            print(f"  [!] Could not process SRTM 1s for coverage: {e}")

    # 10m DEM valid pixels (resample to reference grid)
    if "dem10m" in dem_paths and Path(dem_paths["dem10m"]).exists():
        try:
            dem10m_valid = np.zeros(ref_shape, dtype=np.float32)
            with rasterio.open(dem_paths["dem10m"]) as src:
                dem10m_mask = np.zeros((src.height, src.width), dtype=np.float32)
                block_rows = 2000
                for r0 in range(0, src.height, block_rows):
                    r1 = min(r0 + block_rows, src.height)
                    win = Window(0, r0, src.width, r1 - r0)
                    block = src.read(1, window=win).astype(np.float32)
                    nodata_val = src.nodata if src.nodata is not None else -3.4e38
                    valid_block = ((block != nodata_val) & (block > -1e6) &
                                   (~np.isnan(block))).astype(np.float32)
                    dem10m_mask[r0:r1, :] = valid_block
                    del block, valid_block
                reproject(
                    source=dem10m_mask,
                    destination=dem10m_valid,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.nearest,
                )
                del dem10m_mask
            coverage[dem10m_valid > 0.5] = 3.0
        except Exception as e:
            print(f"  [!] Could not process 10m DEM for coverage: {e}")

    # LiDAR valid pixels -- use bounds-based approximation for huge files
    if "lidar_0_5m" in dem_paths and Path(dem_paths["lidar_0_5m"]).exists():
        try:
            lidar_valid = np.zeros(ref_shape, dtype=np.float32)
            with rasterio.open(dem_paths["lidar_0_5m"]) as src:
                uncompress_gb = src.width * src.height * 4 / (1024**3)
                file_size_gb = Path(dem_paths["lidar_0_5m"]).stat().st_size / (1024**3)
                use_bounds_approx = (uncompress_gb > 10 or file_size_gb > 10)
                if use_bounds_approx:
                    print(f"  LiDAR file is {file_size_gb:.1f} GB -- using bounds approximation")
                    lb = src.bounds
                    xs = np.array([rio_xy(ref_transform, 0, c)[0]
                                   for c in range(ref_shape[1])])
                    ys = np.array([rio_xy(ref_transform, r, 0)[1]
                                   for r in range(ref_shape[0])])
                    x_mask = (xs >= lb.left) & (xs <= lb.right)
                    y_mask = (ys >= lb.bottom) & (ys <= lb.top)
                    lidar_valid[np.ix_(y_mask, x_mask)] = 1.0
                else:
                    lidar_mask = np.zeros((src.height, src.width), dtype=np.float32)
                    block_rows = 2000
                    for r0 in range(0, src.height, block_rows):
                        r1 = min(r0 + block_rows, src.height)
                        win = Window(0, r0, src.width, r1 - r0)
                        block = src.read(1, window=win).astype(np.float32)
                        valid_block = ((block != -9999.0) & (block > -1e6) &
                                       (~np.isnan(block))).astype(np.float32)
                        lidar_mask[r0:r1, :] = valid_block
                        del block, valid_block
                    reproject(
                        source=lidar_mask,
                        destination=lidar_valid,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.nearest,
                    )
                    del lidar_mask
            coverage[lidar_valid > 0.5] = 4.0
        except Exception as e:
            print(f"  [!] Could not process LiDAR for coverage: {e}")

    coverage_smooth = gaussian_filter(coverage, sigma=2)

    extent = [ref_bounds.left, ref_bounds.right, ref_bounds.bottom, ref_bounds.top]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    cmap = LinearSegmentedColormap.from_list("coverage",
        [(0, "#FFFFFF"), (0.25, "#D5F5E3"), (0.5, "#AED6F1"),
         (0.75, "#2E86C1"), (1.0, "#1B4F72")])

    im = ax.imshow(coverage_smooth, cmap=cmap, extent=extent, aspect="equal",
                    vmin=0, vmax=4, origin="upper")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(["No data", "SRTM 3s 90m only",
                          "SRTM 1s 30m + 3s", "GA 10m + SRTM",
                          "LiDAR 0.5m + 10m + SRTM"])

    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_title("Data Coverage Confidence Heatmap  |  Resolution Availability",
                 fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Confidence heatmap -> {out_fig_path.name}")


def plot_multi_resolution_comparison(dem_paths, traces, out_dir,
                                     nodata_srtm3s=-32768.0,
                                     nodata_srtm=-999.0, nodata_10m=-3.4e38,
                                     nodata_lidar=-9999.0, n_tiles=5,
                                     struct_pts=None, fault_proximity_m=500):
    """
    Multi-resolution comparison panels: side-by-side tile panels showing
    the same fault location at 90m, 30m, 10m, and 0.5m resolution.

    Supports fault-aware tile selection: prioritises tiles near fault traces.
    Always uses windowed reads for LiDAR (42.8 GB).
    """
    print(f"\n[Multi-Resolution Comparison] {n_tiles} tiles")
    safe_makedirs(out_dir)

    # Fault-aware tile selection: prioritises traces with structural points nearby
    rng = np.random.default_rng(42)

    # Score traces by proximity to structural points
    if struct_pts is not None and len(struct_pts) > 0:
        pt_coords = struct_pts[["XCoord", "YCoord"]].values.astype(np.float64)
        pt_tree = cKDTree(pt_coords)
        for trace in traces:
            n_near = len(pt_tree.query_ball_point(
                [trace["centroid_x"], trace["centroid_y"]], fault_proximity_m))
            trace["_n_struct_pts"] = n_near
        # Sort by structural point count (most fault-proximate first)
        selected_traces = sorted(
            [t for t in traces if t["length_m"] > 500],
            key=lambda t: -t.get("_n_struct_pts", 0))
    else:
        selected_traces = [t for t in traces if t["length_m"] > 500]
        rng.shuffle(selected_traces)

    if len(selected_traces) > n_tiles:
        selected_traces = selected_traces[:n_tiles]

    for i, trace in enumerate(selected_traces):
        cx, cy = trace["centroid_x"], trace["centroid_y"]
        half = 1000

        panels = []
        labels = []
        extents_list = []

        # SRTM 3s (~90m)
        if "srtm3s" in dem_paths and Path(dem_paths["srtm3s"]).exists():
            try:
                srtm3s_tile, srtm3s_tf, srtm3s_bounds, _ = load_raster_window(
                    dem_paths["srtm3s"], cx, cy, half, nodata=nodata_srtm3s)
                panels.append(srtm3s_tile)
                labels.append("SRTM 90m")
                extents_list.append(srtm3s_bounds)
            except Exception:
                panels.append(None)
                labels.append("SRTM 90m (N/A)")
                extents_list.append(None)

        # SRTM 1s (~30m)
        try:
            srtm_tile, srtm_tf, srtm_bounds, _ = load_raster_window(
                dem_paths["srtm1s"], cx, cy, half, nodata=nodata_srtm)
            panels.append(srtm_tile)
            labels.append("SRTM 30m")
            extents_list.append(srtm_bounds)
        except Exception:
            panels.append(None)
            labels.append("SRTM 30m (N/A)")
            extents_list.append(None)

        # DEM 10m
        try:
            dem10m_tile, dem10m_tf, dem10m_bounds, _ = load_raster_window(
                dem_paths["dem10m"], cx, cy, half, nodata=nodata_10m)
            panels.append(dem10m_tile)
            labels.append("GA 10m")
            extents_list.append(dem10m_bounds)
        except Exception:
            panels.append(None)
            labels.append("GA 10m (N/A)")
            extents_list.append(None)

        # LiDAR 0.5m (always windowed read from .tif)
        try:
            lidar_tile, lidar_tf, lidar_bounds, _ = load_raster_window(
                dem_paths["lidar_0_5m"], cx, cy, half, nodata=nodata_lidar)
            panels.append(lidar_tile)
            labels.append("LiDAR 0.5m")
            extents_list.append(lidar_bounds)
        except Exception:
            panels.append(None)
            labels.append("LiDAR 0.5m (N/A)")
            extents_list.append(None)

        # Plot
        valid_panels = [(p, l, e) for p, l, e in zip(panels, labels, extents_list)
                        if p is not None]
        if len(valid_panels) == 0:
            continue

        n = len(valid_panels)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        for ax, (data, label, ext) in zip(axes, valid_panels):
            data_clean = np.nan_to_num(data, nan=0.0)
            data_clean[data_clean < -1e6] = 0
            lo, hi = np.percentile(data_clean[data_clean > 0], [2, 98]) \
                if np.any(data_clean > 0) else (0, 1)

            if ext is not None:
                im = ax.imshow(data_clean, cmap="terrain", aspect="equal",
                               vmin=lo, vmax=hi, origin="upper",
                               extent=[ext[0], ext[1], ext[2], ext[3]])
                ax.set_xlabel("Easting (m)", fontsize=8)
                ax.set_ylabel("Northing (m)", fontsize=8)
            else:
                im = ax.imshow(data_clean, cmap="terrain", aspect="equal",
                               vmin=lo, vmax=hi, origin="upper")

            ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")

        plt.suptitle(f"Multi-Resolution Comparison  |  Fault {trace['trace_id']}  |  "
                     f"Centre: ({cx:.0f}, {cy:.0f})  EPSG:28355",
                     fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = out_dir / f"multires_{i+1:02d}_{cx:.0f}_{cy:.0f}.png"
        fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Multi-resolution comparison panels saved")


# ===================================================================
# 3.3 CONFIDENCE SCORING
# ===================================================================

def compute_confidence_scores(class_df, indices_df, traces, dem_res=30.0,
                               lidar_class_df=None):
    """
    Compute a composite confidence score (0-10) per fault segment based on:
      - Number of independent geomorphic indicators (scarp, drainage, slope, SVF)
      - Resolution of confirming DEM data
      - Absence of non-tectonic alternative explanations
      - Aleatory/epistemic uncertainty classification
      - Cross-validation with LiDAR results (if available)

    Also assigns aleatory vs. epistemic uncertainty tier.
    """
    print(f"\n[Confidence Scores] Computing per fault segment (DEM res={dem_res}m)")

    min_scarp = MIN_DETECTABLE_SCARP.get(dem_res, 3.0)

    # Build LiDAR cross-validation lookup
    lidar_lookup = {}
    if lidar_class_df is not None and len(lidar_class_df) > 0:
        for _, lr in lidar_class_df.iterrows():
            lidar_lookup[lr["trace_id"]] = lr

    results = []

    for _, row in class_df.iterrows():
        score = 0

        # 1. Scarp evidence (0-3 points)
        h = row.get("mean_scarp_height_m", 0)
        if h >= min_scarp:
            score += 3
        elif h >= min_scarp * 0.5:
            score += 2
        elif h > 0:
            score += 1

        # 2. Detection consistency (0-2 points)
        det_rate = row.get("detection_rate", 0)
        if det_rate >= 0.7:
            score += 2
        elif det_rate >= 0.3:
            score += 1

        # 3. Drainage anomaly corroboration (0-2 points)
        n_drain = row.get("n_drain_anomalies", 0)
        if n_drain >= 3:
            score += 2
        elif n_drain >= 1:
            score += 1

        # 4. Slope/SVF anomaly (0-1 point each)
        if row.get("slope_anomaly_deg", 0) >= EXPR_CLASS2_SLOPE:
            score += 1
        if row.get("svf_anomaly", 0) >= EXPR_CLASS2_SVFDIFF:
            score += 1

        # 5. DEM resolution bonus (0-1)
        if dem_res <= 1.0:
            score += 1  # LiDAR

        # 6. LiDAR cross-validation bonus (0-1)
        tid = row["trace_id"]
        if tid in lidar_lookup:
            lidar_row = lidar_lookup[tid]
            lidar_h = lidar_row.get("mean_scarp_height_m", 0)
            lidar_det = lidar_row.get("detection_rate", 0)
            # If LiDAR confirms the scarp, add bonus
            if lidar_h > 0 or lidar_det > 0:
                score += 1

        # Cap at 10
        score = min(10, score)

        # Aleatory vs. epistemic
        if h < min_scarp and h > 0:
            uncertainty_tier = "aleatory"
            uncertainty_note = (f"Scarp {h:.2f}m below min detectable "
                                f"{min_scarp:.1f}m for {dem_res:.0f}m DEM")
        elif det_rate > 0 and det_rate < 0.5:
            uncertainty_tier = "epistemic"
            uncertainty_note = ("Inconsistent scarp detection -- needs higher-res "
                                "data or field check")
        elif h == 0 and n_drain == 0:
            uncertainty_tier = "aleatory"
            uncertainty_note = "No surface expression detected at this resolution"
        else:
            uncertainty_tier = "low"
            uncertainty_note = "Multiple concordant indicators"

        # Note LiDAR cross-validation status
        lidar_status = "N/A"
        if tid in lidar_lookup:
            lidar_cls = CLASS_MAP_3TO4.get(
                lidar_lookup[tid].get("quality_class", "Class 3"), "C")
            lidar_status = f"LiDAR Class {lidar_cls}"

        results.append({
            "trace_id": row["trace_id"],
            "source": row.get("source", ""),
            "confidence_score": score,
            "uncertainty_tier": uncertainty_tier,
            "uncertainty_note": uncertainty_note,
            "min_detectable_scarp_m": min_scarp,
            "dem_resolution_m": dem_res,
            "lidar_cross_validation": lidar_status,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"  Confidence score distribution:")
        for s in range(0, 11):
            n = (df["confidence_score"] == s).sum()
            if n > 0:
                print(f"    Score {s:2d}: {n} segments")
        print(f"  Uncertainty tiers: {df['uncertainty_tier'].value_counts().to_dict()}")
        if lidar_class_df is not None:
            n_validated = (df["lidar_cross_validation"] != "N/A").sum()
            print(f"  LiDAR cross-validated: {n_validated} segments")
    return df


# ===================================================================
# 3.4 CROSS-VALIDATION AGAINST PUBLISHED ANALOGUES
# ===================================================================

def cross_validate_analogues(class_df, indices_df, lidar_class_df=None):
    """
    Benchmark results against published analogue values:
      - Yallourn Monocline: 1-2m scarps
      - Korean scarps: 2-4m scarps
      - Chinese Smf/Vf ranges for active vs. inactive fronts

    If LiDAR Phase 1 results are available, use them for higher-confidence
    cross-validation of scarp heights.
    """
    print(f"\n[Cross-Validation] Benchmarking against published analogues")

    # Build LiDAR lookup for scarp height cross-validation
    lidar_lookup = {}
    if lidar_class_df is not None and len(lidar_class_df) > 0:
        for _, lr in lidar_class_df.iterrows():
            lidar_lookup[lr["trace_id"]] = lr.get("mean_scarp_height_m", 0)

    results = []

    for _, row in class_df.iterrows():
        h = row.get("mean_scarp_height_m", 0)

        # If LiDAR data available, use the higher-resolution scarp height
        tid = row["trace_id"]
        h_lidar = lidar_lookup.get(tid, None)
        h_best = h  # best available scarp height
        h_source = "coarse_DEM"
        if h_lidar is not None and h_lidar > 0:
            h_best = h_lidar
            h_source = "LiDAR"

        smf = None
        idx_row = indices_df[indices_df["trace_id"] == tid]
        if len(idx_row) > 0:
            smf = idx_row.iloc[0].get("Smf")

        vs_yallourn = "within" if YALLOURN_SCARP_RANGE[0] <= h_best <= YALLOURN_SCARP_RANGE[1] else \
                      "above" if h_best > YALLOURN_SCARP_RANGE[1] else "below"
        vs_korean = "within" if KOREAN_SCARP_RANGE[0] <= h_best <= KOREAN_SCARP_RANGE[1] else \
                    "above" if h_best > KOREAN_SCARP_RANGE[1] else "below"

        smf_assessment = "unknown"
        if smf is not None:
            if REF_SMF_ACTIVE[0] <= smf <= REF_SMF_ACTIVE[1]:
                smf_assessment = "active_range"
            elif smf > REF_SMF_ACTIVE[1]:
                smf_assessment = "inactive_range"
            else:
                smf_assessment = "very_active"

        res_sufficiency = "sufficient" if h_best >= MIN_DETECTABLE_SCARP.get(30.0, 3.0) else \
                          "need_10m" if h_best >= MIN_DETECTABLE_SCARP.get(10.0, 1.0) else \
                          "need_LiDAR"

        results.append({
            "trace_id": tid,
            "source": row.get("source", ""),
            "scarp_height_m": round(h, 2),
            "scarp_height_best_m": round(h_best, 2),
            "height_source": h_source,
            "vs_yallourn_1_2m": vs_yallourn,
            "vs_korean_2_4m": vs_korean,
            "Smf": smf,
            "Smf_assessment": smf_assessment,
            "resolution_sufficiency": res_sufficiency,
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"  Cross-validation summary ({len(df)} segments):")
        print(f"    vs. Yallourn (1-2m): {df['vs_yallourn_1_2m'].value_counts().to_dict()}")
        print(f"    vs. Korean (2-4m): {df['vs_korean_2_4m'].value_counts().to_dict()}")
        print(f"    Resolution sufficiency: {df['resolution_sufficiency'].value_counts().to_dict()}")
        if lidar_class_df is not None:
            n_lidar = (df["height_source"] == "LiDAR").sum()
            print(f"    LiDAR-confirmed scarp heights: {n_lidar}")

    return df


# ===================================================================
# MAIN WORKFLOW
# ===================================================================

def main():
    print("=" * 80)
    print("02-VisualAudit.py -- Visualisation Suite & Uncertainty Assessment")
    print("=" * 80)
    print(f"Available RAM: {mem_gb():.1f} GB")

    # ---- TEST MODE: only process SRTM 3s, skip all other DEMs ----
    TEST_ONLY_SRTM3S = True

    safe_makedirs(OUT_DIR)

    # -- Load data from Phase 1 outputs --
    print("\n--- Loading Phase 1 outputs ---")

    # Structural points
    if not STRUCT_PTS_PATH.exists():
        print("[!] Structural points file not found")
        return

    # Get SRTM 3s or SRTM 1s extent for filtering
    _extent_dem = SRTM3S_RP if (TEST_ONLY_SRTM3S and SRTM3S_RP.exists()) else (
        SRTM_RP if SRTM_RP.exists() else None)
    if _extent_dem is not None and _extent_dem.exists():
        with rasterio.open(_extent_dem) as src:
            b = src.bounds
            extent = (b.left, b.right, b.bottom, b.top)
    else:
        print("[!] No reprojected DEM found -- run 01-DerivativeGen.py first")
        return

    struct_pts = load_structural_points(STRUCT_PTS_PATH, extent=extent)
    traces = reconstruct_fault_traces(struct_pts, cluster_dist_m=30, min_pts=5)

    if not traces:
        print("[!] No fault traces reconstructed -- will use CSV centroids for plotting")
        traces = []

    # -- DEMs to process --
    dems = {}
    if not TEST_ONLY_SRTM3S and SRTM3S_RP.exists():
        dems["srtm3s"] = {"path": SRTM3S_RP, "deriv": SRTM3S_DERIV, "res": 90.0,
                           "nodata": -32768.0}
    if not TEST_ONLY_SRTM3S and SRTM_RP.exists():
        dems["srtm1s"] = {"path": SRTM_RP, "deriv": SRTM_DERIV, "res": 30.0,
                           "nodata": -999.0}
    if not TEST_ONLY_SRTM3S and DEM10M_RP.exists():
        dems["dem10m"] = {"path": DEM10M_RP, "deriv": DEM10M_DERIV, "res": 10.0,
                           "nodata": -3.4e38}
    if not TEST_ONLY_SRTM3S and LIDAR_RP.exists():
        dems["lidar_0_5m"] = {"path": LIDAR_RP, "deriv": LIDAR_DERIV, "res": 0.5,
                               "nodata": -9999.0}

    # In test mode, force only srtm3s
    if TEST_ONLY_SRTM3S and SRTM3S_RP.exists():
        dems = {"srtm3s": {"path": SRTM3S_RP, "deriv": SRTM3S_DERIV, "res": 90.0,
                            "nodata": -32768.0}}

    print(f"\n  DEMs available for visual audit: {list(dems.keys())}")

    # Load Phase 1 classification CSVs
    class_dfs = {}
    class_point_dfs = {}
    scarp_dfs = {}
    drain_dfs = {}
    for dem_key in dems:
        class_csv = AUDIT_DIR / f"{dem_key}_expression_quality.csv"
        class_points_csv = AUDIT_DIR / f"{dem_key}_expression_quality_points.csv"
        scarp_csv = AUDIT_DIR / f"{dem_key}_scarp_analysis.csv"
        drain_csv = AUDIT_DIR / f"{dem_key}_drainage_anomalies.csv"
        if class_points_csv.exists():
            class_point_dfs[dem_key] = pd.read_csv(class_points_csv)
        if class_csv.exists():
            class_dfs[dem_key] = pd.read_csv(class_csv)
        elif dem_key in class_point_dfs:
            class_dfs[dem_key] = _summarise_expression_quality_points(
                class_point_dfs[dem_key], traces=traces)
        if scarp_csv.exists():
            scarp_dfs[dem_key] = pd.read_csv(scarp_csv)
        if drain_csv.exists():
            drain_dfs[dem_key] = pd.read_csv(drain_csv)

    # LiDAR Phase 1 results for cross-validation
    lidar_class_df = class_dfs.get("lidar_0_5m", None)
    if lidar_class_df is not None:
        print(f"  LiDAR Phase 1 classification available ({len(lidar_class_df)} segments)")

    # ===================================================================
    # 2.1 BASE CARTOGRAPHIC PRODUCTS
    # ===================================================================
    print(f"\n{'='*60}")
    print("2.1 BASE CARTOGRAPHIC PRODUCTS")
    print(f"{'='*60}")

    # Collect per-pixel quality counts (returned by plot_expression_quality_4class)
    pixel_quality_counts = {}

    for dem_key, info in dems.items():
        # Hillshade composite
        out_path = OUT_DIR / f"{dem_key}_hillshade_composite.png"
        if not out_path.exists():
            plot_hillshade_composite(dem_key, info["deriv"], out_path,
                                      nodata=info["nodata"], dem_path=info["path"])

        # SVF/Openness comparison
        out_path = OUT_DIR / f"{dem_key}_svf_openness_comparison.png"
        if not out_path.exists():
            plot_svf_openness_comparison(dem_key, info["deriv"], out_path,
                                          nodata=info["nodata"])

    # ===================================================================
    # 2.2 GEOMORPHIC EVIDENCE MAPS
    # ===================================================================
    print(f"\n{'='*60}")
    print("2.2 GEOMORPHIC EVIDENCE MAPS")
    print(f"{'='*60}")

    for dem_key, info in dems.items():
        deriv_dir = info["deriv"]
        dem_stem = get_dem_stem(dem_key)

        # Lineament extraction from PCA
        pca_data, pca_meta = load_derivative(deriv_dir, dem_stem, "pca_composite")
        if pca_data is None:
            # Try PC1
            pca_data, pca_meta = load_derivative(deriv_dir, dem_stem, "pca_pc1")
        if pca_data is None:
            # Try legacy .tif naming
            pca_candidates = list(deriv_dir.glob("*_pca_hillshade.tif"))
            if pca_candidates:
                pca_data, _, _, _ = load_raster(pca_candidates[0])
                pca_meta = None

        if pca_data is not None:
            lineament_df = extract_lineaments_from_pca(
                pca_data, pca_meta, info["path"], OUT_DIR, nodata=info["nodata"]
            )

            # Lineament density map
            out_path = OUT_DIR / f"{dem_key}_lineament_density.png"
            if not out_path.exists() and len(lineament_df) > 0:
                plot_lineament_density_map(dem_key, lineament_df, deriv_dir,
                                            out_path, nodata=info["nodata"])

        # --- Compute REM and HAND for pixel-level river analysis ---
        rem_hand_dir = safe_makedirs(OUT_DIR / "rem_hand")
        rem_hand_result = None
        hand_tif = rem_hand_dir / f"{dem_key}_hand.tif"
        if not hand_tif.exists():
            try:
                rem_hand_result = compute_rem_hand(
                    info["path"], rem_hand_dir, dem_key,
                    nodata=info["nodata"]
                )
            except Exception as e:
                print(f"  [!] REM/HAND computation failed for {dem_key}: {e}")
                rem_hand_result = None
        else:
            # Load pre-computed HAND/REM from .tif
            try:
                with rasterio.open(str(hand_tif)) as src:
                    hand = src.read(1).astype(np.float32)
                    transform = src.transform
                    crs = src.crs
                    extent = _extent_from_transform(transform, hand.shape)
                rem_tif = rem_hand_dir / f"{dem_key}_rem.tif"
                rem = None
                if rem_tif.exists():
                    with rasterio.open(str(rem_tif)) as src:
                        rem = src.read(1).astype(np.float32)
                # Load flow accumulation to reconstruct drainage_mask
                accum_tif = rem_hand_dir / f"{dem_key}_flow_accumulation.tif"
                drainage_mask = None
                if accum_tif.exists():
                    with rasterio.open(str(accum_tif)) as src:
                        accum = src.read(1).astype(np.float32)
                    drainage_mask = accum >= DRAINAGE_ACCUM_THRESHOLD
                else:
                    # Fallback: drainage cells are where HAND == 0
                    drainage_mask = (hand == 0) & ~np.isnan(hand)
                # Load original DEM for context-aware detection
                dem_arr = None
                if info.get("nodata") is not None:
                    with rasterio.open(str(info["path"])) as src:
                        # Read at same resolution as HAND to ensure shape match
                        dem_arr = src.read(1, out_shape=hand.shape,
                                           resampling=ResampEnum.average
                                           ).astype(np.float32)
                    dem_arr[dem_arr == info["nodata"]] = np.nan
                    dem_arr[dem_arr < -1e6] = np.nan
                rem_hand_result = {
                    "hand": hand,
                    "rem": rem,
                    "dem": dem_arr,
                    "drainage_mask": drainage_mask,
                    "transform": transform,
                    "crs": crs,
                    "extent": extent,
                }
                print(f"  Loaded pre-computed HAND/REM for {dem_key}")
            except Exception as e:
                print(f"  [!] Could not load pre-computed HAND/REM: {e}")
                rem_hand_result = None

        # Drainage anomaly map (now with REM/HAND pixel-level results)
        out_path = OUT_DIR / f"{dem_key}_drainage_anomaly_map.png"
        if not out_path.exists():
            drain_csv = AUDIT_DIR / f"{dem_key}_drainage_anomalies.csv"
            plot_drainage_anomaly_map(
                dem_key, drain_csv, traces, deriv_dir, out_path,
                nodata=info["nodata"], rem_hand_result=rem_hand_result,
                dem_path=info["path"]
            )

        # River valley/bed scatter plot + binary .tif masks
        if rem_hand_result is not None:
            rvb_path = rem_hand_dir / f"{dem_key}_river_valley_bed.png"
            if not rvb_path.exists():
                plot_river_valley_bed(
                    dem_key, rem_hand_result, rvb_path,
                    nodata=info["nodata"]
                )

        # Expression quality 4-class per-pixel map (one figure per source)
        if dem_key in class_dfs:
            base_path = OUT_DIR / f"{dem_key}_expression_quality_4class.png"
            pix_counts = plot_expression_quality_4class(
                dem_key, traces, class_dfs[dem_key], deriv_dir,
                base_path, lidar_class_df=lidar_class_df, nodata=info["nodata"],
                dem_path=info["path"])
            if pix_counts:
                pixel_quality_counts[dem_key] = pix_counts

    # ===================================================================
    # 2.3 QUANTITATIVE INDEX VISUALISATION
    # ===================================================================
    print(f"\n{'='*60}")
    print("2.3 QUANTITATIVE INDEX VISUALISATION")
    print(f"{'='*60}")

    for dem_key, info in dems.items():
        deriv_dir = info["deriv"]

        # Compute geomorphic indices
        indices_csv = OUT_DIR / f"{dem_key}_geomorphic_indices.csv"
        if indices_csv.exists():
            indices_df = pd.read_csv(indices_csv)
        else:
            indices_df = compute_geomorphic_indices(
                dem_key, info["path"], deriv_dir, traces, struct_pts,
                nodata=info["nodata"]
            )
            indices_df.to_csv(indices_csv, index=False)

        # Radar charts
        radar_dir = safe_makedirs(OUT_DIR / f"{dem_key}_radar")
        plot_radar_charts(indices_df, radar_dir, max_charts=20)

        # Longitudinal river profiles
        profile_dir = safe_makedirs(OUT_DIR / f"{dem_key}_river_profiles")
        plot_longitudinal_profiles(info["path"], traces[:10], deriv_dir, dem_key,
                                    profile_dir, nodata=info["nodata"])

        # Hypsometric curves
        hypso_dir = safe_makedirs(OUT_DIR / f"{dem_key}_hypsometric")
        plot_hypsometric_curves(info["path"], traces[:10], hypso_dir,
                                 nodata=info["nodata"])

    # ===================================================================
    # 2.4 UNCERTAINTY AND CONFIDENCE LAYERS
    # ===================================================================
    print(f"\n{'='*60}")
    print("2.4 UNCERTAINTY AND CONFIDENCE LAYERS")
    print(f"{'='*60}")

    # Confidence heatmap
    dem_paths = {}
    if SRTM3S_RP.exists():
        dem_paths["srtm3s"] = str(SRTM3S_RP)
    if SRTM_RP.exists():
        dem_paths["srtm1s"] = str(SRTM_RP)
    if DEM10M_RP.exists():
        dem_paths["dem10m"] = str(DEM10M_RP)
    if LIDAR_RP.exists():
        dem_paths["lidar_0_5m"] = str(LIDAR_RP)

    out_path = OUT_DIR / "data_coverage_confidence_heatmap.png"
    if not out_path.exists() and len(dem_paths) >= 2:
        plot_confidence_heatmap(dem_paths, out_path)

    # Multi-resolution comparison panels
    if len(dem_paths) >= 2:
        multires_dir = safe_makedirs(OUT_DIR / "multires_comparison")
        plot_multi_resolution_comparison(
            dem_paths, traces, multires_dir,
            struct_pts=struct_pts
        )

    # ===================================================================
    # 3.3 CONFIDENCE SCORING
    # ===================================================================
    print(f"\n{'='*60}")
    print("3.3 CONFIDENCE SCORING & UNCERTAINTY CLASSIFICATION")
    print(f"{'='*60}")

    for dem_key, info in dems.items():
        if dem_key not in class_dfs:
            continue

        conf_csv = OUT_DIR / f"{dem_key}_confidence_scores.csv"
        if conf_csv.exists():
            conf_df = pd.read_csv(conf_csv)
        else:
            # Load indices
            indices_csv = OUT_DIR / f"{dem_key}_geomorphic_indices.csv"
            if indices_csv.exists():
                indices_df = pd.read_csv(indices_csv)
            else:
                indices_df = pd.DataFrame()

            conf_df = compute_confidence_scores(
                class_dfs[dem_key], indices_df, traces, dem_res=info["res"],
                lidar_class_df=lidar_class_df
            )
            conf_df.to_csv(conf_csv, index=False)
            print(f"  Saved: {conf_csv.name}")

    # ===================================================================
    # 3.4 CROSS-VALIDATION
    # ===================================================================
    print(f"\n{'='*60}")
    print("3.4 CROSS-VALIDATION AGAINST PUBLISHED ANALOGUES")
    print(f"{'='*60}")

    for dem_key, info in dems.items():
        if dem_key not in class_dfs:
            continue

        xval_csv = OUT_DIR / f"{dem_key}_cross_validation.csv"
        if xval_csv.exists():
            continue

        indices_csv = OUT_DIR / f"{dem_key}_geomorphic_indices.csv"
        if indices_csv.exists():
            indices_df = pd.read_csv(indices_csv)
        else:
            indices_df = pd.DataFrame()

        xval_df = cross_validate_analogues(class_dfs[dem_key], indices_df,
                                            lidar_class_df=lidar_class_df)
        xval_df.to_csv(xval_csv, index=False)
        print(f"  Saved: {xval_csv.name}")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'='*80}")
    print("VISUAL AUDIT COMPLETE")
    print(f"  Output directory: {OUT_DIR}")
    print(f"{'='*80}")

    # List all output files
    output_files = sorted(OUT_DIR.rglob("*"))
    n_png = sum(1 for f in output_files if f.suffix == ".png")
    n_csv = sum(1 for f in output_files if f.suffix == ".csv")
    print(f"  PNG figures: {n_png}")
    print(f"  CSV reports: {n_csv}")

    # Print 4-class distribution summary (per-pixel counts)
    print(f"\n  Expression Quality 4-Class Summary (per-pixel):")
    for dem_key in dems:
        if dem_key in pixel_quality_counts:
            counts = pixel_quality_counts[dem_key]
            # Ensure all 4 classes are shown
            for cls in ["A", "B", "C", "D"]:
                counts.setdefault(cls, 0)
            print(f"    {dem_key}: {counts}")
        else:
            print(f"    {dem_key}: (not computed — outputs may already exist)")


if __name__ == "__main__":
    main()
