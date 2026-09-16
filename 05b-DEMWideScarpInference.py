#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05b-DEMWideScarpInference.py -- ML-based wide-area scarp inference.

This script reads the pixel-level candidate points produced by Script 05a
(``all_inferred_points.csv``), enriches each point with local geometric features
sampled from the DEM and HAND rasters, trains a gradient-boosted-tree classifier
on HighCon (positive) vs confirmed-removed (negative) labels, and scores the
unlabeled pool to produce probability-ranked scarp candidates.

Iterative active-learning loop
-------------------------------
Round k:
  1. Train XGB on current labeled set (Trace1..k in HighCon as positives
     + confirmed-removed negatives + explicit confirmed non-scarps)
  2. Score unlabeled pool
  3. Add a new Human-reviewed top-probability points from HighCon (Trace k+1)
  4. Add explicitly confirmed non-scarps to negatives
  5. Re-run from step 1 with the expanded label set

HighCon directory is versioned as ``HighCon/round_1/``, ``round_2/``, ...
so you can always audit which points were labeled in which round and roll
back a bad round.

Trace file auto-detection
-------------------------
The pipeline scans ``data/HighCon`` for ``TraceN.xlsx`` files (Trace1.xlsx,
Trace2.xlsx, ...) and uses all available ones.  If a trace file is missing
(e.g. Trace3.xlsx was removed), that round is skipped gracefully — the
pipeline logs a warning and continues with the next available trace or
completes if no more traces are found.  This replaces the previous
hardcoded ``NUM_TRACES`` count.

Pipeline stages
---------------
Stage 0 — Rule-based pre-filters (border exclusion, flat-terrain/lake exclusion)
Stage 1 — Assemble labelled dataset (HighCon positives, stratified negatives, unlabeled pool)
Stage 2 — Feature engineering (recompute 05a features + add local geometric context
          + cluster shape descriptors for I/L vs U/O filtering)
Stage 3 — Spatial leakage-safe validation (along-trace chunk holdout)
Stage 4 — Model training (XGBoost + LogisticRegression baseline)
Stage 5 — Score & re-integrate (probability per unlabeled point + shape-based rescoring)
Stage 6 — Sanity checks (SHAP, feature importance, threshold metrics, spatial plots)

Outputs are written to ``data/FilteredOutput``.
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import (binary_erosion, label as cc_label,
                           maximum_filter, minimum_filter,
                           map_coordinates, uniform_filter)
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import (connected_components,
                                   minimum_spanning_tree,
                                   shortest_path)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, average_precision_score)

import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from derivative_common import DATA_DIR, OUT_DIR, safe_makedirs

# ── Constants ────────────────────────────────────────────────────────
DEM_SOURCES = {
    "dem10m": {
        "dem": DATA_DIR / "derivatives" / "dem10m_epsg28355_clipped.tif",
        "hand": DATA_DIR / "derivatives" / "visual_audit" / "rem_hand" / "dem10m_hand.tif",
        "csv": DATA_DIR / "derivatives" / "known_trace_scarp_validation" / "dem10m" / "all_inferred_points.csv",
        "filtered_txt": DATA_DIR / "FilteredOutput" / "DEM10.txt",
    },
    "lidar_0_5m": {
        "dem": DATA_DIR / "derivatives" / "lidar_0_5m_epsg28355_10m_smoke.tif",
        "hand": DATA_DIR / "derivatives" / "visual_audit" / "rem_hand" / "lidar_0_5m_hand.tif",
        "csv": DATA_DIR / "derivatives" / "known_trace_scarp_validation" / "lidar_0_5m" / "dem_wide_inferred_points.csv",
        "filtered_txt": DATA_DIR / "FilteredOutput" / "DEM05.txt",
    },
}

HIGHCON_DIR = DATA_DIR / "HighCon"
OUTPUT_DIR = DATA_DIR / "FilteredOutput"

# Iterative active-learning: trace files are auto-detected from data/HighCon.
# The pipeline scans for Trace1.xlsx, Trace2.xlsx, ... and uses all available.
# If a trace file is missing, that round is skipped gracefully — the pipeline
# logs a warning and stops after the last available trace.
# This replaces the old hardcoded NUM_TRACES = 4.
NUM_TRACES = 3  # Current count of high-confidence traces (Trace1.xlsx + Trace2.xlsx)


def _detect_available_traces(highcon_dir=None):
    """Scan ``data/HighCon`` for ``TraceN.xlsx`` files and return sorted 1-based indices.

    Returns a list of integers like [1, 2] for Trace1.xlsx and Trace2.xlsx.
    Missing files are silently skipped — the caller logs a warning.
    """
    if highcon_dir is None:
        highcon_dir = HIGHCON_DIR
    indices = []
    for n in range(1, 100):  # safety cap
        path = highcon_dir / f"Trace{n}.xlsx"
        if path.exists():
            indices.append(n)
        else:
            # Don't stop at first gap — allow Trace1, Trace3 without Trace2.
            # But typically traces are contiguous, so break after 2 consecutive misses.
            next_path = highcon_dir / f"Trace{n + 1}.xlsx"
            if not next_path.exists():
                break
    return indices

# Shape/topology (I/L vs U/O) parameters
CONNECTIVITY_RADIUS_M = 20.0   # tune to point spacing in candidate pool
MIN_CLUSTER_SIZE = 5
STRAIGHTNESS_FLOOR = 0.4       # points in clusters below this are U/O-like

# Stage 0 parameters
EDGE_BUFFER_M = 20.0          # reuse 05a's LSM_EDGE_BUFFER_M
HAND_WATER_THRESHOLD_M = 0.0   # HAND <= this = river/lake bed
HAND_INFLUENCE_THRESHOLD_M = 2.0  # low-HAND zone (05a's constant)

# Stage 2 feature parameters
PROFILE_HALF_WIDTH_M = 50.0
PROFILE_SAMPLE_COUNT = 5
LOCAL_RELIEF_WINDOW_PX = 3   # 3×3 for feature-level local relief
KNN_RADIUS_M = 50.0           # radius for local linearity / density
KNN_MIN_NEIGHBORS = 3
TV_WEIGHTS = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
TARGET_TV_SIGMAS_M = [20.0, 28.0, 40.0, 56.0, 80.0, 113.0, 160.0]
BASE_RAY_LENGTH_M = 800
BASE_BACKGROUND_FRACTION = 0.15
GRAD_DECAY_THRESHOLD = 0.3
RAY_MIN_PIXELS = 50
SCARP_SLOPE_FLAT_THRESHOLD = 0.05
SCARP_SLOPE_DROP_FACTOR = 0.3

# Stage 3 parameters
CHUNK_LENGTH_M = 2000.0     # along-trace chunk size for spatial holdout
TRAIN_FRACTION = 0.7        # fraction of chunks for training

# Stage 4 parameters
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# Feature columns explicitly excluded from training (location / geology proxies)
EXCLUDED_FEATURES = {
    "x", "y", "trace_id", "dist_to_trace", "median_distance_to_known_m",
    "nearest_prior_profile_dist_m", "prior_quality_class", "prior_detection_rate",
    "structural_context", "geological_unit", "label",
}

# ── Stage 0: Rule-based pre-filters ──────────────────────────────────

def _erode_edge_mask(valid_mask, resolution, buffer_m):
    """Erode the valid-data mask by ``buffer_m`` to exclude border pixels."""
    iterations = max(1, int(round(buffer_m / resolution)))
    return binary_erosion(valid_mask, iterations=iterations)


def _build_water_mask(dem, resolution, nodata):
    """Detect water bodies using DEM-fill: areas where the filled DEM has near-zero relief.

    This is a fallback when HAND raster is not available.  It uses a very strict
    relief threshold (0.05 m in a 70 m window) combined with a large-area
    connected-component filter (>= 1 ha) to catch only genuine water bodies.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    finite = np.isfinite(dem) & (dem != nodata)
    dem_safe = np.where(finite, dem, np.nan)

    win_px = max(3, int(round(70.0 / resolution)))
    if win_px % 2 == 0:
        win_px += 1
    max_f = maximum_filter(dem_safe, size=win_px)
    min_f = minimum_filter(dem_safe, size=win_px)
    relief = max_f - min_f
    # Very strict: only near-perfectly flat areas (lakes/reservoirs)
    flat_px = (relief < 0.05) & finite

    labels, n = cc_label(flat_px)
    if n == 0:
        return np.zeros(dem.shape, dtype=bool)

    px_area = resolution ** 2
    areas = np.bincount(labels.ravel()) * px_area
    # Only large contiguous areas (>= 1 ha)
    big_flat_ids = np.where(areas > 10000.0)[0]
    return np.isin(labels, big_flat_ids)


def _build_hand_water_mask(hand, hand_nodata, threshold_m=HAND_WATER_THRESHOLD_M):
    """Build water mask from HAND raster: HAND <= threshold = water."""
    if hand is None:
        return None
    finite = np.isfinite(hand)
    if hand_nodata is not None:
        finite &= ~np.isclose(hand, hand_nodata)
    return (hand <= threshold_m) & finite


def apply_prefilters(dem, resolution, nodata, points_x, points_y, transform,
                     hand=None, hand_transform=None, hand_nodata=None,
                     hand_crs=None, dem_crs=None):
    """Return boolean mask of points that survive Stage-0 pre-filters.

    Filters:
    1. Border exclusion (eroded valid-data mask)
    2. Water/flat exclusion (HAND-based if available, DEM-relief fallback)
    """
    finite = np.isfinite(dem) & (dem != nodata)
    interior = _erode_edge_mask(finite, resolution, EDGE_BUFFER_M)

    # Water mask: prefer HAND, fallback to DEM relief
    water_mask = None
    if hand is not None and hand_transform is not None:
        water_mask = _build_hand_water_mask(hand, hand_nodata)
        if water_mask is not None:
            # Reproject water_mask to DEM grid if transforms differ
            if hand_transform != transform:
                dst = np.zeros(dem.shape, dtype=np.uint8)
                reproject(water_mask.astype(np.uint8), dst,
                          src_transform=hand_transform,
                          src_crs=hand_crs,
                          dst_transform=transform,
                          dst_crs=dem_crs,
                          resampling=Resampling.nearest)
                water_mask = dst.astype(bool)

    if water_mask is None:
        water_mask = _build_water_mask(dem, resolution, nodata)

    rows, cols = rasterio.transform.rowcol(transform, points_x, points_y)

    keep = np.zeros(len(points_x), dtype=bool)
    for i in range(len(points_x)):
        r, c = int(round(rows[i])), int(round(cols[i]))
        if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
            if interior[r, c] and not water_mask[r, c]:
                keep[i] = True
    return keep


# ── Stage 2: Feature engineering ─────────────────────────────────────

def _sample_raster(raster, transform, xs, ys, nodata=None):
    """Bilinear-sample a raster at (x, y) coordinates via map_coordinates."""
    inv = ~transform
    rows, cols = inv * (xs, ys)
    coords = np.vstack([rows, cols])
    values = map_coordinates(raster.astype(np.float64), coords, order=1,
                             mode="nearest")
    if nodata is not None:
        values[np.isclose(values, nodata)] = np.nan
    return values


def precompute_dem_rasters(dem, resolution, nodata):
    """Precompute all DEM-derived rasters once for reuse across point sets.

    Returns a dict of raster arrays sampled at DEM-grid coordinates.
    """
    from skimage.restoration import denoise_tv_chambolle
    from scipy.ndimage import laplace, uniform_filter

    finite = np.isfinite(dem) & (dem != nodata)
    dem_safe = np.where(finite, dem, np.nan)

    rasters = {}

    # Local slope (degrees)
    print("    [Precompute] Local slope...")
    grad_r = np.gradient(dem, axis=0) / resolution
    grad_c = np.gradient(dem, axis=1) / resolution
    slope_rad = np.arctan(np.hypot(grad_r, grad_c))
    rasters["slope_deg"] = np.degrees(slope_rad)

    # Local relief (max-min in 3×3 window)
    print("    [Precompute] Local relief...")
    win_px = LOCAL_RELIEF_WINDOW_PX
    rasters["relief"] = maximum_filter(dem_safe, size=win_px) - \
                          minimum_filter(dem_safe, size=win_px)

    # Terrain Ruggedness Index
    print("    [Precompute] TRI...")
    mean_neighborhood = uniform_filter(dem_safe, size=win_px)
    rasters["tri"] = np.abs(dem_safe - mean_neighborhood)

    # Multi-scale curvature: TV-denoise at each scale, compute Laplacian
    print("    [Precompute] Multi-scale curvature (7 TV scales)...")
    for s_idx, weight in enumerate(TV_WEIGHTS):
        tv_denoised = denoise_tv_chambolle(dem, weight=weight, channel_axis=None)
        rasters[f"curv_scale_{s_idx}"] = laplace(tv_denoised)

    return rasters


def _compute_perpendicular_profile(dem, xs, ys, transform, resolution,
                                     half_width_m=PROFILE_HALF_WIDTH_M):
    """Perpendicular profile shape stats: max_slope, before_slope, after_slope.

    Vectorised: samples all profiles in a single map_coordinates call.
    For each point, the perpendicular direction is estimated from the local
    DEM gradient.
    """
    n = len(xs)
    max_slopes = np.zeros(n, dtype=np.float32)
    before_slopes = np.zeros(n, dtype=np.float32)
    after_slopes = np.zeros(n, dtype=np.float32)

    distances = np.arange(-half_width_m, half_width_m + resolution, resolution)
    n_d = len(distances)

    inv = ~transform
    rows, cols = inv * (xs, ys)

    # Compute gradients at each point (vectorised)
    r_i = np.clip(np.round(rows).astype(int), 1, dem.shape[0] - 2)
    c_i = np.clip(np.round(cols).astype(int), 1, dem.shape[1] - 2)
    grad_r = dem[r_i + 1, c_i] - dem[r_i - 1, c_i]
    grad_c = dem[r_i, c_i + 1] - dem[r_i, c_i - 1]
    grad_mag = np.hypot(grad_r, grad_c)
    valid_grad = grad_mag > 1e-8

    # Perpendicular direction for each point
    perp_r = np.where(valid_grad, -grad_c / np.maximum(grad_mag, 1e-12), 0.0)
    perp_c = np.where(valid_grad, grad_r / np.maximum(grad_mag, 1e-12), 0.0)

    # Build sample coordinates for all (point × distance) pairs
    # Shape: (n, n_d) for both rows and cols
    sample_rows = rows[:, None] + distances[None, :] * perp_r[:, None] / resolution
    sample_cols = cols[:, None] + distances[None, :] * perp_c[:, None] / resolution

    # Flatten and sample in one call
    flat_rows = sample_rows.ravel()
    flat_cols = sample_cols.ravel()
    valid = (flat_rows >= 0) & (flat_rows < dem.shape[0]) & \
            (flat_cols >= 0) & (flat_cols < dem.shape[1])
    flat_values = np.zeros(n * n_d, dtype=np.float64)
    if valid.any():
        flat_values[valid] = map_coordinates(
            dem.astype(np.float64),
            [flat_rows[valid], flat_cols[valid]],
            order=1, mode="nearest")

    profiles = flat_values.reshape(n, n_d)
    # Invalidate profiles where the point has no gradient
    profiles[~valid_grad] = 0.0

    # Compute slopes along each profile (vectorised)
    slopes = np.abs(np.diff(profiles, axis=1))  # shape (n, n_d-1)
    has_slope = slopes.sum(axis=1) > 0

    for i in range(n):
        if not has_slope[i] or not valid_grad[i]:
            continue
        sl = slopes[i]
        max_idx = int(np.argmax(sl))
        before_end = max(1, max_idx - 2)
        after_start = min(len(sl) - 1, max_idx + 2)
        before_slopes[i] = float(np.mean(sl[:before_end])) if before_end > 0 else 0.0
        after_slopes[i] = float(np.mean(sl[after_start:])) if after_start < len(sl) else 0.0
        max_slopes[i] = float(sl[max_idx])

    return max_slopes, before_slopes, after_slopes


def _compute_local_linearity_and_density(xs, ys, radius_m=KNN_RADIUS_M,
                                           min_neighbors=KNN_MIN_NEIGHBORS):
    """Local linearity (PCA eigenvalue ratio) and point density.

    For each point, finds k nearest neighbors within ``radius_m``, fits a PCA
    to the neighbor cloud, and returns:
    - linearity: eigenvalue_ratio = lambda_1 / (lambda_1 + lambda_2)
      (1.0 = perfectly linear string, 0.5 = isotropic cloud)
    - density: number of neighbors within radius
    """
    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)
    n = len(xs)
    linearity = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.int32)

    batch = 10000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        results = tree.query_ball_point(coords[start:end], r=radius_m)

        for i, neighbors in enumerate(results):
            idx = start + i
            if len(neighbors) < min_neighbors:
                linearity[idx] = 0.0
                density[idx] = len(neighbors)
                continue

            neighbor_coords = coords[neighbors]
            centered = neighbor_coords - neighbor_coords.mean(axis=0)
            if centered.shape[0] < 2:
                linearity[idx] = 0.0
                density[idx] = len(neighbors)
                continue

            cov = np.cov(centered.T)
            if cov.ndim == 0:
                linearity[idx] = 0.0
            else:
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                total = eigenvalues.sum()
                if total > 1e-10:
                    linearity[idx] = eigenvalues[0] / total
                else:
                    linearity[idx] = 0.0
            density[idx] = len(neighbors)

    return linearity, density


# ── Stage 2: Shape/topology features (I/L vs U/O) ────────────────────

def compute_shape_features(xs, ys, radius_m=CONNECTIVITY_RADIUS_M,
                            min_cluster_size=MIN_CLUSTER_SIZE):
    """Cluster candidate points into connected traces and score each
    cluster's shape: high straightness = I/L (elongated, open),
    low straightness or has_cycle = U/O (folded back or closed).

    Returns per-point arrays: straightness, has_cycle, cluster_size, cluster_id.
    """
    n = len(xs)
    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)

    # Build sparse adjacency from radius-neighbor pairs
    pairs = tree.query_pairs(r=radius_m, output_type="ndarray")
    if len(pairs) == 0:
        return (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool),
                np.ones(n, dtype=np.int32), np.arange(n))

    dists = np.hypot(coords[pairs[:, 0], 0] - coords[pairs[:, 1], 0],
                      coords[pairs[:, 0], 1] - coords[pairs[:, 1], 1])
    graph = coo_matrix((dists, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    graph = graph + graph.T  # symmetrize

    n_comp, comp_labels = connected_components(graph, directed=False)

    straightness = np.zeros(n, dtype=np.float32)
    has_cycle = np.zeros(n, dtype=bool)
    cluster_size = np.zeros(n, dtype=np.int32)

    for comp_id in range(n_comp):
        idx = np.where(comp_labels == comp_id)[0]
        cluster_size[idx] = len(idx)

        if len(idx) < min_cluster_size:
            straightness[idx] = 1.0   # too small to judge; don't penalize
            continue

        sub_graph = graph[idx][:, idx]
        n_edges = sub_graph.nnz // 2   # symmetrized, so /2
        n_nodes = len(idx)
        cyclic = n_edges >= n_nodes    # a tree has exactly n_nodes-1 edges
        has_cycle[idx] = cyclic

        # Minimum spanning tree gives a single backbone path (no shortcuts
        # from stray cross-links), which is what we want the path length from
        mst = minimum_spanning_tree(sub_graph)
        mst = mst + mst.T

        # Find the two graph-diameter endpoints via BFS/Dijkstra twice
        dist_from_0 = shortest_path(mst, indices=0, method="D",
                                    directed=False, unweighted=False)
        far_a = int(np.argmax(np.where(np.isfinite(dist_from_0), dist_from_0, -1)))
        dist_from_a = shortest_path(mst, indices=far_a, method="D",
                                    directed=False, unweighted=False)
        far_b = int(np.argmax(np.where(np.isfinite(dist_from_a), dist_from_a, -1)))
        path_length = dist_from_a[far_b]

        euclid = np.hypot(coords[idx[far_a], 0] - coords[idx[far_b], 0],
                          coords[idx[far_a], 1] - coords[idx[far_b], 1])

        s = 0.0 if (path_length < 1e-6 or cyclic) else float(euclid / path_length)
        straightness[idx] = np.clip(s, 0.0, 1.0)

    return straightness, has_cycle, cluster_size, comp_labels


def compute_all_features(dem, hand, resolution, dem_nodata, hand_nodata,
                          dem_transform, hand_transform,
                          xs, ys, existing_df, batch_label="",
                          precomputed=None):
    """Compute the complete feature set for a set of points.

    Parameters
    ----------
    dem : 2D array
        The DEM raster.
    hand : 2D array or None
        HAND raster (may be at different resolution/transform).
    resolution : float
        DEM pixel resolution in metres.
    xs, ys : 1D arrays
        Point coordinates.
    existing_df : DataFrame
        DataFrame containing at least ``tier`` and ``break_type`` columns.
    precomputed : dict or None
        Pre-computed DEM-derived rasters from ``precompute_dem_rasters()``.
        If None, computes them on-the-fly (slow for multiple calls).
    """
    n = len(xs)
    print(f"  [Features{batch_label}] Computing features for {n:,} points...")

    features = pd.DataFrame(index=range(n))

    # ── Pull-through features from existing data ──────────────────
    features["tier"] = existing_df["tier"].values if "tier" in existing_df.columns else 2
    features["break_type"] = existing_df["break_type"].values if "break_type" in existing_df.columns else "p"
    features["break_type_p"] = (features["break_type"] == "p").astype(int)

    # ── HAND value ────────────────────────────────────────────────
    print(f"  [Features{batch_label}] Sampling HAND values...")
    if hand is not None:
        features["hand_value"] = _sample_raster(hand, hand_transform, xs, ys, nodata=hand_nodata)
        features["hand_value"] = features["hand_value"].fillna(0.0)
    else:
        features["hand_value"] = 0.0

    # ── DEM-derived features (from precomputed rasters) ───────────
    if precomputed is None:
        precomputed = precompute_dem_rasters(dem, resolution, dem_nodata)

    print(f"  [Features{batch_label}] Sampling precomputed rasters...")
    features["local_slope_deg"] = _sample_raster(precomputed["slope_deg"], dem_transform, xs, ys)
    features["local_relief_m"] = _sample_raster(precomputed["relief"], dem_transform, xs, ys)
    features["tri"] = _sample_raster(precomputed["tri"], dem_transform, xs, ys)

    # Multi-scale curvature vector
    curvatures = np.zeros((n, len(TV_WEIGHTS)), dtype=np.float32)
    for s_idx in range(len(TV_WEIGHTS)):
        vals = _sample_raster(precomputed[f"curv_scale_{s_idx}"], dem_transform, xs, ys)
        curvatures[:, s_idx] = vals
        features[f"curv_scale_{s_idx}"] = vals

    # Curvature consistency: how many scales agree in sign
    curv_signs = np.sign(curvatures)
    n_positive = (curv_signs > 0).sum(axis=1)
    n_negative = (curv_signs < 0).sum(axis=1)
    features["curv_consistency"] = np.maximum(n_positive, n_negative).astype(float) / len(TV_WEIGHTS)
    features["curv_dominant_sign"] = np.where(n_positive > n_negative, 1.0, -1.0)

    # ── Perpendicular profile shape stats ─────────────────────────
    print(f"  [Features{batch_label}] Computing perpendicular profile stats...")
    max_slopes, before_slopes, after_slopes = _compute_perpendicular_profile(
        dem, xs, ys, dem_transform, resolution)
    features["profile_max_slope"] = max_slopes
    features["profile_before_slope"] = before_slopes
    features["profile_after_slope"] = after_slopes
    features["profile_slope_ratio"] = np.where(
        np.maximum(before_slopes, after_slopes) > 1e-8,
        max_slopes / np.maximum(before_slopes, after_slopes),
        0.0).astype(np.float32)

    # ── Local linearity and density ────────────────────────────────
    print(f"  [Features{batch_label}] Computing local linearity and density...")
    linearity, density = _compute_local_linearity_and_density(xs, ys)
    features["local_linearity"] = linearity
    features["local_density"] = density

    # ── Shape/topology features (I/L vs U/O) ────────────────────────
    print(f"  [Features{batch_label}] Computing cluster shape descriptors...")
    straightness, has_cycle, cluster_size, _ = compute_shape_features(xs, ys)
    features["shape_straightness"] = straightness
    features["shape_has_cycle"] = has_cycle.astype(int)
    features["shape_cluster_size"] = cluster_size

    print(f"  [Features{batch_label}] Feature engineering complete: {features.shape[1]} features")
    return features


# ── Stage 1: Assemble labelled dataset ──────────────────────────────

def _build_break_type_lookup(dem_source_key):
    """Build a coordinate -> break_type lookup from the source CSV.

    Returns a dict keyed by (x_rounded, y_rounded) -> break_type.
    """
    csv_path = DEM_SOURCES[dem_source_key]["csv"]
    csv_df = pd.read_csv(csv_path)
    bt_map = {}
    for _, r in csv_df.iterrows():
        bt_map[(round(float(r["x"]), 1), round(float(r["y"]), 1))] = r["break_type"]
    return bt_map


def load_highcon_points(dem_source_key, bt_lookup=None, trace_indices=None):
    """Load HighCon trace points, keeping only those present in this source's CSV.

    HighCon Trace1-2 blend points from both DEM sources. We filter
    to only keep points that actually exist in this source's inferred-points
    CSV, so we don't train on points from the wrong DEM.

    break_type is looked up from the CSV (not defaulted) to avoid label leaks.

    Parameters
    ----------
    trace_indices : list of int, optional
        Which traces to load (1-based).  Default: auto-detect all available
        traces in ``data/HighCon``.  In iterative mode, round k loads traces 1..k.
        Missing trace files are skipped with a warning.
    """
    if bt_lookup is None:
        bt_lookup = _build_break_type_lookup(dem_source_key)

    if trace_indices is None:
        trace_indices = _detect_available_traces()

    traces = []
    for trace_idx in trace_indices:
        path = HIGHCON_DIR / f"Trace{trace_idx}.xlsx"
        if not path.exists():
            print(f"  [Warning] {path} not found, skipping Trace{trace_idx}")
            continue
        df = pd.read_excel(path)
        # Filter: only keep points present in this source's CSV
        keys = [(round(float(x), 1), round(float(y), 1)) for x, y in zip(df["x"], df["y"])]
        in_csv = [k in bt_lookup for k in keys]
        df = df[in_csv].copy()
        if len(df) == 0:
            print(f"  Trace{trace_idx}: 0 points in {dem_source_key} CSV, skipping")
            continue
        # Lookup break_type from CSV
        df["break_type"] = [bt_lookup[(round(float(x), 1), round(float(y), 1))]
                             for x, y in zip(df["x"], df["y"])]
        df["trace_id"] = trace_idx
        df["label"] = 1
        n_total = len(keys)
        print(f"  Trace{trace_idx}: {len(df)}/{n_total} points in {dem_source_key} CSV")
        traces.append(df[["x", "y", "tier", "break_type", "trace_id", "label"]])
    return pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()


def load_confirmed_removed(dem_source_key):
    """Load confirmed-removed points = CSV points NOT in filtered TXT.

    The filtered TXT files contain the surviving (unlabeled + HighCon) pool.
    The difference between the full CSV and the TXT = removed (negative) points.
    """
    csv_path = DEM_SOURCES[dem_source_key]["csv"]
    txt_path = DEM_SOURCES[dem_source_key]["filtered_txt"]

    csv_df = pd.read_csv(csv_path)
    csv_df["key"] = csv_df["x"].round(2).astype(str) + "_" + csv_df["y"].round(2).astype(str)

    txt_df = pd.read_csv(txt_path, sep=";")
    txt_df["key"] = txt_df["x"].round(2).astype(str) + "_" + txt_df["y"].round(2).astype(str)

    txt_keys = set(txt_df["key"])
    removed = csv_df[~csv_df["key"].isin(txt_keys)].copy()
    removed["label"] = 0
    removed = removed[["x", "y", "tier", "break_type", "label"]]

    print(f"  [Negatives] CSV: {len(csv_df):,} | TXT: {len(txt_df):,} | "
          f"Removed (negative): {len(removed):,}")
    return removed


def load_unlabeled_pool(dem_source_key, highcon_keys=None):
    """Load the unlabeled pool = TXT points minus HighCon points.

    The TXT files contain BOTH HighCon and unlabeled points mixed together.
    We must remove HighCon points from the unlabeled pool to avoid
    contamination (positives leaking into the scoring pool).

    break_type is looked up from the CSV (not defaulted) so it matches
    what the model saw during training.
    """
    txt_path = DEM_SOURCES[dem_source_key]["filtered_txt"]
    csv_path = DEM_SOURCES[dem_source_key]["csv"]

    txt_df = pd.read_csv(txt_path, sep=";")
    txt_df["key"] = txt_df["x"].round(2).astype(str) + "_" + txt_df["y"].round(2).astype(str)

    # Remove HighCon points from the unlabeled pool
    if highcon_keys is not None and len(highcon_keys) > 0:
        n_before = len(txt_df)
        txt_df = txt_df[~txt_df["key"].isin(highcon_keys)]
        print(f"  [Unlabeled] Removed {n_before - len(txt_df):,} HighCon points from pool")

    txt_df["label"] = -1  # unlabeled

    # Lookup break_type from CSV
    csv_df = pd.read_csv(csv_path)
    bt_map = {}
    for _, r in csv_df.iterrows():
        bt_map[(round(float(r["x"]), 1), round(float(r["y"]), 1))] = r["break_type"]
    txt_df["break_type"] = [bt_map.get((round(float(x), 1), round(float(y), 1)), "p")
                             for x, y in zip(txt_df["x"], txt_df["y"])]

    return txt_df[["x", "y", "tier", "break_type", "label"]]


# ── Explicit negatives management ────────────────────────────────────

def load_explicit_negatives(dem_source_key, round_num):
    """Load explicitly confirmed non-scarp negatives from previous rounds.

    These are points that were human-reviewed and rejected as non-scarp.
    They accumulate across rounds to sharpen the negative class.

    Returns an empty DataFrame if no explicit negatives file exists yet.
    """
    neg_path = HIGHCON_DIR / f"round_{round_num}" / "explicit_negatives.csv"
    if not neg_path.exists():
        # Also check previous rounds (accumulate from the latest available)
        for r in range(round_num - 1, 0, -1):
            p = HIGHCON_DIR / f"round_{r}" / "explicit_negatives.csv"
            if p.exists():
                neg_path = p
                break
        else:
            return pd.DataFrame(columns=["x", "y", "tier", "break_type", "label"])

    df = pd.read_csv(neg_path)
    df["label"] = 0
    print(f"  [Explicit Neg] Loaded {len(df)} explicit negatives from {neg_path.name}")
    return df[["x", "y", "tier", "break_type", "label"]] if len(df) > 0 else \
        pd.DataFrame(columns=["x", "y", "tier", "break_type", "label"])


def save_explicit_negatives(negatives_df, dem_source_key, round_num):
    """Save explicit negatives for this round to the versioned HighCon directory."""
    round_dir = HIGHCON_DIR / f"round_{round_num}"
    safe_makedirs(round_dir)
    neg_path = round_dir / "explicit_negatives.csv"
    negatives_df[["x", "y", "tier", "break_type"]].to_csv(neg_path, index=False)
    print(f"  [Explicit Neg] Saved {len(negatives_df)} explicit negatives to {neg_path}")


def accumulate_explicit_negatives(dem_source_key, round_num, new_negatives=None):
    """Accumulate explicit negatives across all rounds up to round_num.

    Loads explicit negatives from all previous rounds plus any new ones
    passed in for the current round, deduplicates by coordinate, and
    returns the combined set.
    """
    all_negs = []
    for r in range(1, round_num + 1):
        neg = load_explicit_negatives(dem_source_key, r)
        if len(neg) > 0:
            all_negs.append(neg)

    if new_negatives is not None and len(new_negatives) > 0:
        all_negs.append(new_negatives)

    if not all_negs:
        return pd.DataFrame(columns=["x", "y", "tier", "break_type", "label"])

    combined = pd.concat(all_negs, ignore_index=True)
    # Deduplicate by rounded coordinate
    combined["key"] = combined["x"].round(2).astype(str) + "_" + combined["y"].round(2).astype(str)
    combined = combined.drop_duplicates(subset="key").drop(columns=["key"])
    combined["label"] = 0
    print(f"  [Explicit Neg] Accumulated {len(combined)} unique explicit negatives "
          f"through round {round_num}")
    return combined


def stratify_negatives(negatives, n_positive, dem_source_key):
    """Stratify negatives by removal reason (tier × break_type combination).

    Rather than random sampling, we partition negatives by tier/break_type
    and sample proportionally to ensure the model sees different failure modes.
    """
    negatives["stratum"] = negatives["tier"].astype(str) + "_" + negatives["break_type"].astype(str)
    strata = negatives["stratum"].value_counts()
    print(f"  [Stratify] Negative strata: {strata.to_dict()}")

    # Target: ~3x positive count
    n_target = min(len(negatives), n_positive * 3)
    sampled = []
    for stratum, count in strata.items():
        n_sample = max(1, int(n_target * count / len(negatives)))
        n_sample = min(n_sample, count)
        sampled.append(negatives[negatives["stratum"] == stratum].sample(
            n=n_sample, random_state=42))
    return pd.concat(sampled, ignore_index=True).drop(columns=["stratum"])


# ── Stage 3: Spatial leakage-safe validation ────────────────────────

def assign_spatial_chunks(coords_df, chunk_length_m=CHUNK_LENGTH_M):
    """Bin points into along-trace chunks for spatial holdout.

    Sorts each trace_id by coordinate, segments into ~2km chunks.
    Returns chunk_id column.
    """
    chunks = []
    for trace_id, group in coords_df.groupby("trace_id"):
        group = group.sort_values(["x", "y"]).reset_index(drop=True)
        # Compute cumulative distance along sorted points
        dx = np.diff(group["x"].values, prepend=group["x"].values[0])
        dy = np.diff(group["y"].values, prepend=group["y"].values[0])
        cum_dist = np.cumsum(np.hypot(dx, dy))
        group["chunk_id"] = (cum_dist // chunk_length_m).astype(int)
        # Make globally unique
        group["chunk_id"] = group["chunk_id"] + hash(str(trace_id)) % 10000
        chunks.append(group)
    return pd.concat(chunks, ignore_index=True) if chunks else coords_df


def spatial_train_test_split(labeled_df, train_fraction=TRAIN_FRACTION):
    """Split by whole chunks, not individual points.

    Returns train_mask boolean array.
    """
    train_mask = np.zeros(len(labeled_df), dtype=bool)
    for trace_id, group in labeled_df.groupby("trace_id"):
        chunk_ids = group["chunk_id"].unique()
        np.random.shuffle(chunk_ids)
        n_train = max(1, int(len(chunk_ids) * train_fraction))
        train_chunks = set(chunk_ids[:n_train])
        idx = group.index[group["chunk_id"].isin(train_chunks)]
        train_mask[idx] = True
    return train_mask


# ── Stage 4: Model training ────────────────────────────────────────

def get_feature_columns(df):
    """Return columns usable for training (excluding coordinates, labels, proxies)."""
    cols = [c for c in df.columns
            if c not in EXCLUDED_FEATURES
            and c not in ("chunk_id", "key", "stratum")
            and df[c].dtype != object
            and df[c].notna().any()]
    return cols


def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier with class weighting."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = max(1.0, n_neg / max(n_pos, 1))

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model


def train_logreg_baseline(X_train, y_train):
    """Train a logistic regression baseline for interpretability."""
    # LogReg (sklearn) doesn't handle NaN — impute before scaling
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000,
        random_state=42, solver="lbfgs")
    model.fit(X_train_scaled, y_train)
    return model, scaler


# ── Stage 6: Sanity checks ───────────────────────────────────────────

def plot_shap_summary(model, X, feature_names, output_path):
    """Generate SHAP summary plot."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SHAP] Saved to {output_path}")
    except Exception as e:
        print(f"  [SHAP] Failed: {e}")


def plot_feature_importance(model, feature_names, output_path):
    """Plot XGBoost feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    xgb.plot_importance(model, ax=axes[0], importance_type="weight",
                        max_num_features=20, title="Feature Importance (Weight)")
    xgb.plot_importance(model, ax=axes[1], importance_type="gain",
                        max_num_features=20, title="Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Importance] Saved to {output_path}")


def plot_precision_recall(y_val, y_proba, output_path):
    """Plot precision-recall curve at various thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    avg_prec = average_precision_score(y_val, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"AP={avg_prec:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Held-out Chunks)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PR Curve] Saved to {output_path}")


def plot_score_distribution(y_proba, output_path, hc_proba=None):
    """Plot histogram of model probability scores on unlabeled pool.

    If hc_proba is provided, overlays the HighCon score distribution for
    comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_proba, bins=100, edgecolor="black", alpha=0.5,
            label="Unlabeled pool", density=True)
    if hc_proba is not None:
        ax.hist(hc_proba, bins=50, edgecolor="red", alpha=0.5,
                color="red", label="HighCon positives", density=True)
        ax.legend()
    ax.set_xlabel("Probability Score (Ensemble)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Unlabeled vs HighCon")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Score Dist] Saved to {output_path}")


# ── Spatial plots ────────────────────────────────────────────────────

def plot_spatial_score_map(result_df, output_path, round_num, dem_source_key,
                           prob_col="prob_ensemble",
                           test_x=None, test_y=None, title_suffix=""):
    """Plot a spatial scatter map of probability scores.

    Uses the ensemble score (avg of XGBoost + LR) by default.
    Excludes points that failed the shape test (U/O-shaped clusters).
    Overlay HighCon trace points used as positives this round as gold
    scattered points.

    Parameters
    ----------
    result_df : DataFrame
        Unlabeled pool results with coordinates and probability columns.
    output_path : Path
    round_num : int
    dem_source_key : str
    prob_col : str
        Column to use for coloring (default: prob_ensemble).
    test_x, test_y : array-like or None
        HighCon trace coordinates to overlay as gold points.
    title_suffix : str
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Exclude points that failed the shape test
    plot_df = result_df
    if "shape_straightness" in result_df.columns:
        shape_passed = (result_df["shape_straightness"].values >= STRAIGHTNESS_FLOOR) & \
                       (~result_df["shape_has_cycle"].values.astype(bool))
        plot_df = result_df[shape_passed].copy()
        n_excluded = len(result_df) - len(plot_df)
        print(f"  [Spatial Map] Excluded {n_excluded:,} shape-failed points "
              f"({len(plot_df):,} remaining)")
    else:
        print(f"  [Spatial Map] No shape filter column, plotting all {len(plot_df):,} points")

    # Subsample if too many points for plotting
    n = len(plot_df)
    if n > 50000:
        plot_df = plot_df.sample(n=50000, random_state=42)

    scatter = ax.scatter(plot_df["x"], plot_df["y"],
                         c=plot_df[prob_col], cmap="RdYlBu_r",
                         s=2, alpha=0.6, vmin=0, vmax=1, zorder=3)
    plt.colorbar(scatter, ax=ax, label=prob_col, shrink=0.6)

    # Overlay validation/test set as bright yellow points
    if test_x is not None and test_y is not None:
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        # Subsample if too many
        if len(test_x) > 20000:
            idx = np.random.RandomState(42).choice(len(test_x), 20000, replace=False)
            test_x = test_x[idx]
            test_y = test_y[idx]
        ax.scatter(test_x, test_y, c="gold", edgecolors="dimgray",
                   s=4, alpha=0.7, linewidths=0.2, zorder=5,
                   label=f"HighCon traces ({len(test_x):,})")
        ax.legend(markerscale=2, loc="upper right")

    ax.set_xlabel("Easting (m, EPSG:28355)")
    ax.set_ylabel("Northing (m, EPSG:28355)")
    ax.set_aspect("equal")
    score_label = prob_col.replace("_", " ")
    ax.set_title(f"Round {round_num} -- {dem_source_key} -- Spatial {score_label} Map"
                 f"{title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Spatial Map] Saved to {output_path}")


def plot_shape_filter_map(result_df, output_path, round_num, dem_source_key):
    """Plot spatial map showing which points survived the shape filter.

    Green = survived (I/L-shaped), red = filtered out (U/O-shaped).
    """
    if "shape_straightness" not in result_df.columns:
        print(f"  [Shape Map] No shape features in result, skipping")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    survived = result_df[result_df["shape_straightness"] >= STRAIGHTNESS_FLOOR]
    filtered = result_df[result_df["shape_straightness"] < STRAIGHTNESS_FLOOR]

    # Subsample for plotting
    if len(survived) > 30000:
        survived = survived.sample(n=30000, random_state=42)
    if len(filtered) > 30000:
        filtered = filtered.sample(n=30000, random_state=42)

    ax.scatter(survived["x"], survived["y"], c="green", s=1, alpha=0.3,
               label=f"I/L survived ({len(survived):,})")
    if len(filtered) > 0:
        ax.scatter(filtered["x"], filtered["y"], c="red", s=1, alpha=0.3,
                   label=f"U/O filtered ({len(filtered):,})")

    ax.set_xlabel("Easting (m, EPSG:28355)")
    ax.set_ylabel("Northing (m, EPSG:28355)")
    ax.set_aspect("equal")
    ax.set_title(f"Round {round_num} -- {dem_source_key} -- Shape Filter Map")
    ax.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Shape Map] Saved to {output_path}")


def plot_cross_round_comparison(round_results, output_path, dem_source_key):
    """Plot cross-round comparison: how did the high-confidence pool change?

    For each round, shows the count of points above various thresholds.
    Also plots spatial overlay of new vs old high-conf points.
    """
    if len(round_results) <= 1:
        print(f"  [Cross-Round] Only one round, skipping comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: bar chart of high-conf counts per round
    rounds = [r["round"] for r in round_results]
    hc_counts = [r.get("n_high_conf", 0) for r in round_results]
    pos_counts = [r.get("n_positive", 0) for r in round_results]
    neg_counts = [r.get("n_negative", 0) for r in round_results]

    x = np.arange(len(rounds))
    width = 0.25
    axes[0].bar(x - width, pos_counts, width, label="Positives", color="green")
    axes[0].bar(x, neg_counts, width, label="Negatives", color="red")
    axes[0].bar(x + width, hc_counts, width, label="High-conf candidates", color="blue")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{dem_source_key} -- Label Set Growth Across Rounds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"R{r}" for r in rounds])
    axes[0].legend()
    axes[0].set_yscale("log")

    # Right: AUC per round
    aucs = [r.get("auc_xgb", 0) for r in round_results]
    axes[1].bar(x, aucs, width=0.5, color="purple")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("XGBoost AUC")
    axes[1].set_title(f"{dem_source_key} -- Model AUC Across Rounds")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"R{r}" for r in rounds])
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Cross-Round] Saved to {output_path}")


# ── Main pipeline ────────────────────────────────────────────────────

def run_round(dem_source_key, round_num, dem_data, precomputed,
               unlabeled_cache, cached_unl_features=None,
               max_unlabeled=0, prev_explicit_negatives=None):
    """Run one round of the iterative active-learning pipeline.

    Parameters
    ----------
    dem_source_key : str
    round_num : int
        Current round number (1-based). Round k uses Trace1..k as positives.
    dem_data : dict
        Pre-loaded DEM/HAND rasters with keys: dem, hand, resolution, dem_nodata,
        hand_nodata, dem_transform, hand_transform, dem_crs, hand_crs.
    precomputed : dict
        Pre-computed DEM-derived raster dict from precompute_dem_rasters().
    unlabeled_cache : DataFrame or None
        Cached unlabeled pool from a previous round (to avoid reloading).
        If None, loads and computes features from scratch.
    cached_unl_features : DataFrame or None
        Cached feature DataFrame for the unlabeled pool from a previous round.
        If provided and the unlabeled pool is unchanged, features are reused
        instead of recomputed.  Passed explicitly (not via DataFrame.attrs)
        because attrs is fragile across copy/slice operations.
    max_unlabeled : int
        Subsample cap (0 = no cap).
    prev_explicit_negatives : DataFrame or None
        Accumulated explicit negatives from previous rounds.

    Returns
    -------
    dict with round results: round, auc_xgb, auc_lr, n_positive, n_negative,
    n_high_conf, result_df, unlabeled_cache, unl_features,
    explicit_negatives, feature_cols
    """
    print(f"\n{'='*70}")
    print(f"05b-DEMWideScarpInference -- Round {round_num} -- dem_source={dem_source_key}")
    print(f"{'='*70}")

    safe_makedirs(OUTPUT_DIR)
    src = DEM_SOURCES[dem_source_key]

    dem = dem_data["dem"]
    hand = dem_data["hand"]
    resolution = dem_data["resolution"]
    dem_nodata = dem_data["dem_nodata"]
    hand_nodata = dem_data["hand_nodata"]
    dem_transform = dem_data["dem_transform"]
    hand_transform = dem_data["hand_transform"]
    dem_crs = dem_data["dem_crs"]
    hand_crs = dem_data["hand_crs"]

    # ── Determine which traces to use as positives this round ───────
    # Round 1: Trace1; Round 2: Trace1,2; etc.
    # Uses available traces (auto-detected), so missing files are skipped.
    available = _detect_available_traces()
    trace_indices = available[:min(round_num, len(available))]

    if round_num > len(available):
        print(f"\n  [Skip] Round {round_num} requested but only {len(available)} trace(s) "
              f"available ({available}). No more trace files — stopping iterative loop.")
        return None

    if not trace_indices:
        print(f"\n  [Skip] No trace files available for round {round_num}. Skipping.")
        return None

    # ── Load labelled data ────────────────────────────────────────
    print(f"\n[Stage 1] Loading labelled dataset (round {round_num})...")
    bt_lookup = _build_break_type_lookup(dem_source_key)
    highcon = load_highcon_points(dem_source_key, bt_lookup=bt_lookup,
                                   trace_indices=trace_indices)
    negatives = load_confirmed_removed(dem_source_key)

    # Merge explicit negatives from previous rounds
    if prev_explicit_negatives is not None and len(prev_explicit_negatives) > 0:
        print(f"  Merging {len(prev_explicit_negatives)} explicit negatives from previous rounds")
        negatives = pd.concat([negatives, prev_explicit_negatives],
                               ignore_index=True)
        # Deduplicate
        negatives["key"] = negatives["x"].round(2).astype(str) + "_" + \
                            negatives["y"].round(2).astype(str)
        negatives = negatives.drop_duplicates(subset="key").drop(columns=["key"])

    # Build HighCon key set for unlabeled pool decontamination
    highcon_keys = set()
    if len(highcon) > 0:
        highcon_keys = set(highcon["x"].round(2).astype(str) + "_" +
                           highcon["y"].round(2).astype(str))

    # ── Load unlabeled pool (cache if available) ──────────────────
    if unlabeled_cache is not None:
        print(f"\n[Stage 1] Using cached unlabeled pool ({len(unlabeled_cache):,} points)")
        # Remove any HighCon points that were added this round
        cache_keys = unlabeled_cache["x"].round(2).astype(str) + "_" + \
                     unlabeled_cache["y"].round(2).astype(str)
        unlabeled = unlabeled_cache[~cache_keys.isin(highcon_keys)].copy()
        unlabeled = unlabeled.reset_index(drop=True)
        print(f"  After removing new HighCon: {len(unlabeled):,}")
    else:
        unlabeled = load_unlabeled_pool(dem_source_key, highcon_keys=highcon_keys)

        # ── Stage 0: Pre-filter unlabeled pool ────────────────────
        print(f"\n[Stage 0] Applying rule-based pre-filters to unlabeled pool...")
        keep_mask = apply_prefilters(dem, resolution, dem_nodata,
                                    unlabeled["x"].values, unlabeled["y"].values,
                                    dem_transform,
                                    hand=hand, hand_transform=hand_transform,
                                    hand_nodata=hand_nodata,
                                    hand_crs=hand_crs, dem_crs=dem_crs)
        n_before = len(unlabeled)
        unlabeled = unlabeled[keep_mask].reset_index(drop=True)
        print(f"  Pre-filter: {n_before:,} -> {len(unlabeled):,} "
              f"(removed {n_before - len(unlabeled):,})")

        # Subsample unlabeled pool if requested
        if max_unlabeled > 0 and len(unlabeled) > max_unlabeled:
            unlabeled = unlabeled.sample(n=max_unlabeled, random_state=42).reset_index(drop=True)
            print(f"  Subsampled unlabeled pool to {max_unlabeled:,} points")

    print(f"  HighCon (positive): {len(highcon):,}")
    print(f"  Confirmed-removed (negative): {len(negatives):,}")
    print(f"  Unlabeled pool: {len(unlabeled):,}")

    # Also pre-filter negatives (they may contain edge/flat points)
    if len(negatives) > 0:
        neg_keep = apply_prefilters(dem, resolution, dem_nodata,
                                   negatives["x"].values, negatives["y"].values,
                                   dem_transform,
                                   hand=hand, hand_transform=hand_transform,
                                   hand_nodata=hand_nodata,
                                   hand_crs=hand_crs, dem_crs=dem_crs)
        n_neg_before = len(negatives)
        negatives = negatives[neg_keep].reset_index(drop=True)
        print(f"  Negatives pre-filter: {n_neg_before:,} -> {len(negatives):,}")

    # ── Stratify negatives ─────────────────────────────────────────
    print(f"\n[Stage 1] Stratifying negatives...")
    negatives = stratify_negatives(negatives, len(highcon), dem_source_key)
    print(f"  Sampled negatives: {len(negatives):,}")

    # ── Stage 2: Feature engineering ──────────────────────────────
    print(f"\n[Stage 2] Feature engineering (round {round_num})...")

    # Compute features for each group
    print(f"\n  --- HighCon (positive) ---")
    pos_features = compute_all_features(
        dem, hand, resolution, dem_nodata, hand_nodata,
        dem_transform, hand_transform,
        highcon["x"].values, highcon["y"].values,
        highcon, batch_label=" [HighCon]",
        precomputed=precomputed)

    print(f"\n  --- Confirmed-removed (negative) ---")
    neg_features = compute_all_features(
        dem, hand, resolution, dem_nodata, hand_nodata,
        dem_transform, hand_transform,
        negatives["x"].values, negatives["y"].values,
        negatives, batch_label=" [Neg]",
        precomputed=precomputed)

    if unlabeled_cache is not None and cached_unl_features is not None:
        # Reuse cached unlabeled features, but align to the filtered pool
        # (round 2+ removes HighCon points from the cached unlabeled pool)
        cache_keys = unlabeled_cache["x"].round(2).astype(str) + "_" + \
                     unlabeled_cache["y"].round(2).astype(str)
        unl_keys = unlabeled["x"].round(2).astype(str) + "_" + \
                   unlabeled["y"].round(2).astype(str)
        # cached_unl_features aligns with unlabeled_cache (pre-filter);
        # build a key→row map and select only rows still in the filtered pool
        feat_keys = cache_keys.values
        key_to_feat_idx = {k: i for i, k in enumerate(feat_keys)}
        keep_idx = np.array([key_to_feat_idx[k] for k in unl_keys.values
                             if k in key_to_feat_idx], dtype=int)
        unl_features = cached_unl_features.iloc[keep_idx].reset_index(drop=True)
        print(f"\n  --- Unlabeled pool (cached features, {len(keep_idx):,} aligned) ---")
    else:
        print(f"\n  --- Unlabeled pool ---")
        unl_features = compute_all_features(
            dem, hand, resolution, dem_nodata, hand_nodata,
            dem_transform, hand_transform,
            unlabeled["x"].values, unlabeled["y"].values,
            unlabeled, batch_label=" [Unl]",
            precomputed=precomputed)

    # Assemble labelled set
    labeled_df = pd.concat([
        highcon[["x", "y", "trace_id", "label"]].reset_index(drop=True),
        negatives[["x", "y"]].assign(trace_id=0, label=0).reset_index(drop=True),
    ], ignore_index=True)

    labeled_features = pd.concat([pos_features, neg_features], ignore_index=True)
    labeled_df = pd.concat([labeled_df, labeled_features], axis=1)

    # ── Stage 3: Spatial chunk assignment & split ─────────────────
    print(f"\n[Stage 3] Spatial leakage-safe validation split...")
    labeled_df = assign_spatial_chunks(labeled_df)
    train_mask = spatial_train_test_split(labeled_df)
    n_train = train_mask.sum()
    n_val = len(train_mask) - n_train
    print(f"  Train: {n_train:,} | Val: {n_val:,} (spatial chunks)")

    # ── Stage 4: Model training ───────────────────────────────────
    print(f"\n[Stage 4] Training models (round {round_num})...")
    feature_cols = get_feature_columns(labeled_df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X_train = labeled_df.loc[train_mask, feature_cols].values
    y_train = labeled_df.loc[train_mask, "label"].values
    X_val = labeled_df.loc[~train_mask, feature_cols].values
    y_val = labeled_df.loc[~train_mask, "label"].values

    # XGBoost
    print(f"  Training XGBoost...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    auc_xgb = roc_auc_score(y_val, y_proba_xgb)
    print(f"  XGBoost AUC: {auc_xgb:.4f}")
    print(f"  XGBoost classification report:")
    print(classification_report(y_val, (y_proba_xgb >= 0.5).astype(int),
                                target_names=["negative", "positive"]))

    # Logistic Regression baseline
    print(f"  Training Logistic Regression baseline...")
    lr_model, lr_scaler = train_logreg_baseline(X_train, y_train)
    X_val_scaled = lr_scaler.transform(np.nan_to_num(X_val, nan=0.0))
    y_proba_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    auc_lr = roc_auc_score(y_val, y_proba_lr)
    print(f"  LogReg AUC: {auc_lr:.4f}")

    # ── Stage 5: Score unlabeled pool ─────────────────────────────
    print(f"\n[Stage 5] Scoring unlabeled pool (round {round_num})...")
    X_unl = unl_features[feature_cols].values
    # Handle NaN
    X_unl = np.nan_to_num(X_unl, nan=0.0)

    y_proba_unl_xgb = xgb_model.predict_proba(X_unl)[:, 1]
    X_unl_scaled = lr_scaler.transform(X_unl)
    y_proba_unl_lr = lr_model.predict_proba(X_unl_scaled)[:, 1]

    # XGBoost is the primary scorer; LogReg is a baseline-only sanity floor
    result_df = unlabeled[["x", "y", "tier", "break_type"]].copy()
    result_df["prob_xgb_raw"] = y_proba_unl_xgb

    # ── Shape-based rescoring (Stage 5, after raw probability) ─────
    # Zero out U/O-shaped clusters; keep raw score for audit
    if "shape_straightness" in unl_features.columns:
        shape_straightness_vals = unl_features["shape_straightness"].values
        shape_has_cycle_vals = unl_features["shape_has_cycle"].values.astype(bool)
        shape_mask = (shape_straightness_vals >= STRAIGHTNESS_FLOOR) & (~shape_has_cycle_vals)

        result_df["shape_straightness"] = shape_straightness_vals
        result_df["shape_has_cycle"] = shape_has_cycle_vals
        result_df["prob_xgb"] = np.where(shape_mask, y_proba_unl_xgb, 0.0)

        n_shape_filtered = (~shape_mask).sum()
        print(f"  [Shape Filter] {n_shape_filtered:,} points zeroed by U/O filter "
              f"(straightness < {STRAIGHTNESS_FLOOR} or has_cycle)")
    else:
        result_df["prob_xgb"] = y_proba_unl_xgb

    result_df["prob_lr"] = y_proba_unl_lr  # kept for reference, not used for ranking

    # Ensemble score: average of XGBoost raw and LR (before shape filter)
    result_df["prob_ensemble_raw"] = (y_proba_unl_xgb + y_proba_unl_lr) / 2.0
    if "shape_straightness" in result_df.columns:
        result_df["prob_ensemble"] = np.where(shape_mask,
                                               result_df["prob_ensemble_raw"].values,
                                               0.0)
    else:
        result_df["prob_ensemble"] = result_df["prob_ensemble_raw"]

    # Report threshold statistics (XGBoost primary, post-shape-filter)
    print("  [XGBoost post-shape-filter]")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_above = (result_df["prob_xgb"].values >= thresh).sum()
        n_above_raw = (result_df["prob_xgb_raw"].values >= thresh).sum()
        print(f"    Threshold {thresh}: {n_above:,} (raw: {n_above_raw:,})")

    # ── HighCon sanity check: score held-out positives ──────────
    print(f"\n[Stage 5] HighCon sanity check: scoring held-out positives...")
    X_hc = pos_features[feature_cols].values
    X_hc = np.nan_to_num(X_hc, nan=0.0)
    y_proba_hc_xgb = xgb_model.predict_proba(X_hc)[:, 1]
    X_hc_scaled = lr_scaler.transform(X_hc)
    y_proba_hc_lr = lr_model.predict_proba(X_hc_scaled)[:, 1]

    print(f"  HighCon score distribution (all {len(y_proba_hc_xgb)} positives):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    p{pct}: {np.percentile(y_proba_hc_xgb, pct):.4f}")
    print(f"    mean: {y_proba_hc_xgb.mean():.4f}, std: {y_proba_hc_xgb.std():.4f}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_above = (y_proba_hc_xgb >= thresh).sum()
        print(f"  HighCon >= {thresh}: {n_above}/{len(y_proba_hc_xgb)} "
              f"({100*n_above/len(y_proba_hc_xgb):.1f}%)")

    # Break down by train/val split
    hc_val_mask = ~train_mask[:len(pos_features)]
    if hc_val_mask.any():
        hc_val_scores = y_proba_hc_xgb[hc_val_mask]
        hc_train_scores = y_proba_hc_xgb[~hc_val_mask]
        print(f"  Held-out HighCon (val): mean={hc_val_scores.mean():.4f}, "
              f"median={np.median(hc_val_scores):.4f}")
        print(f"  Train HighCon:          mean={hc_train_scores.mean():.4f}, "
              f"median={np.median(hc_train_scores):.4f}")

    # ── Merge HighCon scores into result_df ───────────────────────
    # Score all available HighCon trace points (Trace1.xlsx, Trace2.xlsx, ...)
    # and append them to result_df so every scored point — unlabeled pool
    # AND known HighCon positives — lives in the same _scored_points.csv.
    hc_result_df = highcon[["x", "y", "tier", "break_type"]].copy()
    hc_result_df["prob_xgb_raw"] = y_proba_hc_xgb

    if "shape_straightness" in pos_features.columns:
        hc_shape_straightness = pos_features["shape_straightness"].values
        hc_shape_has_cycle = pos_features["shape_has_cycle"].values.astype(bool)
        hc_shape_mask = (hc_shape_straightness >= STRAIGHTNESS_FLOOR) & \
                        (~hc_shape_has_cycle)

        hc_result_df["shape_straightness"] = hc_shape_straightness
        hc_result_df["shape_has_cycle"] = hc_shape_has_cycle
        hc_result_df["prob_xgb"] = np.where(hc_shape_mask, y_proba_hc_xgb, 0.0)
    else:
        hc_result_df["prob_xgb"] = y_proba_hc_xgb

    hc_result_df["prob_lr"] = y_proba_hc_lr
    hc_result_df["prob_ensemble_raw"] = (y_proba_hc_xgb + y_proba_hc_lr) / 2.0
    if "shape_straightness" in hc_result_df.columns:
        hc_result_df["prob_ensemble"] = np.where(hc_shape_mask,
                                                   hc_result_df["prob_ensemble_raw"].values,
                                                   0.0)
    else:
        hc_result_df["prob_ensemble"] = hc_result_df["prob_ensemble_raw"]

    # Tag source + trace_id for traceability
    hc_result_df["source"] = "HighCon"
    hc_result_df["trace_id"] = highcon["trace_id"].values
    result_df["source"] = "unlabeled"
    result_df["trace_id"] = -1

    # Append HighCon rows so all scores are in one file
    result_df = pd.concat([result_df, hc_result_df], ignore_index=True)
    print(f"\n  [Merge] Appended {len(hc_result_df):,} HighCon points to result_df "
          f"(total: {len(result_df):,})")

    # ── Save round-specific outputs ────────────────────────────────
    round_suffix = f"_r{round_num}"
    out_csv = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_scored_points.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"  Saved scored points to {out_csv}")

    # Also save a "high-confidence" subset (prob_xgb >= 0.5)
    # Only unlabeled points — HighCon are known positives, not candidates
    unlabeled_only = result_df[result_df["source"] == "unlabeled"]
    high_conf = unlabeled_only[unlabeled_only["prob_xgb"] >= 0.5]
    out_high = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_high_conf_candidates.csv"
    high_conf.to_csv(out_high, index=False)
    print(f"  Saved {len(high_conf):,} high-confidence candidates to {out_high}")

    # Export top candidates for human review (candidates for next trace)
    # Only export if there are more available traces beyond this round
    available_all = _detect_available_traces()
    if round_num < len(available_all):
        top_n = min(5000, len(unlabeled_only))
        top_candidates = unlabeled_only.nlargest(top_n, "prob_xgb")[
            ["x", "y", "tier", "break_type", "prob_xgb", "prob_xgb_raw"]
        ].copy()
        top_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_top_candidates.csv"
        top_candidates.to_csv(top_path, index=False)
        print(f"  Saved {len(top_candidates)} top candidates for human review to {top_path}")

    # ── Versioned HighCon audit directory ────────────────────────
    round_dir = HIGHCON_DIR / f"round_{round_num}"
    safe_makedirs(round_dir)
    audit_df = highcon[["x", "y", "tier", "break_type", "trace_id"]].copy()
    audit_df["round"] = round_num
    audit_df["dem_source"] = dem_source_key
    audit_df["n_points"] = len(audit_df)
    audit_path = round_dir / f"{dem_source_key}_labeled_set_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"  Saved labeled-set audit to {audit_path}")

    # Save round metadata
    meta = {
        "round": round_num,
        "dem_source": dem_source_key,
        "traces_used": trace_indices,
        "n_positive": int(len(highcon)),
        "n_negative": int(len(negatives)),
        "n_unlabeled": int(len(unlabeled)),
        "auc_xgb": float(auc_xgb),
        "auc_lr": float(auc_lr),
        "n_high_conf": int(len(high_conf)),
        "feature_cols": feature_cols,
    }
    meta_path = round_dir / f"{dem_source_key}_round_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved round metadata to {meta_path}")

    # ── Stage 6: Sanity checks + spatial plots ────────────────────
    print(f"\n[Stage 6] Sanity checks (round {round_num})...")
    shap_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_shap_summary.png"
    plot_shap_summary(xgb_model, X_val[:5000], feature_cols, shap_path)

    fi_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_feature_importance.png"
    plot_feature_importance(xgb_model, feature_cols, fi_path)

    pr_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_precision_recall.png"
    plot_precision_recall(y_val, y_proba_xgb, pr_path)

    sd_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_score_distribution.png"
    plot_score_distribution(y_proba_unl_xgb, sd_path,
                            hc_proba=y_proba_hc_xgb)

    # Spatial plots — use unlabeled-only subset (result_df now includes HighCon)
    unlabeled_result = result_df[result_df["source"] == "unlabeled"].copy()
    sp_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_spatial_score_map.png"
    # Overlay HighCon trace points used as positives this round
    hc_x = highcon["x"].values if len(highcon) > 0 else None
    hc_y = highcon["y"].values if len(highcon) > 0 else None
    plot_spatial_score_map(unlabeled_result, sp_path, round_num, dem_source_key,
                           prob_col="prob_ensemble",
                           test_x=hc_x, test_y=hc_y)

    if "shape_straightness" in unlabeled_result.columns:
        sf_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_shape_filter_map.png"
        plot_shape_filter_map(unlabeled_result, sf_path, round_num, dem_source_key)

    # Save model artifacts
    model_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_xgb_model.json"
    xgb_model.save_model(str(model_path))
    print(f"  Saved model to {model_path}")

    # Save feature list
    feat_path = OUTPUT_DIR / f"{dem_source_key}{round_suffix}_feature_columns.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"  Saved feature list to {feat_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Round {round_num} complete for {dem_source_key}")
    print(f"  Traces used: {trace_indices}")
    print(f"  XGBoost AUC: {auc_xgb:.4f}")
    print(f"  LogReg AUC:  {auc_lr:.4f}")
    print(f"  HighCon scored: mean={y_proba_hc_xgb.mean():.4f}, "
          f"median={np.median(y_proba_hc_xgb):.4f}")
    print(f"  Unlabeled scored: {len(result_df):,}")
    print(f"  High-conf unlabeled (>=0.5): {len(high_conf):,}")
    print(f"{'='*70}\n")

    return {
        "round": round_num,
        "auc_xgb": float(auc_xgb),
        "auc_lr": float(auc_lr),
        "n_positive": int(len(highcon)),
        "n_negative": int(len(negatives)),
        "n_high_conf": int(len(high_conf)),
        "result_df": result_df,
        "unlabeled_cache": unlabeled,
        "unl_features": unl_features,
        "feature_cols": feature_cols,
    }


def run_pipeline(dem_source_key, max_unlabeled=0, n_rounds=0):
    """Run the full 05b pipeline for a given DEM source.

    Parameters
    ----------
    dem_source_key : str
    max_unlabeled : int
        Subsample unlabeled pool to N points (0=all).
    n_rounds : int
        Number of iterative active-learning rounds.  0 = auto-detect
        (run one round per available trace file found in ``data/HighCon``).
        Round k uses Trace1..k as positives.

        Missing trace files are skipped: if Trace3.xlsx does not exist,
        the pipeline logs a warning and stops after the last available trace.
        With the current collection of 2 traces (Trace1.xlsx + Trace2.xlsx),
        exactly 2 rounds will run.
    """
    available_traces = _detect_available_traces()
    if n_rounds <= 0:
        n_rounds = len(available_traces)
    else:
        n_rounds = min(n_rounds, len(available_traces))

    if n_rounds == 0:
        print(f"\n[Warning] No trace files found in {HIGHCON_DIR}. Nothing to do.")
        return

    print(f"\n{'#'*70}")
    print(f"# 05b-DEMWideScarpInference -- Iterative Active Learning")
    print(f"# dem_source={dem_source_key}, rounds={n_rounds}")
    print(f"# Available traces: {available_traces}")
    print(f"{'#'*70}")

    safe_makedirs(OUTPUT_DIR)
    src = DEM_SOURCES[dem_source_key]

    # ── Load DEM and HAND once (reused across all rounds) ─────────
    print(f"\n[Setup] Loading DEM and HAND rasters...")
    with rasterio.open(src["dem"]) as ds:
        dem = ds.read(1).astype(np.float32)
        dem_transform = ds.transform
        dem_crs = ds.crs
        dem_nodata = ds.nodata
        resolution = abs(ds.res[0])
    print(f"  DEM: shape={dem.shape}, res={resolution}m, nodata={dem_nodata}")

    hand = None
    hand_transform = None
    hand_nodata = None
    hand_crs = None
    if src["hand"].exists():
        with rasterio.open(src["hand"]) as ds:
            hand = ds.read(1).astype(np.float32)
            hand_transform = ds.transform
            hand_nodata = ds.nodata
            hand_crs = ds.crs
        print(f"  HAND: shape={hand.shape}")
    else:
        print(f"  [Warning] HAND raster not found: {src['hand']}")

    dem_data = {
        "dem": dem, "hand": hand, "resolution": resolution,
        "dem_nodata": dem_nodata, "hand_nodata": hand_nodata,
        "dem_transform": dem_transform, "hand_transform": hand_transform,
        "dem_crs": dem_crs, "hand_crs": hand_crs,
    }

    # ── Precompute DEM-derived rasters once ────────────────────────
    print(f"\n[Setup] Precomputing DEM-derived rasters...")
    precomputed = precompute_dem_rasters(dem, resolution, dem_nodata)

    # ── Iterative active-learning loop ─────────────────────────────
    round_results = []
    unlabeled_cache = None
    cached_unl_features = None
    explicit_negatives = None

    for round_num in range(1, n_rounds + 1):
        result = run_round(
            dem_source_key, round_num, dem_data, precomputed,
            unlabeled_cache, cached_unl_features=cached_unl_features,
            max_unlabeled=max_unlabeled,
            prev_explicit_negatives=explicit_negatives)

        # Trace exhaustion check: run_round returns None when no more trace
        # files are available for the requested round.
        if result is None:
            print(f"\n  [Pipeline] No more trace files available after round {round_num - 1}. "
                  f"Stopping iterative loop.")
            break

        round_results.append(result)

        # Cache unlabeled pool + features for next round (explicit pass,
        # not DataFrame.attrs which is fragile across copy/slice)
        unlabeled_cache = result["unlabeled_cache"]
        cached_unl_features = result["unl_features"]

        # In a real workflow, explicit negatives would be human-reviewed.
        # For now, we auto-generate explicit negatives from the bottom
        # of the scored pool (lowest probability points that are NOT
        # in any HighCon trace) -- these are confident non-scarps.
        result_df = result["result_df"]
        # Only consider unlabeled points for auto-negatives (result_df now
        # also contains HighCon positives with a "source" column)
        unl_for_neg = result_df[result_df["source"] == "unlabeled"] \
            if "source" in result_df.columns else result_df
        if len(unl_for_neg) > 0:
            bottom_n = min(2000, len(unl_for_neg) // 10)
            auto_neg = unl_for_neg.nsmallest(bottom_n, "prob_xgb")[
                ["x", "y", "tier", "break_type"]
            ].copy()
            auto_neg["label"] = 0
            save_explicit_negatives(auto_neg, dem_source_key, round_num)

            # Accumulate
            if explicit_negatives is None:
                explicit_negatives = auto_neg
            else:
                explicit_negatives = pd.concat([explicit_negatives, auto_neg],
                                                ignore_index=True)
                explicit_negatives["key"] = explicit_negatives["x"].round(2).astype(str) + "_" + \
                    explicit_negatives["y"].round(2).astype(str)
                explicit_negatives = explicit_negatives.drop_duplicates(subset="key").drop(columns=["key"])

    # ── Cross-round comparison ────────────────────────────────────
    if len(round_results) > 1:
        print(f"\n[Post-Processing] Generating cross-round comparison...")
        cr_path = OUTPUT_DIR / f"{dem_source_key}_cross_round_comparison.png"
        plot_cross_round_comparison(round_results, cr_path, dem_source_key)

        # Save summary CSV
        summary_rows = []
        for r in round_results:
            summary_rows.append({
                "round": r["round"],
                "n_positive": r["n_positive"],
                "n_negative": r["n_negative"],
                "n_high_conf": r["n_high_conf"],
                "auc_xgb": r["auc_xgb"],
                "auc_lr": r["auc_lr"],
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / f"{dem_source_key}_round_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved round summary to {summary_path}")
        print(summary_df.to_string(index=False))

    print(f"\n{'#'*70}")
    print(f"# Iterative pipeline complete for {dem_source_key}")
    print(f"# Rounds: {len(round_results)}")
    if round_results:
        print(f"# Final round AUC: {round_results[-1]['auc_xgb']:.4f}")
    print(f"{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="05b-DEMWideScarpInference: ML-based wide-area scarp inference "
                    "with iterative active learning")
    parser.add_argument("--dem-source", default="dem10m",
                        choices=list(DEM_SOURCES.keys()),
                        help="DEM source to process (default: dem10m)")
    parser.add_argument("--max-unlabeled", type=int, default=0,
                        help="Subsample unlabeled pool to N points (0=all). "
                             "Useful for quick test runs.")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of iterative active-learning rounds "
                             "(0=auto, one per available trace). "
                             "Round k uses Trace1..k as positives.")
    args = parser.parse_args()

    run_pipeline(args.dem_source, max_unlabeled=args.max_unlabeled,
                 n_rounds=args.rounds)


if __name__ == "__main__":
    main()
