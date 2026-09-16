#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05a-KnownTraceScarpValidation.py -- known-trace, curvature-only scarp test.

This deliberately does *not* import candidate tiles, an autoencoder, cosine
similarities, or Script 05 picks.  It uses the same known-trace reconstruction,
native LiDAR, HAND exclusion, TV scale space, and scarp dimensions as Script
05, but asks the inverse question: which parts of a catalogued trace have an
independently observable pixel-scale scarp base?

Outputs are written to ``data/derivatives/known_trace_scarp_validation``.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.fill import fillnodata
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, Resampling
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           label, laplace, map_coordinates, maximum_filter)
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from skimage.restoration import denoise_tv_chambolle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from derivative_common import (DATA_DIR, OUT_DIR, STRUCT_PTS_PATH, safe_makedirs,
                               reconstruct_fault_traces, _order_points_nn,
                               compute_hillshade)


# Script 05 constants retained here so this script is executable on its own.
FAULT_DB_SOURCES = ["FSD1", "CWvic1", "GeoStru2501", "GeoStru501"]
FAULT_VALIDATION_TOLERANCE_M = 1000
MIN_SCARP_HEIGHT_M = 0.2
SCARP_WIDTH_DISCRETE_MAX_M = 200
LSM_MIN_LENGTH_M = 20
FALLBACK_MAX_SINUOSITY = 2.5
FALLBACK_MAX_PERIODICITY = 0.35
HAND_INFLUENCE_THRESHOLD_M = 2.0
HAND_SEGMENT_MAX_FRACTION = 0.9
LSM_HAND_BUFFER_M = 20
LSM_EDGE_BUFFER_M = 20
LSM_PERSISTENCE_K = 2
LSM_SCALE_NORM_GAMMA = 0.5
LSM_HYSTERESIS_LOW_SIGMA = 0.5
LSM_HYSTERESIS_HIGH_SIGMA = 4.0
LSM_NODATA = -9999.0
MAX_LOCAL_WINDOW_PIXELS = 16_000_000
SEED_WINDOW_M = 1000
CLASS3_WINDOW_M = 400.0
SEED_SPACING_M = 500.0
PRIOR_PROFILE_MATCH_M = 300.0
DRAINAGE_MATCH_M = 300.0
POSITIVE_PRIOR_CLASSES = {"Class 1", "Class 2"}
PHASE1_AUDIT_DIR = OUT_DIR / "phase1_audit"

# Finer than the whole-DEM LSM; all are applied only to a trace buffer.
LOCAL_TV_WEIGHTS = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
TARGET_TV_SIGMAS_M = [20.0, 28.0, 40.0, 56.0, 80.0, 113.0, 160.0]
BASE_RAY_LENGTH_M = 800
BASE_BACKGROUND_FRACTION = 0.15
CHAIN_SAMPLE_SPACING_M = 5.0
LOCAL_FIG_BUFFER_M = 1000.0
LOCAL_FIG_MAX_SCARP_LENGTH_M = 4000.0
MIN_WINDOW_PX = 40
RAY_MIN_PIXELS = 50
RAY_DEBUG_MAX_FAILURES = 3
GRAD_DECAY_THRESHOLD = 0.3
CLUSTER_CONNECT_MAX_GAP_M = 100.0
CLUSTER_CONNECT_MAX_ANGLE_DEG = 30.0
SCARP_SLOPE_FLAT_THRESHOLD = 0.05
SCARP_SLOPE_DROP_FACTOR = 0.3
SCARP_SLOPE_MIN_STEPS = 3
MIN_SCARP_WIDTH_M = 30.0
MAX_SCARP_HEIGHT_WIDTH_RATIO = 0.4
MIN_SCARP_ASYMMETRY_SCORE = 0.3
MIN_LOCAL_SINUOSITY = 1.02
PROFILE_HALF_WIDTH_M = 50.0
PROFILE_SAMPLE_COUNT = 5
SOURCE_COLORS = {"FSD1": "#d73027", "CWvic1": "#4575b4",
                 "GeoStru2501": "#1a9850", "GeoStru501": "#984ea3"}

# Slope break detection thresholds (relaxed for subtle scarp detection).
# These control when _find_slope_break() decides the walking ray has reached
# the base or crest of a scarp.  Relaxed values catch gentler, degraded
# scarps that don't exhibit textbook slope transitions.
SLOPE_BREAK_PRIMARY_RATIO = 0.4       # gentle -> flat transition (was 0.3)
SLOPE_BREAK_INTERMEDIATE_RATIO = 0.7  # steep -> gentle transition (was 0.6)
SLOPE_BREAK_RISE_FACTOR = 2.0         # early-exit when slope rises vs prev (was 1.5)
SLOPE_BREAK_DROP_FACTOR = 0.5         # early-exit when slope drops vs prev (new, mirror)
SLOPE_BREAK_MIN_START_SLOPE = 0.003   # minimum start slope to attempt ray tracing (was 0.01)
SLOPE_BREAK_INFLECTION_RATIO = 0.5    # slope-change inflection: |local - prev| / start < this triggers break
SLOPE_BREAK_INFLECTION_MIN_DELTA = 0.002  # min absolute |local - prev| for inflection (filters noise)

# Soft classify fallback: when ray tracing fails to find slope breaks,
# candidates with strong curvature signal can still be recorded as
# lower-confidence inferred points.
FALLBACK_WALK_M = 50.0                # walk distance for rough height estimate
FALLBACK_MIN_HEIGHT_M = 0.05          # minimum height for fallback points (was 0.1, lowered for gentle scarps)
FALLBACK_MIN_LSM_RATIO = 1.0         # LSM must be >= medium threshold * this ratio
FALLBACK_WALK_ADAPTIVE_M = 150.0       # extended walk distance when initial 50m gives too little height

GENTLE_SLOPE_MODE = False
if GENTLE_SLOPE_MODE:
    MIN_SCARP_HEIGHT_M = 0.2
    LSM_HYSTERESIS_LOW_SIGMA = 1.0
    LSM_HYSTERESIS_HIGH_SIGMA = 1.5
    LOCAL_TV_WEIGHTS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]


def _ramp(value, lo, hi, inverse=False):
    """Continuous 0--1 score; ``inverse`` makes smaller values better."""
    if hi <= lo:
        score = float(value >= hi)
    else:
        score = float(np.clip((value - lo) / (hi - lo), 0, 1))
    return 1.0 - score if inverse else score


def _check_scarp_asymmetry(dem_filled, base_row, base_col, crest_row, crest_col,
                           dr_down, dc_down, dr_up, dc_up, resolution, debug=False):
    """Natural scarps are asymmetric: steep on the scarp face, gentler behind."""
    base_row_i = int(round(base_row))
    base_col_i = int(round(base_col))
    crest_row_i = int(round(crest_row))
    crest_col_i = int(round(crest_col))
    if not (0 <= base_row_i < dem_filled.shape[0] and 0 <= base_col_i < dem_filled.shape[1] and
            0 <= crest_row_i < dem_filled.shape[0] and 0 <= crest_col_i < dem_filled.shape[1]):
        return 0.0

    crest_val = float(dem_filled[crest_row_i, crest_col_i])
    base_val = float(dem_filled[base_row_i, base_col_i])
    face_dist = max(np.hypot(crest_row - base_row, crest_col - base_col) * float(resolution), 1e-6)
    scarp_face_slope = abs(crest_val - base_val) / face_dist

    behind_row = crest_row + dr_up * 3.0
    behind_col = crest_col + dc_up * 3.0
    if (0 <= int(round(behind_row)) < dem_filled.shape[0] and
            0 <= int(round(behind_col)) < dem_filled.shape[1]):
        behind_val = float(dem_filled[int(round(behind_row)), int(round(behind_col))])
        behind_slope = abs(crest_val - behind_val) / (3.0 * float(resolution))
    else:
        behind_slope = 0.0

    ahead_row = base_row + dr_down * 3.0
    ahead_col = base_col + dc_down * 3.0
    if (0 <= int(round(ahead_row)) < dem_filled.shape[0] and
            0 <= int(round(ahead_col)) < dem_filled.shape[1]):
        ahead_val = float(dem_filled[int(round(ahead_row)), int(round(ahead_col))])
        ahead_slope = abs(base_val - ahead_val) / (3.0 * float(resolution))
    else:
        ahead_slope = 0.0

    asymmetry = (scarp_face_slope / (behind_slope + 0.01)) * (scarp_face_slope / (ahead_slope + 0.01))
    asymmetry_score = float(np.clip(asymmetry / 100.0, 0.0, 1.0))
    if debug:
        print(f"    [asym] scarp_face={scarp_face_slope:.4f}, behind={behind_slope:.4f}, "
              f"ahead={ahead_slope:.4f}, ratio={asymmetry:.2f}, score={asymmetry_score:.3f}")
    return asymmetry_score


def _local_sinuosity(points, window_size=5):
    """Compute average sinuosity over short sliding windows."""
    if len(points) < window_size:
        return 0.0
    sinuosities = []
    for idx in range(len(points) - window_size + 1):
        segment = np.asarray(points[idx:idx + window_size], dtype=float)
        path = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
        straight = float(np.linalg.norm(segment[-1] - segment[0]))
        if straight > 0:
            sinuosities.append(path / straight)
    return float(np.mean(sinuosities)) if sinuosities else 0.0


def _compute_segment_confidence(segment_evidence, segment_coords, trace,
                                river_mask, transform, resolution):
    """Compute 0-10 confidence using combined geometric and contextual evidence."""
    evidence = segment_evidence
    coords = np.asarray(segment_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return 0.0, {
            "sinuosity": np.nan,
            "dist_to_trace_m": np.nan,
            "river_clearance_m": np.nan,
            "mean_asymmetry": np.nan,
            "mean_tier": np.nan,
        }

    hw_score = _ramp(float(evidence.height_width_ratio.mean()), 0.01, 0.15)
    pers_score = _ramp(float(evidence.persistence_scales.mean()), 2.0, 7.0)

    path = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(coords[-1] - coords[0]))
    sinuosity = path / straight if straight > 1e-6 else 1.0
    sin_score = _ramp(sinuosity, 1.0, 2.5, inverse=True)

    trace_tree = cKDTree(trace["coords"])
    dist_to_trace = float(np.median(trace_tree.query(coords)[0]))
    dist_score = _ramp(dist_to_trace, 0.0, 500.0, inverse=True)

    cols, rows = (~transform) * (coords[:, 0], coords[:, 1])
    rr = np.clip(np.rint(rows).astype(int), 0, river_mask.shape[0] - 1)
    cc = np.clip(np.rint(cols).astype(int), 0, river_mask.shape[1] - 1)
    river_dist = distance_transform_edt(~river_mask)[rr, cc] * float(resolution)
    river_clearance = float(np.median(river_dist)) if river_dist.size else 0.0
    river_score = _ramp(river_clearance, 20.0, 200.0)

    mean_asymmetry = float(evidence.asymmetry.mean()) if "asymmetry" in evidence.columns else (
        float(evidence.asymmetry_score.mean()) if "asymmetry_score" in evidence.columns else 0.0
    )
    asym_score = _ramp(mean_asymmetry, 0.05, 0.5)

    mean_tier = float(evidence.candidate_tier.mean()) if "candidate_tier" in evidence.columns else 3.0
    tier_score = _ramp(mean_tier, 1.0, 3.0, inverse=True)

    score = 10.0 * (
        0.20 * hw_score +
        0.15 * pers_score +
        0.15 * sin_score +
        0.15 * dist_score +
        0.15 * river_score +
        0.10 * asym_score +
        0.10 * tier_score
    )

    score = float(np.clip(score, 0.0, 10.0))
    diagnostics = {
        "sinuosity": sinuosity,
        "dist_to_trace_m": dist_to_trace,
        "river_clearance_m": river_clearance,
        "mean_asymmetry": mean_asymmetry,
        "mean_tier": mean_tier,
    }
    return score, diagnostics


def _validate_scarp_profile(dem, segment_coords, resolution, transform):
    """Sample perpendicular elevation profiles and score scarp-like shape."""
    coords = np.asarray(segment_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return False, 0.0, []

    direction = coords[-1] - coords[0]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return False, 0.0, []
    direction /= direction_norm
    perp_dir = np.array([-direction[1], direction[0]], dtype=float)

    sample_indices = np.unique(np.linspace(0, len(coords) - 1, num=min(PROFILE_SAMPLE_COUNT, len(coords)), dtype=int))
    distances = np.arange(-PROFILE_HALF_WIDTH_M, PROFILE_HALF_WIDTH_M + float(resolution), float(resolution))
    profile_scores = []
    profile_data = []

    for idx in sample_indices:
        center = coords[idx]
        profile_pts = center + distances[:, None] * perp_dir[None, :]
        cols, rows = (~transform) * (profile_pts[:, 0], profile_pts[:, 1])
        finite = (rows >= 0) & (cols >= 0) & (rows < dem.shape[0]) & (cols < dem.shape[1])
        if np.count_nonzero(finite) < 5:
            continue

        sample_rows = rows[finite]
        sample_cols = cols[finite]
        profile = map_coordinates(dem, [sample_rows, sample_cols], order=1, mode="nearest")
        if profile.size < 5:
            continue

        slope = np.abs(np.diff(profile))
        if slope.size == 0:
            continue

        max_slope_idx = int(np.argmax(slope))
        before_end = max(1, max_slope_idx - 2)
        after_start = min(len(slope) - 1, max_slope_idx + 2)
        before_slope = float(np.mean(slope[:before_end])) if before_end > 0 else 0.0
        after_slope = float(np.mean(slope[after_start:])) if after_start < len(slope) else 0.0
        scarp_slope = float(slope[max_slope_idx])
        score = 1.0 if scarp_slope > 2.0 * max(before_slope, after_slope) else 0.0
        profile_scores.append(score)
        profile_data.append({
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "score": score,
            "max_slope": scarp_slope,
            "before_slope": before_slope,
            "after_slope": after_slope,
        })

    if not profile_scores:
        return False, 0.0, profile_data

    mean_score = float(np.mean(profile_scores))
    return mean_score > 0.5, mean_score, profile_data


def _load_known_points(bounds=None, buffer_m=0.0):
    """Load known-source points, optionally clipped to DEM bounds (+buffer)."""
    if not STRUCT_PTS_PATH.exists():
        raise FileNotFoundError(f"Structural inventory not found: {STRUCT_PTS_PATH}")
    parts = []
    xmin = xmax = ymin = ymax = None
    if bounds is not None:
        xmin = float(bounds.left) - float(buffer_m)
        xmax = float(bounds.right) + float(buffer_m)
        ymin = float(bounds.bottom) - float(buffer_m)
        ymax = float(bounds.top) + float(buffer_m)
    for chunk in pd.read_csv(STRUCT_PTS_PATH, usecols=["XCoord", "YCoord", "source"],
                             chunksize=500_000):
        mask = chunk["source"].isin(FAULT_DB_SOURCES)
        if bounds is not None:
            mask &= ((chunk["XCoord"] >= xmin) & (chunk["XCoord"] <= xmax) &
                     (chunk["YCoord"] >= ymin) & (chunk["YCoord"] <= ymax))
        selected = chunk.loc[mask]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise RuntimeError("No known-fault points found after source/extent filtering.")
    pts = pd.concat(parts, ignore_index=True)
    if pts.empty:
        raise RuntimeError("No points found for the configured known-fault sources.")
    return pts.dropna(subset=["XCoord", "YCoord"])


def _window_for_trace(src, coords, buffer_m):
    return _window_from_bounds_clipped(src, coords.min(axis=0)[0] - buffer_m,
                                       coords.min(axis=0)[1] - buffer_m,
                                       coords.max(axis=0)[0] + buffer_m,
                                       coords.max(axis=0)[1] + buffer_m)


def _min_half_window_buffer_m(resolution, min_window_px=MIN_WINDOW_PX):
    """Half-width buffer needed to ensure a minimum pixel window side length."""
    return 0.5 * float(min_window_px) * float(resolution)


def _tv_weights_for_resolution(resolution):
    """Map target physical smoothing scales to TV weights for this resolution."""
    res = max(float(resolution), 1e-6)
    return [float((sigma / res) ** 2) for sigma in _target_sigmas_for_resolution(res)]


def _target_sigmas_for_resolution(resolution):
    """Return physically meaningful TV smoothing scales for a DEM resolution."""
    res = max(float(resolution), 1e-6)
    if res >= 5.0:
        return [float(sigma) for sigma in TARGET_TV_SIGMAS_M]
    base = res * 2.0
    ratio = np.sqrt(2.0)
    return [float(base * (ratio ** i)) for i in range(7)]


def _effective_persistence_k(resolution, n_scales):
    """Use a slightly lower persistence vote at coarse resolutions."""
    if resolution >= 20:
        return max(2, min(LSM_PERSISTENCE_K - 1, int(n_scales)))
    return max(2, min(LSM_PERSISTENCE_K, int(n_scales)))


def _min_vals_required(interior_pixels):
    """Adaptive minimum support for hysteresis statistics."""
    return int(max(10, min(30, np.ceil(0.02 * float(interior_pixels)))))


def _window_from_bounds_clipped(src, xmin, ymin, xmax, ymax):
    bounds = src.bounds
    xmin = max(float(xmin), float(bounds.left))
    xmax = min(float(xmax), float(bounds.right))
    ymin = max(float(ymin), float(bounds.bottom))
    ymax = min(float(ymax), float(bounds.top))
    if xmax <= xmin or ymax <= ymin:
        return None, "outside_dem"
    raw = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    col_off = max(0, int(np.floor(raw.col_off)))
    row_off = max(0, int(np.floor(raw.row_off)))
    col_end = min(src.width, int(np.ceil(raw.col_off + raw.width)))
    row_end = min(src.height, int(np.ceil(raw.row_off + raw.height)))
    width = col_end - col_off
    height = row_end - row_off
    if width < 3 or height < 3:
        return None, "too_small"
    win = Window(col_off, row_off, width, height)
    if width * height > MAX_LOCAL_WINDOW_PIXELS:
        return None, f"too_large ({width}x{height} px)"
    return win, None


def _load_phase1_priors(dem_source):
    expr_path = PHASE1_AUDIT_DIR / f"{dem_source}_expression_quality.csv"
    expr_points_path = PHASE1_AUDIT_DIR / f"{dem_source}_expression_quality_points.csv"
    scarp_path = PHASE1_AUDIT_DIR / f"{dem_source}_scarp_analysis.csv"
    drain_path = PHASE1_AUDIT_DIR / f"{dem_source}_drainage_anomalies.csv"
    expr = pd.read_csv(expr_path) if expr_path.exists() else pd.DataFrame()
    expr_points = pd.read_csv(expr_points_path) if expr_points_path.exists() else pd.DataFrame()
    scarp = pd.read_csv(scarp_path) if scarp_path.exists() else pd.DataFrame()
    drain = pd.read_csv(drain_path) if drain_path.exists() else pd.DataFrame()
    if not scarp.empty:
        scarp["scarp_type"] = scarp["scarp_type"].fillna("none")
        scarp["has_scarp_evidence"] = ((scarp["scarp_type"].str.lower() != "none") |
                                       (scarp["scarp_height_m"].fillna(0) >= MIN_SCARP_HEIGHT_M))
    if expr.empty and not expr_points.empty:
        class_rank = {"Class 1": 3, "Class 2": 2, "Class 3": 1}
        rows = []
        for tid, grp in expr_points.groupby("trace_id"):
            qclass = sorted(grp["quality_class"].dropna().unique(),
                            key=lambda c: class_rank.get(c, 0), reverse=True)[0]
            rows.append({
                "trace_id": tid,
                "source": grp["source"].iloc[0] if "source" in grp.columns else "",
                "centroid_x": grp["sample_x"].mean() if "sample_x" in grp.columns else np.nan,
                "centroid_y": grp["sample_y"].mean() if "sample_y" in grp.columns else np.nan,
                "trace_length_m": np.nan,
                "quality_class": qclass,
                "mean_scarp_height_m": grp["scarp_height_m"].where(grp["scarp_type"] != "none").mean(),
                "max_scarp_height_m": grp["scarp_height_m"].max(),
                "structural_context": "derived_from_point_expression_quality",
                "n_profiles": len(grp),
                "detection_rate": float((grp["scarp_type"] != "none").mean()) if "scarp_type" in grp.columns else 0.0,
                "mean_scarp_slope_deg": grp["max_slope_deg"].where(grp["scarp_type"] != "none").mean() if "max_slope_deg" in grp.columns else np.nan,
                "n_drain_anomalies": int((grp.get("n_near_drainage_anomalies", pd.Series(dtype=float)) > 0).sum()),
                "slope_anomaly_deg": grp["slope_anomaly_deg"].max() if "slope_anomaly_deg" in grp.columns else np.nan,
                "svf_anomaly": grp["svf_anomaly"].max() if "svf_anomaly" in grp.columns else np.nan,
                "rationale": "derived_from_point_expression_quality",
            })
        expr = pd.DataFrame(rows)
    return {"expression": expr, "expression_points": expr_points, "scarp": scarp, "drainage": drain}


def _attach_trace_priors(traces, priors):
    expr = priors["expression"]
    scarp = priors["scarp"]
    expr_lookup = expr.set_index("trace_id") if not expr.empty and "trace_id" in expr.columns else None
    scarp_groups = scarp.groupby("trace_id") if not scarp.empty and "trace_id" in scarp.columns else None
    for trace in traces:
        trace["prior_quality_class"] = "unclassified"
        trace["prior_detection_rate"] = np.nan
        trace["prior_n_profiles"] = 0
        trace["prior_seed_count"] = 0
        trace["prior_scarp_seed_df"] = pd.DataFrame()
        if expr_lookup is not None and trace["trace_id"] in expr_lookup.index:
            row = expr_lookup.loc[trace["trace_id"]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            trace["prior_quality_class"] = row.get("quality_class", "unclassified")
            trace["prior_detection_rate"] = row.get("detection_rate", np.nan)
            trace["prior_n_profiles"] = int(row.get("n_profiles", 0) or 0)
        if scarp_groups is not None and trace["trace_id"] in scarp_groups.groups:
            trace_scarp = scarp_groups.get_group(trace["trace_id"]).copy()
            trace_scarp = trace_scarp[trace_scarp["has_scarp_evidence"]].copy()
            trace["prior_seed_count"] = len(trace_scarp)
            trace["prior_scarp_seed_df"] = trace_scarp
    return traces


def _coords_in_bounds(coords, bounds):
    return ((coords[:, 0] >= bounds.left) & (coords[:, 0] <= bounds.right) &
            (coords[:, 1] >= bounds.bottom) & (coords[:, 1] <= bounds.top))


def _split_coords_by_bounds(coords, bounds):
    inside = _coords_in_bounds(coords, bounds)
    runs = []
    start = None
    for idx, flag in enumerate(inside):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append(coords[start:idx])
            start = None
    if start is not None:
        runs.append(coords[start:])
    return [run for run in runs if len(run) > 0]


def _estimate_window_pixels(coords, buffer_m, resolution):
    xmin, ymin = coords.min(axis=0) - buffer_m
    xmax, ymax = coords.max(axis=0) + buffer_m
    width_px = max(1.0, (xmax - xmin) / resolution)
    height_px = max(1.0, (ymax - ymin) / resolution)
    return int(np.ceil(width_px * height_px))


def _thin_seed_points(seed_df, min_spacing_m=SEED_SPACING_M):
    if seed_df.empty:
        return seed_df
    keep = []
    last_xy = None
    for row in seed_df.itertuples(index=False):
        xy = np.array([row.sample_x, row.sample_y], dtype=float)
        if last_xy is None or np.linalg.norm(xy - last_xy) >= min_spacing_m:
            keep.append(True)
            last_xy = xy
        else:
            keep.append(False)
    return seed_df.loc[keep].copy()


def _chunk_trace_run(src, coords, buffer_m, kind):
    resolution = abs(src.res[0])
    windows = []
    start = 0
    while start < len(coords):
        stop = start + 1
        last_good = stop
        while stop <= len(coords):
            subset = coords[start:stop]
            if _estimate_window_pixels(subset, buffer_m, resolution) > MAX_LOCAL_WINDOW_PIXELS and stop - start > 1:
                break
            last_good = stop
            stop += 1
        subset = coords[start:last_good]
        win, reason = _window_for_trace(src, subset, buffer_m)
        if win is not None:
            windows.append({"window": win, "kind": kind, "reason": None})
        elif reason != "outside_dem":
            windows.append({"window": None, "kind": kind, "reason": reason})
        if last_good >= len(coords):
            break
        start = max(start + 1, last_good - 1)
    return windows


def _trace_windows(src, trace, buffer_m):
    bounds = src.bounds
    resolution = abs(src.res[0])
    min_half_buffer = _min_half_window_buffer_m(resolution)
    quality_class = trace.get("prior_quality_class", "unclassified")
    scan_buffer_base = buffer_m if quality_class in POSITIVE_PRIOR_CLASSES else min(buffer_m, CLASS3_WINDOW_M)
    scan_buffer_m = max(float(scan_buffer_base), min_half_buffer)
    windows = []
    seeds = trace.get("prior_scarp_seed_df", pd.DataFrame())
    if not seeds.empty:
        seeds = seeds[_coords_in_bounds(seeds[["sample_x", "sample_y"]].values.astype(float), bounds)]
        seeds = _thin_seed_points(seeds)
        # seed_buffer = max(min(scan_buffer_m, SEED_WINDOW_M), min_half_buffer)
        seed_buffer = max(1000.0, min_half_buffer)  # Force 1000m half-window = 2000m full = 200px at 10m
        for row in seeds.itertuples(index=False):
            # Create multiple overlapping windows shifted by half-width
            for offset_x in [0, seed_buffer/2]:
                for offset_y in [0, seed_buffer/2]:
                    win, reason = _window_from_bounds_clipped(
                        src,
                        row.sample_x - seed_buffer + offset_x,
                        row.sample_y - seed_buffer + offset_y,
                        row.sample_x + seed_buffer + offset_x,
                        row.sample_y + seed_buffer + offset_y
                    )
                    windows.append({"window": win, "kind": "seed", "reason": reason})
    for run in _split_coords_by_bounds(trace["coords"], bounds):
        windows.extend(_chunk_trace_run(src, run, scan_buffer_m, "trace"))
    deduped = []
    seen = set()
    for spec in windows:
        win = spec["window"]
        if win is None:
            continue
        key = (int(win.col_off), int(win.row_off), int(win.width), int(win.height))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _annotate_segments_with_priors(segments, trace, priors):
    if not segments:
        return segments
    scarp = trace.get("prior_scarp_seed_df", pd.DataFrame())
    drain = priors["drainage"]
    scarp_tree = None
    drain_tree = None
    if not scarp.empty:
        scarp_xy = scarp[["sample_x", "sample_y"]].values.astype(float)
        scarp_tree = cKDTree(scarp_xy)
    if not drain.empty and {"x", "y"}.issubset(drain.columns):
        drain_tree = cKDTree(drain[["x", "y"]].values.astype(float))
    for seg in segments:
        seg["prior_quality_class"] = trace.get("prior_quality_class", "unclassified")
        seg["prior_detection_rate"] = trace.get("prior_detection_rate", np.nan)
        seg["prior_seed_count"] = trace.get("prior_seed_count", 0)
        seg["nearest_prior_profile_dist_m"] = np.nan
        seg["prior_profile_height_m"] = np.nan
        seg["prior_profile_width_m"] = np.nan
        seg["prior_profile_type"] = None
        seg["height_delta_vs_prior_m"] = np.nan
        seg["width_delta_vs_prior_m"] = np.nan
        seg["prior_profile_agreement"] = "no_prior_seed"
        seg["nearest_drainage_anomaly_m"] = np.nan
        seg["drainage_anomaly_type"] = None
        seg["drainage_corroborates"] = False
        centroid = np.array([seg["centroid_x"], seg["centroid_y"]], dtype=float)
        if scarp_tree is not None:
            dist, idx = scarp_tree.query(centroid)
            prior_row = scarp.iloc[int(idx)]
            seg["nearest_prior_profile_dist_m"] = float(dist)
            seg["prior_profile_height_m"] = float(prior_row.get("scarp_height_m", np.nan))
            seg["prior_profile_width_m"] = float(prior_row.get("scarp_width_m", np.nan))
            seg["prior_profile_type"] = prior_row.get("scarp_type")
            if np.isfinite(seg["prior_profile_height_m"]):
                seg["height_delta_vs_prior_m"] = seg["mean_height_m"] - seg["prior_profile_height_m"]
            if np.isfinite(seg["prior_profile_width_m"]):
                seg["width_delta_vs_prior_m"] = seg["mean_width_m"] - seg["prior_profile_width_m"]
            if dist <= PRIOR_PROFILE_MATCH_M:
                type_match = seg["prior_profile_type"] in (None, "none") or (
                    (seg["prior_profile_type"] == "discrete" and seg["mean_width_m"] <= SCARP_WIDTH_DISCRETE_MAX_M) or
                    (seg["prior_profile_type"] == "monocline" and seg["mean_width_m"] > SCARP_WIDTH_DISCRETE_MAX_M)
                )
                height_match = (not np.isfinite(seg["height_delta_vs_prior_m"])) or abs(seg["height_delta_vs_prior_m"]) <= 1.5
                seg["prior_profile_agreement"] = "matched" if type_match and height_match else "refined"
        if drain_tree is not None:
            dist, idx = drain_tree.query(centroid)
            seg["nearest_drainage_anomaly_m"] = float(dist)
            if dist <= DRAINAGE_MATCH_M:
                seg["drainage_corroborates"] = True
                seg["drainage_anomaly_type"] = drain.iloc[int(idx)].get("anomaly_type")
    return segments


def _read_mask_on_window(mask_paths, dem_transform, dem_crs, shape):
    """Build the Script-05 HAND/river mask and reproject it to one DEM window."""
    if not mask_paths or not mask_paths[0].exists():
        return np.zeros(shape, dtype=bool)
    hand_path, *river_paths = mask_paths
    with rasterio.open(hand_path) as src:
        hand = src.read(1).astype(np.float32)
        river = (hand <= 0) & np.isfinite(hand)
        transform, crs, res = src.transform, src.crs, abs(src.res[0])
    for path in river_paths:
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            value = src.read(1)
            valid = np.isfinite(value.astype(float))
            if src.nodata is not None:
                valid &= value != src.nodata
            # All rem_hand rasters use this same grid.  If that ever changes,
            # reproject each mask separately rather than silently combining it.
            if src.transform != transform or src.crs != crs:
                extra = np.zeros_like(river, dtype=np.uint8)
                reproject(((value > 0) & valid).astype(np.uint8), extra,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.nearest)
                river |= extra.astype(bool)
            else:
                river |= (value > 0) & valid
    river = binary_dilation(river, iterations=max(1, int(LSM_HAND_BUFFER_M / res)))
    dst = np.zeros(shape, dtype=np.uint8)
    reproject(river.astype(np.uint8), dst, src_transform=transform, src_crs=crs,
              dst_transform=dem_transform, dst_crs=dem_crs,
              resampling=Resampling.nearest)
    return dst.astype(bool)


def _local_lsm(dem, valid, river_mask, resolution):
    """Script 05's TV/Laplacian/persistence core, evaluated on one window.

    The river mask is intentionally applied after all smoothing and curvature
    measurements.  ``scale_idx`` identifies the strongest persistent scale so
    ray tracing can use the corresponding signed curvature field.
    """
    diagnostics = {
        "reason": "ok",
        "valid_pixels": int(valid.sum()),
        "interior_pixels": 0,
        "persistent_pixels": 0,
        "vals_count": 0,
        "min_vals_required": 0,
        "persistence_k": 0,
    }
    if valid.sum() < 100:
        diagnostics["reason"] = "valid<100"
        return None, diagnostics
    edge = max(3, int(LSM_EDGE_BUFFER_M / resolution))
    interior = binary_erosion(valid, iterations=edge) if (~valid).any() else valid.copy()
    if interior.sum() < 100:
        interior = valid.copy()
    diagnostics["interior_pixels"] = int(interior.sum())
    if diagnostics["interior_pixels"] < 100:
        diagnostics["reason"] = "interior<100"
        return None, diagnostics
    if (~valid).any():
        seed = float(np.nanmedian(dem[valid])) if np.any(valid) else 0.0
        filled = np.where(valid, dem, seed).astype(np.float32, copy=False)
        filled = fillnodata(filled, mask=valid.astype(np.uint8),
                            max_search_distance=max(8.0, float(edge * 2)),
                            smoothing_iterations=0)
        filled[valid] = dem[valid]
    else:
        filled = np.nan_to_num(dem, nan=0.0)

    local_tv_weights = _tv_weights_for_resolution(resolution)
    signed_stack, norm_stack = [], []
    for weight in local_tv_weights:
        smooth = denoise_tv_chambolle(filled, weight=weight, channel_axis=None)
        signed = laplace(smooth).astype(np.float32)
        sigma = np.sqrt(weight) * resolution
        norm = (sigma ** LSM_SCALE_NORM_GAMMA * np.abs(signed)).astype(np.float32)
        signed_stack.append(signed)
        norm_stack.append(norm)
    norms = np.stack(norm_stack)
    signed = np.stack(signed_stack)
    thresholds = np.array([
        np.percentile(layer[interior & (layer > 0)], 90) * 0.1
        if np.any(interior & (layer > 0)) else np.inf for layer in norms])
    above = norms > thresholds[:, None, None]
    persistence = above.sum(axis=0).astype(np.uint8)
    persistence_k = _effective_persistence_k(resolution, len(local_tv_weights))
    diagnostics["persistence_k"] = int(persistence_k)
    persistent = persistence >= persistence_k
    scale_idx = np.argmax(np.where(above, norms, -np.inf), axis=0).astype(np.int16)
    lsm = np.where(persistent, norms.max(axis=0), 0).astype(np.float32)
    lsm[river_mask | ~interior] = 0
    persistent &= ~river_mask & interior
    diagnostics["persistent_pixels"] = int(np.count_nonzero(persistent))
    vals = lsm[persistent & np.isfinite(lsm)]
    diagnostics["vals_count"] = int(vals.size)
    min_vals = _min_vals_required(diagnostics["interior_pixels"])
    diagnostics["min_vals_required"] = int(min_vals)
    if vals.size < min_vals:
        diagnostics["reason"] = f"vals<{min_vals}"
        return None, diagnostics
    low = float(vals.mean() + LSM_HYSTERESIS_LOW_SIGMA * vals.std())
    high = float(vals.mean() + LSM_HYSTERESIS_HIGH_SIGMA * vals.std())
    medium = float((low + high) / 2.0)
    # Recalibrated thresholds: Tier 1 starts at midpoint, Tier 2 starts at low.
    high_mask = lsm >= medium
    medium_mask = lsm >= low
    low_mask = lsm >= low
    signed_at_strongest = np.take_along_axis(signed, scale_idx[None, :, :], axis=0)[0]
    base_sign = signed_at_strongest > 0
    crest_sign = signed_at_strongest < 0
    lab, n = label(high_mask)
    hyst = np.zeros_like(high_mask)
    for ident in range(1, n + 1):
        hyst |= binary_dilation(lab == ident, iterations=3) & low_mask
    high_peaks = hyst & (maximum_filter(lsm, size=3) == lsm) & persistent & high_mask
    high_base_candidates = high_peaks & base_sign
    high_crest_candidates = high_peaks & crest_sign
    medium_pool = persistent & medium_mask & ~high_peaks
    medium_base_candidates = medium_pool & base_sign
    medium_crest_candidates = medium_pool & crest_sign
    candidates = (high_base_candidates | high_crest_candidates |
                  medium_base_candidates | medium_crest_candidates)
    candidate_tier = np.zeros_like(lsm, dtype=np.uint8)
    candidate_tier[high_base_candidates | high_crest_candidates] = 1
    candidate_tier[medium_base_candidates | medium_crest_candidates] = 2
    candidate_sign = np.zeros_like(lsm, dtype=np.int8)
    candidate_sign[high_base_candidates | medium_base_candidates] = 1
    candidate_sign[high_crest_candidates | medium_crest_candidates] = -1
    return {"lsm": lsm, "persistent": persistent, "persistence": persistence,
            "scale_idx": scale_idx, "signed": signed, "filled": filled,
            "candidates": candidates, "candidate_tier": candidate_tier,
            "candidate_sign": candidate_sign,
            "low": low, "medium": medium, "high": high,
            "tv_weights": local_tv_weights, "persistence_k": persistence_k}, diagnostics

def _ray_crossing(signed_curvature, start_row, start_col, dr, dc, background, max_px,
                  grad_magnitude=None, grad_decay=GRAD_DECAY_THRESHOLD):
    """2-D counterpart to Script 05's ``_find_curvature_crossing``."""
    last_curv = float(map_coordinates(signed_curvature, [[start_row], [start_col]], order=1,
                                      mode="nearest")[0])
    last_grad = None
    if grad_magnitude is not None:
        last_grad = float(map_coordinates(grad_magnitude, [[start_row], [start_col]], order=1,
                                          mode="nearest")[0])
    for step in range(1, max_px + 1):
        r, c = start_row + dr * step, start_col + dc * step
        if r < 1 or c < 1 or r >= signed_curvature.shape[0] - 1 or c >= signed_curvature.shape[1] - 1:
            break
        cur_curv = float(map_coordinates(signed_curvature, [[r], [c]], order=1, mode="nearest")[0])
        if abs(cur_curv) <= background or cur_curv * last_curv < 0:
            return r, c
        if grad_magnitude is not None and last_grad is not None and np.isfinite(last_grad):
            cur_grad = float(map_coordinates(grad_magnitude, [[r], [c]], order=1, mode="nearest")[0])
            if np.isfinite(cur_grad) and cur_grad <= last_grad * float(grad_decay):
                return r, c
            if np.isfinite(cur_grad):
                last_grad = cur_grad
        last_curv = cur_curv
    return None


def _ray_crossing_to_sign(curvature, start_row, start_col, dr, dc, target_sign, max_px):
    """Walk along a ray until curvature reaches a target sign or decays to background."""
    start_val = float(map_coordinates(curvature, [[start_row], [start_col]], order=1,
                                      mode="nearest")[0])
    start_mag = abs(start_val)

    for step in range(1, max_px + 1):
        r = start_row + dr * step
        c = start_col + dc * step
        if r < 1 or c < 1 or r >= curvature.shape[0] - 1 or c >= curvature.shape[1] - 1:
            break

        cur = float(map_coordinates(curvature, [[r], [c]], order=1, mode="nearest")[0])
        cur_sign = np.sign(cur)

        if cur_sign == target_sign or cur_sign == 0:
            return r, c
        if start_mag > 0 and abs(cur) < start_mag * 0.1:
            return r, c
    return None


def _find_slope_break(dem_filled, start_row, start_col, dr, dc, resolution, max_px,
                      direction="downhill", debug=False):
    """Walk along gradient direction and find primary/intermediate slope breaks.

    Returns:
        (primary_break, intermediate_break), where each break is (row, col) or None.
    """
    r1, c1 = float(start_row), float(start_col)
    r2, c2 = r1 + dr * 2.0, c1 + dc * 2.0

    if not (0 <= round(r2) < dem_filled.shape[0] and 0 <= round(c2) < dem_filled.shape[1]):
        return None, None

    z1 = float(map_coordinates(dem_filled, [[r1], [c1]], order=1, mode="nearest")[0])
    z2 = float(map_coordinates(dem_filled, [[r2], [c2]], order=1, mode="nearest")[0])
    start_slope = abs(z1 - z2) / (2.0 * float(resolution))

    if not np.isfinite(start_slope) or start_slope < SLOPE_BREAK_MIN_START_SLOPE:
        return None, None

    prev_slope = start_slope
    prev_r, prev_c = r1, c1
    prev_trend = 0.0  # direction of slope change (positive = steepening)
    intermediate_break = None
    primary_break = None

    for step in range(2, int(max_px), 2):
        r = r1 + dr * step
        c = c1 + dc * step

        if r < 2 or c < 2 or r >= dem_filled.shape[0] - 2 or c >= dem_filled.shape[1] - 2:
            break

        r_ahead = r + dr * 2.0
        c_ahead = c + dc * 2.0
        if not (0 <= round(r_ahead) < dem_filled.shape[0] and 0 <= round(c_ahead) < dem_filled.shape[1]):
            break

        z_here = float(map_coordinates(dem_filled, [[r], [c]], order=1, mode="nearest")[0])
        z_ahead = float(map_coordinates(dem_filled, [[r_ahead], [c_ahead]], order=1, mode="nearest")[0])
        local_slope = abs(z_here - z_ahead) / (2.0 * float(resolution))

        if debug:
            ratio = local_slope / start_slope if start_slope > 0 else float("nan")
            print(f"    [slope] step={step} start={start_slope:.4f} local={local_slope:.4f} ratio={ratio:.2f}")

        # Intermediate break: steep -> gentle transition (relaxed: 0.6 -> 0.7).
        if intermediate_break is None and local_slope < start_slope * SLOPE_BREAK_INTERMEDIATE_RATIO:
            r_next = r + dr * 2.0
            c_next = c + dc * 2.0
            if 0 <= round(r_next) < dem_filled.shape[0] and 0 <= round(c_next) < dem_filled.shape[1]:
                z_next = float(map_coordinates(dem_filled, [[r_next], [c_next]], order=1, mode="nearest")[0])
                next_slope = abs(z_ahead - z_next) / (2.0 * float(resolution))
                if next_slope < start_slope * SLOPE_BREAK_INTERMEDIATE_RATIO:
                    intermediate_break = (r, c)

        # Primary break: gentle -> flat transition (relaxed: 0.3 -> 0.4).
        if local_slope < start_slope * SLOPE_BREAK_PRIMARY_RATIO:
            r_next = r + dr * 2.0
            c_next = c + dc * 2.0
            if 0 <= round(r_next) < dem_filled.shape[0] and 0 <= round(c_next) < dem_filled.shape[1]:
                z_next = float(map_coordinates(dem_filled, [[r_next], [c_next]], order=1, mode="nearest")[0])
                next_slope = abs(z_ahead - z_next) / (2.0 * float(resolution))
                if next_slope < start_slope * SLOPE_BREAK_PRIMARY_RATIO:
                    primary_break = (r, c)
                    break

        # If slope rises or drops sharply vs prev step, assume we passed the break.
        # Rise: walked past the scarp into steeper terrain behind it (was 1.5 -> 2.0).
        if step > 4 and local_slope > prev_slope * SLOPE_BREAK_RISE_FACTOR:
            primary_break = (prev_r, prev_c)
            break
        # Drop: walked past the scarp into much flatter terrain than prev (new).
        if step > 4 and prev_slope > 1e-6 and local_slope <= prev_slope * SLOPE_BREAK_DROP_FACTOR:
            primary_break = (prev_r, prev_c)
            break

        # Primary break relative to previous slope (not just start slope).
        # This catches transitions where start_slope was similar to the
        # post-break slope (e.g. 1% -> 0.5%, start=1%, ratio=0.5 > 0.4
        # primary threshold, but 0.5 < 0.7 intermediate and clearly gentler).
        if step > 6 and prev_slope > SLOPE_BREAK_MIN_START_SLOPE and \
           local_slope < prev_slope * SLOPE_BREAK_INTERMEDIATE_RATIO and \
           local_slope < start_slope * SLOPE_BREAK_INTERMEDIATE_RATIO:
            primary_break = (prev_r, prev_c)
            break

        # Inflection-based break: detect slope-direction reversal.
        # If the slope was increasing (gentle -> steeper) and is now decreasing
        # (steeper -> gentler), or vice versa, we've passed through a convex/
        # concave transition that marks a scarp boundary.  This catches gentle
        # scarps where the absolute slope ratio thresholds above don't fire
        # because all slopes are small (e.g. 0.5% -> 1% -> 0.5%).
        if step > 6 and prev_slope > SLOPE_BREAK_MIN_START_SLOPE:
            slope_delta_prev = local_slope - prev_slope   # current trend
            # Require both trend and current delta to exceed noise threshold.
            # Without this, DEM noise (~1e-4) causes false inflections on
            # near-uniform slopes.
            if (abs(prev_trend) > SLOPE_BREAK_INFLECTION_MIN_DELTA and
                    abs(slope_delta_prev) > SLOPE_BREAK_INFLECTION_MIN_DELTA):
                if prev_trend > 0 and slope_delta_prev < 0:
                    # Was steepening, now flattening -> convex crest break
                    primary_break = (prev_r, prev_c)
                    break
                if prev_trend < 0 and slope_delta_prev > 0:
                    # Was flattening, now steepening -> concave base break
                    primary_break = (prev_r, prev_c)
                    break
            prev_trend = slope_delta_prev

        # Slope-change ratio break: |local - prev| / start > threshold
        # catches transitions between slope domains even when absolute
        # slopes are all gentle (e.g. 3% -> 1% is a 67% change relative
        # to start_slope of 3%).
        if (step > 4 and start_slope > SLOPE_BREAK_MIN_START_SLOPE and
                abs(local_slope - prev_slope) / start_slope > SLOPE_BREAK_INFLECTION_RATIO):
            primary_break = (prev_r, prev_c)
            break

        # Update trend for inflection detection in next iteration
        if step > 4 and prev_slope > 1e-9:
            prev_trend = local_slope - prev_slope
        prev_slope = local_slope
        prev_r, prev_c = r, c

    if primary_break is None:
        primary_break = (prev_r, prev_c)

    return primary_break, intermediate_break



def _classify_scarp_pixels(state, dem, river_mask, resolution, transform,
                           edge_margin_px=None, collect_diagnostics=False):
    """Trace from curvature-signed candidates to the opposite slope break.

    When *collect_diagnostics* is True a tuple ``(records_df, diagnostics_df)``
    is returned.  The diagnostics DataFrame contains one row per candidate
    pixel with its rejection reason (or ``"success"``), height, width,
    asymmetry, and ray-trace outcome.  This lets you tune thresholds with
    data instead of guesswork.

    When ray tracing fails to find a base or crest, a **soft fallback**
    path is attempted: the candidate is kept as a lower-confidence point
    (``classify_method = "curvature_fallback"``) if it has strong curvature
    signal and a minimum elevation drop along the gradient direction.
    """
    if edge_margin_px is None:
        edge_margin_px = max(5, int(BASE_RAY_LENGTH_M / resolution / 2.0))

    rows, cols = np.where(state["candidates"])
    if not len(rows):
        if collect_diagnostics:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()

    # Filter edge candidates only when enough interior remains for ray tracing.
    h, w = dem.shape
    min_interior = 10
    if h < 2 * edge_margin_px + min_interior or w < 2 * edge_margin_px + min_interior:
        print(f"  [classify] WARNING: window {w}x{h} too small for {edge_margin_px}px margin, skipping edge filter")
    else:
        edge_mask = ((rows >= edge_margin_px) & (rows < h - edge_margin_px) &
                     (cols >= edge_margin_px) & (cols < w - edge_margin_px))
        filtered = int((~edge_mask).sum())
        rows = rows[edge_mask]
        cols = cols[edge_mask]
        if filtered:
            print(f"  [classify] filtered {filtered} edge candidates, {len(rows)} remain")

    if not len(rows):
        print(f"  [classify] all candidates removed before ray tracing")
        if collect_diagnostics:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()

    grad_row, grad_col = np.gradient(state["filled"], resolution, resolution)
    grad_mag = np.hypot(grad_row, grad_col)

    # Diagnostic counters to expose where candidate scarp pixels are filtered.
    stats = {"total_candidates": len(rows), "low_gradient": 0,
             "no_crest": 0, "no_base": 0, "out_of_bounds": 0, "river_touch": 0,
             "river_masked_candidate": 0,
             "asymmetry_low": 0, "width_fail": 0, "ratio_fail": 0,
             "height_fail": 0, "ambiguous_sign": 0, "success": 0,
             "fallback_accepted": 0, "fallback_rejected": 0,
             "tier_1": 0, "tier_2": 0, "tier_3": 0}
    ray_debug_failures = 0
    asym_debug_count = 0
    candidate_tier = state.get("candidate_tier")
    candidate_sign = state.get("candidate_sign")

    records = []
    diag_records = []  # A1: per-candidate diagnostics
    for row, col in zip(rows, cols):
        # Common fields for diagnostics
        x_d, y_d = transform * (col, row)
        diag = {"row": int(row), "col": int(col), "x": float(x_d), "y": float(y_d),
                "tier": int(candidate_tier[row, col]) if candidate_tier is not None else 0,
                "local_slope": None, "grad_mag": None, "curvature_sign": None,
                "ray_max_px": None, "ray_length_m": None,
                "base_found": None, "crest_found": None,
                "height_m": None, "width_m": None, "asymmetry": None,
                "touches_river": None, "rejection_reason": None,
                "classify_method": None}

        # Hard filter from HAND/river influence before any ray tracing.
        if river_mask[row, col]:
            stats["river_masked_candidate"] += 1
            diag["rejection_reason"] = "river_masked"
            if collect_diagnostics:
                diag_records.append(diag)
            continue

        g_r, g_c = grad_row[row, col], grad_col[row, col]
        magnitude = float(np.hypot(g_r, g_c))
        diag["grad_mag"] = magnitude
        if not np.isfinite(magnitude) or magnitude < 1e-6:
            stats["low_gradient"] += 1
            diag["rejection_reason"] = "low_gradient"
            if collect_diagnostics:
                diag_records.append(diag)
            continue

        local_slope = float(abs(grad_mag[row, col]))
        diag["local_slope"] = local_slope
        if local_slope < 0.1:
            ray_length_m = float(BASE_RAY_LENGTH_M) * 1.5
            max_px = max(10, int(np.ceil(ray_length_m / resolution)))
        elif local_slope < 0.2:
            ray_length_m = float(BASE_RAY_LENGTH_M)
            max_px = max(5, int(np.ceil(ray_length_m / resolution)))
        else:
            ray_length_m = float(BASE_RAY_LENGTH_M) * 0.5
            max_px = max(3, int(np.ceil(ray_length_m / resolution)))
        diag["ray_max_px"] = max_px
        diag["ray_length_m"] = ray_length_m

        tier = int(candidate_tier[row, col]) if candidate_tier is not None else 0
        if tier == 2:
            stats["tier_2"] += 1
        elif tier == 1:
            stats["tier_1"] += 1
        elif tier == 3:
            stats["tier_3"] += 1

        sign = int(candidate_sign[row, col]) if candidate_sign is not None else 0
        if sign == 0:
            si = int(state["scale_idx"][row, col])
            sign = int(np.sign(state["signed"][si, row, col]))
        if sign == 0:
            stats["ambiguous_sign"] += 1
            diag["rejection_reason"] = "ambiguous_sign"
            if collect_diagnostics:
                diag_records.append(diag)
            continue
        diag["curvature_sign"] = sign

        # Walk direction is controlled by curvature sign at strongest scale.
        dr_up, dc_up = g_r / magnitude, g_c / magnitude
        dr_down, dc_down = -g_r / magnitude, -g_c / magnitude
        if sign > 0:
            base_primary, base_intermediate = _find_slope_break(
                state["filled"], row, col, dr_down, dc_down, resolution, max_px, direction="downhill"
            )
            crest_primary, crest_intermediate = _find_slope_break(
                state["filled"], row, col, dr_up, dc_up, resolution, max_px, direction="uphill"
            )
        else:
            crest_primary, crest_intermediate = _find_slope_break(
                state["filled"], row, col, dr_up, dc_up, resolution, max_px, direction="uphill"
            )
            base_primary, base_intermediate = _find_slope_break(
                state["filled"], row, col, dr_down, dc_down, resolution, max_px, direction="downhill"
            )

        base = base_primary
        crest = crest_primary

        diag["base_found"] = base is not None
        diag["crest_found"] = crest is not None

        if crest is None:
            stats["no_crest"] += 1
        if base is None:
            stats["no_base"] += 1
        if base is None or crest is None:
            if ray_debug_failures < RAY_DEBUG_MAX_FAILURES:
                ray_debug_failures += 1
                print(f"    [slope debug] pixel ({row},{col}) grad_mag={magnitude:.4f} max_px={max_px} "
                      f"base_found={base is not None} crest_found={crest is not None}")

            # B1: Soft classify fallback.
            # When ray tracing fails, check if the candidate has strong
            # curvature signal (LSM above medium threshold) and a minimum
            # elevation drop along the gradient direction.  If so, record
            # it as a lower-confidence inferred point.
            lsm_val = float(state["lsm"][row, col])
            medium_thresh = float(state.get("medium", 0.0))
            n_scales = int(state["persistence"][row, col])

            # Walk a short distance along uphill and downhill to estimate
            # a rough height even without formal slope breaks.  Try 50m first,
            # and if the height is too small, extend to 150m to catch gentle
            # scarps with low relief over a wider span.
            z_start = float(state["filled"][row, col])

            def _fb_walk(walk_m):
                wp = max(5, int(np.ceil(walk_m / resolution)))
                z_u = z_start
                z_d = z_start
                for s in range(1, min(wp, max_px) + 1):
                    ru = row + dr_up * s
                    cu = col + dc_up * s
                    rd_ = row + dr_down * s
                    cd_ = col + dc_down * s
                    if 0 <= round(ru) < dem.shape[0] and 0 <= round(cu) < dem.shape[1]:
                        z_u = float(map_coordinates(
                            state["filled"], [[ru], [cu]], order=1, mode="nearest")[0])
                    if 0 <= round(rd_) < dem.shape[0] and 0 <= round(cd_) < dem.shape[1]:
                        z_d = float(map_coordinates(
                            state["filled"], [[rd_], [cd_]], order=1, mode="nearest")[0])
                    if not (0 <= round(ru) < dem.shape[0]) and not (0 <= round(rd_) < dem.shape[0]):
                        break
                return float(abs(z_u - z_d)), float(wp * 2 * resolution)

            fb_height, fb_width = _fb_walk(FALLBACK_WALK_M)
            # If initial walk gives too little height, try extended walk
            if fb_height < FALLBACK_MIN_HEIGHT_M:
                fb_height_ext, fb_width_ext = _fb_walk(FALLBACK_WALK_ADAPTIVE_M)
                if fb_height_ext > fb_height:
                    fb_height = fb_height_ext
                    fb_width = fb_width_ext
            diag["height_m"] = fb_height
            diag["width_m"] = fb_width

            # Fallback acceptance criteria
            fb_lsm_ok = lsm_val >= medium_thresh * FALLBACK_MIN_LSM_RATIO
            fb_height_ok = np.isfinite(fb_height) and fb_height >= FALLBACK_MIN_HEIGHT_M
            fb_river = bool(river_mask[row, col])

            if fb_lsm_ok and fb_height_ok and not fb_river:
                # Record as fallback point
                x, y = transform * (col, row)
                records.append({"base_x": x, "base_y": y, "crest_x": x,
                                "crest_y": y, "height_m": fb_height,
                                "width_m": fb_width,
                                "height_width_ratio": float(fb_height / fb_width) if fb_width > 1e-9 else np.inf,
                                "asymmetry": 0.0,
                                "asymmetry_score": 0.0,
                                "persistence_scales": n_scales,
                                "lsm": lsm_val,
                                "candidate_tier": tier,
                                "curvature_sign": sign,
                                "tier": tier,
                                "break_type": "p",
                                "local_slope": local_slope,
                                "min_height_m": FALLBACK_MIN_HEIGHT_M,
                                "touches_river": fb_river,
                                "ray_length_m": ray_length_m,
                                "ray_max_px": max_px,
                                "classify_method": "curvature_fallback"})
                stats["success"] += 1
                stats["fallback_accepted"] += 1
                diag["rejection_reason"] = "fallback_accepted"
                diag["classify_method"] = "curvature_fallback"
            else:
                reason = "no_crest" if crest is None else "no_base"
                stats["fallback_rejected"] += 1
                diag["rejection_reason"] = reason + "_fallback_rejected"
                diag["classify_method"] = "none"

            if collect_diagnostics:
                diag_records.append(diag)
            continue

        br, bc = base
        cr, cc = crest
        if not (0 <= round(br) < dem.shape[0] and 0 <= round(bc) < dem.shape[1] and
                0 <= round(cr) < dem.shape[0] and 0 <= round(cc) < dem.shape[1]):
            stats["out_of_bounds"] += 1
            diag["rejection_reason"] = "out_of_bounds"
            diag["classify_method"] = "none"
            if collect_diagnostics:
                diag_records.append(diag)
            continue

        touches_river = bool(river_mask[int(round(br)), int(round(bc))] or
                             river_mask[int(round(cr)), int(round(cc))])
        if touches_river:
            stats["river_touch"] += 1
        diag["touches_river"] = touches_river

        asymmetry_debug = stats["total_candidates"] < 10 and asym_debug_count < 3
        asymmetry = _check_scarp_asymmetry(
            state["filled"], br, bc, cr, cc, dr_down, dc_down, dr_up, dc_up, resolution,
            debug=asymmetry_debug,
        )
        if asymmetry_debug:
            asym_debug_count += 1
        if asymmetry < MIN_SCARP_ASYMMETRY_SCORE:
            stats["asymmetry_low"] += 1
        diag["asymmetry"] = asymmetry

        z_base, z_crest = map_coordinates(dem, [[br, cr], [bc, cc]], order=1, mode="nearest")
        height = float(abs(z_crest - z_base))
        width = float(np.hypot(br - cr, bc - cc) * resolution)
        height_width_ratio = float(height / width) if width > 1e-9 else np.inf
        diag["height_m"] = height
        diag["width_m"] = width

        if not np.isfinite(height) or height < MIN_SCARP_HEIGHT_M:
            stats["height_fail"] += 1
            diag["rejection_reason"] = "height_fail"
            diag["classify_method"] = "none"
            if collect_diagnostics:
                diag_records.append(diag)
            continue

        if width <= 0 or width > SCARP_WIDTH_DISCRETE_MAX_M:
            stats["width_fail"] += 1
            diag["rejection_reason"] = "width_fail"
            diag["classify_method"] = "none"
            if collect_diagnostics:
                diag_records.append(diag)
            continue

        x, y = transform * (bc, br)
        cx, cy = transform * (cc, cr)
        records.append({"base_x": x, "base_y": y, "crest_x": cx,
                        "crest_y": cy, "height_m": height,
                        "width_m": width, "height_width_ratio": height_width_ratio,
                        "asymmetry": asymmetry,
                        "asymmetry_score": asymmetry,
                        "persistence_scales": int(state["persistence"][row, col]),
                        "lsm": float(state["lsm"][row, col]),
                        "candidate_tier": tier,
                        "curvature_sign": sign,
                        "tier": tier,
                        "break_type": "p",
                        "local_slope": local_slope,
                        "min_height_m": MIN_SCARP_HEIGHT_M,
                        "touches_river": touches_river,
                        "ray_length_m": ray_length_m,
                        "ray_max_px": max_px,
                        "classify_method": "ray_traced"})
        stats["success"] += 1
        diag["rejection_reason"] = "success"
        diag["classify_method"] = "ray_traced"
        if collect_diagnostics:
            diag_records.append(diag)

        if base_intermediate is not None:
            br_inter, bc_inter = base_intermediate
            if (0 <= round(br_inter) < dem.shape[0] and 0 <= round(bc_inter) < dem.shape[1] and
                    0 <= round(cr) < dem.shape[0] and 0 <= round(cc) < dem.shape[1]):
                z_base_inter = float(state["filled"][int(round(br_inter)), int(round(bc_inter))])
                z_crest_primary = float(state["filled"][int(round(cr)), int(round(cc))])
                height_inter = float(abs(z_crest_primary - z_base_inter))
                width_inter = float(np.hypot(br_inter - cr, bc_inter - cc) * resolution)
                if width_inter > 0:
                    height_width_ratio_inter = float(height_inter / width_inter)
                else:
                    height_width_ratio_inter = np.inf
                if (np.isfinite(height_inter) and np.isfinite(width_inter) and
                        MIN_SCARP_HEIGHT_M <= height_inter and
                        0 < width_inter <= SCARP_WIDTH_DISCRETE_MAX_M):
                    x_inter, y_inter = transform * (bc_inter, br_inter)
                    records.append({"base_x": x_inter, "base_y": y_inter, "crest_x": cx,
                                    "crest_y": cy, "height_m": height_inter,
                                    "width_m": width_inter,
                                    "height_width_ratio": height_width_ratio_inter,
                                    "asymmetry": asymmetry,
                                    "asymmetry_score": asymmetry,
                                    "persistence_scales": int(state["persistence"][row, col]),
                                    "lsm": float(state["lsm"][row, col]),
                                    "candidate_tier": tier,
                                    "curvature_sign": sign,
                                    "tier": tier,
                                    "break_type": "i",
                                    "local_slope": local_slope,
                                    "min_height_m": MIN_SCARP_HEIGHT_M,
                                    "touches_river": touches_river,
                                    "ray_length_m": ray_length_m,
                                    "ray_max_px": max_px,
                    "classify_method": "ray_traced"})
                    stats["success"] += 1
                    # A1: Record a diagnostic entry for the intermediate break.
                    # This is a *separate* inferred point from the primary,
                    # so it needs its own diag row to keep diagnostics and
                    # inferred points in 1:1 correspondence.
                    if collect_diagnostics:
                        diag_inter = {
                            "row": int(round(br_inter)),
                            "col": int(round(bc_inter)),
                            "x": float(x_inter),
                            "y": float(y_inter),
                            "tier": tier,
                            "local_slope": local_slope,
                            "grad_mag": magnitude,
                            "curvature_sign": sign,
                            "ray_max_px": max_px,
                            "ray_length_m": ray_length_m,
                            "base_found": True,
                            "crest_found": True,
                            "height_m": height_inter,
                            "width_m": width_inter,
                            "asymmetry": asymmetry,
                            "touches_river": touches_river,
                            "rejection_reason": "success_intermediate",
                            "classify_method": "ray_traced",
                        }
                        diag_records.append(diag_inter)

    print(f"  [classify] candidates={stats['total_candidates']} -> "
            f"river_masked={stats['river_masked_candidate']} "
            f"low_grad={stats['low_gradient']} no_crest={stats['no_crest']} "
            f"no_base={stats['no_base']} "
            f"oob={stats['out_of_bounds']} "
            f"river_touch={stats['river_touch']} asym_low={stats['asymmetry_low']} "
                f"ambiguous_sign={stats['ambiguous_sign']} "
                f"h_fail={stats['height_fail']} w_fail={stats['width_fail']} "
                f"ratio={stats['ratio_fail']} -> success={stats['success']} "
                f"fallback_accept={stats['fallback_accepted']} "
                f"fallback_reject={stats['fallback_rejected']} "
                f"tiers={{1:{stats['tier_1']}, 2:{stats['tier_2']}, 3:{stats['tier_3']}}}")
    if collect_diagnostics:
        return pd.DataFrame(records), pd.DataFrame(diag_records)
    return pd.DataFrame(records)


def _periodicity(coords):
    if len(coords) < 5:
        return 0.0
    d = np.diff(coords, axis=0)
    angle = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.diff(angle)
    if len(turn) < 3 or np.dot(turn - turn.mean(), turn - turn.mean()) < 1e-12:
        return 0.0
    centred = turn - turn.mean()
    return float(np.clip(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred), 0, 1))


def _detect_meander_pattern(coords):
    """Detect oscillatory meander-like geometry in ordered coordinates."""
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 10:
        return False

    d = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(d, axis=1)
    if not np.all(np.isfinite(seg_lengths)) or np.sum(seg_lengths) <= 1e-6:
        return False

    start_end_vec = xy[-1] - xy[0]
    start_end_dist = float(np.linalg.norm(start_end_vec))
    if start_end_dist < 1.0:
        return False

    start_end_dir = start_end_vec / start_end_dist
    perp_dir = np.array([-start_end_dir[1], start_end_dir[0]], dtype=float)

    rel = xy - xy[0]
    deviations = rel @ perp_dir
    deviations = deviations.astype(float)

    if not np.all(np.isfinite(deviations)):
        return False

    zero_crossings = int(np.sum(deviations[:-1] * deviations[1:] < 0))

    centered = deviations - float(np.mean(deviations))
    if len(centered) > 10 and float(np.dot(centered, centered)) > 1e-10:
        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        neg_indices = np.where(autocorr[1:] < 0)[0]
        if len(neg_indices) > 0:
            quarter_wave = int(neg_indices[0] + 1)
            if quarter_wave < len(autocorr) - 2 and autocorr[quarter_wave + 1] > -0.3:
                return True

    return zero_crossings >= 2


def _connect_nearby_clusters(clusters_data, max_gap_m=CLUSTER_CONNECT_MAX_GAP_M,
                             max_angle_deg=CLUSTER_CONNECT_MAX_ANGLE_DEG):
    """Connect DBSCAN clusters that are nearby and approximately collinear."""
    if not clusters_data:
        return []

    items = sorted(clusters_data, key=lambda c: c.get("mean_trace_idx", 0.0))
    used = set()
    connected = []
    max_angle_rad = np.deg2rad(max_angle_deg)

    def _best_neighbor(anchor_idx, direction_sign):
        anchor = items[anchor_idx]
        best_idx = None
        best_dist = float(max_gap_m)
        for j, cand in enumerate(items):
            if j in used or j == anchor_idx:
                continue
            dist = float(np.min(cdist(anchor["points"], cand["points"])))
            if dist > max_gap_m:
                continue

            dot = float(np.clip(np.abs(np.dot(anchor["direction"], cand["direction"])), -1.0, 1.0))
            angle = float(np.arccos(dot))
            if angle > max_angle_rad:
                continue

            vec = cand["center"] - anchor["center"]
            proj = float(np.dot(vec, anchor["direction"]))
            if direction_sign > 0 and proj < 0:
                continue
            if direction_sign < 0 and proj > 0:
                continue

            if dist < best_dist:
                best_dist = dist
                best_idx = j
        return best_idx

    for i in range(len(items)):
        if i in used:
            continue

        used.add(i)
        chain = [i]

        extended = True
        while extended:
            extended = False
            nxt = _best_neighbor(chain[-1], direction_sign=1)
            if nxt is not None:
                chain.append(nxt)
                used.add(nxt)
                extended = True

        extended = True
        while extended:
            extended = False
            prev_idx = _best_neighbor(chain[0], direction_sign=-1)
            if prev_idx is not None:
                chain.insert(0, prev_idx)
                used.add(prev_idx)
                extended = True

        connected.append({
            "points": np.vstack([items[idx]["points"] for idx in chain]),
            "cluster_count": len(chain),
            "member_labels": [int(items[idx].get("label", -1)) for idx in chain],
        })

    return connected


def _segments_from_points(points, trace, resolution, river_mask, transform,
                          dem, persistence_k=LSM_PERSISTENCE_K):
    """Chain scarp bases into one fault-like segment via two-stage clustering."""
    if len(points) < 10:
        return []

    coords = np.c_[points.base_x.values, points.base_y.values]

    # Stage 1: tight clustering finds compact scarp nuclei.
    eps_tight = float(resolution) * 2.0
    clustering = DBSCAN(eps=eps_tight, min_samples=3).fit(coords)
    unique_labels = np.unique(clustering.labels_)
    centers = [coords[clustering.labels_ == label_id].mean(axis=0)
               for label_id in unique_labels if label_id >= 0]
    selected_pass = "tight"
    selected_eps = eps_tight

    # Stage 2 fallback: loosen eps when tight nuclei are too sparse.
    if len(centers) < 3:
        eps_loose = float(resolution) * 10.0
        clustering = DBSCAN(eps=eps_loose, min_samples=5).fit(coords)
        unique_labels = np.unique(clustering.labels_)
        centers = [coords[clustering.labels_ == label_id].mean(axis=0)
                   for label_id in unique_labels if label_id >= 0]
        selected_pass = "loose"
        selected_eps = eps_loose

    n_clusters = len([label_id for label_id in unique_labels if label_id >= 0])
    n_noise = int(np.sum(clustering.labels_ == -1))
    cluster_sizes = [int(np.sum(clustering.labels_ == label_id))
                     for label_id in unique_labels if label_id >= 0]
    print(f"  [segments] DBSCAN ({selected_pass}, eps={selected_eps:.1f} m): "
          f"{n_clusters} clusters, sizes={sorted(cluster_sizes)}, noise={n_noise}")

    if len(centers) < 3:
        print(f"  [segments] only {len(centers)} cluster centers, insufficient for segment")
        return []

    trace_tree = cKDTree(trace["coords"])
    source_points_tree = cKDTree(coords)

    def _build_segment_record(ordered, method_name):
        if len(ordered) < 2:
            return None
        path = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
        straight = float(np.linalg.norm(ordered[-1] - ordered[0]))
        sinuosity = path / straight if straight > 1e-6 else np.inf
        periodicity = _periodicity(ordered)
        if path < LSM_MIN_LENGTH_M:
            return None

        _, nearest = source_points_tree.query(ordered)
        evidence = points.iloc[nearest]
        confidence, conf_diag = _compute_segment_confidence(
            evidence, ordered, trace, river_mask, transform, resolution
        )
        if confidence < 3.0:
            return None

        mean_height_width_ratio = float(evidence.height_width_ratio.mean())
        mean_tier = float(conf_diag["mean_tier"])
        mean_asymmetry = float(conf_diag["mean_asymmetry"])

        return {
            "known_trace_id": trace["trace_id"],
            "source": trace["source"],
            "coords": ordered,
            "length_m": path,
            "centroid_x": float(ordered[:, 0].mean()),
            "centroid_y": float(ordered[:, 1].mean()),
            "mean_height_m": float(evidence.height_m.mean()),
            "mean_width_m": float(evidence.width_m.mean()),
            "mean_height_width_ratio": mean_height_width_ratio,
            "mean_persistence_scales": float(evidence.persistence_scales.mean()),
            "mean_candidate_tier": mean_tier,
            "mean_asymmetry": mean_asymmetry,
            "sinuosity": sinuosity,
            "local_sinuosity": _local_sinuosity(ordered, window_size=5),
            "turning_periodicity": periodicity,
            "median_distance_to_known_m": float(conf_diag["dist_to_trace_m"]),
            "river_clearance_m": float(conf_diag["river_clearance_m"]),
            "profile_validation_score": np.nan,
            "profile_validation_passed": np.nan,
            "confidence_score_0_10": round(confidence, 2),
            "segment_method": method_name,
            "profile_samples": [],
        }

    rejection_reasons = {
        "too_few_points": 0,
        "too_short": 0,
        "pca_failed": 0,
        "sinuosity_high": 0,
        "periodicity_high": 0,
        "meander_pattern": 0,
        "connected_meander_pattern": 0,
        "success": 0,
    }

    accepted = []
    for label_id in unique_labels:
        if label_id < 0:
            continue
        cluster_mask = clustering.labels_ == label_id
        n_points = int(cluster_mask.sum())
        if n_points < 5:
            rejection_reasons["too_few_points"] += 1
            continue

        cluster_xy = coords[cluster_mask]
        centered = cluster_xy - cluster_xy.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            principal_dir = vh[0]
            if not np.all(np.isfinite(principal_dir)):
                raise np.linalg.LinAlgError("non-finite principal direction")
            proj = centered @ principal_dir
            ordered = cluster_xy[np.argsort(proj)]
        except np.linalg.LinAlgError:
            rejection_reasons["pca_failed"] += 1
            continue

        trace_dist, trace_idx = trace_tree.query(ordered)

        path = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
        straight = float(np.linalg.norm(ordered[-1] - ordered[0]))
        sinuosity = path / straight if straight > 1e-6 else np.inf
        periodicity = _periodicity(ordered)
        is_meander = _detect_meander_pattern(ordered)

        if path < LSM_MIN_LENGTH_M:
            rejection_reasons["too_short"] += 1
            continue
        if sinuosity > FALLBACK_MAX_SINUOSITY:
            rejection_reasons["sinuosity_high"] += 1
            continue
        if periodicity > FALLBACK_MAX_PERIODICITY:
            rejection_reasons["periodicity_high"] += 1
            continue
        if is_meander:
            rejection_reasons["meander_pattern"] += 1
            continue
        rejection_reasons["success"] += 1
        accepted.append({
            "label": int(label_id),
            "coords": ordered,
            "points": ordered,
            "center": ordered.mean(axis=0),
            "direction": principal_dir,
            "mean_trace_idx": float(np.mean(trace_idx)),
            "median_trace_dist": float(np.median(trace_dist)),
            "is_meander": bool(is_meander),
        })

    print(f"  [segments] rejection reasons: {rejection_reasons}")

    if not accepted:
        print("  [segments] no valid clusters after per-cluster validation")
        return []

    connected_groups = _connect_nearby_clusters(accepted)
    segment_records = []
    for group_idx, group in enumerate(connected_groups):
        pts = np.asarray(group["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue

        # Order chained points by along-trace position before segment checks.
        _, trace_idx = trace_tree.query(pts)
        all_ordered = pts[np.argsort(trace_idx)]
        if len(all_ordered) < 2:
            continue

        if _detect_meander_pattern(all_ordered):
            rejection_reasons["connected_meander_pattern"] += 1
            continue

        record = _build_segment_record(all_ordered, f"dbscan_two_stage_{selected_pass}_connected")
        if record is None:
            continue
        record["segment_id"] = int(group_idx)
        record["n_clusters_chained"] = int(group["cluster_count"])
        segment_records.append(record)

    if not segment_records:
        print("  [segments] no connected cluster groups passed chain validation")
        return []

    print(f"  [segments] {len(points)} points -> {len(segment_records)} connected segment(s) "
          f"(clusters_used={len(accepted)}, eps={selected_eps:.1f} m, noise_points={n_noise})")
    return segment_records


def _write_local_raster(path, state, transform, crs):
    profile = {"driver": "GTiff", "height": state["lsm"].shape[0], "width": state["lsm"].shape[1],
               "count": 1, "dtype": "float32", "crs": crs, "transform": transform, "nodata": LSM_NODATA}
    data = np.where(state["persistent"], state["lsm"], 0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _filter_river_segments(segments, hand_mask_path, hand_threshold_m=HAND_INFLUENCE_THRESHOLD_M,
                           max_fraction=HAND_SEGMENT_MAX_FRACTION):
    """Remove segments that are mostly inside low-HAND terrain."""
    if not segments:
        return []

    hand_path = Path(hand_mask_path)
    if not hand_path.exists():
        print(f"[River Filter] HAND mask not found, skipping: {hand_path}")
        return list(segments)

    with rasterio.open(hand_path) as src:
        hand = src.read(1).astype(float)
        valid = np.isfinite(hand)
        if src.nodata is not None:
            valid &= hand != src.nodata
        hand_influence = (hand <= float(hand_threshold_m)) & valid
        inv_transform = ~src.transform

        kept = []
        for seg in segments:
            coords = np.asarray(seg.get("coords", []), dtype=float)
            if coords.ndim != 2 or coords.shape[0] == 0:
                continue
            cols, rows = inv_transform * (coords[:, 0], coords[:, 1])
            rr = np.clip(np.rint(rows).astype(int), 0, hand.shape[0] - 1)
            cc = np.clip(np.rint(cols).astype(int), 0, hand.shape[1] - 1)
            frac_in_hand = float(np.mean(hand_influence[rr, cc])) if len(rr) else 1.0
            seg["hand_influence_fraction"] = frac_in_hand
            if frac_in_hand < float(max_fraction):
                kept.append(seg)
            else:
                print(f"  [river filter] removed segment {seg.get('segment_id', 'na')}: {frac_in_hand:.1%} in HAND zone")

    return kept


def _summary_and_comparison(traces, segments, out_dir, dem_source):
    flat = []
    for seg in segments:
        row = {k: v for k, v in seg.items() if k != "coords"}
        row["coords_wkt"] = "LINESTRING (" + ", ".join(f"{x:.2f} {y:.2f}" for x, y in seg["coords"]) + ")"
        flat.append(row)
    seg_df = pd.DataFrame(flat)
    seg_df.to_csv(out_dir / "known_trace_scarp_segments.csv", index=False)
    coverage_rows = []
    for trace in traces:
        related = [s for s in segments if s["known_trace_id"] == trace["trace_id"]]
        if related:
            support = np.vstack([s["coords"] for s in related])
            dist = cKDTree(support).query(trace["coords"])[0]
            corroborated = float(np.mean(dist <= 25.0))
        else:
            corroborated = 0.0
        quality_class = trace.get("prior_quality_class", "unclassified")
        if related:
            if quality_class in POSITIVE_PRIOR_CLASSES:
                evidence_delta = "confirmed" if corroborated >= 0.25 else "refined"
            elif quality_class == "Class 3":
                evidence_delta = "discovered"
            else:
                evidence_delta = "detected"
        else:
            evidence_delta = "unconfirmed" if quality_class in POSITIVE_PRIOR_CLASSES else "no_new_signal"
        coverage_rows.append({"known_trace_id": trace["trace_id"], "source": trace["source"],
                              "known_trace_length_m": trace["length_m"], "n_scarp_segments": len(related),
                              "corroborated_length_fraction": corroborated,
                              "corroborated_length_pct": 100 * corroborated,
                              "prior_quality_class": quality_class,
                              "prior_detection_rate": trace.get("prior_detection_rate", np.nan),
                              "prior_n_profiles": trace.get("prior_n_profiles", 0),
                              "prior_seed_count": trace.get("prior_seed_count", 0),
                              "evidence_delta": evidence_delta})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "known_trace_scarp_summary.csv", index=False)
    db_summary = coverage.groupby("source", as_index=False).agg(
        n_known_traces=("known_trace_id", "size"),
        traces_with_scarps=("n_scarp_segments", lambda s: int((s > 0).sum())),
        mean_corroborated_length_pct=("corroborated_length_pct", "mean"))
    db_summary.to_csv(out_dir / "known_trace_scarp_summary_by_database.csv", index=False)

    # Join only on Script 05's stable reconstructed trace ID.  Absence of the
    # ML outputs remains explicit rather than making this independent run fail.
    ml_dir = OUT_DIR / f"ml_trace_reconstruction_{dem_source}" / "validation"
    candidates = [ml_dir / "fault_trace_validation.csv",
                  ml_dir / "fault_trace_overlay" / "fault_trace_validation.csv"]
    ml_path = next((p for p in candidates if p.exists()), None)
    if ml_path:
        ml = pd.read_csv(ml_path)
        common = [c for c in ["trace_id", "known_trace_id"] if c in ml.columns]
        if common:
            key = common[0]
            ml = ml.rename(columns={key: "known_trace_id"})
            coverage.merge(ml, on="known_trace_id", how="outer", suffixes=("_classical", "_ml")).to_csv(
                out_dir / "classical_ml_known_trace_comparison.csv", index=False)
    return seg_df, coverage, db_summary


def _split_polyline_by_length(coords, max_len_m):
    """Split one ordered polyline into chunks with cumulative path <= max_len_m."""
    if len(coords) < 2:
        return [coords]
    if max_len_m <= 0:
        return [coords]
    pieces = []
    current = [coords[0]]
    current_len = 0.0
    for idx in range(1, len(coords)):
        p0 = np.asarray(coords[idx - 1], dtype=float)
        p1 = np.asarray(coords[idx], dtype=float)
        seg_vec = p1 - p0
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= 1e-9:
            continue
        remaining = seg_len
        start = p0.copy()
        while current_len + remaining > max_len_m and remaining > 1e-9:
            need = max_len_m - current_len
            if need <= 1e-9:
                if len(current) >= 2:
                    pieces.append(np.array(current, dtype=float))
                current = [start.copy()]
                current_len = 0.0
                continue
            frac = need / remaining
            split_pt = start + frac * (p1 - start)
            current.append(split_pt)
            if len(current) >= 2:
                pieces.append(np.array(current, dtype=float))
            current = [split_pt]
            current_len = 0.0
            start = split_pt
            remaining = float(np.linalg.norm(p1 - start))
        current.append(p1)
        current_len += remaining
    if len(current) >= 2:
        pieces.append(np.array(current, dtype=float))
    return pieces if pieces else [np.asarray(coords, dtype=float)]


def _chunk_trace_segments(segments, max_len_m):
    """Return ordered scarp chunks for plotting with per-chunk max path length."""
    chunks = []
    current = []
    current_len = 0.0
    for seg in segments:
        for piece in _split_polyline_by_length(seg["coords"], max_len_m):
            if len(piece) < 2:
                continue
            piece_len = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
            if current and current_len + piece_len > max_len_m:
                chunks.append(current)
                current = []
                current_len = 0.0
            current.append({**seg, "coords": piece, "chunk_piece_length_m": piece_len})
            current_len += piece_len
    if current:
        chunks.append(current)
    return chunks


def _window_from_coords(coords, bounds, buffer_m):
    """Window bounds around coordinates, clipped to DEM extent."""
    if coords is None or len(coords) == 0:
        return None
    finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if not np.any(finite):
        return None
    coords = coords[finite]
    xmin = max(float(bounds.left), float(np.min(coords[:, 0]) - buffer_m))
    xmax = min(float(bounds.right), float(np.max(coords[:, 0]) + buffer_m))
    ymin = max(float(bounds.bottom), float(np.min(coords[:, 1]) - buffer_m))
    ymax = min(float(bounds.top), float(np.max(coords[:, 1]) + buffer_m))
    if not np.all(np.isfinite([xmin, xmax, ymin, ymax])):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, xmax, ymin, ymax


def _plot_local_reconstructions(traces, segments, dem_path, out_dir,
                                candidate_points_csv=None,
                                inferred_points_csv=None,
                                trace_points_csv=None,
                                buffer_m=LOCAL_FIG_BUFFER_M,
                                max_scarp_len_m=LOCAL_FIG_MAX_SCARP_LENGTH_M):
    """Write local two-panel figures for scarp chunks and original trace points."""
    local_dir = safe_makedirs(out_dir / "local_reconstruction_plots")
    trace_lookup = {t["trace_id"]: t for t in traces}

    if candidate_points_csv is None:
        candidate_points_csv = out_dir / "all_candidate_points.csv"
    if inferred_points_csv is None:
        inferred_points_csv = out_dir / "all_inferred_points.csv"
    if trace_points_csv is None:
        trace_points_csv = STRUCT_PTS_PATH

    def _read_xy_csv(path, x_col="x", y_col="y"):
        if path is None or not Path(path).exists():
            return np.empty((0, 2), dtype=float)
        df = pd.read_csv(path)
        if x_col not in df.columns or y_col not in df.columns:
            return np.empty((0, 2), dtype=float)
        xy = df[[x_col, y_col]].to_numpy(dtype=float)
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        return xy[finite]

    all_candidates = _read_xy_csv(candidate_points_csv, x_col="x", y_col="y")
    all_inferred = pd.DataFrame(columns=["x", "y", "tier"])
    if inferred_points_csv is not None and Path(inferred_points_csv).exists():
        inf_df = pd.read_csv(inferred_points_csv)
        if {"x", "y"}.issubset(inf_df.columns):
            all_inferred = inf_df.copy()
            if "tier" not in all_inferred.columns:
                if "candidate_tier" in all_inferred.columns:
                    all_inferred["tier"] = all_inferred["candidate_tier"]
                else:
                    all_inferred["tier"] = np.nan
    trace_points_df = pd.DataFrame(columns=["x", "y", "source"])
    if Path(trace_points_csv).exists():
        raw = pd.read_csv(trace_points_csv)
        if {"XCoord", "YCoord", "source"}.issubset(raw.columns):
            trace_points_df = raw.rename(columns={"XCoord": "x", "YCoord": "y"})[["x", "y", "source"]]
        elif {"x", "y", "source"}.issubset(raw.columns):
            trace_points_df = raw[["x", "y", "source"]]
        elif {"x", "y"}.issubset(raw.columns):
            trace_points_df = raw[["x", "y"]].copy()
            trace_points_df["source"] = "unknown"
        finite = np.isfinite(trace_points_df["x"].to_numpy(dtype=float)) & np.isfinite(trace_points_df["y"].to_numpy(dtype=float))
        trace_points_df = trace_points_df.loc[finite].copy()
        if "source" in trace_points_df.columns:
            trace_points_df["source"] = trace_points_df["source"].astype(str)
        if not trace_points_df.empty:
            source_counts = trace_points_df["source"].value_counts().to_dict()
            print(f"[Plot] loaded {len(trace_points_df):,} trace points from CSV: {trace_points_csv}")
            print(f"[Plot] trace-point sources: {source_counts}")
        else:
            print(f"[Plot] trace CSV loaded but no finite points: {trace_points_csv}")
    else:
        print(f"[Plot] trace CSV not found: {trace_points_csv}")

    seg_by_trace = {}
    for seg in segments:
        seg_by_trace.setdefault(seg["known_trace_id"], []).append(seg)
    figure_count = 0
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        for trace_id, trace_segments in seg_by_trace.items():
            trace = trace_lookup.get(trace_id)
            if trace is None or not trace_segments:
                continue
            chunks = _chunk_trace_segments(trace_segments, max_scarp_len_m)
            for chunk_idx, chunk in enumerate(chunks, 1):
                coord_blocks = []
                for seg in chunk:
                    xy = np.asarray(seg["coords"], dtype=float)
                    if xy.ndim != 2 or xy.shape[0] == 0:
                        continue
                    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
                    if np.any(finite):
                        coord_blocks.append(xy[finite])
                if not coord_blocks:
                    print(f"  [plot] trace {trace_id} chunk {chunk_idx}: skipped (no finite scarp coordinates)")
                    continue
                scarp_coords = np.vstack(coord_blocks)
                extent = _window_from_coords(scarp_coords, bounds, buffer_m)
                if extent is None:
                    print(f"  [plot] trace {trace_id} chunk {chunk_idx}: skipped (invalid extent)")
                    continue
                xmin, xmax, ymin, ymax = extent
                win, reason = _window_from_bounds_clipped(src, xmin, ymin, xmax, ymax)
                if win is None:
                    if reason not in (None, "outside_dem"):
                        print(f"  [plot] trace {trace_id} chunk {chunk_idx}: skipped ({reason})")
                    continue

                dem = src.read(1, window=win, out_shape=(min(900, int(win.height)), min(900, int(win.width))),
                               resampling=Resampling.bilinear).astype(float)
                if src.nodata is not None:
                    dem[dem == src.nodata] = np.nan
                win_transform = src.window_transform(win)
                hs = compute_hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), abs(src.res[0]) * win.height / dem.shape[0])
                win_bounds = rasterio.windows.bounds(win, src.transform)

                fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
                x0 = float(win_bounds[0])
                x1 = float(win_bounds[2])
                y0 = float(win_bounds[1])
                y1 = float(win_bounds[3])
                if not np.all(np.isfinite([x0, x1, y0, y1])) or x1 <= x0 or y1 <= y0:
                    print(f"  [plot] trace {trace_id} chunk {chunk_idx}: skipped (non-finite window bounds)")
                    plt.close(fig)
                    continue
                for ax in axes:
                    ax.imshow(hs, cmap="gray",
                              extent=(x0, x1, y0, y1),
                              origin="upper", alpha=0.8)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_aspect("equal", adjustable="box")
                    # Use fixed ticks to avoid auto-locator failures on pathological extents.
                    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                        ax.set_xticks(np.linspace(xmin, xmax, 5))
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                        ax.set_yticks(np.linspace(ymin, ymax, 5))
                    ax.set_xlabel("Easting (m)")
                    ax.set_ylabel("Northing (m)")

                # Left panel: all candidates + inferred locations (from CSV), clipped to panel extent.
                if len(all_candidates):
                    cand_mask = ((all_candidates[:, 0] >= xmin) & (all_candidates[:, 0] <= xmax) &
                                 (all_candidates[:, 1] >= ymin) & (all_candidates[:, 1] <= ymax))
                    if np.any(cand_mask):
                        axes[0].scatter(all_candidates[cand_mask, 0], all_candidates[cand_mask, 1],
                                        s=0.5, c="yellow", alpha=0.30, edgecolors="none", label="Candidates")
                if not all_inferred.empty:
                    panel_inferred = all_inferred[
                        (all_inferred["x"] >= xmin) & (all_inferred["x"] <= xmax) &
                        (all_inferred["y"] >= ymin) & (all_inferred["y"] <= ymax)
                    ]
                    if not panel_inferred.empty:
                        tier_colors = {1: "red", 2: "orange", 3: "yellow"}
                        for tier_value, tier_color in tier_colors.items():
                            grp = panel_inferred[panel_inferred["tier"].fillna(-1).astype(float) == float(tier_value)]
                            if not grp.empty:
                                axes[0].scatter(grp["x"].to_numpy(dtype=float), grp["y"].to_numpy(dtype=float),
                                                s=6, c=tier_color, alpha=0.35, edgecolors="none",
                                                label=f"Inferred tier {tier_value}")
                        grp_other = panel_inferred[~panel_inferred["tier"].fillna(-1).isin([1, 2, 3])]
                        if not grp_other.empty:
                            axes[0].scatter(grp_other["x"].to_numpy(dtype=float), grp_other["y"].to_numpy(dtype=float),
                                            s=6, c="red", alpha=0.35, edgecolors="none", label="Inferred")


                # Right panel: original fault trace points from CSV, clipped to panel extent.
                if not trace_points_df.empty:
                    panel_points = trace_points_df[
                        (trace_points_df["x"] >= xmin) & (trace_points_df["x"] <= xmax) &
                        (trace_points_df["y"] >= ymin) & (trace_points_df["y"] <= ymax)
                    ]
                    if not panel_points.empty:
                        for source_name, grp in panel_points.groupby("source"):
                            axes[1].scatter(grp["x"].to_numpy(dtype=float), grp["y"].to_numpy(dtype=float), s=5,
                                            c=SOURCE_COLORS.get(source_name, "white"), alpha=0.75,
                                            edgecolors="none", label=f"{source_name} points")

                chunk_len = sum(s.get("chunk_piece_length_m", 0.0) for s in chunk)
                axes[0].set_title("Candidates (yellow) and inferred locations (red)")
                axes[1].set_title("Original fault-trace points (CSV)")
                for ax in axes:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
                fig.suptitle(
                    f"Trace {trace_id} ({trace.get('source', 'unknown')}), chunk {chunk_idx}/{len(chunks)}"
                    f" | scarp length {chunk_len:.0f} m"
                )
                out_png = local_dir / f"trace_{trace_id:05d}_chunk_{chunk_idx:03d}.png"
                try:
                    fig.tight_layout()
                    fig.savefig(out_png, dpi=200)
                except ValueError as exc:
                    if "cannot convert float NaN to integer" not in str(exc):
                        raise
                    print(f"  [plot] trace {trace_id} chunk {chunk_idx}: retrying save after NaN tick-space fallback")
                    for ax in axes:
                        ax.set_aspect("auto")
                        ax.set_xlim(x0, x1)
                        ax.set_ylim(y0, y1)
                        if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                            ax.set_xticks(np.linspace(x0, x1, 5))
                        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                            ax.set_yticks(np.linspace(y0, y1, 5))
                        ax.set_xlabel("Easting (m)")
                        ax.set_ylabel("Northing (m)")
                    fig.savefig(out_png, dpi=200, bbox_inches="tight")
                plt.close(fig)
                figure_count += 1
    print(f"[Plot] Wrote {figure_count} local reconstruction figures -> {local_dir}")


def _plot_summary(traces, segments, dem_path, out_path):
    """Compact hillshade overview, using the same terrain context as Script 05."""
    fig, ax = plt.subplots(figsize=(14, 11))
    with rasterio.open(dem_path) as src:
        out_h, out_w = min(1200, src.height), min(1200, src.width)
        dem = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        hs = compute_hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), abs(src.res[0]) * src.height / out_h)
        b = src.bounds
        ax.imshow(hs, cmap="gray", extent=(b.left, b.right, b.bottom, b.top), origin="upper", alpha=.75)
        ax.set_xlim(float(b.left), float(b.right))
        ax.set_ylim(float(b.bottom), float(b.top))
    for trace in traces:
        xy = trace["coords"]
        mask = ((xy[:, 0] >= b.left) & (xy[:, 0] <= b.right) &
                (xy[:, 1] >= b.bottom) & (xy[:, 1] <= b.top))
        if np.any(mask):
            ax.scatter(xy[mask, 0], xy[mask, 1], s=2,
                       c=SOURCE_COLORS.get(trace["source"], "white"), alpha=.4,
                       edgecolors="none")
    for seg in segments:
        xy = seg["coords"]
        mask = ((xy[:, 0] >= b.left) & (xy[:, 0] <= b.right) &
                (xy[:, 1] >= b.bottom) & (xy[:, 1] <= b.top))
        if np.any(mask):
            conf = float(np.clip(seg.get("confidence_score_0_10", 0.0) / 10.0, 0.0, 1.0))
            ax.plot(xy[mask, 0], xy[mask, 1], color=plt.cm.RdYlGn(conf), lw=1.5)
    ax.set_title("Known traces (scatter) and curvature-only scarp segments (lines)")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-source", default="lidar_0_5m", choices=["lidar_0_5m", "dem10m", "srtm1s"])
    parser.add_argument("--buffer-m", type=float, default=FAULT_VALIDATION_TOLERANCE_M)
    parser.add_argument("--max-traces", type=int, default=None, help="Optional development limit.")
    parser.add_argument("--clip-points-to-dem", action="store_true",
                        help="Clip known-fault points to DEM bounds before trace reconstruction.")
    parser.add_argument("--point-buffer-m", type=float, default=0.0,
                        help="Optional buffer for --clip-points-to-dem (meters).")
    parser.add_argument("--max-windows-per-trace", type=int, default=None,
                        help="Optional development cap on local windows processed per trace.")
    parser.add_argument("--write-local-lsm", action="store_true", help="Save one local LSM GeoTIFF per trace.")
    args = parser.parse_args()
    dem_name = {"lidar_0_5m": "lidar_0_5m_epsg28355_10m_smoke.tif", "dem10m": "dem10m_epsg28355_clipped.tif",
                "srtm1s": "srtm1s_epsg28355_clipped.tif"}[args.dem_source]
    dem_path = OUT_DIR / dem_name
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    out_dir = safe_makedirs(OUT_DIR / "known_trace_scarp_validation" / args.dem_source)
    lsm_dir = safe_makedirs(out_dir / "local_lsm")
    rem = OUT_DIR / "visual_audit" / "rem_hand"
    masks = [rem / f"{args.dem_source}_hand.tif", rem / f"{args.dem_source}_river_bed.tif",
             rem / f"{args.dem_source}_river_surface.tif", rem / f"{args.dem_source}_river_valley.tif"]
    priors = _load_phase1_priors(args.dem_source)
    expr_path = PHASE1_AUDIT_DIR / f"{args.dem_source}_expression_quality.csv"
    expr_points_path = PHASE1_AUDIT_DIR / f"{args.dem_source}_expression_quality_points.csv"
    scarp_path = PHASE1_AUDIT_DIR / f"{args.dem_source}_scarp_analysis.csv"
    print(f"[Priors] expression_quality exists={expr_path.exists()} ({expr_path})")
    print(f"[Priors] expression_quality_points exists={expr_points_path.exists()} ({expr_points_path})")
    print(f"[Priors] scarp_analysis exists={scarp_path.exists()} ({scarp_path})")

    with rasterio.open(dem_path) as dem_src:
        dem_bounds = dem_src.bounds
    known_points = _load_known_points(dem_bounds if args.clip_points_to_dem else None,
                                      buffer_m=args.point_buffer_m)
    if args.clip_points_to_dem:
        print(f"[Known Points] DEM-clipped points: {len(known_points):,} (buffer={args.point_buffer_m:.1f} m)")
        print("[Known Points] NOTE: clipping before reconstruction may change trace_id numbering")
        print("[Known Points] and reduce direct matching to phase1 prior tables.")

    traces = reconstruct_fault_traces(known_points)
    traces = [t for t in traces if t["source"] in FAULT_DB_SOURCES]
    traces = _attach_trace_priors(traces, priors)
    source_priority = {"FSD1": 0, "CWvic1": 1, "GeoStru2501": 2, "GeoStru501": 3}
    traces = sorted(traces, key=lambda t: (source_priority.get(t.get("source"), 99), t.get("trace_id", 0)))
    class_counts = Counter(t.get("prior_quality_class", "unclassified") for t in traces)
    pos_count = sum(class_counts.get(c, 0) for c in POSITIVE_PRIOR_CLASSES)
    print(
        "[Priors] trace prior classes: "
        f"Class1/2={pos_count}, Class3={class_counts.get('Class 3', 0)}, "
        f"unclassified={class_counts.get('unclassified', 0)}, total={len(traces)}"
    )
    if args.max_traces:
        traces = traces[:args.max_traces]
    all_segments = []
    window_diagnostics = []
    all_scarp_bases_rows = []
    all_candidate_xy = []
    with rasterio.open(dem_path) as src:
        resolution = abs(src.res[0])
        min_half_buffer = _min_half_window_buffer_m(resolution)
        print(
            f"[Scale] DEM resolution={resolution:.3f} m/px, "
            f"minimum enforced window side={2 * min_half_buffer:.1f} m ({MIN_WINDOW_PX} px)"
        )
        for number, trace in enumerate(traces, 1):
            windows = _trace_windows(src, trace, args.buffer_m)
            if not windows:
                print(f"[{number}/{len(traces)}] trace {trace['trace_id']}: skipped (no in-extent local windows)")
                continue
            trace_pixels = 0
            trace_segments = []
            signal_windows = 0
            for window_idx, spec in enumerate(windows):
                if args.max_windows_per_trace is not None and window_idx >= args.max_windows_per_trace:
                    break
                if window_idx == 0 or (window_idx + 1) % 10 == 0:
                    print(
                        f"    trace {trace['trace_id']}: window {window_idx + 1}/{len(windows)} "
                        f"({spec['kind']}, {int(spec['window'].width)}x{int(spec['window'].height)} px)"
                    )
                window = spec["window"]
                dem = src.read(1, window=window).astype(np.float32)
                valid = np.isfinite(dem) if src.nodata is None else np.isfinite(dem) & (dem != src.nodata)
                transform = src.window_transform(window)
                river = _read_mask_on_window(masks, transform, src.crs, dem.shape)
                state, lsm_diag = _local_lsm(np.where(valid, dem, np.nan), valid, river, abs(src.res[0]))
                window_diagnostics.append({
                    "trace_id": trace["trace_id"],
                    "source": trace["source"],
                    "window_index": window_idx,
                    "window_kind": spec["kind"],
                    "window_width_px": int(window.width),
                    "window_height_px": int(window.height),
                    "window_pixels": int(window.width * window.height),
                    "prior_quality_class": trace.get("prior_quality_class", "unclassified"),
                    "lsm_status": "ok" if state is not None else lsm_diag.get("reason", "unknown"),
                    "valid_pixels": lsm_diag.get("valid_pixels", 0),
                    "interior_pixels": lsm_diag.get("interior_pixels", 0),
                    "persistent_pixels": lsm_diag.get("persistent_pixels", 0),
                    "vals_count": lsm_diag.get("vals_count", 0),
                    "min_vals_required": lsm_diag.get("min_vals_required", 0),
                    "persistence_k": lsm_diag.get("persistence_k", 0),
                })
                if state is None:
                    continue
                signal_windows += 1
                cand_rows, cand_cols = np.where(state["candidates"])
                if len(cand_rows):
                    cand_x, cand_y = transform * (cand_cols, cand_rows)
                    all_candidate_xy.append(np.c_[cand_x, cand_y])
                pixels = _classify_scarp_pixels(state, dem, river, abs(src.res[0]), transform)
                trace_pixels += len(pixels)
                if not pixels.empty:
                    keep_cols = ["base_x", "base_y", "tier", "break_type"]
                    if "break_type" not in pixels.columns:
                        keep_cols = ["base_x", "base_y", "tier"]
                    elif "tier" not in pixels.columns:
                        keep_cols = ["base_x", "base_y", "break_type"]
                    all_scarp_bases_rows.append(pixels[keep_cols].copy())
                segments = _segments_from_points(
                    pixels,
                    trace,
                    abs(src.res[0]),
                    river,
                    transform,
                    dem,
                    persistence_k=state.get("persistence_k", LSM_PERSISTENCE_K),
                )
                for segment in segments:
                    segment["window_kind"] = spec["kind"]
                    segment["window_index"] = window_idx
                trace_segments.extend(segments)
                if args.write_local_lsm:
                    _write_local_raster(lsm_dir / f"trace_{trace['trace_id']:05d}_w{window_idx:03d}_{spec['kind']}_lsm.tif",
                                        state, transform, src.crs)
            trace_segments = _annotate_segments_with_priors(trace_segments, trace, priors)
            all_segments.extend(trace_segments)
            print(f"[{number}/{len(traces)}] trace {trace['trace_id']}: {len(windows)} windows, "
                  f"{signal_windows} with signal, {trace_pixels} scarp bases, {len(trace_segments)} segments, "
                  f"prior={trace.get('prior_quality_class', 'unclassified')}")

    segments_before_filter = len(all_segments)
    all_segments = _filter_river_segments(
        all_segments,
        masks[0],
        hand_threshold_m=HAND_INFLUENCE_THRESHOLD_M,
        max_fraction=HAND_SEGMENT_MAX_FRACTION,
    )
    print(
        f"[River Filter] kept={len(all_segments)} removed={segments_before_filter - len(all_segments)} "
        f"(hand<={HAND_INFLUENCE_THRESHOLD_M}m, frac>={HAND_SEGMENT_MAX_FRACTION} removed)"
    )

    diag_df = pd.DataFrame(window_diagnostics)
    if not diag_df.empty:
        diag_df.to_csv(out_dir / "window_diagnostics.csv", index=False)
        status_counts = diag_df["lsm_status"].value_counts(dropna=False).to_dict()
        print(f"[Diagnostics] Wrote window diagnostics -> {out_dir / 'window_diagnostics.csv'}")
        print(f"[Diagnostics] LSM status counts: {status_counts}")
    _summary_and_comparison(traces, all_segments, out_dir, args.dem_source)
    _plot_summary(traces, all_segments, dem_path, out_dir / "known_trace_scarp_summary.png")

    all_candidates = np.vstack(all_candidate_xy) if all_candidate_xy else np.empty((0, 2), dtype=float)
    pd.DataFrame(all_candidates, columns=["x", "y"]).to_csv(out_dir / "all_candidate_points.csv", index=False)
    if all_scarp_bases_rows:
        inferred_df = pd.concat(all_scarp_bases_rows, ignore_index=True)
        inferred_df = inferred_df.rename(columns={"base_x": "x", "base_y": "y"})
        if "tier" not in inferred_df.columns:
            inferred_df["tier"] = np.nan
        inferred_df.to_csv(out_dir / "all_inferred_points.csv", index=False)
    else:
        pd.DataFrame(columns=["x", "y", "tier"]).to_csv(out_dir / "all_inferred_points.csv", index=False)
    trace_rows = []
    for trace in traces:
        xy = np.asarray(trace["coords"], dtype=float)
        if xy.ndim != 2 or xy.shape[0] == 0:
            continue
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if not np.any(finite):
            continue
        xy = xy[finite]
        trace_rows.append(pd.DataFrame({
            "trace_id": int(trace["trace_id"]),
            "source": trace.get("source", ""),
            "x": xy[:, 0],
            "y": xy[:, 1],
        }))
    if trace_rows:
        pd.concat(trace_rows, ignore_index=True).to_csv(out_dir / "known_trace_points.csv", index=False)
    else:
        pd.DataFrame(columns=["trace_id", "source", "x", "y"]).to_csv(
            out_dir / "known_trace_points.csv", index=False
        )

    _plot_local_reconstructions(
        traces,
        all_segments,
        dem_path,
        out_dir,
        candidate_points_csv=out_dir / "all_candidate_points.csv",
        inferred_points_csv=out_dir / "all_inferred_points.csv",
        trace_points_csv=STRUCT_PTS_PATH,
    )
    print(f"Done. Outputs: {out_dir}")


if __name__ == "__main__":
    main()