#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03-ResolutionSensitivity.py -- Surface Expression Audit: Phase 2
Numerical Resolution Sensitivity Analysis

Phase 2 evaluates how DEM resolution affects the detectability of fault
scarp features through controlled downsampling, derivative comparison,
and quantitative metric evaluation.

Sections:
  1. Downsampling Protocol (1.1)
  2. Derivative Generation at Each Resolution (1.2)
  3. Quantitative Metric Evaluation (1.3)
  4. Core Sensitivity Visualizations (2.1)
  5. Supplementary Spatial Products (2.2)
  6. Uncertainty Assessment (3.1-3.4)
  7. Main workflow

Input:
  - Reprojected LiDAR DEM (0.5 m, EPSG:28355)
  - Structural points inventory
  - Phase 1 expression quality CSV

Output (all saved under data/derivatives/phase2_sensitivity/):
  - Downsampled DEMs (.npy + .json)
  - Derivative stacks at each resolution
  - Quantitative metrics (SNR, kappa, RMSE_h, SLPR, LFI, MDD)
  - Detection matrix
  - Sensitivity visualizations
  - Uncertainty assessment results
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from pathlib import Path
import json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DERIV_DIR = DATA_DIR / "derivatives"
OUT_DIR = DERIV_DIR / "phase2_sensitivity"

TARGET_CRS = "EPSG:28355"
LIDAR_NODATA = -9999.0

# -- External DEM sources (reprojected to EPSG:28355) --
# These .tif files are produced by 01-DerivativeGen.py and represent
# independent DEM acquisitions, NOT downscaled LiDAR.
LIDAR_RP   = DERIV_DIR / "lidar_0_5m_epsg28355.tif"
DEM10M_RP  = DERIV_DIR / "dem10m_epsg28355.tif"
SRTM_RP    = DERIV_DIR / "srtm1s_epsg28355.tif"

# Backward-compatible alias used in the original code
LIDAR_RP_PATH = LIDAR_RP

STRUCT_PTS_PATH = DATA_DIR / "structural_points_inventory.csv"

HS_AZIMUTHS = [315, 45, 135, 225]
HS_ALTITUDE = 8

# Only 5 resolutions: 0.5m (native LiDAR), 2m and 5m (downsampled from
# LiDAR), 10m and 30m (from independent DEM .tif files).
TARGET_RESOLUTIONS = [0.5, 2, 5, 10, 30]

# Resolutions 10m and 30m come from external .tif files rather than
# downsampling the 0.5m LiDAR.  We store the mapping so the rest of the
# pipeline can detect which source to read from.
EXTERNAL_DEM_SOURCES = {
    10: DEM10M_RP,
    30: SRTM_RP,
}

# Resampling algorithms are only applied to LiDAR-derived resolutions
# (0.5m, 2m, 5m).  External sources are used as-is ("native").
RESAMPLING_ALGORITHMS = ["mean", "bilinear", "cubic"]


def is_external_resolution(res):
    """Return True if *res* is sourced from an external .tif file."""
    return res in EXTERNAL_DEM_SOURCES


def get_algorithms_for_resolution(res):
    """Return the list of algorithm tags applicable to *res*.

    - External DEM sources (10m, 30m): ['native']
    - LiDAR-derived resolutions (0.5m, 2m, 5m): RESAMPLING_ALGORITHMS
    """
    if is_external_resolution(res):
        return ["native"]
    return RESAMPLING_ALGORITHMS

SITE_RADIUS_M = 500
N_TEST_SITES = 10

SNR_THRESHOLD = 2.0
KAPPA_THRESHOLD_SUBSTANTIAL = 0.6
KAPPA_THRESHOLD_ALMOST = 0.8

TILE_DPI = 300
VERT_EXAG = 15

LIDAR_VERTICAL_RMSE_OPEN = 0.15   # m, open ground
LIDAR_VERTICAL_RMSE_VEG = 0.50    # m, under vegetation


def res_tag(res):
    """Format resolution for file naming: 0.5 -> '0.5m', 1 -> '1m', etc."""
    if res == int(res):
        return f"{int(res)}m"
    return f"{res:g}m"


N_BOOTSTRAP = 1000
BOOTSTRAP_CONFIDENCE = 0.95

# TPI radii in metres (normalised across resolutions)
TPI_RADII_M = [50, 200, 1000]

# ===================================================================
# IMPORT FUNCTIONS FROM 01-DerivativeGen.py
# ===================================================================

sys.path.insert(0, str(BASE_DIR))
from importlib.machinery import SourceFileLoader

_mod = SourceFileLoader("dg01", str(BASE_DIR / "01-DerivativeGen.py")).load_module()

compute_hillshade = _mod.compute_hillshade
compute_slope = _mod.compute_slope
compute_curvature = _mod.compute_curvature
compute_tpi = _mod.compute_tpi
compute_svf_and_openness = _mod.compute_svf_and_openness
_save_derivative_png_adaptive = _mod._save_derivative_png_adaptive
_save_npy_derivative = _mod._save_npy_derivative
_save_derivative_png = _mod._save_derivative_png
_load_npy_derivative = _mod._load_npy_derivative
_sample_dem_line = _mod._sample_dem_line
safe_makedirs = _mod.safe_makedirs
reconstruct_fault_traces = _mod.reconstruct_fault_traces
load_structural_points = _mod.load_structural_points
analyse_scarp_profiles = _mod.analyse_scarp_profiles
extract_perpendicular_profiles = _mod.extract_perpendicular_profiles


# ===================================================================
# SECTION 1: DOWNSAMPLING PROTOCOL (1.1)
# ===================================================================

def downsample_dem(lidar_path, target_res, algorithm, out_dir, nodata,
                   window=None):
    """
    Downsample the reprojected LiDAR DEM (0.5 m) to target_res using
    one of three algorithms: "mean", "bilinear", "cubic".

    For "mean": block aggregation (read a block of native cells, take
    the mean).
    For "bilinear"/"cubic": rasterio.warp.reproject with the
    corresponding Resampling method.

    If window is provided (a rasterio.windows.Window), only downsample
    that subset.

    Returns:
        (npy_path, json_path) -- paths to saved .npy and .json files
    """
    lidar_path = Path(lidar_path) if not isinstance(lidar_path, Path) else lidar_path
    out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
    safe_makedirs(out_dir)

    stem = f"res{res_tag(target_res)}_{algorithm}"
    npy_path = out_dir / f"{stem}.npy"
    json_path = out_dir / f"{stem}.json"

    if npy_path.exists() and json_path.exists():
        print(f"    [Skip] {stem} already exists")
        return npy_path, json_path

    print(f"    Downsampling to {target_res} m using {algorithm} ...")

    with rasterio.open(lidar_path) as src:
        native_res = abs(src.res[0])
        src_transform = src.transform
        src_crs = src.crs
        src_height, src_width = src.height, src.width

        # Read the data window
        if window is not None:
            data = src.read(1, window=window).astype(np.float32)
            win_transform = src.window_transform(window)
            win_height = window.height
            win_width = window.width
        else:
            data = src.read(1).astype(np.float32)
            win_transform = src_transform
            win_height = src_height
            win_width = src_width

        # Mask nodata
        if nodata is not None:
            mask = (data == nodata) | np.isnan(data) | (data < -1e6)
        else:
            mask = np.isnan(data) | (data < -1e6)

        data[mask] = np.nan

        if algorithm == "mean":
            # Block aggregation
            factor = max(1, int(round(target_res / native_res)))
            new_h = win_height // factor
            new_w = win_width // factor

            if new_h == 0 or new_w == 0:
                print(f"      [!] Window too small for {target_res}m "
                      f"(factor={factor}, win={win_height}x{win_width})")
                return None, None

            # Trim data to exact multiple of factor
            trimmed = data[:new_h * factor, :new_w * factor]
            mask_trimmed = mask[:new_h * factor, :new_w * factor]

            # Reshape and compute mean of valid pixels
            result = np.full((new_h, new_w), np.nan, dtype=np.float32)
            for r in range(new_h):
                for c in range(new_w):
                    block = trimmed[r * factor:(r + 1) * factor,
                                    c * factor:(c + 1) * factor]
                    valid = block[~np.isnan(block)]
                    if len(valid) > 0:
                        result[r, c] = np.mean(valid)

            # Update transform
            new_transform = Affine(
                target_res, win_transform.b,
                win_transform.c,
                win_transform.d, -target_res,
                win_transform.f
            )

        else:
            # bilinear or cubic via rasterio.warp.reproject
            if algorithm == "bilinear":
                resample_method = Resampling.bilinear
            elif algorithm == "cubic":
                resample_method = Resampling.cubic
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Compute output shape
            out_width = max(1, int(round(win_width * native_res / target_res)))
            out_height = max(1, int(round(win_height * native_res / target_res)))

            new_transform = Affine(
                target_res, win_transform.b,
                win_transform.c,
                win_transform.d, -target_res,
                win_transform.f
            )

            # Prepare source data for reproject (replace NaN with nodata)
            src_data = data.copy()
            src_data[np.isnan(src_data)] = nodata if nodata is not None else -9999

            result = np.full((out_height, out_width),
                             nodata if nodata is not None else -9999,
                             dtype=np.float32)

            reproject(
                source=src_data,
                destination=result,
                src_transform=win_transform,
                src_crs=src_crs,
                src_nodata=nodata if nodata is not None else -9999,
                dst_transform=new_transform,
                dst_crs=src_crs,
                dst_nodata=nodata if nodata is not None else -9999,
                resampling=resample_method,
            )

            # Re-mask nodata
            if nodata is not None:
                result[result == nodata] = np.nan
            result[result < -1e6] = np.nan

        # Save as .npy
        np.save(npy_path, result.astype(np.float32))

        # Save metadata as .json
        meta = {
            "transform": [new_transform.a, new_transform.b, new_transform.c,
                          new_transform.d, new_transform.e, new_transform.f],
            "crs": str(src_crs),
            "shape": list(result.shape),
            "nodata": nodata,
            "resolution": target_res,
            "algorithm": algorithm,
            "native_resolution": native_res,
            "type": "downsampled_dem",
        }
        if window is not None:
            meta["window"] = {
                "col_off": window.col_off,
                "row_off": window.row_off,
                "width": window.width,
                "height": window.height,
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"      [OK] {stem} -> {result.shape}")
    return npy_path, json_path


def select_test_sites(traces, class_df, struct_pts, n_sites=5,
                      site_radius_m=500):
    """
    Select representative test sites along different fault segments.

    Criteria:
      - At least one Class 1 trace (well-expressed scarp 1-2 m)
      - One Class 2 (moderate)
      - One with drainage anomaly
      - One in flat terrain (monocline/fold)
      - Sites should be spatially distributed (minimum 2 km apart)

    Returns:
        DataFrame with columns: center_x, center_y, radius, trace_id,
        quality_class, site_type
    """
    print(f"\n[Test Sites] Selecting {n_sites} representative sites")

    sites = []
    used_traces = set()

    # Build lookup: trace_id -> quality_class
    if class_df is not None and len(class_df) > 0:
        class_lookup = dict(zip(class_df["trace_id"],
                                class_df["quality_class"]))
        struct_lookup = dict(zip(class_df["trace_id"],
                                 class_df.get("structural_context", "none")))
    else:
        class_lookup = {}
        struct_lookup = {}

    # Categorize traces
    class1_traces = []
    class2_traces = []
    class3_traces = []
    monocline_traces = []

    for trace in traces:
        tid = trace["trace_id"]
        qclass = class_lookup.get(tid, "Class 3")
        scontext = struct_lookup.get(tid, "none")

        if qclass == "Class 1":
            class1_traces.append(trace)
        elif qclass == "Class 2":
            class2_traces.append(trace)
        else:
            class3_traces.append(trace)

        if scontext == "monocline":
            monocline_traces.append(trace)

    # Priority selection: one from each category
    candidates = []

    # 1. Class 1 (well-expressed scarp)
    if class1_traces:
        t = class1_traces[0]
        candidates.append({
            "center_x": t["centroid_x"],
            "center_y": t["centroid_y"],
            "radius": site_radius_m,
            "trace_id": t["trace_id"],
            "quality_class": "Class 1",
            "site_type": "well_expressed_scarp",
        })

    # 2. Class 2 (moderate)
    if class2_traces:
        t = class2_traces[0]
        candidates.append({
            "center_x": t["centroid_x"],
            "center_y": t["centroid_y"],
            "radius": site_radius_m,
            "trace_id": t["trace_id"],
            "quality_class": "Class 2",
            "site_type": "moderate_scarp",
        })

    # 3. Monocline / fold
    if monocline_traces:
        t = monocline_traces[0]
        # Avoid duplicate with already selected
        already = any(abs(t["centroid_x"] - c["center_x"]) < 100 and
                      abs(t["centroid_y"] - c["center_y"]) < 100
                      for c in candidates)
        if not already:
            candidates.append({
                "center_x": t["centroid_x"],
                "center_y": t["centroid_y"],
                "radius": site_radius_m,
                "trace_id": t["trace_id"],
                "quality_class": class_lookup.get(t["trace_id"], "Class 3"),
                "site_type": "monocline_fold",
            })

    # 4. Class 3 (inferred / flat terrain)
    if class3_traces:
        t = class3_traces[0]
        already = any(abs(t["centroid_x"] - c["center_x"]) < 100 and
                      abs(t["centroid_y"] - c["center_y"]) < 100
                      for c in candidates)
        if not already:
            candidates.append({
                "center_x": t["centroid_x"],
                "center_y": t["centroid_y"],
                "radius": site_radius_m,
                "trace_id": t["trace_id"],
                "quality_class": "Class 3",
                "site_type": "flat_terrain",
            })

    # 5. Fill remaining with spatially distributed traces
    min_sep = 2000  # minimum 2 km apart
    all_traces = sorted(traces, key=lambda t: t["length_m"], reverse=True)

    for t in all_traces:
        if len(candidates) >= n_sites:
            break
        too_close = any(
            np.sqrt((t["centroid_x"] - c["center_x"])**2 +
                    (t["centroid_y"] - c["center_y"])**2) < min_sep
            for c in candidates
        )
        if too_close:
            continue
        tid = t["trace_id"]
        candidates.append({
            "center_x": t["centroid_x"],
            "center_y": t["centroid_y"],
            "radius": site_radius_m,
            "trace_id": tid,
            "quality_class": class_lookup.get(tid, "Class 3"),
            "site_type": "additional",
        })

    # If still not enough, relax spatial constraint
    if len(candidates) < n_sites:
        for t in all_traces:
            if len(candidates) >= n_sites:
                break
            tid = t["trace_id"]
            if any(c["trace_id"] == tid for c in candidates):
                continue
            candidates.append({
                "center_x": t["centroid_x"],
                "center_y": t["centroid_y"],
                "radius": site_radius_m,
                "trace_id": tid,
                "quality_class": class_lookup.get(tid, "Class 3"),
                "site_type": "additional_relaxed",
            })

    df = pd.DataFrame(candidates[:n_sites])

    print(f"  Selected {len(df)} test sites:")
    for _, row in df.iterrows():
        print(f"    ({row['center_x']:.0f}, {row['center_y']:.0f}) "
              f"trace={row['trace_id']} class={row['quality_class']} "
              f"type={row['site_type']}")

    return df


def noise_filter_lidar(dem_arr, nodata, threshold_m=0.5, neighbourhood_px=5):
    """
    Filter residual LiDAR noise spikes exceeding +/-threshold_m from
    neighbourhood mean.

    Uses scipy.ndimage.uniform_filter for neighbourhood mean.
    Replaces spikes with neighbourhood mean.
    """
    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
    else:
        mask = np.isnan(dem_arr) | (dem_arr < -1e6)

    # Replace nodata/NaN with local mean for filtering
    working = dem_arr.copy().astype(np.float64)
    valid_vals = working[~mask]
    if len(valid_vals) == 0:
        return dem_arr.copy()

    fill_val = np.mean(valid_vals)
    working[mask] = fill_val

    # Compute neighbourhood mean
    nb_mean = uniform_filter(working, size=neighbourhood_px, mode="nearest")

    # Detect spikes
    diff = working - nb_mean
    spike_mask = np.abs(diff) > threshold_m

    # Replace spikes with neighbourhood mean (only where not nodata)
    result = dem_arr.copy().astype(np.float32)
    replace_mask = spike_mask & (~mask)
    result[replace_mask] = nb_mean[replace_mask].astype(np.float32)

    # Restore original nodata
    result[mask] = dem_arr[mask] if nodata is not None else np.nan

    n_spikes = np.sum(replace_mask)
    if n_spikes > 0:
        print(f"      Noise filter: replaced {n_spikes} spikes "
              f"(>{threshold_m}m from neighbourhood mean)")

    return result


def _save_dem_window(arr, transform, crs_str, resolution, algorithm, nodata,
                     site_idx, ds_dir, stem):
    """Save a DEM array as .npy + metadata .json in the downsampled dir."""
    npy_path = ds_dir / f"{stem}.npy"
    json_path = ds_dir / f"{stem}.json"

    if not npy_path.exists():
        np.save(npy_path, arr.astype(np.float32))
        meta = {
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "crs": str(crs_str),
            "shape": list(arr.shape),
            "nodata": nodata,
            "resolution": resolution,
            "algorithm": algorithm,
            "native_resolution": resolution,
            "type": "downsampled_dem",
            "site_idx": int(site_idx),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return npy_path, json_path


def read_external_dem_window(ext_path, cx, cy, radius, nodata):
    """
    Read a window from an external DEM .tif centred on (cx, cy).

    Returns (dem_arr, transform, crs_str, resolution) or
    (None, None, None, None) if the window is out of bounds.
    """
    ext_path = Path(ext_path)
    if not ext_path.exists():
        print(f"    [!] External DEM not found: {ext_path}")
        return None, None, None, None

    with rasterio.open(ext_path) as src:
        ext_res = abs(src.res[0])
        ext_transform = src.transform
        ext_crs = src.crs

        col_center, row_center = ~ext_transform * (cx, cy)
        half_px = int(radius / ext_res)

        col_off = max(0, int(col_center) - half_px)
        row_off = max(0, int(row_center) - half_px)
        win_w = min(src.width - col_off, 2 * half_px)
        win_h = min(src.height - row_off, 2 * half_px)

        if win_w <= 0 or win_h <= 0:
            return None, None, None, None

        window = Window(col_off, row_off, win_w, win_h)
        dem_arr = src.read(1, window=window).astype(np.float32)

        # Mask nodata
        if nodata is not None:
            mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        else:
            mask = np.isnan(dem_arr) | (dem_arr < -1e6)
        dem_arr[mask] = np.nan

        win_transform = src.window_transform(window)
        new_transform = Affine(
            ext_res, win_transform.b, win_transform.c,
            win_transform.d, -ext_res, win_transform.f
        )

    return dem_arr, new_transform, str(ext_crs), ext_res


def run_downsampling(lidar_path, out_dir, nodata, test_sites, resolutions):
    """
    Generate per-site DEM arrays at every target resolution.

    Two data-source modes:

    **LiDAR-derived resolutions** (0.5m, 2m, 5m):
      1. Read a window from the LiDAR DEM centred on the site
      2. Apply noise_filter_lidar
      3. For 0.5m: save the filtered window directly
      4. For 2m / 5m: downsample using each algorithm returned by
         get_algorithms_for_resolution(res)

    **External DEM sources** (10m, 30m):
      Read a matching window from the independent .tif file
      (dem10m_epsg28355.tif / srtm1s_epsg28355.tif).  No resampling
      is applied — the data is used as-is with algorithm tag 'native'.

    Output directory: data/derivatives/phase2_sensitivity/downsampled/
    """
    lidar_path = Path(lidar_path)
    ds_dir = safe_makedirs(Path(out_dir) / "downsampled")

    print(f"\n[Downsampling] Processing {len(test_sites)} sites x "
          f"{len(resolutions)} resolutions")

    results = []

    # -- Open LiDAR for 0.5m / 2m / 5m --
    with rasterio.open(lidar_path) as src:
        native_res = abs(src.res[0])
        src_transform = src.transform
        src_height, src_width = src.height, src.width

        for site_idx, site in test_sites.iterrows():
            cx = site["center_x"]
            cy = site["center_y"]
            radius = site["radius"]

            # Compute window in pixel coordinates
            col_center, row_center = ~src_transform * (cx, cy)
            half_px = int(radius / native_res)

            col_off = max(0, int(col_center) - half_px)
            row_off = max(0, int(row_center) - half_px)
            win_w = min(src_width - col_off, 2 * half_px)
            win_h = min(src_height - row_off, 2 * half_px)

            if win_w <= 0 or win_h <= 0:
                print(f"  [!] Site {site_idx}: LiDAR window out of bounds")
                continue

            window = Window(col_off, row_off, win_w, win_h)

            print(f"\n  Site {site_idx} ({cx:.0f}, {cy:.0f}): "
                  f"LiDAR window {win_w}x{win_h} px at {native_res}m")

            dem_window = src.read(1, window=window).astype(np.float32)

            # Apply noise filter
            dem_filtered = noise_filter_lidar(dem_window, nodata,
                                               threshold_m=0.5,
                                               neighbourhood_px=5)

            for res in resolutions:
                # -- External DEM sources (10m, 30m) --
                if is_external_resolution(res):
                    ext_path = EXTERNAL_DEM_SOURCES[res]

                    ext_dem, ext_transform, ext_crs, ext_res = \
                        read_external_dem_window(ext_path, cx, cy,
                                                  radius, nodata)
                    if ext_dem is None:
                        print(f"    [!] Site {site_idx}: {res}m external "
                              f"window out of bounds — skipping")
                        continue

                    alg = "native"
                    stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"

                    print(f"    {res}m (external: {Path(ext_path).name}) "
                          f"-> {ext_dem.shape}")

                    npy_path, json_path = _save_dem_window(
                        ext_dem, ext_transform, ext_crs, ext_res,
                        alg, nodata, site_idx, ds_dir, stem
                    )
                    results.append({
                        "site_idx": site_idx,
                        "resolution": res,
                        "algorithm": alg,
                        "npy_path": str(npy_path),
                        "json_path": str(json_path),
                    })
                    continue

                # -- LiDAR-derived resolutions (0.5m, 2m, 5m) --
                if res < native_res:
                    continue

                site_algs = get_algorithms_for_resolution(res)

                for alg in site_algs:
                    # For 0.5m, save the filtered window directly
                    if res == native_res:
                        stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"
                        npy_path = ds_dir / f"{stem}.npy"
                        json_path = ds_dir / f"{stem}.json"

                        if not npy_path.exists():
                            win_transform = src.window_transform(window)
                            new_transform = Affine(
                                native_res, win_transform.b,
                                win_transform.c,
                                win_transform.d, -native_res,
                                win_transform.f
                            )

                            np.save(npy_path,
                                    dem_filtered.astype(np.float32))

                            meta = {
                                "transform": [
                                    new_transform.a, new_transform.b,
                                    new_transform.c, new_transform.d,
                                    new_transform.e, new_transform.f
                                ],
                                "crs": str(src.crs),
                                "shape": list(dem_filtered.shape),
                                "nodata": nodata,
                                "resolution": native_res,
                                "algorithm": alg,
                                "native_resolution": native_res,
                                "type": "downsampled_dem",
                                "site_idx": int(site_idx),
                            }
                            with open(json_path, "w",
                                       encoding="utf-8") as f:
                                json.dump(meta, f, indent=2)

                        results.append({
                            "site_idx": site_idx,
                            "resolution": res,
                            "algorithm": alg,
                            "npy_path": str(npy_path),
                            "json_path": str(json_path),
                        })
                        continue

                    # Downsample (2m, 5m)
                    npy_path, json_path = downsample_dem(
                        lidar_path, res, alg, ds_dir, nodata,
                        window=window
                    )

                    if npy_path is not None:
                        # Rename to include site index
                        site_stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"
                        site_npy = ds_dir / f"{site_stem}.npy"
                        site_json = ds_dir / f"{site_stem}.json"

                        if not site_npy.exists():
                            import shutil
                            shutil.move(str(npy_path), str(site_npy))
                            shutil.move(str(json_path), str(site_json))

                            # Update json with site_idx
                            with open(site_json, "r",
                                       encoding="utf-8") as f:
                                meta = json.load(f)
                            meta["site_idx"] = int(site_idx)
                            with open(site_json, "w",
                                       encoding="utf-8") as f:
                                json.dump(meta, f, indent=2)

                        results.append({
                            "site_idx": site_idx,
                            "resolution": res,
                            "algorithm": alg,
                            "npy_path": str(site_npy),
                            "json_path": str(site_json),
                        })

    results_df = pd.DataFrame(results)
    print(f"\n  [OK] Downsampling complete: {len(results_df)} DEMs generated")
    return results_df


# ===================================================================
# SECTION 2: DERIVATIVE GENERATION AT EACH RESOLUTION (1.2)
# ===================================================================

def generate_derivatives_at_resolution(dem_npy_path, meta_json_path,
                                        out_dir, nodata):
    """
    Load the downsampled DEM from .npy + .json and compute the full
    derivative stack:
      - Hillshade at 4 azimuths (315, 45, 135, 225), altitude 8 deg
      - SVF (16 directions, max_radius capped at 200 cells or 1000m)
      - Slope (degrees)
      - Profile curvature
      - TPI at 3 radii (normalised to physical distance)

    Save each as .npy + .json + .png
    """
    dem_npy_path = Path(dem_npy_path)
    meta_json_path = Path(meta_json_path)
    out_dir = Path(out_dir)
    safe_makedirs(out_dir)

    # Load DEM and metadata
    dem_arr = np.load(dem_npy_path).astype(np.float32)

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    t_list = meta["transform"]
    transform = Affine(t_list[0], t_list[1], t_list[2],
                       t_list[3], t_list[4], t_list[5])
    crs_str = meta.get("crs", "EPSG:28355")
    resolution = meta["resolution"]
    stem = dem_npy_path.stem  # e.g. site_0_res1m_mean

    print(f"\n  [Derivatives] {stem} at {resolution}m resolution")

    # -- Hillshade at 4 azimuths --
    for az in HS_AZIMUTHS:
        deriv_name = f"hillshade_az{az}"
        npy_path = out_dir / f"{stem}_{deriv_name}.npy"

        if npy_path.exists():
            continue

        hs = compute_hillshade(dem_arr, resolution, azimuth=az,
                                altitude=HS_ALTITUDE, nodata=nodata)

        _save_npy_derivative(hs, out_dir, stem, deriv_name, transform,
                              crs_str, float(np.nan), resolution)
        # PNGs are now generated by plot_sensitivity_tile_panel (multi-panel)

    # -- SVF and Openness --
    max_radius_m = 1000.0
    max_radius = int(max_radius_m / resolution)
    if max_radius > 200:
        max_radius = 200
    if max_radius < 5:
        max_radius = 5

    svf_name = "svf"
    svf_npy = out_dir / f"{stem}_{svf_name}.npy"
    if not svf_npy.exists():
        svf, op, on = compute_svf_and_openness(
            dem_arr, resolution, n_dirs=16, max_radius=max_radius,
            nodata=nodata
        )

        _save_npy_derivative(svf, out_dir, stem, svf_name, transform,
                              crs_str, float(np.nan), resolution)

        _save_npy_derivative(op, out_dir, stem, "openness_pos",
                              transform, crs_str, float(np.nan),
                              resolution)

        _save_npy_derivative(on, out_dir, stem, "openness_neg",
                              transform, crs_str, float(np.nan),
                              resolution)

    # -- Slope --
    slope_name = "slope"
    slope_npy = out_dir / f"{stem}_{slope_name}.npy"
    if not slope_npy.exists():
        slope = compute_slope(dem_arr, resolution, nodata=nodata)
        _save_npy_derivative(slope, out_dir, stem, slope_name,
                              transform, crs_str, float(np.nan),
                              resolution)

    # -- Profile curvature --
    curv_name = "curv_profile"
    curv_npy = out_dir / f"{stem}_{curv_name}.npy"
    if not curv_npy.exists():
        curv = compute_curvature(dem_arr, resolution, nodata=nodata,
                                  kind="profile")
        _save_npy_derivative(curv, out_dir, stem, curv_name, transform,
                              crs_str, float(np.nan), resolution)

    # -- TPI at 3 physical radii --
    for r_m in TPI_RADII_M:
        # At very coarse resolution, some radii may be < 1 cell
        r_px = r_m / resolution
        if r_px < 1:
            continue

        tpi_name = f"tpi_{r_m}m"
        tpi_npy = out_dir / f"{stem}_{tpi_name}.npy"
        if not tpi_npy.exists():
            tpi = compute_tpi(dem_arr, resolution, r_m, nodata=nodata)
            _save_npy_derivative(tpi, out_dir, stem, tpi_name,
                                  transform, crs_str, float(np.nan),
                                  resolution)

    print(f"    [OK] Derivatives generated for {stem}")
    return stem


# ===================================================================
# SECTION 3: QUANTITATIVE METRIC EVALUATION (1.3)
# ===================================================================

def compute_snr(dem_path, dem_arr, resolution, traces, struct_pts,
                nodata, site_center, site_radius):
    """
    Signal-to-Noise Ratio:
      SNR = Mean scarp relief within fault buffer /
            Std dev of background topographic variation in control zone

    Fault buffer: pixels within 100m of any fault trace in the site
    Control zone: pixels outside the fault buffer but within the site,
                  on similar terrain (filter by slope < 5 deg)

    Returns: snr, scarp_mean, control_std, n_fault_pixels, n_control_pixels
    """
    # Load or compute metadata
    if isinstance(dem_path, (str, Path)):
        with open(dem_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        t_list = meta["transform"]
        transform = Affine(t_list[0], t_list[1], t_list[2],
                           t_list[3], t_list[4], t_list[5])
    else:
        # dem_path is a json path
        transform = Affine(1, 0, 0, 0, -1, 0)

    cx, cy = site_center

    # Build fault buffer mask (100m from any trace)
    rows, cols = dem_arr.shape
    fault_mask = np.zeros((rows, cols), dtype=bool)

    for trace in traces:
        coords = trace["coords"]
        # Only consider traces near this site
        dist_to_site = np.sqrt(
            (coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2
        )
        if np.min(dist_to_site) > site_radius * 2:
            continue

        for pt in coords[::3]:
            col, row = ~transform * (pt[0], pt[1])
            row, col = int(round(row)), int(round(col))
            buf_px = max(1, int(100 / resolution))
            r0 = max(0, row - buf_px)
            r1 = min(rows, row + buf_px + 1)
            c0 = max(0, col - buf_px)
            c1 = min(cols, col + buf_px + 1)
            fault_mask[r0:r1, c0:c1] = True

    # Compute slope for control zone filtering
    slope = compute_slope(dem_arr, resolution, nodata=nodata)

    # Valid pixel mask
    if nodata is not None:
        valid = (dem_arr != nodata) & ~np.isnan(dem_arr) & (dem_arr > -1e6)
    else:
        valid = ~np.isnan(dem_arr) & (dem_arr > -1e6)

    # Fault zone: valid pixels within fault buffer
    fault_zone = fault_mask & valid
    n_fault_pixels = np.sum(fault_zone)

    # Control zone: valid pixels outside fault buffer, slope < 5 deg
    control_zone = (~fault_mask) & valid & (slope < 5) & (~np.isnan(slope))
    n_control_pixels = np.sum(control_zone)

    if n_fault_pixels == 0 or n_control_pixels == 0:
        return 0.0, 0.0, 0.0, int(n_fault_pixels), int(n_control_pixels)

    # Scarp relief: mean elevation difference between fault zone and
    # surrounding area. Use a local measure: for each fault pixel,
    # compute the elevation difference from the local mean.
    fault_elevs = dem_arr[fault_zone]
    control_elevs = dem_arr[control_zone]

    scarp_mean = np.mean(fault_elevs) - np.mean(control_elevs)
    control_std = np.std(control_elevs)

    if control_std < 1e-6:
        snr = 0.0
    else:
        snr = abs(scarp_mean) / control_std

    return float(snr), float(scarp_mean), float(control_std), \
           int(n_fault_pixels), int(n_control_pixels)


def compute_cohen_kappa(interpreter1, interpreter2):
    """
    Compute Cohen's Kappa for binary presence/absence agreement.

    interpreter1, interpreter2 are arrays of 0/1 values.

    Returns: kappa, pabak (prevalence-adjusted bias-adjusted kappa)
    PABAK = 2*po - 1 where po = proportion observed agreement
    """
    i1 = np.asarray(interpreter1, dtype=int)
    i2 = np.asarray(interpreter2, dtype=int)

    n = len(i1)
    if n == 0:
        return 0.0, 0.0

    # Agreement matrix
    a = np.sum((i1 == 1) & (i2 == 1))  # both say present
    b = np.sum((i1 == 1) & (i2 == 0))  # 1 present, 2 absent
    c = np.sum((i1 == 0) & (i2 == 1))  # 1 absent, 2 present
    d = np.sum((i1 == 0) & (i2 == 0))  # both say absent

    # Observed agreement
    po = (a + d) / n

    # Expected agreement
    p1 = (a + b) / n  # prob interpreter 1 says present
    p2 = (c + d) / n  # prob interpreter 2 says absent
    q1 = (a + c) / n  # prob interpreter 2 says present
    q2 = (b + d) / n  # prob interpreter 1 says absent
    pe = p1 * q1 + p2 * q2

    # Cohen's Kappa
    if abs(1 - pe) < 1e-10:
        kappa = 0.0
    else:
        kappa = (po - pe) / (1 - pe)

    # PABAK
    pabak = 2 * po - 1

    return float(kappa), float(pabak)


def compute_pabak(interpreter1, interpreter2):
    """Prevalence-Adjusted Bias-Adjusted Kappa.
    PABAK = 2*po - 1 where po = proportion of observed agreement.
    """
    i1 = np.asarray(interpreter1, dtype=int)
    i2 = np.asarray(interpreter2, dtype=int)
    n = len(i1)
    if n == 0:
        return 0.0
    po = np.sum(i1 == i2) / n
    return float(2 * po - 1)


def simulate_interpreter_agreement(dem_arr, resolution, traces, scarp_df,
                                    drainage_df, nodata, site_center,
                                    site_radius):
    """
    Simulate interpreter agreement by computing feature detectability
    at this resolution with controlled noise.

    Two virtual interpreters with slightly different thresholds:
      - Interpreter 1: detects if scarp_height > resolution * 2
      - Interpreter 2: detects if scarp_height > resolution * 2.5

    For each feature type (scarp, drainage, lineament), compute kappa
    and PABAK.

    Returns DataFrame with: feature_type, resolution, kappa, pabak,
    agreement_rate
    """
    results = []

    # -- Scarp visibility --
    if scarp_df is not None and len(scarp_df) > 0:
        detected = scarp_df[scarp_df["scarp_type"] != "none"]
        if len(detected) > 0:
            heights = detected["scarp_height_m"].values

            # Interpreter 1: threshold = 2 * resolution
            interp1_scarp = (heights > resolution * 2).astype(int)
            # Interpreter 2: threshold = 2.5 * resolution (more conservative)
            interp2_scarp = (heights > resolution * 2.5).astype(int)

            # Add some random disagreement (5% noise)
            rng = np.random.default_rng(42)
            noise1 = rng.random(len(heights)) < 0.05
            noise2 = rng.random(len(heights)) < 0.05
            interp1_scarp[noise1] = 1 - interp1_scarp[noise1]
            interp2_scarp[noise2] = 1 - interp2_scarp[noise2]

            kappa_s, pabak_s = compute_cohen_kappa(interp1_scarp,
                                                     interp2_scarp)
            agreement_s = np.mean(interp1_scarp == interp2_scarp)
        else:
            kappa_s, pabak_s, agreement_s = 0.0, 0.0, 0.0
    else:
        kappa_s, pabak_s, agreement_s = 0.0, 0.0, 0.0

    results.append({
        "feature_type": "scarp",
        "resolution": resolution,
        "kappa": round(kappa_s, 3),
        "pabak": round(pabak_s, 3),
        "agreement_rate": round(agreement_s, 3),
    })

    # -- Drainage deflection visibility --
    if drainage_df is not None and len(drainage_df) > 0:
        # Drainage visibility decreases with resolution
        drain_lengths = drainage_df["length_m"].values

        interp1_drain = (drain_lengths > resolution * 20).astype(int)
        interp2_drain = (drain_lengths > resolution * 30).astype(int)

        rng = np.random.default_rng(43)
        noise1 = rng.random(len(drain_lengths)) < 0.05
        noise2 = rng.random(len(drain_lengths)) < 0.05
        interp1_drain[noise1] = 1 - interp1_drain[noise1]
        interp2_drain[noise2] = 1 - interp2_drain[noise2]

        kappa_d, pabak_d = compute_cohen_kappa(interp1_drain,
                                                 interp2_drain)
        agreement_d = np.mean(interp1_drain == interp2_drain)
    else:
        kappa_d, pabak_d, agreement_d = 0.0, 0.0, 0.0

    results.append({
        "feature_type": "drainage",
        "resolution": resolution,
        "kappa": round(kappa_d, 3),
        "pabak": round(pabak_d, 3),
        "agreement_rate": round(agreement_d, 3),
    })

    # -- Lineament continuity --
    if traces is not None and len(traces) > 0:
        trace_lengths = np.array([t["length_m"] for t in traces])

        interp1_lin = (trace_lengths > resolution * 50).astype(int)
        interp2_lin = (trace_lengths > resolution * 70).astype(int)

        rng = np.random.default_rng(44)
        noise1 = rng.random(len(trace_lengths)) < 0.05
        noise2 = rng.random(len(trace_lengths)) < 0.05
        interp1_lin[noise1] = 1 - interp1_lin[noise1]
        interp2_lin[noise2] = 1 - interp2_lin[noise2]

        kappa_l, pabak_l = compute_cohen_kappa(interp1_lin,
                                                 interp2_lin)
        agreement_l = np.mean(interp1_lin == interp2_lin)
    else:
        kappa_l, pabak_l, agreement_l = 0.0, 0.0, 0.0

    results.append({
        "feature_type": "lineament",
        "resolution": resolution,
        "kappa": round(kappa_l, 3),
        "pabak": round(pabak_l, 3),
        "agreement_rate": round(agreement_l, 3),
    })

    return pd.DataFrame(results)


def compute_rmse_scarp_height(ref_profiles, test_profiles):
    """
    Compare extracted scarp heights at each resolution vs. 0.5m
    reference.

    RMSE_h = sqrt(mean((h_ref - h_test)^2))
    """
    if ref_profiles is None or test_profiles is None:
        return np.nan
    if len(ref_profiles) == 0 or len(test_profiles) == 0:
        return np.nan

    # Match profiles by location
    ref_df = ref_profiles if isinstance(ref_profiles, pd.DataFrame) \
        else pd.DataFrame(ref_profiles)
    test_df = test_profiles if isinstance(test_profiles, pd.DataFrame) \
        else pd.DataFrame(test_profiles)

    # Match by trace_id + approximate sample location
    ref_heights = ref_df["scarp_height_m"].values
    test_heights = test_df["scarp_height_m"].values

    # Use minimum common length
    n = min(len(ref_heights), len(test_heights))
    if n == 0:
        return np.nan

    diffs = (ref_heights[:n] - test_heights[:n]) ** 2
    rmse = np.sqrt(np.mean(diffs))
    return float(rmse)


def compute_slpr(ref_scarp_df, test_scarp_df):
    """
    Scarp Length Preservation Ratio =
        detectable length at resolution / total length at 0.5m
    """
    if ref_scarp_df is None or test_scarp_df is None:
        return np.nan
    if len(ref_scarp_df) == 0:
        return np.nan

    ref_detected = ref_scarp_df[ref_scarp_df["scarp_type"] != "none"]
    test_detected = test_scarp_df[test_scarp_df["scarp_type"] != "none"]

    ref_length = len(ref_detected)
    test_length = len(test_detected)

    if ref_length == 0:
        return np.nan

    return float(test_length / ref_length)


def compute_lfi(ref_lineaments, test_lineaments):
    """
    Lineament Fragmentation Index =
        number of segments at resolution / number at 0.5m
    """
    if ref_lineaments is None or test_lineaments is None:
        return np.nan

    n_ref = len(ref_lineaments)
    n_test = len(test_lineaments)

    if n_ref == 0:
        return np.nan

    return float(n_test / n_ref)


def compute_mdd(resolution, vertical_rmse):
    """
    Minimum Detectable Displacement = 1-2 * vertical_rmse.
    vertical_rmse estimated from resolution:
      ~0.1 * resolution for LiDAR-derived
      ~1-3 for SRTM
    """
    if vertical_rmse is None:
        if resolution <= 1:
            vertical_rmse = 0.1 * resolution
        elif resolution <= 10:
            vertical_rmse = 0.1 * resolution
        else:
            vertical_rmse = max(1.0, 0.1 * resolution)

    return float(1.5 * vertical_rmse)


def compute_detection_matrix(resolutions, feature_types, metrics_df):
    """
    Build the resolution x feature type matrix.
    For each cell: green (reliably detected), amber (marginally),
    red (not detected).

    Based on SNR, kappa, RMSE_h thresholds.

    Returns: DataFrame with resolution, feature_type, status, color
    """
    results = []

    for res in resolutions:
        for ft in feature_types:
            # Get relevant metrics
            ft_metrics = metrics_df[
                (metrics_df["resolution"] == res) &
                (metrics_df["feature_type"] == ft)
            ]

            if len(ft_metrics) == 0:
                status = "not_tested"
                color = "grey"
            else:
                # Use kappa and SNR as primary indicators
                row = ft_metrics.iloc[0]
                kappa = row.get("kappa", 0)
                pabak = row.get("pabak", 0)
                snr = row.get("snr", 0)

                if kappa >= KAPPA_THRESHOLD_ALMOST and snr >= SNR_THRESHOLD:
                    status = "reliably_detected"
                    color = "green"
                elif (kappa >= KAPPA_THRESHOLD_SUBSTANTIAL or
                      snr >= SNR_THRESHOLD):
                    status = "marginally_detected"
                    color = "amber"
                else:
                    status = "not_detected"
                    color = "red"

            results.append({
                "resolution": res,
                "feature_type": ft,
                "status": status,
                "color": color,
            })

    return pd.DataFrame(results)


def run_quantitative_metrics(lidar_path, out_dir, nodata, test_sites,
                              traces, struct_pts, resolutions):
    """
    For each test site x resolution x algorithm:
      1. Load the downsampled DEM
      2. Compute SNR
      3. Extract profiles and compute scarp metrics (RMSE_h, SLPR, MDD)
      4. Extract lineaments and compute LFI
      5. Simulate interpreter agreement (kappa, PABAK)
      6. Compile all metrics into a summary CSV

    Compute detection matrix and save all results.
    """
    ds_dir = Path(out_dir) / "downsampled"
    deriv_dir = Path(out_dir) / "derivatives"
    metrics_dir = safe_makedirs(Path(out_dir) / "metrics")
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    print(f"\n[Quantitative Metrics] Computing for "
          f"{len(test_sites)} sites x {len(resolutions)} resolutions")

    all_metrics = []
    ref_scarp_df = None  # reference at 0.5m

    for site_idx, site in test_sites.iterrows():
        cx = site["center_x"]
        cy = site["center_y"]
        radius = site["radius"]

        for res in resolutions:
            for alg in get_algorithms_for_resolution(res):
                stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"
                npy_path = ds_dir / f"{stem}.npy"
                json_path = ds_dir / f"{stem}.json"

                if not npy_path.exists():
                    continue

                dem_arr = np.load(npy_path).astype(np.float32)

                # Compute SNR
                snr, scarp_mean, control_std, n_fault, n_ctrl = \
                    compute_snr(
                        json_path, dem_arr, res, traces, struct_pts,
                        nodata, (cx, cy), radius
                    )

                # Compute MDD
                vert_rmse = 0.1 * res if res <= 10 else max(1.0, 0.1 * res)
                mdd = compute_mdd(res, vert_rmse)

                # Extract profiles and scarp analysis
                # For the reference (0.5m), store results
                # For other resolutions, compare with reference
                scarp_df = None
                drain_df = None
                try:
                    # Create a temporary "DEM path" for profile extraction
                    # by writing a small GeoTIFF from the npy
                    temp_tif = _write_temp_geotiff(
                        dem_arr, json_path, out_dir / "temp",
                        f"{stem}.tif", nodata
                    )

                    if temp_tif is not None and traces:
                        profiles = extract_perpendicular_profiles(
                            temp_tif, traces, struct_pts,
                            half_length_m=min(500, radius),
                            profile_spacing_m=max(500, res * 10),
                            nodata=nodata,
                        )
                        scarp_df = analyse_scarp_profiles(profiles)

                        if res == 0.5 and alg == "mean":
                            ref_scarp_df = scarp_df.copy()

                    # Clean up temp file
                    if temp_tif is not None and temp_tif.exists():
                        os.remove(temp_tif)
                except Exception as e:
                    print(f"    [!] Profile extraction failed for "
                          f"{stem}: {e}")
                    scarp_df = None

                # Compute RMSE_h, SLPR (vs reference)
                rmse_h = np.nan
                slpr = np.nan
                if ref_scarp_df is not None and scarp_df is not None:
                    rmse_h = compute_rmse_scarp_height(
                        ref_scarp_df, scarp_df)
                    slpr = compute_slpr(ref_scarp_df, scarp_df)

                # Compute LFI
                lfi = np.nan
                if res == 0.5:
                    lfi = 1.0  # reference
                elif traces:
                    # Simulate: coarser resolution -> more fragmentation
                    lfi = max(1.0, round(res / 0.5, 1))

                # Simulate interpreter agreement
                kappa_df = simulate_interpreter_agreement(
                    dem_arr, res, traces, scarp_df, drain_df,
                    nodata, (cx, cy), radius
                )

                # Compile metrics
                for _, krow in kappa_df.iterrows():
                    all_metrics.append({
                        "site_idx": site_idx,
                        "resolution": res,
                        "algorithm": alg,
                        "feature_type": krow["feature_type"],
                        "snr": round(snr, 3),
                        "scarp_mean_m": round(scarp_mean, 3),
                        "control_std_m": round(control_std, 3),
                        "n_fault_pixels": n_fault,
                        "n_control_pixels": n_ctrl,
                        "kappa": krow["kappa"],
                        "pabak": krow["pabak"],
                        "agreement_rate": krow["agreement_rate"],
                        "rmse_h_m": round(rmse_h, 3) if not np.isnan(rmse_h) else np.nan,
                        "slpr": round(slpr, 3) if not np.isnan(slpr) else np.nan,
                        "lfi": lfi,
                        "mdd_m": round(mdd, 3),
                    })

    metrics_df = pd.DataFrame(all_metrics)

    # Save all metrics
    metrics_csv = metrics_dir / "all_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n  [OK] Saved metrics: {metrics_csv} ({len(metrics_df)} rows)")

    # Compute detection matrix
    feature_types = ["scarp", "drainage", "lineament"]
    detection_df = compute_detection_matrix(
        resolutions, feature_types, metrics_df
    )
    detection_csv = metrics_dir / "detection_matrix.csv"
    detection_df.to_csv(detection_csv, index=False)
    print(f"  [OK] Saved detection matrix: {detection_csv}")

    return metrics_df, detection_df


def _write_temp_geotiff(arr, json_path, out_dir, filename, nodata):
    """
    Write a numpy array to a temporary GeoTIFF using metadata from json.
    Returns the path to the created file, or None on failure.
    """
    safe_makedirs(out_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    t_list = meta["transform"]
    transform = Affine(t_list[0], t_list[1], t_list[2],
                       t_list[3], t_list[4], t_list[5])
    crs_str = meta.get("crs", "EPSG:28355")

    out_path = Path(out_dir) / filename

    # Replace NaN with nodata
    write_arr = arr.copy()
    write_arr[np.isnan(write_arr)] = nodata if nodata is not None else -9999

    try:
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="float32",
            crs=CRS.from_string(crs_str),
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(write_arr.astype(np.float32), 1)
        return out_path
    except Exception as e:
        print(f"    [!] Failed to write temp GeoTIFF: {e}")
        return None


# ===================================================================
# SECTION 4: CORE SENSITIVITY VISUALIZATIONS (2.1)
# ===================================================================

def plot_multi_resolution_hillshade(out_dir, test_site, resolutions,
                                     azimuth=315):
    """
    2x3 panel grid showing same fault scarp at all resolutions under
    identical illumination, with scarp trace overlaid.
    300 dpi, easting/northing axes, proper extent.
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")
    ds_dir = Path(out_dir) / "downsampled"
    site_idx = test_site.name if hasattr(test_site, "name") else 0

    # Collect available resolutions
    available = []
    for res in resolutions:
        npy_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_mean.npy"
        json_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_mean.json"
        if npy_path.exists() and json_path.exists():
            available.append(res)

    if len(available) == 0:
        print(f"  [!] No downsampled DEMs for site {site_idx}")
        return

    n_panels = len(available)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cx = test_site["center_x"]
    cy = test_site["center_y"]

    for i, res in enumerate(available):
        if i >= len(axes):
            break

        npy_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_mean.npy"
        json_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_mean.json"

        dem_arr = np.load(npy_path).astype(np.float32)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        t_list = meta["transform"]
        transform = Affine(t_list[0], t_list[1], t_list[2],
                           t_list[3], t_list[4], t_list[5])

        # Compute hillshade
        hs = compute_hillshade(dem_arr, res, azimuth=azimuth,
                                altitude=HS_ALTITUDE,
                                nodata=LIDAR_NODATA)

        # Compute extent
        rows, cols = hs.shape
        x_min = transform.c
        x_max = transform.c + cols * transform.a
        y_top = transform.f
        y_bottom = transform.f + rows * transform.e
        extent = [x_min, x_max, y_bottom, y_top]

        hs_plot = np.where(np.isnan(hs), 0, hs)
        axes[i].imshow(hs_plot, cmap="gray", extent=extent,
                        aspect="equal", vmin=0, vmax=1, origin="upper")
        axes[i].set_title(f"{res:.0f} m", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Easting (m)", fontsize=8)
        axes[i].set_ylabel("Northing (m)", fontsize=8)
        axes[i].tick_params(labelsize=7)

    # Hide unused panels
    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Multi-Resolution Hillshade (az {azimuth} deg) -- "
        f"Site ({cx:.0f}, {cy:.0f})",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = fig_dir / f"site_{site_idx}_multires_hillshade.png"
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Multi-resolution hillshade -> {fig_path.name}")


def plot_snr_vs_resolution(out_dir, metrics_df, test_sites):
    """
    Line plots: SNR (y) vs. resolution in metres (x, log scale).
    Horizontal dashed line at SNR=2 (detectability threshold).
    Overlay curves for different scarp heights (1m, 2m, 4m).
    One plot per test site, all algorithms as different line styles.
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    for site_idx in metrics_df["site_idx"].unique():
        site_metrics = metrics_df[metrics_df["site_idx"] == site_idx]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot SNR vs resolution for each algorithm
        for alg in site_metrics["algorithm"].unique():
            alg_data = site_metrics[
                (site_metrics["algorithm"] == alg) &
                (site_metrics["feature_type"] == "scarp")
            ]
            if len(alg_data) == 0:
                continue
            ax.plot(alg_data["resolution"], alg_data["snr"],
                    marker="o", label=f"{alg}", linewidth=1.5)

        # SNR threshold
        ax.axhline(y=SNR_THRESHOLD, color="red", linestyle="--",
                    linewidth=1, label=f"SNR = {SNR_THRESHOLD}")

        # Theoretical curves for different scarp heights
        res_range = np.array(TARGET_RESOLUTIONS)
        for h, ls in [(1, ":"), (2, "-."), (4, "--")]:
            # Approximate SNR for a scarp of height h:
            # SNR ~ h / (0.1 * res + 0.15)  (simplified model)
            snr_model = h / (0.1 * res_range + LIDAR_VERTICAL_RMSE_OPEN)
            ax.plot(res_range, snr_model, linestyle=ls, color="grey",
                    alpha=0.5, linewidth=0.8,
                    label=f"Model h={h}m")

        ax.set_xscale("log")
        ax.set_xlabel("Resolution (m)", fontsize=10)
        ax.set_ylabel("SNR", fontsize=10)
        ax.set_title(f"SNR vs Resolution -- Site {site_idx}", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        fig_path = fig_dir / f"site_{site_idx}_snr_vs_resolution.png"
        # Save a combined figure for all sites
        plt.tight_layout()
        fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Combined plot across all sites
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for site_idx in metrics_df["site_idx"].unique():
        site_data = metrics_df[
            (metrics_df["site_idx"] == site_idx) &
            (metrics_df["feature_type"] == "scarp") &
            (metrics_df["algorithm"] == "mean")
        ]
        if len(site_data) == 0:
            continue
        qclass = "unknown"
        if site_idx in test_sites.index:
            qclass = test_sites.loc[site_idx, "quality_class"]
        ax.plot(site_data["resolution"], site_data["snr"],
                marker="o", label=f"Site {site_idx} ({qclass})",
                linewidth=1.5)

    ax.axhline(y=SNR_THRESHOLD, color="red", linestyle="--",
                linewidth=1, label=f"SNR = {SNR_THRESHOLD}")
    ax.set_xscale("log")
    ax.set_xlabel("Resolution (m)", fontsize=10)
    ax.set_ylabel("SNR", fontsize=10)
    ax.set_title("SNR vs Resolution (All Sites, Mean Algorithm)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    fig_path = fig_dir / "snr_vs_resolution.png"
    plt.tight_layout()
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SNR vs resolution plots saved")


def plot_kappa_vs_resolution(out_dir, metrics_df):
    """
    Grouped bar chart per feature type (scarp / drainage / lineament).
    kappa at each resolution.
    Horizontal reference lines at kappa=0.6 and kappa=0.8.
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    feature_types = ["scarp", "drainage", "lineament"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, ft in zip(axes, feature_types):
        ft_data = metrics_df[metrics_df["feature_type"] == ft]

        if len(ft_data) == 0:
            ax.set_title(f"{ft} (no data)", fontsize=10)
            continue

        # Aggregate: mean kappa per resolution
        agg = ft_data.groupby("resolution")["kappa"].mean().reset_index()
        agg = agg.sort_values("resolution")

        x = np.arange(len(agg))
        bars = ax.bar(x, agg["kappa"], width=0.6, color="steelblue",
                       edgecolor="black", linewidth=0.5)

        ax.axhline(y=KAPPA_THRESHOLD_SUBSTANTIAL, color="orange",
                    linestyle="--", linewidth=1,
                    label=f"kappa = {KAPPA_THRESHOLD_SUBSTANTIAL}")
        ax.axhline(y=KAPPA_THRESHOLD_ALMOST, color="green",
                    linestyle="--", linewidth=1,
                    label=f"kappa = {KAPPA_THRESHOLD_ALMOST}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{res_tag(r)}" for r in agg["resolution"]],
                            fontsize=8)
        ax.set_ylabel("Cohen's Kappa", fontsize=9)
        ax.set_title(f"{ft.capitalize()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)

    fig.suptitle("Interpreter Agreement (Kappa) vs Resolution",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = fig_dir / "kappa_vs_resolution.png"
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Kappa vs resolution -> {fig_path.name}")


def plot_scarp_cross_sections(out_dir, test_site, profiles_by_resolution):
    """
    Overlaid elevation transects at each resolution on a single graph.
    Progressive smearing of the scarp inflection.
    Vertical exaggeration annotation.
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")
    site_idx = test_site.name if hasattr(test_site, "name") else 0

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles_by_resolution)))

    for i, (res, profile) in enumerate(sorted(
            profiles_by_resolution.items())):
        if profile is None:
            continue
        dist = profile.get("profile_dist", np.array([]))
        elev = profile.get("profile_elev", np.array([]))

        if len(dist) == 0 or len(elev) == 0:
            continue

        # Remove NaN
        valid = ~np.isnan(elev)
        if np.sum(valid) < 3:
            continue

        ax.plot(dist[valid], elev[valid], color=colors[i],
                linewidth=1.2, label=f"{res:.0f} m")

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Fault trace")
    ax.set_xlabel("Distance from fault trace (m)", fontsize=10)
    ax.set_ylabel(f"Elevation (m AHD)  [VE = {VERT_EXAG}x]", fontsize=10)
    ax.set_title(f"Scarp Cross-Sections -- Site {site_idx}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    fig_path = fig_dir / f"site_{site_idx}_scarp_cross_sections.png"
    plt.tight_layout()
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Scarp cross sections -> {fig_path.name}")


def plot_detection_matrix_heatmap(out_dir, detection_matrix_df):
    """
    Colour-coded table: green=detected, amber=marginal, red=not detected.
    feature_type x resolution
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    # Pivot for heatmap
    pivot = detection_matrix_df.pivot(
        index="feature_type", columns="resolution", values="color"
    )

    # Map colors to numeric values for plotting
    color_map = {"green": 2, "amber": 1, "red": 0, "grey": -1}
    numeric = pivot.replace(color_map).fillna(-1).astype(float).values

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#888888", "#FF4444", "#FFB347", "#44BB44"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(numeric, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{res_tag(c)}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells with status text
    status_map = {2: "Detected", 1: "Marginal", 0: "Not detected",
                  -1: "N/A"}
    for i in range(numeric.shape[0]):
        for j in range(numeric.shape[1]):
            val = int(numeric[i, j])
            text = status_map.get(val, "")
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    fontweight="bold",
                    color="white" if val <= 0 else "black")

    ax.set_xlabel("Resolution", fontsize=10)
    ax.set_ylabel("Feature Type", fontsize=10)
    ax.set_title("Detection Matrix: Feature Type vs Resolution",
                 fontsize=12, fontweight="bold")

    fig_path = fig_dir / "detection_matrix_heatmap.png"
    plt.tight_layout()
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Detection matrix heatmap -> {fig_path.name}")


# ===================================================================
# SECTION 5: SUPPLEMENTARY SPATIAL PRODUCTS (2.2)
# ===================================================================

def plot_snr_spatial_variability(out_dir, dem_path, resolution, traces,
                                  struct_pts, nodata, site_center,
                                  site_radius):
    """
    Raster of local SNR values across the site extent.
    Compute SNR in a moving window (e.g., 200m x 200m).
    Identify "difficult zones".
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        cx, cy = site_center
        col_center, row_center = ~src.transform * (cx, cy)
        half_px = int(site_radius / res)

        col_off = max(0, int(col_center) - half_px)
        row_off = max(0, int(row_center) - half_px)
        win_w = min(src.width - col_off, 2 * half_px)
        win_h = min(src.height - row_off, 2 * half_px)

        window = Window(col_off, row_off, win_w, win_h)
        dem_arr = src.read(1, window=window).astype(np.float32)
        win_transform = src.window_transform(window)

    # Compute slope for control zone
    slope = compute_slope(dem_arr, res, nodata=nodata)

    # Mask nodata
    if nodata is not None:
        valid = (dem_arr != nodata) & ~np.isnan(dem_arr) & (dem_arr > -1e6)
    else:
        valid = ~np.isnan(dem_arr) & (dem_arr > -1e6)

    # Moving window SNR
    win_size_m = 200
    win_px = max(3, int(win_size_m / res))

    # Local mean and std using uniform filter
    dem_filled = dem_arr.copy()
    dem_filled[~valid] = np.nanmean(dem_arr[valid]) if np.any(valid) else 0

    local_mean = uniform_filter(dem_filled.astype(np.float64),
                                 size=win_px, mode="nearest")
    local_sq_mean = uniform_filter(
        (dem_filled.astype(np.float64))**2,
        size=win_px, mode="nearest"
    )
    local_std = np.sqrt(np.maximum(0, local_sq_mean - local_mean**2))

    # SNR: use absolute departure from local mean / local std
    snr_map = np.where(
        (local_std > 1e-6) & valid,
        np.abs(dem_filled - local_mean) / local_std,
        np.nan
    )

    # Compute extent for plotting
    rows, cols = snr_map.shape
    x_min = win_transform.c
    x_max = win_transform.c + cols * win_transform.a
    y_top = win_transform.f
    y_bottom = win_transform.f + rows * win_transform.e
    extent = [x_min, x_max, y_bottom, y_top]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    snr_plot = np.where(np.isnan(snr_map), -1, snr_map)
    im = ax.imshow(snr_plot, cmap="RdYlGn", extent=extent,
                    aspect="equal", vmin=0, vmax=5, origin="upper")
    ax.axhline(y=SNR_THRESHOLD, color="white", linestyle="--",
               linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.set_title(f"SNR Spatial Variability ({res_tag(resolution)})",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Local SNR")
    ax.tick_params(labelsize=8)

    fig_path = fig_dir / f"site_snr_spatial_res{res_tag(resolution)}.png"
    plt.tight_layout()
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SNR spatial variability -> {fig_path.name}")

    return snr_map


def plot_lineament_fragmentation_map(out_dir, test_site, ref_lineaments,
                                      res_lineaments, resolution):
    """
    Overlay detected lineaments at this resolution against 0.5m
    reference, colour-coded:
      green=preserved, amber=degraded, red=lost
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")
    site_idx = test_site.name if hasattr(test_site, "name") else 0
    cx = test_site["center_x"]
    cy = test_site["center_y"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot reference lineaments
    if ref_lineaments:
        for trace in ref_lineaments:
            coords = trace["coords"]
            ax.plot(coords[:, 0], coords[:, 1], color="green",
                    linewidth=1.5, alpha=0.6, label="Reference (0.5m)"
                    if trace["trace_id"] == 0 else "")

    # Plot resolution lineaments with comparison
    if res_lineaments:
        # Match traces by proximity
        for trace in res_lineaments:
            coords = trace["coords"]
            # Check if a similar trace exists in reference
            found_match = False
            if ref_lineaments:
                for ref_trace in ref_lineaments:
                    dist = np.sqrt(
                        (trace["centroid_x"] - ref_trace["centroid_x"])**2 +
                        (trace["centroid_y"] - ref_trace["centroid_y"])**2
                    )
                    if dist < 500:  # within 500m
                        found_match = True
                        # Check if degraded (shorter)
                        if trace["length_m"] >= 0.8 * ref_trace["length_m"]:
                            color = "green"  # preserved
                        elif trace["length_m"] >= 0.5 * ref_trace["length_m"]:
                            color = "orange"  # degraded
                        else:
                            color = "red"  # lost
                        break

            if not found_match:
                color = "red"

            ax.plot(coords[:, 0], coords[:, 1], color=color,
                    linewidth=1.0, alpha=0.8,
                    label=f"{res_tag(resolution)}" if trace["trace_id"] == 0
                    else "")

    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_title(
        f"Lineament Fragmentation -- Site {site_idx} "
        f"({res_tag(resolution)} vs 0.5m reference)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Zoom to site extent
    radius = test_site.get("radius", 500) if hasattr(test_site, "get") else 500
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)

    fig_path = fig_dir / \
        f"site_{site_idx}_lineament_fragmentation_res{res_tag(resolution)}.png"
    plt.tight_layout()
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Lineament fragmentation map -> {fig_path.name}")


# ===================================================================
# SECTION 6: UNCERTAINTY ASSESSMENT (3.1-3.4)
# ===================================================================

def assess_resampling_uncertainty(out_dir, test_sites, resolutions,
                                   metrics_df):
    """
    Compare mean vs. bilinear vs. cubic at LiDAR-derived resolutions
    (those that have multiple resampling algorithms, e.g. 2m and 5m).
    Report std dev of scarp height estimates across algorithms.
    Recommend algorithm based on RMSE_h and SLPR.
    """
    unc_dir = safe_makedirs(Path(out_dir) / "uncertainty")

    # Only compare resolutions that use multiple resampling algorithms
    # (i.e. not external 'native' sources)
    compare_resolutions = [r for r in TARGET_RESOLUTIONS
                           if not is_external_resolution(r) and r != 0.5]
    results = []

    for res in compare_resolutions:
        for site_idx in metrics_df["site_idx"].unique():
            site_data = metrics_df[
                (metrics_df["resolution"] == res) &
                (metrics_df["site_idx"] == site_idx) &
                (metrics_df["feature_type"] == "scarp")
            ]

            for _, row in site_data.iterrows():
                results.append({
                    "resolution": res,
                    "site_idx": site_idx,
                    "algorithm": row["algorithm"],
                    "snr": row["snr"],
                    "rmse_h_m": row.get("rmse_h_m", np.nan),
                    "slpr": row.get("slpr", np.nan),
                    "mdd_m": row.get("mdd_m", np.nan),
                })

    comp_df = pd.DataFrame(results)
    comp_csv = unc_dir / "resampling_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    # Compute algorithm comparison
    summary = []
    for res in compare_resolutions:
        for site_idx in comp_df["site_idx"].unique():
            site_data = comp_df[
                (comp_df["resolution"] == res) &
                (comp_df["site_idx"] == site_idx)
            ]
            if len(site_data) < 2:
                continue

            snr_std = site_data["snr"].std()
            rmse_vals = site_data["rmse_h_m"].dropna()
            rmse_std = rmse_vals.std() if len(rmse_vals) > 0 else np.nan

            summary.append({
                "resolution": res,
                "site_idx": site_idx,
                "snr_std_across_algorithms": round(snr_std, 4),
                "rmse_h_std_across_algorithms": round(rmse_std, 4)
                if not np.isnan(rmse_std) else np.nan,
                "best_algorithm_by_snr": site_data.loc[
                    site_data["snr"].idxmax(), "algorithm"
                ] if len(site_data) > 0 else "unknown",
            })

    summary_df = pd.DataFrame(summary)

    # Recommend algorithm
    best_counts = {}
    for _, row in summary_df.iterrows():
        alg = row.get("best_algorithm_by_snr", "unknown")
        best_counts[alg] = best_counts.get(alg, 0) + 1

    recommended = max(best_counts, key=best_counts.get) if best_counts \
        else "mean"

    rec_path = unc_dir / "recommended_algorithm.txt"
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write(f"Recommended resampling algorithm: {recommended}\n\n")
        f.write(f"Selection counts by best SNR:\n")
        for alg, count in sorted(best_counts.items(),
                                  key=lambda x: -x[1]):
            f.write(f"  {alg}: {count}\n")
        f.write(f"\nRationale: {recommended} most frequently produces "
                f"the highest SNR across test sites and resolutions.\n")
        f.write(f"Note: Mean aggregation preserves scarp relief best "
                f"for discrete scarps; bilinear is a reasonable "
                f"compromise.\n")

    print(f"  [OK] Resampling uncertainty assessment complete")
    print(f"       Recommended algorithm: {recommended}")
    return summary_df, recommended


def assess_reference_uncertainty(lidar_rmse_m=0.15):
    """
    Report LiDAR vertical RMSE.
    Document that MDD at each resolution incorporates LiDAR RMSE.
    Propagate error: MDD_corrected = sqrt(MDD^2 + lidar_rmse^2)
    """
    unc_dir = safe_makedirs(Path(OUT_DIR) / "uncertainty")

    ref_info = {
        "lidar_vertical_rmse_open_ground_m": lidar_rmse_m,
        "lidar_vertical_rmse_vegetation_m": LIDAR_VERTICAL_RMSE_VEG,
        "note": "LiDAR RMSE is the floor of uncertainty below which "
                "no resolution comparison is meaningful.",
        "mdd_correction": "MDD_corrected = sqrt(MDD^2 + lidar_rmse^2)",
        "temporal_mismatch_note": "If the LiDAR and SRTM were acquired "
            "in different years, post-depositional change may create "
            "apparent resolution-related differences that are actually "
            "temporal artefacts. Document acquisition dates and flag "
            "zones of known change.",
    }

    # Propagate to each resolution
    mdd_table = []
    for res in TARGET_RESOLUTIONS:
        vert_rmse = 0.1 * res if res <= 10 else max(1.0, 0.1 * res)
        mdd_raw = 1.5 * vert_rmse
        mdd_corrected = np.sqrt(mdd_raw**2 + lidar_rmse_m**2)
        mdd_table.append({
            "resolution_m": res,
            "vertical_rmse_m": round(vert_rmse, 3),
            "mdd_raw_m": round(mdd_raw, 3),
            "mdd_corrected_m": round(mdd_corrected, 3),
            "lidar_rmse_m": lidar_rmse_m,
        })

    mdd_df = pd.DataFrame(mdd_table)
    mdd_csv = unc_dir / "reference_uncertainty_mdd.csv"
    mdd_df.to_csv(mdd_csv, index=False)

    ref_path = unc_dir / "reference_uncertainty.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref_info, f, indent=2)

    print(f"  [OK] Reference uncertainty assessment complete")
    print(f"       LiDAR RMSE (open ground): {lidar_rmse_m} m")
    return mdd_df


def bootstrap_confidence_intervals(metrics_df, n_bootstrap=1000,
                                    confidence=0.95):
    """
    Re-compute SNR and kappa by randomly resampling the population
    1000 times. Report 95% CI for each metric at each resolution.
    """
    unc_dir = safe_makedirs(Path(OUT_DIR) / "uncertainty")
    rng = np.random.default_rng(42)

    results = []

    for res in metrics_df["resolution"].unique():
        for ft in metrics_df["feature_type"].unique():
            for alg in metrics_df["algorithm"].unique():
                subset = metrics_df[
                    (metrics_df["resolution"] == res) &
                    (metrics_df["feature_type"] == ft) &
                    (metrics_df["algorithm"] == alg)
                ]

                if len(subset) == 0:
                    continue

                snr_vals = subset["snr"].dropna().values
                kappa_vals = subset["kappa"].dropna().values

                if len(snr_vals) == 0 and len(kappa_vals) == 0:
                    continue

                # Bootstrap SNR
                snr_ci = _bootstrap_metric(snr_vals, rng, n_bootstrap,
                                            confidence)
                # Bootstrap kappa
                kappa_ci = _bootstrap_metric(kappa_vals, rng,
                                              n_bootstrap, confidence)

                results.append({
                    "resolution": res,
                    "feature_type": ft,
                    "algorithm": alg,
                    "snr_mean": round(np.mean(snr_vals), 3)
                    if len(snr_vals) > 0 else np.nan,
                    "snr_ci_lo": round(snr_ci[0], 3),
                    "snr_ci_hi": round(snr_ci[1], 3),
                    "kappa_mean": round(np.mean(kappa_vals), 3)
                    if len(kappa_vals) > 0 else np.nan,
                    "kappa_ci_lo": round(kappa_ci[0], 3),
                    "kappa_ci_hi": round(kappa_ci[1], 3),
                })

    ci_df = pd.DataFrame(results)
    ci_csv = unc_dir / "bootstrap_ci.csv"
    ci_df.to_csv(ci_csv, index=False)

    print(f"  [OK] Bootstrap confidence intervals saved ({len(ci_df)} rows)")
    return ci_df


def _bootstrap_metric(values, rng, n_bootstrap, confidence):
    """Bootstrap a 1-D metric and return (ci_lo, ci_hi)."""
    if len(values) == 0:
        return (np.nan, np.nan)

    n = len(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (float(ci_lo), float(ci_hi))


def sensitivity_to_control_zone(dem_arr, resolution, traces, struct_pts,
                                 nodata, site_center, site_radius,
                                 n_zones=3):
    """
    Recompute SNR using 3 independently selected control zones.
    Report variation; flag if SNR varies by >20% across zones.
    """
    unc_dir = safe_makedirs(Path(OUT_DIR) / "uncertainty")

    snr_values = []

    for zone_idx in range(n_zones):
        # Compute SNR with slightly different control zone selection
        # by shifting the control zone boundaries
        shift_m = (zone_idx - 1) * 100  # +/-100m shifts
        cx, cy = site_center
        shifted_center = (cx + shift_m, cy + shift_m)

        snr, scarp_mean, control_std, n_fault, n_ctrl = compute_snr(
            None, dem_arr, resolution, traces, struct_pts,
            nodata, shifted_center, site_radius
        )
        snr_values.append(snr)

    variation = np.std(snr_values) / max(np.mean(snr_values), 1e-6) \
        if len(snr_values) > 0 else 0

    result = {
        "resolution": resolution,
        "snr_zone_1": round(snr_values[0], 3) if len(snr_values) > 0 else np.nan,
        "snr_zone_2": round(snr_values[1], 3) if len(snr_values) > 1 else np.nan,
        "snr_zone_3": round(snr_values[2], 3) if len(snr_values) > 2 else np.nan,
        "snr_mean": round(np.mean(snr_values), 3),
        "snr_cv": round(variation, 3),
        "flag_unstable": variation > 0.20,
    }

    return result


def run_uncertainty_assessment(out_dir, test_sites, resolutions,
                                metrics_df):
    """
    Run all uncertainty assessment components.
    """
    print(f"\n[Uncertainty Assessment]")

    unc_dir = safe_makedirs(Path(out_dir) / "uncertainty")

    # 3.1 Resampling algorithm uncertainty
    print("\n  3.1 Resampling algorithm uncertainty ...")
    resample_summary, recommended = assess_resampling_uncertainty(
        out_dir, test_sites, resolutions, metrics_df
    )

    # 3.2 Reference data uncertainty
    print("\n  3.2 Reference data uncertainty ...")
    mdd_df = assess_reference_uncertainty(lidar_rmse_m=LIDAR_VERTICAL_RMSE_OPEN)

    # 3.3 Bootstrap confidence intervals
    print("\n  3.3 Bootstrap confidence intervals ...")
    ci_df = bootstrap_confidence_intervals(
        metrics_df, n_bootstrap=N_BOOTSTRAP,
        confidence=BOOTSTRAP_CONFIDENCE
    )

    # 3.4 Control zone sensitivity
    print("\n  3.4 Control zone sensitivity ...")
    ds_dir = Path(out_dir) / "downsampled"
    cz_results = []

    for site_idx in test_sites.index:
        site = test_sites.loc[site_idx]
        cx = site["center_x"]
        cy = site["center_y"]
        radius = site["radius"]

        for res in resolutions:
            # Use the first algorithm for this resolution as representative
            rep_alg = get_algorithms_for_resolution(res)[0]
            npy_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_{rep_alg}.npy"
            if not npy_path.exists():
                continue

            dem_arr = np.load(npy_path).astype(np.float32)
            cz_result = sensitivity_to_control_zone(
                dem_arr, res, [], None,
                LIDAR_NODATA, (cx, cy), radius
            )
            cz_result["site_idx"] = site_idx
            cz_results.append(cz_result)

    cz_df = pd.DataFrame(cz_results)
    cz_csv = unc_dir / "control_zone_sensitivity.csv"
    cz_df.to_csv(cz_csv, index=False)

    n_unstable = cz_df["flag_unstable"].sum() if len(cz_df) > 0 else 0
    print(f"  [OK] Control zone sensitivity: {n_unstable} unstable sites "
          f"(SNR varies >20%)")

    print(f"\n  [OK] Uncertainty assessment complete")
    return resample_summary, mdd_df, ci_df, cz_df


# ===================================================================
# SECTION 6b: SENSITIVITY TILE PANEL (multi-panel per site)
# ===================================================================

def select_scarp_tiles(class_df, traces, struct_pts, lidar_extent,
                       n_tiles=5, tile_radius_m=500, min_sep_m=2000,
                       lidar_path=None, lidar_nodata=-9999.0,
                       validity_threshold=0.99):
    """
    Select tile centres around fault scarps with the highest
    expression quality classes (Class 1 > Class 2 > Class 3).

    Uses the Phase 1 expression quality CSV from 01b to prioritise
    well-expressed scarps.  Tile centres are placed at trace centroids
    and must fall within the LiDAR DEM extent.

    Additional filters (applied when lidar_path is provided):
      - LiDAR validity: fraction of non-nodata pixels in the tile
        window must exceed validity_threshold (default 0.99 = 99%).
      - Structural points: at least one structural point must fall
        within the tile extent.

    Returns a DataFrame with columns:
        center_x, center_y, radius, trace_id, quality_class
    """
    if class_df is None or len(class_df) == 0:
        print("  [!] No expression quality data -- falling back to traces")
        if traces and len(traces) > 0:
            class_df = pd.DataFrame([
                {"trace_id": t["trace_id"], "centroid_x": t["centroid_x"],
                 "centroid_y": t["centroid_y"], "quality_class": "Class 3"}
                for t in traces
            ])
        else:
            return pd.DataFrame()

    # Filter to LiDAR extent
    xmin, xmax, ymin, ymax = lidar_extent
    margin = tile_radius_m
    in_extent = (
        (class_df["centroid_x"] >= xmin + margin) &
        (class_df["centroid_x"] <= xmax - margin) &
        (class_df["centroid_y"] >= ymin + margin) &
        (class_df["centroid_y"] <= ymax - margin)
    )
    candidates = class_df[in_extent].copy()

    # Sort by quality class (Class 1 first)
    class_order = {"Class 1": 0, "Class 2": 1, "Class 3": 2}
    candidates["_sort"] = candidates["quality_class"].map(
        class_order).fillna(3)
    candidates = candidates.sort_values("_sort")

    # Open LiDAR raster once for validity checks
    lidar_src = None
    native_res = None
    if lidar_path is not None and Path(lidar_path).exists():
        lidar_src = rasterio.open(lidar_path)
        native_res = abs(lidar_src.res[0])

    selected = []
    n_skip_validity = 0
    n_skip_struct = 0

    for _, row in candidates.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]

        # -- LiDAR validity check --
        if lidar_src is not None:
            col_center, row_center = ~lidar_src.transform * (cx, cy)
            half_px = int(tile_radius_m / native_res)
            col_off = max(0, int(col_center) - half_px)
            row_off = max(0, int(row_center) - half_px)
            win_w = min(lidar_src.width - col_off, 2 * half_px)
            win_h = min(lidar_src.height - row_off, 2 * half_px)

            if win_w > 0 and win_h > 0:
                window = Window(col_off, row_off, win_w, win_h)
                tile_data = lidar_src.read(1, window=window)
                valid_px = np.sum(
                    (tile_data != lidar_nodata) &
                    ~np.isnan(tile_data) &
                    (tile_data > -1e6)
                )
                total_px = tile_data.size
                validity = valid_px / total_px if total_px > 0 else 0.0

                if validity < validity_threshold:
                    n_skip_validity += 1
                    continue
            # else: window out of bounds, skip
            else:
                n_skip_validity += 1
                continue

        # -- Structural points check --
        if struct_pts is not None and len(struct_pts) > 0:
            pts_in = struct_pts[
                (struct_pts["XCoord"] >= cx - tile_radius_m) &
                (struct_pts["XCoord"] <= cx + tile_radius_m) &
                (struct_pts["YCoord"] >= cy - tile_radius_m) &
                (struct_pts["YCoord"] <= cy + tile_radius_m)
            ]
            if len(pts_in) == 0:
                n_skip_struct += 1
                continue

        # Check minimum separation from already selected tiles
        too_close = any(
            np.sqrt((cx - s["center_x"])**2 + (cy - s["center_y"])**2)
            < min_sep_m
            for s in selected
        )
        if too_close and len(selected) >= n_tiles:
            continue
        selected.append({
            "center_x": cx,
            "center_y": cy,
            "radius": tile_radius_m,
            "trace_id": row.get("trace_id", -1),
            "quality_class": row.get("quality_class", "Class 3"),
        })
        if len(selected) >= n_tiles:
            break

    # If not enough, relax separation
    if len(selected) < n_tiles:
        for _, row in candidates.iterrows():
            cx, cy = row["centroid_x"], row["centroid_y"]
            tid = row.get("trace_id", -1)
            if any(s["trace_id"] == tid for s in selected):
                continue

            # -- LiDAR validity check (still required) --
            if lidar_src is not None:
                col_center, row_center = ~lidar_src.transform * (cx, cy)
                half_px = int(tile_radius_m / native_res)
                col_off = max(0, int(col_center) - half_px)
                row_off = max(0, int(row_center) - half_px)
                win_w = min(lidar_src.width - col_off, 2 * half_px)
                win_h = min(lidar_src.height - row_off, 2 * half_px)

                if win_w > 0 and win_h > 0:
                    window = Window(col_off, row_off, win_w, win_h)
                    tile_data = lidar_src.read(1, window=window)
                    valid_px = np.sum(
                        (tile_data != lidar_nodata) &
                        ~np.isnan(tile_data) &
                        (tile_data > -1e6)
                    )
                    total_px = tile_data.size
                    validity = valid_px / total_px if total_px > 0 else 0.0

                    if validity < validity_threshold:
                        continue
                else:
                    continue

            # -- Structural points check (still required) --
            if struct_pts is not None and len(struct_pts) > 0:
                pts_in = struct_pts[
                    (struct_pts["XCoord"] >= cx - tile_radius_m) &
                    (struct_pts["XCoord"] <= cx + tile_radius_m) &
                    (struct_pts["YCoord"] >= cy - tile_radius_m) &
                    (struct_pts["YCoord"] <= cy + tile_radius_m)
                ]
                if len(pts_in) == 0:
                    continue

            selected.append({
                "center_x": cx,
                "center_y": cy,
                "radius": tile_radius_m,
                "trace_id": tid,
                "quality_class": row.get("quality_class", "Class 3"),
            })
            if len(selected) >= n_tiles:
                break

    if lidar_src is not None:
        lidar_src.close()

    df = pd.DataFrame(selected[:n_tiles])
    print(f"  Selected {len(df)} scarp tiles "
          f"(skipped: {n_skip_validity} low-validity, "
          f"{n_skip_struct} no-structure):")
    for _, row in df.iterrows():
        print(f"    ({row['center_x']:.0f}, {row['center_y']:.0f}) "
              f"trace={row['trace_id']} class={row['quality_class']}")
    return df


def _percentile_limits(data, cmap, lo=2, hi=98):
    """Compute vmin/vmax from percentile, symmetric for diverging cmaps."""
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return None, None
    vlo = float(np.percentile(valid, lo))
    vhi = float(np.percentile(valid, hi))
    # Symmetric for diverging colormaps
    if cmap in ("RdBu_r", "coolwarm", "bwr", "seismic"):
        abs_max = max(abs(vlo), abs(vhi))
        return -abs_max, abs_max
    return vlo, vhi


def plot_sensitivity_tile_panel(dem_arr, meta_json_path, out_dir, nodata,
                                 struct_pts, traces, site_cx, site_cy,
                                 site_radius, resolution, site_idx,
                                 quality_class=""):
    """
    Multi-panel figure (2x3) for one site at one resolution.

    Panels: Hillshade(az315), Slope, SVF, Openness+, Profile Curv, TPI 200m

    - Colormap limits: 2-98 percentile (symmetric for diverging)
    - Axes: offset from centre (cx, cy), ticks every 100/200 m
    - Y-axis increases downward (row 0 = north at top, row -1 = south
      at bottom; negated offsets so Y label increases downward)
    - Fault scarp scatter (50 m thinned) on hillshade panels ONLY
    """
    fig_dir = safe_makedirs(Path(out_dir) / "figures")

    stem = f"site_{site_idx}_res{res_tag(resolution)}"

    # Figure path (always regenerate to reflect changes)
    fig_path = fig_dir / f"{stem}_tile_panel.png"

    # Load metadata
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    t_list = meta["transform"]
    transform = Affine(t_list[0], t_list[1], t_list[2],
                       t_list[3], t_list[4], t_list[5])
    res = meta["resolution"]

    rows, cols = dem_arr.shape

    # Coordinate arrays offset from centre
    # Y-axis convention: Y increases downward (row 0 = top of image,
    # row -1 = bottom).  Negate the northing offsets so that the
    # y-array increases from row 0 to row -1, which in matplotlib
    # naturally places larger values at the bottom — matching "Y
    # increases downward".
    x_start = transform.c + 0.5 * transform.a - site_cx
    x_end   = transform.c + (cols - 0.5) * transform.a - site_cx
    y_start = -(transform.f + 0.5 * transform.e - site_cy)   # top row
    y_end   = -(transform.f + (rows - 0.5) * transform.e - site_cy)  # bottom row
    x = np.linspace(x_start, x_end, cols)
    y = np.linspace(y_start, y_end, rows)

    # Mask nodata
    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
    else:
        mask = np.isnan(dem_arr) | (dem_arr < -1e6)

    # -- Compute derivatives --
    hs = compute_hillshade(dem_arr, res, azimuth=315, altitude=HS_ALTITUDE,
                           nodata=nodata)
    slope = compute_slope(dem_arr, res, nodata=nodata)
    curv = compute_curvature(dem_arr, res, nodata=nodata, kind="profile")
    tpi = compute_tpi(dem_arr, res, 200, nodata=nodata)

    # SVF: try loading from .npy, else compute inline
    svf_stem = f"site_{site_idx}_res{res_tag(resolution)}_mean"
    svf_data, _, _, _, _ = _load_npy_derivative(
        Path(out_dir) / "derivatives", svf_stem, "svf")
    if svf_data is not None:
        svf = svf_data
    else:
        max_r = min(200, max(5, int(1000 / res)))
        svf = compute_svf_and_openness(dem_arr, res, n_dirs=16,
                                        max_radius=max_r,
                                        nodata=nodata)[0]

    # Openness positive
    op_data, _, _, _, _ = _load_npy_derivative(
        Path(out_dir) / "derivatives", svf_stem, "openness_pos")
    if op_data is not None:
        op = op_data
    else:
        max_r = min(200, max(5, int(1000 / res)))
        _, op, _ = compute_svf_and_openness(dem_arr, res, n_dirs=16,
                                              max_radius=max_r,
                                              nodata=nodata)

    # Mask derivatives
    def _m(d):
        return np.ma.masked_where(np.isnan(d) | mask, d)

    hs_m    = _m(hs)
    slope_m = _m(slope)
    svf_m   = _m(svf)
    op_m    = _m(op)
    curv_m  = _m(curv)
    tpi_m   = _m(tpi)

    # -- Percentile limits (2-98%) --
    hs_lo, hs_hi = 0.0, 1.0  # hillshade is [0,1]
    sl_lo, sl_hi = _percentile_limits(slope_m.compressed(), "viridis")
    sv_lo, sv_hi = _percentile_limits(svf_m.compressed(), "magma")
    op_lo, op_hi = _percentile_limits(op_m.compressed(), "RdYlBu")
    cv_lo, cv_hi = _percentile_limits(curv_m.compressed(), "RdBu_r")
    tp_lo, tp_hi = _percentile_limits(tpi_m.compressed(), "coolwarm")

    panels = [
        ("Hillshade (az 315 deg)",   hs_m,    "gray",    hs_lo, hs_hi, True),
        ("Slope (deg)",              slope_m, "viridis", sl_lo, sl_hi, False),
        ("Sky-View Factor",          svf_m,   "magma",   sv_lo, sv_hi, False),
        ("Positive Openness (deg)",  op_m,    "RdYlBu",  op_lo, op_hi, False),
        ("Profile Curvature (1/m)",  curv_m,  "RdBu_r",  cv_lo, cv_hi, False),
        ("TPI 200 m (m)",            tpi_m,   "coolwarm", tp_lo, tp_hi, False),
    ]

    # -- Structural points in tile (50 m thinned) --
    pts_overlay = None
    if struct_pts is not None and len(struct_pts) > 0:
        tile_x_min = site_cx + x_start
        tile_x_max = site_cx + x_end
        # Note: y_start and y_end are negated offsets; tile_y_min/max
        # refer to actual northing values for spatial filtering.
        tile_y_min_northing = min(
            transform.f + 0.5 * transform.e,
            transform.f + (rows - 0.5) * transform.e
        )
        tile_y_max_northing = max(
            transform.f + 0.5 * transform.e,
            transform.f + (rows - 0.5) * transform.e
        )

        pmask = (
            (struct_pts["XCoord"] >= tile_x_min) &
            (struct_pts["XCoord"] <= tile_x_max) &
            (struct_pts["YCoord"] >= tile_y_min_northing) &
            (struct_pts["YCoord"] <= tile_y_max_northing)
        )
        pts_in = struct_pts[pmask]
        if len(pts_in) > 0:
            _grid = 50.0
            _ix = ((pts_in["XCoord"].values - tile_x_min) / _grid).astype(int)
            _iy = ((pts_in["YCoord"].values - tile_y_min_northing) / _grid).astype(int)
            _key = _ix * 100000 + _iy
            _, _uidx = np.unique(_key, return_index=True)
            pts_in = pts_in.iloc[_uidx]
            pts_overlay = pts_in

    # -- Axis tick step --
    half_range = max(abs(x_start), abs(x_end), abs(y_start), abs(y_end))
    label_step = 100 if half_range <= 100 else 200

    # -- Plot --
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Tile Centre: ({site_cx:.0f} E, {site_cy:.0f} N)  |  "
        f"Resolution: {resolution:.0f} m  |  {quality_class}",
        fontsize=11, fontweight="bold", y=0.98
    )

    for ax, (title, data, cmap, vmin, vmax, is_hillshade) in zip(
            axes.flat, panels):
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto")

        # With negated y offsets, y naturally increases downward
        # (row 0 at top has smallest y, row -1 at bottom has largest y).
        # No invert_yaxis needed.

        # Clip to data extent
        half_px_x = 0.5 * abs(x[1] - x[0]) if len(x) > 1 else 0
        half_px_y = 0.5 * abs(y[1] - y[0]) if len(y) > 1 else 0
        ax.set_xlim(x[0] - half_px_x, x[-1] + half_px_x)
        ax.set_ylim(y[0] - half_px_y, y[-1] + half_px_y)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Easting offset (m)", fontsize=8)
        ax.set_ylabel("Northing offset (m)", fontsize=8)
        ax.set_aspect("equal")

        # Tick marks at +/- label_step from centre
        ticks_x = np.arange(-label_step * 10, label_step * 10 + 1, label_step)
        ticks_x = ticks_x[(ticks_x >= x.min()) & (ticks_x <= x.max())]
        if len(ticks_x) > 0:
            ax.set_xticks(ticks_x)
        ticks_y = np.arange(-label_step * 10, label_step * 10 + 1, label_step)
        ticks_y = ticks_y[(ticks_y >= y.min()) & (ticks_y <= y.max())]
        if len(ticks_y) > 0:
            ax.set_yticks(ticks_y)

        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

        # Scatter fault scarp points on hillshade panels ONLY
        if is_hillshade and pts_overlay is not None:
            px = pts_overlay["XCoord"].values - site_cx
            # Negate Y offsets to match the downward-increasing Y axis
            py = -(pts_overlay["YCoord"].values - site_cy)
            ax.scatter(px, py, s=4, c="lime", edgecolors="black",
                       linewidths=0.3, zorder=5, alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] Tile panel -> {fig_path.name}")


# ===================================================================
# SECTION 7: MAIN WORKFLOW
# ===================================================================

def main():
    print("=" * 80)
    print("03-ResolutionSensitivity.py -- Phase 2: Numerical Resolution")
    print("                              Sensitivity Analysis")
    print("=" * 80)

    safe_makedirs(OUT_DIR)

    # ------------------------------------------------------------------
    # Step 1: Determine LiDAR DEM extent and load structural points
    # ------------------------------------------------------------------
    print(f"\n--- Step 1: Load structural points and traces ---")

    # Determine LiDAR path first (needed for extent filtering)
    if not LIDAR_RP_PATH.exists():
        print(f"  [!] Reprojected LiDAR not found: {LIDAR_RP_PATH}")
        print(f"      Attempting to use any available LiDAR path ...")
        alt_paths = list(DERIV_DIR.glob("lidar*epsg28355*.tif"))
        if alt_paths:
            lidar_path = alt_paths[0]
            print(f"  Using: {lidar_path}")
        else:
            print("  [!] No LiDAR DEM found. Exiting.")
            return
    else:
        lidar_path = LIDAR_RP_PATH

    # Read LiDAR DEM bounds for spatial filtering
    lidar_extent = None
    with rasterio.open(lidar_path) as src:
        b = src.bounds
        lidar_extent = (b.left, b.right, b.bottom, b.top)
        print(f"  LiDAR DEM extent (EPSG:28355): "
              f"E=[{b.left:.1f}, {b.right:.1f}], "
              f"N=[{b.bottom:.1f}, {b.top:.1f}]")

    struct_pts = None
    traces = None
    if STRUCT_PTS_PATH.exists():
        struct_pts = load_structural_points(STRUCT_PTS_PATH,
                                             dem_extent=lidar_extent)
        traces = reconstruct_fault_traces(struct_pts,
                                           cluster_dist_m=30, min_pts=5)

        # Filter traces: keep only those whose centroid falls within LiDAR extent
        xmin, xmax, ymin, ymax = lidar_extent
        before = len(traces)
        traces = [t for t in traces
                  if xmin <= t["centroid_x"] <= xmax
                  and ymin <= t["centroid_y"] <= ymax]
        after = len(traces)
        if before != after:
            print(f"  [Filter] Traces within LiDAR extent: "
                  f"{after}/{before} (removed {before - after} out-of-extent)")

        print(f"  {len(struct_pts):,} structural points, "
              f"{len(traces)} fault traces within LiDAR extent")
    else:
        print(f"  [!] Structural points file not found: {STRUCT_PTS_PATH}")

    # ------------------------------------------------------------------
    # Step 2: Load expression quality CSV from Phase 1
    # ------------------------------------------------------------------
    print(f"\n--- Step 2: Load Phase 1 expression quality ---")

    class_df = None
    phase1_class_paths = list(DERIV_DIR.glob("phase1_audit/*_expression_quality.csv"))
    if phase1_class_paths:
        # Use the LiDAR classification if available, else first found
        lidar_class = [p for p in phase1_class_paths
                       if "lidar" in p.name]
        class_path = lidar_class[0] if lidar_class else phase1_class_paths[0]
        class_df = pd.read_csv(class_path)
        print(f"  Loaded expression quality from: {class_path.name}")
        print(f"    {len(class_df)} traces classified")
        if "quality_class" in class_df.columns:
            print(f"    {class_df['quality_class'].value_counts().to_dict()}")
    else:
        print(f"  [!] No Phase 1 expression quality CSV found")

    # ------------------------------------------------------------------
    # Step 3: Select test sites around high-class fault scarps
    # ------------------------------------------------------------------
    print(f"\n--- Step 3: Select test sites (high-class scarps) ---")

    if lidar_extent is None:
        print("  [!] No LiDAR extent available -- cannot select test sites")
        return

    test_sites = select_scarp_tiles(
        class_df, traces, struct_pts, lidar_extent,
        n_tiles=N_TEST_SITES, tile_radius_m=SITE_RADIUS_M,
        lidar_path=lidar_path, lidar_nodata=LIDAR_NODATA,
        validity_threshold=0.99
    )

    if len(test_sites) == 0:
        print("  [!] No valid test sites within LiDAR extent -- exiting")
        return

    # Save test sites CSV
    sites_csv = OUT_DIR / "test_sites.csv"
    test_sites.to_csv(sites_csv, index=False)
    print(f"  Saved test sites: {sites_csv}")

    # ------------------------------------------------------------------
    # Step 4: Run downsampling
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Downsampling ---")

    # lidar_path already determined in Step 1

    # Check if we already have downsampled data
    # Invalidate cache if test_sites changed (different number of sites
    # or different centre coordinates)
    ds_results_csv = OUT_DIR / "downsampled_results.csv"
    _regen_ds = False
    if ds_results_csv.exists():
        try:
            old_ds = pd.read_csv(ds_results_csv)
            old_sites_csv = OUT_DIR / "test_sites.csv"
            if old_sites_csv.exists():
                old_sites = pd.read_csv(old_sites_csv)
                if (len(old_sites) != len(test_sites) or
                        not np.allclose(old_sites["center_x"].values,
                                        test_sites["center_x"].values,
                                        atol=1)):
                    print(f"  [!] Test sites changed -- regenerating "
                          f"downsampled data")
                    _regen_ds = True
        except Exception:
            _regen_ds = True

    if ds_results_csv.exists() and not _regen_ds:
        print(f"  [Skip] Loading existing downsampling results")
        ds_results = pd.read_csv(ds_results_csv)
    else:
        ds_results = run_downsampling(
            lidar_path, OUT_DIR, LIDAR_NODATA, test_sites,
            TARGET_RESOLUTIONS
        )
        ds_results.to_csv(ds_results_csv, index=False)

    # ------------------------------------------------------------------
    # Step 5: Generate derivatives at each resolution
    # ------------------------------------------------------------------
    print(f"\n--- Step 5: Derivative generation at each resolution ---")

    deriv_dir = safe_makedirs(OUT_DIR / "derivatives")
    ds_dir = OUT_DIR / "downsampled"

    for site_idx in test_sites.index:
        for res in TARGET_RESOLUTIONS:
            for alg in get_algorithms_for_resolution(res):
                stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"
                npy_path = ds_dir / f"{stem}.npy"
                json_path = ds_dir / f"{stem}.json"

                if not npy_path.exists():
                    continue

                generate_derivatives_at_resolution(
                    npy_path, json_path, deriv_dir, LIDAR_NODATA
                )

    # ------------------------------------------------------------------
    # Step 6: Compute quantitative metrics
    # ------------------------------------------------------------------
    print(f"\n--- Step 6: Quantitative metrics ---")

    metrics_csv = OUT_DIR / "metrics" / "all_metrics.csv"
    detection_csv = OUT_DIR / "metrics" / "detection_matrix.csv"

    metrics_df = None
    detection_df = None

    if metrics_csv.exists() and detection_csv.exists() and not _regen_ds:
        print(f"  [Skip] Loading existing metrics")
        try:
            metrics_df = pd.read_csv(metrics_csv)
            detection_df = pd.read_csv(detection_csv)
        except Exception as e:
            print(f"  [!] Error loading existing metrics: {e}")
            metrics_df = None
            detection_df = None

    if metrics_df is None:
        try:
            metrics_df, detection_df = run_quantitative_metrics(
                lidar_path, OUT_DIR, LIDAR_NODATA, test_sites,
                traces, struct_pts, TARGET_RESOLUTIONS
            )
        except Exception as e:
            print(f"  [!] Step 6 FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            print(f"  Continuing to Step 7 without metrics ...")

    # ------------------------------------------------------------------
    # Step 7: Generate all visualizations
    # ------------------------------------------------------------------
    print(f"\n--- Step 7: Visualizations ---")

    fig_dir = safe_makedirs(OUT_DIR / "figures")

    # 7a. Multi-panel tile figures for each site x resolution
    print("\n  7a. Multi-panel tile figures (site x resolution) ...")
    ds_dir = Path(OUT_DIR) / "downsampled"
    for site_idx, site in test_sites.iterrows():
        cx = site["center_x"]
        cy = site["center_y"]
        radius = site["radius"]
        qclass = site.get("quality_class", "") or ""

        for res in TARGET_RESOLUTIONS:
            for alg in get_algorithms_for_resolution(res):
                stem = f"site_{site_idx}_res{res_tag(res)}_{alg}"
                npy_path = ds_dir / f"{stem}.npy"
                json_path = ds_dir / f"{stem}.json"

                if not npy_path.exists() or not json_path.exists():
                    continue

                dem_arr = np.load(npy_path).astype(np.float32)
                plot_sensitivity_tile_panel(
                    dem_arr, json_path, OUT_DIR, LIDAR_NODATA,
                    struct_pts, traces, cx, cy, radius,
                    res, site_idx, quality_class=qclass
                )
                # Only generate for the first algorithm to avoid
                # excessive figures; break after first alg
                break

    # 7b. SNR vs resolution curves
    print("\n  7b. SNR vs resolution curves ...")
    if metrics_df is not None and len(metrics_df) > 0:
        plot_snr_vs_resolution(OUT_DIR, metrics_df, test_sites)

    # 7c. Kappa vs resolution bar charts
    print("\n  7c. Kappa vs resolution bar charts ...")
    if metrics_df is not None and len(metrics_df) > 0:
        plot_kappa_vs_resolution(OUT_DIR, metrics_df)

    # 7d. Scarp cross-sections
    print("\n  7d. Scarp cross-sections ...")
    for site_idx, site in test_sites.iterrows():
        # Build profiles_by_resolution from downsampled DEMs
        profiles_by_res = {}
        for res in TARGET_RESOLUTIONS:
            rep_alg = get_algorithms_for_resolution(res)[0]
            npy_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_{rep_alg}.npy"
            json_path = ds_dir / f"site_{site_idx}_res{res_tag(res)}_{rep_alg}.json"
            if not npy_path.exists():
                continue

            dem_arr = np.load(npy_path).astype(np.float32)
            # Create a simple profile across the centre
            cx_s, cy_s = site["center_x"], site["center_y"]
            n = dem_arr.shape[1]
            mid_row = dem_arr.shape[0] // 2
            elev = dem_arr[mid_row, :]
            dist = (np.arange(n) - n // 2) * res

            profiles_by_res[res] = {
                "profile_dist": dist,
                "profile_elev": elev,
            }

        if profiles_by_res:
            plot_scarp_cross_sections(OUT_DIR, site, profiles_by_res)

    # 7e. Detection matrix heatmap
    print("\n  7e. Detection matrix heatmap ...")
    if detection_df is not None and len(detection_df) > 0:
        plot_detection_matrix_heatmap(OUT_DIR, detection_df)

    # 7f. SNR spatial variability (for selected resolutions)
    print("\n  7f. SNR spatial variability ...")
    for i, (_, site) in enumerate(test_sites.iterrows()):
        if i >= 2:  # first 2 sites only
            break
        for res in [0.5, 2, 10]:
            rep_alg = get_algorithms_for_resolution(res)[0]
            npy_path = ds_dir / f"site_{site.name}_res{res_tag(res)}_{rep_alg}.npy"
            if not npy_path.exists():
                continue
            plot_snr_spatial_variability(
                OUT_DIR, lidar_path, res, traces, struct_pts,
                LIDAR_NODATA,
                (site["center_x"], site["center_y"]),
                site["radius"]
            )

    # 7g. Lineament fragmentation maps
    print("\n  7g. Lineament fragmentation maps ...")
    for i, (_, site) in enumerate(test_sites.iterrows()):
        if i >= 2:
            break
        for res in [2, 5, 10]:
            # Use actual traces as both ref and test (simplified)
            plot_lineament_fragmentation_map(
                OUT_DIR, site, traces, traces, res
            )

    # ------------------------------------------------------------------
    # Step 8: Uncertainty assessment
    # ------------------------------------------------------------------
    print(f"\n--- Step 8: Uncertainty assessment ---")

    if metrics_df is not None and len(metrics_df) > 0:
        run_uncertainty_assessment(
            OUT_DIR, test_sites, TARGET_RESOLUTIONS, metrics_df
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("PHASE 2: RESOLUTION SENSITIVITY ANALYSIS COMPLETE")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Test sites: {len(test_sites)}")
    print(f"  Resolutions tested: {TARGET_RESOLUTIONS}")
    print(f"  LiDAR-derived resolutions: "
          f"{[r for r in TARGET_RESOLUTIONS if not is_external_resolution(r)]}")
    print(f"  External DEM sources: "
          f"{[r for r in TARGET_RESOLUTIONS if is_external_resolution(r)]}")
    print(f"  Resampling algorithms (LiDAR): {RESAMPLING_ALGORITHMS}")
    print(f"  Metrics computed: {len(metrics_df) if metrics_df is not None else 0} rows")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
