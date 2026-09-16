#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06b-ScarpParameterProfiling.py -- Detailed scarp parameter profiling.

Calculate quantitative scarp parameters at 10 m intervals along validated
scarp trace point clouds from ``data/HighCon/``.  For each 10 m station a
1 km perpendicular transect is extracted from the DEM, regional
hanging-wall / footwall slopes are fitted (Theil-Sen), scarp height/throw
is decomposed, and strike/dip of the scarp surface is estimated from
adjacent crest-base pairs.

Outputs are written to ``data/derivatives/scarp_parameter_analysis/``:

  parameter_database.csv          All parameters at 10 m intervals
  scarp_segment_summaries.csv      Segment-averaged parameters
  cross_sections/                  Panel cross-section plots
  along_scarp_profiles/            Parameter-vs-distance plots
  3d_visualizations/               Multi-angle 3D renderings, per-segment
                                    chunks and per-original-trace overviews
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds, transform as window_transform, Window
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.stats import theilslopes
from scipy.special import erf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.ticker as mticker

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from derivative_common import (
    DATA_DIR, OUT_DIR, safe_makedirs, _order_points_nn,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# Constants
# ===================================================================
DEM_PATH = OUT_DIR / "lidar_0_5m_epsg28355_10m_smoke.tif"
DEM10M_PATCH_PATH = OUT_DIR / "dem10m_epsg28355_clipped.tif"
# High-resolution DEM for Rule-of-V local dip analysis (2 m pixel).
# Now the same file as DEM_PATH (both use the 2 m smoke DEM).
DEM_HR_PATH = OUT_DIR / "lidar_0_5m_epsg28355_2.0m_smoke.tif"
STRUCT_PTS_PATH = DATA_DIR / "structural_points_inventory.csv"
SCORED_POINTS_PATH = DATA_DIR / "FilteredOutput" / "lidar_0_5m_r3_scored_points.csv"
HIGHCON_DIR = DATA_DIR / "HighCon"
CLEANED_DIR = HIGHCON_DIR / "cleaned_traces"
OUTPUT_DIR = OUT_DIR / "scarp_parameter_analysis"
PROFILES_DIR = OUTPUT_DIR / "profiles_cache"  # .npz per trace for plot regen

PROFILE_HALF_LENGTH_M = 500.0      # 500 m each side -> 1 km transect
PROFILE_RESOLUTION_M = 10.0        # sampling interval along transect
STATION_SPACING_M = 10.0           # distance along trace between profiles
REGIONAL_FARFIELD_DIST_M = 200.0   # min distance from scarp centre for regional slope
VERTICAL_EXAGGERATION = 5.0
CROSS_SECTION_PANEL_SIZE = 4       # profiles per figure
CROSS_SECTION_INTERVAL_M = 100.0   # plot every N m along trace
CHUNK_3D_LENGTH_M = 2000.0         # 3D viz chunk size
TRACE_CORRIDOR_BUFFER_M = 2000.0   # perpendicular corridor half-width for by-trace 3D plot
POLYLINE_SMOOTH_SIGMA = 3.0        # gaussian sigma (stations) for best-fit polyline
PLOT_DPI = 350

# --- Piecewise crest/base detection parameters ---
SCARP_MAX_WIDTH_M = 150.0          # maximum scarp width (geometric constraint)
SCARP_CENTER_RADIUS_M = 200.0      # crest/base must be within +-200 m of centre
NOISE_GATE_MULTIPLIER = 3.0        # fitted height must exceed N * local roughness
SG_SMOOTH_WINDOW_M = 40.0          # Savitzky-Golay window length in metres
SG_SMOOTH_POLYORDER = 2            # Savitzky-Golay polynomial order

# Fault-source colours (consistent with 05a)
TRACE_COLOURS = {0: "#d73027", 1: "#4575b4", 2: "#74add1"}

# Gold-to-orange colormap for trace segments (used in 3D overview and
# per-trace parameter profiles).  Fragments are always grey.
from matplotlib.colors import LinearSegmentedColormap
_GOLD_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "gold_orange", ["#470047","#6F0B5B","#A02B68","#FFD700", "#FFB300", "#FF8C00"]
)

def _gold_orange_colors(n):
    """Return *n* colours sampled from the gold-to-orange gradient."""
    if n <= 0:
        return np.array([])
    return _GOLD_ORANGE_CMAP(np.linspace(0.15, 0.95, n))

# Consistent brown for annotation text / boxes in parameter profiles.
_BROWN = "#5D4037"
_FRAGMENT_GREY = "#888888"


def _safe_savefig(fig, path, **kwargs):
    """Save figure, tolerating OneDrive file-locking OSError."""
    try:
        fig.savefig(path, **kwargs)
    except OSError:
        print(f"  [Warning] Could not save {path} (OneDrive lock?)")
    plt.close(fig)


# ===================================================================
# 1.  TRACE LOADING AND PRE-PROCESSING
# ===================================================================

def load_traces(highcon_dir):
    """Load pre-cleaned scarp traces produced by 06a-ScarpTraceCleaning.py.

    Reads ``data/HighCon/cleaned_traces/trace_XX.csv`` files (columns: x, y,
    one row per ordered vertex) and the accompanying ``manifest.csv``.

    If the cleaned directory does not exist, falls back to the legacy inline
    dedup/split/NN-order path so the script remains usable without 06a.
    """
    cleaned_dir = CLEANED_DIR
    manifest_path = cleaned_dir / "manifest.csv"

    # ------------------------------------------------------------------
    # Primary path: read pre-cleaned CSVs from 06a
    # ------------------------------------------------------------------
    if cleaned_dir.exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        all_traces = []
        for _, row in manifest.iterrows():
            csv_path = cleaned_dir / row["csv_path"]
            if not csv_path.exists():
                print(f"[Warning] {csv_path.name} listed in manifest but missing; skipping")
                continue
            df = pd.read_csv(csv_path)
            coords = df[["x", "y"]].values.astype(np.float64)
            if len(coords) < 2:
                continue

            # trace_type: "segment" (normal) or "fragment" (short trace kept
            # for visualisation only).  Default to "segment" for backwards
            # compatibility with manifests produced by older 06a versions.
            trace_type = str(row.get("trace_type", "segment") or "segment")

            seg_lengths = np.sqrt(np.diff(coords[:, 0])**2 +
                                  np.diff(coords[:, 1])**2)
            total_len = seg_lengths.sum()
            # Fragments are short by definition; skip the 50 m minimum filter.
            if trace_type != "fragment" and total_len < 50.0:
                continue

            # Overall azimuth (from north)
            dx = coords[-1, 0] - coords[0, 0]
            dy = coords[-1, 1] - coords[0, 1]
            azimuth = np.degrees(np.arctan2(dx, dy)) % 360

            trace_id = int(row["trace_id"])
            # Determine file_idx from source_file (Trace1→0, Trace2→1, Trace3→2)
            src = row["source_file"]
            if "Trace3" in src:
                file_idx = 2
            elif "Trace2" in src:
                file_idx = 1
            else:
                file_idx = 0
            all_traces.append({
                "trace_id": trace_id,
                "file_idx": file_idx,
                "file_name": src,
                "coords": coords,
                "length_m": total_len,
                "azimuth_deg": azimuth,
                "n_points": len(coords),
                "centroid_x": float(np.mean(coords[:, 0])),
                "centroid_y": float(np.mean(coords[:, 1])),
                "trace_type": trace_type,
            })
            type_tag = f" [{trace_type}]" if trace_type != "segment" else ""
            print(f"[Trace {trace_id}]{type_tag} {row['source_file']}: "
                  f"{len(coords)} vertices, {total_len:.0f} m, "
                  f"az={azimuth:.0f} deg, qc={row.get('qc_flag', 'N/A')}")

        # ------------------------------------------------------------------
        # Renumber traces: group by big trace (Trace1/Trace2/Trace3),
        # within each group sort segments N→S then E→W and number 00..N,
        # then sort fragments the same way and continue numbering.
        # ------------------------------------------------------------------
        from itertools import groupby

        def _sort_key_north_south_east_west(t):
            """Sort key: north (high y) first, then west (low x)."""
            return (-t["centroid_y"], t["centroid_x"])

        # Group by file_name (Trace1.xlsx, Trace2.xlsx, Trace3.xlsx)
        # preserving file order Trace1 < Trace2 < Trace3.
        def _file_order_key(fn):
            if "Trace3" in fn:
                return 2
            if "Trace2" in fn:
                return 1
            return 0

        all_traces.sort(key=lambda t: (_file_order_key(t["file_name"]),
                                       -t["centroid_y"],
                                       t["centroid_x"]))

        # Separate segments and fragments per big-trace group, renumber.
        renumbered = []
        new_id = 0
        # Build groups in file order
        groups = {}
        for t in all_traces:
            key = _file_order_key(t["file_name"])
            groups.setdefault(key, []).append(t)

        for fk in sorted(groups.keys()):
            group = groups[fk]
            # Segments first, sorted N→S then E→W
            segs = [t for t in group if t.get("trace_type", "segment") != "fragment"]
            segs.sort(key=_sort_key_north_south_east_west)
            for t in segs:
                t["trace_id"] = new_id
                renumbered.append(t)
                new_id += 1

            # Fragments next, sorted N→S then E→W
            frags = [t for t in group if t.get("trace_type", "segment") == "fragment"]
            frags.sort(key=_sort_key_north_south_east_west)
            for t in frags:
                t["trace_id"] = new_id
                renumbered.append(t)
                new_id += 1

        all_traces = renumbered

        n_segments = sum(1 for t in all_traces if t.get("trace_type", "segment") == "segment")
        n_fragments = sum(1 for t in all_traces if t.get("trace_type", "segment") == "fragment")
        print(f"\n[Traces] {n_segments} segments + {n_fragments} fragments loaded "
              f"from {cleaned_dir} (renumbered N->S, E->W per big trace)")
        return all_traces

    # ------------------------------------------------------------------
    # Fallback: legacy inline dedup/split/NN-order on raw Excel files
    # ------------------------------------------------------------------
    print(f"[Warning] Cleaned trace directory not found: {cleaned_dir}")
    print(f"          Falling back to legacy inline processing of raw "
          f"Trace*.xlsx files.")
    print(f"          Run 06a-ScarpTraceCleaning.py first for better results.")

    trace_files = sorted(highcon_dir.glob("Trace*.xlsx"))
    if not trace_files:
        raise FileNotFoundError(
            f"No Trace*.xlsx files in {highcon_dir} and no cleaned traces "
            f"in {cleaned_dir}. Run 06a-ScarpTraceCleaning.py first."
        )

    all_traces = []
    trace_id = 0
    for fi, fpath in enumerate(trace_files):
        df = pd.read_excel(fpath)
        df = df.rename(columns={"x": "XCoord", "y": "YCoord"})
        coords = df[["XCoord", "YCoord"]].values.astype(np.float64)

        print(f"[Trace {trace_id}] {fpath.name}: {len(coords)} points loaded")

        # Remove exact duplicates (rounded to 0.1 m)
        rounded = np.round(coords, 1)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_idx.sort()
        coords = coords[unique_idx]
        print(f"  Deduplicated: {len(coords)} unique points")

        # Split at sequential gaps > max_gap_m
        sub_traces = _split_sequential(coords, max_gap_m=200.0)
        print(f"  Split into {len(sub_traces)} continuous sub-traces")

        for st in sub_traces:
            if len(st) < 5:
                continue
            # Only NN-order small sub-traces; large ones are already sequential
            if len(st) < 500:
                ordered = _order_points_nn(st)
            else:
                ordered = st
            seg_lengths = np.sqrt(np.diff(ordered[:, 0])**2 +
                                  np.diff(ordered[:, 1])**2)
            total_len = seg_lengths.sum()
            if total_len < 50.0:
                continue
            # Overall azimuth (from north)
            dx = ordered[-1, 0] - ordered[0, 0]
            dy = ordered[-1, 1] - ordered[0, 1]
            azimuth = np.degrees(np.arctan2(dx, dy)) % 360
            all_traces.append({
                "trace_id": trace_id,
                "file_idx": fi,
                "file_name": fpath.name,
                "coords": ordered,
                "length_m": total_len,
                "azimuth_deg": azimuth,
                "n_points": len(ordered),
                "centroid_x": float(np.mean(ordered[:, 0])),
                "centroid_y": float(np.mean(ordered[:, 1])),
            })
            print(f"  Sub-trace {trace_id}: {len(ordered)} pts, "
                  f"{total_len:.0f} m, az={azimuth:.0f} deg")
            trace_id += 1

    print(f"\n[Traces] {len(all_traces)} continuous traces prepared")
    return all_traces


def _split_sequential(coords, max_gap_m=200.0):
    """Split sequentially-ordered points at gaps > *max_gap_m*.

    Much faster than BFS clustering: O(n) instead of O(n^2) for large
    point clouds.
    """
    if len(coords) <= 2:
        return [coords]

    # Compute consecutive distances
    diffs = np.diff(coords, axis=0)
    dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

    # Find split locations
    split_idx = np.where(dists > max_gap_m)[0]

    # Split into sub-arrays
    sub_traces = []
    start = 0
    for si in split_idx:
        sub = coords[start:si + 1]
        if len(sub) >= 3:
            sub_traces.append(sub)
        start = si + 1
    # Last segment
    sub = coords[start:]
    if len(sub) >= 3:
        sub_traces.append(sub)

    return sub_traces


def resample_trace(coords, spacing_m=10.0):
    """Resample an ordered polyline at uniform *spacing_m* intervals.

    Returns:
        stations : (N, 2) array of (x, y) at uniform spacing
        distances : (N,) array of cumulative distance along trace
        azimuths  : (N,) array of local azimuth (deg from north) at each station
    """
    seg_vecs = np.diff(coords, axis=0)
    seg_lens = np.sqrt(seg_vecs[:, 0]**2 + seg_vecs[:, 1]**2)
    cum_lens = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    n_stations = max(2, int(total_len / spacing_m) + 1)
    target_dists = np.linspace(0, total_len, n_stations)

    stations = np.zeros((n_stations, 2))
    azimuths = np.zeros(n_stations)

    for i, td in enumerate(target_dists):
        # Find segment containing this distance
        seg_idx = np.searchsorted(cum_lens, td, side="right") - 1
        seg_idx = min(seg_idx, len(seg_vecs) - 1)
        # Interpolate position
        seg_start = cum_lens[seg_idx]
        seg_len = seg_lens[seg_idx]
        if seg_len < 1e-9:
            stations[i] = coords[seg_idx]
        else:
            t = (td - seg_start) / seg_len
            stations[i] = coords[seg_idx] + t * seg_vecs[seg_idx]
        # Local azimuth from segment direction (from north)
        vx, vy = seg_vecs[seg_idx]
        azimuths[i] = np.degrees(np.arctan2(vx, vy)) % 360

    return stations, target_dists, azimuths


# ===================================================================
# 2.  PROFILE EXTRACTION
# ===================================================================

def extract_perpendicular_profile(dem_data, transform, center_xy, azimuth_deg,
                                  half_length_m=500.0, resolution_m=10.0,
                                  dem_shape=None, nodata=None):
    """Extract elevation profile perpendicular to scarp at *center_xy*.

    The profile is oriented so that negative distances are on the hanging-wall
    side and positive distances are on the footwall side whenever the DEM shows
    the expected elevation contrast.
    """
    # Perpendicular direction = strike + 90 deg
    perp_az = (azimuth_deg + 90.0) % 360.0
    rad = np.radians(perp_az)
    dx = np.sin(rad)
    dy = np.cos(rad)

    n_samples = int(2 * half_length_m / resolution_m) + 1
    distances = np.linspace(-half_length_m, half_length_m, n_samples)

    xs = center_xy[0] + distances * dx
    ys = center_xy[1] + distances * dy

    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)

    # Track the perpendicular azimuth actually used to sample `distances`,
    # so callers can correctly reconstruct map (x, y) from signed distances
    # even after the hanging-wall/footwall flip below.
    perp_azimuth_used = perp_az

    elevations = np.full(n_samples, np.nan, dtype=np.float64)
    valid = (cols >= 0) & (cols < dem_shape[1]) & (rows >= 0) & (rows < dem_shape[0])
    if not np.any(valid):
        return distances, elevations, perp_azimuth_used

    coords_rc = np.array([rows[valid], cols[valid]])
    sampled = map_coordinates(dem_data.astype(np.float64), coords_rc,
                              order=1, mode="constant", cval=np.nan)
    elevations[valid] = sampled

    if nodata is not None:
        elevations[elevations == nodata] = np.nan
    elevations[~np.isfinite(elevations)] = np.nan

    # Ensure perpendicular orientation points from hanging wall to footwall.
    # If the positive side is actually higher, flip the direction so the
    # lower footwall side becomes positive and the hanging wall side becomes
    # negative.
    if np.isfinite(elevations).sum() >= 6:
        left_mask = distances < -50.0
        right_mask = distances > 50.0
        left_vals = elevations[left_mask]
        right_vals = elevations[right_mask]
        if left_vals.size >= 3 and right_vals.size >= 3:
            left_mean = np.nanmean(left_vals)
            right_mean = np.nanmean(right_vals)
            if np.isfinite(left_mean) and np.isfinite(right_mean) and right_mean > left_mean:
                distances = -distances
                perp_azimuth_used = (perp_az + 180.0) % 360.0
                xs = center_xy[0] + distances * dx
                ys = center_xy[1] + distances * dy
                cols, rows = inv_transform * (xs, ys)
                elevations = np.full(n_samples, np.nan, dtype=np.float64)
                valid = (cols >= 0) & (cols < dem_shape[1]) & (rows >= 0) & (rows < dem_shape[0])
                if np.any(valid):
                    coords_rc = np.array([rows[valid], cols[valid]])
                    sampled = map_coordinates(dem_data.astype(np.float64), coords_rc,
                                              order=1, mode="constant", cval=np.nan)
                    elevations[valid] = sampled
                    if nodata is not None:
                        elevations[elevations == nodata] = np.nan
                    elevations[~np.isfinite(elevations)] = np.nan

    return distances, elevations, perp_azimuth_used


# ===================================================================
# 3.  SCARP HEIGHT / THROW CALCULATION
# ===================================================================

def _plot_debug_profile(d, z, d_crest, z_crest, d_base, z_base, title="scarp_debug"):
    """Save a simple diagnostic plot for break detection quality."""
    debug_dir = OUTPUT_DIR / "debug_plots"
    safe_makedirs(debug_dir)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d, z, "k-", linewidth=1.5, label="Profile")
    if np.isfinite(d_crest) and np.isfinite(z_crest):
        ax.axvline(d_crest, color="tab:red", linestyle="--", alpha=0.8, label="Crest")
        ax.scatter([d_crest], [z_crest], color="tab:red", s=30, zorder=5)
    if np.isfinite(d_base) and np.isfinite(z_base):
        ax.axvline(d_base, color="tab:blue", linestyle="--", alpha=0.8, label="Base")
        ax.scatter([d_base], [z_base], color="tab:blue", s=30, zorder=5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    out_path = debug_dir / f"{title}.png"
    _safe_savefig(fig, out_path, dpi=150, bbox_inches="tight")


def _diagnose_profile(distances, elevations, trace_id, station_idx, output_dir=None):
    """Save diagnostic plot for a single profile to understand what's happening."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / "diagnostics"
    safe_makedirs(output_dir)

    valid = np.isfinite(elevations)
    d = distances[valid]
    z = elevations[valid]

    if len(d) < 10:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    ax = axes[0]
    ax.plot(distances, elevations, 'k-', linewidth=0.5, alpha=0.5, label='Raw')
    ax.plot(d, z, 'b-', linewidth=1.5, label='Valid points')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Trace {trace_id}, Station {station_idx}: Raw Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Profile center')
    ax.axvspan(-REGIONAL_FARFIELD_DIST_M, REGIONAL_FARFIELD_DIST_M,
               alpha=0.1, color='yellow', label='Near-field zone')
    ax.legend()

    ax = axes[1]
    dz = np.gradient(z, d)
    d2z = np.gradient(dz, d)
    ax.plot(d, dz, 'r-', linewidth=1.5, label='1st derivative (slope)')
    ax.plot(d, d2z, 'b-', linewidth=1.5, label='2nd derivative (curvature)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Derivative')
    ax.set_xlabel('Distance along profile (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(d, z, 'k-', linewidth=1.5, label='Profile')
    z_max = np.max(z)
    z_min = np.min(z)
    relief = z_max - z_min
    dz_abs = np.abs(dz)
    threshold = np.percentile(dz_abs, 80)
    steep_zone = dz_abs >= threshold

    for i in range(len(d) - 1):
        if steep_zone[i]:
            ax.axvspan(d[i], d[i+1], alpha=0.3, color='orange')

    z_20 = z_min + 0.2 * relief
    z_80 = z_min + 0.8 * relief
    above_80 = z >= z_80
    below_20 = z <= z_20

    if np.any(above_80) and np.any(below_20):
        crest_candidates = np.where(above_80)[0]
        base_candidates = np.where(below_20)[0]

        if len(crest_candidates) > 0 and len(base_candidates) > 0:
            crest_idx = crest_candidates[-1] if d[crest_candidates[0]] < 0 else crest_candidates[0]
            base_idx = base_candidates[0] if d[base_candidates[0]] > 0 else base_candidates[-1]

            ax.plot(d[crest_idx], z[crest_idx], 'r^', markersize=15,
                    label=f'Crest: d={d[crest_idx]:.0f}m, z={z[crest_idx]:.1f}m')
            ax.plot(d[base_idx], z[base_idx], 'bv', markersize=15,
                    label=f'Base: d={d[base_idx]:.0f}m, z={z[base_idx]:.1f}m')

            width = abs(d[base_idx] - d[crest_idx])
            height = abs(z[crest_idx] - z[base_idx])
            ax.set_title(f'Simple Detection: width={width:.0f}m, height={height:.2f}m, slope={np.degrees(np.arctan(height/width)):.1f} deg')

    ax.set_xlabel('Distance along profile (m)')
    ax.set_ylabel('Elevation (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"diagnostic_trace{trace_id}_station{station_idx:04d}.png"
    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except OSError:
        pass
    plt.close(fig)


def _empty_throw():
    return {
        "z_crest_primary": np.nan, "z_base_primary": np.nan,
        "d_crest": np.nan, "d_base": np.nan,
        "z_hw_projected": np.nan, "z_fw_projected": np.nan,
        "primary_throw": np.nan,
        "total_cumulative_throw": np.nan, "vertical_separation": np.nan,
        "true_vertical_displacement": np.nan,
        "VS_max": np.nan, "VS_min": np.nan,
        "net_slip": np.nan, "heave": np.nan,
        "vertical_displacement": np.nan,
        "scarp_width_m": np.nan, "scarp_slope_deg": np.nan,
        "max_local_slope_deg": np.nan,
        "HW_regional_slope": np.nan, "FW_regional_slope": np.nan,
        "scarp_slope": np.nan,
        "HW_regional_slope_deg": np.nan, "FW_regional_slope_deg": np.nan,
        "is_reverse_scarp": False,
    }


# ===================================================================
# 4.  STRIKE / DIP CALCULATION
# ===================================================================

def calculate_scarp_geometry(crest_points, base_points):
    """Calculate strike and dip from multiple crest-base pairs.

    Uses a more robust method: calculate local scarp surface orientation
    from the 3D geometry of crest-base pairs.
    """
    if not crest_points or not base_points:
        return _empty_geometry()

    if len(crest_points) < 2 or len(base_points) < 2:
        return _empty_geometry()

    crest_pts = np.array(crest_points)
    base_pts = np.array(base_points)

    from scipy.spatial import cKDTree
    base_tree = cKDTree(base_pts[:, :2])

    scarp_vectors = []
    scarp_strikes = []

    for crest in crest_pts:
        dist, idx = base_tree.query(crest[:2])
        base = base_pts[idx]
        vector_3d = crest - base
        h_dist = np.sqrt(vector_3d[0]**2 + vector_3d[1]**2)
        v_dist = abs(vector_3d[2])

        if h_dist > 1.0 and v_dist > 0.1:
            dip_angle = np.degrees(np.arctan2(v_dist, h_dist))
            dip_dir = np.degrees(np.arctan2(-vector_3d[0], -vector_3d[1])) % 360
            strike = (dip_dir - 90) % 360

            scarp_vectors.append(vector_3d)
            scarp_strikes.append(strike)

    if len(scarp_strikes) < 2:
        return _empty_geometry()

    strike_rad = np.radians(scarp_strikes)
    mean_sin = np.mean(np.sin(2 * strike_rad))
    mean_cos = np.mean(np.cos(2 * strike_rad))
    mean_strike = np.degrees(np.arctan2(mean_sin, mean_cos)) / 2 % 180

    dips = []
    for vec in scarp_vectors:
        h_dist = np.sqrt(vec[0]**2 + vec[1]**2)
        v_dist = abs(vec[2])
        if h_dist > 0:
            dip = np.degrees(np.arctan2(v_dist, h_dist))
            dips.append(dip)

    mean_dip = np.median(dips) if dips else np.nan
    dip_direction = (mean_strike + 90) % 360

    strike_std = np.std(scarp_strikes) if len(scarp_strikes) > 1 else 0
    dip_std = np.std(dips) if len(dips) > 1 else 0
    plane_quality = 1.0 / (1.0 + (strike_std + dip_std) / 2.0)

    return {
        "strike": float(mean_strike),
        "dip": float(mean_dip),
        "dip_direction": float(dip_direction),
        "plane_quality_rms": float(plane_quality),
    }


def _empty_geometry():
    return {
        "strike": np.nan, "dip": np.nan,
        "dip_direction": np.nan, "plane_quality_rms": np.nan,
    }


# ===================================================================
# 5.  CORE PROCESSING PIPELINE
# ===================================================================

def _piecewise_fit_crest_base(d, z, mh, bh, mf, bf, hw_on_left,
                              max_w=SCARP_MAX_WIDTH_M,
                              center_r=SCARP_CENTER_RADIUS_M,
                              noise_mult=NOISE_GATE_MULTIPLIER,
                              sg_window_m=SG_SMOOTH_WINDOW_M,
                              sg_polyorder=SG_SMOOTH_POLYORDER):
    """Continuity-constrained 3-segment piecewise fit for crest/base detection.

    Replaces the former ``_find_crossings`` + ``_select_crest_base_by_area``
    two-stage pipeline with a single optimisation that jointly fits all
    three segments (HW regional + scarp + FW regional) while enforcing
    continuity at the crest and base breakpoints.

    Model (hw_on_left=True):
      HW:    z = bh + mh*d         for d < d[i]  (slope fixed to mh)
      Scarp: z = bh + mh*d[i] + ms*(d - d[i])  for d[i] <= d <= d[j]
      FW:    z = bf + mf*d         for d > d[j]  (slope fixed to mf)
      Continuity at d[i]: HW line value = scarp line value
      Continuity at d[j]: scarp line value = FW line value

    The HW/FW slopes are **fixed** to the far-field Theil-Sen values.
    Only the two breakpoints (i, j) and the scarp slope (ms) are free.
    The intercepts bh and bf are derived from the continuity constraint
    and the OLS solution on each segment, so they are not free parameters
    either.

    Optimisation:
      1. Coarse grid search over candidate (i, j) pairs satisfying the
         geometric constraints (|d| <= center_r, width <= max_w).
      2. Fine refinement around the best coarse candidate.
      3. Objective: minimise normalised total RSS (continuity-constrained).
         Tiebreaker: widest scarp within an additive tolerance floor.
      4. Noise gate: fitted height > noise_mult * local roughness sigma.
         If the gate is not passed, returns None ("no scarp detected").

    Parameters
    ----------
    d, z : sorted profile arrays (1-D, same length)
    mh, bh : float -- HW regional slope and intercept (Theil-Sen far-field)
    mf, bf : float -- FW regional slope and intercept (Theil-Sen far-field)
    hw_on_left : bool -- whether HW is on the left side of the profile
    max_w : float -- maximum scarp width (geometric constraint)
    center_r : float -- crest/base must be within +-center_r of profile centre
    noise_mult : float -- fitted height must exceed this multiple of roughness
    sg_window_m : float -- Savitzky-Golay smoothing window in metres (0 to disable)
    sg_polyorder : int -- Savitzky-Golay polynomial order

    Returns
    -------
    (crest_d, crest_z, base_d, base_z, scarp_slope, scarp_intercept,
     fitted_h, roughness, snr) or None if no scarp detected.
    """
    n = len(d)
    dx = float(np.median(np.diff(d)))
    grid_step = max(dx, 5.0)

    d_min, d_max = d[0], d[-1]
    search_lo = max(d_min, -center_r)
    search_hi = min(d_max, center_r)

    # --- Optional Savitzky-Golay smoothing ---
    z_fit = z.copy()
    if sg_window_m > 0:
        sg_w = max(5, int(sg_window_m / dx))
        if sg_w % 2 == 0:
            sg_w += 1
        if sg_w < n:
            z_fit = savgol_filter(z_fit, sg_w, sg_polyorder)

    # Precompute
    y_h = z_fit - mh * d
    y_f = z_fit - mf * d

    cum_yh = np.concatenate([[0.0], np.cumsum(y_h)])
    cum_yh2 = np.concatenate([[0.0], np.cumsum(y_h**2)])
    cum_yf = np.concatenate([[0.0], np.cumsum(y_f)])
    cum_yf2 = np.concatenate([[0.0], np.cumsum(y_f**2)])

    suff_yh = np.concatenate([np.cumsum(y_h[::-1])[::-1], [0.0]])
    suff_yh2 = np.concatenate([np.cumsum(y_h[::-1]**2)[::-1], [0.0]])
    suff_yf = np.concatenate([np.cumsum(y_f[::-1])[::-1], [0.0]])
    suff_yf2 = np.concatenate([np.cumsum(y_f[::-1]**2)[::-1], [0.0]])

    def hw_prefix(i):
        cnt = i
        if cnt < 3:
            return 0.0, np.inf, 0
        s = cum_yh[i]
        bh_ = s / cnt
        s2 = cum_yh2[i]
        rss = s2 - 2.0 * bh_ * s + cnt * bh_**2
        return bh_, rss, cnt

    def hw_suffix(k):
        cnt = n - k
        if cnt < 3:
            return 0.0, np.inf, 0
        s = suff_yh[k]
        bh_ = s / cnt
        s2 = suff_yh2[k]
        rss = s2 - 2.0 * bh_ * s + cnt * bh_**2
        return bh_, rss, cnt

    def fw_rss_constrained(bf_, start, end):
        cnt = end - start
        if cnt < 3:
            return np.inf, 0
        if start == 0:
            s = cum_yf[end]
            s2 = cum_yf2[end]
        elif end == n:
            s = suff_yf[start]
            s2 = suff_yf2[start]
        else:
            s = cum_yf[end] - cum_yf[start]
            s2 = cum_yf2[end] - cum_yf2[start]
        rss = s2 - 2.0 * bf_ * s + cnt * bf_**2
        return rss, cnt

    def scarp_continuous(i, j, bh_, left_hw):
        ds_ = d[i:j + 1]
        zs_ = z_fit[i:j + 1]
        ns = len(ds_)
        if ns < 3:
            return np.inf, 0.0, 0.0, 0
        if left_hw:
            anchor = d[i]
        else:
            anchor = d[j]
        u = zs_ - bh_ - mh * anchor
        v = ds_ - anchor
        sv2 = float(np.dot(v, v))
        if sv2 < 1e-12:
            return np.inf, 0.0, 0.0, 0
        suv = float(np.dot(u, v))
        ms = suv / sv2
        su2 = float(np.dot(u, u))
        rss = su2 - suv**2 / sv2
        if left_hw:
            bf_ = bh_ + (mh - ms) * d[i] + (ms - mf) * d[j]
        else:
            bf_ = bh_ + (mh - ms) * d[j] + (ms - mf) * d[i]
        return rss, ms, bf_, ns

    def evaluate_pair(i, j):
        width = abs(float(d[j] - d[i]))
        if width < 0.5 or width > max_w:
            return np.inf, 0.0, 0
        # --- Base-distance constraint ---
        # The cleaned trace is biased towards the base, so the distance
        # from the trace centre (d=0) to the detected base must always
        # be smaller than the distance to the crest:
        #     |d_base| < |d_crest|
        # When hw_on_left=True: crest=d[i], base=d[j]  ->  |d[j]| < |d[i]|
        # When hw_on_left=False: crest=d[j], base=d[i] ->  |d[i]| < |d[j]|
        if hw_on_left:
            if abs(float(d[j])) >= abs(float(d[i])):
                return np.inf, 0.0, 0
        else:
            if abs(float(d[i])) >= abs(float(d[j])):
                return np.inf, 0.0, 0
        if hw_on_left:
            bh_, rss_hw, nh = hw_prefix(i)
            rss_sc, ms, bf_cont, ns = scarp_continuous(i, j, bh_, True)
            rss_fw, nf = fw_rss_constrained(bf_cont, j + 1, n)
        else:
            bh_, rss_hw, nh = hw_suffix(j + 1)
            rss_sc, ms, bf_cont, ns = scarp_continuous(i, j, bh_, False)
            rss_fw, nf = fw_rss_constrained(bf_cont, 0, i)
        total = rss_hw + rss_sc + rss_fw
        tn = nh + ns + nf
        if tn < 9:
            return np.inf, 0.0, 0
        return total / tn, width, tn

    # --- Coarse grid search ---
    grid = np.arange(search_lo, search_hi + grid_step, grid_step)
    grid_indices = np.unique(np.clip(np.searchsorted(d, grid, side='left'), 0, n - 1))
    if len(grid_indices) < 3:
        return None

    best_rss = np.inf
    best_i = best_j = None
    best_width = 0.0
    tol = 1e-8

    for ci in range(len(grid_indices)):
        i = int(grid_indices[ci])
        for cj in range(ci + 1, len(grid_indices)):
            j = int(grid_indices[cj])
            rss, width, _ = evaluate_pair(i, j)
            if not np.isfinite(rss):
                continue
            if rss < best_rss - tol:
                best_rss = rss
                best_i = i
                best_j = j
                best_width = width
            elif abs(rss - best_rss) <= tol and width > best_width:
                best_rss = rss
                best_i = i
                best_j = j
                best_width = width

    if best_i is None:
        return None

    # --- Fine refinement ---
    refine_n = max(1, int(grid_step / dx) + 1)
    lo_i = max(0, best_i - refine_n)
    hi_i = min(n - 1, best_i + refine_n)
    lo_j = max(0, best_j - refine_n)
    hi_j = min(n - 1, best_j + refine_n)

    for i in range(lo_i, hi_i + 1):
        for j in range(max(i + 1, lo_j), hi_j + 1):
            rss, width, _ = evaluate_pair(i, j)
            if not np.isfinite(rss):
                continue
            if rss < best_rss - tol:
                best_rss = rss
                best_i = i
                best_j = j
                best_width = width
            elif abs(rss - best_rss) <= tol and width > best_width:
                best_rss = rss
                best_i = i
                best_j = j
                best_width = width

    # --- Extract final results ---
    i, j = best_i, best_j

    if hw_on_left:
        bh_, _, _ = hw_prefix(i)
        _, ms, bf_, _ = scarp_continuous(i, j, bh_, True)
        xc = float(d[i])
        xb = float(d[j])
        hw_mask = np.arange(n) < i
        fw_mask = np.arange(n) > j
    else:
        bh_, _, _ = hw_suffix(j + 1)
        _, ms, bf_, _ = scarp_continuous(i, j, bh_, False)
        xc = float(d[j])  # crest on right
        xb = float(d[i])  # base on left
        hw_mask = np.arange(n) > j
        fw_mask = np.arange(n) < i
    sc_mask = (np.arange(n) >= i) & (np.arange(n) <= j)

    # Theil-Sen refit on scarp segment (using the smoothed profile for
    # robustness, then applying to original for downstream parameters)
    ds, zs = d[sc_mask], z_fit[sc_mask]
    if len(ds) >= 5:
        sf = theilslopes(zs, ds)
        ms_ts = float(sf.slope)
        bs_ts = float(sf.intercept)
    elif len(ds) >= 2:
        ms_ts, bs_ts = np.polyfit(ds, zs, 1)
    else:
        ms_ts, bs_ts = 0.0, float(np.mean(zs))

    # Fitted height (from continuity-constrained model)
    fit_h = abs(ms * (float(d[j]) - float(d[i])))

    # Roughness: std of HW + FW residuals (outside scarp)
    dh, zh = d[hw_mask], z[hw_mask]
    df_, zf = d[fw_mask], z[fw_mask]
    ih = float(np.mean(zh - mh * dh)) if len(dh) > 0 else bh
    iff = float(np.mean(zf - mf * df_)) if len(df_) > 0 else bf
    hw_resid = zh - (mh * dh + ih) if len(dh) > 0 else np.array([])
    fw_resid = zf - (mf * df_ + iff) if len(df_) > 0 else np.array([])
    all_resid = np.concatenate([hw_resid, fw_resid])
    rough = float(np.std(all_resid)) if len(all_resid) > 2 else 0.0
    snr = fit_h / rough if rough > 1e-10 else np.inf

    # Noise gate: if fitted height does not exceed noise_mult * roughness,
    # report "no scarp detected"
    if fit_h < noise_mult * rough:
        return None

    # ------------------------------------------------------------------
    # Crest/base elevations
    # ------------------------------------------------------------------
    # IMPORTANT:
    # xc and xb are locations on the observed DEM profile.  The previous
    # implementation returned elevations from the projected regional
    # surfaces:
    #
    #     zc = mh * xc + ih
    #     zb = mf * xb + iff
    #
    # These are useful projected elevations, but they are NOT the actual
    # elevations on the DEM profile.
    #
    # Primary crest/base elevations should therefore be sampled directly
    # from the observed profile z at xc and xb.
    #
    # Use a small local median rather than a single interpolated point to
    # make the reported elevations less sensitive to pixel-scale DEM noise.
    def local_profile_elevation(x_target, half_width_m=3.0):
        local_mask = np.abs(d - x_target) <= half_width_m

        if np.count_nonzero(local_mask) >= 2:
            return float(np.nanmedian(z[local_mask]))

        # Fallback when the profile is sparsely sampled
        return float(np.interp(x_target, d, z))


    z_crest_profile = local_profile_elevation(xc)
    z_base_profile = local_profile_elevation(xb)

    # Projected elevations from the regional HW/FW surfaces.
    z_crest_projected = float(mh * xc + ih)
    z_base_projected = float(mf * xb + iff)
    return (xc,z_crest_profile,xb,z_base_profile,ms_ts,bs_ts,fit_h,rough,snr,z_crest_projected,z_base_projected,)


def calculate_reverse_fault_parameters(distances, elevations, dip_deg=None):
    """Calculate reverse fault parameters using continuity-constrained
    piecewise fit.

    Architecture (3-stage pipeline):

      Stage A -- Fit HW/FW regional slopes from **fixed far-field windows**
                 (|d| > 200 m), independent of any crest/base detection.
      Stage B -- Continuity-constrained 3-segment piecewise fit
                 (HW regional + scarp + FW regional) with breakpoints
                 optimised under geometric constraints (|d| <= 200 m,
                 width <= 150 m).  Objective: minimise normalised total
                 RSS with additive tolerance + widest-scarp tiebreaker.
                 A noise gate rejects profiles where fitted height < 3x
                 local roughness, reporting "no scarp detected".
      Stage C -- Fit scarp slope (Theil-Sen on the detected scarp segment)
                 and compute all downstream displacement parameters.
    """
    valid = np.isfinite(elevations)
    if valid.sum() < 20:
        return _empty_throw()

    d = distances[valid]
    z = elevations[valid]

    # Ensure data is sorted by distance
    sort_idx = np.argsort(d)
    d = d[sort_idx]
    z = z[sort_idx]

    # ------------------------------------------------------------------
    # Stage A: Fit HW/FW regional slopes from fixed far-field windows
    # ------------------------------------------------------------------
    # Use points beyond REGIONAL_FARFIELD_DIST_M (200 m) on each side.
    # This is independent of any crest/base detection -- the far-field
    # surfaces are, by definition, away from the scarp zone.
    hw_far_mask = d <= -REGIONAL_FARFIELD_DIST_M
    fw_far_mask = d >=  REGIONAL_FARFIELD_DIST_M

    # Fall back to outer 20 % if the 200 m window has too few points
    if hw_far_mask.sum() < 5 or fw_far_mask.sum() < 5:
        n_outer = max(3, int(len(d) * 0.2))
        hw_far_mask = np.zeros(len(d), dtype=bool)
        fw_far_mask = np.zeros(len(d), dtype=bool)
        hw_far_mask[:n_outer] = True
        fw_far_mask[-n_outer:] = True

    hw_d = d[hw_far_mask]
    hw_z = z[hw_far_mask]
    fw_d = d[fw_far_mask]
    fw_z = z[fw_far_mask]

    if hw_d.size < 3 or fw_d.size < 3:
        return _empty_throw()

    # Determine HW/FW sides from far-field median elevations
    left_far_elev = np.nanmedian(z[d < 0])
    right_far_elev = np.nanmedian(z[d >= 0])
    if left_far_elev > right_far_elev:
        hw_on_left = True
    else:
        hw_on_left = False

    # Theil-Sen robust linear fit for regional slopes
    if len(hw_d) >= 5:
        hw_fit = theilslopes(hw_z, hw_d)
        mh = float(hw_fit.slope)
        bh = float(hw_fit.intercept)
    else:
        try:
            mh, bh = np.polyfit(hw_d, hw_z, 1)
        except (TypeError, ValueError):
            return _empty_throw()

    if len(fw_d) >= 5:
        fw_fit = theilslopes(fw_z, fw_d)
        mf = float(fw_fit.slope)
        bf = float(fw_fit.intercept)
    else:
        try:
            mf, bf = np.polyfit(fw_d, fw_z, 1)
        except (TypeError, ValueError):
            return _empty_throw()

    # ------------------------------------------------------------------
    # Stage B: Continuity-constrained piecewise fit for crest/base
    # ------------------------------------------------------------------
    result = _piecewise_fit_crest_base(
        d, z, mh, bh, mf, bf, hw_on_left,
    )

    if result is None:
        # Noise gate rejected: no scarp detected
        return _empty_throw()

    (x1,z_crest_profile,x2,z_base_profile,ms,bs,fit_h,roughness,snr,z_crest_projected,z_base_projected) = result

    # ------------------------------------------------------------------
    # Stage C: Compute all downstream displacement parameters.
    # ------------------------------------------------------------------
    # ms and bs (scarp slope/intercept) are already returned by
    # _piecewise_fit_crest_base (Theil-Sen refit on the scarp segment).
    # Define scarp bounds for the max_local_slope calculation below.
    scarp_lo = min(x1, x2)
    scarp_hi = max(x1, x2)

    # ------------------------------------------------------------------
    # Stage C (continued): Compute all downstream displacement parameters.
    # ------------------------------------------------------------------
    scarp_height = abs(z_crest_profile - z_base_profile)

    # Vertical separation
    VS_max = abs(x1 * (mh - mf) + (bh - bf))
    VS_min = abs(x2 * (mh - mf) + (bh - bf))
    VS = (VS_max + VS_min) / 2.0

    # Total cumulative throw: the vertical separation between the projected
    # HW and FW far-field surfaces evaluated at the scarp midpoint.
    scarp_mid_x = (x1 + x2) / 2.0
    hw_z_at_mid = mh * scarp_mid_x + bh
    fw_z_at_mid = mf * scarp_mid_x + bf
    total_cumulative_throw = abs(hw_z_at_mid - fw_z_at_mid)
    if total_cumulative_throw < scarp_height:
        total_cumulative_throw = max(VS, scarp_height)

    scarp_width = abs(x2 - x1)

    scarp_slope_deg = float(np.degrees(np.arctan(abs(ms))))
    hw_slope_deg = float(np.degrees(np.arctan(mh)))
    fw_slope_deg = float(np.degrees(np.arctan(mf)))

    # Maximum local slope within the scarp window (steepest point on the
    # scarp face).  Computed from a smoothed slope array restricted to the
    # crest-base interval.
    from scipy.ndimage import gaussian_filter1d
    z_smooth = gaussian_filter1d(z, sigma=max(3, len(d) // 50) / 3)
    dz_smooth = np.gradient(z_smooth, d)
    slope_angles = np.degrees(np.arctan(np.abs(dz_smooth)))
    scarp_lo_idx = np.searchsorted(d, scarp_lo, side='left')
    scarp_hi_idx = np.searchsorted(d, scarp_hi, side='right')
    if scarp_hi_idx > scarp_lo_idx:
        local_slopes_in_scarp = slope_angles[scarp_lo_idx:scarp_hi_idx]
        max_local_slope_deg = float(np.nanmax(local_slopes_in_scarp))
    else:
        max_local_slope_deg = scarp_slope_deg

    # Sanity check on scarp width
    if scarp_width > 500 or scarp_width < 0.5:
        return _empty_throw()
    
    # Calculate displacements
    if dip_deg is not None and np.isfinite(dip_deg) and dip_deg > 1.0:
        theta = np.radians(dip_deg)
        
        if abs(mh - mf) < 0.01:
            alpha_h = np.arctan(mh)
            DS = VS * np.cos(alpha_h) / np.sin(theta + alpha_h)
            HD = VS / (np.tan(theta) + np.tan(alpha_h))
            VD = VS * np.tan(theta) / (np.tan(theta) + np.tan(alpha_h))
        else:
            x = (x1 + x2) / 2.0
            y = (z_crest_profile + z_base_profile) / 2.0
            
            numerator_AP = x * (ms - mf) + (bs - bf)
            denominator_AP = np.sin(theta) - mf * np.cos(theta)
            AP = numerator_AP / denominator_AP if abs(denominator_AP) > 1e-10 else 0
            
            numerator_BP = x * (mh - ms) + (bh - bs)
            denominator_BP = np.sin(theta) - mh * np.cos(theta)
            BP = numerator_BP / denominator_BP if abs(denominator_BP) > 1e-10 else 0
            
            DS = abs(AP + BP)
            VD = DS * np.sin(theta)
            HD = DS * np.cos(theta)
    else:
        DS = np.nan
        HD = np.nan
        VD = np.nan
    
    # Projected HW and FW elevations: evaluate the regional slope lines
    # at the crest and base intersection points respectively.  The
    # "projected H" is the vertical difference between the HW regional
    # slope at the crest and the FW regional slope at the base.
    #
    # In the common case the crest is on the hanging-wall (HW) side and
    # the base is on the footwall (FW) side, so the HW regional slope is
    # projected to the crest and the FW regional slope to the base.
    #
    # In rare cases the base appears at a more negative distance than the
    # crest (base comes before crest in the cross-section).  In that
    # situation the crest is on the FW side and the base is on the HW
    # side, so the projection directions are swapped to follow the actual
    # crest-base ordering.
    if x1 <= x2:
        # Normal ordering: crest (x1) is more negative (HW side),
        # base (x2) is more positive (FW side).
        z_hw_proj = mh * x1 + bh
        z_fw_proj = mf * x2 + bf
    else:
        # Base comes before crest: crest (x1) is more positive (FW side),
        # base (x2) is more negative (HW side).  Swap projection direction
        # to match the actual geometry.
        z_hw_proj = mh * x2 + bh   # HW projection at base location
        z_fw_proj = mf * x1 + bf   # FW projection at crest location
    projected_H = abs(z_hw_proj - z_fw_proj)

    hw_elev = np.nanmean(hw_z)
    fw_elev = np.nanmean(fw_z)
    is_reverse_scarp = (hw_elev > fw_elev) and (scarp_height > 0)
    
    return {
        "z_crest_primary": float(z_crest_profile),
        "z_base_primary": float(z_base_profile),
        "d_crest": float(x1),
        "d_base": float(x2),
        "z_hw_projected": float(z_hw_proj),
        "z_fw_projected": float(z_fw_proj),
        "projected_H": float(projected_H),
        "projected_hw_fw_distance": float(abs(x2 - x1)),
        "primary_throw": float(scarp_height),
        "true_vertical_displacement": float(VS),
        "vertical_separation": float(VS),
        "VS_max": float(VS_max),
        "VS_min": float(VS_min),
        "total_cumulative_throw": float(total_cumulative_throw),
        "net_slip": float(DS) if np.isfinite(DS) else np.nan,
        "heave": float(HD) if np.isfinite(HD) else np.nan,
        "vertical_displacement": float(VD) if np.isfinite(VD) else np.nan,
        "scarp_width_m": float(scarp_width),
        "scarp_slope_deg": float(scarp_slope_deg),
        "max_local_slope_deg": float(max_local_slope_deg),
        "HW_regional_slope": float(mh),
        "FW_regional_slope": float(mf),
        "scarp_slope": float(ms),
        "HW_regional_slope_deg": float(hw_slope_deg),
        "FW_regional_slope_deg": float(fw_slope_deg),
        "is_reverse_scarp": bool(is_reverse_scarp),
    }

def process_trace(trace, dem_data, transform, dem_shape, nodata,
                  station_spacing=STATION_SPACING_M):
    """Process a single trace: extract profiles and compute all parameters.

    All cross-sections use the **same average azimuth** for the entire
    trace (stored in ``trace["azimuth_deg"]``), so that every profile
    is perpendicular to the segment-averaged direction rather than to a
    noisy per-station local azimuth.
    """
    coords = trace["coords"]
    trace_id = trace["trace_id"]

    stations, distances, azimuths = resample_trace(coords, station_spacing)
    n_stations = len(stations)
    print(f"\n[Trace {trace_id}] {n_stations} stations at {station_spacing} m spacing "
          f"(trace length: {trace['length_m']:.0f} m)")

    # --- Use the segment-averaged azimuth for ALL cross sections ---
    # The per-station azimuths from resample_trace are noisy and produce
    # inconsistent profile orientations.  The average azimuth (from
    # first to last point of the trace) gives a stable, consistent
    # direction for all profiles.
    avg_azimuth = trace.get("azimuth_deg", azimuths.mean())
    print(f"  Average azimuth: {avg_azimuth:.1f} deg (used for all cross sections)")

    results = []
    profiles_cache = []

    for i in range(n_stations):
        x, y = stations[i]
        az = avg_azimuth  # uniform azimuth for all stations
        d_along = distances[i]

        prof_d, prof_z, perp_azimuth_used = extract_perpendicular_profile(
            dem_data, transform, (x, y), az,
            half_length_m=PROFILE_HALF_LENGTH_M,
            resolution_m=PROFILE_RESOLUTION_M,
            dem_shape=dem_shape, nodata=nodata,
        )

        if i < 3:
            _diagnose_profile(prof_d, prof_z, trace_id, i)

        profiles_cache.append((prof_d.copy(), prof_z.copy()))

        valid_frac = np.isfinite(prof_z).sum() / len(prof_z)
        if valid_frac < 0.5:
            results.append(_empty_result(trace_id, d_along, x, y))
            continue

        throw = calculate_reverse_fault_parameters(prof_d, prof_z, dip_deg=None)

        result = {
            "trace_id": trace_id,
            "distance_along": float(d_along),
            "x": float(x),
            "y": float(y),
            "azimuth": float(az),
            "perp_azimuth_used": float(perp_azimuth_used),
            "HW_regional_slope_deg": throw.get("HW_regional_slope_deg", np.nan),
            "FW_regional_slope_deg": throw.get("FW_regional_slope_deg", np.nan),
            "HW_regional_slope": throw.get("HW_regional_slope", np.nan),
            "FW_regional_slope": throw.get("FW_regional_slope", np.nan),
            "scarp_slope": throw.get("scarp_slope", np.nan),
            "local_relief": throw.get("VS_max", np.nan) - throw.get("VS_min", np.nan),
            "z_crest_primary": throw["z_crest_primary"],
            "z_base_primary": throw["z_base_primary"],
            "d_crest": throw.get("d_crest", np.nan),
            "d_base": throw.get("d_base", np.nan),
            "z_hw_projected": throw["z_hw_projected"],
            "z_fw_projected": throw["z_fw_projected"],
            "projected_H": throw.get("projected_H", np.nan),
            "projected_hw_fw_distance": throw.get("projected_hw_fw_distance", np.nan),
            "primary_throw": throw["primary_throw"],
            "true_vertical_displacement": throw["true_vertical_displacement"],
            "vertical_separation": throw["vertical_separation"],
            "total_cumulative_throw": throw.get("total_cumulative_throw", np.nan),
            "VS_max": throw.get("VS_max", np.nan),
            "VS_min": throw.get("VS_min", np.nan),
            "net_slip": throw["net_slip"],
            "heave": throw["heave"],
            "vertical_displacement": throw.get("vertical_displacement", np.nan),
            "scarp_width_m": throw["scarp_width_m"],
            "scarp_slope_deg": throw["scarp_slope_deg"],
            "max_local_slope_deg": throw.get("max_local_slope_deg", np.nan),
            "is_reverse_scarp": throw.get("is_reverse_scarp", False),
            "profile_quality": float(valid_frac),
            "strike": np.nan,
            "dip": np.nan,
            "dip_direction": np.nan,
            "plane_quality_rms": np.nan,
        }

        results.append(result)

    _batch_strike_dip(results, profiles_cache, stations, distances)

    for i, r in enumerate(results):
        if np.isfinite(r.get("dip", np.nan)) and r["dip"] > 1.0:
            prof_d, prof_z = profiles_cache[i]
            updated_throw = calculate_reverse_fault_parameters(
                prof_d, prof_z, dip_deg=r["dip"]
            )
            r["net_slip"] = updated_throw["net_slip"]
            r["heave"] = updated_throw["heave"]
            r["vertical_displacement"] = updated_throw.get("vertical_displacement", np.nan)

    return results, profiles_cache


def _batch_strike_dip(results, profiles_cache, stations, distances):
    """Calculate strike/dip from sliding window of crest-base pairs.

    Strike is derived from the local trace azimuth (circular mean over the
    sliding window) rather than from crest-base 3D vectors, because the
    crest-base vector direction is unreliable (depends on which side is
    HW/FW and on the quality of crest/base picks).  Dip is still estimated
    from the 3D geometry of crest-base pairs via
    ``calculate_scarp_geometry``.
    """
    n = len(results)
    window_radius = 5

    for i in range(n):
        lo = max(0, i - window_radius)
        hi = min(n, i + window_radius + 1)

        crest_pts = []
        base_pts = []
        local_azimuths = []

        for j in range(lo, hi):
            r = results[j]
            if not isinstance(r, dict):
                continue

            z_crest = r.get("z_crest_primary", np.nan)
            z_base = r.get("z_base_primary", np.nan)
            d_crest = r.get("d_crest", np.nan)
            d_base = r.get("d_base", np.nan)
            az = r.get("azimuth", np.nan)

            if np.isfinite(az) and az != 0:
                local_azimuths.append(az)

            if (np.isfinite(z_crest) and
                np.isfinite(z_base) and
                np.isfinite(d_crest) and
                np.isfinite(d_base)):

                station_x, station_y = stations[j]
                perp_az = r.get("perp_azimuth_used", np.nan)

                if np.isfinite(perp_az):
                    rad = np.radians(perp_az)

                    cx = station_x + d_crest * np.sin(rad)
                    cy = station_y + d_crest * np.cos(rad)
                    crest_pts.append([cx, cy, z_crest])

                    bx = station_x + d_base * np.sin(rad)
                    by = station_y + d_base * np.cos(rad)
                    base_pts.append([bx, by, z_base])

        # --- Strike from local trace azimuth (circular mean) ---
        if local_azimuths:
            az_arr = np.array(local_azimuths)
            sin_mean = np.mean(np.sin(np.radians(az_arr)))
            cos_mean = np.mean(np.cos(np.radians(az_arr)))
            mean_strike = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
            results[i]["strike"] = float(mean_strike)
            results[i]["dip_direction"] = (mean_strike + 90.0) % 360.0
        else:
            results[i]["strike"] = np.nan
            results[i]["dip_direction"] = np.nan

        # --- Dip from 3D crest-base geometry ---
        if len(crest_pts) >= 3 and len(base_pts) >= 3:
            geom = calculate_scarp_geometry(crest_pts, base_pts)

            results[i]["dip"] = geom["dip"]
            results[i]["plane_quality_rms"] = geom["plane_quality_rms"]

            if np.isfinite(geom["dip"]) and geom["dip"] > 5:
                true_vd = results[i].get("true_vertical_displacement", np.nan)
                if np.isfinite(true_vd):
                    dip_rad = np.radians(geom["dip"])
                    results[i]["net_slip"] = float(true_vd / np.sin(dip_rad))
                    results[i]["heave"] = float(true_vd / np.tan(dip_rad))


def _empty_result(trace_id, d_along, x, y):
    return {
        "trace_id": trace_id,
        "distance_along": float(d_along),
        "x": float(x), "y": float(y),
        "azimuth": np.nan,
        "perp_azimuth_used": np.nan,
        "HW_regional_slope_deg": np.nan,
        "FW_regional_slope_deg": np.nan,
        "HW_regional_slope": np.nan,
        "FW_regional_slope": np.nan,
        "local_relief": np.nan,
        "z_crest_primary": np.nan,
        "z_base_primary": np.nan,
        "d_crest": np.nan,
        "d_base": np.nan,
        "z_hw_projected": np.nan,
        "z_fw_projected": np.nan,
        "projected_H": np.nan,
        "projected_hw_fw_distance": np.nan,
        "primary_throw": np.nan,
        "true_vertical_displacement": np.nan,
        "vertical_separation": np.nan,
        "total_cumulative_throw": np.nan,
        "net_slip": np.nan,
        "heave": np.nan,
        "scarp_width_m": np.nan,
        "scarp_slope_deg": np.nan,
        "max_local_slope_deg": np.nan,
        "profile_quality": 0.0,
        "strike": np.nan, "dip": np.nan,
        "dip_direction": np.nan, "plane_quality_rms": np.nan,
    }


# ===================================================================
# 8b.  1D DIFFUSION EQUATION SOLUTION  (kappa * t)
# ===================================================================

def compute_diffused_profile(d, hw_slope, hw_intercept, fw_slope,
                             fw_intercept, x_crest, x_base, H,
                             kappa_t):
    """Compute the 1-D diffusion equation solution for a scarp profile.

    The 1-D linear diffusion equation  dz/dt = kappa * d2z/dx2  with
    piecewise-linear initial conditions (HW regional slope on the left,
    a steep scarp face, and FW regional slope on the right) has a
    well-known analytic solution expressed via error functions.

    The initial topography is:
      * Left of crest  (x < x_crest):  z = hw_slope * x + hw_intercept
      * Right of base  (x > x_base):   z = fw_slope * x + fw_intercept
      * Scarp face (x_crest <= x <= x_base): linear interpolation
        between (x_crest, z_crest) and (x_base, z_base)

    The far-field boundary conditions are the two regional slopes.
    The diffused profile is::

        z(x, t) = hw_line(x)
                  + (fw_line(x) - hw_line(x))
                    * 0.5 * [1 + erf((x - x_mid) / (2*sqrt(kappa*t)))]

    where ``x_mid`` is the midpoint of the scarp.  This smoothly
    transitions from HW slope to FW slope over a width controlled by
    ``sqrt(kappa*t)``.

    Parameters
    ----------
    d : array of distances along the cross-section
    hw_slope, hw_intercept : slope and intercept of HW regional line
    fw_slope, fw_intercept : slope and intercept of FW regional line
    x_crest, x_base : crest and base distances
    H : scarp height (z_crest - z_base), used for the vertical offset
    kappa_t : diffusivity * time (m^2).  If <= 0, returns the initial
              (undiffused) piecewise-linear profile.

    Returns
    -------
    z_diffused : array of elevations after diffusion
    """
    z_hw = hw_slope * d + hw_intercept
    z_fw = fw_slope * d + fw_intercept

    x_mid = (x_crest + x_base) / 2.0

    if kappa_t <= 0:
        # Undiffused: piecewise linear
        z_scarp = np.interp(d, [x_crest, x_base],
                            [hw_slope * x_crest + hw_intercept,
                             fw_slope * x_base + fw_intercept])
        z_init = np.where(d < x_crest, z_hw,
                          np.where(d > x_base, z_fw, z_scarp))
        return z_init

    sigma = 2.0 * np.sqrt(kappa_t)
    # Smooth transition from HW to FW using error function
    transition = 0.5 * (1.0 + erf((d - x_mid) / sigma))
    z_diffused = z_hw + (z_fw - z_hw) * transition

    return z_diffused


def estimate_kappa_t_for_45deg(hw_slope, fw_slope, H, x_crest, x_base):
    """Estimate kappa*t such that the maximum scarp-face slope is 45 deg.

    For a 1-D linear diffusion model with an initial scarp of height H,
    the maximum slope of the diffused profile is approximately::

        max_slope ~ H / (2 * sqrt(pi * kappa * t))

    Setting max_slope = tan(45 deg) = 1 and solving for kappa*t::

        kappa_t ~ H^2 / (4 * pi)

    Parameters
    ----------
    hw_slope, fw_slope : regional slopes (dimensionless, rise/run)
    H : scarp height (projected_H or primary_throw)
    x_crest, x_base : crest and base distances (used to compute the
        total elevation offset between HW and FW lines at the midpoint)

    Returns
    -------
    kappa_t : float, estimated kappa*t in m^2
    """
    x_mid = (x_crest + x_base) / 2.0
    delta_z = abs((hw_slope - fw_slope) * x_mid) + H
    if delta_z < 0.1:
        return 0.0

    kappa_t = delta_z**2 / (4.0 * np.pi)
    return kappa_t


def kappa_t_to_years(kappa_t, kappa):
    """Convert kappa*t (m^2) to estimated age in years.

    Since kappa_t = kappa * t,  t = kappa_t / kappa.

    Parameters
    ----------
    kappa_t : float, diffusivity * time in m^2
    kappa : float, diffusivity in m^2/yr

    Returns
    -------
    t_years : float, estimated age in years
    """
    if kappa <= 0 or kappa_t <= 0:
        return 0.0
    return kappa_t / kappa


# ===================================================================
# 9.  VISUALISATION: CROSS-SECTION PANELS
# ===================================================================


def plot_cross_sections(results, profiles_cache, trace_id, output_dir):
    """Generate improved cross-section panel plots for a trace."""
    n = len(results)
    if n == 0:
        return

    cs_dir = output_dir / "cross_sections" / f"segment_{trace_id}"
    safe_makedirs(cs_dir)

    # Select profiles at regular intervals
    d_along = np.array([r["distance_along"] for r in results])
    max_d = d_along.max() if len(d_along) > 0 else 0

    selected_indices = []
    target_d = 0.0
    while target_d <= max_d:
        idx = np.argmin(np.abs(d_along - target_d))
        if idx not in selected_indices:
            selected_indices.append(idx)
        target_d += CROSS_SECTION_INTERVAL_M

    # Always include the last profile
    if (n - 1) not in selected_indices:
        selected_indices.append(n - 1)

    panel_size = CROSS_SECTION_PANEL_SIZE
    panel_num = 0

    for panel_start in range(0, len(selected_indices), panel_size):
        panel_indices = selected_indices[panel_start:panel_start + panel_size]
        n_plots = len(panel_indices)

        fig, axes = plt.subplots(n_plots, 1, figsize=(22, 4.5 * n_plots),
                                 squeeze=False)

        for ax_i, idx in enumerate(panel_indices):
            ax = axes[ax_i, 0]
            r = results[idx]
            prof_d, prof_z = profiles_cache[idx]

            valid = np.isfinite(prof_z)
            if not np.any(valid):
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Plot elevation profile.
            ax.plot(prof_d, prof_z, "k-", linewidth=1.0, alpha=0.8,
                    label="Elevation profile", zorder=2)

            # Plot regional slope projections.
            hw_slope = r.get("HW_regional_slope", np.nan)
            fw_slope = r.get("FW_regional_slope", np.nan)

            hw_intercept = np.nan
            fw_intercept = np.nan

            if np.isfinite(hw_slope):
                hw_d_mask = (prof_d < -50.0) & valid
                if hw_d_mask.sum() > 3:
                    hw_vals = prof_z[hw_d_mask]
                    hw_d_vals = prof_d[hw_d_mask]
                    hw_intercept = np.nanmedian(hw_vals - hw_slope * hw_d_vals)
                    plot_d = np.linspace(prof_d.min(), prof_d.max(), 200)
                    plot_z_hw = hw_slope * plot_d + hw_intercept
                    ax.plot(plot_d, plot_z_hw, "r--", linewidth=1.5, alpha=0.7,
                            label="HW regional slope", zorder=3)

            if np.isfinite(fw_slope):
                fw_d_mask = (prof_d > 50.0) & valid
                if fw_d_mask.sum() > 3:
                    fw_vals = prof_z[fw_d_mask]
                    fw_d_vals = prof_d[fw_d_mask]
                    fw_intercept = np.nanmedian(fw_vals - fw_slope * fw_d_vals)
                    plot_d = np.linspace(prof_d.min(), prof_d.max(), 200)
                    plot_z_fw = fw_slope * plot_d + fw_intercept
                    ax.plot(plot_d, plot_z_fw, "b--", linewidth=1.5, alpha=0.7,
                            label="FW regional slope", zorder=3)

            # Mark detected crest/base.
            if np.isfinite(r.get("z_crest_primary", np.nan)):
                ax.axhline(r["z_crest_primary"], color="red", linestyle=":",
                           alpha=0.6, linewidth=1.0)
                ax.scatter([r.get("d_crest", 0.0)], [r["z_crest_primary"]],
                           color="red", s=80, zorder=5, marker="^",
                           edgecolors="darkred", linewidth=1.0,
                           label="Crest (detected)")

            if np.isfinite(r.get("z_base_primary", np.nan)):
                ax.axhline(r["z_base_primary"], color="blue", linestyle=":",
                           alpha=0.6, linewidth=1.0)
                ax.scatter([r.get("d_base", 0.0)], [r["z_base_primary"]],
                           color="blue", s=80, zorder=5, marker="v",
                           edgecolors="darkblue", linewidth=1.0,
                           label="Base (detected)")

            # Mark projected HW/FW at crest and base positions.
            if (np.isfinite(r.get("z_hw_projected", np.nan)) and
                    np.isfinite(r.get("z_fw_projected", np.nan))):
                d_crest = r.get("d_crest", 0.0)
                d_base = r.get("d_base", 0.0)
                #ax.scatter([d_crest], [r["z_hw_projected"]],color="green", s=60, zorder=5, marker="s",edgecolors="darkgreen", linewidth=1.0,label="Projected HW")
                #ax.scatter([d_base], [r["z_fw_projected"]],color="lime", s=60, zorder=5, marker="D",edgecolors="darkgreen", linewidth=1.0,label="Projected FW")
                ax.plot([d_crest, d_base],
                        [r["z_hw_projected"], r["z_fw_projected"]],
                        color="green", linewidth=2.5, linestyle="-",
                        alpha=0.7, zorder=4)

                proj_H = r.get("projected_H", np.nan)
                proj_dist = r.get("projected_hw_fw_distance", np.nan)
                #if np.isfinite(proj_H):
                #    mid_z = (r["z_hw_projected"] + r["z_fw_projected"]) / 2.0
                #    mid_d = (d_crest + d_base) / 2.0
                #    ax.annotate(
                #        f"Projected H: {proj_H:.1f}m\n"
                #        f"Proj. dist: {proj_dist:.1f}m",
                #        xy=(mid_d + 20.0, mid_z),
                #        fontsize=14, color="darkgreen",
                #        bbox=dict(boxstyle="round,pad=0.3",
                #                  facecolor="lightgreen", alpha=0.7),
                #    )

            # Shade scarp zone.
            if (np.isfinite(r.get("d_crest", np.nan)) and
                    np.isfinite(r.get("d_base", np.nan))):
                ax.axvspan(r["d_crest"], r["d_base"], alpha=0.1,
                           color="orange", zorder=1, label="Scarp zone")

            # Plot 1-D diffusion model curves.
            x_crest = r.get("d_crest", np.nan)
            x_base = r.get("d_base", np.nan)
            H = r.get("projected_H", r.get("primary_throw", np.nan))
            if (np.isfinite(hw_intercept) and np.isfinite(fw_intercept) and
                    np.isfinite(x_crest) and np.isfinite(x_base) and
                    np.isfinite(H) and H > 0):
                kappa_values = [0.001, 0.005]  # m^2/yr
                # A single kappa_t is estimated from the scarp geometry;
                # each kappa value then implies a different age t = kt / kappa.
                kt = estimate_kappa_t_for_45deg(
                    hw_slope, fw_slope, H, x_crest, x_base)

                plot_d_fine = np.linspace(prof_d.min(), prof_d.max(), 500)
                diff_colors = ["#e41a1c", "#377eb8"]
                for ki, kv in enumerate(kappa_values):
                    if kt > 0:
                        t_years = kappa_t_to_years(kt, kv)
                        z_diff = compute_diffused_profile(
                            plot_d_fine,
                            hw_slope, hw_intercept,
                            fw_slope, fw_intercept,
                            x_crest, x_base, H, kt)
                        if t_years >= 1000:
                            age_str = f"{t_years/1000:.1f} kyr"
                        else:
                            age_str = f"{t_years:.0f} yr"
                        ax.plot(plot_d_fine, z_diff,
                                color=diff_colors[ki], linewidth=1.5,
                                alpha=0.8, linestyle="-.",
                                label=f"k={kv:.3f} m2/yr -> age {age_str}",
                                zorder=6)

                # Subtitle with kappa and age estimates
                t1 = kappa_t_to_years(kt, 0.001)
                t2 = kappa_t_to_years(kt, 0.005)
                def _fmt_age(yr):
                    if yr >= 1000:
                        return f"{yr/1000:.1f} kyr"
                    return f"{yr:.0f} yr"
                subtitle = (f"kt={kt:.1f} m2  |  "
                            f"k=0.001 -> age={_fmt_age(t1)}  |  "
                            f"k=0.005 -> age={_fmt_age(t2)}")
                ax.annotate(subtitle, xy=(0.02, 0.02),
                            xycoords="axes fraction",
                            fontsize=11, style="italic", color="gray",
                            verticalalignment="bottom")

            title_parts = [
                f"d={r['distance_along']:.0f}m",
                f"Projected H={r.get('projected_H', np.nan):.1f}m",
                f"Proj. dist={r.get('projected_hw_fw_distance', np.nan):.1f}m",
                f"True VD={r.get('true_vertical_displacement', np.nan):.1f}m",
            ]
            if np.isfinite(r.get("net_slip", np.nan)):
                title_parts.append(f"Net slip={r['net_slip']:.1f}m")
            title_parts.extend([
                f"HW slope={r.get('HW_regional_slope_deg', np.nan):.1f} deg",
                f"FW slope={r.get('FW_regional_slope_deg', np.nan):.1f} deg",
                f"Dip={r.get('dip', np.nan):.1f} deg",
            ])

            ax.set_title("  |  ".join(title_parts), fontsize=16, pad=8)
            ax.set_xlabel("Distance along transect (m)", fontsize=14)
            ax.set_ylabel("Elevation (m)", fontsize=14)
            ax.tick_params(labelsize=14)
            ax.legend(fontsize=10, loc="center left",
                      bbox_to_anchor=(1.01, 0.5),
                      framealpha=0.9, ncol=1)
            ax.grid(True, alpha=0.3, linestyle="--")

            z_min = np.nanmin(prof_z) - 5.0
            z_max = np.nanmax(prof_z) + 5.0
            ax.set_ylim(z_min, z_max)

        fig.suptitle(f"Trace {trace_id} - Cross-sections (panel {panel_num + 1})",
                 fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.78, 0.97])

        out_path = cs_dir / f"panel_{panel_num + 1:03d}.png"
        _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.3)
        panel_num += 1

    print(f"  [Cross-sections] {panel_num} panels saved to {cs_dir}")


# ===================================================================
# 10.  ALONG-SCARP PARAMETER PROFILES
# ===================================================================

def plot_along_scarp_profiles(results, trace_id, output_dir):
    """Plot parameters vs. distance along scarp."""
    if not results:
        return

    df = pd.DataFrame(results)
    d = df["distance_along"].values

    asp_dir = output_dir / "along_scarp_profiles"
    safe_makedirs(asp_dir)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Subplot 1: Scarp height and throw
    ax = axes[0]
    ax.plot(d, df["primary_throw"], "r-", linewidth=1.0, label="Primary throw")
    ax.plot(d, df["total_cumulative_throw"], "b-", linewidth=1.0,
            label="Total cumulative throw", alpha=0.7)
    ax.set_ylabel("Throw (m)", fontsize=14)
    ax.legend(fontsize=14, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Trace {trace_id} - Along-scarp parameter profiles",
                 fontsize=16)

    # Subplot 2: Strike and dip
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(d, df["strike"], "k-", linewidth=1.0, label="Strike")
    ax2.plot(d, df["dip"], "r-", linewidth=1.0, label="Dip", alpha=0.7)
    ax.set_ylabel("Strike (deg)", fontsize=14)
    ax2.set_ylabel("Dip (deg)", fontsize=14, color="r")
    ax.legend(fontsize=14, loc="upper left")
    ax2.legend(fontsize=14, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Subplot 3: Regional slopes
    ax = axes[2]
    ax.plot(d, df["HW_regional_slope_deg"], "r-", linewidth=1.0,
            label="HW slope")
    ax.plot(d, df["FW_regional_slope_deg"], "b-", linewidth=1.0,
            label="FW slope")
    ax.set_ylabel("Regional slope (deg)", fontsize=14)
    ax.legend(fontsize=14, loc="best")
    ax.grid(True, alpha=0.3)

    # Subplot 4: Quality metrics
    ax = axes[3]
    ax.plot(d, df["profile_quality"], "k-", linewidth=1.0,
            label="Profile quality")
    ax.set_ylabel("Quality", fontsize=14)
    ax.set_xlabel("Distance along trace (m)", fontsize=14)
    ax.legend(fontsize=14, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = asp_dir / f"segment_{trace_id}_parameters.png"
    _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"  [Along-scarp] Saved to {out_path}")


# ===================================================================
# 10b.  PER-ORIGINAL-TRACE PARAMETER PROFILES
# ===================================================================

# Colour palette for segment lines in per-trace figures.
_PER_TRACE_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#e5c494", "#b3b3b3", "#8dd3c7", "#ffffb3", "#bebada",
    "#fb8072", "#80b1d3", "#fdb462", "#d9d9d9", "#bc80bd",
]


def _sort_segments_north_to_south(segments):
    """Return segments sorted by mean northing (descending = north first)."""
    def mean_y(seg):
        results = seg.get("results", [])
        ys = [r.get("y", np.nan) for r in results
              if np.isfinite(r.get("y", np.nan))]
        if not ys:
            coords = seg.get("trace", {}).get("coords", np.array([]))
            if len(coords) > 0:
                return float(np.mean(coords[:, 1]))
            return -np.inf
        return float(np.mean(ys))
    return sorted(segments, key=mean_y, reverse=True)


def _segment_mean_y(seg):
    """Mean northing of a segment (from results or raw coords)."""
    results = seg.get("results", [])
    ys = [r.get("y", np.nan) for r in results
          if np.isfinite(r.get("y", np.nan))]
    if ys:
        return float(np.mean(ys))
    coords = seg.get("trace", {}).get("coords", np.array([]))
    if len(coords) > 0:
        return float(np.mean(coords[:, 1]))
    return 0.0


def _circular_mean_deg(angles_deg):
    """Circular mean of angles in degrees (0-360).

    Returns 0 for 350+10, not 180.  Excludes NaN.
    """
    valid = np.array([a for a in angles_deg
                     if np.isfinite(a) and a != 0.0], dtype=float)
    if len(valid) == 0:
        valid = np.array([a for a in angles_deg if np.isfinite(a)],
                         dtype=float)
    if len(valid) == 0:
        return np.nan
    rad = np.deg2rad(valid)
    mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    mean_deg = np.rad2deg(mean_rad)
    if mean_deg < 0:
        mean_deg += 360.0
    return mean_deg


def _plot_ref_lines(ax, values, label_prefix="", circular=False):
    """Draw horizontal reference lines for average and maximum.

    Excludes NaN values.  If circular=True, uses circular mean for
    azimuth-like quantities (so 350+10 -> 0, not 180).  Maximum is
    computed on the raw values (not circular).
    """
    valid_raw = np.array([v for v in values if np.isfinite(v)])
    if len(valid_raw) == 0:
        return

    if circular:
        avg = _circular_mean_deg(valid_raw)
    else:
        valid_nz = valid_raw[valid_raw != 0.0]
        if len(valid_nz) == 0:
            return
        avg = float(np.mean(valid_nz))

    mx = float(np.max(valid_raw))

    ax.axhline(avg, color=_BROWN, linestyle="--", alpha=0.7, linewidth=2.0,
               label=f"{label_prefix}avg={avg:.1f}")
    ax.axhline(mx, color=_BROWN, linestyle=":", alpha=0.6, linewidth=2.5,
              label=f"{label_prefix}max={mx:.1f}")


# ===================================================================
# 10b.1  LOCAL DIP DIRECTION ESTIMATION (per-segment plane fitting)
# ===================================================================

def _estimate_local_dip_direction(xs, ys, elev, window_half=150.0,
                                   step=50.0):
    """Estimate dip direction along a fault trace using sliding-window
    plane fitting.

    At each window centre, fit a plane  z = a*x + b*y + c  to the
    elevation points within the window.  The dip direction is the
    direction of steepest descent (opposite to the gradient).  The dip
    angle is  arctan(sqrt(a^2 + b^2)).

    Valley/ridge detection: points where local curvature of the fault
    trace exceeds a threshold are flagged as potential valley/ridge
    intersections.  The V-shape consistency is checked by comparing
    the curvature sign change before and after the intersection.

    Parameters
    ----------
    xs, ys, elev : arrays of easting, northing, elevation along the fault
    window_half : half-window size in metres
    step : sampling step along the trace (metres)

    Returns
    -------
    dict with keys:
        centres_x, centres_y : window centre coordinates
        dip_direction_deg : dip direction at each window
        dip_angle_deg : dip angle at each window
        confidence : R^2 of the plane fit (0-1)
        is_valley : boolean array, True at valley/ridge intersections
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    elev = np.asarray(elev, dtype=float)
    n = len(xs)
    if n < 5:
        return None

    # Compute cumulative distance along trace for windowing
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg_len = np.sqrt(dx**2 + dy**2)
    cum_len = np.zeros(n)
    cum_len[1:] = np.cumsum(seg_len)
    total_len = cum_len[-1]

    if total_len < window_half * 2:
        # Entire trace fits in one window
        window_centres = [total_len / 2.0]
    else:
        window_centres = np.arange(window_half, total_len - window_half + 1,
                                   step)

    centres_x = []
    centres_y = []
    dip_direction_deg = []
    dip_angle_deg = []
    confidence = []
    is_valley_flags = []

    # --- Valley/ridge detection along the trace ---
    # Use curvature of the fault trace itself: where the trace
    # bends sharply, it may indicate a valley/ridge intersection.
    if n >= 7:
        # Smooth the trace coordinates slightly to reduce noise
        from scipy.ndimage import uniform_filter1d
        smooth_x = uniform_filter1d(xs, size=5, mode="nearest")
        smooth_y = uniform_filter1d(ys, size=5, mode="nearest")
        # Bearing at each point
        ddx = np.diff(smooth_x)
        ddy = np.diff(smooth_y)
        bearings = np.arctan2(ddy, ddx)
        # Curvature = change in bearing
        bearing_diff = np.diff(bearings)
        # Wrap to [-pi, pi]
        bearing_diff = np.arctan2(np.sin(bearing_diff),
                                  np.cos(bearing_diff))
        # Absolute curvature
        abs_curv = np.abs(bearing_diff)
        # Pad to length n
        curv = np.zeros(n)
        curv[1:-1] = abs_curv
        # Flag high-curvature points (valleys/ridges)
        curv_threshold = np.percentile(curv[curv > 0], 75) if np.any(curv > 0) else 1.0
        valley_flags = curv > curv_threshold
    else:
        valley_flags = np.zeros(n, dtype=bool)
        curv = np.zeros(n)

    for wc in window_centres:
        # Select points within the window
        mask = (cum_len >= wc - window_half) & (cum_len <= wc + window_half)
        if np.sum(mask) < 5:
            continue

        wx = xs[mask]
        wy = ys[mask]
        wz = elev[mask]

        # Centre the coordinates to improve conditioning
        cx = np.mean(wx)
        cy = np.mean(wy)
        wx_c = wx - cx
        wy_c = wy - cy
        wz_c = wz - np.mean(wz)

        # Fit plane z = a*x + b*y + c via least squares
        A = np.column_stack([wx_c, wy_c, np.ones(len(wx_c))])
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, wz_c,
                                                           rcond=None)
            a, b, c_fit = result
        except np.linalg.LinAlgError:
            continue

        # Gradient magnitude
        grad_mag = np.sqrt(a**2 + b**2)
        dip_angle = np.degrees(np.arctan(grad_mag))

        # Dip direction: direction of steepest descent (opposite to
        # the gradient).  Gradient points uphill, so dip direction
        # is atan2(-a, -b) in (east, north) convention.
        dip_dir = np.degrees(np.arctan2(-a, -b))
        if dip_dir < 0:
            dip_dir += 360.0

        # Confidence: R^2 of the fit
        wz_pred = A @ result
        ss_res = np.sum((wz_c - wz_pred)**2)
        ss_tot = np.sum(wz_c**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Check if any valley/ridge points are in this window
        has_valley = np.any(valley_flags[mask])

        centres_x.append(cx)
        centres_y.append(cy)
        dip_direction_deg.append(dip_dir)
        dip_angle_deg.append(dip_angle)
        confidence.append(max(0.0, min(1.0, r2)))
        is_valley_flags.append(has_valley)

    return {
        "centres_x": np.array(centres_x),
        "centres_y": np.array(centres_y),
        "dip_direction_deg": np.array(dip_direction_deg),
        "dip_angle_deg": np.array(dip_angle_deg),
        "confidence": np.array(confidence),
        "is_valley": np.array(is_valley_flags),
    }


# ===================================================================
# 10a.1  FOLDED-FAULT DIP ESTIMATION (V-rule model)
# ===================================================================


def estimate_folded_fault_dip(segments, dem_data, transform, dem_shape,
                               dem10m_data=None, window_half=500.0):
    """Estimate fault dip using the folded-fault V-rule model.

    Implements a two-wing folded-fault model where the fault trace forms
    a V-shape in plan view.  For each sliding window along the trace:

    1. PCA determines the along-trace (s) and perpendicular (n) axes.
    2. A V-shape is detected when the perpendicular projection n has a
       clear interior extremum (apex).
    3. If a V-shape is found, two separate wing planes are fitted:
         z = a_i * s + b_i * n + c_i   (i = 1, 2)
       The dip on each wing is  arctan(sqrt(a_i^2 + b_i^2)).
    4. If no V-shape, a single plane is fitted for the entire window.
    5. A V-rule constrained dip is also computed from the outcrop
       geometry:  sin(alpha) = H / W  where H is the vertical relief
       (primary_throw) and W is the scarp width.

    The mathematical framework follows the "Rule of V" for folded faults:

    - Terrain:  h_terrain(s, n) = slope_s * s + const  (flat in n)
    - Fault wing plane:  h_fault = a * s + b * n + c
    - At the outcrop line:  a + b * (dn/ds) = slope_s
    - Dip:  tan(alpha) = sqrt(a^2 + b^2)
    - Outcrop width:  W = H / sin(alpha)

    When the fault is folded (two wings), each wing has its own
    (a_i, b_i) satisfying  a_i + b_i * (dn/ds)_i = slope_s.  The
    outcrop height constraint (H ~ 1 m) limits the horizontal width
    W = H / sin(alpha), providing a cross-check on the plane-fit dip.

    Parameters
    ----------
    segments : list of segment dicts, each with "results" (list of
        station dicts containing "x", "y", "primary_throw",
        "scarp_width_m", etc.)
    dem_data : 2-D DEM elevation array
    transform : affine transform for the DEM
    dem_shape : (rows, cols) of the DEM
    dem10m_data : optional fallback DEM for nodata patches
    window_half : half-window size in metres for the sliding analysis

    Returns
    -------
    dict mapping trace_id -> list of per-station dicts with keys:
        y, x, folded_dip_deg, v_rule_dip_deg, terrain_slope,
        cross_slope_b, wing_label, confidence
    """
    result = {}

    for seg in segments:
        results = seg.get("results", [])
        trace_id = seg.get("trace_id", "")
        if len(results) < 5:
            continue

        # --- Extract station coordinates ---
        xs_all = np.array([r.get("x", np.nan) for r in results],
                          dtype=float)
        ys_all = np.array([r.get("y", np.nan) for r in results],
                          dtype=float)

        valid_xy = np.isfinite(xs_all) & np.isfinite(ys_all)
        if int(valid_xy.sum()) < 5:
            continue

        xs = xs_all[valid_xy]
        ys = ys_all[valid_xy]
        results_xy = [results[i] for i in range(len(results))
                      if valid_xy[i]]

        # --- Sample DEM elevations at station positions ---
        zs = _sample_dem_elevations_patched(
            xs, ys, dem_data, transform, dem_shape, dem10m_data)

        valid_z = np.isfinite(zs)
        if int(valid_z.sum()) < 5:
            continue

        xs = xs[valid_z]
        ys = ys[valid_z]
        zs = zs[valid_z]
        results_valid = [results_xy[i] for i in range(len(results_xy))
                         if valid_z[i]]

        n_st = len(xs)

        # --- Cumulative distance along trace for windowing ---
        dx = np.diff(xs)
        dy = np.diff(ys)
        seg_len = np.sqrt(dx ** 2 + dy ** 2)
        cum_len = np.zeros(n_st)
        cum_len[1:] = np.cumsum(seg_len)
        total_len = cum_len[-1]

        if total_len < window_half * 2:
            window_centres = [total_len / 2.0]
        else:
            step = min(50.0, total_len / 20)
            window_centres = np.arange(window_half,
                                       total_len - window_half + 1, step)

        # --- Per-station storage ---
        st_dip = np.full(n_st, np.nan)
        st_v_rule = np.full(n_st, np.nan)
        st_terrain = np.full(n_st, np.nan)
        st_cross = np.full(n_st, np.nan)
        st_conf = np.zeros(n_st)
        st_wing = np.array([""] * n_st, dtype=object)

        for wc in window_centres:
            mask = ((cum_len >= wc - window_half) &
                     (cum_len <= wc + window_half))
            n_in = int(mask.sum())
            if n_in < 5:
                continue

            wx = xs[mask]
            wy = ys[mask]
            wz = zs[mask]
            w_res = [results_valid[i] for i in range(n_st) if mask[i]]

            # --- PCA: along-trace (principal) and perp (secondary) ---
            cx, cy = float(np.mean(wx)), float(np.mean(wy))
            ddx = wx - cx
            ddy = wy - cy
            cov = np.cov(ddx, ddy)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, -1]
            secondary = eigvecs[:, 0]

            # Project onto (s, n)
            s = ddx * principal[0] + ddy * principal[1]
            n = ddx * secondary[0] + ddy * secondary[1]

            # Sort by s
            sort_idx = np.argsort(s)
            s = s[sort_idx]
            n = n[sort_idx]
            wz_s = wz[sort_idx]

            s_range = float(np.max(s) - np.min(s))
            n_range = float(np.max(np.abs(n)))

            # --- V-shape detection ---
            # V-shape: n has a clear interior extremum (apex), with the
            # perpendicular displacement reversing direction.
            has_v = False
            apex_idx = None

            if s_range > 0 and n_range > 0.05 * s_range:
                max_n_idx = int(np.argmax(np.abs(n)))
                if 0 < max_n_idx < len(n) - 1:
                    before = n[:max_n_idx + 1]
                    after = n[max_n_idx:]
                    if len(before) > 1 and len(after) > 1:
                        b_diff = before[-1] - before[0]
                        a_diff = after[-1] - after[0]
                        if b_diff * a_diff < 0:
                            has_v = True
                            apex_idx = max_n_idx

            dip_val = np.nan
            a_fit, b_fit, r2 = np.nan, np.nan, 0.0
            wing_label = ""

            if (has_v and apex_idx is not None and
                    apex_idx > 2 and apex_idx < len(s) - 2):
                # --- Two-wing folded-fault model ---
                # Wing 1: stations before apex
                # Wing 2: stations after apex
                # Each wing is a plane  z = a*s + b*n + c
                wing_dips = []
                wing_labels = ["wing1", "wing2"]

                for wi, (w0, w1) in enumerate(
                        [(0, apex_idx + 1), (apex_idx, len(s))]):
                    ws_w = s[w0:w1]
                    wn_w = n[w0:w1]
                    wz_w = wz_s[w0:w1]
                    if len(ws_w) < 3:
                        continue

                    A = np.column_stack(
                        [ws_w, wn_w, np.ones(len(ws_w))])
                    try:
                        sol = np.linalg.lstsq(A, wz_w, rcond=None)[0]
                        a_w, b_w, _ = sol
                        dip_w = float(np.degrees(
                            np.arctan(np.sqrt(a_w ** 2 + b_w ** 2))))
                        wing_dips.append(dip_w)

                        z_pred = A @ sol
                        ss_r = float(np.sum((wz_w - z_pred) ** 2))
                        ss_t = float(
                            np.sum((wz_w - np.mean(wz_w)) ** 2))
                        r2_w = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0

                        if wi == 0:
                            a_fit, b_fit, r2 = a_w, b_w, r2_w
                            wing_label = wing_labels[wi]
                    except np.linalg.LinAlgError:
                        continue

                if wing_dips:
                    dip_val = float(np.mean(wing_dips))
                else:
                    continue

            else:
                # --- Single-plane fit (no V-shape detected) ---
                A = np.column_stack([s, n, np.ones(len(s))])
                try:
                    sol = np.linalg.lstsq(A, wz_s, rcond=None)[0]
                    a_fit, b_fit, _ = sol
                    dip_val = float(np.degrees(
                        np.arctan(np.sqrt(a_fit ** 2 + b_fit ** 2))))

                    z_pred = A @ sol
                    ss_r = float(np.sum((wz_s - z_pred) ** 2))
                    ss_t = float(np.sum((wz_s - np.mean(wz_s)) ** 2))
                    r2 = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
                    wing_label = "single"
                except np.linalg.LinAlgError:
                    continue

            # --- V-rule constrained dip from outcrop geometry ---
            # sin(alpha) = H / W, where H = vertical relief (throw),
            # W = scarp width.  This is an independent cross-check.
            pt_vals = [r.get("primary_throw", np.nan) for r in w_res]
            sw_vals = [r.get("scarp_width_m", np.nan) for r in w_res]
            pt_med = (float(np.nanmedian(pt_vals))
                      if len(pt_vals) > 0 else np.nan)
            sw_med = (float(np.nanmedian(sw_vals))
                      if len(sw_vals) > 0 else np.nan)

            v_rule_dip = np.nan
            if (np.isfinite(pt_med) and np.isfinite(sw_med)
                    and sw_med > 0.1):
                ratio = pt_med / sw_med
                if 0 < ratio < 1:
                    v_rule_dip = float(np.degrees(np.arcsin(ratio)))

            # --- Assign to stations in the window ---
            # Prefer the highest-confidence estimate per station.
            window_indices = np.where(mask)[0]
            for idx in window_indices:
                if np.isnan(st_dip[idx]) or r2 > st_conf[idx]:
                    st_dip[idx] = dip_val
                    st_v_rule[idx] = v_rule_dip
                    st_terrain[idx] = a_fit
                    st_cross[idx] = b_fit
                    st_conf[idx] = r2
                    st_wing[idx] = wing_label

        # --- Build per-station result list ---
        # Dip category classification based on the V-rule model:
        #   0 = <45 deg (low dip,不符合目标区间)
        #   1 = 45-55 deg (目标倾角区间)
        #   2 = >55 deg (陡倾角,超出目标区间)
        # Uses the plane-fit dip (folded_dip_deg) as the primary estimate,
        # cross-checked with the V-rule dip (v_rule_dip_deg) when available.
        def _classify_dip(plane_dip, v_rule_dip):
            """Classify dip into 3 categories per the V-rule model."""
            candidates = []
            if np.isfinite(plane_dip):
                candidates.append(plane_dip)
            if np.isfinite(v_rule_dip):
                candidates.append(v_rule_dip)
            if not candidates:
                return -1  # unknown
            # Use the mean of available estimates
            best = float(np.mean(candidates))
            if best < 45.0:
                return 0
            elif best <= 55.0:
                return 1
            else:
                return 2

        station_list = []
        for i in range(n_st):
            dip_deg = float(st_dip[i]) if np.isfinite(st_dip[i]) else np.nan
            v_deg = float(st_v_rule[i]) if np.isfinite(st_v_rule[i]) else np.nan
            station_list.append({
                "y": float(ys[i]),
                "x": float(xs[i]),
                "folded_dip_deg": dip_deg,
                "v_rule_dip_deg": v_deg,
                "dip_category": _classify_dip(dip_deg, v_deg),
                "terrain_slope": float(st_terrain[i])
                if np.isfinite(st_terrain[i]) else np.nan,
                "cross_slope_b": float(st_cross[i])
                if np.isfinite(st_cross[i]) else np.nan,
                "confidence": float(st_conf[i]),
                "wing_label": str(st_wing[i]),
            })

        result[trace_id] = station_list

    return result


def plot_per_trace_parameter_profiles(file_name, segments, dem_data,
                                       transform, dem_shape, output_dir,
                                       dem10m_data=None,
                                       folded_dip_data=None):
    """Generate a parameter-vs-northing figure per original trace.

    Panels (top to bottom):
      1. Fault Displacement box plot (30 deg - 60 deg range from scarp throw,
         with 45 deg smoothed curve)
      2. Scarp Slope (steepest local slope within scarp width)
      3. Azimuth rose diagram

    X-axis uses absolute northing (y) of each station, not
    distance_along.  Segments are sorted north-to-south.  Each
    segment is a separate coloured line (gold-to-orange gradient)
    with its trace number annotated in brown.  Horizontal reference
    lines mark the average and maximum per panel.

    Top-5 locations ranked by deepest scarp throw + largest scarp slope
    are highlighted with grey background bands, with northing values
    listed in an off-chart legend.
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba

    active_segs = [s for s in segments
                   if s.get("results") and len(s["results"]) > 0]
    if not active_segs:
        return

    trace_stub = Path(file_name).stem
    asp_dir = output_dir / "per_trace_parameter_profiles"
    safe_makedirs(asp_dir)

    sorted_segs = _sort_segments_north_to_south(active_segs)

    n_panels = 3
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=[3, 3, 3.5],
                          hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    ax_rose = fig.add_subplot(gs[2], projection='polar')

    # Gold-to-orange colours for segments (consistent with 3D overview)
    seg_colors = _gold_orange_colors(len(sorted_segs))

    # Helper: circular mean
    def _circ_mean(angles_deg):
        if len(angles_deg) == 0:
            return np.nan
        angles_rad = np.deg2rad(angles_deg)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        mean_angle_rad = np.arctan2(mean_sin, mean_cos)
        mean_angle_deg = np.rad2deg(mean_angle_rad)
        return mean_angle_deg % 360

    # Helper: stepped average
    def _stepped_avg(y_vals, v_vals, max_points=10, is_circular=False):
        if len(y_vals) <= max_points:
            return y_vals, v_vals
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        bin_edges = np.linspace(y_min, y_max, max_points + 1)
        bin_centers = []
        bin_means = []
        for i in range(max_points):
            mask = (y_vals >= bin_edges[i]) & (y_vals < bin_edges[i+1])
            if i == max_points - 1:
                mask = (y_vals >= bin_edges[i]) & (y_vals <= bin_edges[i+1])
            if np.any(mask):
                bin_centers.append(np.mean(y_vals[mask]))
                if is_circular:
                    bin_means.append(_circ_mean(v_vals[mask]))
                else:
                    bin_means.append(np.mean(v_vals[mask]))
        return np.array(bin_centers), np.array(bin_means)

    # Collect all station data for Top-5 ranking
    all_station_data = []  # list of (y, primary_throw, max_local_slope_deg, trace_id)

    # ==================================================================
    # Panel 1: Fault Displacement box plot (30 deg - 60 deg)
    # ==================================================================
    ax_disp = axes[0]
    all_disp_vals = []
    all_d45_vals = []

    for si, seg in enumerate(sorted_segs):
        results = seg["results"]
        trace_id = seg["trace_id"]
        y_vals = np.array([r.get("y", np.nan) for r in results], dtype=float)
        x_vals = np.array([r.get("x", np.nan) for r in results], dtype=float)
        sort_idx = np.lexsort((x_vals, y_vals))
        y_vals = y_vals[sort_idx]
        x_vals = x_vals[sort_idx]
        results = [results[i] for i in sort_idx]

        color = seg_colors[si]
        trace_length = len(results)
        if trace_length > 100:
            max_points = 15
        elif trace_length > 50:
            max_points = 10
        else:
            max_points = 5

        # Compute displacement at 30, 45, 60 degrees from primary_throw
        pt_vals = np.array([r.get("primary_throw", np.nan) for r in results],
                           dtype=float)
        valid = np.isfinite(pt_vals) & (pt_vals > 0) & np.isfinite(y_vals)
        if not np.any(valid):
            continue

        y_v = y_vals[valid]
        pt_v = pt_vals[valid]
        d_30 = pt_v / np.sin(np.radians(30))   # low bar
        d_60 = pt_v / np.sin(np.radians(60))   # high bar
        d_45 = pt_v / np.sin(np.radians(45))   # mid

        # Stepped average for this segment only
        y_s, d30_s = _stepped_avg(y_v, d_30, max_points)
        _, d60_s = _stepped_avg(y_v, d_60, max_points)
        _, d45_s = _stepped_avg(y_v, d_45, max_points)

        # Draw vertical bars (30 deg low to 60 deg high) per segment
        for i in range(len(y_s)):
            ax_disp.plot([y_s[i], y_s[i]], [d30_s[i], d60_s[i]],
                         color=color, linewidth=0.8, alpha=0.4)

        # Draw per-segment 45-degree line (connected within segment only)
        ax_disp.plot(y_s, d45_s, '-', color=color, linewidth=2.0, alpha=0.9,
                     zorder=10)
        ax_disp.plot(y_s, d45_s, 'o', color=color, markersize=4, alpha=0.8,
                     markeredgecolor='white', markeredgewidth=0.5, zorder=11)

        all_disp_vals.extend(d30_s.tolist())
        all_disp_vals.extend(d60_s.tolist())
        all_d45_vals.extend(d45_s.tolist())

        # Collect station data for Top-5
        for r in results:
            y_val = r.get("y", np.nan)
            pt = r.get("primary_throw", np.nan)
            mls = r.get("max_local_slope_deg", np.nan)
            if np.isfinite(y_val) and np.isfinite(pt):
                all_station_data.append((y_val, pt, mls, trace_id))

        # Annotate trace ID
        mid_idx = len(y_s) // 2
        if mid_idx < len(y_s):
            y_pos = float(np.median(d45_s))
            ax_disp.annotate(
                f"{trace_id}", xy=(y_s[mid_idx], y_pos),
                fontsize=8, fontweight="bold", color=_BROWN,
                xytext=(4, 4), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=_BROWN, alpha=0.9, linewidth=0.5),
                zorder=15,
            )

    # Reference lines: use only 45-deg displacement for average/max
    _plot_ref_lines(ax_disp, all_d45_vals, label_prefix="Disp 45 ")

    # Compute average displacement range for legend
    valid_d45 = np.array([v for v in all_d45_vals if np.isfinite(v)])
    if len(valid_d45) > 0:
        avg_disp = float(np.mean(valid_d45))
        max_disp = float(np.max(valid_d45))
        min_disp = float(np.min(valid_d45))
        max_diff = max_disp - avg_disp
        min_diff = avg_disp - min_disp
    else:
        avg_disp = np.nan
        max_disp = np.nan
        min_disp = np.nan
        max_diff = 0.0
        min_diff = 0.0

    ax_disp.set_ylabel("Displacement (m)", fontsize=14)
    ax_disp.set_title("Along-strike fault displacement",
                       fontsize=14, fontweight="bold")
    ax_disp.grid(True, alpha=0.3, linestyle="--")
    ax_disp.tick_params(labelsize=12)

    # Custom legend for displacement panel with superscript/subscript
    # Build legend label: range^(+max diff)_(-min diff)
    range_str = f"{min_disp:.1f}-{max_disp:.1f}"
    legend_label = (f"Disp avg={avg_disp:.1f} "
                    f"[{range_str}]$^{{(+{max_diff:.1f})}}_{{(-{min_diff:.1f})}}$")

    disp_legend = [
        Line2D([0], [0], color='gray', linewidth=2.0, marker='o', markersize=4,
               label="Displacement at 45 deg"),
        Line2D([0], [0], color='gray', linewidth=0.8, alpha=0.4,
               label="Range 30 deg - 60 deg"),
        Line2D([0], [0], color='brown', linewidth=0, marker='',
               label=legend_label),
    ]
    handles, labels = ax_disp.get_legend_handles_labels()
    ax_disp.legend(handles=handles + disp_legend,
                   labels=labels + ["Displacement at 45 deg",
                                   "Range 30 deg - 60 deg",
                                   legend_label],
                   fontsize=18, loc='upper left',bbox_to_anchor=(1.02, 1.0), framealpha=0.9, ncol=1)

    # ==================================================================
    # Panel 2: Scarp Slope (max local slope within scarp width)
    # ==================================================================
    ax_slope = axes[1]
    all_slope_vals = []

    for si, seg in enumerate(sorted_segs):
        results = seg["results"]
        trace_id = seg["trace_id"]
        y_vals = np.array([r.get("y", np.nan) for r in results], dtype=float)
        x_vals = np.array([r.get("x", np.nan) for r in results], dtype=float)
        sort_idx = np.lexsort((x_vals, y_vals))
        y_vals = y_vals[sort_idx]
        x_vals = x_vals[sort_idx]
        results = [results[i] for i in sort_idx]

        color = seg_colors[si]
        trace_length = len(results)
        if trace_length > 100:
            max_points = 15
        elif trace_length > 50:
            max_points = 10
        else:
            max_points = 5

        slope_vals = np.array([r.get("max_local_slope_deg", np.nan) for r in results],
                               dtype=float)
        valid = np.isfinite(slope_vals) & np.isfinite(y_vals)
        if np.any(valid):
            y_s, v_s = _stepped_avg(y_vals[valid], slope_vals[valid], max_points)
            ax_slope.plot(y_s, v_s, '-', color=color, linewidth=1.5, alpha=0.8)
            ax_slope.plot(y_s, v_s, 'o', color=color, markersize=4, alpha=0.8,
                          markeredgecolor='white', markeredgewidth=0.5)
            all_slope_vals.extend(v_s.tolist())

        # Annotate trace ID
        mid_idx = len(y_vals) // 2
        if mid_idx < len(y_vals) and np.isfinite(y_vals[mid_idx]):
            sl_all = [r.get("max_local_slope_deg", np.nan) for r in results]
            sl_valid = [v for v in sl_all if np.isfinite(v)]
            if sl_valid:
                y_pos = float(np.median(sl_valid))
                ax_slope.annotate(
                    f"{trace_id}", xy=(y_vals[mid_idx], y_pos),
                    fontsize=8, fontweight="bold", color=_BROWN,
                    xytext=(4, 4), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=_BROWN, alpha=0.9, linewidth=0.5),
                    zorder=15,
                )

    _plot_ref_lines(ax_slope, all_slope_vals, label_prefix="Slope ")
    ax_slope.set_ylabel("Slope (deg)", fontsize=14)
    ax_slope.set_title("Maximum slope within scarp width",
                        fontsize=14, fontweight="bold")
    ax_slope.grid(True, alpha=0.3, linestyle="--")
    ax_slope.tick_params(labelsize=12)
    handles, labels = ax_slope.get_legend_handles_labels()
    if handles:
        ax_slope.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

    # ==================================================================
    # Top-5 highlight: rank by deepest throw + largest slope
    # ==================================================================
    if len(all_station_data) >= 5:
        # Normalise throw and slope, then combine
        throws = np.array([d[1] for d in all_station_data])
        slopes = np.array([d[2] if np.isfinite(d[2]) else 0 for d in all_station_data])
        throw_norm = (throws - throws.min()) / (throws.max() - throws.min() + 1e-10)
        slope_norm = (slopes - slopes.min()) / (slopes.max() - slopes.min() + 1e-10)
        combined_score = throw_norm + slope_norm
        top5_idx = np.argsort(combined_score)[-5:]

        # Add grey bands to displacement and slope panels
        for idx in top5_idx:
            y_val = all_station_data[idx][0]
            # Estimate band width from neighbouring stations
            all_y = sorted([d[0] for d in all_station_data])
            y_pos = all_y.index(y_val) if y_val in all_y else 0
            if y_pos > 0 and y_pos < len(all_y) - 1:
                band_half = (all_y[min(y_pos + 1, len(all_y)-1)] -
                             all_y[max(y_pos - 1, 0)]) / 2
            else:
                band_half = 50.0
            band_half = max(band_half, 20.0)

            for ax in axes[:2]:
                ax.axvspan(y_val - band_half, y_val + band_half,
                           color='gray', alpha=0.15, zorder=0)

        # Off-chart legend with northing values
        top5_y = [all_station_data[idx][0] for idx in top5_idx]
        top5_y.sort()
        legend_lines = [
            Line2D([0], [0], color='gray', alpha=0.3, linewidth=8,
                   label=f"N={y_val:.0f} m")
            for y_val in top5_y
        ]
        # Place legend outside the figure (bottom-right)
        fig.legend(handles=legend_lines, loc='lower right',
                   fontsize=9, framealpha=0.9, ncol=1,
                   bbox_to_anchor=(1.0, 0.02),
                   title="Top-5 locations (throw + slope)")

    # ==================================================================
    # Panel 3: Azimuth rose diagram (length-weighted, trace-coloured)
    # ==================================================================
    bin_deg = 10
    bins = np.arange(0, 370, bin_deg)
    n_bins = len(bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    theta = np.deg2rad(bin_centers)
    width = np.deg2rad(bin_deg)

    # Compute trace length per segment (in km)
    trace_lengths_km = []
    for seg in sorted_segs:
        tr = seg.get("trace", {})
        length_m = tr.get("length_m", 0)
        trace_lengths_km.append(length_m / 1000.0)

    # Build per-trace, per-bin weighted counts
    # weighted_counts[si] = array of length(n_bins) with length-weighted counts
    weighted_counts = np.zeros((len(sorted_segs), n_bins))

    for si, seg in enumerate(sorted_segs):
        results = seg["results"]
        if not results:
            continue
        azs = np.array([r.get("azimuth", np.nan) for r in results],
                       dtype=float)
        valid_az = azs[np.isfinite(azs) & (azs != 0)]
        if len(valid_az) == 0:
            continue
        # Weight each azimuth by trace length / n_stations so the total
        # weight for this trace equals its length in km
        n_stations = len(valid_az)
        per_station_weight = trace_lengths_km[si] / n_stations
        counts, _ = np.histogram(valid_az, bins=bins)
        weighted_counts[si] = counts * per_station_weight

    # Cumulative (stacked) counts per bin
    cumulative = np.sum(weighted_counts, axis=0)
    max_cum = np.max(cumulative) if np.max(cumulative) > 0 else 1.0

    # Draw stacked bars: each trace gets its own colour
    bottom = np.zeros(n_bins)
    for si in range(len(sorted_segs)):
        wc = weighted_counts[si]
        if np.max(wc) > 0:
            color = seg_colors[si]
            ax_rose.bar(theta, wc, width=width, bottom=bottom,
                        color=[color] * n_bins, alpha=0.8,
                        edgecolor='white', linewidth=0.3,
                        label=f"Trace {sorted_segs[si]['trace_id']}")
            bottom += wc

    # Mean azimuth line (length-weighted)
    all_az_weighted = []
    for si, seg in enumerate(sorted_segs):
        results = seg["results"]
        for r in results:
            az = r.get("azimuth", np.nan)
            if np.isfinite(az) and az != 0:
                all_az_weighted.append((az, trace_lengths_km[si] /
                                       max(1, len([r2 for r2 in results
                                                   if np.isfinite(r2.get("azimuth", np.nan))
                                                   and r2.get("azimuth") != 0]))))

    if all_az_weighted:
        az_arr = np.array([a for a, _ in all_az_weighted])
        w_arr = np.array([w for _, w in all_az_weighted])
        sin_mean = np.sum(w_arr * np.sin(np.radians(az_arr))) / np.sum(w_arr)
        cos_mean = np.sum(w_arr * np.cos(np.radians(az_arr))) / np.sum(w_arr)
        mean_az = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
        ax_rose.plot(np.array([0, np.deg2rad(mean_az)]),
                      np.array([0, max_cum]),
                      color=_BROWN, linewidth=2.0, linestyle='--',
                      label=f"Mean={mean_az:.0f} deg")

    ax_rose.set_title("Azimuth Rose Diagram (length-weighted, km)",
                       fontsize=14, fontweight="bold", pad=15)
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    ax_rose.tick_params(labelsize=10)

    # Move legend to bottom, outside the polar panel
    handles_rose, labels_rose = ax_rose.get_legend_handles_labels()
    if handles_rose:
        ax_rose.legend(handles_rose, labels_rose,
                       loc='upper center', bbox_to_anchor=(0.5, -0.12),
                       fontsize=10, framealpha=0.9, ncol=min(4, len(handles_rose)),
                       title="Traces")

    # ==================================================================
    # Final formatting
    # ==================================================================
    axes[1].set_xlabel("Northing (m)", fontsize=14)

    fig.suptitle(f"{trace_stub} - Per-trace parameter profiles (N to S)",
                 fontsize=16, fontweight="bold", y=0.995)

    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    out_path = asp_dir / f"{trace_stub}_parameter_profiles.png"
    _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"  [Per-trace params] Saved to {out_path}")

# ===================================================================
# 10c.  PER-ORIGINAL-TRACE ELEVATION vs EASTING (per-segment dip)
# ===================================================================


def plot_elevation_easting_dip(file_name, segments, dem_data, transform,
                                dem_shape, output_dir, dem10m_data=None,
                                dem_hr_path=None):
    """Generate elevation-vs-along-trace-distance plots with local dip estimation.

    Parameters
    ----------
    dem_hr_path : Path or None
        Path to a high-resolution DEM GeoTIFF (e.g. 2 m LiDAR).
        When provided, local DEM windows for Rule-of-V analysis are
        sampled from this DEM at its native resolution.

    Workflow:
    1. Sample DEM elevations along fault trace
    2. Calculate along-trace distance
    3. Identify individual V (valley) and ^ (ridge) features
    4. For each feature, analyze local DEM window
    5. Estimate local dip using Rule of V's geometry
    6. Visualize all components

    The key principle: each V/^ is analyzed locally, not globally.
    """
    from scipy.signal import find_peaks, savgol_filter
    from scipy.ndimage import gaussian_filter
    
    trace_stub = Path(file_name).stem
    eed_dir = output_dir / "per_trace_elevation_profiles"
    safe_makedirs(eed_dir)

    def _calculate_along_trace_distance(x_coords, y_coords):
        """Calculate cumulative distance along fault trace."""
        if len(x_coords) < 2:
            return np.zeros(len(x_coords))
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        segment_distances = np.sqrt(dx**2 + dy**2)
        
        cumulative_dist = np.zeros(len(x_coords))
        cumulative_dist[1:] = np.cumsum(segment_distances)
        
        return cumulative_dist

    def _identify_vw_features(s_dist, z_vals, min_prominence=5.0, 
                              min_distance=100.0):
        """Identify V (valley) and ∧ (ridge) features using scipy find_peaks.
        
        The min_prominence is adapted to the actual elevation range of
        the profile: if the total relief is small, the threshold is
        scaled down so that features can still be detected in low-relief
        fault scarps.
        """
        empty_arr = np.array([], dtype=int)
        if len(s_dist) < 5:
            return empty_arr, empty_arr, None, None
        
        # Smooth the profile to reduce noise
        window_length = min(9, len(z_vals)//2*2+1)
        if window_length >= 3 and window_length <= len(z_vals):
            try:
                z_smooth = savgol_filter(z_vals, window_length, 2)
            except:
                z_smooth = gaussian_filter(z_vals, sigma=1.0)
        else:
            z_smooth = gaussian_filter(z_vals, sigma=1.0)
        
        # Adapt prominence threshold to the actual elevation range
        z_range = np.nanmax(z_smooth) - np.nanmin(z_smooth)
        if z_range < 1.0:
            return empty_arr, empty_arr, None, None
        
        # Scale prominence: use 10% of z_range, but at least 0.3m
        # and at most the original min_prominence
        adaptive_prominence = max(0.3, min(min_prominence, z_range * 0.10))
        
        # Convert distance to samples for find_peaks
        if len(s_dist) > 1:
            avg_spacing = np.mean(np.diff(s_dist))
            min_distance_samples = max(1, int(min_distance / avg_spacing))
        else:
            min_distance_samples = 1
        
        # Find ridges (peaks) and valleys (troughs)
        ridge_indices, ridge_props = find_peaks(
            z_smooth, 
            prominence=adaptive_prominence,
            distance=min_distance_samples
        )
        
        valley_indices, valley_props = find_peaks(
            -z_smooth,
            prominence=adaptive_prominence,
            distance=min_distance_samples
        )
        
        return valley_indices, ridge_indices, valley_props, ridge_props

    def _estimate_local_azimuth(x_vals, y_vals, center_idx, window_size=10,
                                global_azimuth=None):
        """Estimate local fault azimuth using points around the feature.
        
        Uses PCA on the local point cloud to find the principal axis,
        which works for all trace orientations (unlike linear regression
        which fails for near-N/S traces). Falls back to global azimuth
        if provided, or 0.0 as last resort.
        """
        start_idx = max(0, center_idx - window_size)
        end_idx = min(len(x_vals), center_idx + window_size + 1)
        
        if end_idx - start_idx < 3:
            if global_azimuth is not None and np.isfinite(global_azimuth):
                return float(global_azimuth)
            return _estimate_global_azimuth(x_vals, y_vals)
        
        local_x = x_vals[start_idx:end_idx]
        local_y = y_vals[start_idx:end_idx]
        
        # Use PCA to estimate local direction (robust for all orientations)
        try:
            coords = np.column_stack([local_x, local_y])
            coords_centered = coords - np.mean(coords, axis=0)
            
            # PCA via covariance matrix
            cov_matrix = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Principal axis = eigenvector with largest eigenvalue
            principal_axis = eigenvectors[:, -1]
            
            # Azimuth measured clockwise from north
            # If dx = principal_axis[0], dy = principal_axis[1]
            # azimuth = atan2(dx, dy)
            azimuth = np.degrees(np.arctan2(principal_axis[0], principal_axis[1]))
            azimuth = azimuth % 360.0
            
            # Validate: if the local spread is too small, result is unreliable
            if eigenvalues[-1] < 1e-6:
                if global_azimuth is not None and np.isfinite(global_azimuth):
                    return float(global_azimuth)
                return _estimate_global_azimuth(x_vals, y_vals)
            
            return float(azimuth)
        except Exception:
            if global_azimuth is not None and np.isfinite(global_azimuth):
                return float(global_azimuth)
            return _estimate_global_azimuth(x_vals, y_vals)

    def _estimate_global_azimuth(x_vals, y_vals):
        """Estimate global fault azimuth using PCA."""
        if len(x_vals) < 2:
            return 0.0
        
        coords = np.column_stack([x_vals, y_vals])
        coords_centered = coords - np.mean(coords, axis=0)
        
        try:
            cov_matrix = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            principal_axis = eigenvectors[:, -1]
            
            azimuth = np.degrees(np.arctan2(principal_axis[0], principal_axis[1]))
            azimuth = azimuth % 360.0
            return azimuth
        except:
            return 0.0

    def _extract_local_dem_window(x_center, y_center, azimuth, dem_data,
                              transform, dem_shape, window_size=200.0,
                              dem10m_data=None,
                              dem_hr_path=None):
        """Extract a local DEM window around a V/W feature.

        When *dem_hr_path* is provided and the file exists, the window
        is read directly from the high-resolution DEM (e.g. 2 m LiDAR)
        via rasterio windowed reading.  This avoids loading the entire
        ~2.3 GB grid into memory and preserves the full spatial
        resolution needed to resolve small Rule-of-V displacements.

        The window is expressed in a local coordinate system:
            s = along-trace direction
            n = perpendicular-to-trace direction

        Returns
        -------
        dict or None
            Local DEM coordinates and elevations.
        """

        # ---------------------------------------------------------------
        # Choose which DEM to use for local sampling.
        # ---------------------------------------------------------------
        use_hr_dem = False
        if dem_hr_path is not None:
            try:
                from pathlib import Path
                p = Path(dem_hr_path)
                if p.exists():
                    use_hr_dem = True
            except Exception:
                pass

        # ---------------------------------------------------------------
        # Local coordinate system
        # ---------------------------------------------------------------
        azimuth_rad = np.deg2rad(azimuth)
        perp_azimuth = (azimuth + 90.0) % 360.0
        perp_azimuth_rad = np.deg2rad(perp_azimuth)

        along_vector = np.array([
            np.sin(azimuth_rad),
            np.cos(azimuth_rad)
        ], dtype=float)

        perp_vector = np.array([
            np.sin(perp_azimuth_rad),
            np.cos(perp_azimuth_rad)
        ], dtype=float)

        half_window = float(window_size) / 2.0

        dem_resolution = 5.0  # default; overwritten below if known

        # ---------------------------------------------------------------
        # Build the sample grid in local (s, n) coordinates, then
        # convert to map (x, y) coordinates for DEM sampling.
        # ---------------------------------------------------------------
        if use_hr_dem:
            # Determine the native pixel size of the HR DEM by opening
            # it once (metadata-only read, no pixel data loaded).
            try:
                with rasterio.open(str(dem_hr_path)) as src_hr:
                    hr_transform = src_hr.transform
                    hr_shape = (src_hr.height, src_hr.width)
                    res_x = abs(float(hr_transform.a))
                    res_y = abs(float(hr_transform.e))
                    dem_resolution = min(res_x, res_y)
            except Exception:
                use_hr_dem = False
                dem_resolution = 5.0

        if not use_hr_dem:
            # Fall back to the main DEM (10 m).
            try:
                res_x = abs(float(transform.a))
                res_y = abs(float(transform.e))
                dem_resolution = min(res_x, res_y)
            except Exception:
                dem_resolution = 5.0

        if not np.isfinite(dem_resolution) or dem_resolution <= 0:
            dem_resolution = 5.0

        # Sample at native DEM resolution.
        sample_spacing = dem_resolution

        n_samples = int(np.ceil(window_size / sample_spacing)) + 1
        # Safety cap
        n_samples = min(n_samples, 501)

        s_offsets = np.linspace(-half_window, half_window, n_samples)
        n_offsets = np.linspace(-half_window, half_window, n_samples)

        ss, nn = np.meshgrid(s_offsets, n_offsets, indexing='ij')

        xx = x_center + ss * along_vector[0] + nn * perp_vector[0]
        yy = y_center + ss * along_vector[1] + nn * perp_vector[1]

        # ---------------------------------------------------------------
        # Sample elevations from the chosen DEM.
        # ---------------------------------------------------------------
        if use_hr_dem:
            # Windowed read from the HR DEM file: read only the bounding
            # box that covers all sample points, then bilinearly
            # interpolate within that sub-array.
            x_min, x_max = float(xx.min()), float(xx.max())
            y_min, y_max = float(yy.min()), float(yy.max())

            try:
                with rasterio.open(str(dem_hr_path)) as src_hr:
                    # Add a small buffer (1 pixel) to avoid edge effects.
                    buf = dem_resolution * 2
                    win = from_bounds(
                        x_min - buf, y_min - buf,
                        x_max + buf, y_max + buf,
                        src_hr.transform
                    )
                    # Clamp window to raster bounds
                    x_off = max(0, int(win.col_off))
                    y_off = max(0, int(win.row_off))
                    w_end = min(src_hr.width, int(win.col_off + win.width))
                    h_end = min(src_hr.height, int(win.row_off + win.height))
                    win = Window(
                        x_off, y_off,
                        max(1, w_end - x_off),
                        max(1, h_end - y_off)
                    )
                    hr_data = src_hr.read(
                        1, window=win,
                        boundless=False, fill_value=np.nan
                    ).astype(np.float64)
                    hr_win_transform = window_transform(
                        win, src_hr.transform
                    )

                    # Use the patched sampler with the HR sub-array
                    elevations = _sample_dem_elevations_patched(
                        xx.ravel(), yy.ravel(),
                        hr_data, hr_win_transform, hr_data.shape,
                        dem10m_data=None  # HR DEM should not need patching
                    )
                    elevations = elevations.reshape(xx.shape)
            except Exception:
                # If HR read fails, fall back to the main DEM
                elevations = _sample_dem_elevations_patched(
                    xx.ravel(), yy.ravel(),
                    dem_data, transform, dem_shape,
                    dem10m_data=dem10m_data
                )
                elevations = elevations.reshape(xx.shape)
        else:
            elevations = _sample_dem_elevations_patched(
                xx.ravel(), yy.ravel(),
                dem_data, transform, dem_shape,
                dem10m_data=dem10m_data
            )
            elevations = elevations.reshape(xx.shape)

        valid = np.isfinite(elevations)
        if np.sum(valid) < 10:
            return None

        return {
            'x': xx[valid],
            'y': yy[valid],
            's_offset': ss[valid],
            'n_offset': nn[valid],
            'elevation': elevations[valid],
            'x_grid': xx,
            'y_grid': yy,
            's_grid': ss,
            'n_grid': nn,
            'elevation_grid': elevations,
            'along_vector': along_vector,
            'perp_vector': perp_vector,
            'dem_resolution': float(dem_resolution),
            'sample_spacing': float(
                min(
                    abs(s_offsets[1] - s_offsets[0]) if len(s_offsets) > 1 else sample_spacing,
                    abs(n_offsets[1] - n_offsets[0]) if len(n_offsets) > 1 else sample_spacing
                )
            ),
            'window_size': float(window_size),
            'perp_azimuth': float(perp_azimuth)
        }

    def _estimate_local_dip_from_v_geometry(local_dem, feature_z,
                                            along_vector, perp_vector):
        """
        Estimate local fault dip using two complementary methods:

        1. **Contour V-deflection** (primary): When a dipping fault
           scarp crosses topographic relief, contour lines on the
           scarp face are deflected.  Fitting delta_z (plateau
           elevation deviation) vs delta_n (contour position
           deviation within each level) gives tan(dip).

        2. **Scarp-face slope** (fallback): When the V-deflection
           signal is too weak (low-relief terrain), the slope of
           the scarp face itself provides a lower bound on the
           fault dip.  This is because the scarp face is the
           exhumed fault plane, degraded by erosion so its slope
           is gentler than the original dip.

        Quality checks
        --------------
        - Perpendicular relief (scarp height) must be > 0.5 m.
        - At least 3 along-trace profiles with valid data.
        - V-deflection: r² >= 0.15, dip 0.5–85°
        - Scarp-face slope: r² >= 0.3, dip 0.5–85°

        Returns
        -------
        dict or None
        """
        if local_dem is None:
            return None

        z_grid = local_dem.get('elevation_grid')
        if z_grid is None:
            return None
        s_grid = local_dem.get('s_grid')
        n_grid = local_dem.get('n_grid')
        if s_grid is None or n_grid is None:
            return None

        z = np.asarray(z_grid, dtype=float)
        if z.ndim != 2:
            return None
        finite = np.isfinite(z)
        if np.sum(finite) < 25:
            return None
        z_valid = z[finite]
        z_range = float(np.max(z_valid) - np.min(z_valid))
        if z_range < 0.3:
            return None

        sample_spacing = float(local_dem.get(
            'sample_spacing', local_dem.get('dem_resolution', 5.0)))

        perp_azimuth = (
            np.degrees(np.arctan2(perp_vector[0], perp_vector[1])) % 360.0
        )

        n_vals_1d = n_grid[0, :]
        s_vals_1d = s_grid[:, 0]
        s_limit = 0.40 * float(np.nanmax(np.abs(s_vals_1d)))

        # -----------------------------------------------------------
        # Phase 1: Per-profile scarp identification
        # -----------------------------------------------------------
        # For each along-trace position, find the scarp face,
        # collect plateau elevations, scarp center position,
        # and per-profile contour crossings.
        # -----------------------------------------------------------

        profile_data = []

        for si in range(len(s_vals_1d)):
            s_val = float(s_vals_1d[si])
            if abs(s_val) > s_limit:
                continue

            z_profile = z[si, :].copy()
            valid_n = np.isfinite(z_profile)
            if np.sum(valid_n) < 5:
                continue

            n_prof = n_vals_1d[valid_n]
            z_prof = z_profile[valid_n]
            sort_idx = np.argsort(n_prof)
            n_prof = n_prof[sort_idx]
            z_prof = z_prof[sort_idx]

            if len(z_prof) < 5:
                continue

            # Scarp face: zone of maximum slope
            dz_dn = np.gradient(z_prof, n_prof)
            abs_grad = np.abs(dz_dn)
            max_grad = float(np.nanmax(abs_grad)) if np.any(np.isfinite(abs_grad)) else 0
            grad_threshold = 0.25 * max_grad
            if grad_threshold < 0.005:
                continue

            above = abs_grad > grad_threshold
            if not np.any(above):
                continue

            best_start, best_len = 0, 0
            cur_start, cur_len = 0, 0
            for k in range(len(above)):
                if above[k]:
                    if cur_len == 0:
                        cur_start = k
                    cur_len += 1
                else:
                    if cur_len > best_len:
                        best_len, best_start = cur_len, cur_start
                    cur_len = 0
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            if best_len < 3:
                continue

            scarp_start = best_start
            scarp_end = best_start + best_len

            # Plateau elevations
            if scarp_start > 0:
                z_upper = float(np.nanmedian(z_prof[:scarp_start]))
            else:
                z_upper = float(z_prof[0])
            if scarp_end < len(z_prof):
                z_lower = float(np.nanmedian(z_prof[scarp_end:]))
            else:
                z_lower = float(z_prof[-1])
            if z_upper < z_lower:
                z_upper, z_lower = z_lower, z_upper

            scarp_height = z_upper - z_lower
            if scarp_height < 0.3:
                continue

            # Scarp center: midpoint of scarp zone in n
            n_scarp_zone = n_prof[scarp_start:scarp_end + 1]
            z_scarp_zone = z_prof[scarp_start:scarp_end + 1]
            n_center = float(np.mean(n_scarp_zone))

            # Scarp-face slope: linear fit z vs n within scarp zone
            if len(z_scarp_zone) >= 3:
                try:
                    from scipy.stats import linregress as _lr
                    sf_reg = _lr(n_scarp_zone, z_scarp_zone)
                    scarp_slope = float(sf_reg.slope)
                    scarp_r2 = float(sf_reg.rvalue) ** 2
                except Exception:
                    if len(n_scarp_zone) > 1:
                        sf_A = np.column_stack([n_scarp_zone, np.ones(len(n_scarp_zone))])
                        sf_res = np.linalg.lstsq(sf_A, z_scarp_zone, rcond=None)[0]
                        scarp_slope = float(sf_res[0])
                        sf_pred = sf_A @ sf_res
                        sf_ss_res = np.sum((z_scarp_zone - sf_pred) ** 2)
                        sf_ss_tot = np.sum((z_scarp_zone - np.mean(z_scarp_zone)) ** 2)
                        scarp_r2 = 1.0 - sf_ss_res / sf_ss_tot if sf_ss_tot > 0 else 0.0
                    else:
                        scarp_slope = 0.0
                        scarp_r2 = 0.0
            else:
                scarp_slope = 0.0
                scarp_r2 = 0.0

            # Per-profile contour crossings at adaptive levels
            n_contours = min(8, max(3, int(scarp_height / 0.5)))
            z_levels = np.linspace(
                z_lower + 0.15 * scarp_height,
                z_upper - 0.15 * scarp_height,
                n_contours
            )

            n_cross_by_level = {}
            for z_c in z_levels:
                diffs = z_scarp_zone - z_c
                sign_changes = np.where(np.diff(np.sign(diffs)))[0]
                if len(sign_changes) == 0:
                    continue
                k = sign_changes[0]
                if k >= len(n_scarp_zone) - 1:
                    continue
                dz = z_scarp_zone[k + 1] - z_scarp_zone[k]
                if abs(dz) < 1e-9:
                    n_cross = float(n_scarp_zone[k])
                else:
                    frac = (z_c - z_scarp_zone[k]) / dz
                    n_cross = float(n_scarp_zone[k] + frac * (n_scarp_zone[k + 1] - n_scarp_zone[k]))
                n_cross_by_level[float(z_c)] = n_cross

            profile_data.append({
                's': s_val,
                'z_upper': z_upper,
                'z_lower': z_lower,
                'h': scarp_height,
                'n_center': n_center,
                'scarp_slope': scarp_slope,
                'scarp_r2': scarp_r2,
                'n_cross': n_cross_by_level
            })

        if len(profile_data) < 3:
            return None

        # -----------------------------------------------------------
        # Phase 2: V-deflection method (primary)
        # -----------------------------------------------------------
        # For each pair of profiles that share a contour level,
        # compute delta_n = n_cross - n_bar(level) and
        # delta_z = z_upper - mean(z_upper).
        # Fit delta_z vs delta_n → tan(dip).
        # -----------------------------------------------------------

        z_uppers = np.array([p['z_upper'] for p in profile_data])
        z_upper_mean = float(np.mean(z_uppers))
        delta_z_all = z_uppers - z_upper_mean

        # Build shared contour levels across profiles
        # Use the intersection of all profiles' contour levels
        # (rounded to handle float comparison)
        def _round_key(z):
            return round(z, 2)

        # Collect all contour level keys
        all_levels = set()
        for p in profile_data:
            for z_c in p['n_cross']:
                all_levels.add(_round_key(z_c))

        # For each level, find profiles that have it
        level_to_profiles = {}
        for z_c in all_levels:
            level_to_profiles[z_c] = []
            for i, p in enumerate(profile_data):
                for z_key in p['n_cross']:
                    if _round_key(z_key) == z_c:
                        level_to_profiles[z_c].append(i)
                        break

        # Valid levels: those present in >= 3 profiles
        valid_levels = [z_c for z_c, idxs in level_to_profiles.items() if len(idxs) >= 3]

        delta_n_all = []
        delta_z_paired = []

        for z_c in valid_levels:
            idxs = level_to_profiles[z_c]
            # Get n_cross for each profile at this level
            n_vals_at_level = []
            for i in idxs:
                p = profile_data[i]
                for z_key in p['n_cross']:
                    if _round_key(z_key) == z_c:
                        n_vals_at_level.append(p['n_cross'][z_key])
                        break

            if len(n_vals_at_level) < 3:
                continue

            n_bar = float(np.mean(n_vals_at_level))

            for j, i in enumerate(idxs):
                delta_n = n_vals_at_level[j] - n_bar
                delta_n_all.append(delta_n)
                delta_z_paired.append(delta_z_all[i])

        v_deflection_result = None
        if len(delta_n_all) >= 6:
            delta_n_arr = np.array(delta_n_all, dtype=float)
            delta_z_arr = np.array(delta_z_paired, dtype=float)

            if np.std(delta_n_arr) > 1e-6 and np.std(delta_z_arr) > 1e-6:
                try:
                    from scipy.stats import linregress
                    reg = linregress(delta_n_arr, delta_z_arr)
                    v_m = float(reg.slope)
                    v_c = float(reg.intercept)
                    v_r2 = float(reg.rvalue) ** 2
                except Exception:
                    A = np.column_stack([delta_n_arr, np.ones(len(delta_n_arr))])
                    res = np.linalg.lstsq(A, delta_z_arr, rcond=None)[0]
                    v_m = float(res[0])
                    v_c = float(res[1])
                    v_pred = A @ res
                    ss_res = np.sum((delta_z_arr - v_pred) ** 2)
                    ss_tot = np.sum((delta_z_arr - np.mean(delta_z_arr)) ** 2)
                    v_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                if np.isfinite(v_m) and np.isfinite(v_r2):
                    v_dip = np.degrees(np.arctan(abs(v_m)))
                    if (
                        np.isfinite(v_dip)
                        and 0.5 <= v_dip <= 85.0
                        and v_r2 >= 0.15
                    ):
                        n_deflection_range = float(np.max(delta_n_arr) - np.min(delta_n_arr))
                        v_deflection_result = {
                            'dip_angle': float(v_dip),
                            'm_n': float(v_m),
                            'c': float(v_c),
                            'r2': float(v_r2),
                            'n_points': int(len(delta_n_arr)),
                            'n_deflection_range': n_deflection_range,
                            'delta_n': delta_n_arr.tolist(),
                            'delta_z': delta_z_arr.tolist(),
                        }

        # -----------------------------------------------------------
        # Phase 3: Scarp-face slope method (fallback)
        # -----------------------------------------------------------
        # Average scarp-face slope across all profiles.
        # This is a lower bound on fault dip.
        # -----------------------------------------------------------

        scarp_slopes = [p['scarp_slope'] for p in profile_data if p['scarp_r2'] >= 0.3]
        scarp_r2s = [p['scarp_r2'] for p in profile_data if p['scarp_r2'] >= 0.3]

        scarp_face_result = None
        if len(scarp_slopes) >= 3:
            # Weighted average (weight by r²)
            weights = np.array(scarp_r2s)
            slopes = np.array(scarp_slopes)
            # Take absolute values (slope can be + or - depending
            # on n direction)
            abs_slopes = np.abs(slopes)
            w_sum = np.sum(weights)
            if w_sum > 0:
                mean_slope = float(np.sum(abs_slopes * weights) / w_sum)
            else:
                mean_slope = float(np.mean(abs_slopes))

            sf_dip = np.degrees(np.arctan(mean_slope))
            mean_r2 = float(np.mean(scarp_r2s))

            if (
                np.isfinite(sf_dip)
                and 0.5 <= sf_dip <= 85.0
                and mean_r2 >= 0.3
            ):
                scarp_face_result = {
                    'dip_angle': float(sf_dip),
                    'scarp_slope': float(mean_slope),
                    'r2': float(mean_r2),
                    'n_profiles': int(len(scarp_slopes)),
                }

        # -----------------------------------------------------------
        # Phase 4: Select best result
        # -----------------------------------------------------------

        result = None
        method = None

        if v_deflection_result is not None:
            result = v_deflection_result
            method = 'contour_v_deflection'
        elif scarp_face_result is not None:
            result = scarp_face_result
            method = 'scarp_face_slope'

        if result is None:
            return None

        dip_angle = result['dip_angle']

        # Dip direction
        if method == 'contour_v_deflection':
            m_n = result['m_n']
            if m_n < 0:
                dip_direction = perp_azimuth
            else:
                dip_direction = (perp_azimuth + 180.0) % 360.0
        else:
            # Scarp-face slope: scarp faces downdip
            slope = result.get('scarp_slope', 0)
            if slope < 0:
                dip_direction = perp_azimuth
            else:
                dip_direction = (perp_azimuth + 180.0) % 360.0

        # -----------------------------------------------------------
        # Statistics
        # -----------------------------------------------------------
        scarp_heights = np.array([p['h'] for p in profile_data])
        median_scarp_height = float(np.median(scarp_heights))
        along_trace_relief = float(np.max(z_uppers) - np.min(z_uppers))
        v_signal_ratio = along_trace_relief / median_scarp_height if median_scarp_height > 0 else 0

        # Confidence
        r2 = result.get('r2', 0)
        n_pts = result.get('n_points', len(profile_data))

        if (
            method == 'contour_v_deflection'
            and n_pts >= 15
            and r2 >= 0.5
            and v_signal_ratio > 0.10
        ):
            confidence = 'high'
        elif (
            method == 'contour_v_deflection'
            and n_pts >= 8
            and r2 >= 0.25
        ):
            confidence = 'moderate'
        elif method == 'scarp_face_slope' and r2 >= 0.5:
            confidence = 'moderate'
        else:
            confidence = 'low'

        # RMSE
        if 'delta_n' in result:
            dn = np.array(result['delta_n'])
            dz = np.array(result['delta_z'])
            dz_pred = result['m_n'] * dn + result.get('c', 0)
            rmse = float(np.sqrt(np.mean((dz - dz_pred) ** 2)))
        else:
            rmse = 0.0

        return {
            'dip_angle': float(dip_angle),
            'dip_direction': float(dip_direction),

            'm_n': float(result.get('m_n', result.get('scarp_slope', 0))),
            'c': float(result.get('c', 0)),

            'r2': float(r2),
            'rmse': rmse,

            'n_contours': int(n_pts),
            'n_profiles': int(len(profile_data)),

            'scarp_height': float(median_scarp_height),
            'along_trace_relief': float(along_trace_relief),
            'v_signal_ratio': float(v_signal_ratio),

            'confidence': confidence,

            'contour_interval': 0.0,
            'sample_spacing': float(sample_spacing),

            'apex_s': [p['s'] for p in profile_data],
            'apex_n': result.get('delta_n', [p['n_center'] for p in profile_data]),
            'apex_z': result.get('delta_z', [p['z_upper'] for p in profile_data]),

            'method': method,
        }

    
    # NOTE: The former _estimate_local_dip_gradient_fallback has been
    # removed. It measured the average perpendicular terrain gradient
    # (dz/dn across the DEM window), which is terrain slope, not fault
    # dip. It was returning ~0.3-2 degree values that looked like real
    # fault dips but were actually just the topographic slope.
    #
    # If contour-based Rule-of-V estimation fails, we return None and
    # report that the dip could not be estimated at that feature.


    # Main processing
    seg_data = []
    for seg in segments:
        trace = seg["trace"]
        trace_id = seg["trace_id"]
        is_fragment = trace.get("trace_type") == "fragment"
        coords = trace.get("coords", np.array([]))

        if is_fragment or not seg.get("results"):
            if len(coords) < 2:
                continue
            elev = _sample_dem_elevations_patched(
                coords[:, 0], coords[:, 1],
                dem_data, transform, dem_shape, dem10m_data=dem10m_data,
            )
            valid = np.isfinite(elev)
            if not np.any(valid):
                continue
            x_valid = coords[valid, 0]
            y_valid = coords[valid, 1]
            z_valid = elev[valid]
            s_dist = _calculate_along_trace_distance(x_valid, y_valid)
            
            seg_data.append({
                "trace_id": trace_id,
                "is_fragment": True,
                "x": x_valid,
                "y": y_valid,
                "z": z_valid,
                "s": s_dist,
                "color": _PER_TRACE_PALETTE[len(seg_data) % len(_PER_TRACE_PALETTE)],
                "vw_features": []
            })
        else:
            results = seg["results"]
            xs = np.array([r.get("x", np.nan) for r in results], dtype=float)
            ys = np.array([r.get("y", np.nan) for r in results], dtype=float)
            valid_xy = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(valid_xy):
                continue
            xs = xs[valid_xy]
            ys = ys[valid_xy]
            
            elev = _sample_dem_elevations_patched(
                xs, ys, dem_data, transform, dem_shape,
                dem10m_data=dem10m_data,
            )
            valid = np.isfinite(elev)
            if not np.any(valid):
                continue
            
            x_valid = xs[valid]
            y_valid = ys[valid]
            z_valid = elev[valid]
            
            # Sort by along-trace order (preserve original station ordering)
            # The results array is already ordered along the trace; use that
            # ordering rather than lexsort which scrambles the sequence
            original_indices = np.where(valid)[0]
            # Maintain the original order from the results array
            x_valid = xs[valid]
            y_valid = ys[valid]
            z_valid = elev[valid]
            
            # Calculate along-trace distance
            s_dist = _calculate_along_trace_distance(x_valid, y_valid)
            
            # Identify V/∧ features
            valley_idx, ridge_idx, valley_props, ridge_props = _identify_vw_features(
                s_dist, z_valid
            )
            
            z_range = float(np.nanmax(z_valid) - np.nanmin(z_valid)) if len(z_valid) > 0 else 0.0
            print(f"  [EED] Trace {trace_id}: {len(x_valid)} pts, z_range={z_range:.1f}m, "
                  f"{len(valley_idx)} valleys, {len(ridge_idx)} ridges")
            
            # Compute global azimuth for fallback
            global_az = _estimate_global_azimuth(x_valid, y_valid)
            
            # Process each V/∧ feature locally
            vw_features = []
            
            for idx, feature_type in [(v, 'V') for v in valley_idx] + [(r, '∧') for r in ridge_idx]:
                # Get feature location
                x_feature = x_valid[idx]
                y_feature = y_valid[idx]
                z_feature = z_valid[idx]
                s_feature = s_dist[idx]
                
                # Estimate local azimuth with global fallback
                local_azimuth = _estimate_local_azimuth(
                    x_valid, y_valid, idx, global_azimuth=global_az
                )
                
                # Extract local DEM window (200 m, high-res DEM)
                local_dem = _extract_local_dem_window(
                    x_feature, y_feature, local_azimuth,
                    dem_data, transform, dem_shape,
                    window_size=200.0,  # 100 m each side
                    dem10m_data=dem10m_data,
                    dem_hr_path=dem_hr_path
                )
                
                # Estimate local dip using Rule-of-V contour geometry
                dip_result = None
                if local_dem is not None:
                    dip_result = _estimate_local_dip_from_v_geometry(
                        local_dem, z_feature,
                        local_dem['along_vector'], local_dem['perp_vector']
                    )
                
                # If dip estimation failed, try with wider window as fallback
                if dip_result is None:
                    local_dem_wide = _extract_local_dem_window(
                        x_feature, y_feature, local_azimuth,
                        dem_data, transform, dem_shape,
                        window_size=400.0,  # 200 m each side
                        dem10m_data=dem10m_data,
                        dem_hr_path=dem_hr_path
                    )
                    if local_dem_wide is not None:
                        dip_result = _estimate_local_dip_from_v_geometry(
                            local_dem_wide, z_feature,
                            local_dem_wide['along_vector'], local_dem_wide['perp_vector']
                        )
                
                
                if dip_result is not None:
                    method = dip_result.get('method', 'contour')
                    print(f"    -> Feature {feature_type} at s={s_feature:.0f}m: "
                          f"dip={dip_result['dip_angle']:.1f}deg, "
                          f"dir={dip_result['dip_direction']:.0f}deg, "
                          f"r2={dip_result['r2']:.2f} ({method})")
                else:
                    # Report failure — do NOT use terrain-slope fallback
                    print(f"    -> Feature {feature_type} at s={s_feature:.0f}m: "
                          f"dip estimation FAILED (insufficient contour V geometry)")
                
                vw_features.append({
                    'type': feature_type,
                    'index': idx,
                    's': s_feature,
                    'x': x_feature,
                    'y': y_feature,
                    'z': z_feature,
                    'azimuth': local_azimuth,
                    'dip': dip_result
                })
            
            seg_data.append({
                "trace_id": trace_id,
                "is_fragment": False,
                "x": x_valid,
                "y": y_valid,
                "z": z_valid,
                "s": s_dist,
                "color": _PER_TRACE_PALETTE[len(seg_data) % len(_PER_TRACE_PALETTE)],
                "vw_features": vw_features
            })

    if not seg_data:
        return

    # Visualization
    vert_exag = 10.0
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], 
                          hspace=0.3, wspace=0.3)
    ax_profile = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_dip_summary = fig.add_subplot(gs[1, :])

    # Plot each segment
    for sd in seg_data:
        s_dist = sd["s"]
        zs = sd["z"]
        zs_plot = zs * vert_exag
        color = sd["color"]
        
        marker = "D" if sd["is_fragment"] else "o"
        size = 20 if sd["is_fragment"] else 12
        
        # Main profile plot
        ax_profile.scatter(s_dist, zs_plot, color=color, s=size, alpha=0.6,
                    marker=marker, edgecolors="black", linewidth=0.3,
                    label=f"Trace {sd['trace_id']}")
        ax_profile.plot(s_dist, zs_plot, "-", color=color, 
                       linewidth=1.0, alpha=0.4)
        
        # Plot V/∧ features
        for feature in sd.get("vw_features", []):
            if feature['type'] == 'V':
                marker_style = 'v'
                marker_color = 'red'
                marker_size = 120
            else:  # ∧
                marker_style = '^'
                marker_color = 'blue'
                marker_size = 120
            
            ax_profile.scatter(feature['s'], feature['z'] * vert_exag,
                             marker=marker_style, s=marker_size, 
                             color=marker_color,
                             edgecolors="black", linewidth=1.5,
                             zorder=10, alpha=0.9)
            
            # Annotate with dip if available
            if feature.get('dip'):
                dip = feature['dip']
                ax_profile.annotate(
                    f"{dip['dip_angle']:.0f}°",
                    xy=(feature['s'], feature['z'] * vert_exag),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, fontweight="bold",
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                             edgecolor='black', alpha=0.8),
                    zorder=15
                )
        
        # Trace ID annotation
        if len(s_dist) > 0:
            cs = float(np.median(s_dist))
            cz = float(np.median(zs_plot))
            ax_profile.annotate(
                f"{sd['trace_id']}", xy=(cs, cz),
                fontsize=8, fontweight="bold", color=color,
                xytext=(6, 6), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.8),
                zorder=15,
            )
        
        # Map view
        if len(sd["x"]) > 0:
            ax_map.scatter(sd["x"], sd["y"], color=color, s=8,
                          alpha=0.6, marker=marker)
            ax_map.plot(sd["x"], sd["y"], "-", color=color, 
                       linewidth=1.0, alpha=0.4)
            
            # Plot V/∧ features on map with dip directions
            for feature in sd.get("vw_features", []):
                if feature['type'] == 'V':
                    marker_style = 'v'
                    marker_color = 'red'
                else:
                    marker_style = '^'
                    marker_color = 'blue'
                
                ax_map.scatter(feature['x'], feature['y'],
                             marker=marker_style, s=100,
                             color=marker_color,
                             edgecolors="black", linewidth=1.0,
                             zorder=10, alpha=0.8)
                
                # Plot dip direction arrow if available
                if feature.get('dip'):
                    dip = feature['dip']
                    dip_azimuth_rad = np.deg2rad(dip['dip_direction'])
                    dip_vector = np.array([np.sin(dip_azimuth_rad), 
                                          np.cos(dip_azimuth_rad)])
                    
                    # Scale arrow based on dip angle
                    arrow_length = 200.0  # meters
                    arrow_start = np.array([feature['x'], feature['y']])
                    arrow_end = arrow_start + dip_vector * arrow_length
                    
                    ax_map.annotate('', xy=arrow_end, xytext=arrow_start,
                                  arrowprops=dict(arrowstyle='->', 
                                                color='green', linewidth=2, 
                                                alpha=0.7))
        
        # Dip summary panel
        if sd.get("vw_features"):
            feature_data = []
            for feature in sd["vw_features"]:
                if feature.get('dip'):
                    feature_data.append([
                        feature['s'],
                        feature['dip']['dip_angle'],
                        feature['type'],
                        feature['dip']['r2']
                    ])
            
            if feature_data:
                feature_data = np.array(feature_data)
                s_values = feature_data[:, 0]
                dip_values = feature_data[:, 1]
                feature_types = feature_data[:, 2]
                r2_values = feature_data[:, 3]
                
                # Separate V and ∧
                v_mask = feature_types == 'V'
                ridge_mask = feature_types == '∧'
                
                # Plot dip angles for each feature
                ax_dip_summary.scatter(s_values[v_mask], dip_values[v_mask],
                                     marker='v', s=80, color='red',
                                     edgecolors='black', linewidth=0.5,
                                     alpha=0.7, label=f"Trace {sd['trace_id']} V")
                ax_dip_summary.scatter(s_values[ridge_mask], dip_values[ridge_mask],
                                     marker='^', s=80, color='blue',
                                     edgecolors='black', linewidth=0.5,
                                     alpha=0.7, label=f"Trace {sd['trace_id']} ∧")
                
                # Connect features with line for same trace
                sort_idx = np.argsort(s_values)
                ax_dip_summary.plot(s_values[sort_idx], dip_values[sort_idx],
                                  '--', color=color, linewidth=1.0, alpha=0.3)

    # Set up profile axes
    ax_profile.set_xlabel("Distance along fault trace (m)", fontsize=14)
    ax_profile.set_ylabel(f"Elevation (m × {vert_exag:.0f})", fontsize=14)
    ax_profile.set_title(f"{trace_stub} - Elevation Profile with Local Dip Estimates\n"
                        "(▼: valleys, ▲: ridges, numbers: dip angle)",
                 fontsize=14, fontweight="bold")
    ax_profile.tick_params(labelsize=12)
    ax_profile.grid(True, alpha=0.3, linestyle="--")
    ax_profile.legend(fontsize=9, loc="best", framealpha=0.85, ncol=2)

    # Set up map axes
    ax_map.set_xlabel("Easting (m)", fontsize=12)
    ax_map.set_ylabel("Northing (m)", fontsize=12)
    ax_map.set_title("Map View with Local Dip Directions", fontsize=14, fontweight="bold")
    ax_map.tick_params(labelsize=10)
    ax_map.grid(True, alpha=0.3, linestyle="--")
    ax_map.set_aspect('equal', adjustable='box')

    # Set up dip summary axes
    ax_dip_summary.set_xlabel("Distance along fault trace (m)", fontsize=14)
    ax_dip_summary.set_ylabel("Local Dip Angle (degrees)", fontsize=14)
    ax_dip_summary.set_title("Local Dip Estimates from V/∧ Features", 
                            fontsize=14, fontweight="bold")
    ax_dip_summary.tick_params(labelsize=12)
    ax_dip_summary.grid(True, alpha=0.3, linestyle="--")
    ax_dip_summary.legend(fontsize=9, loc='best', framealpha=0.85, ncol=3)

    # --- Add significant‑figure formatting ---
    # --- Add significant‑figure formatting ---
    # y-axis: 2 significant figures
    ax_dip_summary.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2g}")
    )

    # x-axis: 4 significant figures
    ax_dip_summary.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.4g}")
    )

    fig.tight_layout()
    out_path = eed_dir / f"{trace_stub}_local_dip_analysis.png"
    _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"  [Local Dip Analysis] Saved to {out_path}")


# ===================================================================
# 11.  3D VISUALISATIONS
# ===================================================================

def plot_3d_visualizations(results, profiles_cache, trace, dem_data,
                           transform, dem_shape, output_dir):
    """Generate improved 3D visualisations for ~2 km chunks of a trace."""
    if not results:
        return

    coords = trace["coords"]
    trace_id = trace["trace_id"]
    total_len = trace["length_m"]

    n_chunks = max(1, int(np.ceil(total_len / CHUNK_3D_LENGTH_M)))
    chunk_len = total_len / n_chunks

    viz_dir = output_dir / "3d_visualizations" / f"segment_{trace_id}"
    safe_makedirs(viz_dir)

    # Use stronger local exaggeration in 3D plots for better readability.
    vert_exag = 8.0

    for chunk_i in range(n_chunks):
        d_start = chunk_i * chunk_len
        d_end = (chunk_i + 1) * chunk_len

        # Select stations in this chunk
        d_along = np.array([r["distance_along"] for r in results])
        chunk_mask = (d_along >= d_start) & (d_along <= d_end)
        chunk_indices = np.where(chunk_mask)[0]

        if len(chunk_indices) < 3:
            continue

        # Build a local DEM grid for the chunk
        chunk_stations = np.array([[results[i]["x"], results[i]["y"]]
                                   for i in chunk_indices])

        # Buffer around the chunk
        buffer_m = 300.0
        x_min = chunk_stations[:, 0].min() - buffer_m
        x_max = chunk_stations[:, 0].max() + buffer_m
        y_min = chunk_stations[:, 1].min() - buffer_m
        y_max = chunk_stations[:, 1].max() + buffer_m

        # Read local DEM window
        inv_transform = ~transform
        col_min, row_min = inv_transform * (x_min, y_max)
        col_max, row_max = inv_transform * (x_max, y_min)

        col_min = max(0, int(col_min))
        col_max = min(dem_shape[1], int(col_max) + 1)
        row_min = max(0, int(row_min))
        row_max = min(dem_shape[0], int(row_max) + 1)

        if col_max <= col_min or row_max <= row_min:
            continue

        local_dem = dem_data[row_min:row_max, col_min:col_max].astype(float)

        # Subsample for 3D plotting (max ~120x120 grid)
        h, w = local_dem.shape
        step = max(1, max(h, w) // 120)

        # Create coordinate grids
        local_cols = np.arange(0, w, step)
        local_rows = np.arange(0, h, step)
        grid_cols, grid_rows = np.meshgrid(local_cols, local_rows)

        # Convert pixel offsets to CRS coordinates.
        full_cols = col_min + grid_cols
        full_rows = row_min + grid_rows
        grid_x = np.zeros_like(grid_cols, dtype=float)
        grid_y = np.zeros_like(grid_rows, dtype=float)

        for rr in range(grid_rows.shape[0]):
            for cc in range(grid_cols.shape[1]):
                gx, gy = transform * (full_cols[rr, cc], full_rows[rr, cc])
                grid_x[rr, cc] = gx
                grid_y[rr, cc] = gy

        grid_z = local_dem[::step, ::step]
        valid_mask = np.isfinite(grid_z)
        grid_z_plot = np.where(valid_mask, grid_z * vert_exag, np.nan)

        z_min = np.nanmin(grid_z)
        z_max = np.nanmax(grid_z)
        z_norm = (grid_z - z_min) / (z_max - z_min + 1e-10)
        terrain_colors = plt.cm.terrain(z_norm)

        # Extract cleaned trace points that lie on this terrain chunk.
        trace_x, trace_y, trace_z = [], [], []
        for pt in coords:
            pt_col, pt_row = inv_transform * (pt[0], pt[1])
            ri, ci = int(round(pt_row)), int(round(pt_col))
            if 0 <= ri < dem_shape[0] and 0 <= ci < dem_shape[1]:
                elev = dem_data[ri, ci]
                if (np.isfinite(elev) and
                        x_min <= pt[0] <= x_max and
                        y_min <= pt[1] <= y_max):
                    trace_x.append(pt[0])
                    trace_y.append(pt[1])
                    trace_z.append(elev * vert_exag)

        # Three views: plan, oblique NE, oblique SW
        views = [
            ("view_plan", 90, -90, "Plan View"),
            ("view_oblique_NE", 35, 45, "Oblique NE View"),
            ("view_oblique_SW", 35, 225, "Oblique SW View"),
        ]

        for view_name, elev_angle, azim_angle, title_suffix in views:
            fig = plt.figure(figsize=(16, 14))
            ax = fig.add_subplot(111, projection="3d")

            # Plot terrain surface
            ax.plot_surface(
                grid_x, grid_y, grid_z_plot,
                facecolors=terrain_colors,
                alpha=0.85, linewidth=0, antialiased=True,
                shade=True, rstride=1, cstride=1,
            )

            mappable = plt.cm.ScalarMappable(cmap=plt.cm.terrain)
            mappable.set_array(grid_z[valid_mask])
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.12)
            cbar.set_label("Elevation (m)", fontsize=10, labelpad=10)
            cbar.ax.tick_params(labelsize=8)

            if trace_x:
                ax.scatter(
                    trace_x, trace_y, trace_z,
                    color="red", s=20, alpha=0.9,
                    edgecolors="darkred", linewidth=0.3,
                    label="Cleaned Trace", zorder=6,
                )

            # Plot crest/base points at profile-derived coordinates.
            crest_x, crest_y, crest_z = [], [], []
            base_x, base_y, base_z = [], [], []

            for idx in chunk_indices[::3]:
                r = results[idx]

                station_x, station_y = r["x"], r["y"]
                station_az = r.get("azimuth", trace.get("azimuth_deg", 0.0))
                perp_az = (station_az + 90.0) % 360.0
                rad = np.radians(perp_az)

                if np.isfinite(r.get("z_crest_primary", np.nan)) and np.isfinite(r.get("d_crest", np.nan)):
                    cx = station_x + r["d_crest"] * np.sin(rad)
                    cy = station_y + r["d_crest"] * np.cos(rad)
                    cz = r["z_crest_primary"]
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        crest_x.append(cx)
                        crest_y.append(cy)
                        crest_z.append(cz * vert_exag)

                if np.isfinite(r.get("z_base_primary", np.nan)) and np.isfinite(r.get("d_base", np.nan)):
                    bx = station_x + r["d_base"] * np.sin(rad)
                    by = station_y + r["d_base"] * np.cos(rad)
                    bz = r["z_base_primary"]
                    if x_min <= bx <= x_max and y_min <= by <= y_max:
                        base_x.append(bx)
                        base_y.append(by)
                        base_z.append(bz * vert_exag)

            if crest_x:
                ax.scatter(
                    crest_x, crest_y, crest_z,
                    color="orange", s=40, alpha=0.9,
                    edgecolors="darkorange", linewidth=0.5,
                    label="Scarp Crest", zorder=5,
                )

            if base_x:
                ax.scatter(
                    base_x, base_y, base_z,
                    color="cyan", s=40, alpha=0.9,
                    edgecolors="darkcyan", linewidth=0.5,
                    label="Scarp Base", zorder=5,
                )

            # Draw crest-base links in profile order where both points exist.
            for idx in chunk_indices[::3]:
                r = results[idx]
                if not (np.isfinite(r.get("d_crest", np.nan)) and np.isfinite(r.get("d_base", np.nan))):
                    continue
                if not (np.isfinite(r.get("z_crest_primary", np.nan)) and np.isfinite(r.get("z_base_primary", np.nan))):
                    continue

                station_x, station_y = r["x"], r["y"]
                station_az = r.get("azimuth", trace.get("azimuth_deg", 0.0))
                perp_az = (station_az + 90.0) % 360.0
                rad = np.radians(perp_az)

                cx = station_x + r["d_crest"] * np.sin(rad)
                cy = station_y + r["d_crest"] * np.cos(rad)
                cz = r["z_crest_primary"] * vert_exag

                bx = station_x + r["d_base"] * np.sin(rad)
                by = station_y + r["d_base"] * np.cos(rad)
                bz = r["z_base_primary"] * vert_exag

                if (x_min <= cx <= x_max and y_min <= cy <= y_max and
                        x_min <= bx <= x_max and y_min <= by <= y_max):
                    ax.plot(
                        [cx, bx], [cy, by], [cz, bz],
                        color="purple", linewidth=1.5,
                        linestyle="--", alpha=0.5, zorder=4,
                    )

            # Add a north arrow in map coordinates.
            north_x = x_max - (x_max - x_min) * 0.1
            north_y = y_max - (y_max - y_min) * 0.15
            arrow_length = (x_max - x_min) * 0.08

            col_n, row_n = inv_transform * (north_x, north_y)
            ri, ci = int(row_n), int(col_n)
            if 0 <= ri < dem_shape[0] and 0 <= ci < dem_shape[1]:
                north_z = float(dem_data[ri, ci]) * vert_exag
                if not np.isfinite(north_z):
                    north_z = float(np.nanmedian(grid_z_plot))
            else:
                north_z = float(np.nanmedian(grid_z_plot))

            ax.quiver(
                north_x, north_y, north_z,
                0.0, arrow_length, 0.0,
                color="black", arrow_length_ratio=0.3,
                linewidth=2.0, zorder=10,
            )
            ax.text(
                north_x, north_y + arrow_length * 0.3, north_z,
                "N", fontsize=16, fontweight="bold",
                color="black", zorder=10,
            )

            ax.set_xlabel("Easting (m)", fontsize=12, labelpad=12)
            ax.set_ylabel("Northing (m)", fontsize=12, labelpad=12)
            ax.set_zlabel(f"Elevation (m x {vert_exag:.0f})",
                          fontsize=12, labelpad=12)
            ax.view_init(elev=elev_angle, azim=azim_angle)
            ax.set_title(f"Trace {trace_id} - Chunk {chunk_i + 1}/{n_chunks}\n"
                         f"{title_suffix} (d={d_start:.0f}-{d_end:.0f} m)",
                         fontsize=13, pad=20)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

            z_span = np.nanmax(grid_z_plot) - np.nanmin(grid_z_plot)
            ax.set_box_aspect([
                (x_max - x_min),
                (y_max - y_min),
                z_span * 0.5 if np.isfinite(z_span) and z_span > 0 else 1.0,
            ])

            chunk_results = [results[i] for i in chunk_indices if results[i] is not None]
            if chunk_results:
                median_params = {
                    "Vertical separation (m)": np.nanmedian([r.get("primary_throw", np.nan) for r in chunk_results]),
                    "Potential Vertical Offset (m)": np.nanmedian([r.get("true_vertical_displacement", np.nan) for r in chunk_results]),
                    "Scarp Width (m)": np.nanmedian([r.get("scarp_width_m", np.nan) for r in chunk_results]),
                    "Scarp Slope (deg)": np.nanmedian([r.get("scarp_slope_deg", np.nan) for r in chunk_results]),
                    "Scarp Strike (deg)": np.nanmedian([r.get("strike", np.nan) for r in chunk_results]),
                    "Scarp Dip (deg)": np.nanmedian([r.get("dip", np.nan) for r in chunk_results]),
                    "Profile Quality": np.nanmedian([r.get("profile_quality", np.nan) for r in chunk_results]),
                }
                median_params = {k: v for k, v in median_params.items() if np.isfinite(v)}

                if median_params:
                    table_lines = ["Median Parameters:", "-" * 30]
                    for param, value in median_params.items():
                        table_lines.append(f"{param}: {value:.2f}")
                    plt.figtext(
                        0.02, 0.02, "\n".join(table_lines),
                        fontsize=7, fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                  alpha=0.85, edgecolor="gray"),
                        verticalalignment="bottom", horizontalalignment="left",
                        zorder=20,
                    )

            out_path = viz_dir / f"{view_name}_chunk{chunk_i + 1:02d}.png"
            _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.5)

        print(f"  [3D] Chunk {chunk_i + 1}/{n_chunks}: 3 views saved to {viz_dir}")


# ===================================================================
# 11b.  PER-ORIGINAL-TRACE (Trace1..N) CORRIDOR 3D OVERVIEW
# ===================================================================

def _smooth_segment_polyline(stations_xy, sigma=POLYLINE_SMOOTH_SIGMA):
    """Gaussian-smooth a segment's station (x, y) sequence into a bendable
    best-fit centerline (distinct from the raw digitized trace points)."""
    stations_xy = np.asarray(stations_xy, dtype=np.float64)
    if len(stations_xy) < 3:
        return stations_xy
    xs = gaussian_filter1d(stations_xy[:, 0], sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(stations_xy[:, 1], sigma=sigma, mode="nearest")
    return np.column_stack([xs, ys])


def _sample_dem_elevations(xs, ys, dem_data, transform, dem_shape):
    """Bilinearly sample DEM elevations at arbitrary map (x, y) points."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    elevations = np.full(len(xs), np.nan, dtype=np.float64)
    valid = (cols >= 0) & (cols < dem_shape[1]) & (rows >= 0) & (rows < dem_shape[0])
    if not np.any(valid):
        return elevations
    coords_rc = np.array([rows[valid], cols[valid]])
    elevations[valid] = map_coordinates(dem_data.astype(np.float64), coords_rc,
                                        order=1, mode="constant", cval=np.nan)
    return elevations


def _build_trace_corridor_grid(trace_points_xy, dem_data, transform, dem_shape,
                               buffer_m=TRACE_CORRIDOR_BUFFER_M, max_grid=250,
                               dem10m_data=None):
    """Build a subsampled DEM grid covering the full rectangular extent
    determined by the trace points plus a buffer. No distance-to-trace mask is
    applied; the entire rectangle is returned, ready to fill the 3D figure.

    If *dem10m_data* is provided, nodata cells in *dem_data* are patched
    with values from *dem10m_data* (must share the same transform/shape).
    Returns an extra *lidar_valid_mask* indicating which grid cells had
    original lidar coverage (before patching).
    """
    pts = np.asarray(trace_points_xy, dtype=np.float64)
    if len(pts) < 2:
        return None

    x_min = float(pts[:, 0].min() - buffer_m)
    x_max = float(pts[:, 0].max() + buffer_m)
    y_min = float(pts[:, 1].min() - buffer_m)
    y_max = float(pts[:, 1].max() + buffer_m)

    inv_transform = ~transform
    col_min, row_min = inv_transform * (x_min, y_max)
    col_max, row_max = inv_transform * (x_max, y_min)
    col_min = max(0, int(col_min))
    col_max = min(dem_shape[1], int(col_max) + 1)
    row_min = max(0, int(row_min))
    row_max = min(dem_shape[0], int(row_max) + 1)
    if col_max <= col_min or row_max <= row_min:
        return None

    local_dem = dem_data[row_min:row_max, col_min:col_max].astype(float)
    lidar_valid_local = np.isfinite(local_dem) & (local_dem > -1000)

    if dem10m_data is not None:
        local_dem10m = dem10m_data[row_min:row_max, col_min:col_max].astype(float)
        patch_mask = ~lidar_valid_local & np.isfinite(local_dem10m) & (local_dem10m > -1000)
        if np.any(patch_mask):
            local_dem[patch_mask] = local_dem10m[patch_mask]

    h, w = local_dem.shape
    step = max(1, max(h, w) // max_grid)

    local_cols = np.arange(0, w, step)
    local_rows = np.arange(0, h, step)
    grid_cols, grid_rows = np.meshgrid(local_cols, local_rows)
    full_cols = col_min + grid_cols
    full_rows = row_min + grid_rows

    grid_x = np.zeros_like(grid_cols, dtype=float)
    grid_y = np.zeros_like(grid_rows, dtype=float)
    for rr in range(grid_rows.shape[0]):
        for cc in range(grid_cols.shape[1]):
            gx, gy = transform * (full_cols[rr, cc], full_rows[rr, cc])
            grid_x[rr, cc] = gx
            grid_y[rr, cc] = gy

    grid_z = local_dem[::step, ::step]
    lidar_valid_grid = lidar_valid_local[::step, ::step]

    # No corridor mask: use the whole rectangular extent.
    valid_mask = np.isfinite(grid_z) & (grid_z > -1000)

    return grid_x, grid_y, grid_z, valid_mask, (x_min, x_max, y_min, y_max), lidar_valid_grid


def _sample_dem_elevations_patched(xs, ys, dem_data, transform, dem_shape,
                                   dem10m_data=None):
    """Bilinearly sample DEM elevations at arbitrary map (x, y) points,
    falling back to *dem10m_data* where *dem_data* has nodata."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    elevations = np.full(len(xs), np.nan, dtype=np.float64)
    valid = (cols >= 0) & (cols < dem_shape[1]) & (rows >= 0) & (rows < dem_shape[0])
    if not np.any(valid):
        return elevations
    coords_rc = np.array([rows[valid], cols[valid]])
    elevations[valid] = map_coordinates(dem_data.astype(np.float64), coords_rc,
                                         order=1, mode="constant", cval=np.nan)
    if dem10m_data is not None:
        need_patch = valid & ~np.isfinite(elevations)
        if np.any(need_patch):
            coords_rc_patch = np.array([rows[need_patch], cols[need_patch]])
            elevations[need_patch] = map_coordinates(
                dem10m_data.astype(np.float64), coords_rc_patch,
                order=1, mode="constant", cval=np.nan)
    return elevations


def _load_fsd1_points(buffer_bounds):
    """Load FSD1 structural points within the given (x_min, x_max, y_min, y_max)
    bounds.  Returns an (N, 2) array of (x, y) coordinates."""
    x_min, x_max, y_min, y_max = buffer_bounds
    try:
        df = pd.read_csv(STRUCT_PTS_PATH, usecols=["XCoord", "YCoord", "source"])
    except FileNotFoundError:
        print(f"  [Warning] {STRUCT_PTS_PATH.name} not found; skipping FSD1 points.")
        return None
    fsd1 = df[df["source"] == "FSD1"]
    in_bounds = ((fsd1["XCoord"] >= x_min) & (fsd1["XCoord"] <= x_max) &
                 (fsd1["YCoord"] >= y_min) & (fsd1["YCoord"] <= y_max))
    pts = fsd1[in_bounds][["XCoord", "YCoord"]].to_numpy(dtype=np.float64)
    print(f"  [FSD1] {len(pts)} points within buffer bounds")
    return pts


# Module-level cache for scored points (loaded once, reused across all traces)
_scored_points_cache = None  # type: ignore[var-annotated]


def _load_scored_points():
    """Load scored points from ``lidar_0_5m_r4_scored_points.csv`` once and
    cache the result.  Points with ``prob_xgb > 0.5`` are retained by default,
    but if none are found we fall back to all points with a positive probability.
    """
    global _scored_points_cache
    if _scored_points_cache is not None:
        return _scored_points_cache

    try:
        df = pd.read_csv(SCORED_POINTS_PATH)
    except FileNotFoundError:
        print(f"  [Warning] {SCORED_POINTS_PATH.name} not found; skipping scored points.")
        _scored_points_cache = False
        return None

    # Try to find the required columns; allow common alternatives.
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("x", "xcoord", "easting", "east"):
            col_map["x"] = col
        elif cl in ("y", "ycoord", "northing", "north"):
            col_map["y"] = col
        elif cl in ("prob_xgb", "prob", "probability", "score"):
            col_map["prob_xgb"] = col

    if "x" not in col_map or "y" not in col_map or "prob_xgb" not in col_map:
        print(f"  [Warning] Scored points file missing required columns; found: {list(df.columns)}")
        _scored_points_cache = False
        return None

    df = df.rename(columns={
        col_map["x"]: "x",
        col_map["y"]: "y",
        col_map["prob_xgb"]: "prob_xgb",
    })
    total = len(df)
    high = df[df["prob_xgb"] > 0.5].copy()
    if len(high) == 0:
        print(f"  [Scored pts] No points with prob_xgb > 0.5; using all {total} points")
        high = df[df["prob_xgb"] > 0.0].copy()

    pts = high[["x", "y", "prob_xgb"]].to_numpy(dtype=np.float64)
    _scored_points_cache = pts
    print(f"  [Scored pts] {len(pts)} points loaded (of {total} total)")
    return pts


def _nearest_scored_median(trace_coords, buffer_m=1000.0):
    """Find scored points (prob_xgb > 0.5, or fallback all) within ``buffer_m``
    of the trace coordinates, then return the median prob_xgb value rounded to
    1 significant figure.

    Parameters
    ----------
    trace_coords : np.ndarray
        (N, 2) array of (x, y) trace coordinates.
    buffer_m : float
        Search radius around each trace point (default 1000 m).

    Returns
    -------
    float or np.nan
        Median prob_xgb of nearby scored points, rounded to 1 significant
        figure, or NaN if none found.
    """
    pts = _load_scored_points()
    if pts is None or not isinstance(pts, np.ndarray) or len(pts) == 0 or len(trace_coords) == 0:
        print("  [Score debug] No scored points available")
        return np.nan

    tc = np.asarray(trace_coords)
    x_min = tc[:, 0].min() - buffer_m
    x_max = tc[:, 0].max() + buffer_m
    y_min = tc[:, 1].min() - buffer_m
    y_max = tc[:, 1].max() + buffer_m

    mask = ((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
            (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max))
    candidates = pts[mask]
    print(f"  [Score debug] Bounding box: x[{x_min:.0f},{x_max:.0f}] y[{y_min:.0f},{y_max:.0f}]")
    print(f"  [Score debug] Scored points in bbox: {len(candidates)}")
    if len(candidates) == 0:
        print("  [Score debug] No candidates in bounding box; try increasing buffer_m")
        return np.nan

    tree = cKDTree(candidates[:, :2])
    trace_xy = tc[:, :2]
    dists, idxs = tree.query(trace_xy, k=1)
    within = dists <= buffer_m
    n_within = np.sum(within)
    print(f"  [Score debug] Trace vertices with a scored point within {buffer_m}m: {n_within}/{len(trace_xy)}")
    if n_within == 0:
        print("  [Score debug] No scored point within buffer of any trace vertex")
        return np.nan

    nearby_scores = candidates[idxs[within], 2]
    median_score = float(np.median(nearby_scores))

    # Round to 1 significant figure
    if median_score == 0:
        return 0.0
    from math import floor, log10
    digits = -floor(log10(abs(median_score)))
    return round(median_score, digits)


def _place_annotation_boxes(ax, ann_texts, ann_positions_3d, grid_z_plot,
                            x_min, x_max, y_min, y_max):
    """Place annotation boxes close to their target points.

    Boxes are positioned locally around each target rather than along the
    plot edges. Candidate positions are tested around each target, and the
    position producing the shortest connector line is selected subject to
    non-overlap with previously placed boxes.

    The boxes are kept within the plot bounds and the orange connector lines
    therefore remain as short as possible.
    """
    n = len(ann_texts)
    if n == 0:
        return

    # ------------------------------------------------------------------
    # Z position: keep all annotation boxes slightly above the terrain.
    # ------------------------------------------------------------------
    finite_z = grid_z_plot[np.isfinite(grid_z_plot)]
    if len(finite_z) == 0:
        return

    z_top = float(np.nanmax(finite_z))
    z_med = float(np.nanmedian(finite_z))

    z_offset = (z_top - z_med) * 0.3 + 5.0
    box_z = z_top + z_offset

    x_range = x_max - x_min
    y_range = y_max - y_min

    # ------------------------------------------------------------------
    # Approximate size of an annotation box in map coordinates.
    #
    # These values are deliberately conservative so that boxes that are
    # visually close to one another are treated as overlapping.
    # ------------------------------------------------------------------
    box_half_width = x_range * 0.035
    box_half_height = y_range * 0.018

    # Minimum gap between neighbouring boxes.
    box_gap_x = x_range * 0.008
    box_gap_y = y_range * 0.008

    # Initial distance of a box from its target.
    initial_offset = min(x_range, y_range) * 0.025
    # Do not draw a connector if the annotation box is this close
    # to the target point.
    min_connector_length = min(x_range, y_range) * 0.1

    # ------------------------------------------------------------------
    # Candidate directions around each target.
    #
    # Ordered approximately from shortest / most natural positions to
    # progressively less desirable ones.
    # ------------------------------------------------------------------
    directions = [
        (1.0, 0.0),     # right
        (-1.0, 0.0),    # left
        (0.0, 1.0),     # north
        (0.7, 0.7),     # NE
        (-0.7, 0.7),    # NW
    ]

    # Store placed boxes as:
    # (box_x, box_y, tx, ty, text)
    placed_boxes = []

    # ------------------------------------------------------------------
    # Process targets in order of their y position.
    #
    # This gives a reasonably stable arrangement while still allowing
    # each box to move locally around its own target.
    # ------------------------------------------------------------------
    items = list(zip(ann_texts, ann_positions_3d))
    items.sort(key=lambda item: -item[1][1])

    box_positions = []

    # ------------------------------------------------------------------
    # Helper: determine whether a candidate box overlaps an existing box.
    # ------------------------------------------------------------------
    def overlaps(candidate_x, candidate_y):
        for existing_x, existing_y, *_ in placed_boxes:

            x_overlap = (
                abs(candidate_x - existing_x)
                < (2.0 * box_half_width + box_gap_x)
            )

            y_overlap = (
                abs(candidate_y - existing_y)
                < (2.0 * box_half_height + box_gap_y)
            )

            if x_overlap and y_overlap:
                return True

        return False

    # ------------------------------------------------------------------
    # Helper: keep the complete box within the plotting region.
    # ------------------------------------------------------------------
    def inside_plot(candidate_x, candidate_y):
        return (
            candidate_x - box_half_width >= x_min and
            candidate_x + box_half_width <= x_max and
            candidate_y - box_half_height >= y_min and
            candidate_y + box_half_height <= y_max
        )

    # ------------------------------------------------------------------
    # Place each annotation.
    # ------------------------------------------------------------------
    for text, (tx, ty, tz) in items:

        best_candidate = None
        best_distance = np.inf

        # Try progressively larger search radii.
        #
        # Most annotations should be placed at the first radius,
        # producing a very short orange line.
        for radius_multiplier in [1.0, 1.5, 2.0, 2.5, 3.0,
                                  4.0, 5.0, 6.0]:

            radius = initial_offset * radius_multiplier

            candidates = []

            for dx, dy in directions:

                # Normalise diagonal directions so that their actual
                # distance from the target is approximately equal.
                norm = np.hypot(dx, dy)
                ux = dx / norm
                uy = dy / norm

                candidate_x = tx + ux * radius
                candidate_y = ty + uy * radius

                if not inside_plot(candidate_x, candidate_y):
                    continue

                if overlaps(candidate_x, candidate_y):
                    continue

                # Actual 2-D connector length.
                distance = np.hypot(
                    candidate_x - tx,
                    candidate_y - ty
                )

                candidates.append(
                    (distance, candidate_x, candidate_y)
                )

            if candidates:
                candidates.sort(key=lambda c: c[0])
                best_candidate = candidates[0]
                best_distance = best_candidate[0]
                break

        # ------------------------------------------------------------------
        # If no completely non-overlapping position could be found,
        # choose the nearest position anyway.
        #
        # This is a fallback for very dense annotation sets.
        # ------------------------------------------------------------------
        if best_candidate is None:

            fallback_candidates = []

            for radius_multiplier in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
                radius = initial_offset * radius_multiplier

                for dx, dy in directions:

                    norm = np.hypot(dx, dy)
                    ux = dx / norm
                    uy = dy / norm

                    candidate_x = tx + ux * radius
                    candidate_y = ty + uy * radius

                    if not inside_plot(candidate_x, candidate_y):
                        continue

                    distance = np.hypot(
                        candidate_x - tx,
                        candidate_y - ty
                    )

                    fallback_candidates.append(
                        (distance, candidate_x, candidate_y)
                    )

                if fallback_candidates:
                    break

            if fallback_candidates:
                fallback_candidates.sort(key=lambda c: c[0])
                best_candidate = fallback_candidates[0]

        # ------------------------------------------------------------------
        # Save the selected position.
        # ------------------------------------------------------------------
        if best_candidate is not None:

            _, box_x, box_y = best_candidate

            placed_boxes.append(
                (box_x, box_y, tx, ty, text)
            )

            box_positions.append(
                [box_x, box_y, box_z, text, tx, ty, tz]
            )

    # ------------------------------------------------------------------
    # Draw the annotation boxes and short orange connector lines.
    # ------------------------------------------------------------------
    for bp in box_positions:

        box_x, box_y, box_z_i, text, tx, ty, tz = bp

        # Annotation box.
        ax.text(
            box_x, box_y, box_z_i, text,
            fontsize=7,
            fontweight="bold",
            bbox=dict(
                facecolor="none",
                edgecolor="none",
                alpha=0,
            ),
            zorder=200,
        )
        # Draw connector only when the annotation box is sufficiently
        # far from the target point.
        connector_length = np.hypot(
            box_x - tx,
            box_y - ty
        )

        if connector_length >= min_connector_length:
            ax.plot(
                [box_x, tx],
                [box_y, ty],
                [box_z_i, tz],
                color="black",
                linewidth=1,
                alpha=0.9,
                zorder=199,
            )



def _save_trace_parameter_table(file_name, segments, output_dir):
    """Save the per-segment parameter table as a CSV alongside the 3D
    overview figures in ``3d_visualizations/by_original_trace/<stub>/``.

    Columns: trace_id, trace_type, n_stations, n_points, length_m,
    median_VS_m, median_scarp_width_m, azimuth_deg, median_scored_prob
    """
    trace_stub = Path(file_name).stem
    viz_dir = output_dir / "3d_visualizations" / "by_original_trace" / trace_stub
    safe_makedirs(viz_dir)
    csv_path = viz_dir / f"{trace_stub}_parameter_table.csv"

    rows = []
    for seg in segments:
        trace_id = seg["trace_id"]
        trace = seg["trace"]
        results = seg["results"]
        is_fragment = trace.get("trace_type") == "fragment"

        n_stations = len(results)
        n_points = trace.get("n_points", len(trace.get("coords", [])))
        length_m = trace.get("length_m", 0.0)
        az = trace.get("azimuth_deg", np.nan)

        if is_fragment or n_stations == 0:
            rows.append({
                "trace_id": trace_id,
                "trace_type": "fragment" if is_fragment else "segment",
                "n_stations": 0,
                "n_points": n_points,
                "length_m": length_m,
                "median_VS_m": np.nan,
                "median_scarp_width_m": np.nan,
                "azimuth_deg": az,
                "median_scored_prob": np.nan,
            })
            continue

        vs_vals = [r.get("true_vertical_displacement", np.nan) for r in results]
        width_vals = [r.get("scarp_width_m", np.nan) for r in results]
        coords = trace.get("coords", np.array([]))
        scored_median = _nearest_scored_median(coords) if len(coords) > 0 else np.nan

        rows.append({
            "trace_id": trace_id,
            "trace_type": "segment",
            "n_stations": n_stations,
            "n_points": n_points,
            "length_m": length_m,
            "median_VS_m": float(np.nanmedian(vs_vals)) if np.any(np.isfinite(vs_vals)) else np.nan,
            "median_scarp_width_m": float(np.nanmedian(width_vals)) if np.any(np.isfinite(width_vals)) else np.nan,
            "azimuth_deg": az,
            "median_scored_prob": scored_median,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  [Table] Saved parameter table to {csv_path}")
    return csv_path


def plot_trace_overview_3d(file_name, segments, dem_data, transform, dem_shape,
                           output_dir, dem10m_data=None):
    """Per-original-trace (e.g. Trace1.xlsx) 3D overview: terrain corridor
    ~2 km either side of the trace, raw digitized points kept, a smoothed
    best-fit polyline per segment, and per-segment parameter annotations.

    Fragments (short traces with trace_type="fragment") are rendered with a
    distinct visual style (diamond markers, warm colour) so they are visible
    in the overview even though they lack profiling results.
    """
    # Save parameter table CSV alongside the 3D overview figures.
    _save_trace_parameter_table(file_name, segments, output_dir)

    # Build corridor from segment stations AND fragment coordinates so that
    # fragments influence the terrain extent.
    all_station_xy = []
    for seg in segments:
        for r in seg["results"]:
            if np.isfinite(r.get("x", np.nan)) and np.isfinite(r.get("y", np.nan)):
                all_station_xy.append([r["x"], r["y"]])
        # Include fragment coords in corridor extent calculation
        if seg["trace"].get("trace_type") == "fragment":
            coords = seg["trace"]["coords"]
            for pt in coords:
                all_station_xy.append([pt[0], pt[1]])
    if len(all_station_xy) < 2:
        return

    grid = _build_trace_corridor_grid(all_station_xy, dem_data, transform, dem_shape,
                                      dem10m_data=dem10m_data)
    if grid is None:
        return
    grid_x, grid_y, grid_z, valid_mask, (x_min, x_max, y_min, y_max), lidar_valid_grid = grid

    vert_exag = 8.0
    segment_offset_m = 10.0

    grid_z_plot = np.where(valid_mask, grid_z * vert_exag, np.nan)
    z_min = np.nanmin(grid_z[valid_mask]) if np.any(valid_mask) else 0.0
    z_max = np.nanmax(grid_z[valid_mask]) if np.any(valid_mask) else 1.0
    z_norm = (grid_z - z_min) / (z_max - z_min + 1e-10)
    terrain_colors = plt.cm.terrain(z_norm)

    seg_colors = _gold_orange_colors(max(sum(seg["trace"].get("trace_type") == "segment" for seg in segments), 1))

    # Load FSD1 neotectonic points within the buffer bounds
    fsd1_pts = _load_fsd1_points((x_min, x_max, y_min, y_max))

    trace_stub = Path(file_name).stem
    viz_dir = output_dir / "3d_visualizations" / "by_original_trace" / trace_stub
    safe_makedirs(viz_dir)

    views = [
        ("view_plan", 90, -90, "Plan View"),
        ("view_oblique_NE", 35, 45, "Oblique NE View"),
        ("view_oblique_SW", 35, 225, "Oblique SW View"),
    ]

    for view_name, elev_angle, azim_angle, title_suffix in views:
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111, projection="3d")

        # Surface with low zorder to ensure it stays below points/lines
        ax.plot_surface(
            grid_x, grid_y, grid_z_plot,
            facecolors=terrain_colors,
            alpha=0.85, linewidth=0, antialiased=True,
            shade=True, rstride=1, cstride=1,
            zorder=1,
        )

        mappable = plt.cm.ScalarMappable(cmap=plt.cm.terrain)
        mappable.set_array(grid_z[valid_mask])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.12)
        cbar.set_label("Elevation (m)", fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        # Plot FSD1 neotectonic feature points and connecting line
        if fsd1_pts is not None and len(fsd1_pts) >= 2:
            fsd1_z = _sample_dem_elevations_patched(
                fsd1_pts[:, 0], fsd1_pts[:, 1],
                dem_data, transform, dem_shape, dem10m_data=dem10m_data)
            fsd1_valid = np.isfinite(fsd1_z)
            if np.any(fsd1_valid):
                fv = fsd1_pts[fsd1_valid]
                fz = fsd1_z[fsd1_valid]
                # Subsample if too many points for 3D rendering
                if len(fv) > 800:
                    idx = np.linspace(0, len(fv) - 1, 800, dtype=int)
                    fv = fv[idx]
                    fz = fz[idx]
                ax.scatter(
                    fv[:, 0], fv[:, 1], (fz + segment_offset_m) * vert_exag,
                    color="red", s=8, alpha=1.0,
                    edgecolors="none",
                    label="neotectonic feature", zorder=90,
                )


        ann_texts = []
        ann_positions_3d = []
        table_lines = [f"{trace_stub} - Trace Parameters:", "-" * 40]

        for si, seg in enumerate(segments):
            trace_id = seg["trace_id"]
            results = seg["results"]
            coords = seg["trace"]["coords"]
            is_fragment = seg["trace"].get("trace_type") == "fragment"

            if is_fragment:
                # --- Fragment rendering: distinct style, no profiling ---
                frag_z = _sample_dem_elevations_patched(
                    coords[:, 0], coords[:, 1],
                    dem_data, transform, dem_shape, dem10m_data=dem10m_data)
                in_bounds = ((coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                             (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
                             np.isfinite(frag_z))
                if np.any(in_bounds):
                    ax.scatter(
                        coords[in_bounds, 0], coords[in_bounds, 1],
                        (frag_z[in_bounds] + segment_offset_m) * vert_exag,
                        color=_FRAGMENT_GREY, s=30, alpha=1.0,
                        marker="D",
                        edgecolors="black", linewidth=0.5,
                        zorder=105,
                    )
                    # Connect fragment points with a thin dashed line
                    if in_bounds.sum() >= 2:
                        ax.plot(
                            coords[in_bounds, 0], coords[in_bounds, 1],
                            (frag_z[in_bounds] + segment_offset_m) * vert_exag,
                            color=_FRAGMENT_GREY, linewidth=1.5, alpha=1.0,
                            linestyle="--", zorder=104,
                        )

                    # --------------------------------------------------------------
                    # Fragment annotation target
                    # Use the midpoint of the valid fragment coordinates.
                    # --------------------------------------------------------------
                    valid_coords = coords[in_bounds]
                    valid_z = frag_z[in_bounds]

                    mid_idx = len(valid_coords) // 2

                    frag_mid_x = float(valid_coords[mid_idx, 0])
                    frag_mid_y = float(valid_coords[mid_idx, 1])
                    frag_mid_z = float(
                        (valid_z[mid_idx] + segment_offset_m) * vert_exag
                    )
                    # Add fragment annotation.
                    ann_texts.append(f"{trace_id}")
                    ann_positions_3d.append(
                        (frag_mid_x, frag_mid_y, frag_mid_z)
                    )

                # Add fragment to the parameter table with a label
                frag_len = seg["trace"].get("length_m", 0)
                table_lines.append(
                    f"Trace {trace_id} [fragment]: "
                    f"{len(coords)} pts, {frag_len:.0f}m"
                )
                continue

            # --- Normal segment rendering ---
            color = seg_colors[si % len(seg_colors)]

            # Raw digitized trace points (kept, unmodified).
            raw_z = _sample_dem_elevations_patched(
                coords[:, 0], coords[:, 1],
                dem_data, transform, dem_shape, dem10m_data=dem10m_data)
            in_bounds = ((coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                         (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
                         np.isfinite(raw_z))
            if np.any(in_bounds):
                ax.scatter(
                    coords[in_bounds, 0], coords[in_bounds, 1],
                    (raw_z[in_bounds] + segment_offset_m) * vert_exag,
                    color=color, s=14, alpha=1.0,
                    edgecolors="black", linewidth=0.4,
                    zorder=100,
                )

            # Smoothed best-fit polyline (can bend), sampled onto the DEM.
            station_xy = np.array([[r["x"], r["y"]] for r in results
                                   if np.isfinite(r.get("x", np.nan))])
            smooth_mid_x = smooth_mid_y = smooth_mid_z = None
            if len(station_xy) >= 3:
                smooth_xy = _smooth_segment_polyline(station_xy)
                smooth_z = _sample_dem_elevations_patched(
                    smooth_xy[:, 0], smooth_xy[:, 1],
                    dem_data, transform, dem_shape, dem10m_data=dem10m_data)
                poly_valid = np.isfinite(smooth_z)
                if poly_valid.sum() >= 2:
                    ax.plot(
                        smooth_xy[poly_valid, 0], smooth_xy[poly_valid, 1],
                        (smooth_z[poly_valid] + segment_offset_m) * vert_exag,
                        color=color, linewidth=3.0, alpha=1.0,
                        zorder=110,
                    )
                    mid_idx = len(smooth_xy[poly_valid]) // 2
                    smooth_mid_x = float(smooth_xy[poly_valid][mid_idx, 0])
                    smooth_mid_y = float(smooth_xy[poly_valid][mid_idx, 1])
                    smooth_mid_z = float((smooth_z[poly_valid][mid_idx] + segment_offset_m) * vert_exag)

            vs = np.nanmedian([r.get("true_vertical_displacement", np.nan) for r in results])
            width = np.nanmedian([r.get("scarp_width_m", np.nan) for r in results])
            az = seg["trace"].get("azimuth_deg", np.nan)
            seg_len = seg["trace"].get("length_m", 0.0)

            # Nearest scored-point prob_xgb median (prob_xgb > 0.5 only)
            scored_median = _nearest_scored_median(coords)

            # Build parameter table line for figtext
            scored_str = f"{scored_median:.1g}" if np.isfinite(scored_median) else "N/A"
            table_lines.append(
                f"Trace {trace_id} (n={len(results)}): "
                f"VS={vs:.1f}m  W={width:.0f}m  "
                f"Az={az:.0f}deg  Len={seg_len:.0f}m  "
                f"Score={scored_str}"
            )

            # Annotation box: only the trace number
            ann_text = f"{trace_id}"
            if smooth_mid_x is not None:
                ann_texts.append(ann_text)
                ann_positions_3d.append((smooth_mid_x, smooth_mid_y, smooth_mid_z))

        # Place non-overlapping annotation boxes with arrows
        _place_annotation_boxes(ax, ann_texts, ann_positions_3d, grid_z_plot,
                                x_min, x_max, y_min, y_max)

        # North arrow in map coordinates.
        north_x = x_max - (x_max - x_min) * 0.1
        north_y = y_max - (y_max - y_min) * 0.15
        arrow_length = (x_max - x_min) * 0.06
        north_z = float(np.nanmedian(grid_z_plot))
        ax.quiver(
            north_x, north_y, north_z,
            0.0, arrow_length, 0.0,
            color="black", arrow_length_ratio=0.3,
            linewidth=2.0, zorder=10,
        )
        ax.text(
            north_x, north_y + arrow_length * 0.3, north_z,
            "N", fontsize=16, fontweight="bold", color="black", zorder=10,
        )

        ax.set_xlabel("Easting (m)", fontsize=12, labelpad=12)
        ax.set_ylabel("Northing (m)", fontsize=12, labelpad=12)
        ax.set_zlabel(f"Elevation (m)", fontsize=12, labelpad=12)
        ax.view_init(elev=elev_angle, azim=azim_angle)
        ax.set_title(f"{trace_stub} - {title_suffix}\n"
                     f"(corridor +/-{TRACE_CORRIDOR_BUFFER_M:.0f} m)",
                     fontsize=13, pad=20)
        ax.tick_params(labelsize=9)

        # Only show legend for neotectonic feature (no legend for trace points/lines)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=1)

        z_span = np.nanmax(grid_z_plot) - np.nanmin(grid_z_plot)
        ax.set_box_aspect([
            (x_max - x_min),
            (y_max - y_min),
            z_span * 0.5 if np.isfinite(z_span) and z_span > 0 else 1.0,
        ])

        # Parameter table at the bottom of the figure
        plt.figtext(
            0.02, 0.02, "\n".join(table_lines),
            fontsize=7, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.85, edgecolor="gray"),
            verticalalignment="bottom", horizontalalignment="left",
            zorder=20,
        )

        out_path = viz_dir / f"{view_name}.png"
        _safe_savefig(fig, out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.5)

    print(f"  [3D by-trace] {trace_stub}: 3 views saved to {viz_dir}")


# ===================================================================
# 12.  SUMMARY AND OUTPUT
# ===================================================================

def generate_segment_summaries(all_results):
    """Generate per-trace segment summaries."""
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    summary = df.groupby("trace_id").agg(
        n_stations=("distance_along", "count"),
        trace_length_m=("distance_along", "max"),
        mean_primary_throw=("primary_throw", "mean"),
        max_primary_throw=("primary_throw", "max"),
        mean_total_cumulative_throw=("total_cumulative_throw", "mean"),
        mean_HW_slope_deg=("HW_regional_slope_deg", "mean"),
        mean_FW_slope_deg=("FW_regional_slope_deg", "mean"),
        mean_strike=("strike", "mean"),
        mean_dip=("dip", "mean"),
        mean_scarp_width_m=("scarp_width_m", "mean"),
        mean_scarp_slope_deg=("scarp_slope_deg", "mean"),
        mean_profile_quality=("profile_quality", "mean"),
        mean_local_relief=("local_relief", "mean"),
    ).reset_index()

    return summary


# ===================================================================
# 12b.  PROFILE CACHE (for decoupled plot regeneration)
# ===================================================================

def save_profiles_cache(trace_id, profiles_cache, output_dir):
    """Save profile arrays to .npz so plots can be regenerated without reprocessing."""
    cache_dir = output_dir / "profiles_cache"
    safe_makedirs(cache_dir)
    n = len(profiles_cache)
    if n == 0:
        return
    # All profiles share the same distance array
    d_arr = profiles_cache[0][0]
    z_matrix = np.full((n, len(d_arr)), np.nan, dtype=np.float32)
    for i, (d, z) in enumerate(profiles_cache):
        z_matrix[i] = z
    path = cache_dir / f"trace_{trace_id}.npz"
    np.savez_compressed(path, distances=d_arr, elevations=z_matrix)
    print(f"  [Cache] Saved {n} profiles to {path.name}")


def load_profiles_cache(trace_id, output_dir):
    """Load cached profiles for a trace. Returns list of (distances, elevations)."""
    cache_dir = output_dir / "profiles_cache"
    path = cache_dir / f"trace_{trace_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    d_arr = data["distances"]
    z_matrix = data["elevations"]
    return [(d_arr.copy(), z_matrix[i].copy()) for i in range(z_matrix.shape[0])]


def _append_to_csv(results, csv_path):
    """Append rows to CSV (create header if new)."""
    if not results:
        return
    df = pd.DataFrame(results)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ===================================================================
# 13.  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-source", default=str(DEM_PATH),
                        help="Path to DEM GeoTIFF (default: lidar_0_5m_epsg28355_2.0m_smoke)")
    parser.add_argument("--max-traces", type=int, default=None,
                        help="Limit number of traces to process (for testing)")
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Limit stations per trace (for testing)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (CSV only)")
    parser.add_argument("--skip-3d", action="store_true",
                        help="Skip 3D visualisation generation")
    parser.add_argument("--start-trace", type=int, default=0,
                        help="Start from trace index N (0-based, for batch runs)")
    parser.add_argument("--end-trace", type=int, default=None,
                        help="Stop before trace index N (0-based, exclusive)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip traces already in parameter_database.csv")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip computation, regenerate plots from saved CSV + cache")
    args = parser.parse_args()

    print("=" * 70)
    print("06b-ScarpParameterProfiling.py")
    print("Detailed scarp parameter profiling at 10 m intervals")
    print("=" * 70)

    safe_makedirs(OUTPUT_DIR)
    param_path = OUTPUT_DIR / "parameter_database.csv"
    summary_path = OUTPUT_DIR / "scarp_segment_summaries.csv"

    # ------------------------------------------------------------------
    # PLOTS-ONLY MODE: regenerate all visualisations from saved data
    # ------------------------------------------------------------------
    if args.plots_only:
        if not param_path.exists():
            print("[Error] parameter_database.csv not found. Run computation first.")
            sys.exit(1)

        print(f"\n[Plots-only] Loading {param_path.name}...")
        param_df = pd.read_csv(param_path)

        # Load DEM (needed for 3D)
        dem_path = Path(args.dem_source)
        print(f"[DEM] Loading {dem_path.name}...")
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(np.float32)
            transform = src.transform
            dem_shape = (src.height, src.width)
            nodata = src.nodata if src.nodata is not None else -9999.0
        dem_data[dem_data == nodata] = np.nan

        # Load dem10m for patching lidar nodata gaps in 3D overviews
        dem10m_data = None
        if DEM10M_PATCH_PATH.exists():
            print(f"[DEM] Loading patch DEM: {DEM10M_PATCH_PATH.name}...")
            with rasterio.open(DEM10M_PATCH_PATH) as src10:
                dem10m_data = src10.read(1).astype(np.float32)
                nd10 = src10.nodata if src10.nodata is not None else -9999.0
            dem10m_data[dem10m_data == nd10] = np.nan
        else:
            print(f"[DEM] Patch DEM not found: {DEM10M_PATCH_PATH.name}")

        traces = load_traces(HIGHCON_DIR)
        trace_dict = {t["trace_id"]: t for t in traces}

        print("\n[Visualisations] Regenerating plots...")
        segments_by_file = {}
        for trace_id, group in param_df.groupby("trace_id"):
            results = group.to_dict("records")
            profiles = load_profiles_cache(trace_id, OUTPUT_DIR)

            if profiles is None:
                print(f"  Trace {trace_id}: no cached profiles, skipping cross-sections/3D")
                profiles = []

            trace = trace_dict.get(trace_id, {"coords": np.array([]), "trace_id": trace_id,
                                              "length_m": 0, "source_idx": 0})

            print(f"\n  Trace {trace_id} ({len(results)} stations):")

            if profiles:
                plot_cross_sections(results, profiles, trace_id, OUTPUT_DIR)

            plot_along_scarp_profiles(results, trace_id, OUTPUT_DIR)

            if not args.skip_3d and profiles:
                plot_3d_visualizations(
                    results, profiles, trace, dem_data,
                    transform, dem_shape, OUTPUT_DIR,
                )

            file_name = trace.get("file_name", f"trace_{trace_id}")
            segments_by_file.setdefault(file_name, []).append(
                {"trace_id": trace_id, "trace": trace, "results": results}
            )

        # Add fragments (no parameter rows in param_df) to segments_by_file
        # so they appear in the per-original-trace overview plots.
        for tid, tr in trace_dict.items():
            if tr.get("trace_type") == "fragment" and tid not in param_df["trace_id"].values:
                file_name = tr.get("file_name", f"trace_{tid}")
                segments_by_file.setdefault(file_name, []).append(
                    {"trace_id": tid, "trace": tr, "results": []}
                )

        if not args.skip_3d:
            print("\n[Visualisations] Regenerating per-original-trace overviews...")
            for file_name, segments in segments_by_file.items():
                plot_trace_overview_3d(
                    file_name, segments, dem_data, transform, dem_shape, OUTPUT_DIR,
                    dem10m_data=dem10m_data,
                )

        # Per-trace parameter profiles and elevation-easting dip figures
        for file_name, segments in segments_by_file.items():
            folded_dip_data = estimate_folded_fault_dip(
                segments, dem_data, transform, dem_shape,
                dem10m_data=dem10m_data,
            )
            plot_per_trace_parameter_profiles(
                file_name, segments, dem_data, transform, dem_shape,
                OUTPUT_DIR, dem10m_data=dem10m_data,
                folded_dip_data=folded_dip_data,
            )
            plot_elevation_easting_dip(
                file_name, segments, dem_data, transform, dem_shape,
                OUTPUT_DIR, dem10m_data=dem10m_data,
                dem_hr_path=DEM_HR_PATH,
            )

        print("\n" + "=" * 70)
        print("DONE (plots-only mode)")
        print(f"  Output directory: {OUTPUT_DIR}")
        print("=" * 70)
        return

    # ------------------------------------------------------------------
    # COMPUTE MODE
    # ------------------------------------------------------------------
    dem_path = Path(args.dem_source)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    print(f"\n[DEM] Loading {dem_path.name}...")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        transform = src.transform
        dem_shape = (src.height, src.width)
        nodata = src.nodata if src.nodata is not None else -9999.0
        dem_crs = src.crs
    # Replace nodata with NaN so interpolation doesn't create ghost values
    dem_data[dem_data == nodata] = np.nan
    print(f"  Shape: {dem_shape[0]} x {dem_shape[1]}, "
          f"CRS: {dem_crs}, nodata: {nodata}")

    # Load dem10m for patching lidar nodata gaps in 3D overviews
    dem10m_data = None
    if DEM10M_PATCH_PATH.exists():
        print(f"[DEM] Loading patch DEM: {DEM10M_PATCH_PATH.name}...")
        with rasterio.open(DEM10M_PATCH_PATH) as src10:
            dem10m_data = src10.read(1).astype(np.float32)
            nd10 = src10.nodata if src10.nodata is not None else -9999.0
        dem10m_data[dem10m_data == nd10] = np.nan
    else:
        print(f"[DEM] Patch DEM not found: {DEM10M_PATCH_PATH.name}")

    # Load traces
    traces = load_traces(HIGHCON_DIR)

    # Apply start/end/max slicing
    start_idx = args.start_trace
    end_idx = args.end_trace if args.end_trace is not None else len(traces)
    if args.max_traces:
        end_idx = min(end_idx, start_idx + args.max_traces)
    traces = traces[start_idx:end_idx]
    print(f"\n[Traces] Processing traces [{start_idx}:{end_idx}] "
          f"({len(traces)} traces)")

    print(f"\n[Output] {OUTPUT_DIR}")

    # Resume: find which trace_ids are already in the CSV
    done_trace_ids = set()
    if args.resume and param_path.exists():
        existing_df = pd.read_csv(param_path, usecols=["trace_id"])
        done_trace_ids = set(existing_df["trace_id"].unique())
        print(f"[Resume] {len(done_trace_ids)} traces already in database, "
              f"will skip them")

    # If not resuming and starting fresh, remove old CSV
    if not args.resume and start_idx == 0 and param_path.exists():
        param_path.unlink()
        print("[Database] Cleared old parameter_database.csv")

    # Process each trace
    all_results = []
    all_profiles = {}
    batch_count = 0

    for trace in traces:
        trace_id = trace["trace_id"]

        # Fragments are too short for parameter profiling; store them with
        # empty results so they can still appear in overview plots.
        if trace.get("trace_type") == "fragment":
            print(f"\n[Trace {trace_id}] Fragment (short trace) - skipping computation, "
                  f"will show in overview plots")
            all_profiles[trace_id] = {
                "results": [],
                "profiles": [],
                "trace": trace,
            }
            continue

        if trace_id in done_trace_ids:
            print(f"\n[Trace {trace_id}] Skipping (already in database)")
            continue

        results, profiles_cache = process_trace(
            trace, dem_data, transform, dem_shape, nodata,
            station_spacing=STATION_SPACING_M,
        )

        if args.max_stations:
            results = results[:args.max_stations]
            profiles_cache = profiles_cache[:args.max_stations]

        all_results.extend(results)
        all_profiles[trace_id] = {
            "results": results,
            "profiles": profiles_cache,
            "trace": trace,
        }

        # Progress
        n_valid = sum(1 for r in results if np.isfinite(r["primary_throw"]))
        print(f"  -> {n_valid}/{len(results)} stations with valid scarp parameters")

        # Save profiles cache for this trace (enables plots-only mode)
        save_profiles_cache(trace_id, profiles_cache, OUTPUT_DIR)

        # Append results to CSV immediately (crash-safe)
        _append_to_csv(results, param_path)
        batch_count += 1

        # Save summaries periodically (every 10 traces)
        if batch_count % 10 == 0:
            print(f"\n  [Checkpoint] Saved {batch_count} traces so far...")
            # Regenerate summaries from full CSV
            if param_path.exists():
                full_df = pd.read_csv(param_path)
                summary_df = generate_segment_summaries(full_df.to_dict("records"))
                summary_df.to_csv(summary_path, index=False)

    # Final save: summaries
    print(f"\n[Database] Appending to {param_path}")
    if param_path.exists():
        full_df = pd.read_csv(param_path)
        all_results = full_df.to_dict("records")

    print(f"[Database] {len(all_results)} total rows in {param_path}")

    # Save segment summaries
    summary_df = generate_segment_summaries(all_results)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Summary] {len(summary_df)} segments saved to {summary_path}")

    # Visualisations
    if not args.no_plots:
        print("\n[Visualisations] Generating plots...")
        for trace_id, data in all_profiles.items():
            results = data["results"]
            profiles = data["profiles"]
            trace = data["trace"]

            # Fragments have no profiling results; skip per-trace plots.
            if trace.get("trace_type") == "fragment":
                print(f"\n  Trace {trace_id} [fragment]: skipping per-trace plots")
                continue

            print(f"\n  Trace {trace_id} ({len(results)} stations):")

            # Cross-sections
            plot_cross_sections(results, profiles, trace_id, OUTPUT_DIR)

            # Along-scarp profiles
            plot_along_scarp_profiles(results, trace_id, OUTPUT_DIR)

            # 3D visualisations
            if not args.skip_3d:
                plot_3d_visualizations(
                    results, profiles, trace, dem_data,
                    transform, dem_shape, OUTPUT_DIR,
                )

        # Per-original-trace (Trace1..N) corridor overviews, additive to the
        # per-segment chunked 3D plots above.  Fragments are included here so
        # they appear in the by-original-trace overview.
        segments_by_file = {}
        for trace_id, data in all_profiles.items():
            file_name = data["trace"].get("file_name", f"trace_{trace_id}")
            segments_by_file.setdefault(file_name, []).append(
                {"trace_id": trace_id, "trace": data["trace"], "results": data["results"]}
            )

        if not args.skip_3d:
            print("\n[Visualisations] Generating per-original-trace overviews...")
            for file_name, segments in segments_by_file.items():
                plot_trace_overview_3d(
                    file_name, segments, dem_data, transform, dem_shape, OUTPUT_DIR,
                    dem10m_data=dem10m_data,
                )

        # Per-trace parameter profiles and elevation-easting dip figures
        for file_name, segments in segments_by_file.items():
            folded_dip_data = estimate_folded_fault_dip(
                segments, dem_data, transform, dem_shape,
                dem10m_data=dem10m_data,
            )
            plot_per_trace_parameter_profiles(
                file_name, segments, dem_data, transform, dem_shape,
                OUTPUT_DIR, dem10m_data=dem10m_data,
                folded_dip_data=folded_dip_data,
            )
            plot_elevation_easting_dip(
                file_name, segments, dem_data, transform, dem_shape,
                OUTPUT_DIR, dem10m_data=dem10m_data,
                dem_hr_path=DEM_HR_PATH,
            )

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Parameter database: {param_path}")
    print(f"  Segment summaries:  {summary_path}")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
