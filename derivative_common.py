#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
derivative_common.py -- Shared utilities, constants, and core functions
for DEM derivative generation and tile sampling.

This module is imported by both:
  - 01a-DerivativeGen.py  (derivative generation)
  - 01b-TileSampling.py   (tile sampling & visualisation)
"""


import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
import tempfile
import shutil
from rasterio.crs import CRS
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from rasterio.transform import from_bounds, array_bounds, Affine
from rasterio.windows import Window
from rasterio.enums import Resampling as ResampEnum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    generic_filter,
    label,
    find_objects,
    sobel as scipy_sobel,
)
from scipy.signal import convolve2d, fftconvolve
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.filters import sobel as sk_sobel
from skimage.feature import canny as sk_canny
from sklearn.decomposition import PCA as SkPCA
from pyproj import Transformer
import psutil
from tqdm import tqdm
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# RESOURCE DETECTION
# ===================================================================



import subprocess
import re

class SystemResources:
    """Detect and report available system resources; adapt processing config.
    Enhanced with multi-GPU support for RTX 2000 Ada.
    """

    def __init__(self, gpu_id=None):
        """
        Args:
            gpu_id: GPU device ID to use.  None = auto-detect (pick first
                    available device).  If the requested ID does not exist,
                    falls back to device 0.
        """
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.available_ram_gb = psutil.virtual_memory().available / (1024**3)
        self.cpu_count = psutil.cpu_count(logical=False) or 1
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_vram_gb = 0.0
        self._cp = None

        # -- Auto-detect the correct GPU device --
        resolved_gpu_id = self._resolve_gpu_id(gpu_id)
        self.gpu_id = resolved_gpu_id

        # Try CuPy with resolved GPU
        try:
            import cupy as cp
            # Set the GPU device to use
            cp.cuda.Device(resolved_gpu_id).use()
            
            # Get GPU info
            device = cp.cuda.Device(resolved_gpu_id)

            # Get GPU properties
            props = cp.cuda.runtime.getDeviceProperties(resolved_gpu_id)

            # GPU name
            self.gpu_name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else props["name"]

            # Total VRAM (bytes -> GB)
            self.gpu_vram_gb = device.mem_info[1] / (1024 ** 3)

            # Smoke-test the backend before trusting it. Some CUDA/CuPy
            # combinations fail at kernel compile time even though import
            # succeeds, which would otherwise crash later in the derivatives.
            if self._probe_cupy_backend(cp):
                self.gpu_available = True
                self._cp = cp
                print(
                    f"  [GPU] Using {self.gpu_name} "
                    f"(ID: {resolved_gpu_id}) with {self.gpu_vram_gb:.1f} GB VRAM"
                )
            else:
                print(f"  [GPU] CuPy backend failed validation; falling back to CPU")
                self.gpu_name = None
                self.gpu_vram_gb = 0.0
                self._cp = None
            
        except Exception as e:
            print(f"  [GPU] CuPy initialization failed: {e}")
            print(f"  [GPU] Falling back to CPU mode")
            self._cp = None

        # Try PyTorch as fallback for GPU
        if not self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    n_torch = torch.cuda.device_count()
                    safe_id = min(resolved_gpu_id, n_torch - 1) if n_torch > 0 else 0
                    torch.cuda.set_device(safe_id)
                    self.gpu_name = torch.cuda.get_device_name(safe_id)
                    self.gpu_vram_gb = torch.cuda.get_device_properties(safe_id).total_memory / (1024**3)
                    self.gpu_available = True
                    self.gpu_id = safe_id
                    self._backend = "torch"
                    print(f"  [GPU] Using PyTorch with {self.gpu_name} (ID: {safe_id})")
            except Exception as e:
                print(f"  [GPU] PyTorch initialization failed: {e}")

        # Adaptive config
        self.n_workers = min(self.cpu_count, 8)
        self.use_gpu = self.gpu_available
        self.tile_size = self._adapt_tile_size()
        self.max_array_gb = self.available_ram_gb * 0.6

    @staticmethod
    def _resolve_gpu_id(gpu_id):
        """Auto-detect the first valid GPU device ID.

        If *gpu_id* is None, enumerate all CUDA devices and return 0
        (the first one).  If an explicit *gpu_id* is given but is out
        of range, fall back to device 0 with a warning.
        """
        try:
            import cupy as cp
            n_devices = cp.cuda.runtime.getDeviceCount()
            if n_devices == 0:
                return 0  # no GPU; CuPy init will fail gracefully later
            if gpu_id is None:
                return 0
            if gpu_id >= n_devices:
                print(f"  [GPU] Requested gpu_id={gpu_id} but only {n_devices} device(s) found; "
                      f"falling back to device 0")
                return 0
            return gpu_id
        except Exception:
            # CuPy not installed or CUDA not available – just return 0
            if gpu_id is not None:
                return gpu_id
            return 0

    def _probe_cupy_backend(self, cp):
        """Validate that CuPy can execute a simple reduction on this machine."""
        try:
            arr = cp.asarray(np.zeros((2, 2), dtype=np.float32))
            cp.any(arr)
            return True
        except Exception as e:
            print(f"  [GPU] CuPy smoke test failed: {e}")
            return False

    def _adapt_tile_size(self):
        """Choose tile size based on available VRAM and RAM."""
        if self.gpu_available and self.gpu_vram_gb > 0:
            # Use GPU VRAM as primary constraint
            # Each tile needs: DEM + slope + curvature + SVF (4 arrays) 
            # ~4 bytes per pixel * 4 arrays = 16 bytes per pixel
            vram_budget_gb = self.gpu_vram_gb * 0.7  # Use 70% of VRAM
            max_px_vram = int(np.sqrt(vram_budget_gb * (1024**3) / (4 * 4)))  # 4 arrays of float32
        else:
            max_px_vram = float('inf')

        # RAM constraint (same as before)
        budget_per_worker_gb = (self.available_ram_gb * 0.5) / max(self.n_workers, 1)
        max_px_ram = int(np.sqrt(budget_per_worker_gb * (1024**3) / (4 * 3)))

        max_px = min(max_px_vram, max_px_ram, 8000)
        max_px = max(max_px, 2000)
        # Round to nearest 500
        max_px = (max_px // 500) * 500
        
        print(f"  [Config] Tile size: {max_px} px (VRAM: {max_px_vram}, RAM: {max_px_ram})")
        return max_px

    def xp(self):
        """Return cupy if GPU available, else numpy."""
        if self.use_gpu and self._cp is not None:
            return self._cp
        return np

    def to_numpy(self, arr):
        """Ensure array is numpy (move from GPU if needed)."""
        if self.use_gpu and self._cp is not None:
            if isinstance(arr, self._cp.ndarray):
                return self._cp.asnumpy(arr)
        return np.asarray(arr)

    def to_device(self, arr):
        """Move array to GPU if available."""
        if self.use_gpu and self._cp is not None:
            return self._cp.asarray(arr)
        return arr

    def report(self):
        """Print a resource summary."""
        print("=" * 60)
        print("SYSTEM RESOURCES")
        print("=" * 60)
        print(f"  Total RAM:       {self.total_ram_gb:.1f} GB")
        print(f"  Available RAM:   {self.available_ram_gb:.1f} GB")
        print(f"  CPU cores:       {self.cpu_count} physical / {self.cpu_count_logical} logical")
        print(f"  GPU:             {'Yes -- ' + self.gpu_name if self.gpu_available else 'No'}")
        if self.gpu_available:
            print(f"  GPU VRAM:        {self.gpu_vram_gb:.1f} GB")
            print(f"  GPU ID:          {self.gpu_id}")
        print("-" * 60)
        print(f"  Workers:         {self.n_workers}")
        print(f"  GPU accel:       {'Yes' if self.use_gpu else 'No'}")
        print(f"  Tile size:       {self.tile_size} px")
        print(f"  Max array:       {self.max_array_gb:.1f} GB")
        print("=" * 60)


# ===================================================================
# GLOBAL FLAGS FOR MULTIPROCESSING
# ===================================================================

# Flag to prevent resource detection in worker processes
_IN_WORKER_PROCESS = False

def set_worker_mode():
    """Mark the current process as a worker (suppresses resource detection)."""
    global _IN_WORKER_PROCESS
    _IN_WORKER_PROCESS = True


# Global resource instance (lazy)
_res = None

def get_resources(gpu_id=None):
    """Get or create the global SystemResources instance.
    
    Args:
        gpu_id: GPU device ID to use. None = auto-detect (default).
                If the requested ID is invalid, SystemResources will
                fall back to device 0 automatically.
    """
    global _res, _IN_WORKER_PROCESS
    
    # In worker processes, return a minimal config without printing
    if _IN_WORKER_PROCESS:
        # Return a minimal resource object for workers
        class MinimalResources:
            def __init__(self):
                self.total_ram_gb = 0
                self.available_ram_gb = 0
                self.cpu_count = 1
                self.cpu_count_logical = 1
                self.gpu_available = False
                self.gpu_name = None
                self.gpu_vram_gb = 0.0
                self.n_workers = 1
                self.use_gpu = False
                self.tile_size = 4000
                self.max_array_gb = 0
                self._cp = None
            
            def xp(self):
                return np
            
            def to_numpy(self, arr):
                return np.asarray(arr)
            
            def to_device(self, arr):
                return arr
            
            def report(self):
                pass
        
        return MinimalResources()
    
    if _res is None:
        _res = SystemResources(gpu_id=gpu_id)
        _res.report()
    return _res


# ===================================================================
# CONFIGURATION
# ===================================================================


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUT_DIR  = DATA_DIR / "derivatives"

# Input DEM paths
SRTM_PATH   = DATA_DIR / "srtm1s_axe.tif"
SRTM3S_PATH = DATA_DIR / "srtm3s_axe.tif"
DEM10M_PATH = DATA_DIR / "Dem10m_axe.tif"
LIDAR_PATH  = DATA_DIR / "axedale_2020nov01_dem50cm_v10cm_epsg7855.tif"

# Structural points
STRUCT_PTS_PATH = DATA_DIR / "structural_points_inventory.csv"

# Target CRS -- GDA94 / MGA zone 55
TARGET_CRS = CRS.from_epsg(28355)

# LiDAR nodata value (from the user)
LIDAR_NODATA = -9999.0

# Hillshade parameters
HS_AZIMUTHS   = [315, 45, 135, 225]
HS_ALTITUDE   = 8       # degrees -- low angle for scarp contrast

# TPI radii (in pixels, will be converted based on resolution)
TPI_RADII_M = [50, 200, 1000]  # metres

# Tile sampling
TILE_SIZE_M       = 2000     # 2 km x 2 km
N_TILES           = 10
MIN_VALIDITY      = 0.95     # at least 95 % valid pixels
TILE_DPI          = 300
TILE_HALF_LABEL_M = 100      # +/-100 m axis labels (or 200 m -- chosen per DEM res)

# Block size for memory-safe raster I/O
BLOCK_SIZE = 4096

# Reprojection version -- increment when algorithm changes to force regeneration
REPROJECT_VERSION = 3

# -- Phase 1: Scarp Profile Parameters --
PROFILE_HALF_LENGTH_M = 500   # half-length of perpendicular profile (m)
PROFILE_STEP_M        = None  # sampling step (None = native resolution)
SCARP_MIN_HEIGHT_M    = 0.5   # minimum scarp height to flag (m)
SCARP_MAX_WIDTH_M     = 200   # maximum scarp width for discrete rupture (m)
PROFILE_SPACING_M     = 500   # spacing between sample points along fault (m)

# -- Phase 1: Expression Quality Thresholds --
EXPR_CLASS1_SLOPE     = 5.0   # deg -- strong gradient break
EXPR_CLASS2_SLOPE     = 2.0   # deg -- moderate gradient break
EXPR_CLASS1_SVFDIFF   = 0.15  # SVF anomaly threshold (strong)
EXPR_CLASS2_SVFDIFF   = 0.08  # SVF anomaly threshold (moderate)
EXPR_MIN_PROFILES     = 3     # minimum profiles needed for classification
EXPR_POINT_DRAIN_BUFFER_M = 1000.0  # drainage corroboration radius per sample point (m)

# -- Phase 1: Drainage Anomaly --
DRAIN_CURV_THRESHOLD  = 0.005 # curvature threshold for linear valley detection
DRAIN_SLOPE_MIN       = 1.0   # min slope along drainage anomaly (deg)
DRAIN_MIN_LENGTH_M    = 200   # minimum length of anomaly (m)

# -- LiDAR tiled processing --
LIDAR_TILE_SIZE = 4000        # pixels per tile for LiDAR derivatives (overridden by SystemResources)
LIDAR_TILE_OVERLAP = 50       # overlap in pixels for edge effects
LIDAR_OVERVIEW_RES = 50.0     # resolution for overview PNG (metres)


# ===================================================================
# TIMING HELPERS
# ===================================================================


class Timer:
    """Context manager for timing blocks; prints ETA for iterative work."""

    def __init__(self, label, total_items=None):
        self.label = label
        self.total_items = total_items
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        print(f"  [Timer] {self.label} ...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.elapsed > 60:
            m, s = divmod(self.elapsed, 60)
            print(f"  [Timer] {self.label}: {int(m)}m {s:.1f}s")
        else:
            print(f"  [Timer] {self.label}: {self.elapsed:.1f}s")

    def eta(self, completed, total=None):
        """Return estimated time remaining as a string."""
        if total is None:
            total = self.total_items
        if completed <= 0 or total is None:
            return "N/A"
        elapsed = time.time() - self.start
        rate = elapsed / completed
        remaining = rate * (total - completed)
        if remaining > 3600:
            return f"{remaining/3600:.1f}h"
        elif remaining > 60:
            return f"{remaining/60:.1f}m"
        else:
            return f"{remaining:.0f}s"


# ===================================================================
# HELPERS
# ===================================================================

def mem_gb():
    """Return available RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def check_nodata(path):
    """Open a raster, print its nodata, and return (nodata_value, crs, shape, resolution)."""
    with rasterio.open(path) as src:
        nd = src.nodata
        crs = src.crs
        shape = (src.height, src.width)
        res = src.res
    print(f"  {path.name}: nodata={nd}, CRS={crs}, shape={shape}, res={res}")
    return nd, crs, shape, res


def safe_makedirs(d):
    """Create directory tree if it doesn't exist."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def infer_nodata_value(path):
    """Return a float nodata value for the raster -- handles known special values."""
    with rasterio.open(path) as src:
        nd = src.nodata
        if nd is not None:
            return float(nd)
        # No nodata declared -- scan a small window for common sentinel values
        h, w = min(256, src.height), min(256, src.width)
        sample = src.read(1, window=Window(0, 0, w, h))
        # Check for very large negative values that are likely nodata
        if np.issubdtype(sample.dtype, np.floating):
            minval = np.nanmin(sample)
            if minval < -1e6:
                return float(minval)
        return None


def _cleanup_old_tif(out_dir, dem_stem, deriv_name):
    """Remove old .tif derivative files that have been replaced by .npy."""
    old_tif = out_dir / f"{dem_stem}_{deriv_name}.tif"
    if old_tif.exists():
        print(f"  [Cleanup] Removing old .tif: {old_tif.name}")
        os.remove(old_tif)


def _save_npy_derivative(data, out_dir, dem_stem, deriv_name, transform, crs_str,
                          nodata, resolution):
    """Save a derivative array as .npy + metadata .json, and clean up old .tif."""
    safe_makedirs(out_dir)
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    json_path = out_dir / f"{dem_stem}_{deriv_name}.json"

    np.save(npy_path, data.astype(np.float32))

    # Metadata JSON: transform stored as [a, b, c, d, e, f]
    t = transform
    meta = {
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "crs": crs_str,
        "shape": list(data.shape),
        "nodata": nodata,
        "resolution": resolution,
        "type": deriv_name,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Remove old .tif if it exists
    _cleanup_old_tif(out_dir, dem_stem, deriv_name)

    return npy_path


def _load_npy_derivative(out_dir, dem_stem, deriv_name):
    """Load a .npy derivative with its metadata JSON.

    Returns (data, transform, crs_str, nodata, resolution) or (None, ...) if not found.
    """
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    json_path = out_dir / f"{dem_stem}_{deriv_name}.json"

    if not npy_path.exists():
        return None, None, None, None, None

    data = np.load(npy_path)
    if not json_path.exists():
        return data, None, None, None, None

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    t = meta.get("transform", [1, 0, 0, 0, -1, 0])
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    return data, transform, meta.get("crs", "EPSG:28355"), meta.get("nodata", np.nan), meta.get("resolution", 1.0)


# Add this configuration at the top of the file
KEEP_DERIVATIVES = {
    # Key: derivative name, Value: whether to keep full-resolution .npy
    "hillshade": True,       # Needed for visualisation
    "slope": True,           # Needed for drainage and classification
    "curv_profile": True,    # Needed for drainage anomalies
    "curv_plan": True,       # Kept for visualisation
    "svf": True,             # Needed for classification
    "openness_pos": False,   # Not used in downstream processing
    "openness_neg": False,   # Not used in downstream processing
    "tpi": False,            # Not used in downstream processing
    "pca": False,            # Not used in downstream processing
}

# Derivative types that can be safely deleted after phase 1 audit
TRANSIENT_DERIVATIVES = ["openness_pos", "openness_neg", "tpi_*", "pca_*"]


def cleanup_transient_derivatives(out_dir, dem_key, keep_list=None):
    """
    Delete derivative .npy files that are not needed for downstream processing.
    
    This is the key to reducing storage from 700GB+ to a manageable size.
    """
    if keep_list is None:
        keep_list = ["hillshade", "slope", "curv_profile", "curv_plan", "svf"]
    
    deriv_dir = out_dir / dem_key
    if not deriv_dir.exists():
        return
    
    deleted_count = 0
    deleted_size = 0
    
    for f in deriv_dir.glob("*.npy"):
        # Extract derivative name from filename
        # The full stem is e.g. "dem10m_epsg28355_clipped_hillshade_az315"
        # We need to strip the DEM-key prefix AND any CRS/clipped suffix
        # to get just "hillshade_az315" for keep-pattern matching.
        name = f.stem
        # Remove dem_key prefix if present
        if name.startswith(f"{dem_key}_"):
            name = name[len(dem_key) + 1:]
        
        # Also strip common CRS/clipped suffixes: "epsg28355_" or "epsg28355_clipped_"
        # These are inserted between the dem_key and the derivative name
        for strip_prefix in ("epsg28355_clipped_", "epsg28355_"):
            if name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
                break
        
        # Check if this derivative should be kept
        should_keep = False
        for keep_pattern in keep_list:
            if name.startswith(keep_pattern) or name == keep_pattern:
                should_keep = True
                break
            # Handle wildcard patterns like tpi_*
            if keep_pattern.endswith("*") and name.startswith(keep_pattern[:-1]):
                should_keep = True
                break
        
        if not should_keep:
            size_mb = f.stat().st_size / (1024 * 1024)
            deleted_size += size_mb
            f.unlink()
            deleted_count += 1
            
            # Also delete associated JSON metadata
            json_path = f.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
    
    if deleted_count > 0:
        print(f"  [Cleanup] Removed {deleted_count} transient files ({deleted_size:.1f} MB) from {dem_key}")


def cleanup_phase1_artifacts(audit_dir, keep_analysis=True):
    """
    Clean up intermediate files created during phase 1 audit.
    
    Args:
        audit_dir: Path to phase1_audit directory
        keep_analysis: If True, keep CSV files (scarp_analysis, drainage_anomalies, expression_quality)
    """
    if not audit_dir.exists():
        return
    
    deleted_count = 0
    deleted_size = 0
    
    # Remove profile PNGs (can be regenerated)
    profiles_dir = audit_dir / "*_profiles"
    for d in audit_dir.glob("*_profiles"):
        if d.is_dir():
            for f in d.glob("*.png"):
                size_mb = f.stat().st_size / (1024 * 1024)
                deleted_size += size_mb
                f.unlink()
                deleted_count += 1
            # Remove empty directory
            try:
                d.rmdir()
            except OSError:
                pass
    
    # Remove intermediate overlay maps (can be regenerated)
    # NOTE: *_expression_quality_map.png is kept because it is now a key
    #       A/B/C/D scatter output, not a disposable intermediate.
    for pattern in ["*_fault_overlay.png"]:
        for f in audit_dir.glob(pattern):
            size_mb = f.stat().st_size / (1024 * 1024)
            deleted_size += size_mb
            f.unlink()
            deleted_count += 1
    
    print(f"  [Cleanup] Removed {deleted_count} phase1 artifacts ({deleted_size:.1f} MB)")


def _save_derivative_png(data, transform, png_path, cmap="viridis", vmin=None, vmax=None,
                          title=None, downsample_max=2000):
    """Save a 300 dpi PNG with easting/northing axes and colorbar.

    For very large arrays, downsample before plotting to keep PNG size reasonable.
    Uses pcolormesh with correct geographic extent.
    """
    rows, cols = data.shape

    # Downsample if needed
    if rows > downsample_max or cols > downsample_max:
        ds = max(int(np.ceil(rows / downsample_max)), int(np.ceil(cols / downsample_max)))
        data = data[::ds, ::ds]
        rows, cols = data.shape

    # Compute cell edge coordinates from transform
    x_min = transform.c
    x_max = transform.c + cols * transform.a
    # For north-up raster, transform.e is negative
    y_top = transform.f
    y_bottom = transform.f + rows * transform.e

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_top, y_bottom, rows + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="flat")
    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

def _save_derivative_png_adaptive(data, transform, png_path, cmap="viridis", 
                                   vmin=None, vmax=None, title=None, 
                                   downsample_max=2000, use_percentile_clip=True,
                                   clip_low=2, clip_high=98):
    """
    Save a 300 dpi PNG with adaptive colormap limits based on data distribution.
    
    Args:
        use_percentile_clip: If True, clip colormap to percentile range
        clip_low: Lower percentile (default 2)
        clip_high: Upper percentile (default 98)
    """
    rows, cols = data.shape

    # Downsample if needed
    if rows > downsample_max or cols > downsample_max:
        ds = max(int(np.ceil(rows / downsample_max)), int(np.ceil(cols / downsample_max)))
        data = data[::ds, ::ds]
        rows, cols = data.shape

    # Compute cell edge coordinates from transform
    x_min = transform.c
    x_max = transform.c + cols * transform.a
    y_top = transform.f
    y_bottom = transform.f + rows * transform.e

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_top, y_bottom, rows + 1)

    # --- NEW: Adaptive colormap limits based on histogram ---
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        if use_percentile_clip:
            vmin = np.percentile(valid_data, clip_low)
            vmax = np.percentile(valid_data, clip_high)
            # For symmetric data (like curvature), ensure symmetric limits
            if cmap in ["RdBu_r", "coolwarm"]:
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max
        else:
            # Auto-scale using standard deviation
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            if vmin is None:
                vmin = mean_val - 3 * std_val
            if vmax is None:
                vmax = mean_val + 3 * std_val
    
    # Save histogram for debugging
    hist_path = png_path.parent / f"{png_path.stem}_histogram.png"
    _save_histogram(valid_data, hist_path, title)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="flat")
    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_aspect('equal')
    if title:
        ax.set_title(f"{title}  |  Range: [{vmin:.3f}, {vmax:.3f}]", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _save_histogram(data, hist_path, title=None):
    """Save histogram of data values for diagnostic purposes."""
    if len(data) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    if title:
        ax.set_title(f"Histogram: {title}")
    
    # Add percentile markers
    for p in [2, 50, 98]:
        val = np.percentile(data, p)
        ax.axvline(val, color='red', linestyle='--', alpha=0.5, 
                   label=f"{p}th percentile = {val:.3f}")
    ax.legend()
    fig.savefig(hist_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

# ===================================================================
# 1.  DEM REPROJECTION & ALIGNMENT
# ===================================================================

def reproject_dem(src_path, dst_path, target_crs=TARGET_CRS, target_res=None):
    """
    Reproject a DEM to target_crs and optionally resample.

    BUG FIX: The old block-wise approach passed the FULL dst_transform
    to every reproject() call, causing every block to receive the same
    data (top-left corner repeated). Now uses:
      - Full-array reproject for DEMs <= 2 GB uncompressed
      - Windowed reproject with correct per-window destination transform
        for DEMs > 2 GB

    Returns the output profile dict.
    """
    src_path = Path(src_path) if not isinstance(src_path, Path) else src_path
    dst_path = Path(dst_path) if not isinstance(dst_path, Path) else dst_path
    print(f"\n[Reproject] {src_path.name} -> {target_crs}")
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else -9999.0

        if target_res is None:
            # Keep approximate source resolution
            if src_crs.is_geographic:
                center_lat = (src.bounds.bottom + src.bounds.top) / 2
                m_per_deg = 111320 * np.cos(np.radians(center_lat))
                target_res = abs(src.res[0]) * m_per_deg
            else:
                target_res = abs(src.res[0])

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_crs,
            src.width, src.height,
            *src.bounds,
            resolution=target_res,
        )

        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": src_nodata,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

        print(f"  Source CRS={src_crs}  ->  Target CRS={target_crs}")
        print(f"  Target resolution: {target_res:.2f} m")
        print(f"  Output shape: {dst_height}x{dst_width}")

        # Estimate uncompressed output size
        uncompressed_gb = dst_height * dst_width * np.dtype(profile["dtype"]).itemsize / (1024**3)
        print(f"  Uncompressed output size: {uncompressed_gb:.2f} GB")

        # Use BigTIFF format for large files (> 4 GB)
        if uncompressed_gb > 4.0:
            profile["BIGTIFF"] = "YES"

        safe_makedirs(dst_path.parent)

        if uncompressed_gb <= 2.0:
            # -- Full-array reproject (safe for <= 2 GB) --
            print("  Using full-array reprojection (<= 2 GB)")
            dst_data = np.full((1, dst_height, dst_width), src_nodata, dtype=profile["dtype"])
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                dst_nodata=src_nodata,
                resampling=Resampling.bilinear,
            )
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(dst_data)
        else:
            # -- Windowed reproject with correct per-window transform (> 2 GB) --
            print("  Using windowed reprojection (> 2 GB)")
            tile_px = BLOCK_SIZE
            with rasterio.open(dst_path, "w", **profile) as dst:
                for row_off in tqdm(range(0, dst_height, tile_px), desc="  Reprojecting", leave=False):
                    for col_off in range(0, dst_width, tile_px):
                        win_h = min(tile_px, dst_height - row_off)
                        win_w = min(tile_px, dst_width - col_off)

                        out_win = Window(col_off, row_off, win_w, win_h)
                        dst_data = np.full((1, win_h, win_w), src_nodata, dtype=profile["dtype"])

                        # Compute the transform for this specific window
                        win_transform = dst_transform * Affine.translation(col_off, row_off)

                        reproject(
                            source=rasterio.band(src, 1),
                            destination=dst_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=src_nodata,
                            dst_transform=win_transform,
                            dst_crs=target_crs,
                            dst_nodata=src_nodata,
                            resampling=Resampling.bilinear,
                        )
                        dst.write(dst_data, window=out_win)

    print(f"  [OK] Saved: {dst_path}")
    return profile


def load_reprojected(dem_key, src_path, out_dir):
    """
    Ensure a DEM is reprojected to TARGET_CRS.
    Uses a version marker to force regeneration when the algorithm changes.
    Returns (reprojected_path, nodata_value).
    """
    rp_path = out_dir / f"{dem_key}_epsg28355.tif"
    version_path = out_dir / f"{dem_key}_epsg28355.version"

    # Check if reprojected file exists with correct version
    if rp_path.exists() and version_path.exists():
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                v = int(f.read().strip())
            if v >= REPROJECT_VERSION:
                print(f"  [Skip] {rp_path.name} already exists (v{v})")
                nd = infer_nodata_value(rp_path)
                return rp_path, nd
        except (ValueError, IOError):
            pass

    # Force regeneration -- remove old file if present
    if rp_path.exists():
        print(f"  [!] Removing outdated/corrupt {rp_path.name}")
        os.remove(rp_path)
    if version_path.exists():
        os.remove(version_path)

    # Determine target resolution based on DEM type
    # Check '3s' first so it does not fall into the generic 'srtm' branch
    if "3s" in dem_key.lower():
        tgt_res = 90.0
    elif "srtm" in dem_key.lower() or "1s" in dem_key.lower():
        tgt_res = 30.0
    elif "10m" in dem_key.lower():
        tgt_res = 10.0
    elif "50cm" in dem_key.lower() or "0_5m" in dem_key.lower():
        tgt_res = 0.5
    else:
        tgt_res = None

    reproject_dem(src_path, rp_path, target_res=tgt_res)

    # Write version marker
    with open(version_path, "w", encoding="utf-8") as f:
        f.write(str(REPROJECT_VERSION))

    nd = infer_nodata_value(rp_path)
    return rp_path, nd


def clip_to_common_extent(dems_to_process, out_dir, reference_key="lidar_0_5m"):
    """
    Clip all reprojected DEMs to a common extent aligned with the reference DEM.

    The reference DEM (by default LiDAR) defines the target grid: its origin,
    pixel size, and extent. All other DEMs are clipped to the intersection of
    their own extent with the reference extent, snapped to the reference grid.

    This ensures that derivative products from different resolutions share
    exactly the same geographic footprint, making visual comparison and
    overlay straightforward.

    Args:
        dems_to_process: dict {dem_key: {"path": ..., "nodata": ..., "label": ...}}
        out_dir:         output directory (Path)
        reference_key:   which DEM key to use as the reference extent (default: lidar_0_5m)

    Returns:
        Updated dems_to_process dict with paths pointing to the clipped files.
    """
    if reference_key not in dems_to_process:
        print(f"  [!] Reference DEM '{reference_key}' not found in dems_to_process, skipping clip")
        return dems_to_process

    # Read the reference DEM extent and grid
    ref_path = dems_to_process[reference_key]["path"]
    with rasterio.open(ref_path) as ref_src:
        ref_bounds = ref_src.bounds
        ref_res = abs(ref_src.res[0])
        ref_transform = ref_src.transform

    print(f"\n[Align] Clipping all DEMs to common extent (reference: {reference_key})")
    print(f"  Reference extent: E=[{ref_bounds.left:.1f}, {ref_bounds.right:.1f}], "
          f"N=[{ref_bounds.bottom:.1f}, {ref_bounds.top:.1f}]")
    print(f"  Reference resolution: {ref_res} m")

    for dem_key, info in dems_to_process.items():
        src_path = info["path"]

        # Output path for the clipped file
        clipped_path = out_dir / f"{dem_key}_epsg28355_clipped.tif"

        # ---- Resume check: skip if clipped file already exists with correct extent ----
        if clipped_path.exists():
            with rasterio.open(clipped_path) as clipped_src:
                clipped_bounds = clipped_src.bounds
                # Check if the clipped file's extent is within the reference extent
                # (allowing tiny floating-point tolerance)
                tol = 0.5  # half a metre tolerance
                if (clipped_bounds.left >= ref_bounds.left - tol and
                    clipped_bounds.right <= ref_bounds.right + tol and
                    clipped_bounds.bottom >= ref_bounds.bottom - tol and
                    clipped_bounds.top <= ref_bounds.top + tol):
                    print(f"  [Skip] {dem_key}: clipped file already exists with correct extent")
                    dems_to_process[dem_key]["path"] = clipped_path
                    dems_to_process[dem_key]["nodata"] = infer_nodata_value(clipped_path)
                    continue
                else:
                    print(f"  [!] {dem_key}: clipped file exists but extent mismatch, re-clipping")

        with rasterio.open(src_path) as src:
            src_bounds = src.bounds
            src_res = abs(src.res[0])
            src_nodata = info["nodata"]

            # Compute intersection of source and reference extents
            inter_left   = max(src_bounds.left, ref_bounds.left)
            inter_right  = min(src_bounds.right, ref_bounds.right)
            inter_bottom = max(src_bounds.bottom, ref_bounds.bottom)
            inter_top    = min(src_bounds.top, ref_bounds.top)

            if inter_left >= inter_right or inter_bottom >= inter_top:
                print(f"  [!] {dem_key}: No overlap with reference extent, skipping")
                continue

            # Snap the intersection bounds to the reference grid so that
            # pixel edges align across all datasets.
            # For each edge, find the nearest grid line in the reference grid.
            # Reference grid: x = ref_transform.c + col * ref_transform.a
            #                 y = ref_transform.f + row * ref_transform.e

            # Snap left/right to reference grid columns
            col_min = int(np.floor((inter_left - ref_transform.c) / ref_transform.a))
            col_max = int(np.ceil((inter_right - ref_transform.c) / ref_transform.a))
            snap_left  = ref_transform.c + col_min * ref_transform.a
            snap_right = ref_transform.c + col_max * ref_transform.a

            # Snap top/bottom to reference grid rows
            row_min = int(np.floor((inter_bottom - ref_transform.f) / ref_transform.e))
            row_max = int(np.ceil((inter_top - ref_transform.f) / ref_transform.e))
            snap_bottom = ref_transform.f + row_min * ref_transform.e
            snap_top    = ref_transform.f + row_max * ref_transform.e

            # Compute output grid for this DEM's resolution
            # The output must cover [snap_left, snap_right] x [snap_bottom, snap_top]
            # at the DEM's native resolution, but aligned so that pixel centres
            # coincide with the reference grid where possible.
            out_res = src_res
            out_width  = int(round((snap_right - snap_left) / out_res))
            out_height = int(round((snap_top - snap_bottom) / out_res))

            # Adjust right/top to match exact pixel count
            out_right = snap_left + out_width * out_res
            out_top   = snap_bottom + out_height * out_res

            # Build the output transform (same origin alignment as reference)
            out_transform = Affine(out_res, 0.0, snap_left,
                                   0.0, -out_res, out_top)

            # Check if clipping is actually needed
            src_left_rounded   = round(src_bounds.left, 1)
            src_right_rounded  = round(src_bounds.right, 1)
            src_bottom_rounded = round(src_bounds.bottom, 1)
            src_top_rounded    = round(src_bounds.top, 1)
            snap_left_rounded   = round(snap_left, 1)
            snap_right_rounded  = round(out_right, 1)
            snap_bottom_rounded = round(snap_bottom, 1)
            snap_top_rounded    = round(out_top, 1)

            needs_clip = not (
                src_left_rounded == snap_left_rounded and
                src_right_rounded == snap_right_rounded and
                src_bottom_rounded == snap_bottom_rounded and
                src_top_rounded == snap_top_rounded
            )

            if not needs_clip:
                print(f"  [Skip] {dem_key}: already aligned to common extent")
                continue

            print(f"  [Clip] {dem_key}: "
                  f"E=[{snap_left:.0f}, {out_right:.0f}], N=[{snap_bottom:.0f}, {out_top:.0f}] "
                  f"({out_width}x{out_height} px at {out_res}m)")

            # Use windowed read to extract the clipped region
            # Convert target bounds to source pixel coordinates
            from rasterio.windows import from_bounds as window_from_bounds

            clip_window = window_from_bounds(
                snap_left, snap_bottom, out_right, out_top,
                transform=src.transform
            )
            # Clamp to source bounds
            full_window = Window(0, 0, src.width, src.height)
            clip_window = clip_window.intersection(full_window)
            clip_window = clip_window.round_offsets().round_lengths()

            # Build output profile
            profile = src.profile.copy()
            profile.update({
                "height": out_height,
                "width": out_width,
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            })

            # Remove old clipped file if it exists
            if clipped_path.exists():
                os.remove(clipped_path)

            safe_makedirs(clipped_path.parent)

            # Write the clipped raster
            with rasterio.open(clipped_path, "w", **profile, BIGTIFF="YES") as dst:
                # Read in blocks for memory efficiency
                block_size = src.profile.get("blockxsize", 256)
                row_start = int(clip_window.row_off)
                col_start = int(clip_window.col_off)
                row_end = int(clip_window.row_off + clip_window.height)
                col_end = int(clip_window.col_off + clip_window.width)

                for row in range(row_start, row_end, block_size):
                    for col in range(col_start, col_end, block_size):
                        win = Window(
                            col,
                            row,
                            min(block_size, col_end - col),
                            min(block_size, row_end - row)
                        )
                        data = src.read(window=win)

                        # Destination window
                        dst_window = Window(
                            col_off=win.col_off - clip_window.col_off,
                            row_off=win.row_off - clip_window.row_off,
                            width=win.width,
                            height=win.height
                        )
                        dst.write(data, window=dst_window)

            print(f"  [OK] Clipped: {clipped_path.name}")

            # Remove old derivative files (.npy, .png, .json) that were computed
            # from a previous (unclipped or differently-clipped) DEM run.
            # Clean up both the unclipped stem and the clipped stem to avoid
            # stale derivatives from a previous clip with different bounds.
            deriv_dir = out_dir / dem_key
            stems_to_remove = [
                src_path.stem,          # e.g. "dem10m_epsg28355"
                clipped_path.stem,      # e.g. "dem10m_epsg28355_clipped"
            ]
            if deriv_dir.exists():
                n_removed = 0
                for f in list(deriv_dir.iterdir()):
                    if f.suffix in (".npy", ".png", ".json"):
                        for old_stem in stems_to_remove:
                            if f.stem.startswith(old_stem):
                                f.unlink()
                                n_removed += 1
                                break
                if n_removed > 0:
                    print(f"  [Cleanup] Removed {n_removed} old derivative files with stem '{old_stem}'")

            # Update the path in dems_to_process
            dems_to_process[dem_key]["path"] = clipped_path
            # Update nodata in case it changed
            dems_to_process[dem_key]["nodata"] = infer_nodata_value(clipped_path)

    return dems_to_process


# ===================================================================
# 2.  DERIVATIVE GENERATION (saved as .npy + .json + .png)
# ===================================================================

# -- 2a. Hillshade ------------------------------------------------

def compute_hillshade(dem_arr, resolution, azimuth=315, altitude=45, nodata=None):
    """
    Compute hillshade from a 2-D DEM array using analytical formula.
    Works with numpy arrays (no GDAL dependency).
    Falls back to the CPU implementation if CuPy is unavailable or fails.
    """
    res = get_resources()
    if not getattr(res, "use_gpu", False) or getattr(res, "_cp", None) is None:
        return compute_hillshade_cpu(dem_arr, resolution, azimuth=azimuth,
                                    altitude=altitude, nodata=nodata)

    xp = res.xp()

    try:
        # Ensure the input array lives on the same backend as `xp` (CuPy or NumPy).
        dem_arr = res.to_device(dem_arr)

        if nodata is not None:
            mask = (dem_arr == nodata) | xp.isnan(dem_arr) | (dem_arr < -1e6)
            dem = dem_arr.copy()
            dem[mask] = xp.nan
        else:
            dem = dem_arr.copy()
            mask = xp.isnan(dem)

        # Fill NaN for gradient calculation
        if xp.any(mask):
            filled = dem.copy()
            if xp.any(~xp.isnan(dem)):
                fill_value = xp.nanmean(dem)
            else:
                fill_value = 0.0
            filled[mask] = fill_value
            for _ in range(3):
                blurred = gaussian_filter(res.to_numpy(filled), sigma=2)
                blurred = res.to_device(blurred)
                weights = gaussian_filter(res.to_numpy((~xp.isnan(filled)).astype(float)), sigma=2)
                weights = res.to_device(weights)
                filled = xp.where(xp.isnan(filled), blurred / xp.maximum(weights, 1e-10), filled)
            dem_grad = filled
        else:
            dem_grad = dem

        # Compute gradients on device
        # CuPy doesn't have np.gradient, so use numpy for gradients then transfer
        if res.use_gpu:
            dem_np = res.to_numpy(dem_grad)
            dy, dx = np.gradient(dem_np, resolution, resolution)
            dy = res.to_device(dy)
            dx = res.to_device(dx)
        else:
            dy, dx = np.gradient(dem_grad, resolution, resolution)

        slope_rad = xp.arctan(xp.sqrt(dx**2 + dy**2))
        aspect_rad = xp.arctan2(-dy, dx)   # aspect from north, clockwise

        az_rad = xp.radians(azimuth)
        alt_rad = xp.radians(altitude)

        # Hillshade formula
        hs = (
            xp.sin(alt_rad) * xp.cos(slope_rad)
            + xp.cos(alt_rad) * xp.sin(slope_rad) * xp.cos(az_rad - aspect_rad)
        )
        hs = xp.clip(hs, 0, 1)

        # Restore nodata mask
        hs[mask] = xp.nan

        return res.to_numpy(hs)
    except Exception as e:
        print(f"  [GPU] Hillshade fallback: {e}")
        return compute_hillshade_cpu(dem_arr, resolution, azimuth=azimuth,
                                     altitude=altitude, nodata=nodata)


def compute_hillshade_cpu(dem_arr, resolution, azimuth=315, altitude=45, nodata=None):
    """CPU fallback for hillshade computation."""
    dem_arr = np.asarray(dem_arr)
    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = np.nan
    else:
        dem = dem_arr.copy()
        mask = np.isnan(dem)

    if np.any(mask):
        filled = dem.copy()
        if np.any(~np.isnan(dem)):
            fill_value = np.nanmean(dem)
        else:
            fill_value = 0.0
        filled[mask] = fill_value
        for _ in range(3):
            blurred = gaussian_filter(np.where(np.isnan(filled), 0, filled), sigma=2)
            weights = gaussian_filter((~np.isnan(filled)).astype(float), sigma=2)
            filled = np.where(np.isnan(filled), blurred / np.maximum(weights, 1e-10), filled)
        dem_grad = filled
    else:
        dem_grad = dem

    dy, dx = np.gradient(dem_grad, resolution, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    hs = (
        np.sin(alt_rad) * np.cos(slope_rad)
        + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    )
    hs = np.clip(hs, 0, 1)
    hs[mask] = np.nan
    return hs


def compute_hillshade_gpu(dem_arr, resolution, azimuth=315, altitude=45, nodata=None):
    """
    Compute hillshade using GPU acceleration (CuPy).
    Much faster for large arrays on RTX 2000 Ada.
    """
    res = get_resources()
    if not getattr(res, "use_gpu", False) or getattr(res, "_cp", None) is None:
        return compute_hillshade_cpu(dem_arr, resolution, azimuth=azimuth,
                                    altitude=altitude, nodata=nodata)

    xp = res.xp()

    if nodata is not None:
        mask = (dem_arr == nodata) | xp.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = xp.nan
    else:
        dem = dem_arr.copy()
        mask = xp.isnan(dem)

    # Fill NaN for gradient calculation
    if xp.any(mask):
        filled = dem.copy()
        if xp.any(~xp.isnan(dem)):
            fill_value = xp.nanmean(dem)
        else:
            fill_value = 0.0
        filled[mask] = fill_value
        # GPU-accelerated Gaussian filter
        # Note: CuPy doesn't have gaussian_filter, use scipy then transfer back
        if res.use_gpu:
            filled_np = res.to_numpy(filled)
            blurred = gaussian_filter(filled_np, sigma=2)
            weights = gaussian_filter((~np.isnan(filled_np)).astype(float), sigma=2)
            blurred = res.to_device(blurred)
            weights = res.to_device(weights)
            filled = xp.where(xp.isnan(filled), blurred / xp.maximum(weights, 1e-10), filled)
        else:
            blurred = gaussian_filter(res.to_numpy(filled), sigma=2)
            weights = gaussian_filter((~np.isnan(filled)).astype(float), sigma=2)
            filled = xp.where(xp.isnan(filled), blurred / xp.maximum(weights, 1e-10), filled)
        dem_grad = filled
    else:
        dem_grad = dem

    # Compute gradients on GPU using cupy's gradient implementation
    if res.use_gpu:
        # CuPy has cp.gradient similar to numpy
        dy, dx = xp.gradient(dem_grad, resolution, resolution)
    else:
        dy, dx = np.gradient(dem_grad, resolution, resolution)

    slope_rad = xp.arctan(xp.sqrt(dx**2 + dy**2))
    aspect_rad = xp.arctan2(-dy, dx)

    az_rad = xp.radians(azimuth)
    alt_rad = xp.radians(altitude)

    hs = (xp.sin(alt_rad) * xp.cos(slope_rad) +
          xp.cos(alt_rad) * xp.sin(slope_rad) * xp.cos(az_rad - aspect_rad))
    hs = xp.clip(hs, 0, 1)

    hs[mask] = xp.nan

    return res.to_numpy(hs)

def compute_multi_hillshade(
    dem_path,
    out_dir,
    nodata=None,
    azimuths=HS_AZIMUTHS,
    altitude=HS_ALTITUDE
):
    """
    Generate hillshade rasters at multiple azimuths and save as .npy + .json + .png.
    Returns dict {azimuth: output_npy_path}.
    """

    print(f"\n[Hillshade] {dem_path.name}  (azimuths={azimuths}, altitude={altitude} deg)")
    hs_paths = {}

    with rasterio.open(dem_path) as src:

        resolution = abs(src.res[0])
        transform = src.transform
        crs_str = str(src.crs)
        dem_stem = dem_path.stem

        print(f"  Reading DEM ({src.height}x{src.width}) ...")

        # Load DEM (CPU first)
        dem = src.read(1).astype(np.float32)

        # GPU transfer if available
        resources = get_resources()
        if resources.use_gpu:
            dem = resources.to_device(dem)

        for az in azimuths:

            deriv_name = f"hillshade_az{az}"
            npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"

            # Skip if already computed
            if npy_path.exists():
                print(f"    [Skip] az={az} deg already exists")
                hs_paths[az] = npy_path
                continue

            # Compute hillshade (THIS MUST BE OUTSIDE skip block)
            hs = compute_hillshade_gpu(
                dem,
                resolution,
                azimuth=az,
                altitude=altitude,
                nodata=nodata
            )

            # Save .npy + metadata
            saved_path = _save_npy_derivative(
                hs,
                out_dir,
                dem_stem,
                deriv_name,
                transform,
                crs_str,
                float(np.nan),
                resolution
            )
            hs_paths[az] = saved_path

            # Save PNG
            png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
            _save_derivative_png(
                hs,
                transform,
                png_path,
                cmap="gray",
                vmin=0,
                vmax=1,
                title=f"Hillshade az={az} deg, alt={altitude} deg"
            )

            print(f"    [OK] az={az} deg -> {npy_path.name}")

    return hs_paths



# -- 2b. Sky-View Factor (SVF) & Openness (OPTIMISED) ------------

def compute_svf_and_openness(dem_arr, resolution, n_dirs=16, max_radius=10,
                              nodata=None):
    """
    Compute Sky-View Factor and positive/negative topographic openness.

    OPTIMISED (v3):
      - Uses slice-based arithmetic instead of allocating a full 'shifted' array
        per (direction, radius) step. For each step, we only need the shifted
        values in the overlap region, so we index directly into dem_arr.
      - Avoids creating the 'valid' mask per step (precompute once).
      - Uses in-place accumulation to reduce memory allocations.

    Algorithm (Yokoyama et al. 2002):
      For each of n_dirs search directions, trace a ray up to max_radius cells.
      At each step, compute the horizon angle (max slope from cell to line-of-sight).
      SVF = (1/n_dirs) * sum cos(horizon_angle_i)   -- for each direction, take max upward angle
      Positive openness = mean of horizon angles across all directions
      Negative openness = mean of downward-looking angles (nadir-side)

    Args:
        dem_arr:     2-D float32 elevation array
        resolution:  cell size in metres
        n_dirs:      number of search directions (8, 16, or 32)
        max_radius:  search distance in cells
        nodata:      nodata value

    Returns:
        svf, openness_pos, openness_neg  (2-D float32 arrays)
    """
    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = np.nanmean(dem[~mask]) if np.any(~mask) else 0
    else:
        dem = dem_arr.copy()
        mask = np.isnan(dem)

    rows, cols = dem.shape

    # Pre-compute direction vectors
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dx_dirs = np.cos(angles)
    dy_dirs = np.sin(angles)

    # Output accumulators -- accumulate cos(horizon) and angles directly
    # to avoid storing n_dirs full-size arrays
    cos_horizon_sum = np.zeros((rows, cols), dtype=np.float64)
    horizon_up_sum  = np.zeros((rows, cols), dtype=np.float64)
    horizon_dn_sum  = np.zeros((rows, cols), dtype=np.float64)
    # Track per-pixel max/min angles per direction
    horizon_up_max  = np.full((rows, cols), -np.pi / 2, dtype=np.float32)
    horizon_dn_min  = np.full((rows, cols),  np.pi / 2, dtype=np.float32)

    for d_idx in range(n_dirs):
        dx = dx_dirs[d_idx]
        dy = dy_dirs[d_idx]

        # Reset per-direction trackers
        horizon_up_max[:] = -np.pi / 2
        horizon_dn_min[:] =  np.pi / 2

        for r in range(1, max_radius + 1):
            # Offset in cells (rounded)
            ci = int(round(dx * r))
            ri = int(round(dy * r))
            dist_m = np.sqrt((ci * resolution) ** 2 + (ri * resolution) ** 2)

            if dist_m == 0:
                continue

            # Compute valid index ranges for source and destination
            # Instead of allocating a full shifted array, compute dz directly
            # using slice indexing

            # Destination slice (where the shifted values end up)
            if ri >= 0:
                dst_r0, dst_r1 = ri, rows
                src_r0, src_r1 = 0, rows - ri
            else:
                dst_r0, dst_r1 = 0, rows + ri
                src_r0, src_r1 = -ri, rows
            if ci >= 0:
                dst_c0, dst_c1 = ci, cols
                src_c0, src_c1 = 0, cols - ci
            else:
                dst_c0, dst_c1 = 0, cols + ci
                src_c0, src_c1 = -ci, cols

            if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
                continue

            # Elevation difference directly via slices (no temporary array)
            dz = dem[src_r0:src_r1, src_c0:src_c1] - dem[dst_r0:dst_r1, dst_c0:dst_c1]
            angle = np.arctan2(dz, dist_m)

            # Update per-direction max/min
            np.maximum(horizon_up_max[dst_r0:dst_r1, dst_c0:dst_c1], angle,
                       out=horizon_up_max[dst_r0:dst_r1, dst_c0:dst_c1])
            np.minimum(horizon_dn_min[dst_r0:dst_r1, dst_c0:dst_c1], angle,
                       out=horizon_dn_min[dst_r0:dst_r1, dst_c0:dst_c1])

        # Accumulate across directions
        cos_horizon_sum += np.cos(np.clip(horizon_up_max, -np.pi / 2, np.pi / 2))
        horizon_up_sum  += horizon_up_max
        horizon_dn_sum  += horizon_dn_min

    # SVF: average of cos(horizon_angle) for upward-looking
    svf = (cos_horizon_sum / n_dirs).astype(np.float32)
    svf = np.clip(svf, 0, 1)

    # Positive openness: mean of horizon angles (upward)
    openness_pos_deg = np.degrees(horizon_up_sum / n_dirs).astype(np.float32)

    # Negative openness: mean of downward horizon angles (flip sign for "openness below")
    openness_neg_deg = np.degrees(-horizon_dn_sum / n_dirs).astype(np.float32)

    # Restore mask
    svf[mask] = np.nan
    openness_pos_deg[mask] = np.nan
    openness_neg_deg[mask] = np.nan

    return svf, openness_pos_deg, openness_neg_deg


def compute_svf_gpu(dem_arr, resolution, n_dirs=16, max_radius=10, nodata=None):
    """
    GPU-accelerated SVF computation.
    RTX 2000 Ada with 12GB VRAM can handle large tiles efficiently.
    """
    res = get_resources()
    xp = res.xp()

    if nodata is not None:
        mask = (dem_arr == nodata) | xp.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = xp.nanmean(dem[~mask]) if xp.any(~mask) else 0
    else:
        dem = dem_arr.copy()
        mask = xp.isnan(dem)

    rows, cols = dem.shape
    angles = xp.linspace(0, 2 * xp.pi, n_dirs, endpoint=False)
    dx_dirs = xp.cos(angles)
    dy_dirs = xp.sin(angles)

    cos_horizon_sum = xp.zeros((rows, cols), dtype=xp.float64)
    
    # Process on GPU if available
    if res.use_gpu:
        for d_idx in range(n_dirs):
            dx = dx_dirs[d_idx]
            dy = dy_dirs[d_idx]
            
            horizon_up_max = xp.full((rows, cols), -xp.pi / 2, dtype=xp.float32)
            
            for r in range(1, max_radius + 1):
                ci = int(round(float(dx) * r))
                ri = int(round(float(dy) * r))
                dist_m = xp.sqrt((ci * resolution) ** 2 + (ri * resolution) ** 2)
                
                if dist_m == 0:
                    continue
                
                # Use GPU slicing
                if ri >= 0:
                    dst_r0, dst_r1 = ri, rows
                    src_r0, src_r1 = 0, rows - ri
                else:
                    dst_r0, dst_r1 = 0, rows + ri
                    src_r0, src_r1 = -ri, rows
                    
                if ci >= 0:
                    dst_c0, dst_c1 = ci, cols
                    src_c0, src_c1 = 0, cols - ci
                else:
                    dst_c0, dst_c1 = 0, cols + ci
                    src_c0, src_c1 = -ci, cols
                
                if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
                    continue
                
                dz = dem[src_r0:src_r1, src_c0:src_c1] - dem[dst_r0:dst_r1, dst_c0:dst_c1]
                angle = xp.arctan2(dz, dist_m)
                
                horizon_up_max[dst_r0:dst_r1, dst_c0:dst_c1] = xp.maximum(
                    horizon_up_max[dst_r0:dst_r1, dst_c0:dst_c1], angle
                )
            
            cos_horizon_sum += xp.cos(xp.clip(horizon_up_max, -xp.pi / 2, xp.pi / 2))
        
        svf = (cos_horizon_sum / n_dirs).astype(xp.float32)
        svf = xp.clip(svf, 0, 1)
        svf[mask] = xp.nan
        
        return res.to_numpy(svf)
    else:
        # Fallback to CPU implementation
        return compute_svf_and_openness(dem_arr, resolution, n_dirs, max_radius, nodata)[0]


def generate_svf_openness(dem_path, out_dir, nodata=None, n_dirs=16, max_radius_m=None):
    """Compute and save SVF + openness with adaptive colormaps."""
    print(f"\n[SVF/Openness] {dem_path.name}")

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        transform = src.transform
        crs_str = str(src.crs)
        dem_stem = dem_path.stem

        total_px = src.height * src.width
        print(f"  Resolution: {res:.2f} m,  Shape: {src.height}x{src.width},  Total px: {total_px:,}")

        if max_radius_m is None:
            max_radius_m = 1000.0
        max_radius = int(max_radius_m / res)

        if max_radius > 200:
            print(f"  Capping max_radius from {max_radius} to 200 cells for performance")
            max_radius = 200

        print(f"  max_radius={max_radius} cells ({max_radius * res:.0f} m),  n_dirs={n_dirs}")

        if total_px > 50_000_000:
            print("  [!] Large raster -- processing in tiles")
            svf_full, op_full, on_full = _svf_tiled(src, res, n_dirs, max_radius, nodata)
        else:
            dem = src.read(1).astype(np.float32)
            with Timer(f"SVF/Openness {dem_path.name}"):
                svf_full, op_full, on_full = compute_svf_and_openness(
                    dem, res, n_dirs=n_dirs, max_radius=max_radius, nodata=nodata
                )

    # Save with adaptive colormaps
    for arr, suffix, cmap in [
        (svf_full, "_svf", "inferno"),
        (op_full, "_openness_pos", "RdYlBu"),
        (on_full, "_openness_neg", "RdYlBu_r"),
    ]:
        deriv_name = suffix.lstrip("_")
        npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"

        if npy_path.exists():
            print(f"  [Skip] {npy_path.name} already exists")
            continue

        _save_npy_derivative(arr, out_dir, dem_stem, deriv_name, transform, crs_str,
                              float(np.nan), res)

        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        
        # Adaptive colormap limits per derivative type
        if deriv_name == "svf":
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                vmin = max(0.3, np.percentile(valid, 2))
                vmax = min(1.0, np.percentile(valid, 98))
            else:
                vmin, vmax = 0.3, 1.0
        else:
            # Openness: use symmetric percentile clipping
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                vmin = np.percentile(valid, 2)
                vmax = np.percentile(valid, 98)
                # Ensure symmetry for RdYlBu colormaps
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = None, None
            
        _save_derivative_png_adaptive(arr, transform, png_path, cmap=cmap, 
                                       vmin=vmin, vmax=vmax,
                                       title=f"{deriv_name} | {dem_path.name}")
        print(f"  [OK] {npy_path.name}")


def _svf_tiled(src, res, n_dirs, max_radius, nodata):
    """Process SVF/openness in overlapping tiles for large rasters."""
    h, w = src.height, src.width
    tile_px = 2000  # process in 2000x2000 tiles
    overlap = max_radius + 10

    svf_acc = np.full((h, w), np.nan, dtype=np.float32)
    op_acc  = np.full((h, w), np.nan, dtype=np.float32)
    on_acc  = np.full((h, w), np.nan, dtype=np.float32)

    n_tiles_r = (h + tile_px - 1) // tile_px
    n_tiles_c = (w + tile_px - 1) // tile_px
    total_tiles = n_tiles_r * n_tiles_c

    t_start = time.time()
    tile_count = 0

    for r0 in tqdm(range(0, h, tile_px), desc="  SVF tiles"):
        for c0 in range(0, w, tile_px):
            # Read with overlap
            r_start = max(0, r0 - overlap)
            c_start = max(0, c0 - overlap)
            r_end = min(h, r0 + tile_px + overlap)
            c_end = min(w, c0 + tile_px + overlap)

            win = Window(c_start, r_start, c_end - c_start, r_end - r_start)
            dem_tile = src.read(1, window=win).astype(np.float32)

            svf_t, op_t, on_t = compute_svf_and_openness(
                dem_tile, res, n_dirs=n_dirs, max_radius=max_radius, nodata=nodata
            )

            # Write back only the inner (non-overlap) region
            inner_r0 = r0 - r_start
            inner_c0 = c0 - c_start
            inner_r1 = inner_r0 + min(tile_px, h - r0)
            inner_c1 = inner_c0 + min(tile_px, w - c0)

            svf_acc[r0:r0 + inner_r1 - inner_r0, c0:c0 + inner_c1 - inner_c0] = \
                svf_t[inner_r0:inner_r1, inner_c0:inner_c1]
            op_acc[r0:r0 + inner_r1 - inner_r0, c0:c0 + inner_c1 - inner_c0] = \
                op_t[inner_r0:inner_r1, inner_c0:inner_c1]
            on_acc[r0:r0 + inner_r1 - inner_r0, c0:c0 + inner_c1 - inner_c0] = \
                on_t[inner_r0:inner_r1, inner_c0:inner_c1]

            tile_count += 1
            if tile_count % 5 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / tile_count
                eta = rate * (total_tiles - tile_count)
                print(f"    SVF progress: {tile_count}/{total_tiles}, ETA: "
                      f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m")

    return svf_acc, op_acc, on_acc


# -- 2c. Slope, Curvature ----------------------------------------

def compute_slope(dem_arr, resolution, nodata=None):
    """Compute slope in degrees using Zevenbergen & Thorne method.
    Falls back to the CPU implementation if GPU acceleration is unavailable."""
    res_obj = get_resources()
    if getattr(res_obj, "use_gpu", False) and getattr(res_obj, "_cp", None) is not None:
        try:
            return compute_slope_gpu(dem_arr, resolution, nodata=nodata)
        except Exception as e:
            print(f"  [GPU] Slope fallback: {e}")

    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = np.nan
    else:
        dem = dem_arr.copy()
        mask = np.isnan(dem)

    # Fill NaN for gradient
    if np.any(mask):
        filled = dem.copy()
        filled[mask] = np.nanmean(dem[~mask])
        for _ in range(3):
            blurred = gaussian_filter(np.where(np.isnan(filled), 0, filled), sigma=2)
            weights  = gaussian_filter((~np.isnan(filled)).astype(float), sigma=2)
            filled = np.where(np.isnan(filled), blurred / np.maximum(weights, 1e-10), filled)
    else:
        filled = dem

    dy, dx = np.gradient(filled, resolution, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[mask] = np.nan
    return slope_deg


def compute_curvature(dem_arr, resolution, nodata=None, kind="profile"):
    """
    Compute curvature (1/m) using finite differences.
    Falls back to the CPU implementation if GPU acceleration is unavailable.

    Profile curvature  = d^2z/ds^2  (along-slope curvature -- convexity/concavity)
    Plan curvature     = d^2z/dn^2  (across-slope curvature -- lateral convergence)

    Uses the Zevenbergen & Thorne (1987) 3x3 kernel approach.
    """
    res_obj = get_resources()
    if getattr(res_obj, "use_gpu", False) and getattr(res_obj, "_cp", None) is not None:
        try:
            return compute_curvature_gpu(dem_arr, resolution, nodata=nodata, kind=kind)
        except Exception as e:
            print(f"  [GPU] Curvature fallback: {e}")

    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = np.nan
    else:
        dem = dem_arr.copy()
        mask = np.isnan(dem)

    # Fill for convolution
    if np.any(mask):
        filled = dem.copy()
        filled[mask] = np.nanmean(dem[~mask])
        for _ in range(3):
            blurred = gaussian_filter(np.where(np.isnan(filled), 0, filled), sigma=2)
            weights  = gaussian_filter((~np.isnan(filled)).astype(float), sigma=2)
            filled = np.where(np.isnan(filled), blurred / np.maximum(weights, 1e-10), filled)
    else:
        filled = dem

    cell = resolution
    g = cell

    # Pad array
    padded = np.pad(filled, 1, mode="edge")

    # Neighbours (3x3 window):
    #  z7  z2  z6
    #  z3  z0  z1
    #  z4  z5  z8

    z0 = padded[1:-1, 1:-1]
    z1 = padded[1:-1, 2:]
    z2 = padded[:-2, 1:-1]
    z3 = padded[1:-1, :-2]
    z4 = padded[2:,  :-2]
    z5 = padded[2:,  1:-1]
    z6 = padded[:-2, 2:]
    z7 = padded[:-2, :-2]
    z8 = padded[2:,  2:]

    D = (z1 + z3) / (2 * g**2) - z0 / g**2
    E = (z2 + z5) / (2 * g**2) - z0 / g**2
    F = (z6 + z4 - z7 - z8) / (4 * g**2)
    G = (z1 - z3) / (2 * g)
    H = (z2 - z5) / (2 * g)

    denom = (G**2 + H**2)
    safe_denom = np.maximum(denom, 1e-12)
    sqrt_denom = np.sqrt(safe_denom)

    if kind == "profile":
        raw = -2 * (D * G**2 + E * H**2 + F * G * H) / (safe_denom * sqrt_denom)
        curv = np.where(denom > 1e-12, raw, 0.0)
    elif kind == "plan":
        raw = 2 * (D * H**2 + E * G**2 - F * G * H) / (safe_denom * sqrt_denom)
        curv = np.where(denom > 1e-12, raw, 0.0)
    else:
        curv = D + E  # total (Laplacian)

    curv[mask] = np.nan
    return curv


def compute_slope_gpu(dem_arr, resolution, nodata=None):
    """GPU-accelerated slope computation using CuPy.
    Falls back to CPU compute_slope() if GPU unavailable."""
    res_obj = get_resources()
    if not res_obj.use_gpu or res_obj._cp is None:
        return compute_slope(dem_arr, resolution, nodata=nodata)

    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian

    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
    else:
        mask = np.isnan(dem_arr)

    # Move to GPU
    dem = cp.asarray(dem_arr.astype(np.float32))
    mask_gpu = cp.asarray(mask)

    fill_val = float(cp.nanmean(dem[~mask_gpu]))
    dem[mask_gpu] = fill_val

    # Simple iterative fill (3 passes of Gaussian blur)
    for _ in range(3):
        blurred = cp_gaussian(cp.where(mask_gpu, 0, dem), sigma=2)
        weights = cp_gaussian((~mask_gpu).astype(cp.float32), sigma=2)
        dem = cp.where(mask_gpu, blurred / cp.maximum(weights, 1e-10), dem)

    # Central differences
    dy = cp.zeros_like(dem)
    dx = cp.zeros_like(dem)
    dy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2 * resolution)
    dx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / (2 * resolution)
    # Edge handling
    dy[0, :] = (dem[1, :] - dem[0, :]) / resolution
    dy[-1, :] = (dem[-1, :] - dem[-2, :]) / resolution
    dx[:, 0] = (dem[:, 1] - dem[:, 0]) / resolution
    dx[:, -1] = (dem[:, -1] - dem[:, -2]) / resolution

    slope_deg = cp.degrees(cp.arctan(cp.sqrt(dx**2 + dy**2)))
    slope_deg[mask_gpu] = cp.nan

    return cp.asnumpy(slope_deg)


def compute_curvature_gpu(dem_arr, resolution, nodata=None, kind="profile"):
    """GPU-accelerated curvature computation using CuPy.
    Falls back to CPU compute_curvature() if GPU unavailable."""
    res_obj = get_resources()
    if not res_obj.use_gpu or res_obj._cp is None:
        return compute_curvature(dem_arr, resolution, nodata=nodata, kind=kind)

    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian
    g = float(resolution)

    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
    else:
        mask = np.isnan(dem_arr)

    # Move to GPU
    dem = cp.asarray(dem_arr.astype(np.float32))
    mask_gpu = cp.asarray(mask)

    fill_val = float(cp.nanmean(dem[~mask_gpu]))
    dem[mask_gpu] = fill_val

    # Simple iterative fill (3 passes)
    for _ in range(3):
        blurred = cp_gaussian(cp.where(mask_gpu, 0, dem), sigma=2)
        weights = cp_gaussian((~mask_gpu).astype(cp.float32), sigma=2)
        dem = cp.where(mask_gpu, blurred / cp.maximum(weights, 1e-10), dem)

    # Pad
    padded = cp.pad(dem, 1, mode='edge')

    z0 = padded[1:-1, 1:-1]
    z1 = padded[1:-1, 2:]
    z2 = padded[:-2, 1:-1]
    z3 = padded[1:-1, :-2]
    z4 = padded[2:,  :-2]
    z5 = padded[2:,  1:-1]
    z6 = padded[:-2, 2:]
    z7 = padded[:-2, :-2]
    z8 = padded[2:,  2:]

    D = (z1 + z3) / (2 * g**2) - z0 / g**2
    E = (z2 + z5) / (2 * g**2) - z0 / g**2
    F = (z6 + z4 - z7 - z8) / (4 * g**2)
    G = (z1 - z3) / (2 * g)
    H = (z2 - z5) / (2 * g)

    denom = (G**2 + H**2)
    safe_denom = cp.maximum(denom, 1e-12)
    sqrt_denom = cp.sqrt(safe_denom)

    if kind == "profile":
        raw = -2 * (D * G**2 + E * H**2 + F * G * H) / (safe_denom * sqrt_denom)
        curv = cp.where(denom > 1e-12, raw, cp.float32(0.0))
    elif kind == "plan":
        raw = 2 * (D * H**2 + E * G**2 - F * G * H) / (safe_denom * sqrt_denom)
        curv = cp.where(denom > 1e-12, raw, cp.float32(0.0))
    else:
        curv = D + E

    curv[mask_gpu] = cp.nan
    return cp.asnumpy(curv)



def generate_slope_curvature(dem_path, out_dir, nodata=None):
    """Compute and save slope, profile curvature, plan curvature with adaptive colormaps."""
    print(f"\n[Slope/Curvature] {dem_path.name}")

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        transform = src.transform
        crs_str = str(src.crs)
        dem_stem = dem_path.stem
        dem = src.read(1).astype(np.float32)

    # Slope
    deriv_name = "slope"
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    if not npy_path.exists():
        with Timer(f"Slope {dem_path.name}"):
            slope = compute_slope(dem, res, nodata=nodata)
        _save_npy_derivative(slope, out_dir, dem_stem, deriv_name, transform, crs_str,
                              float(np.nan), res)
        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        # Slope: clip at 98th percentile to avoid outliers
        valid = slope[~np.isnan(slope)]
        vmax = np.percentile(valid, 98) if len(valid) > 0 else 1.0
        _save_derivative_png_adaptive(slope, transform, png_path, cmap="viridis", 
                                       vmin=0, vmax=vmax,
                                       title=f"Slope (deg) | {dem_path.name}")
        print(f"  [OK] slope -> {npy_path.name}")
    else:
        print(f"  [Skip] slope already exists")

    # Profile curvature - use adaptive symmetric limits
    deriv_name = "curv_profile"
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    if not npy_path.exists():
        with Timer(f"Profile curvature {dem_path.name}"):
            prof_c = compute_curvature(dem, res, nodata=nodata, kind="profile")
        _save_npy_derivative(prof_c, out_dir, dem_stem, deriv_name, transform, crs_str,
                              float(np.nan), res)
        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        _save_derivative_png_adaptive(prof_c, transform, png_path, cmap="RdBu_r",
                                       use_percentile_clip=True, clip_low=2, clip_high=98,
                                       title=f"Profile Curvature (1/m) | {dem_path.name}")
        print(f"  [OK] profile curvature -> {npy_path.name}")
    else:
        print(f"  [Skip] profile curvature already exists")

    # Plan curvature - use adaptive symmetric limits
    deriv_name = "curv_plan"
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    if not npy_path.exists():
        with Timer(f"Plan curvature {dem_path.name}"):
            plan_c = compute_curvature(dem, res, nodata=nodata, kind="plan")
        _save_npy_derivative(plan_c, out_dir, dem_stem, deriv_name, transform, crs_str,
                              float(np.nan), res)
        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        _save_derivative_png_adaptive(plan_c, transform, png_path, cmap="RdBu_r",
                                       use_percentile_clip=True, clip_low=2, clip_high=98,
                                       title=f"Plan Curvature (1/m) | {dem_path.name}")
        print(f"  [OK] plan curvature -> {npy_path.name}")
    else:
        print(f"  [Skip] plan curvature already exists")


# -- 2d. Topographic Position Index (TPI) -- FFT-optimised --------

def compute_tpi(dem_arr, resolution, radius_m, nodata=None):
    """
    Compute Topographic Position Index (TPI):
      TPI = elevation - mean(elevation within annulus or circle of radius)

    OPTIMISED: Uses FFT-based convolution for large radii (>100 px),
    which is O(N log N) vs O(N * kernel_size^2) for uniform_filter.
    For small radii, falls back to scipy uniform_filter.
    """
    if nodata is not None:
        mask = (dem_arr == nodata) | np.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = np.nan
    else:
        dem = dem_arr.copy()
        mask = np.isnan(dem)

    radius_px = max(1, int(round(radius_m / resolution)))

    if np.any(mask):
        filled = dem.copy()
        filled[mask] = np.nanmean(dem[~mask])
    else:
        filled = dem

    # Choose algorithm based on kernel size
    if radius_px > 100:
        # FFT-based: build a circular kernel and use fftconvolve
        local_mean = _fft_circular_mean(filled.astype(np.float64), radius_px)
    else:
        local_mean = uniform_filter(filled.astype(np.float64), size=2 * radius_px + 1,
                                     mode="nearest")

    tpi = dem - local_mean.astype(np.float32)
    tpi[mask] = np.nan
    return tpi


def compute_tpi_gpu(dem_arr, resolution, radius_m, nodata=None):
    """
    Compute TPI using GPU acceleration for the convolution.
    RTX 2000 Ada excels at this type of operation.
    """
    res = get_resources()
    xp = res.xp()

    if nodata is not None:
        mask = (dem_arr == nodata) | xp.isnan(dem_arr) | (dem_arr < -1e6)
        dem = dem_arr.copy()
        dem[mask] = xp.nan
    else:
        dem = dem_arr.copy()
        mask = xp.isnan(dem)

    radius_px = max(1, int(round(radius_m / resolution)))

    if np.any(mask):
        filled = dem.copy()
        filled[mask] = xp.nanmean(dem[~mask]) if xp.any(~mask) else 0
    else:
        filled = dem

    # GPU-accelerated convolution
    if res.use_gpu and radius_px > 50:
        # Build circular kernel on GPU
        ksize = 2 * radius_px + 1
        kernel = xp.zeros((ksize, ksize), dtype=xp.float64)
        cy, cx = radius_px, radius_px
        y, x = xp.ogrid[:ksize, :ksize]
        kernel[(x - cx)**2 + (y - cy)**2 <= radius_px**2] = 1.0
        kernel /= kernel.sum()

        # CuPy convolution (similar to scipy.signal.fftconvolve but on GPU)
        from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
        local_mean = cp_fftconvolve(filled.astype(xp.float64), kernel, mode="same")
    elif radius_px > 100:
        # Fallback to CPU FFT
        local_mean = _fft_circular_mean(res.to_numpy(filled).astype(np.float64), radius_px)
        local_mean = xp.asarray(local_mean)
    else:
        # Use uniform filter on CPU for small kernels
        local_mean = uniform_filter(res.to_numpy(filled).astype(np.float64), 
                                     size=2 * radius_px + 1, mode="nearest")
        local_mean = xp.asarray(local_mean)

    tpi = dem - local_mean.astype(xp.float32)
    tpi[mask] = xp.nan

    return res.to_numpy(tpi)



def _fft_circular_mean(arr, radius_px):
    """Compute circular mean filter using FFT convolution.
    Much faster than uniform_filter for large radii."""
    ksize = 2 * radius_px + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    cy, cx = radius_px, radius_px
    for y in range(ksize):
        for x in range(ksize):
            if (x - cx)**2 + (y - cy)**2 <= radius_px**2:
                kernel[y, x] = 1.0
    kernel /= kernel.sum()

    # FFT convolution -- handles large kernels efficiently
    result = fftconvolve(arr, kernel, mode="same")
    return result


def generate_tpi(dem_path, out_dir, radii_m=TPI_RADII_M, nodata=None):
    """Compute and save TPI at multiple radii as .npy + .json + .png."""
    print(f"\n[TPI] {dem_path.name}  (radii={radii_m} m)")

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        transform = src.transform
        crs_str = str(src.crs)
        dem_stem = dem_path.stem
        dem = src.read(1).astype(np.float32)

    for r_m in radii_m:
        deriv_name = f"tpi_{r_m}m"
        npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
        if npy_path.exists():
            print(f"  [Skip] TPI {r_m}m already exists")
            continue
        with Timer(f"TPI {r_m}m ({r_m/res:.0f} px radius)"):
            tpi = compute_tpi(dem, res, r_m, nodata=nodata)
        _save_npy_derivative(tpi, out_dir, dem_stem, deriv_name, transform, crs_str,
                              float(np.nan), res)
        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        _save_derivative_png(tpi, transform, png_path, cmap="coolwarm",
                              title=f"TPI {r_m}m (m) | {dem_path.name}")
        print(f"  [OK] TPI {r_m}m -> {npy_path.name}")


# -- 2e. PCA on Multi-Azimuth Hillshades -------------------------

def generate_pca_composite(hs_paths, out_dir, dem_stem):
    """
    Stack multi-azimuth hillshades and compute PCA.
    First PC captures topographically coherent ridges/scarp regardless of sun angle.
    Saves result as .npy + .json + .png.
    """
    print(f"\n[PCA] {dem_stem}  ({len(hs_paths)} hillshade bands)")

    bands = []
    ref_transform = None
    ref_crs = None
    ref_res = None
    for az, p in sorted(hs_paths.items()):
        p = Path(p) if not isinstance(p, Path) else p
        if p.suffix == ".npy":
            data = np.load(p).astype(np.float32)
            # Load metadata for transform
            json_path = p.with_suffix(".json")
            if json_path.exists() and ref_transform is None:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                t = meta.get("transform", [1, 0, 0, 0, -1, 0])
                ref_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
                ref_crs = meta.get("crs", "EPSG:28355")
                ref_res = meta.get("resolution", 1.0)
        else:
            # Fallback: load from .tif
            with rasterio.open(p) as src:
                if ref_transform is None:
                    ref_transform = src.transform
                    ref_crs = str(src.crs)
                    ref_res = abs(src.res[0])
                data = src.read(1).astype(np.float32)
        # Replace NaN with 0 for PCA
        data = np.nan_to_num(data, nan=0.0)
        bands.append(data)

    if ref_transform is None:
        print("  [!] Could not determine georeferencing for PCA output")
        return None

    # Stack: (n_bands, rows, cols)
    stack = np.stack(bands, axis=0)
    n_bands, rows, cols = stack.shape

    # Reshape for PCA: (n_samples, n_features)
    X = stack.reshape(n_bands, -1).T  # (pixels, bands)

    print(f"  Running PCA on {X.shape[0]:,} pixels x {X.shape[1]} bands ...")
    with Timer("PCA computation"):
        pca = SkPCA(n_components=min(n_bands, 3))
        X_pca = pca.fit_transform(X)
    print(f"  Explained variance ratios: {pca.explained_variance_ratio_}")

    # Save first 3 PCs as separate .npy files
    for i in range(min(n_bands, 3)):
        deriv_name = f"pca_pc{i+1}"
        pc = X_pca[:, i].reshape(rows, cols).astype(np.float32)
        _save_npy_derivative(pc, out_dir, dem_stem, deriv_name, ref_transform,
                              ref_crs, float(np.nan), ref_res)
        png_path = out_dir / f"{dem_stem}_{deriv_name}.png"
        _save_derivative_png(pc, ref_transform, png_path, cmap="viridis",
                              title=f"PCA PC{i+1} ({pca.explained_variance_ratio_[i]:.1%}) | {dem_stem}")

    # Also save a 3-band composite as .npy
    composite = np.stack([X_pca[:, i].reshape(rows, cols) for i in range(min(n_bands, 3))],
                          axis=0).astype(np.float32)
    deriv_name = "pca_composite"
    npy_path = out_dir / f"{dem_stem}_{deriv_name}.npy"
    np.save(npy_path, composite)

    t = ref_transform
    meta = {
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "crs": ref_crs,
        "shape": list(composite.shape),
        "nodata": float(np.nan),
        "resolution": ref_res,
        "type": deriv_name,
        "n_components": min(n_bands, 3),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    json_path = out_dir / f"{dem_stem}_{deriv_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _cleanup_old_tif(out_dir, dem_stem, deriv_name)

    print(f"  [OK] PCA composite -> {npy_path.name}")
    return npy_path


# ===================================================================
# 3.  LiDAR-SPECIFIC TILED DERIVATIVE GENERATION (OPTIMISED)
# ===================================================================

def _process_lidar_tile_worker(args):
    """
    Worker function for parallel LiDAR tile processing.

    OPTIMISATION: Reads DEM once, computes ALL derivatives for the tile.
    This is the single biggest speedup -- the old code read the same tile
    7+ times (once per derivative type).
    """
    
    # Set worker mode to suppress resource detection
    set_worker_mode()

    (dem_path_str, r0, c0, tile_h, tile_w, overlap, h, w,
     nodata, res, azimuths, altitude, max_radius_svf, out_dir_str, dem_stem) = args

    out_dir = Path(out_dir_str)

    # Check if ALL derivative tiles for this spatial tile already exist
    deriv_names = [f"hillshade_az{az}" for az in azimuths] + \
                  ["slope", "curv_profile", "curv_plan", "svf", "openness_pos", "openness_neg"]
    all_exist = True
    for dn in deriv_names:
        tile_npy = out_dir / f"tile_{r0:04d}_{c0:04d}_{dn}.npy"
        if not tile_npy.exists():
            all_exist = False
            break
    if all_exist:
        return (r0, c0, 0, "skip")

    # Read DEM tile with overlap
    with rasterio.open(dem_path_str) as src:
        r_start = max(0, r0 - overlap)
        c_start = max(0, c0 - overlap)
        r_end = min(h, r0 + tile_h + overlap)
        c_end = min(w, c0 + tile_w + overlap)

        win = Window(c_start, r_start, c_end - c_start, r_end - r_start)
        data = src.read(1, window=win).astype(np.float32)

    # Strip overlap for saving
    inner_r0 = r0 - r_start
    inner_c0 = c0 - c_start
    inner_r1 = inner_r0 + tile_h
    inner_c1 = inner_c0 + tile_w

    results = {}
    n_computed = 0

    # 1. Hillshades (cheap)
    for az in azimuths:
        dn = f"hillshade_az{az}"
        tile_npy = out_dir / f"tile_{r0:04d}_{c0:04d}_{dn}.npy"
        if tile_npy.exists():
            continue
        hs = compute_hillshade(data, res, azimuth=az, altitude=altitude, nodata=nodata)
        inner = hs[inner_r0:inner_r1, inner_c0:inner_c1]
        np.save(tile_npy, inner.astype(np.float32))
        n_computed += 1

    # 2. Slope (cheap)
    dn = "slope"
    tile_npy = out_dir / f"tile_{r0:04d}_{c0:04d}_{dn}.npy"
    if not tile_npy.exists():
        slope = compute_slope(data, res, nodata=nodata)
        inner = slope[inner_r0:inner_r1, inner_c0:inner_c1]
        np.save(tile_npy, inner.astype(np.float32))
        n_computed += 1
    else:
        # Load for reuse if needed
        pass

    # 3. Curvature (cheap)
    for kind, dn in [("profile", "curv_profile"), ("plan", "curv_plan")]:
        tile_npy = out_dir / f"tile_{r0:04d}_{c0:04d}_{dn}.npy"
        if tile_npy.exists():
            continue
        curv = compute_curvature(data, res, nodata=nodata, kind=kind)
        inner = curv[inner_r0:inner_r1, inner_c0:inner_c1]
        np.save(tile_npy, inner.astype(np.float32))
        n_computed += 1

    # 4. SVF + Openness (EXPENSIVE -- compute ONCE, save all 3)
    # This is the critical fix: the old code had 3 separate lambda entries
    # that each called compute_svf_and_openness, wasting 66% of the time.
    svf_needed = not (out_dir / f"tile_{r0:04d}_{c0:04d}_svf.npy").exists()
    op_needed = not (out_dir / f"tile_{r0:04d}_{c0:04d}_openness_pos.npy").exists()
    on_needed = not (out_dir / f"tile_{r0:04d}_{c0:04d}_openness_neg.npy").exists()

    if svf_needed or op_needed or on_needed:
        svf, op, on = compute_svf_and_openness(
            data, res, n_dirs=16, max_radius=max_radius_svf, nodata=nodata
        )
        if svf_needed:
            inner = svf[inner_r0:inner_r1, inner_c0:inner_c1]
            np.save(out_dir / f"tile_{r0:04d}_{c0:04d}_svf.npy", inner.astype(np.float32))
            n_computed += 1
        if op_needed:
            inner = op[inner_r0:inner_r1, inner_c0:inner_c1]
            np.save(out_dir / f"tile_{r0:04d}_{c0:04d}_openness_pos.npy", inner.astype(np.float32))
            n_computed += 1
        if on_needed:
            inner = on[inner_r0:inner_r1, inner_c0:inner_c1]
            np.save(out_dir / f"tile_{r0:04d}_{c0:04d}_openness_neg.npy", inner.astype(np.float32))
            n_computed += 1

    return (r0, c0, n_computed, "ok")


def generate_lidar_derivatives(dem_path, out_dir, nodata):
    """
    Generate derivatives for LiDAR DEM using tiled processing.

    OPTIMISATIONS (v3):
      1. Tile-centric: reads DEM once per tile, computes all derivatives
      2. SVF+Openness computed ONCE (was 3x due to lambda bug)
      3. Multiprocessing: N workers process tiles in parallel
      4. Adaptive tile size based on available RAM
      5. Progress with ETA estimation

    Saves individual tile .npy files with a master metadata JSON.
    Also generates downsampled overview .npy + .png.
    """
    print(f"\n[LiDAR Tiled Derivatives] {dem_path.name}")
    safe_makedirs(out_dir)

    res_obj = get_resources()

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        h, w = src.height, src.width
        transform = src.transform
        crs_str = str(src.crs)
        dem_stem = dem_path.stem

        # Use adaptive tile size from resource detection
        tile_size = min(res_obj.tile_size, LIDAR_TILE_SIZE)
        overlap = LIDAR_TILE_OVERLAP

        # Compute tiling scheme
        tile_list = []
        for r0 in range(0, h, tile_size):
            for c0 in range(0, w, tile_size):
                tile_h = min(tile_size, h - r0)
                tile_w = min(tile_size, w - c0)
                tile_list.append((r0, c0, tile_h, tile_w))

        print(f"  Tile size: {tile_size}px, Overlap: {overlap}px, Total tiles: {len(tile_list)}")
        print(f"  Workers: {res_obj.n_workers}, GPU: {res_obj.use_gpu}")

        # SVF max_radius for LiDAR
        max_radius_lidar = min(200, int(1000 / res))
        if max_radius_lidar < 10:
            max_radius_lidar = 10

        # Estimate per-tile processing time
        # Based on: 4 hillshades + slope + 2 curvatures + SVF (dominant)
        # SVF dominates: n_dirs * max_radius iterations over tile_size^2 pixels
        svf_ops = 16 * max_radius_lidar * tile_size**2
        approx_sec_per_tile = svf_ops / 500_000_000  # rough: ~500M ops/sec single core
        total_est = approx_sec_per_tile * len(tile_list) / max(res_obj.n_workers, 1)
        print(f"  Estimated SVF ops/tile: {svf_ops:,.0f}")
        print(f"  Approx time/tile: {approx_sec_per_tile:.1f}s")
        print(f"  Estimated total: {total_est/3600:.1f}h (with {res_obj.n_workers} workers)")

        # Prepare work items
        work_items = []
        for r0, c0, tile_h, tile_w in tile_list:
            work_items.append((
                str(dem_path), r0, c0, tile_h, tile_w, overlap, h, w,
                nodata, res, HS_AZIMUTHS, HS_ALTITUDE, max_radius_lidar,
                str(out_dir), dem_stem
            ))

        # Process tiles with multiprocessing
        n_workers = res_obj.n_workers
        t_start = time.time()
        completed = 0
        n_total = len(work_items)
        n_computed_total = 0

        if n_workers > 1 and n_total > 1:
            print(f"  Processing {n_total} tiles with {n_workers} workers ...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_lidar_tile_worker, item): i
                           for i, item in enumerate(work_items)}

                for future in as_completed(futures):
                    try:
                        r0, c0, n_computed, status = future.result()
                        completed += 1
                        n_computed_total += n_computed

                        if completed % max(1, n_total // 20) == 0 or completed == n_total:
                            elapsed = time.time() - t_start
                            rate = elapsed / completed
                            eta = rate * (n_total - completed)
                            if eta > 3600:
                                eta_str = f"{eta/3600:.1f}h"
                            elif eta > 60:
                                eta_str = f"{eta/60:.0f}m"
                            else:
                                eta_str = f"{eta:.0f}s"
                            print(f"    Progress: {completed}/{n_total} tiles, "
                                  f"ETA: {eta_str}, computed: {n_computed_total} derivatives")
                    except Exception as e:
                        print(f"    [!] Tile failed: {e}")
                        completed += 1
        else:
            # Single-process fallback
            print(f"  Processing {n_total} tiles (single process) ...")
            for item in tqdm(work_items, desc="  LiDAR tiles"):
                try:
                    r0, c0, n_computed, status = _process_lidar_tile_worker(item)
                    completed += 1
                    n_computed_total += n_computed
                except Exception as e:
                    print(f"    [!] Tile ({item[1]},{item[2]}) failed: {e}")
                    completed += 1

        elapsed = time.time() - t_start
        print(f"  LiDAR derivatives complete: {n_computed_total} products in "
              f"{elapsed/60:.1f}m ({n_total} tiles)")

        # Save master metadata for each derivative type
        deriv_names = [f"hillshade_az{az}" for az in HS_AZIMUTHS] + \
                      ["slope", "curv_profile", "curv_plan", "svf", "openness_pos", "openness_neg"]

        for deriv_name in deriv_names:
            meta_path = out_dir / f"{dem_stem}_{deriv_name}_meta.json"
            if meta_path.exists():
                continue

            t = transform
            master_meta = {
                "crs": crs_str,
                "resolution": res,
                "nodata": nodata,
                "tile_size": tile_size,
                "overlap": overlap,
                "type": deriv_name,
                "full_shape": [h, w],
                "full_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                "n_tiles": len(tile_list),
                "tile_files": [f"tile_{r0:04d}_{c0:04d}_{deriv_name}.npy"
                                for r0, c0, _, _ in tile_list],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(master_meta, f, indent=2)

            # Generate downsampled overview PNG
            _generate_lidar_overview_png(dem_path, out_dir, dem_stem, deriv_name, nodata, res)

    # Clean up old .tif derivative files
    for dn in deriv_names:
        _cleanup_old_tif(out_dir, dem_stem, dn)


def _generate_lidar_overview_png(dem_path, out_dir, dem_stem, deriv_name, nodata, native_res):
    """Generate a downsampled overview PNG for a LiDAR derivative."""
    overview_res = LIDAR_OVERVIEW_RES
    png_path = out_dir / f"{dem_stem}_{deriv_name}_overview.png"

    if png_path.exists():
        return

    print(f"    Generating overview PNG for {deriv_name} at {overview_res}m resolution ...")

    try:
        # Read DEM at reduced resolution and compute derivative on-the-fly
        with rasterio.open(dem_path) as src:
            # Compute overview shape
            overview_factor = max(1, int(overview_res / native_res))
            ov_height = src.height // overview_factor
            ov_width = src.width // overview_factor

            # Read at reduced resolution using rasterio resampling
            from rasterio.enums import Resampling as Rsp
            data = src.read(1,
                            out_shape=(ov_height, ov_width),
                            resampling=Rsp.average).astype(np.float32)
            transform = src.transform * src.transform.scale(
                float(src.width) / ov_width,
                float(src.height) / ov_height
            )
            # Capture CRS while file is still open
            src_crs = str(src.crs)

        # Compute the derivative at overview resolution
        if deriv_name.startswith("hillshade_az"):
            az = int(deriv_name.split("az")[1])
            result = compute_hillshade(data, overview_res, azimuth=az,
                                        altitude=HS_ALTITUDE, nodata=nodata)
            cmap, vmin, vmax = "gray", 0, 1
        elif deriv_name == "slope":
            result = compute_slope(data, overview_res, nodata=nodata)
            cmap, vmin, vmax = "viridis", 0, None
        elif deriv_name == "curv_profile":
            result = compute_curvature(data, overview_res, nodata=nodata, kind="profile")
            cmap, vmin, vmax = "RdBu_r", None, None
        elif deriv_name == "curv_plan":
            result = compute_curvature(data, overview_res, nodata=nodata, kind="plan")
            cmap, vmin, vmax = "RdBu_r", None, None
        elif deriv_name == "svf":
            mr = min(50, int(500 / overview_res))
            result, _, _ = compute_svf_and_openness(data, overview_res, n_dirs=8,
                                                      max_radius=mr, nodata=nodata)
            cmap, vmin, vmax = "inferno", 0.3, 1.0
        elif deriv_name == "openness_pos":
            mr = min(50, int(500 / overview_res))
            _, result, _ = compute_svf_and_openness(data, overview_res, n_dirs=8,
                                                      max_radius=mr, nodata=nodata)
            cmap, vmin, vmax = "RdYlBu", None, None
        elif deriv_name == "openness_neg":
            mr = min(50, int(500 / overview_res))
            _, _, result = compute_svf_and_openness(data, overview_res, n_dirs=8,
                                                      max_radius=mr, nodata=nodata)
            cmap, vmin, vmax = "RdYlBu_r", None, None
        else:
            return

        # Ensure result is a plain numpy float32 array (not CuPy)
        result = np.asarray(result, dtype=np.float32)

        # Save overview as .npy + .png
        ov_npy_path = out_dir / f"{dem_stem}_{deriv_name}_overview.npy"
        np.save(ov_npy_path, result)

        # Save metadata for overview
        ov_json_path = out_dir / f"{dem_stem}_{deriv_name}_overview.json"
        t = transform
        ov_meta = {
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "crs": src_crs,
            "shape": list(result.shape),
            "nodata": nodata,
            "resolution": overview_res,
            "type": f"{deriv_name}_overview",
        }
        with open(ov_json_path, "w", encoding="utf-8") as f:
            json.dump(ov_meta, f, indent=2)

        _save_derivative_png(result, transform, png_path, cmap=cmap, vmin=vmin, vmax=vmax,
                              title=f"{deriv_name} overview ({overview_res}m) | {dem_path.name}",
                              downsample_max=3000)
    except Exception as e:
        print(f"    [!] Overview PNG failed for {deriv_name}: {e}")


def load_lidar_tile_data(out_dir, dem_stem, deriv_name, row_off, col_off, tile_size=LIDAR_TILE_SIZE):
    """Load a specific LiDAR derivative tile .npy file."""
    tile_npy = out_dir / f"tile_{row_off:04d}_{col_off:04d}_{deriv_name}.npy"
    if tile_npy.exists():
        return np.load(tile_npy)
    return None


# ===================================================================
# 4.  TILE SAMPLING & VISUALISATION (fault-aware)
# ===================================================================

def sample_valid_tiles(dem_path, tile_size_m=TILE_SIZE_M, n_tiles=N_TILES,
                        min_validity=MIN_VALIDITY, nodata=None, seed=42,
                        struct_pts=None, fault_proximity_m=500):
    """
    Sample tiles with >= min_validity (95%) valid (non-nodata) pixels.

    CHANGE 4: Fault-aware tile selection.
      1. First, identify tile locations that overlap with known fault traces
         (from structural points).
      2. Prioritize tiles with the most structural points nearby.
      3. If not enough fault-overlapping tiles exist, fall back to random valid tiles.

    Returns list of dicts with keys: center_x, center_y, row_off, col_off, tile_h, tile_w,
                                        valid_frac, n_struct_pts
    """
    dem_path = Path(dem_path) if not isinstance(dem_path, Path) else dem_path
    print(f"\n[Tile Sampling] {dem_path.name}  (tile={tile_size_m}m, n={n_tiles}, "
          f"validity>={min_validity:.0%})")

    rng = np.random.default_rng(seed)

    with rasterio.open(dem_path) as src:
        res = abs(src.res[0])
        transform = src.transform
        tile_px = int(round(tile_size_m / res))
        tile_h = min(tile_px, src.height)
        tile_w = min(tile_px, src.width)

        print(f"  Tile: {tile_w}x{tile_h} px  ({tile_w * res:.0f}x{tile_h * res:.0f} m)")

        max_row = src.height - tile_h
        max_col = src.width  - tile_w
        if max_row <= 0 or max_col <= 0:
            print("  [!] Raster too small for a single tile")
            return []

        # Build a KD-tree of structural points for fast proximity queries
        pt_tree = None
        pt_coords = None
        if struct_pts is not None and len(struct_pts) > 0:
            pt_coords = struct_pts[["XCoord", "YCoord"]].values.astype(np.float64)
            pt_tree = cKDTree(pt_coords)
            print(f"  Structural points available: {len(struct_pts):,}")

        # Candidate pool -- try many, keep the best
        n_candidates = n_tiles * 50
        candidates = []

        for _ in range(n_candidates):
            r0 = rng.integers(0, max_row + 1)
            c0 = rng.integers(0, max_col + 1)
            win = Window(c0, r0, tile_w, tile_h)
            data = src.read(1, window=win)

            if nodata is not None:
                if np.isnan(nodata):
                    valid_frac = np.mean(~np.isnan(data) & (data > -1e6))
                else:
                    valid_frac = np.mean((data != nodata) & ~np.isnan(data) & (data > -1e6))
            else:
                valid_frac = np.mean(~np.isnan(data) & (data > -1e6))

            if valid_frac < min_validity:
                continue

            # Compute center in CRS coordinates
            center_row = r0 + tile_h / 2
            center_col = c0 + tile_w / 2
            center_x, center_y = rasterio.transform.xy(transform, center_row, center_col)

            # Count structural points within proximity
            n_pts_near = 0
            if pt_tree is not None:
                # Query points within fault_proximity_m of the tile center
                indices = pt_tree.query_ball_point([center_x, center_y], fault_proximity_m)
                n_pts_near = len(indices)

            candidates.append({
                "center_x": center_x,
                "center_y": center_y,
                "row_off": r0,
                "col_off": c0,
                "tile_h": tile_h,
                "tile_w": tile_w,
                "valid_frac": valid_frac,
                "n_struct_pts": n_pts_near,
            })

        # Sort: prioritize tiles with structural points nearby (most points first),
        # then by validity for ties
        candidates.sort(key=lambda t: (-t["n_struct_pts"], -t["valid_frac"]))

        # Ensure spatial diversity: greedy selection with min distance
        selected = []
        min_dist = tile_size_m * 0.5  # at least half a tile apart

        # First pass: select fault-proximate tiles
        for c in candidates:
            if c["n_struct_pts"] == 0:
                break  # no more fault-proximate tiles
            if len(selected) >= n_tiles:
                break
            too_close = False
            for s in selected:
                dist = np.sqrt((c["center_x"] - s["center_x"])**2 +
                               (c["center_y"] - s["center_y"])**2)
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append(c)

        # Second pass: fill remaining with any valid tiles (including non-fault)
        if len(selected) < n_tiles:
            for c in candidates:
                if len(selected) >= n_tiles:
                    break
                if c in selected:
                    continue
                too_close = False
                for s in selected:
                    dist = np.sqrt((c["center_x"] - s["center_x"])**2 +
                                   (c["center_y"] - s["center_y"])**2)
                    if dist < min_dist:
                        too_close = True
                        break
                if not too_close:
                    selected.append(c)

        # Final fallback: no diversity constraint
        if len(selected) < n_tiles:
            for c in candidates:
                if len(selected) >= n_tiles:
                    break
                if c not in selected:
                    selected.append(c)

        n_fault_tiles = sum(1 for t in selected if t["n_struct_pts"] > 0)
        print(f"  Sampled {len(selected)} tiles (from {len(candidates)} candidates, "
              f"{n_fault_tiles} fault-proximate)")
        for i, t in enumerate(selected):
            tag = " *FAULT*" if t["n_struct_pts"] > 0 else ""
            print(f"    Tile {i+1}: centre=({t['center_x']:.0f}, {t['center_y']:.0f}), "
                  f"validity={t['valid_frac']:.2%}, pts={t['n_struct_pts']}{tag}")

    return selected


def plot_tile_panel(dem_path, tile_info, derivative_dir, out_fig_path,
                    nodata=None, struct_pts=None, traces=None):
    """
    Plot a multi-panel tile figure showing:
      1. Hillshaded DEM (az 315 deg)
      2. Slope
      3. SVF
      4. Positive openness
      5. Profile curvature
      6. TPI (200 m)

    CHANGE 4: Overlays fault traces from reconstructed traces as colored lines.

    Title = centre coordinates.  Axis labels = easting/northing offsets from centre.
    """
    r0 = tile_info["row_off"]
    c0 = tile_info["col_off"]
    th = tile_info["tile_h"]
    tw = tile_info["tile_w"]
    cx = tile_info["center_x"]
    cy = tile_info["center_y"]
    win = Window(c0, r0, tw, th)

    with rasterio.open(dem_path) as src:
        dem = src.read(1, window=win).astype(np.float32)
        res = abs(src.res[0])
        # Compute coordinate arrays for this tile
        tile_transform = src.window_transform(win)

    # Build x, y coordinate arrays (in metres from centre)
    x_start = tile_transform.c + 0.5 * tile_transform.a - cx
    x_end   = tile_transform.c + (tw - 0.5) * tile_transform.a - cx
    y_start = tile_transform.f + 0.5 * tile_transform.e - cy  # top row
    y_end   = tile_transform.f + (th - 0.5) * tile_transform.e - cy  # bottom row

    x = np.linspace(x_start, x_end, tw)
    y = np.linspace(y_start, y_end, th)

    # Mask nodata
    if nodata is not None:
        if np.isnan(nodata):
            mask = np.isnan(dem) | (dem < -1e6)
        else:
            mask = (dem == nodata) | np.isnan(dem) | (dem < -1e6)
    else:
        mask = np.isnan(dem) | (dem < -1e6)
    dem_masked = np.ma.masked_where(mask, dem)

    # -- Load or compute derivatives for this tile --
    base = dem_path.stem

    # Hillshade
    hs = compute_hillshade(dem, res, azimuth=315, altitude=HS_ALTITUDE, nodata=nodata)
    hs_masked = np.ma.masked_where(np.isnan(hs), hs)

    # Slope
    slope = compute_slope(dem, res, nodata=nodata)
    slope_masked = np.ma.masked_where(np.isnan(slope), slope)

    # SVF -- try to load from .npy, else compute inline
    svf_data, svf_transform, _, _, _ = _load_npy_derivative(derivative_dir, base, "svf")
    if svf_data is not None:
        # Extract the tile window from the loaded array
        svf = svf_data[r0:r0+th, c0:c0+tw]
    else:
        svf = compute_svf_and_openness(dem, res, n_dirs=16, max_radius=min(50, int(500/res)),
                                         nodata=nodata)[0]
    svf_masked = np.ma.masked_where(np.isnan(svf), svf)

    # Openness (positive)
    op_data, _, _, _, _ = _load_npy_derivative(derivative_dir, base, "openness_pos")
    if op_data is not None:
        op = op_data[r0:r0+th, c0:c0+tw]
    else:
        _, op, _ = compute_svf_and_openness(dem, res, n_dirs=16, max_radius=min(50, int(500/res)),
                                              nodata=nodata)
    op_masked = np.ma.masked_where(np.isnan(op), op)

    # Profile curvature
    prof_c = compute_curvature(dem, res, nodata=nodata, kind="profile")
    prof_c_masked = np.ma.masked_where(np.isnan(prof_c), prof_c)

    # TPI 200m
    tpi = compute_tpi(dem, res, 200, nodata=nodata)
    tpi_masked = np.ma.masked_where(np.isnan(tpi), tpi)

    # -- Structural points overlay (thinned to ~50 m spacing) --
    pts_in_tile = None
    if struct_pts is not None and len(struct_pts) > 0:
        tile_x_min = cx + x_start
        tile_x_max = cx + x_end
        tile_y_min = cy + y_end    # y_end is the bottom (smaller value in MGA)
        tile_y_max = cy + y_start  # y_start is the top

        pts_mask = (
            (struct_pts["XCoord"] >= tile_x_min) &
            (struct_pts["XCoord"] <= tile_x_max) &
            (struct_pts["YCoord"] >= tile_y_min) &
            (struct_pts["YCoord"] <= tile_y_max)
        )
        pts_in_tile = struct_pts[pts_mask]

        # Thin structural points to ~50 m spacing (source data is ~10 m)
        # Use a grid-based approach: divide tile into 50 m cells, keep
        # at most one point per cell to reduce visual clutter.
        if len(pts_in_tile) > 0:
            _thin_grid = 50.0  # metres
            _ix = ((pts_in_tile["XCoord"].values - tile_x_min) / _thin_grid).astype(int)
            _iy = ((pts_in_tile["YCoord"].values - tile_y_min) / _thin_grid).astype(int)
            _cell_key = _ix * 100000 + _iy
            _, _unique_idx = np.unique(_cell_key, return_index=True)
            pts_in_tile = pts_in_tile.iloc[_unique_idx]

    # -- Fault traces overlay (CHANGE 4) --
    trace_segments = []
    if traces is not None and len(traces) > 0:
        tile_x_min = cx + x_start
        tile_x_max = cx + x_end
        tile_y_min = cy + y_end
        tile_y_max = cy + y_start

        for trace in traces:
            coords = trace["coords"]
            # Check if any part of the trace intersects the tile
            in_tile = (
                (coords[:, 0] >= tile_x_min) & (coords[:, 0] <= tile_x_max) &
                (coords[:, 1] >= tile_y_min) & (coords[:, 1] <= tile_y_max)
            )
            if np.any(in_tile):
                trace_segments.append({
                    "coords": coords,
                    "source": trace["source"],
                    "in_tile_mask": in_tile,
                })

    # -- Determine axis label step --
    half_range = max(abs(x_start), abs(x_end), abs(y_start), abs(y_end))
    if half_range <= 100:
        label_step = 100
    else:
        label_step = 200

    # Source colour palette for fault traces
    source_colors = {
        "CWvic1": "#FF4444",
        "FSD1": "#44FF44",
        "GeoStru501": "#4488FF",
        "GeoStru2501": "#FFAA00",
    }

    # -- Plot --
    # SVF: normalise to [0, 1] using percentile-based scaling for informative colour
    svf_vmin, svf_vmax = 0.3, 1.0  # fallback defaults
    if svf_masked.count() > 0:
        svf_p2 = np.nanpercentile(svf_masked.compressed(), 2)
        svf_p98 = np.nanpercentile(svf_masked.compressed(), 98)
        if svf_p98 > svf_p2:
            svf_vmin, svf_vmax = svf_p2, svf_p98

    po_vmin, po_vmax = 0.3, 1.0  # fallback defaults
    if op_masked.count() > 0:
        po_p2 = np.nanpercentile(op_masked.compressed(), 2)
        po_p98 = np.nanpercentile(op_masked.compressed(), 98)
        if po_p98 > po_p2:
            po_vmin, po_vmax = po_p2, po_p98

    pc_vmin, pc_vmax = 0.3, 1.0  # fallback defaults
    if prof_c_masked.count() > 0:
        pc_p2 = np.nanpercentile(prof_c_masked.compressed(), 2)
        pc_p98 = np.nanpercentile(prof_c_masked.compressed(), 98)
        if pc_p98 > pc_p2:
            pc_vmin, pc_vmax = pc_p2, pc_p98

    panels = [
        ("Hillshade (az 315 deg, alt 8 deg)", hs_masked, "gray", None, None),
        ("Slope (deg)", slope_masked, "viridis", 0, np.nanpercentile(slope_masked.compressed(), 99) if slope_masked.count() > 0 else 1),
        ("Sky-View Factor", svf_masked, "magma", svf_vmin, svf_vmax),
        ("Positive Openness (deg)", op_masked, "RdYlBu", po_vmin, po_vmax),
        ("Profile Curvature (1/m)", prof_c_masked, "RdBu_r", pc_vmin, pc_vmax),
        ("TPI 200 m (m)", tpi_masked, "coolwarm", None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Tile Centre: ({cx:.0f} E, {cy:.0f} N)  EPSG:28355  |  "
                 f"DEM: {dem_path.name}",
                 fontsize=11, fontweight="bold", y=0.98)

    for ax, (title, data, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

        # Invert y-axis: in MGA, y increases upward, but row 0 is the top (north)
        ax.invert_yaxis()

        # Clip axis extent exactly to tile window (no padding beyond data)
        half_px_x = 0.5 * abs(x[1] - x[0]) if len(x) > 1 else 0
        half_px_y = 0.5 * abs(y[1] - y[0]) if len(y) > 1 else 0
        ax.set_xlim(x[0] - half_px_x, x[-1] + half_px_x)
        ax.set_ylim(y[-1] - half_px_y, y[0] + half_px_y)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Easting offset (m)", fontsize=8)
        ax.set_ylabel("Northing offset (m)", fontsize=8)
        ax.set_aspect('equal')

        # Axis ticks at +/-label_step
        ticks = np.arange(-label_step * 10, label_step * 10 + 1, label_step)
        ticks = ticks[(ticks >= x.min()) & (ticks <= x.max())]
        if len(ticks) > 0:
            ax.set_xticks(ticks)
        ticks_y = np.arange(-label_step * 10, label_step * 10 + 1, label_step)
        ticks_y = ticks_y[(ticks_y >= y.min()) & (ticks_y <= y.max())]
        if len(ticks_y) > 0:
            ax.set_yticks(ticks_y)

        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

        # Overlay structural points
        if pts_in_tile is not None and len(pts_in_tile) > 0:
            px = pts_in_tile["XCoord"].values - cx
            py = pts_in_tile["YCoord"].values - cy
            ax.scatter(px, py, s=4, c="lime", edgecolors="black", linewidths=0.3,
                       zorder=5, alpha=0.7)

        # Overlay fault traces (clipped to tile boundary)
        if trace_segments:
            for seg in trace_segments:
                coords = seg["coords"]
                color = source_colors.get(seg["source"], "yellow")
                # Plot only the in-tile portion of the trace
                in_tile = seg["in_tile_mask"]
                if not np.any(in_tile):
                    continue
                # Find contiguous in-tile segments and plot each
                diffs = np.diff(in_tile.astype(int))
                starts = np.where(diffs == 1)[0] + 1
                ends = np.where(diffs == -1)[0] + 1
                # Handle edge cases
                if in_tile[0]:
                    starts = np.concatenate([[0], starts])
                if in_tile[-1]:
                    ends = np.concatenate([ends, [len(in_tile)]])
                for s, e in zip(starts, ends):
                    seg_coords = coords[s:e]
                    ax.plot(seg_coords[:, 0] - cx, seg_coords[:, 1] - cy,
                            color=color, linewidth=1.0, alpha=0.6, zorder=4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fig_path, dpi=TILE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Tile figure -> {out_fig_path.name}")


# ===================================================================
# 5.  STRUCTURAL POINT LOADING
# ===================================================================

def load_structural_points(csv_path, dem_extent=None, crs="EPSG:28355"):
    """
    Load structural point inventory.
    If dem_extent is given (xmin, xmax, ymin, ymax in TARGET_CRS),
    only return points within that extent.
    """
    print(f"\n[Structural Points] Loading {csv_path.name}")
    chunks = pd.read_csv(csv_path, chunksize=500_000)
    dfs = []
    for chunk in tqdm(chunks, desc="  Reading CSV"):
        if dem_extent is not None:
            xmin, xmax, ymin, ymax = dem_extent
            mask = (
                (chunk["XCoord"] >= xmin) & (chunk["XCoord"] <= xmax) &
                (chunk["YCoord"] >= ymin) & (chunk["YCoord"] <= ymax)
            )
            chunk = chunk[mask]
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} points within extent")
    if len(df) > 0:
        print(f"  Sources: {df['source'].value_counts().to_dict()}")
    return df


# ===================================================================
# 6.  FAULT TRACE RECONSTRUCTION FROM POINTS
# ===================================================================

def reconstruct_fault_traces(struct_pts, cluster_dist_m=30, min_pts=5):
    """
    Reconstruct fault trace linestrings from the dense point inventory.

    The structural_points_inventory.csv encodes fault traces as densely
    sampled point coordinates (~10 m spacing). This function groups points
    by source, then clusters them into individual fault traces using
    BFS clustering with a distance threshold.

    Args:
        struct_pts:     DataFrame with XCoord, YCoord, source columns
        cluster_dist_m: maximum distance between consecutive points in
                        the same trace (m). Points further apart start
                        a new segment.
        min_pts:        minimum number of points for a valid trace

    Returns:
        List of dicts, each with keys:
            trace_id, source, coords (Nx2 array), length_m, azimuth_deg,
            n_points, centroid_x, centroid_y
    """
    print(f"\n[Fault Traces] Reconstructing from {len(struct_pts):,} points")
    print(f"  Cluster distance: {cluster_dist_m} m,  Min points: {min_pts}")

    traces = []
    trace_id = 0

    for src_name in struct_pts["source"].unique():
        pts = struct_pts[struct_pts["source"] == src_name].copy()
        if len(pts) < min_pts:
            continue

        coords = pts[["XCoord", "YCoord"]].values.astype(np.float64)

        # Build KD-tree for fast neighbour queries
        tree = cKDTree(coords)

        # Find connected components using BFS with distance threshold
        visited = np.zeros(len(coords), dtype=bool)
        component_ids = np.full(len(coords), -1, dtype=int)
        comp_idx = 0

        for start in range(len(coords)):
            if visited[start]:
                continue
            # BFS
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

        # Build traces from components
        for ci in range(comp_idx):
            mask = component_ids == ci
            comp_coords = coords[mask]
            if len(comp_coords) < min_pts:
                continue

            # Order points along the trace using nearest-neighbour chain
            ordered = _order_points_nn(comp_coords)

            # Compute trace properties
            diffs = np.diff(ordered, axis=0)
            seg_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            total_length = np.sum(seg_lengths)

            # Overall azimuth (first to last point)
            dx = ordered[-1, 0] - ordered[0, 0]
            dy = ordered[-1, 1] - ordered[0, 1]
            azimuth = np.degrees(np.arctan2(dx, dy)) % 360  # from north

            centroid_x = np.mean(ordered[:, 0])
            centroid_y = np.mean(ordered[:, 1])

            traces.append({
                "trace_id": trace_id,
                "source": src_name,
                "coords": ordered,
                "length_m": total_length,
                "azimuth_deg": azimuth,
                "n_points": len(ordered),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            })
            trace_id += 1

    print(f"  Reconstructed {len(traces)} fault traces")
    if len(traces) > 0:
        lengths = [t["length_m"] for t in traces]
        print(f"  Lengths: min={min(lengths):.0f}, median={np.median(lengths):.0f}, "
              f"max={max(lengths):.0f} m")
    return traces


def _order_points_nn(coords):
    """Order a set of 2-D points using nearest-neighbour chain starting
    from the point closest to the centroid of all points."""
    if len(coords) <= 2:
        return coords

    # Start from the point nearest to the centroid
    centroid = coords.mean(axis=0)
    dists = np.sqrt((coords[:, 0] - centroid[0])**2 + (coords[:, 1] - centroid[1])**2)
    start_idx = np.argmin(dists)

    ordered = [coords[start_idx]]
    remaining = np.ones(len(coords), dtype=bool)
    remaining[start_idx] = False

    current = coords[start_idx]
    while np.any(remaining):
        rem_coords = coords[remaining]
        d = np.sqrt((rem_coords[:, 0] - current[0])**2 + (rem_coords[:, 1] - current[1])**2)
        nearest = np.argmin(d)
        nearest_idx = np.where(remaining)[0][nearest]
        ordered.append(coords[nearest_idx])
        remaining[nearest_idx] = False
        current = coords[nearest_idx]

    return np.array(ordered)


# ===================================================================
# 7.  DATA OVERLAY AND DESKTOP REVIEW MAPS
# ===================================================================

def plot_fault_overlay_map(dem_path, deriv_dir, struct_pts, traces,
                            out_fig_path, nodata=None, azimuth=315):
    """
    Generate a desktop review map: hillshade basemap with fault traces
    from all sources overlaid, coloured by source database.

    Works with .npy hillshade derivatives, falling back to on-the-fly computation.
    """
    print(f"\n[Fault Overlay Map] {dem_path.name}")

    dem_stem = dem_path.stem

    # Load hillshade -- try .npy first, then compute from DEM
    hs_data, hs_transform, _, _, _ = _load_npy_derivative(deriv_dir, dem_stem,
                                                             f"hillshade_az{azimuth}")
    if hs_data is not None and hs_transform is not None:
        hs = hs_data
    else:
        # Compute hillshade on-the-fly
        with rasterio.open(dem_path) as src:
            res = abs(src.res[0])
            hs_transform = src.transform
            hs_bounds = src.bounds
            # For very large DEMs, compute at reduced resolution
            total_px = src.height * src.width
            if total_px > 50_000_000:
                # Read at reduced resolution
                ds_factor = max(1, int(np.sqrt(total_px / 5_000_000)))
                ov_h = src.height // ds_factor
                ov_w = src.width // ds_factor
                dem = src.read(1, out_shape=(ov_h, ov_w),
                                resampling=ResampEnum.average).astype(np.float32)
                hs_transform = src.transform * src.transform.scale(
                    float(src.width) / ov_w, float(src.height) / ov_h)
                res = abs(src.res[0]) * ds_factor
            else:
                dem = src.read(1).astype(np.float32)
                res = abs(src.res[0])
                hs_bounds = src.bounds
        hs = compute_hillshade(dem, res, azimuth=azimuth, altitude=HS_ALTITUDE,
                                nodata=nodata)

    # Get bounds from transform and data shape
    rows, cols = hs.shape
    x_min = hs_transform.c
    x_max = hs_transform.c + cols * hs_transform.a
    y_top = hs_transform.f
    y_bottom = hs_transform.f + rows * hs_transform.e
    extent = [x_min, x_max, y_bottom, y_top]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot hillshade
    hs_plot = np.where(np.isnan(hs), 0, hs)
    ax.imshow(hs_plot, cmap="gray", extent=extent, aspect="equal",
              vmin=0, vmax=1, origin="upper")

    # Colour palette for sources
    source_colors = {
        "CWvic1": "#FF4444",       # red -- Central-Western Victoria
        "FSD1": "#44FF44",         # green -- Fault Source Database
        "GeoStru501": "#4488FF",   # blue -- Geological Structure 501
        "GeoStru2501": "#FFAA00",  # orange -- Geological Structure 2501
    }

    # Plot fault traces
    for trace in traces:
        color = source_colors.get(trace["source"], "white")
        coords = trace["coords"]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.8,
                alpha=0.7, label=trace["source"] if trace["trace_id"] == 0 else "")

    # Clean legend (one entry per source)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="lower right",
                  fontsize=9, framealpha=0.8)

    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_aspect('equal')
    ax.set_title(f"Fault Trace Overlay on Hillshade (az {azimuth} deg)  |  "
                 f"DEM: {dem_path.name}", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Fault overlay map -> {out_fig_path.name}")


# ===================================================================
# 8.  PERPENDICULAR PROFILE EXTRACTION AND SCARP ANALYSIS
# ===================================================================

def extract_perpendicular_profiles(dem_path, traces, struct_pts,
                                    half_length_m=PROFILE_HALF_LENGTH_M,
                                    profile_spacing_m=PROFILE_SPACING_M,
                                    nodata=None):
    """
    Extract elevation profiles perpendicular to each fault trace.

    For each trace, sample points at profile_spacing_m intervals along
    the trace. At each sample point, compute the local fault direction
    using neighbouring points, then extract a profile perpendicular to it.

    Works with any DEM size via _sample_dem_line (windowed reads).
    """
    print(f"\n[Profiles] Extracting perpendicular profiles")
    print(f"  Half-length: {half_length_m} m,  Spacing: {profile_spacing_m} m")

    with rasterio.open(dem_path) as src:
        dem_transform = src.transform
        dem_crs = src.crs
        dem_shape = (src.height, src.width)
        dem_res = abs(src.res[0])

    profiles = []

    for trace in tqdm(traces, desc="  Traces", leave=False):
        coords = trace["coords"]
        trace_len = trace["length_m"]

        if trace_len < profile_spacing_m:
            continue

        # Sample points along the trace at regular intervals
        n_samples = max(1, int(trace_len / profile_spacing_m))
        sample_indices = np.linspace(0, len(coords) - 1, n_samples + 2, dtype=int)[1:-1]

        for si in sample_indices:
            # Local fault direction: use neighbouring points
            i_prev = max(0, si - 3)
            i_next = min(len(coords) - 1, si + 3)

            dx = coords[i_next, 0] - coords[i_prev, 0]
            dy = coords[i_next, 1] - coords[i_prev, 1]

            if dx == 0 and dy == 0:
                continue

            # Fault direction angle (from east, counterclockwise)
            fault_angle = np.arctan2(dy, dx)

            # Perpendicular direction (90 deg counter-clockwise)
            perp_angle = fault_angle + np.pi / 2

            # Sample point on the fault
            sample_x = coords[si, 0]
            sample_y = coords[si, 1]

            # Build profile line: sample at native resolution
            step = dem_res
            n_steps = int(half_length_m / step)
            distances = np.arange(-n_steps, n_steps + 1) * step

            prof_x = sample_x + distances * np.cos(perp_angle)
            prof_y = sample_y + distances * np.sin(perp_angle)

            # Sample DEM along profile
            elev = _sample_dem_line(dem_path, prof_x, prof_y, nodata=nodata)

            fault_az_deg = np.degrees(fault_angle) % 360

            profiles.append({
                "trace_id": trace["trace_id"],
                "source": trace["source"],
                "sample_x": sample_x,
                "sample_y": sample_y,
                "profile_dist": distances,
                "profile_elev": elev,
                "fault_azimuth": fault_az_deg,
                "local_direction_deg": np.degrees(perp_angle) % 360,
            })

    print(f"  Extracted {len(profiles)} profiles from {len(traces)} traces")
    return profiles


def _sample_dem_line(dem_path, x_coords, y_coords, nodata=None):
    """Sample DEM elevations along a polyline defined by x, y coordinates.
    Uses bilinear interpolation via rasterio.sample."""
    with rasterio.open(dem_path) as src:
        # rasterio.sample expects (x, y) pairs
        xy = list(zip(x_coords, y_coords))
        values = list(src.sample(xy))
        values = np.array([v[0] for v in values])

        if nodata is not None:
            values[values == nodata] = np.nan
        values[values < -1e6] = np.nan

    return values.astype(np.float32)


def analyse_scarp_profiles(profiles, min_height_m=SCARP_MIN_HEIGHT_M,
                           max_width_m=SCARP_MAX_WIDTH_M):
    """
    Analyse each perpendicular profile for scarp characteristics.

    Detects the fault scarp by finding the steepest gradient zone
    near the profile centre (where the fault trace intersects).
    Classifies the scarp type as:
      - "discrete": sharp break, narrow transition zone
      - "monocline": broad flexure, gradual elevation change
      - "none": no detectable scarp

    Returns a DataFrame with one row per profile.
    """
    print(f"\n[Scarp Analysis] Analysing {len(profiles)} profiles")

    results = []

    for prof in tqdm(profiles, desc="  Analysing", leave=False):
        dist = prof["profile_dist"]
        elev = prof["profile_elev"].copy()

        # Skip profiles with too many missing values
        valid = ~np.isnan(elev)
        if np.sum(valid) < len(elev) * 0.5:
            results.append(_empty_scarp_result(prof))
            continue

        # Smooth profile slightly to reduce noise
        if np.any(~valid):
            elev[~valid] = np.nan
            valid_idx = np.where(valid)[0]
            if len(valid_idx) < 3:
                results.append(_empty_scarp_result(prof))
                continue
            elev = np.interp(np.arange(len(elev)), valid_idx, elev[valid_idx])

        # Apply mild smoothing (5-cell moving average)
        kernel_size = min(5, len(elev) // 3)
        if kernel_size >= 3:
            elev_smooth = uniform_filter(elev.astype(float), size=kernel_size)
        else:
            elev_smooth = elev.astype(float)

        # Compute slope along the profile
        if len(elev_smooth) < 3:
            results.append(_empty_scarp_result(prof))
            continue

        delev = np.gradient(elev_smooth, dist[1] - dist[0] if len(dist) > 1 else 1.0)
        slope_profile = np.abs(delev)

        # Find the steepest zone near the centre
        centre_idx = len(dist) // 2
        search_radius = len(dist) // 3
        search_start = max(0, centre_idx - search_radius)
        search_end = min(len(dist), centre_idx + search_radius)

        if search_end <= search_start:
            results.append(_empty_scarp_result(prof))
            continue

        local_slope = slope_profile[search_start:search_end]
        peak_idx = np.argmax(local_slope) + search_start

        # Measure scarp height
        n_side = max(5, len(dist) // 10)
        left_idx = max(0, peak_idx - n_side)
        right_idx = min(len(elev_smooth) - 1, peak_idx + n_side)

        left_mean = np.nanmean(elev_smooth[left_idx:peak_idx])
        right_mean = np.nanmean(elev_smooth[peak_idx:right_idx + 1])

        scarp_height = abs(right_mean - left_mean)
        max_slope = local_slope.max()

        max_slope_deg = np.degrees(np.arctan(max_slope))

        # Scarp width
        if scarp_height > 0.1:
            threshold_lo = min(left_mean, right_mean) + 0.1 * scarp_height
            threshold_hi = min(left_mean, right_mean) + 0.9 * scarp_height

            elev_section = elev_smooth[left_idx:right_idx + 1]
            dist_section = dist[left_idx:right_idx + 1]

            above_lo = np.where(elev_section >= threshold_lo)[0]
            above_hi = np.where(elev_section >= threshold_hi)[0]

            if len(above_lo) > 0 and len(above_hi) > 0:
                scarp_width = abs(dist_section[above_hi[0]] - dist_section[above_lo[0]])
            else:
                scarp_width = abs(dist[right_idx] - dist[left_idx])
        else:
            scarp_width = 0

        # Offset from centre
        offset_m = dist[peak_idx]

        # Classify scarp type
        if scarp_height < min_height_m:
            scarp_type = "none"
        elif scarp_width <= max_width_m:
            scarp_type = "discrete"
        else:
            scarp_type = "monocline"

        results.append({
            "trace_id": prof["trace_id"],
            "source": prof["source"],
            "sample_x": prof["sample_x"],
            "sample_y": prof["sample_y"],
            "fault_azimuth": prof["fault_azimuth"],
            "scarp_height_m": round(scarp_height, 2),
            "scarp_width_m": round(scarp_width, 1),
            "max_slope_deg": round(max_slope_deg, 2),
            "scarp_type": scarp_type,
            "offset_from_centre_m": round(offset_m, 1),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"  Scarp detection summary:")
        print(f"    discrete: {(df.scarp_type=='discrete').sum()}")
        print(f"    monocline: {(df.scarp_type=='monocline').sum()}")
        print(f"    none: {(df.scarp_type=='none').sum()}")
        if (df.scarp_type != 'none').any():
            detected = df[df.scarp_type != 'none']
            print(f"    Height range: {detected.scarp_height_m.min():.2f} - "
                  f"{detected.scarp_height_m.max():.2f} m")
    return df


def _empty_scarp_result(prof):
    """Return an empty scarp analysis result for a profile."""
    return {
        "trace_id": prof["trace_id"],
        "source": prof["source"],
        "sample_x": prof["sample_x"],
        "sample_y": prof["sample_y"],
        "fault_azimuth": prof["fault_azimuth"],
        "scarp_height_m": 0.0,
        "scarp_width_m": 0.0,
        "max_slope_deg": 0.0,
        "scarp_type": "none",
        "offset_from_centre_m": 0.0,
    }


# ===================================================================
# 9.  SCARP PROFILE VISUALISATION
# ===================================================================

def plot_scarp_profiles(profiles, scarp_df, out_dir, max_plots=20):
    """
    Plot annotated elevation profiles showing detected scarps.
    """
    print(f"\n[Profile Plots] Generating up to {max_plots} scarp profile figures")

    safe_makedirs(out_dir)

    if len(scarp_df) == 0:
        print("  No profiles to plot")
        return

    detected = scarp_df[scarp_df["scarp_type"] != "none"].copy()
    if len(detected) == 0:
        print("  No scarps detected in any profile")
        detected = scarp_df.head(max_plots)
    else:
        detected = detected.sort_values("scarp_height_m", ascending=False)
        detected = detected.head(max_plots)

    # Build lookup: (trace_id, sample_x, sample_y) -> profile
    prof_lookup = {}
    for p in profiles:
        key = (p["trace_id"], round(p["sample_x"], 0), round(p["sample_y"], 0))
        prof_lookup[key] = p

    for idx, (_, row) in enumerate(detected.iterrows()):
        key = (row["trace_id"], round(row["sample_x"], 0), round(row["sample_y"], 0))
        prof = prof_lookup.get(key)
        if prof is None:
            continue

        dist = prof["profile_dist"]
        elev = prof["profile_elev"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(dist, elev, "k-", linewidth=0.8, label="Elevation")

        if row["scarp_type"] != "none":
            delev = np.abs(np.gradient(np.nan_to_num(elev, nan=0.0),
                                       dist[1] - dist[0] if len(dist) > 1 else 1.0))
            peak_idx = np.argmax(delev)
            scarp_half = int(row["scarp_width_m"] / 2 / (dist[1] - dist[0])
                             if len(dist) > 1 else 5)
            scarp_half = max(3, scarp_half)

            left = max(0, peak_idx - scarp_half)
            right = min(len(dist), peak_idx + scarp_half)

            ax.axvspan(dist[left], dist[right - 1], alpha=0.2, color="red",
                       label=f"Scarp zone ({row['scarp_type']})")

            mid_idx = peak_idx
            ax.annotate(f"h = {row['scarp_height_m']:.2f} m\n"
                        f"w = {row['scarp_width_m']:.0f} m\n"
                        f"slope = {row['max_slope_deg']:.1f} deg",
                        xy=(dist[mid_idx], elev[mid_idx] if not np.isnan(elev[mid_idx])
                            else np.nanmean(elev)),
                        xytext=(20, 20), textcoords="offset points",
                        fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.5))

        ax.axvline(0, color="blue", linestyle="--", linewidth=0.5, alpha=0.5,
                   label="Fault trace")

        ax.set_xlabel("Distance from fault trace (m)", fontsize=9)
        ax.set_ylabel("Elevation (m AHD)", fontsize=9)
        ax.set_title(f"Profile at ({row['sample_x']:.0f} E, {row['sample_y']:.0f} N)  |  "
                     f"Source: {row['source']}  |  Type: {row['scarp_type']}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / f"scarp_profile_{idx+1:02d}_" \
                   f"{row['sample_x']:.0f}_{row['sample_y']:.0f}.png"
        fig.savefig(fig_path, dpi=400, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Saved {len(detected)} scarp profile figures")


# ===================================================================
# 10. DRAINAGE ANOMALY DETECTION
# ===================================================================

def detect_drainage_anomalies(dem_path, deriv_dir, traces, struct_pts,
                               nodata=None, dem_stem=None,
                               curv_threshold=DRAIN_CURV_THRESHOLD,
                               slope_min=DRAIN_SLOPE_MIN,
                               min_length_m=DRAIN_MIN_LENGTH_M,
                               is_large_dem=False):
    """
    Detect drainage anomalies near fault traces that may indicate
    tectonic deflection or fault-controlled drainage.

    Method:
      1. Compute a "drainage anomaly index" = |profile curvature| * slope
      2. Extract pixels within a buffer of fault traces.
      3. Identify connected anomaly clusters along fault corridors.
      4. Classify as: deflected stream, linear valley, or none.

    For small DEMs: loads slope/curvature from .npy files.
    For large DEMs (LiDAR): computes on-the-fly per fault trace corridor.

    Returns:
        DataFrame of anomaly features with location, type, and metrics.
    """
    print(f"\n[Drainage Anomalies] Detecting near fault traces")

    if is_large_dem:
        return _detect_drainage_anomalies_large(dem_path, traces, struct_pts,
                                                  nodata, curv_threshold, slope_min,
                                                  min_length_m)

    # -- Small DEM path: load from .npy or compute from full DEM --
    if dem_stem is None:
        dem_stem = Path(dem_path).stem

    # Load slope
    slope_data, slope_transform, _, _, slope_res = _load_npy_derivative(
        deriv_dir, dem_stem, "slope")
    if slope_data is None:
        # Compute from DEM
        with rasterio.open(dem_path) as src:
            slope_res = abs(src.res[0])
            slope_transform = src.transform
            dem = src.read(1).astype(np.float32)
        slope_data = compute_slope(dem, slope_res, nodata=nodata)

    # Load curvature
    curv_data, _, _, _, _ = _load_npy_derivative(deriv_dir, dem_stem, "curv_profile")
    if curv_data is None:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
        curv_data = compute_curvature(dem, slope_res, nodata=nodata, kind="profile")

    res = slope_res
    transform = slope_transform

    # Drainage anomaly index
    dai = np.abs(curv_data) * slope_data
    dai[np.isnan(dai)] = 0
    dai[slope_data < slope_min] = 0

    # Create fault buffer mask
    print("  Building fault buffer mask...")
    fault_mask = np.zeros(dai.shape, dtype=bool)

    for trace in traces:
        coords = trace["coords"]
        for pt in coords[::5]:  # subsample for speed
            col, row = ~transform * (pt[0], pt[1])
            row, col = int(row), int(col)
            buf_px = int(2000 / res)
            r0 = max(0, row - buf_px)
            r1 = min(dai.shape[0], row + buf_px + 1)
            c0 = max(0, col - buf_px)
            c1 = min(dai.shape[1], col + buf_px + 1)
            fault_mask[r0:r1, c0:c1] = True

    # Apply fault buffer
    dai_near_fault = np.where(fault_mask, dai, 0)

    # Threshold
    if np.any(dai_near_fault > 0):
        threshold = np.nanpercentile(dai_near_fault[dai_near_fault > 0], 95)
    else:
        threshold = curv_threshold

    anomaly_binary = dai_near_fault > threshold

    # Find connected components
    labelled, n_features = label(anomaly_binary)
    print(f"  Found {n_features} candidate anomaly clusters")

    # Analyse each cluster
    anomalies = _analyse_anomaly_clusters(labelled, n_features, dai_near_fault, res, transform,
                                            traces)

    df = pd.DataFrame(anomalies)
    if len(df) > 0:
        print(f"  Detected {len(df)} drainage anomalies")
        print(f"    Types: {df['anomaly_type'].value_counts().to_dict()}")
    else:
        print(f"  No significant drainage anomalies detected")

    return df


def _detect_drainage_anomalies_large(dem_path, traces, struct_pts, nodata,
                                       curv_threshold, slope_min, min_length_m):
    """Detect drainage anomalies for large DEMs (LiDAR) using on-the-fly computation
    per fault trace corridor.
    
    Optimisation strategy:
      1. Downsample: for LiDAR (0.5 m), resample corridors to
         LIDAR_DRAIN_RES_M (default 2 m) -- anomalies at 200 m+ scale
         are well-resolved at 2 m.  This gives a ~6x speedup.
      2. GPU-accelerated slope/curvature via CuPy when available.
      3. Pre-filter: skip traces shorter than min_length_m (they cannot
         produce anomalies longer than themselves).
      4. Merge nearby corridors into non-overlapping tiles to avoid
         redundant DEM reads, then split oversized tiles.
      5. Optimised cluster analysis (find_objects + covariance orientation).
    """
    print(f"  [LiDAR] Computing drainage anomalies on-the-fly per fault corridor")
    res_obj = get_resources()
    use_gpu = res_obj.use_gpu and res_obj._cp is not None

    # --- Configuration ---
    LIDAR_DRAIN_RES_M = 2.0  # target resolution for drainage analysis (metres)

    with rasterio.open(dem_path) as src:
        native_res = abs(src.res[0])
        dem_transform = src.transform
        h, w = src.height, src.width

    downsample_factor = max(1, int(round(LIDAR_DRAIN_RES_M / native_res)))
    effective_res = native_res * downsample_factor
    print(f"  [LiDAR] Native res: {native_res:.2f} m -> drain analysis at {effective_res:.2f} m "
          f"(downsample {downsample_factor}x)")
    if use_gpu:
        print(f"  [LiDAR] GPU-accelerated slope/curvature enabled")

    # --- Pre-filter short traces ---
    filtered_traces = [t for t in traces if t["length_m"] >= min_length_m]
    skipped = len(traces) - len(filtered_traces)
    if skipped > 0:
        print(f"  [LiDAR] Skipped {skipped} traces shorter than {min_length_m} m")
    print(f"  [LiDAR] Processing {len(filtered_traces)} corridors")

    # --- Merge overlapping corridors ---
    merged_corridors = _merge_fault_corridors(filtered_traces, buf_m=2000)
    print(f"  [LiDAR] Merged into {len(merged_corridors)} non-overlapping tiles "
          f"(from {len(filtered_traces)} traces)")

    # --- Process each tile ---
    all_anomalies = []

    for corridor_info in tqdm(merged_corridors, desc="  LiDAR drainage tiles", leave=False):
        x_min, x_max, y_min, y_max, corridor_traces = corridor_info

        try:
            with rasterio.open(dem_path) as src:
                # Convert to pixel coordinates
                c_min_f, r_min_f = ~dem_transform * (x_min, y_max)
                c_max_f, r_max_f = ~dem_transform * (x_max, y_min)
                r0 = max(0, int(r_min_f))
                r1 = min(h, int(r_max_f))
                c0 = max(0, int(c_min_f))
                c1 = min(w, int(c_max_f))

                if r1 <= r0 or c1 <= c0:
                    continue

                # Read at downsampled resolution
                out_h = max(1, (r1 - r0) // downsample_factor)
                out_w = max(1, (c1 - c0) // downsample_factor)

                win = Window(c0, r0, c1 - c0, r1 - r0)
                dem_tile = src.read(1, out_shape=(out_h, out_w),
                                     resampling=ResampEnum.average,
                                     window=win).astype(np.float32)
                tile_res = effective_res
                tile_transform = rasterio.transform.from_bounds(
                    x_min, y_min, x_max, y_max, out_w, out_h)

            # Compute slope and curvature (GPU or CPU)
            if use_gpu:
                slope_tile = compute_slope_gpu(dem_tile, tile_res, nodata=nodata)
                curv_tile = compute_curvature_gpu(dem_tile, tile_res, nodata=nodata, kind="profile")
            else:
                slope_tile = compute_slope(dem_tile, tile_res, nodata=nodata)
                curv_tile = compute_curvature(dem_tile, tile_res, nodata=nodata, kind="profile")

            # DAI
            dai = np.abs(curv_tile) * slope_tile
            dai[np.isnan(dai)] = 0
            dai[slope_tile < slope_min] = 0

            # Threshold
            if np.any(dai > 0):
                threshold = np.nanpercentile(dai[dai > 0], 95)
            else:
                threshold = curv_threshold

            anomaly_binary = dai > threshold
            labelled, n_features = label(anomaly_binary)

            if n_features == 0:
                continue

            # Analyse clusters
            cluster_anomalies = _analyse_anomaly_clusters(
                labelled, n_features, dai, tile_res, tile_transform, corridor_traces)
            all_anomalies.extend(cluster_anomalies)

        except Exception as e:
            print(f"  [Warning] Tile failed: {e}")

    df = pd.DataFrame(all_anomalies)
    if len(df) > 0:
        # Deduplicate by proximity
        if len(df) > 1:
            pts = df[["x", "y"]].values
            keep = [0]
            for i in range(1, len(pts)):
                dists = np.sqrt((pts[i, 0] - pts[keep, 0])**2 +
                                (pts[i, 1] - pts[keep, 1])**2)
                if dists.min() > 500:  # at least 500m apart
                    keep.append(i)
            df = df.iloc[keep].reset_index(drop=True)

        print(f"  Detected {len(df)} drainage anomalies (LiDAR, deduplicated)")
        print(f"    Types: {df['anomaly_type'].value_counts().to_dict()}")
    else:
        print(f"  No significant drainage anomalies detected (LiDAR)")

    return df


def _merge_fault_corridors(traces, buf_m=2000):
    """Merge overlapping fault corridors to avoid redundant DEM reads,
    and split oversized corridors into manageable tiles.
    
    Returns list of (x_min, x_max, y_min, y_max, [trace, ...]) tuples
    representing non-overlapping merged corridors, each small enough
    to fit in memory.
    """
    if not traces:
        return []

    # Build initial corridors: one per trace
    corridors = []
    for trace in traces:
        coords = trace["coords"]
        x_min = coords[:, 0].min() - buf_m
        x_max = coords[:, 0].max() + buf_m
        y_min = coords[:, 1].min() - buf_m
        y_max = coords[:, 1].max() + buf_m
        corridors.append((x_min, x_max, y_min, y_max, [trace]))

    # Iteratively merge overlapping corridors
    merged = True
    while merged:
        merged = False
        new_corridors = []
        used = set()
        for i in range(len(corridors)):
            if i in used:
                continue
            x1_min, x1_max, y1_min, y1_max, t1 = corridors[i]
            for j in range(i + 1, len(corridors)):
                if j in used:
                    continue
                x2_min, x2_max, y2_min, y2_max, t2 = corridors[j]
                # Check overlap
                if (x1_min < x2_max and x1_max > x2_min and
                        y1_min < y2_max and y1_max > y2_min):
                    # Merge
                    x1_min = min(x1_min, x2_min)
                    x1_max = max(x1_max, x2_max)
                    y1_min = min(y1_min, y2_min)
                    y1_max = max(y1_max, y2_max)
                    t1 = t1 + t2
                    used.add(j)
                    merged = True
            new_corridors.append((x1_min, x1_max, y1_min, y1_max, t1))
            used.add(i)
        corridors = new_corridors

    # Split oversized corridors into tiles along the Y-axis.
    # Each tile should be at most TILE_SPAN_M in each dimension to
    # keep memory reasonable (at 2 m res, a 10km x 10km tile = 5000x5000 px).
    TILE_SPAN_M = 10000  # 10 km
    final_corridors = []
    for x_min, x_max, y_min, y_max, trace_list in corridors:
        span_x = x_max - x_min
        span_y = y_max - y_min

        # Number of tiles needed in each dimension
        n_x = max(1, int(np.ceil(span_x / TILE_SPAN_M)))
        n_y = max(1, int(np.ceil(span_y / TILE_SPAN_M)))

        if n_x == 1 and n_y == 1:
            final_corridors.append((x_min, x_max, y_min, y_max, trace_list))
            continue

        tile_x = span_x / n_x
        tile_y = span_y / n_y

        for ix in range(n_x):
            for iy in range(n_y):
                tx0 = x_min + ix * tile_x
                tx1 = min(x_min + (ix + 1) * tile_x, x_max)
                ty0 = y_min + iy * tile_y
                ty1 = min(y_min + (iy + 1) * tile_y, y_max)
                # Find traces that overlap this tile
                chunk_traces = []
                for t in trace_list:
                    tc = t["coords"]
                    if (tc[:, 0].min() - buf_m < tx1 and
                            tc[:, 0].max() + buf_m > tx0 and
                            tc[:, 1].min() - buf_m < ty1 and
                            tc[:, 1].max() + buf_m > ty0):
                        chunk_traces.append(t)
                if chunk_traces:
                    final_corridors.append((tx0, tx1, ty0, ty1, chunk_traces))

    return final_corridors



def _analyse_anomaly_clusters(labelled, n_features, dai, res, transform, traces):
    """Analyse connected anomaly clusters and classify them.
    
    Optimised version: uses find_objects for fast bounding-box extraction
    instead of creating a full boolean mask per feature.  Replaces PCA
    with a direct covariance-matrix orientation estimate.
    """
    anomalies = []

    # Fast bounding-box extraction
    slices = find_objects(labelled)
    if slices is None:
        return anomalies

    for feat_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        # Extract the local patch
        local = labelled[slc] == feat_id
        n_pixels = np.sum(local)
        if n_pixels < 3:
            continue

        # Bounding-box dimensions
        rows_feat_local, cols_feat_local = np.where(local)
        length_px = max(rows_feat_local.max() - rows_feat_local.min(),
                        cols_feat_local.max() - cols_feat_local.min())
        length_m = length_px * res

        if length_m < DRAIN_MIN_LENGTH_M:
            continue

        # Centroid (in global pixel coords)
        r_offset = slc[0].start
        c_offset = slc[1].start
        cy_px = np.mean(rows_feat_local) + r_offset
        cx_px = np.mean(cols_feat_local) + c_offset
        cx_coord, cy_coord = rasterio.transform.xy(transform, cy_px, cx_px)

        # Mean DAI (only within the local patch + mask)
        dai_local = dai[slc]
        mean_dai = np.mean(dai_local[local])

        # Orientation from covariance matrix (faster than PCA)
        if n_pixels >= 3:
            rc = np.column_stack([cols_feat_local, rows_feat_local]).astype(float)
            rc_centered = rc - rc.mean(axis=0)
            cov = (rc_centered.T @ rc_centered) / n_pixels
            # Eigenvector of largest eigenvalue
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
        else:
            angle = 0

        # Classify
        span_rows = rows_feat_local.max() - rows_feat_local.min()
        span_cols = cols_feat_local.max() - cols_feat_local.min()
        aspect_ratio = length_px / max(1, min(span_rows, span_cols))

        if aspect_ratio > 3:
            nearest_az = _nearest_fault_azimuth(cx_coord, cy_coord, traces)
            angle_diff = abs(angle - nearest_az) % 180
            if angle_diff < 30 or angle_diff > 150:
                anomaly_type = "linear_valley"
            elif 60 < angle_diff < 120:
                anomaly_type = "deflected_stream"
            else:
                anomaly_type = "linear_valley"
        else:
            anomaly_type = "depression_cluster"

        anomalies.append({
            "anomaly_id": len(anomalies),
            "x": round(cx_coord, 0),
            "y": round(cy_coord, 0),
            "length_m": round(length_m, 0),
            "aspect_ratio": round(aspect_ratio, 1),
            "orientation_deg": round(angle, 0),
            "mean_dai": round(mean_dai, 4),
            "n_pixels": int(n_pixels),
            "anomaly_type": anomaly_type,
        })

    return anomalies


def _nearest_fault_azimuth(x, y, traces):
    """Find the azimuth of the nearest fault trace to a point."""
    min_dist = float("inf")
    nearest_az = 0

    for trace in traces:
        coords = trace["coords"]
        dists = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        d = dists.min()
        if d < min_dist:
            min_dist = d
            nearest_az = trace["azimuth_deg"]

    return nearest_az


# ===================================================================
# 11. EXPRESSION QUALITY CLASSIFICATION
# ===================================================================

def annotate_scarp_with_drainage_corroboration(scarp_df, drainage_df,
                                               buffer_m=EXPR_POINT_DRAIN_BUFFER_M):
    """Annotate each scarp sample with nearby drainage anomalies."""
    if scarp_df is None:
        return pd.DataFrame()
    if len(scarp_df) == 0:
        return scarp_df.copy()

    df = scarp_df.copy()
    df["n_near_drainage_anomalies"] = 0
    df["nearest_drainage_distance_m"] = np.nan
    df["drainage_corroborates"] = False
    df["drainage_anomaly_ids"] = ""
    df["drainage_anomaly_types"] = ""

    if (drainage_df is None or len(drainage_df) == 0 or
            not {"x", "y"}.issubset(drainage_df.columns)):
        return df

    drain = drainage_df.copy().reset_index(drop=True)
    drain_xy = drain[["x", "y"]].values.astype(np.float64)
    drain_tree = cKDTree(drain_xy)
    sample_xy = df[["sample_x", "sample_y"]].values.astype(np.float64)
    nearest_dist, _ = drain_tree.query(sample_xy, k=1)
    near_lists = drain_tree.query_ball_point(sample_xy, r=buffer_m)

    for row_idx, near_idx in enumerate(near_lists):
        if len(near_idx) == 0:
            continue
        near_rows = drain.iloc[near_idx]
        df.at[df.index[row_idx], "n_near_drainage_anomalies"] = int(len(near_rows))
        df.at[df.index[row_idx], "nearest_drainage_distance_m"] = float(nearest_dist[row_idx])
        df.at[df.index[row_idx], "drainage_corroborates"] = True
        if "anomaly_id" in near_rows.columns:
            ids = "|".join(str(int(v)) for v in sorted(set(near_rows["anomaly_id"].dropna().astype(int))))
            df.at[df.index[row_idx], "drainage_anomaly_ids"] = ids
        if "anomaly_type" in near_rows.columns:
            types = "|".join(sorted(set(str(v) for v in near_rows["anomaly_type"].dropna())))
            df.at[df.index[row_idx], "drainage_anomaly_types"] = types

    return df


def classify_expression_quality_points(scarp_df, drainage_df, dem_path,
                                       deriv_dir=None, nodata=None,
                                       is_large_dem=False):
    """Classify expression quality at profile sample points."""
    n_points = 0 if scarp_df is None else len(scarp_df)
    print(f"\n[Expression Quality] Classifying {n_points} profile sample points")

    if scarp_df is None or len(scarp_df) == 0:
        return pd.DataFrame()

    dem_stem = Path(dem_path).stem
    slope_data = None
    svf_data = None
    slope_transform = None
    svf_transform = None

    if deriv_dir is not None and not is_large_dem:
        slope_data, slope_transform, _, _, _ = _load_npy_derivative(deriv_dir, dem_stem, "slope")
        svf_data, svf_transform, _, _, _ = _load_npy_derivative(deriv_dir, dem_stem, "svf")

    point_df = annotate_scarp_with_drainage_corroboration(scarp_df, drainage_df)
    results = []

    for row in point_df.itertuples(index=False):
        sample_x = float(row.sample_x)
        sample_y = float(row.sample_y)
        scarp_height = float(getattr(row, "scarp_height_m", 0.0) or 0.0)
        scarp_width = float(getattr(row, "scarp_width_m", 0.0) or 0.0)
        max_slope = float(getattr(row, "max_slope_deg", 0.0) or 0.0)
        scarp_type = getattr(row, "scarp_type", "none") or "none"
        n_drain = int(getattr(row, "n_near_drainage_anomalies", 0) or 0)

        slope_anomaly = _sample_derivative_at_point(
            slope_data, slope_transform, sample_x, sample_y, nodata=nodata)
        if slope_anomaly == 0.0 and is_large_dem:
            slope_anomaly = _compute_derivative_on_the_fly(
                dem_path, "slope", sample_x, sample_y, nodata=nodata)

        svf_val = _sample_derivative_at_point(
            svf_data, svf_transform, sample_x, sample_y, nodata=nodata)
        if svf_val == 0.0 and is_large_dem:
            svf_val = _compute_derivative_on_the_fly(
                dem_path, "svf", sample_x, sample_y, nodata=nodata)
        svf_anomaly = 1.0 - svf_val if svf_val > 0 else 0.0

        has_scarp = scarp_type != "none"
        strong_geomorphic = (max_slope >= EXPR_CLASS1_SLOPE or
                             slope_anomaly >= EXPR_CLASS1_SLOPE or
                             svf_anomaly >= EXPR_CLASS1_SVFDIFF)
        moderate_geomorphic = (max_slope >= EXPR_CLASS2_SLOPE or
                               slope_anomaly >= EXPR_CLASS2_SLOPE or
                               svf_anomaly >= EXPR_CLASS2_SVFDIFF)

        if has_scarp and scarp_height >= 1.0 and (strong_geomorphic or n_drain >= 1):
            quality_class = "Class 1"
            rationale = (f"scarp_h={scarp_height:.2f}m,type={scarp_type},"
                         f"drain={n_drain},slope={max(max_slope, slope_anomaly):.1f}deg")
        elif ((has_scarp and scarp_height >= 0.5) or moderate_geomorphic or n_drain >= 1):
            quality_class = "Class 2"
            rationale = (f"scarp_h={scarp_height:.2f}m,type={scarp_type},"
                         f"svf_anom={svf_anomaly:.3f},drain={n_drain}")
        else:
            quality_class = "Class 3"
            rationale = "no_surface_expression"

        results.append({
            "trace_id": int(row.trace_id),
            "source": row.source,
            "sample_x": round(sample_x, 3),
            "sample_y": round(sample_y, 3),
            "fault_azimuth": round(float(getattr(row, "fault_azimuth", 0.0) or 0.0), 3),
            "quality_class": quality_class,
            "scarp_height_m": round(scarp_height, 2),
            "scarp_width_m": round(scarp_width, 1),
            "max_slope_deg": round(max_slope, 2),
            "scarp_type": scarp_type,
            "offset_from_centre_m": round(float(getattr(row, "offset_from_centre_m", 0.0) or 0.0), 1),
            "n_near_drainage_anomalies": n_drain,
            "nearest_drainage_distance_m": round(float(getattr(row, "nearest_drainage_distance_m", np.nan)), 1)
                                           if pd.notna(getattr(row, "nearest_drainage_distance_m", np.nan)) else np.nan,
            "drainage_corroborates": bool(getattr(row, "drainage_corroborates", False)),
            "drainage_anomaly_ids": getattr(row, "drainage_anomaly_ids", ""),
            "drainage_anomaly_types": getattr(row, "drainage_anomaly_types", ""),
            "slope_anomaly_deg": round(slope_anomaly, 1),
            "svf_anomaly": round(svf_anomaly, 3),
            "rationale": rationale,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        print("  Point expression quality summary:")
        for cls in ["Class 1", "Class 2", "Class 3"]:
            n = (df["quality_class"] == cls).sum()
            print(f"    {cls}: {n} points")
    return df


def summarise_expression_quality_points(point_df, traces=None):
    """Aggregate point-based expression quality into the legacy trace summary."""
    if point_df is None or len(point_df) == 0:
        return pd.DataFrame(columns=[
            "trace_id", "source", "centroid_x", "centroid_y", "trace_length_m",
            "quality_class", "mean_scarp_height_m", "max_scarp_height_m",
            "structural_context", "n_profiles", "detection_rate",
            "mean_scarp_slope_deg", "n_drain_anomalies", "slope_anomaly_deg",
            "svf_anomaly", "rationale"
        ])

    trace_lookup = {t["trace_id"]: t for t in (traces or [])}
    class_rank = {"Class 1": 3, "Class 2": 2, "Class 3": 1}
    results = []

    for tid, grp in point_df.groupby("trace_id"):
        detected = grp[grp["scarp_type"] != "none"].copy()
        trace = trace_lookup.get(tid)
        source = grp["source"].iloc[0] if "source" in grp.columns else trace.get("source", "") if trace else ""
        cx = trace["centroid_x"] if trace is not None else grp["sample_x"].mean()
        cy = trace["centroid_y"] if trace is not None else grp["sample_y"].mean()
        trace_len = trace["length_m"] if trace is not None else np.nan

        detection_rate = float((grp["scarp_type"] != "none").mean()) if len(grp) > 0 else 0.0
        mean_height = float(detected["scarp_height_m"].mean()) if len(detected) > 0 else 0.0
        max_height = float(detected["scarp_height_m"].max()) if len(detected) > 0 else 0.0
        mean_slope = float(detected["max_slope_deg"].mean()) if len(detected) > 0 else 0.0
        n_discrete = int((detected["scarp_type"] == "discrete").sum()) if len(detected) > 0 else 0
        n_monocline = int((detected["scarp_type"] == "monocline").sum()) if len(detected) > 0 else 0
        if n_discrete > n_monocline:
            struct_context = "discrete_rupture"
        elif n_monocline > n_discrete:
            struct_context = "monocline"
        elif len(detected) > 0:
            struct_context = "mixed"
        else:
            struct_context = "none"

        unique_drain_ids = set()
        if "drainage_anomaly_ids" in grp.columns:
            for val in grp["drainage_anomaly_ids"].dropna():
                if val:
                    unique_drain_ids.update(v for v in str(val).split("|") if v)
        n_drain = len(unique_drain_ids)
        slope_anomaly = float(grp["slope_anomaly_deg"].max()) if "slope_anomaly_deg" in grp.columns else 0.0
        svf_anomaly = float(grp["svf_anomaly"].max()) if "svf_anomaly" in grp.columns else 0.0
        quality_class = sorted(grp["quality_class"].dropna().unique(),
                               key=lambda c: class_rank.get(c, 0), reverse=True)[0]

        if quality_class == "Class 1":
            rationale = f"best_point={quality_class},det={detection_rate:.0%},max_h={max_height:.2f}m"
        elif quality_class == "Class 2":
            rationale = f"moderate_point_support,drain={n_drain},svf={svf_anomaly:.3f}"
        else:
            rationale = "no_surface_expression"

        results.append(_class_result(
            tid, source, cx, cy, trace_len, quality_class,
            mean_height, max_height, struct_context, len(grp), detection_rate,
            mean_slope, n_drain, slope_anomaly, svf_anomaly, rationale
        ))

    return pd.DataFrame(results)

def classify_expression_quality(scarp_df, drainage_df, traces, dem_path,
                                deriv_dir=None, nodata=None, is_large_dem=False):
    """
    Classify each fault trace by expression quality.

    Class 1 (Well-constrained): Clear scarp with >1 m displacement,
        discrete or monocline, confirmed by drainage anomalies
    Class 2 (Moderately-constrained): Detectable scarp <1 m, or
        scarp present but inconsistent along strike
    Class 3 (Inferred): No detectable scarp in DEM, fault position
        based on database alone

    For small DEMs: loads slope/SVF from .npy files.
    For large DEMs (LiDAR): computes on-the-fly from DEM windows.
    """
    point_df = classify_expression_quality_points(
        scarp_df, drainage_df, dem_path,
        deriv_dir=deriv_dir, nodata=nodata, is_large_dem=is_large_dem
    )
    df = summarise_expression_quality_points(point_df, traces=traces)
    if len(df) > 0:
        print(f"  Expression quality summary:")
        for cls in ["Class 1", "Class 2", "Class 3"]:
            n = (df["quality_class"] == cls).sum()
            print(f"    {cls}: {n} traces")
    return df


def _sample_derivative_at_point(data, transform, x, y, nodata=None):
    """Sample a derivative array at a geographic point.
    Returns 0.0 if data/transform unavailable or point is out of bounds."""
    if data is None or transform is None:
        return 0.0
    col, row = ~transform * (x, y)
    row, col = int(round(row)), int(round(col))
    if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
        val = data[row, col]
        if nodata is not None and not np.isnan(nodata) and val == nodata:
            return 0.0
        if np.isnan(val) or val < -1e6:
            return 0.0
        return float(val)
    return 0.0


def _compute_derivative_on_the_fly(dem_path, deriv_type, x, y, nodata=None,
                                    window_m=500):
    """Compute a derivative value at a location on-the-fly from DEM window.
    Used for LiDAR where pre-computed arrays are too large to load fully.
    
    For high-resolution DEMs (>2 m), downsamples to 2 m before computing
    derivatives to keep runtime practical.
    """
    try:
        with rasterio.open(dem_path) as src:
            res = abs(src.res[0])
            transform = src.transform

            # Convert to pixel coordinates
            col, row = ~transform * (x, y)
            row, col = int(round(row)), int(round(col))

            # For high-res DEMs, downsample aggressively
            downsample = max(1, int(round(res / 2.0))) if res < 2 else 1
            eff_res = res * downsample

            # Read a window around the point
            half_px = int(window_m / res)
            r0 = max(0, row - half_px)
            c0 = max(0, col - half_px)
            r1 = min(src.height, row + half_px + 1)
            c1 = min(src.width, col + half_px + 1)

            win = Window(c0, r0, c1 - c0, r1 - r0)

            if downsample > 1:
                out_h = max(1, (r1 - r0) // downsample)
                out_w = max(1, (c1 - c0) // downsample)
                data = src.read(1, out_shape=(out_h, out_w),
                                 resampling=ResampEnum.average,
                                 window=win).astype(np.float32)
            else:
                data = src.read(1, window=win).astype(np.float32)

            if deriv_type == "slope":
                result = compute_slope(data, eff_res, nodata=nodata)
            elif deriv_type == "svf":
                max_radius = min(50, int(500 / eff_res))
                svf, _, _ = compute_svf_and_openness(data, eff_res, n_dirs=16,
                                                       max_radius=max_radius, nodata=nodata)
                result = svf
            elif deriv_type == "curv_profile":
                result = compute_curvature(data, eff_res, nodata=nodata, kind="profile")
            else:
                return 0.0

            # Sample at the center of the window
            if downsample > 1:
                center_r = (row - r0) // downsample
                center_c = (col - c0) // downsample
            else:
                center_r = row - r0
                center_c = col - c0
            if 0 <= center_r < result.shape[0] and 0 <= center_c < result.shape[1]:
                val = result[center_r, center_c]
                if np.isnan(val) or val < -1e6:
                    return 0.0
                return float(val)
            return 0.0
    except Exception as e:
        print(f"    [!] On-the-fly derivative computation failed: {e}")
        return 0.0


def _class_result(tid, src, cx, cy, trace_len, quality_class,
                   mean_height=0, max_height=0, struct_context="none",
                   n_profiles=0, detection_rate=0, mean_slope=0,
                   n_drain_anomalies=0, slope_anomaly=0, svf_anomaly=0,
                   rationale=""):
    """Build a classification result dict."""
    return {
        "trace_id": tid,
        "source": src,
        "centroid_x": round(cx, 0),
        "centroid_y": round(cy, 0),
        "trace_length_m": round(trace_len, 0),
        "quality_class": quality_class,
        "mean_scarp_height_m": round(mean_height, 2),
        "max_scarp_height_m": round(max_height, 2),
        "structural_context": struct_context,
        "n_profiles": n_profiles,
        "detection_rate": round(detection_rate, 2),
        "mean_scarp_slope_deg": round(mean_slope, 1),
        "n_drain_anomalies": n_drain_anomalies,
        "slope_anomaly_deg": round(slope_anomaly, 1),
        "svf_anomaly": round(svf_anomaly, 3),
        "rationale": rationale,
    }


# ===================================================================
# 12. EXPRESSION QUALITY MAP
# ===================================================================

def plot_expression_quality_map(dem_path, deriv_dir, traces, class_df,
                                 out_fig_path, nodata=None, azimuth=315,
                                 point_df=None):
    """
    Plot original fault database points on hillshade as a scatter map,
    coloured by expression quality class (A/B/C/D).

    Uses the original structural-point coordinates from the fault trace
    reconstruction (no interpolated lines).  Each point inherits the
    quality class of its parent trace:
      A = Well-constrained (Class 1)
      B = Moderately-constrained (Class 2)
      C = Inferred (Class 3)
      D = No surface expression detected

    If *point_df* (point-based expression quality) is supplied, the
    per-point sample_x / sample_y positions are scatter-plotted instead
    and coloured by each point's own quality_class, giving a finer-grained
    view of expression quality along each trace.
    """
    print(f"\n[Quality Map] Plotting expression quality classification (A/B/C/D scatter)")

    dem_stem = dem_path.stem

    # Load hillshade -- try .npy first, then compute on-the-fly
    hs_data, hs_transform, _, _, _ = _load_npy_derivative(deriv_dir, dem_stem,
                                                             f"hillshade_az{azimuth}")
    if hs_data is None or hs_transform is None:
        with rasterio.open(dem_path) as src:
            hs_transform = src.transform
            hs_bounds = src.bounds
            total_px = src.height * src.width
            if total_px > 50_000_000:
                ds_factor = max(1, int(np.sqrt(total_px / 5_000_000)))
                ov_h = src.height // ds_factor
                ov_w = src.width // ds_factor
                dem = src.read(1, out_shape=(ov_h, ov_w),
                                resampling=ResampEnum.average).astype(np.float32)
                hs_transform = src.transform * src.transform.scale(
                    float(src.width) / ov_w, float(src.height) / ov_h)
                res = abs(src.res[0]) * ds_factor
            else:
                dem = src.read(1).astype(np.float32)
                res = abs(src.res[0])
        hs_data = compute_hillshade(dem, res, azimuth=azimuth, altitude=HS_ALTITUDE,
                                      nodata=nodata)

    rows, cols = hs_data.shape
    x_min = hs_transform.c
    x_max = hs_transform.c + cols * hs_transform.a
    y_top = hs_transform.f
    y_bottom = hs_transform.f + rows * hs_transform.e
    extent = [x_min, x_max, y_bottom, y_top]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    hs_plot = np.where(np.isnan(hs_data), 0, hs_data)
    ax.imshow(hs_plot, cmap="gray", extent=extent, aspect="equal",
              vmin=0, vmax=1, origin="upper")

    # A/B/C/D class system (matches downstream scripts 07a/08)
    class_3to4 = {"Class 1": "A", "Class 2": "B", "Class 3": "C"}
    class_colors = {
        "A": "#FF0000",     # red -- well-constrained
        "B": "#FF8800",     # orange -- moderately-constrained
        "C": "#FFFF00",     # yellow -- inferred
        "D": "#888888",     # grey -- no surface expression
    }
    class_markers = {"A": "o", "B": "s", "C": "^", "D": "x"}
    class_sizes = {"A": 12, "B": 10, "C": 8, "D": 5}

    if point_df is not None and len(point_df) > 0:
        # -- Per-point scatter using point-based expression quality --
        pdf = point_df.copy()
        pdf["letter"] = pdf["quality_class"].map(class_3to4).fillna("C")
        for letter in ["A", "B", "C", "D"]:
            sub = pdf[pdf["letter"] == letter]
            if len(sub) > 0:
                ax.scatter(sub["sample_x"].values, sub["sample_y"].values,
                           c=class_colors[letter], s=class_sizes[letter],
                           marker=class_markers[letter], alpha=0.8,
                           edgecolors="black", linewidths=0.2,
                           label=f"Class {letter}")
        n_points = len(pdf)
        print(f"  Plotted {n_points:,} point-based quality samples")
    else:
        # -- Trace-level scatter using original fault database points --
        # Build trace_id -> A/B/C/D lookup
        class_lookup = {}
        for _, row in class_df.iterrows():
            qclass = row.get("quality_class", "Class 3")
            letter = class_3to4.get(qclass, "C")
            # Detect D: no surface expression at all
            if (row.get("detection_rate", 0) == 0 and
                row.get("n_drain_anomalies", 0) == 0 and
                row.get("mean_scarp_height_m", 0) == 0):
                letter = "D"
            class_lookup[row["trace_id"]] = letter

        # Collect original fault database points (trace coords) by class
        n_points = 0
        for letter in ["A", "B", "C", "D"]:
            xs, ys = [], []
            for trace in traces:
                if class_lookup.get(trace["trace_id"], "D") == letter:
                    coords = trace["coords"]
                    xs.extend(coords[:, 0].tolist())
                    ys.extend(coords[:, 1].tolist())
            if xs:
                ax.scatter(xs, ys,
                           c=class_colors[letter], s=class_sizes[letter],
                           marker=class_markers[letter], alpha=0.7,
                           edgecolors="black", linewidths=0.2,
                           label=f"Class {letter}")
                n_points += len(xs)
        print(f"  Plotted {n_points:,} original fault database points across {len(traces)} traces")

    ax.legend(loc="lower right", fontsize=10, framealpha=0.8,
              title="Expression Quality")
    ax.set_xlabel("Easting (m, EPSG:28355)", fontsize=10)
    ax.set_ylabel("Northing (m, EPSG:28355)", fontsize=10)
    ax.set_aspect('equal')
    ax.set_title(f"Fault Expression Quality Classification (A/B/C/D)  |  "
                 f"DEM: {dem_path.name}", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Quality map -> {out_fig_path.name}")


# ===================================================================
# 13. PHASE 1 AUDIT WORKFLOW
# ===================================================================

def run_phase1_audit(dems_to_process, struct_pts):
    """
    Run the complete Phase 1 Surface Expression Audit:
      1. Reconstruct fault traces from point inventory
      2. Generate fault overlay maps (desktop review)
      3. Extract perpendicular profiles across faults
      4. Analyse scarp morphology from profiles
      5. Detect drainage anomalies near faults
      6. Classify expression quality (Class 1-3)
      7. Generate output maps and CSV reports

    CHANGE 3: Now supports LiDAR via windowed reads and on-the-fly computation.
    """
    print(f"\n{'='*60}")
    print("PHASE 1: SURFACE EXPRESSION AUDIT")
    print(f"{'='*60}")

    audit_dir = safe_makedirs(OUT_DIR / "phase1_audit")

    # -- Step 1: Reconstruct fault traces --
    print(f"\n--- Step 1: Fault Trace Reconstruction ---")
    traces = reconstruct_fault_traces(struct_pts, cluster_dist_m=30, min_pts=5)

    if not traces:
        print("  [!] No fault traces could be reconstructed")
        return

    # Save trace summary
    trace_summary = pd.DataFrame([{
        "trace_id": t["trace_id"],
        "source": t["source"],
        "length_m": round(t["length_m"], 0),
        "azimuth_deg": round(t["azimuth_deg"], 1),
        "n_points": t["n_points"],
        "centroid_x": round(t["centroid_x"], 0),
        "centroid_y": round(t["centroid_y"], 0),
    } for t in traces])
    trace_csv = audit_dir / "fault_traces_summary.csv"
    trace_summary.to_csv(trace_csv, index=False)
    print(f"  Saved trace summary: {trace_csv.name}")

    # -- Process each DEM resolution --
    for dem_key, info in dems_to_process.items():
        dem_path = info["path"]
        nodata   = info["nodata"]
        deriv_dir = OUT_DIR / dem_key

        with rasterio.open(dem_path) as src:
            total_px = src.height * src.width
            res = abs(src.res[0])

        is_large = total_px > 100_000_000  # >100M pixels

        if is_large:
            print(f"\n--- Processing {dem_key} (res={res:.1f} m, LARGE DEM -- windowed) ---")
        else:
            print(f"\n--- Processing {dem_key} (res={res:.1f} m) ---")

        # -- Step 2: Fault overlay map --
        print(f"\n--- Step 2: Fault Overlay Map ---")
        overlay_path = audit_dir / f"{dem_key}_fault_overlay.png"
        if not overlay_path.exists():
            plot_fault_overlay_map(dem_path, deriv_dir, struct_pts, traces,
                                    overlay_path, nodata=nodata)

        # -- Step 3: Perpendicular profiles --
        # Works for all DEMs via _sample_dem_line (windowed reads)
        print(f"\n--- Step 3: Perpendicular Profiles ---")
        scarp_csv_path = audit_dir / f"{dem_key}_scarp_analysis.csv"

        if scarp_csv_path.exists():
            scarp_df = pd.read_csv(scarp_csv_path)
            print(f"  [Skip] Scarp analysis already exists ({len(scarp_df)} profiles)")
        else:
            with Timer(f"Profiles + Scarp analysis ({dem_key})"):
                profiles = extract_perpendicular_profiles(
                    dem_path, traces, struct_pts,
                    half_length_m=min(PROFILE_HALF_LENGTH_M, 500),
                    profile_spacing_m=max(PROFILE_SPACING_M, res * 10),
                    nodata=nodata
                )

                # -- Step 4: Scarp analysis --
                print(f"\n--- Step 4: Scarp Analysis ---")
                scarp_df = analyse_scarp_profiles(profiles)
                scarp_df.to_csv(scarp_csv_path, index=False)
                print(f"  Saved: {scarp_csv_path}")

                # -- Step 4b: Scarp profile plots --
                profile_plot_dir = safe_makedirs(audit_dir / f"{dem_key}_profiles")
                plot_scarp_profiles(profiles, scarp_df, profile_plot_dir, max_plots=20)

        # -- Step 5: Drainage anomalies --
        print(f"\n--- Step 5: Drainage Anomaly Detection ---")
        drain_csv_path = audit_dir / f"{dem_key}_drainage_anomalies.csv"

        if drain_csv_path.exists():
            drain_df = pd.read_csv(drain_csv_path)
            print(f"  [Skip] Drainage analysis already exists ({len(drain_df)} anomalies)")
        else:
            with Timer(f"Drainage anomalies ({dem_key})"):
                drain_df = detect_drainage_anomalies(
                    dem_path, deriv_dir, traces, struct_pts,
                    nodata=nodata, dem_stem=dem_path.stem,
                    is_large_dem=is_large
                )
                drain_df.to_csv(drain_csv_path, index=False)
                print(f"  Saved: {drain_csv_path}")

        required_scarp_cols = {
            "n_near_drainage_anomalies", "nearest_drainage_distance_m",
            "drainage_corroborates", "drainage_anomaly_ids",
            "drainage_anomaly_types"
        }
        if not required_scarp_cols.issubset(set(scarp_df.columns)):
            scarp_df = annotate_scarp_with_drainage_corroboration(scarp_df, drain_df)
            scarp_df.to_csv(scarp_csv_path, index=False)
            print(f"  Updated scarp analysis with drainage corroboration: {scarp_csv_path.name}")

        # -- Step 6: Expression quality classification --
        print(f"\n--- Step 6: Expression Quality Classification ---")
        class_csv_path = audit_dir / f"{dem_key}_expression_quality.csv"
        class_points_csv_path = audit_dir / f"{dem_key}_expression_quality_points.csv"

        if class_csv_path.exists() and class_points_csv_path.exists():
            class_df = pd.read_csv(class_csv_path)
            class_points_df = pd.read_csv(class_points_csv_path) if class_points_csv_path.exists() else None
            print(f"  [Skip] Classification already exists")
        else:
            class_points_df = classify_expression_quality_points(
                scarp_df, drain_df, dem_path,
                deriv_dir=deriv_dir, nodata=nodata, is_large_dem=is_large
            )
            class_points_df.to_csv(class_points_csv_path, index=False)
            print(f"  Saved: {class_points_csv_path}")

            class_df = summarise_expression_quality_points(class_points_df, traces=traces)
            class_df.to_csv(class_csv_path, index=False)
            print(f"  Saved: {class_csv_path}")

        # -- Step 7: Expression quality map (A/B/C/D scatter) --
        print(f"\n--- Step 7: Expression Quality Map ---")
        quality_map_path = audit_dir / f"{dem_key}_expression_quality_map.png"
        if not quality_map_path.exists():
            plot_expression_quality_map(dem_path, deriv_dir, traces, class_df,
                                         quality_map_path, nodata=nodata,
                                         point_df=class_points_df)

    # -- Final summary --
    print(f"\n{'='*60}")
    print("PHASE 1 AUDIT COMPLETE")
    print(f"  Output directory: {audit_dir}")
    print(f"{'='*60}")


# ===================================================================
# 14. MAIN WORKFLOW
# ===================================================================

