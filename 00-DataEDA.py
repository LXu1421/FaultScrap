# Import required libraries
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import os
import psutil
from tqdm import tqdm

# For geometric registration
import cv2
from skimage.registration import phase_cross_correlation

# For memory check
import platform

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Total memory: {mem.total / 1e9:.2f} GB, Available: {mem.available / 1e9:.2f} GB")
    return mem.available / 1e9

# Resource-aware profile for larger-memory machines
mem_total_gb = psutil.virtual_memory().total / (1024**3)
HIGH_MEMORY_MODE = mem_total_gb > 50.0
RESOURCE_PROFILE = "high-memory" if HIGH_MEMORY_MODE else "low-memory"
PATCH_SIZE_KM = 1.0 
PATCH_STRIDE_KM = 0.1
print(f"Resource profile: {RESOURCE_PROFILE} (RAM={mem_total_gb:.1f} GB)")

check_memory()

# Helper functions to load DEM data from multiple formats
import glob
from rasterio.warp import transform_bounds
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
    print("✓ pyproj is available for coordinate transformations")
except ImportError:
    PYPROJ_AVAILABLE = False
    print("⚠️  pyproj not available - coordinate transformations may not work correctly")


def load_geotiff_metadata(path, use_native_crs=False):
    """
    Load GeoTIFF metadata. Can work in native CRS or target CRS.
    Does not load full raster to save memory with large files.
    
    Args:
        path: Path to .tif file
        use_native_crs: If True, keep extent in native CRS (for .tif files in same CRS)
    Returns:
        Metadata dictionary with extent and resolution
    """
    print(f"  Loading GeoTIFF metadata: {path}...")
    
    with rasterio.open(path) as src:
        # Original CRS and bounds
        src_crs = src.crs
        src_bounds = src.bounds
        src_resolution = src.res[0]  # Native resolution
        
        print(f"    Source CRS: {src_crs}")
        print(f"    Source bounds: {src_bounds}")
        print(f"    Shape: {src.height} rows × {src.width} cols")
        
        # Check if source is in geographic coordinates (degrees)
        is_geographic = src_crs.is_geographic
        if is_geographic:
            print(f"    Native resolution: {src_resolution:.8f}° (geographic)")
            # Estimate meters at center latitude for reference
            center_lat = (src_bounds.bottom + src_bounds.top) / 2
            meters_per_degree = 111320 * np.cos(np.radians(center_lat))
            approx_resolution_m = src_resolution * meters_per_degree
            print(f"    Approximate resolution at center: ~{approx_resolution_m:.1f}m")
        else:
            print(f"    Native resolution: {src_resolution:.2f}m (projected)")
        
        # If use_native_crs, keep in source CRS (for .tif processing)
        if use_native_crs:
            extent = (src_bounds.left, src_bounds.right, 
                     src_bounds.bottom, src_bounds.top)  # xmin, xmax, ymin, ymax
            resolution = src_resolution
            working_crs = src_crs
            print(f"    Working in native CRS (degrees or meters depending on source)")
        else:
            # Transform to EPSG:28355 (GDA94 / MGA zone 55)
            # Most files are already in this CRS, covers Victoria region in meters
            target_crs = 'EPSG:28355'  # GDA94 / MGA zone 55
            target_crs_obj = CRS.from_string(target_crs)
            
            if src_crs != target_crs_obj:
                print(f"    Transforming from {src_crs} to {target_crs}...")
                print(f"    Source bounds: {src_bounds}")
                
                # Use pyproj Transformer directly for reliable transformation
                if PYPROJ_AVAILABLE:
                    try:
                        # Get EPSG codes
                        src_epsg = src_crs.to_epsg()
                        dst_epsg = target_crs_obj.to_epsg()
                        
                        if src_epsg is None or dst_epsg is None:
                            raise ValueError(f"Cannot get EPSG codes for CRS: {src_crs} -> {target_crs_obj}")
                        
                        transformer = Transformer.from_crs(
                            f"EPSG:{src_epsg}", 
                            f"EPSG:{dst_epsg}",
                            always_xy=True  # Ensure (x, y) / (lon, lat) order
                        )
                        
                        # Transform the corners
                        xmin_t, ymin_t = transformer.transform(src_bounds.left, src_bounds.bottom)
                        xmax_t, ymax_t = transformer.transform(src_bounds.right, src_bounds.top)
                        
                        print(f"    DEBUG: Pyproj transformed corners:")
                        print(f"      Lower-left: ({src_bounds.left:.6f}, {src_bounds.bottom:.6f}) → ({xmin_t:.2f}, {ymin_t:.2f})")
                        print(f"      Upper-right: ({src_bounds.right:.6f}, {src_bounds.top:.6f}) → ({xmax_t:.2f}, {ymax_t:.2f})")
                        
                        extent = (xmin_t, xmax_t, ymin_t, ymax_t)
                        working_crs = target_crs_obj
                        
                        print(f"    ✓ Transformation successful!")
                        print(f"    Transformed extent (xmin, xmax, ymin, ymax): ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}, {extent[3]:.2f})")
                        
                    except Exception as e:
                        print(f"    ERROR: Pyproj transformation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"    Falling back to native extent (will cause errors later)")
                        extent = (src_bounds.left, src_bounds.right, src_bounds.bottom, src_bounds.top)
                        working_crs = src_crs
                else:
                    # Fallback to rasterio's transform_bounds
                    print(f"    WARNING: Using rasterio's transform_bounds (may not work correctly)")
                    target_bounds = transform_bounds(
                        src_crs, target_crs_obj,
                        src_bounds.left, src_bounds.bottom,
                        src_bounds.right, src_bounds.top
                    )
                    
                    print(f"    DEBUG: transform_bounds returned: {target_bounds}")
                    extent = (target_bounds[0], target_bounds[2], 
                             target_bounds[1], target_bounds[3])
                    working_crs = target_crs_obj
                    print(f"    DEBUG: Created extent (xmin, xmax, ymin, ymax): {extent}")
                
                # For resolution: if both CRS are projected (meters), keep source resolution
                # Only recalculate if transforming from geographic to projected
                if is_geographic:
                    # Geographic to projected: need to calculate resolution in meters
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_crs, target_crs_obj,
                        src.width, src.height,
                        *src_bounds
                    )
                    resolution = dst_transform[0]  # Pixel width in target CRS
                    print(f"    Estimated resolution in target CRS: {resolution:.2f}m")
                else:
                    # Both projected: preserve source resolution (both in meters)
                    resolution = src_resolution
                    print(f"    Keeping source resolution: {resolution:.2f}m (both CRS in meters)")
            else:
                extent = (src_bounds.left, src_bounds.right, 
                         src_bounds.bottom, src_bounds.top)
                resolution = src_resolution
                working_crs = src_crs
        
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        
        unit = "°" if is_geographic and use_native_crs else "m"
        scale = 111.32 if (is_geographic and use_native_crs) else 1000  # km conversion
        
        print(f"    Extent:")
        print(f"      X: [{extent[0]:.6f}, {extent[1]:.6f}] ({x_range*scale:.2f} km)")
        print(f"      Y: [{extent[2]:.6f}, {extent[3]:.6f}] ({y_range*scale:.2f} km)")
        
        return {
            'path': path,
            'extent': extent,
            'resolution': resolution,
            'src_crs': src_crs,
            'working_crs': working_crs,
            'is_geographic': is_geographic,
            'shape': (src.height, src.width)
        }


def extract_geotiff_window(tif_path, x_min, x_max, y_min, y_max, 
                           working_crs=None, target_resolution=None, verbose=False,
                           nodata_threshold=-100):
    """
    Extract a window from a GeoTIFF file. Handles reprojection if needed.
    
    Args:
        tif_path: Path to .tif file
        x_min, x_max, y_min, y_max: Window bounds in working CRS
        working_crs: CRS of the bounds (if None, uses source CRS)
        target_resolution: Output resolution (if None, uses source resolution)
        verbose: Print debug information
        nodata_threshold: Values below this are treated as nodata (default: -100)
    Returns:
        data: Point cloud array [x, y, z]
        resolution: Effective resolution
    """
    with rasterio.open(tif_path) as src:
        # Determine working CRS
        if working_crs is None:
            working_crs = src.crs
        elif isinstance(working_crs, str):
            working_crs = CRS.from_string(working_crs)
        
        if verbose:
            print(f"    Source CRS: {src.crs}, Working CRS: {working_crs}")
            print(f"    Source bounds: {src.bounds}")
            print(f"    Requested bounds (working CRS): ({x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f})")
        
        # Transform window bounds to source CRS if needed
        if src.crs != working_crs:
            if PYPROJ_AVAILABLE:
                try:
                    # Get EPSG codes
                    working_epsg = working_crs.to_epsg() if hasattr(working_crs, 'to_epsg') else None
                    src_epsg = src.crs.to_epsg()
                    
                    if working_epsg is None or src_epsg is None:
                        raise ValueError("Cannot get EPSG codes for transformation")
                    
                    transformer = Transformer.from_crs(
                        f"EPSG:{working_epsg}",
                        f"EPSG:{src_epsg}",
                        always_xy=True
                    )
                    xmin_src, ymin_src = transformer.transform(x_min, y_min)
                    xmax_src, ymax_src = transformer.transform(x_max, y_max)
                    src_bounds = (xmin_src, ymin_src, xmax_src, ymax_src)
                except Exception as e:
                    if verbose:
                        print(f"    ERROR: Pyproj transformation failed: {e}")
                    return np.array([]), src.res[0]
            else:
                src_bounds = transform_bounds(
                    working_crs, src.crs,
                    x_min, y_min, x_max, y_max
                )
            if verbose:
                print(f"    Transformed to source CRS: {src_bounds}")
        else:
            src_bounds = (x_min, y_min, x_max, y_max)
        
        # Check if window is within source bounds
        if (src_bounds[0] >= src.bounds.right or src_bounds[2] <= src.bounds.left or
            src_bounds[1] >= src.bounds.top or src_bounds[3] <= src.bounds.bottom):
            if verbose:
                print(f"    WARNING: Window completely outside source bounds!")
            return np.array([]), src.res[0]
        
        # Read window from source
        try:
            window = rasterio.windows.from_bounds(
                src_bounds[0], src_bounds[1], src_bounds[2], src_bounds[3],
                transform=src.transform
            )
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to create window: {e}")
            return np.array([]), src.res[0]
        
        # Check if window is valid (not outside bounds or empty)
        if window.width <= 0 or window.height <= 0:
            if verbose:
                print(f"    WARNING: Invalid window dimensions {window.height}×{window.width}")
            return np.array([]), src.res[0]
        
        # Read data
        try:
            data_array = src.read(1, window=window, masked=True)
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to read window: {e}")
            return np.array([]), src.res[0]
        
        # Check if read was successful
        if data_array is None or data_array.size == 0:
            if verbose:
                print(f"    WARNING: Empty data array read")
            return np.array([]), src.res[0]
        
        if verbose:
            print(f"    Read array: {data_array.shape}, valid pixels: {np.sum(~np.ma.is_masked(data_array))}")
        
        # Convert to point cloud format in working CRS
        if src.crs == working_crs:
            # No reprojection needed
            resolution = src.res[0]
            points = []
            for i in range(data_array.shape[0]):
                for j in range(data_array.shape[1]):
                    if not np.ma.is_masked(data_array[i, j]):
                        z = data_array[i, j]
                        # Filter out nodata values (e.g., -999, -9999)
                        if z > nodata_threshold and not np.isnan(z) and not np.isinf(z):
                            x = x_min + (j + 0.5) * resolution
                            y = y_max - (i + 0.5) * resolution
                            points.append([x, y, z])
        else:
            # Reprojection needed
            window_transform = src.window_transform(window)
            
            # Calculate output dimensions
            if target_resolution is None:
                dst_transform, _, _ = calculate_default_transform(
                    src.crs, working_crs,
                    src.width, src.height,
                    *src.bounds
                )
                target_resolution = dst_transform[0]
            
            out_width = int(np.ceil((x_max - x_min) / target_resolution))
            out_height = int(np.ceil((y_max - y_min) / target_resolution))
            
            # Safety check for valid dimensions
            if out_width <= 0 or out_height <= 0:
                print(f"    WARNING: Invalid output dimensions {out_height}×{out_width}")
                print(f"    Window: ({x_min}, {y_min}, {x_max}, {y_max})")
                print(f"    Size: {x_max-x_min} × {y_max-y_min}")
                print(f"    Resolution: {target_resolution}")
                return np.array([]), target_resolution
            
            # Check if window read is empty
            if data_array.shape[0] == 0 or data_array.shape[1] == 0:
                print(f"    WARNING: Empty window read from source")
                return np.array([]), target_resolution
            
            # Prepare output array
            out_array = np.empty((out_height, out_width), dtype=data_array.dtype)
            
            # Define output transform
            out_transform = rasterio.transform.from_bounds(
                x_min, y_min, x_max, y_max,
                out_width, out_height
            )
            
            # Reproject
            reproject(
                source=data_array,
                destination=out_array,
                src_transform=window_transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=working_crs,
                resampling=Resampling.bilinear
            )
            
            data_array = np.ma.masked_invalid(out_array)
            resolution = target_resolution
            
            # Convert to points
            points = []
            for i in range(data_array.shape[0]):
                for j in range(data_array.shape[1]):
                    if not np.ma.is_masked(data_array[i, j]):
                        z = data_array[i, j]
                        # Filter out nodata values (e.g., -999, -9999)
                        if z > nodata_threshold and not np.isnan(z) and not np.isinf(z):
                            x = x_min + (j + 0.5) * resolution
                            y = y_max - (i + 0.5) * resolution
                            points.append([x, y, z])
        
        return np.array(points), resolution


def load_dem_metadata(path, use_native_crs=False):
    """
    Load DEM data and estimate actual resolution based on point density.
    Supports .txt (point cloud), .asc (ASCII grid), and .tif (GeoTIFF) formats.
    Returns metadata without creating full grid to avoid memory issues.
    
    Args:
        use_native_crs: For .tif files, work in native CRS instead of transforming
    """
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext == '.tif' or file_ext == '.tiff':
        # Load GeoTIFF format (new support for large files)
        return load_geotiff_metadata(path, use_native_crs=use_native_crs)
    
    elif file_ext == '.asc':
        # Load ASCII grid format
        data, resolution = load_asc_file(path)
    
    elif file_ext == '.txt':
        # Load text point cloud format
        print(f"  Loading TXT file: {path}...")
        data = np.loadtxt(path, delimiter=';', skiprows=1)
        
        # Estimate resolution from point density
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        n_points = len(data)
        
        area_m2 = x_range * y_range
        points_per_m2 = n_points / area_m2
        avg_spacing = 1.0 / np.sqrt(points_per_m2)
        
        # Check unique coordinates for refined estimate
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        
        if len(unique_x) > 1:
            x_diffs = np.diff(np.sort(unique_x))
            x_spacing = np.median(x_diffs[x_diffs > avg_spacing * 0.5])
        else:
            x_spacing = avg_spacing
        
        if len(unique_y) > 1:
            y_diffs = np.diff(np.sort(unique_y))
            y_spacing = np.median(y_diffs[y_diffs > avg_spacing * 0.5])
        else:
            y_spacing = avg_spacing
        
        resolution = max(x_spacing, y_spacing, avg_spacing)
        print(f"    Estimated resolution: {resolution:.2f}m")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .txt, .asc, or .tif")
    
    # For non-GeoTIFF formats, extract extent from data
    if file_ext != '.tif' and file_ext != '.tiff':
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        z_values = data[:, 2]
        
        n_points = len(z_values)
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        extent = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())
        
        print(f"    Total valid points: {n_points:,}")
        print(f"    X range: {x_coords.min():.2f} to {x_coords.max():.2f} ({x_range/1000:.2f} km)")
        print(f"    Y range: {y_coords.min():.2f} to {y_coords.max():.2f} ({y_range/1000:.2f} km)")
        print(f"    Z range: [{z_values.min():.2f}, {z_values.max():.2f}]")
        print(f"    Resolution: {resolution:.2f}m")
        print(f"    Expected grid dimensions: {int(y_range/resolution)} × {int(x_range/resolution)}")
        
        return {
            'data': data,
            'extent': extent,
            'resolution': resolution,
            'n_points': n_points,
            'working_crs': CRS.from_epsg(28355),  # Assume MGA zone 55 for .asc/.txt
            'is_geographic': False
        }


# Expected resolutions for validation
expected_resolutions = {
    'DEM1s': 30,   # 1 arc-second ≈ 30m
    'DEM5m': 0.5,  # 0.5 meter resolution
}

# Scan for DEM files in multiple locations
dem_files = {}
tif_dirs = {
    'DEM': [
        r'data',
        'data'
    ],
}

print("Scanning for DEM files...")
print("=" * 60)

for key, paths in tif_dirs.items():
    # Ensure paths is iterable (tuple or list)
    if isinstance(paths, (str, bytes)):
        paths = [paths]

    found_dir = None

    # Try each possible directory until one exists
    for directory in paths:
        if os.path.exists(directory):
            found_dir = directory
            break

    if found_dir is None:
        print(f"[{key}] No valid directory found in: {paths}")
        continue

    # Scan for .tif files in the resolved directory
    tif_files = glob.glob(os.path.join(found_dir, '*.tif'))

    if tif_files:
        print(f"[{key}] Found {len(tif_files)} .tif file(s) in {found_dir}")
        for tif in tif_files:
            file_key = f"{key}_{os.path.basename(tif)}"
            dem_files[file_key] = tif
            print(f"  - {os.path.basename(tif)}")
    else:
        print(f"[{key}] Directory exists but contains no .tif files: {found_dir}")


if len(dem_files) == 0:
    raise FileNotFoundError("Could not find any DEM files in specified directories")

print(f"\n{'='*60}")
print(f"Found {len(dem_files)} DEM file(s) to process")
print('='*60)

# First pass: check CRS of all files to determine strategy
print("\nChecking coordinate systems...")
file_crs_list = []
for path in dem_files.values():
    file_ext = os.path.splitext(path)[1].lower()
    if file_ext in ['.tif', '.tiff']:
        with rasterio.open(path) as src:
            file_crs_list.append((path, src.crs))
            print(f"  {os.path.basename(path)}: {src.crs}")

# Check if all files have the same CRS type (all geographic or all projected)
all_same_crs = len(set(str(crs) for _, crs in file_crs_list)) == 1
all_projected = all(not crs.is_geographic for _, crs in file_crs_list)
all_geographic = all(crs.is_geographic for _, crs in file_crs_list)

# Check if majority are in the same CRS (for complementary datasets)
from collections import Counter
crs_counts = Counter(str(crs) for _, crs in file_crs_list)
most_common_crs, most_common_count = crs_counts.most_common(1)[0]

# Determine strategy
if all_same_crs and all_projected:
    # All files in same projected CRS - can work natively
    use_native_crs = True
    print("\nStrategy: All files in same projected CRS - will work in native CRS (meters)")
elif all_same_crs and all_geographic:
    # All files in same geographic CRS - can work natively
    use_native_crs = True
    print("\nStrategy: All files in same geographic CRS - will work in native CRS (degrees)")
elif most_common_count >= 2 and not all_same_crs:
    # Majority in one CRS - use that CRS, ignore others
    use_native_crs = True
    print(f"\nStrategy: Majority of files ({most_common_count}/{len(file_crs_list)}) in {most_common_crs}")
    print(f"           Will work in native CRS and use only files in the majority CRS")
    print(f"           Other files will be skipped (likely covering different geographic regions)")
else:
    # Mixed CRS - must transform to common CRS
    use_native_crs = False
    print("\nStrategy: Mixed coordinate systems detected - will transform to common CRS (GDA94 / MGA zone 55 / EPSG:28355)")
    print("           All extents will be in meters for proper comparison")

# Load metadata for all DEMs
dem_metadata = {}

# Determine target CRS if using native CRS with majority strategy
target_crs_for_filter = None
if use_native_crs and len(file_crs_list) > 0:
    from collections import Counter
    crs_counts = Counter(str(crs) for _, crs in file_crs_list)
    most_common_crs_str, most_common_count = crs_counts.most_common(1)[0]
    if most_common_count >= 2 and len(set(str(crs) for _, crs in file_crs_list)) > 1:
        # Using majority strategy - filter files
        target_crs_for_filter = most_common_crs_str
        print(f"  Will only load files in CRS: {target_crs_for_filter}\n")

for key, path in dem_files.items():
    print(f"\n{'='*60}")
    print(f"Loading metadata for {key}")
    print('='*60)
    
    # Check if we should skip this file (different CRS than majority)
    if target_crs_for_filter is not None:
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext in ['.tif', '.tiff']:
            with rasterio.open(path) as src:
                if str(src.crs) != target_crs_for_filter:
                    print(f"  Skipping: File is in {src.crs}, but majority CRS is {target_crs_for_filter}")
                    print(f"  (This file likely covers a different geographic region)")
                    continue
    
    try:
        metadata = load_dem_metadata(path, use_native_crs=use_native_crs)
        
        # Determine expected resolution
        if '1s' in key.lower() or '1_second' in key.lower():
            expected_res = expected_resolutions['DEM1s']
        elif '5m' in key.lower():
            expected_res = expected_resolutions['DEM5m']
        else:
            expected_res = metadata['resolution']
        
        # For geographic coordinates, estimate resolution in meters
        if metadata.get('is_geographic', False):
            # Estimate at center latitude
            extent = metadata['extent']
            center_lat = (extent[2] + extent[3]) / 2
            meters_per_degree = 111320 * np.cos(np.radians(center_lat))
            approx_res_m = metadata['resolution'] * meters_per_degree
            
            # Override with expected resolution for processing
            if abs(approx_res_m - expected_res) / expected_res > 0.5:
                print(f"  INFO: Approximate resolution {approx_res_m:.1f}m differs from expected {expected_res:.2f}m")
                print(f"  Using expected resolution for processing: {expected_res:.2f}m")
            metadata['resolution_meters'] = expected_res
        else:
            metadata['resolution_meters'] = metadata['resolution']
        
        dem_metadata[key] = metadata
    except Exception as e:
        print(f"  ERROR loading {key}: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Skipping this file...")

if len(dem_metadata) == 0:
    raise ValueError("No DEM files could be loaded successfully")

print(f"\n{'='*60}")
print(f"DEM metadata loaded for {len(dem_metadata)} file(s)!")
print('='*60)

# Print individual extents for debugging
print(f"\nIndividual DEM extents:")
for key, metadata in dem_metadata.items():
    extent = metadata['extent']
    is_geo = metadata.get('is_geographic', False)
    unit = "°" if is_geo else "m"
    print(f"  {key}:")
    print(f"    X: [{extent[0]:.6f}, {extent[1]:.6f}] {unit}")
    print(f"    Y: [{extent[2]:.6f}, {extent[3]:.6f}] {unit}")

# Find common extent across all DEMs
all_extents = [m['extent'] for m in dem_metadata.values()]
common_extent = (
    max([e[0] for e in all_extents]),  # xmin
    min([e[1] for e in all_extents]),  # xmax
    max([e[2] for e in all_extents]),  # ymin
    min([e[3] for e in all_extents])   # ymax
)

# Check if common extent is valid
if common_extent[1] <= common_extent[0] or common_extent[3] <= common_extent[2]:
    print(f"\n{'='*60}")
    print("ERROR: DEMs do not overlap!")
    print('='*60)
    print(f"Common extent would be:")
    print(f"  X: [{common_extent[0]:.6f}, {common_extent[1]:.6f}] (range: {common_extent[1]-common_extent[0]:.6f})")
    print(f"  Y: [{common_extent[2]:.6f}, {common_extent[3]:.6f}] (range: {common_extent[3]-common_extent[2]:.6f})")
    print(f"\nPossible issues:")
    print(f"  1. Files are in different geographic regions")
    print(f"  2. Wrong directories specified")
    raise ValueError("DEMs do not overlap - cannot find common extent")

# Determine units and scale for display
is_geo = list(dem_metadata.values())[0].get('is_geographic', False)
unit = "°" if is_geo else "m"
km_scale = 111.32 if is_geo else 1000

print(f"\nCommon extent (intersection of all DEMs):")
print(f"  X: [{common_extent[0]:.6f}, {common_extent[1]:.6f}] {unit} ({(common_extent[1]-common_extent[0])*km_scale:.2f} km)")
print(f"  Y: [{common_extent[2]:.6f}, {common_extent[3]:.6f}] {unit} ({(common_extent[3]-common_extent[2])*km_scale:.2f} km)")

# CRITICAL CHECK: Verify extents are actually in the expected CRS
if not use_native_crs:
    # Should be in meters (EPSG:28355 - MGA zone 55)
    if common_extent[0] < 100000 or common_extent[1] < 100000:
        print(f"\n⚠️  ERROR: Extents appear to be in degrees but should be in meters!")
        print(f"   This indicates coordinate transformation failed.")
        print(f"   Common extent X: [{common_extent[0]:.2f}, {common_extent[1]:.2f}]")
        print(f"   Expected: X in range [200000, 900000] for EPSG:28355 (MGA zone 55)")
        raise ValueError("Coordinate transformation failed - extents still in degrees!")

# Organize by resolution
lr_dems = {k: v for k, v in dem_metadata.items() if v.get('resolution_meters', v['resolution']) >= 20}
hr_dems = {k: v for k, v in dem_metadata.items() if v.get('resolution_meters', v['resolution']) < 20}

# CRITICAL: Verify all extents are in same units
extent_check_x = [m['extent'][0] for m in dem_metadata.values()]
if not use_native_crs:
    # Should all be in meters (large numbers)
    if any(x < 1000 for x in extent_check_x):
        print(f"\n⚠️  CRITICAL ERROR: Some extents still in degrees after transformation!")
        for key, meta in dem_metadata.items():
            ext = meta['extent']
            print(f"  {key}: X=[{ext[0]:.2f}, {ext[1]:.2f}] ({meta['working_crs']})")
        print(f"\n  ACTION REQUIRED: Re-run this cell (Cell 4) to reload metadata with proper transformation!")
        raise ValueError("Extents in wrong coordinate system - re-run metadata loading!")

print(f"\nResolution summary:")
if len(lr_dems) > 0:
    first_lr = list(lr_dems.values())[0]
    lr_res = first_lr.get('resolution_meters', first_lr['resolution'])
    print(f"  Low-res DEMs (1s): {len(lr_dems)} file(s), ~{lr_res:.1f}m resolution")
else:
    print(f"  Low-res DEMs (1s): 0 files")
    
if len(hr_dems) > 0:
    first_hr = list(hr_dems.values())[0]
    hr_res = first_hr.get('resolution_meters', first_hr['resolution'])
    print(f"  High-res DEMs (>10m): {len(hr_dems)} file(s), ~{hr_res:.1f}m resolution")
else:
    print(f"  High-res DEMs (>10m): 0 files")
    

# =============================================================================
# Diagnose actual coverage of ALL HR DEMs
# =============================================================================

print("\n" + "=" * 80)
print("HR DATA COVERAGE SUMMARY")
print("=" * 80)

hr_actual_extents = {}

for hr_key in hr_dems:

    hr_path = dem_metadata[hr_key]['path']

    print("\n" + "-" * 80)
    print(f"HR DEM : {hr_key}")
    print(f"Path   : {hr_path}")

    with rasterio.open(hr_path) as src:

        print("\nMetadata")
        print(f"  CRS        : {src.crs}")
        print(f"  Shape      : {src.height:,} x {src.width:,}")
        print(f"  Resolution : {src.res}")
        print(f"  Bounds     : {src.bounds}")
        print(f"  Nodata     : {src.nodata}")

        print("\nScanning valid-data footprint...")

        valid_rows = []
        valid_cols = []

        block_size = 1024

        # ------------------------------------------------------------
        # Scan rows
        # ------------------------------------------------------------
        for row in tqdm(
            range(0, src.height, block_size),
            leave=False,
            desc=f"{hr_key} rows"
        ):

            h = min(block_size, src.height - row)

            block = src.read(
                1,
                window=rasterio.windows.Window(
                    0,
                    row,
                    src.width,
                    h
                ),
                masked=True
            )

            mask = ~np.ma.getmaskarray(block)

            if np.any(mask):
                local_rows = np.where(mask.any(axis=1))[0]
                valid_rows.extend(row + local_rows)

        # ------------------------------------------------------------
        # Scan columns
        # ------------------------------------------------------------
        for col in tqdm(
            range(0, src.width, block_size),
            leave=False,
            desc=f"{hr_key} cols"
        ):

            w = min(block_size, src.width - col)

            block = src.read(
                1,
                window=rasterio.windows.Window(
                    col,
                    0,
                    w,
                    src.height
                ),
                masked=True
            )

            mask = ~np.ma.getmaskarray(block)

            if np.any(mask):
                local_cols = np.where(mask.any(axis=0))[0]
                valid_cols.extend(col + local_cols)

        if len(valid_rows) == 0 or len(valid_cols) == 0:

            print("WARNING: No valid pixels found.")
            continue

        row_min = min(valid_rows)
        row_max = max(valid_rows)

        col_min = min(valid_cols)
        col_max = max(valid_cols)

        x_left, y_top = rasterio.transform.xy(
            src.transform,
            row_min,
            col_min,
            offset='center'
        )

        x_right, y_bottom = rasterio.transform.xy(
            src.transform,
            row_max,
            col_max,
            offset='center'
        )

        actual_extent = (
            min(x_left, x_right),
            max(x_left, x_right),
            min(y_bottom, y_top),
            max(y_bottom, y_top)
        )

        hr_actual_extents[hr_key] = actual_extent

        bbox_area = (
            (src.bounds.right - src.bounds.left)
            * (src.bounds.top - src.bounds.bottom)
        )

        valid_area = (
            (actual_extent[1] - actual_extent[0])
            * (actual_extent[3] - actual_extent[2])
        )

        coverage_pct = (
            valid_area / bbox_area * 100
            if bbox_area > 0 else 0
        )

        print("\nActual valid-data extent")
        print(f"  X : [{actual_extent[0]:.2f}, {actual_extent[1]:.2f}]")
        print(f"  Y : [{actual_extent[2]:.2f}, {actual_extent[3]:.2f}]")
        print(f"  Bounding-box coverage ≈ {coverage_pct:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"LR datasets : {len(lr_dems)}")
print(f"HR datasets : {len(hr_dems)}")

print("\nPatch extraction strategy")
print("  • Every patch is tested against every HR dataset.")
print("  • Coverage is evaluated independently.")
print("  • Sparse or flat HR patches are rejected.")
print("  • The HR dataset with the highest valid coverage is selected.")
print("  • Sub-5m LiDAR datasets are Lanczos-resampled to 5m before use.")

print("=" * 80)