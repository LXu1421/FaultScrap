"""
ClipHelper.py -- Clip a raster to the extent of a reference raster with an optional buffer.

Usage:
  Edit the configuration section below, then run:
    python ClipHelper.py

Configuration:
  - input_raster:   Path to the large source raster to clip
  - reference_raster: Path to the reference raster whose extent defines the clip area
  - output_raster:  Path for the clipped output file
  - buffer_m:       Buffer in metres to add around the reference extent (default: 1000)
"""

from pathlib import Path
import psutil
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds


# ===================================================================
# CONFIGURATION
# ===================================================================

input_raster = "data/3secSRTM_DEM.tif"
reference_raster = "data/axedale_2020nov01_dem50cm_v10cm_epsg7855.tif"
output_raster = "data/srtm3s_axe.tif"

# Buffer in metres to add around the reference extent before clipping.
# This ensures the clipped area extends slightly beyond the reference,
# which avoids edge effects when reprojecting or resampling later.
buffer_m = 1000  # 1 km buffer


# ===================================================================
# MEMORY CHECK
# ===================================================================

def check_memory():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    print(f"Available RAM: {available_gb:.2f} GB")

    if available_gb < 8:
        print("  [!] Low RAM available for raster processing.")

check_memory()


# ===================================================================
# GET REFERENCE BOUNDS (with buffer)
# ===================================================================

with rasterio.open(reference_raster) as ref:
    ref_bounds = ref.bounds
    ref_crs = ref.crs

    # Apply buffer in the reference CRS coordinate system (metres for projected CRS)
    buffered_bounds = (
        ref_bounds.left   - buffer_m,
        ref_bounds.bottom - buffer_m,
        ref_bounds.right  + buffer_m,
        ref_bounds.top    + buffer_m,
    )

    print(f"Reference extent: E=[{ref_bounds.left:.0f}, {ref_bounds.right:.0f}], "
          f"N=[{ref_bounds.bottom:.0f}, {ref_bounds.top:.0f}]")
    print(f"Buffered extent:  E=[{buffered_bounds[0]:.0f}, {buffered_bounds[2]:.0f}], "
          f"N=[{buffered_bounds[1]:.0f}, {buffered_bounds[3]:.0f}] ({buffer_m}m buffer)")


# ===================================================================
# CLIP RASTER
# ===================================================================

with rasterio.open(input_raster) as src:

    # Transform buffered reference extent into source CRS
    bounds = transform_bounds(
        ref_crs,
        src.crs,
        *buffered_bounds,
        densify_pts=21
    )

    print(f"Buffered bounds in source CRS ({src.crs}): "
          f"left={bounds[0]:.1f}, bottom={bounds[1]:.1f}, right={bounds[2]:.1f}, top={bounds[3]:.1f}")

    # Build clipping window in source pixel coordinates
    clip_window = from_bounds(*bounds, transform=src.transform)

    # Clamp to raster bounds
    full_window = Window(0, 0, src.width, src.height)
    clip_window = clip_window.intersection(full_window)

    clip_window = clip_window.round_offsets().round_lengths()

    print(f"Clip window: {clip_window}")
    print(f"Clipped size: {int(clip_window.width)} x {int(clip_window.height)} px")

    # Output metadata
    profile = src.profile.copy()
    profile.update({
        "height": int(clip_window.height),
        "width": int(clip_window.width),
        "transform": src.window_transform(clip_window)
    })

    block_size = src.block_shapes[0][0]  # assume square-ish blocks

    with rasterio.open(output_raster, "w", **profile) as dst:

        # Iterate ONLY over clipped region
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

                # Read only this small chunk
                data = src.read(window=win)

                # Write to output at same window position
                # Convert source window to destination window
                dst_window = Window(
                    col_off=win.col_off - clip_window.col_off,
                    row_off=win.row_off - clip_window.row_off,
                    width=win.width,
                    height=win.height
                )

                dst.write(data, window=dst_window)

print(f"Clipping complete. Output: {output_raster}")
