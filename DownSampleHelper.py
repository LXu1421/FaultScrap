
def ensure_smoke_dem(dem_source, target_res=2.0):
    """Create a downsampled DEM if it doesn't exist (BigTIFF enabled)."""
    from pathlib import Path
    import rasterio
    from rasterio.warp import reproject, Resampling

    orig_path = Path(f"data/derivatives/{dem_source}.tif")
    smoke_path = Path(f"data/derivatives/{dem_source}_{target_res}m_smoke.tif")

    if not smoke_path.exists():
        print(f"[SMOKE] Creating {smoke_path.name}...")

        with rasterio.open(orig_path) as src:
            # Compute scale factor (e.g., 0.5m → 2m = scale 4)
            scale = target_res / src.res[0]

            new_height = int(src.height / scale)
            new_width = int(src.width / scale)

            # Build new transform correctly
            new_transform = src.transform * src.transform.scale(scale, scale)

            # Build new profile
            profile = src.profile.copy()
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "dtype": "float32",
                "BIGTIFF": "YES",          # <--- IMPORTANT
                "tiled": True,             # <--- recommended for large rasters
                "compress": "deflate",     # <--- safe + fast
                "blockxsize": 512,
                "blockysize": 512,
            })

            # Create the output raster
            with rasterio.open(smoke_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                )

        print(f"[SMOKE] Created {smoke_path.name} ({new_width}×{new_height})")

    return smoke_path

input_raster = r"data/derivatives/lidar_0_5m_epsg28355.tif"
output_raster = r"data/derivatives/lidar_2m_epsg28355.tif"

# Create the smoke DEM (2m)
smoke_path = ensure_smoke_dem("lidar_0_5m_epsg28355", target_res=10.0)
print("Smoke DEM:", smoke_path)
