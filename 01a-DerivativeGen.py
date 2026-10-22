#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01a-DerivativeGen.py -- Multi-Resolution DEM Derivative Generation

Steps:
  1. Load & reproject DEMs to common CRS (EPSG:28355)
  1b. Clip all DEMs to common extent (aligned with LiDAR)
  2. Generate derivatives per DEM:
     - Multi-azimuth hillshades
     - SVF / Openness
     - Slope / Curvature
     - TPI at multiple radii
     - PCA composite from hillshades
     (For LiDAR: tiled processing with multiprocessing)
  2b. Cleanup transient derivatives to save disk space

Run after ClipHelper.py has prepared the source DEM files.
Output goes to data/derivatives/.
"""

from derivative_common import *


def main():
    """Step 1-2: Reproject, align, and generate all DEM derivatives."""

    # ---- TEST MODE: only process SRTM 3s, skip all other DEMs ----
    TEST_ONLY_SRTM3S = False

    # -- Resource detection (GPU auto-detected) --
    res_obj = get_resources()
    safe_makedirs(OUT_DIR)
    tiles_dir = safe_makedirs(OUT_DIR / "tiles")

    # -- Step 1: Identify which DEMs to process --
    dems_to_process = {}

    # SRTM 3s (~90 m)
    if SRTM3S_PATH.exists():
        print(f"\n{'-'*60}")
        print(f"SRTM 3-arc-second: {SRTM3S_PATH.name}")
        rp, nd = load_reprojected("srtm3s", SRTM3S_PATH, OUT_DIR)
        dems_to_process["srtm3s"] = {"path": rp, "nodata": nd, "label": "SRTM 3s (~90 m)"}
    else:
        print(f"[!] SRTM 3s not found: {SRTM3S_PATH}")

    # SRTM 1s (~30 m)
    if not TEST_ONLY_SRTM3S and SRTM_PATH.exists():
        print(f"\n{'-'*60}")
        print(f"SRTM 1-arc-second: {SRTM_PATH.name}")
        rp, nd = load_reprojected("srtm1s", SRTM_PATH, OUT_DIR)
        dems_to_process["srtm1s"] = {"path": rp, "nodata": nd, "label": "SRTM 1s (~30 m)"}
    elif not TEST_ONLY_SRTM3S:
        print(f"[!] SRTM not found: {SRTM_PATH}")

    # DEM 10 m
    if not TEST_ONLY_SRTM3S and DEM10M_PATH.exists():
        print(f"\n{'-'*60}")
        print(f"GA 10 m DEM: {DEM10M_PATH.name}")
        rp, nd = load_reprojected("dem10m", DEM10M_PATH, OUT_DIR)
        dems_to_process["dem10m"] = {"path": rp, "nodata": nd, "label": "GA 10 m DEM"}
    elif not TEST_ONLY_SRTM3S:
        print(f"[!] DEM10m not found: {DEM10M_PATH}")

    # LiDAR 0.5 m -- SKIPPED (70 GB reprojection too slow for this run)
    if not TEST_ONLY_SRTM3S and LIDAR_PATH.exists():
         rp, nd = load_reprojected("lidar_0_5m", LIDAR_PATH, OUT_DIR)
         dems_to_process["lidar_0_5m"] = {"path": rp, "nodata": LIDAR_NODATA,
                                            "label": "LiDAR 0.5 m"}

    print(f"\n{'='*60}")
    print(f"DEMs ready for derivative generation: {len(dems_to_process)}")
    for k, v in dems_to_process.items():
        print(f"  {k}: {v['path'].name}  (nodata={v['nodata']})")
    print(f"{'='*60}")

    # -- Step 1b: Clip all DEMs to common extent (aligned with reference) --
    # Without LiDAR, use srtm3s as reference
    ref_key = "srtm3s" if TEST_ONLY_SRTM3S else "srtm3s"
    dems_to_process = clip_to_common_extent(dems_to_process, OUT_DIR, reference_key=ref_key)

    print(f"\n{'='*60}")
    print(f"DEMs after alignment: {len(dems_to_process)}")
    for k, v in dems_to_process.items():
        print(f"  {k}: {v['path'].name}  (nodata={v['nodata']})")
    print(f"{'='*60}")

    # -- Step 2: Generate derivatives for each DEM --
    for dem_key, info in dems_to_process.items():
        dem_path = info["path"]
        nodata   = info["nodata"]
        label    = info["label"]
        deriv_dir = safe_makedirs(OUT_DIR / dem_key)

        print(f"\n{'='*60}")
        print(f"Processing: {label}  ->  {deriv_dir}")
        print(f"{'='*60}")

        with rasterio.open(dem_path) as src:
            total_px = src.height * src.width
            res_m = abs(src.res[0])

        is_large = total_px > 100_000_000

        if is_large:
            print(f"  [!] Large raster ({total_px:,} px) -- using tiled derivative generation")
            with Timer(f"LiDAR derivatives ({dem_key})"):
                generate_lidar_derivatives(dem_path, deriv_dir, nodata)
        else:
            # 2a. Multi-azimuth hillshades
            hs_paths = compute_multi_hillshade(dem_path, deriv_dir, nodata=nodata)

            # 2b. SVF and Openness
            with Timer(f"SVF/Openness {dem_path.name}"):
                generate_svf_openness(dem_path, deriv_dir, nodata=nodata)

            # 2c. Slope, Curvature
            with Timer(f"Slope/Curvature {dem_path.name}"):
                generate_slope_curvature(dem_path, deriv_dir, nodata=nodata)

            # 2d. TPI
            with Timer(f"TPI {dem_path.name}"):
                generate_tpi(dem_path, deriv_dir, nodata=nodata)

            # 2e. PCA composite from multi-azimuth hillshades
            dem_stem = dem_path.stem
            all_hs_exist = True
            for az in HS_AZIMUTHS:
                npy_path = deriv_dir / f"{dem_stem}_hillshade_az{az}.npy"
                if not npy_path.exists():
                    all_hs_exist = False
                    break
            if all_hs_exist:
                generate_pca_composite(hs_paths, deriv_dir, dem_stem)

        # -- Cleanup transient derivatives --
        print(f"\n[Cleanup] Removing transient derivatives for {dem_key}")
        cleanup_transient_derivatives(OUT_DIR, dem_key,
                                       keep_list=["hillshade", "slope", "curv_profile",
                                                  "curv_plan", "svf"])

    # -- Storage summary --
    print(f"\n{'='*60}")
    print("STORAGE SUMMARY")
    print(f"{'='*60}")
    total_size = 0
    for dem_key in dems_to_process.keys():
        deriv_dir = OUT_DIR / dem_key
        if deriv_dir.exists():
            size_mb = sum(f.stat().st_size for f in deriv_dir.glob("**/*.npy")
                         if f.is_file()) / (1024 * 1024)
            png_size_mb = sum(f.stat().st_size for f in deriv_dir.glob("**/*.png")
                             if f.is_file()) / (1024 * 1024)
            print(f"  {dem_key}: {size_mb:.1f} MB (.npy) + {png_size_mb:.1f} MB (.png)")
            total_size += size_mb + png_size_mb

    print(f"  TOTAL: {total_size/1024:.2f} GB")
    print(f"{'='*60}")

    print(f"\n{'='*80}")
    print("01a DONE -- All derivatives saved to:")
    print(f"  {OUT_DIR}")
    print("Next: run 01b-TileSampling.py for tile sampling & visualisation")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
