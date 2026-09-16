#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01b-TileSampling.py -- Tile Sampling, Visualisation & Phase 1 Surface Expression Audit

Steps:
  3. Fault-aware tile sampling & multi-panel visualisation
  4. Phase 1 Surface Expression Audit:
     - Fault overlay maps
     - Perpendicular scarp profiles
     - Drainage anomaly detection
     - Expression quality classification

Prerequisites: Run 01a-DerivativeGen.py first to generate all derivatives.
Output goes to data/derivatives/ and data/derivatives/tiles/.
"""

from derivative_common import *


def main():
    """Step 3-4: Tile sampling, visualisation, and Phase 1 audit."""

    tiles_dir = safe_makedirs(OUT_DIR / "tiles")

    # -- Reconstruct dems_to_process from existing files --
    dems_to_process = {}

    for dem_key, label, default_nodata in [
        ("srtm1s", "SRTM 1s (~30 m)", -999.0),
        ("dem10m", "GA 10 m DEM", -3.4028234663852886e+38),
        ("lidar_0_5m", "LiDAR 0.5 m", -9999.0),
    ]:
        # Prefer clipped file if it exists
        clipped_path = OUT_DIR / f"{dem_key}_epsg28355_clipped.tif"
        unclipped_path = OUT_DIR / f"{dem_key}_epsg28355.tif"

        if clipped_path.exists():
            nd = infer_nodata_value(clipped_path)
            dems_to_process[dem_key] = {"path": clipped_path, "nodata": nd, "label": label}
        elif unclipped_path.exists():
            nd = infer_nodata_value(unclipped_path)
            dems_to_process[dem_key] = {"path": unclipped_path, "nodata": nd, "label": label}

    if not dems_to_process:
        print("[!] No reprojected DEM files found. Run 01a-DerivativeGen.py first.")
        return

    print(f"\n{'='*60}")
    print(f"DEMs available for tile sampling: {len(dems_to_process)}")
    for k, v in dems_to_process.items():
        print(f"  {k}: {v['path'].name}  (nodata={v['nodata']})")
    print(f"{'='*60}")

    # -- Step 3: Tile sampling & visualisation --
    print(f"\n{'='*60}")
    print("TILE SAMPLING & VISUALISATION")
    print(f"{'='*60}")

    struct_pts = None
    traces = None
    if STRUCT_PTS_PATH.exists():
        all_extents = []
        for info in dems_to_process.values():
            with rasterio.open(info["path"]) as src:
                b = src.bounds
                all_extents.append((b.left, b.right, b.bottom, b.top))
        if all_extents:
            combined_extent = (
                min(e[0] for e in all_extents),
                max(e[1] for e in all_extents),
                min(e[2] for e in all_extents),
                max(e[3] for e in all_extents),
            )
            struct_pts = load_structural_points(STRUCT_PTS_PATH, dem_extent=combined_extent)
            traces = reconstruct_fault_traces(struct_pts, cluster_dist_m=30, min_pts=5)
    else:
        print("[!] Structural points file not found")

    for dem_key, info in dems_to_process.items():
        dem_path = info["path"]
        nodata   = info["nodata"]

        tiles = sample_valid_tiles(dem_path, nodata=nodata, seed=42,
                                    struct_pts=struct_pts)

        if not tiles:
            print(f"  [!] No valid tiles found for {dem_key}")
            continue

        deriv_dir = OUT_DIR / dem_key
        tile_fig_dir = safe_makedirs(tiles_dir / dem_key)

        for i, tile in enumerate(tiles):
            fig_path = tile_fig_dir / f"tile_{i+1:02d}_{tile['center_x']:.0f}_{tile['center_y']:.0f}.png"
            if fig_path.exists():
                print(f"  [Skip] {fig_path.name} already exists")
                continue
            plot_tile_panel(dem_path, tile, deriv_dir, fig_path, nodata=nodata,
                           struct_pts=struct_pts, traces=traces)

    # -- Step 4: Phase 1 Surface Expression Audit --
    if struct_pts is not None and len(dems_to_process) > 0:
        run_phase1_audit(dems_to_process, struct_pts)
        
        # -- NEW: Cleanup phase1 artifacts to save space --
        audit_dir = OUT_DIR / "phase1_audit"
        print(f"\n[Cleanup] Removing intermediate phase1 artifacts")
        cleanup_phase1_artifacts(audit_dir, keep_analysis=True)

    # -- Final storage summary --
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
    
    audit_dir = OUT_DIR / "phase1_audit"
    if audit_dir.exists():
        audit_size_mb = sum(f.stat().st_size for f in audit_dir.glob("**/*") 
                           if f.is_file()) / (1024 * 1024)
        print(f"  phase1_audit: {audit_size_mb:.1f} MB")
        total_size += audit_size_mb
    
    print(f"  TOTAL: {total_size/1024:.2f} GB")
    print(f"{'='*60}")

    print(f"\n{'='*80}")
    print("DONE -- All derivatives and visualisations saved to:")
    print(f"  {OUT_DIR}")
    print(f"{'='*80}")




if __name__ == "__main__":
    main()
