# FaultScrap: Surface Expression Audit of Existing Fault Databases

## Overview

This project implements a multi-resolution surface expression audit of existing fault
databases in the Axedale region (Central-Western Victoria, Australia). The workflow
analyses three DEMs at different resolutions (30 m SRTM, 10 m GA DEM, 0.5 m LiDAR)
to classify fault expression quality and assess uncertainty.

The processing pipeline consists of the following scripts. Scripts are grouped by
current status (active in project root, or archived in `code_old/`):

**Active scripts (project root):**

1. **00-DataEDA.py** -- Pre-pipeline: Exploratory data analysis of input DEMs
   (CRS, resolution, extent, overlap, valid-data footprint)
2. **01a-DerivativeGen.py** -- Phase 1a: DEM reprojection, alignment, and
   terrain derivative generation (hillshade, SVF, openness, slope, curvature, TPI, PCA)
3. **01b-TileSampling.py** -- Phase 1b: Fault-aware tile sampling, multi-panel
   visualisation, and Phase 1 surface expression audit (scarp detection, drainage
   anomalies, expression quality classification)
4. **02-VisualAudit.py** -- Phase 2: Visualisation suite, geomorphic indices,
   uncertainty assessment, and cross-validation
5. **03-ResolutionSensitivity.py** -- Phase 3: Numerical resolution sensitivity
   analysis (downsampling, metric evaluation, detection matrix)
6. **05a-KnownTraceScarpValidation_dem10m.py** / **05a-KnownTraceScarpValidation_hs.py**
   -- Phase 5a: Known-trace, curvature-only validation that tests where catalogued
   traces are independently corroborated by pixel-scale scarp morphology (no ML
   candidate import). Split into two DEM-specific variants: `_dem10m` (10 m GA DEM)
   and `_hs` (high-resolution LiDAR).
7. **05b-DEMWideScarpInference.py** -- Phase 5b: ML-based wide-area scarp
   inference with iterative active learning, shape/topology filtering, and
   XGBoost classification
8. **06a-ScarpTraceCleaning.py** -- Phase 6a: Pre-process raw scarp trace point
   clouds (`Trace*.xlsx`) into cleaned, ordered polylines via density filtering,
   DBSCAN clustering, break-type classification, and Savitzky-Golay smoothing
9. **06b-ScarpParameterProfiling.py** -- Phase 6b: Detailed scarp parameter
   profiling at 10 m intervals along cleaned traces -- perpendicular transect
   extraction, cumulative relief analysis, throw decomposition, strike/dip estimation,
   and Nash (1980) linear diffusion fitting

**Shared library:**

- **derivative_common.py** -- 4,700-line shared module imported by 01a, 01b,
  05a, 05b, 06a, and 06b. Contains all DEM derivative computation functions
  (hillshade, SVF, openness, slope, curvature, TPI, PCA), data I/O utilities, CRS
  transforms, and path constants. Script 02 reads derivatives from disk but does
  not import derivative_common. Script 03 imports from `01-DerivativeGen.py`
  via `SourceFileLoader` (which itself imports derivative_common).

**Utility scripts:**

- **ClipHelper.py** -- Clips a raster to a reference raster extent (standalone).
- **DownSampleHelper.py** -- Creates downsampled smoke-test DEMs.
- **GPU_test.py** -- CuPy GPU vs CPU benchmark diagnostic.
- **Structural_InvHelper.py** -- Chunked structural-point ingestion from .txt files.

**Archived scripts (`code_old/`):**

The following scripts have been moved to `code_old/` and are no longer in the
project root. They are referenced in some sections of this README for historical
context but are not part of the active workflow:

- **01-DerivativeGen.py** -- Legacy monolith, replaced by 01a + 01b. Still
  imported by `03-ResolutionSensitivity.py` via `SourceFileLoader` for
  derivative computation functions.
- **04-LineamentML.py** -- Phase 4: Unsupervised ML (CNN autoencoder + Glass Box
  UMAP + HDBSCAN/KMeans) for lineament discovery in LiDAR tiles
- **05-MLTraceReconstruction.py** -- Phase 5: ML-guided fault trace reconstruction
  and lateral continuity validation
- **06-ChannelImportance.py** -- Phase 6: Channel ablation and gradient saliency
  to quantify per-channel contribution to ML retrieval
- **07-MultiResolutionSummary.py** -- Phase 7: Multi-resolution summary overlay
  of ML-inferred fault traces from all three DEM sources
- **07a-MultiResolutionReframe.py** -- Phase 7a: Post-pipeline synthesis that
  reframes "fault detection confidence" as "lineament expression confidence"
- **08-GeologicalContext.py** -- Phase 8: Geological context analysis and
  composite confidence scoring

---

## Data

### Input DEMs (in `data/`)

| DEM | File | CRS | Resolution | Nodata | Size |
|-----|------|-----|-----------|--------|------|
| SRTM 1s | `srtm1s_axe.tif` | EPSG:28355 | ~30 m | -999 | 3.3M px |
| GA 10 m | `Dem10m_axe.tif` | EPSG:3111 (Vicgrid) | 10 m | -3.4e38 | 49M px |
| LiDAR 0.5 m | `axedale_2020nov01_dem50cm_v10cm_epsg7855.tif` | EPSG:7855 | 0.5 m | -9999 | 42.8 GB |

### Structural Point Inventory

`structural_points_inventory.csv` -- 3,378,533 points encoding known fault traces
as dense coordinate pairs (~10 m spacing). Columns: XCoord, YCoord, source.
Four sources: CWvic1 (~1M), GeoStru2501 (~1M), GeoStru501 (~1M), FSD1 (~378K).
All coordinates in EPSG:28355.

Note: There are no vector fault trace linestrings -- all fault data is encoded as
point coordinates that must be clustered and ordered to reconstruct traces.

### Geological Unit Data

`GeoUnit(Point).txt` -- 90,832 points on a 250 m grid in EPSG:28355, each
carrying a `GEOGRAPHIC` ID that links to a geological unit. Semicolon-delimited.

`GeoUnit(Exp).txt` -- 934 geological unit descriptions, tab-delimited. Each row
maps a `GEOGRAPHIC` ID to an `ABBREVIATE` code (e.g. "Qc1", "Ocl"), a
`FORMATTED_` name (e.g. "colluvium( Qc1): generic"), and a `TEXT_DESCR` with
lithology and depositional environment. Used by Script 08 for geological
context analysis and by Script 05 for the geology panel in fault trace detail
figures.

---

## 00-DataEDA.py

### What It Does

This script performs exploratory data analysis (EDA) on all input DEM files before
the main pipeline begins. It scans the `data/` directory for GeoTIFF files (`.tif`)
and also supports ASCII grid (`.asc`) and point cloud (`.txt`) formats. It loads
their metadata (without reading full raster data into memory), checks coordinate
reference systems, computes valid-data footprints, and reports overlap and resolution
statistics. This is a diagnostic script -- it produces no output files, only console
output for the user to verify data integrity before running the derivative generation
pipeline.

### Methods

#### 1. DEM File Discovery

Scans the `data/` directory for `.tif` files using glob. Each file is registered
with a key based on the directory and filename.

#### 2. CRS Detection and Strategy Selection

All GeoTIFF files are opened with `rasterio` to read their CRS. The script then
determines a processing strategy:
- **All same projected CRS** -- Work natively (meters)
- **All same geographic CRS** -- Work natively (degrees)
- **Majority in one CRS** -- Use majority CRS, skip minority files
- **Mixed CRS** -- Transform all to EPSG:28355 (GDA94 / MGA zone 55)

Uses `pyproj.Transformer` for coordinate transformations, with a fallback to
`rasterio.warp.transform_bounds` if pyproj is unavailable.

#### 3. Metadata Extraction

For each DEM, extracts and reports:
- Source CRS and bounds
- Resolution (native units, with estimated meters for geographic CRS)
- Shape (height x width in pixels)
- Extent in working CRS (xmin, xmax, ymin, ymax)
- Coverage area in km

#### 4. Valid-Data Footprint Scanning

For high-resolution DEMs (< 20 m resolution), scans the raster in 1024-pixel-wide
row and column strips to determine the actual valid-data extent (as opposed to the
bounding box). This is critical for LiDAR data where nodata borders can be large. Reports:
- Actual valid-data extent (x, y bounds)
- Bounding-box coverage percentage (valid area / bbox area)

#### 5. Overlap and Common Extent Computation

Computes the intersection of all DEM extents to verify spatial overlap. Raises an
error if DEMs do not overlap. Reports the common extent in both native units and km.

### Output

No files are written. All results are printed to console. The script serves as a
pre-flight check to confirm:
- All DEMs are discoverable and readable
- CRS information is consistent or transformable
- DEMs spatially overlap
- High-resolution DEMs have valid data coverage

---

## 01a-DerivativeGen.py

### What It Does

This script performs Steps 1-2 of Phase 1: DEM reprojection, alignment, and terrain
derivative generation. It replaces the monolithic `01-DerivativeGen.py` as the
first half of a two-script split (01a generates derivatives, 01b performs tile
sampling and the surface expression audit).

1. **DEM Reprojection & Alignment** -- Reprojects all DEMs to EPSG:28355 and clips
   to a common extent aligned with the LiDAR footprint
2. **Derivative Generation** -- Computes the full terrain derivative stack for each
   DEM (hillshade, SVF, openness, slope, curvature, TPI, PCA composite)
3. **Cleanup** -- Removes transient derivatives to save disk space, keeping only
   the essential set (hillshade, slope, profile curvature, plan curvature, SVF)

### Methods

#### 1. DEM Reprojection (Step 1)

Each DEM is reprojected to EPSG:28355 (GDA94 / MGA zone 55) using bilinear
resampling. For DEMs with uncompressed size <= 2 GB, a single full-array
reproject is used. For very large DEMs (LiDAR, 42.8 GB), a windowed approach
is used where each output tile gets a correctly offset destination transform.

#### 1b. Common Extent Alignment

All reprojected DEMs are clipped to a common extent aligned with the LiDAR
footprint (via `clip_to_common_extent` from `derivative_common`). This ensures
all DEMs cover the same geographic area for multi-resolution comparison.

#### 2. Terrain Derivatives (Step 2)

All derivatives are saved as `.npy` arrays with companion `.json` metadata
(transform, CRS, shape, nodata, resolution, type) and 300 dpi PNG visualisation
with easting/northing axes and colorbar.

| Derivative | Method | Parameters |
|-----------|--------|-----------|
| Hillshade | Analytical formula (numpy central-difference gradient) | Azimuths 315/45/135/225 deg, altitude 8 deg |
| Sky-View Factor (SVF) | Yokoyama et al. 2002 | 16 directions, max radius = 1000 m (capped at 200 cells) |
| Positive/Negative Openness | Yokoyama et al. 2002 | Same parameters as SVF |
| Slope | Zevenbergen & Thorne (1987) | Output in degrees |
| Profile Curvature | Zevenbergen & Thorne (1987) | 1/m, along-slope convexity/concavity |
| Plan Curvature | Zevenbergen & Thorne (1987) | 1/m, across-slope convergence |
| TPI | Uniform filter (small radii) / circular FFT filter (large radii) | 50 m, 200 m, 1000 m radii |
| PCA Composite | SkPCA on multi-azimuth hillshades | First 3 principal components (saved as separate `pca_pc1`, `pca_pc2`, `pca_pc3` files) |

For LiDAR (0.5 m, 42.8 GB), derivatives are computed tile-by-tile (up to 4000 x 4000
pixels with 50-pixel overlap, adaptively scaled down based on available VRAM/RAM)
and saved as tiled `.npy` files (`tile_RRRR_CCCC_{name}.npy`) with a master metadata
JSON. Overview PNGs are generated at 50 m resolution for visualisation.

#### 2b. Transient Derivative Cleanup

After derivative generation, transient products (openness, TPI, PCA composite)
are removed to save disk space. The following are retained: hillshade (all
azimuths), slope, profile curvature, plan curvature, and SVF. Note: PNG
visualisations of transient derivatives are retained on disk (only `.npy` and
`.json` files are deleted).

Output: `data/derivatives/{dem_key}/{dem_stem}_{derivative}.npy` + `.json` + `.png`

### Figures (01a-DerivativeGen.py)

| Figure | What It Shows | How to Interpret |
|--------|-------------|-----------------|
| `{dem_key}_hillshade_az{NNN}.png` | Hillshade at azimuth NNN deg, altitude 8 deg | Low sun angle enhances subtle scarps and lineaments |
| `{dem_key}_svf.png` | Sky-View Factor | Uniform illumination; low values = deep valleys / scarps |
| `{dem_key}_openness_pos.png` | Positive topographic openness | High values = ridges/crests; low = valleys |
| `{dem_key}_openness_neg.png` | Negative topographic openness | High values = depressions; low = peaks |
| `{dem_key}_slope.png` | Terrain slope in degrees | Steep slopes may indicate fault scarps |
| `{dem_key}_curv_profile.png` | Profile curvature (1/m) | Positive = convex (crest); negative = concave (valley) |
| `{dem_key}_curv_plan.png` | Plan curvature (1/m) | Positive = divergent; negative = convergent |
| `{dem_key}_tpi_{R}m.png` | Topographic Position Index at radius R | Positive = ridge/crest; negative = valley |
| `{dem_key}_pca_pc1.png` | First principal component of multi-azimuth hillshade | PC1 captures the dominant topographic structure regardless of sun angle |
| `{dem_key}_pca_pc2.png` | Second principal component | PC2 captures orthogonal/subordinate structure |
| `{dem_key}_pca_pc3.png` | Third principal component | PC3 captures residual variation |

---

## 01b-TileSampling.py

### What It Does

This script performs Steps 3-4 of Phase 1: fault-aware tile sampling, multi-panel
tile visualisation, and the Phase 1 surface expression audit. It replaces the second
half of the monolithic `01-DerivativeGen.py`.

1. **Tile Visualisation** -- Generates 2 km x 2 km tile panels with fault overlays
2. **Phase 1 Audit** -- Fault trace reconstruction, scarp detection, drainage
   anomaly detection, and expression quality classification

### Methods

#### 3. Tile Visualisation (Step 3)

Ten 2 km x 2 km tiles are sampled per DEM. Tile selection prioritises locations
with known fault traces (structural points within 500 m of tile centre), then
falls back to random valid tiles. Each tile is plotted as a 6-panel figure
(300 dpi) showing hillshade, slope, SVF, positive openness, profile curvature, and TPI 200 m.
Fault traces are overlaid as coloured lines.

Output: `data/derivatives/tiles/{dem_key}/tile_XX_EEEEE_NNNNN.png`

#### 4. Phase 1 Audit (Step 4)

**Step 1: Fault Trace Reconstruction** -- Structural points are grouped by source
database, then clustered into individual fault traces using BFS with a 30 m distance
threshold and a minimum of 5 points per trace. Points within each cluster are ordered
via nearest-neighbour chaining. Each trace gets an ID, length, azimuth, and centroid.

**Step 2: Fault Overlay Map** -- All fault traces are plotted on a hillshade basemap,
coloured by source database (CWvic1=red, FSD1=green, GeoStru501=blue, GeoStru2501=orange).

**Step 3: Perpendicular Profile Extraction** -- At 500 m intervals along each fault
trace, a profile is extracted perpendicular to the local fault direction. Profile
half-length = 500 m. Elevation is sampled from the DEM via bilinear interpolation.

**Step 4: Scarp Analysis** -- Each profile is analysed for scarp morphology:
- Scarp height = elevation difference across the steepest gradient zone
- Scarp width = distance over which 80% of elevation change occurs
- Classification: **discrete** (width <= 200 m) or **monocline** (width > 200 m)
- Minimum detectable scarp height = 0.5 m

**Step 5: Drainage Anomaly Detection** -- A Drainage Anomaly Index (DAI) = |profile
curvature| x slope highlights linear valleys near faults. Anomalies are classified as:
- **deflected_stream** -- perpendicular to fault, elongated
- **linear_valley** -- aligned with fault, elongated
- **depression_cluster** -- compact, not elongated

**Step 6: Expression Quality Classification** -- Each point is classified using
slope, SVF anomaly, and drainage anomaly criteria (point-based, not trace-level):
- **Class 1** (well-constrained): Scarp height >= 1 m AND strong geomorphic signal
  (max slope >= 5 deg, or slope anomaly >= 5 deg, or SVF anomaly >= 0.15) or drainage anomaly present
- **Class 2** (moderately-constrained): Scarp height >= 0.5 m, OR moderate geomorphic
  signal (max slope >= 2 deg, or slope anomaly >= 2 deg, or SVF anomaly >= 0.08) or drainage anomaly present
- **Class 3** (inferred): No detectable scarp or geomorphic signal
- **Class D** (no surface expression): No scarp, no drainage anomalies, no slope/SVF signal

Trace-level classification takes the highest class among all points in the trace.

Output: `data/derivatives/phase1_audit/`
- `fault_traces_summary.csv`
- `{dem_key}_fault_overlay.png` (deleted by cleanup step after audit)
- `{dem_key}_scarp_analysis.csv`
- `{dem_key}_profiles/` (scarp profile PNGs; deleted by cleanup step after audit)
- `{dem_key}_drainage_anomalies.csv`
- `{dem_key}_expression_quality.csv`
- `{dem_key}_expression_quality_points.csv` (point-level classification)
- `{dem_key}_expression_quality_map.png`

Note: A cleanup step removes intermediate artefacts (fault overlay PNGs and profile
PNGs) after the audit is complete to save disk space.

### Figures (01b-TileSampling.py)

| Figure | What It Shows | How to Interpret |
|--------|-------------|-----------------|
| `tile_XX_*.png` | 6-panel tile view at fault-proximate locations | Compare derivative responses at known fault locations |
| `{dem_key}_fault_overlay.png` | Fault traces on hillshade by source | Evaluate database agreement/disagreement |
| `scarp_profile_XX_*.png` | Annotated elevation profile across fault | Red zone = scarp; h = height, w = width, slope = max gradient |
| `{dem_key}_expression_quality_map.png` | Fault traces coloured by quality class | Red = Class 1 (well-constrained), Orange = Class 2, Yellow = Class 3, Grey = Class D (no expression) |

---

## 02-VisualAudit.py

### What It Does

This script implements Phase 2 of the audit plan: visualisation of geomorphic
evidence, quantitative indices, uncertainty assessment, and cross-validation with
published analogues. It reads derivatives from 01a-DerivativeGen.py (.npy + .json format)
and Phase 1 audit CSVs produced by 01b-TileSampling.py.

### Methods

#### 2.1 Base Cartographic Products

**Multi-Azimuth Hillshade Composite** -- Blends 4 azimuth hillshades into:
- Grayscale mean (all azimuths averaged)
- RGB false-colour composite (R=315 deg, G=45 deg, B=135 deg)

Each azimuth highlights different terrain features; the composite reveals structures
visible from multiple illumination directions.

**SVF/Openness Comparison** -- Side-by-side display of hillshade, SVF, and positive
openness. SVF provides uniform illumination (no azimuth bias) and openness shows
terrain morphology from a nadir perspective.

#### 2.2 Geomorphic Evidence Maps

**Lineament Extraction** -- Semi-automated lineament extraction from the PCA hillshade
composite using Canny edge detection (sigma=2.0) followed by connected component
chaining. Lineaments >= 500 m are retained with azimuth, length, and centroid.
A Kernel Density Estimation (KDE, bandwidth=2 km) map shows structural corridors.

**Drainage Anomaly Map** -- Anomalous reaches from Phase 1 plotted on hillshade with
fault traces. The script also implements a pixel-level river valley/bed detection
system using REM, HAND, TPI, and local relief criteria, producing river valley and
river bed masks. Colour-coded by type: cyan = deflected stream, magenta = linear valley,
yellow = depression cluster.

**Expression Quality 4-Class Map** -- Upgrades the Phase 1 three-class system to
four classes, implemented as a per-pixel classification within a 200 m buffer of
fault traces using slope and SVF thresholds. One figure is produced per fault source
(FSD1, CWvic1, GeoStru2501, GeoStru501):
- **Class A** (well-expressed): Clear, unambiguous geomorphic signature (from Phase 1 Class 1)
- **Class B** (probable): Multi-indicator concordance (from Phase 1 Class 2)
- **Class C** (inferred): Single indicator or ambiguous context (from Phase 1 Class 3)
- **Class D** (no expression): No scarp, no drainage anomalies, no slope/SVF signal

#### 2.3 Quantitative Index Visualisation

**Geomorphic Indices** -- Per fault segment (note: Vf, HI, and AF are proxy/placeholder
values; SL is computed from slope):
- Smf (Mountain Front Sinuosity) = trace_length / straight_distance; low = active
- Vf (Valley Floor Width-to-Height Ratio) = 0.5 (proxy; not fully computed)
- HI (Hypsometric Integral) = 0.5 (proxy; not fully computed)
- SL (Stream Length-Gradient Index) = slope * 100; anomalous peaks = knickpoints
- AF (Drainage Basin Asymmetry Factor) = 50.0 (proxy; not fully computed)

**Radar/Spider Charts** -- Normalised geomorphic indices per fault segment,
allowing rapid visual comparison. Smf is inverted (1 - normalised) so all axes
point toward "more active".

**Longitudinal River Profiles** -- Elevation profiles across fault traces with
SL index annotation. Knickpoints are flagged where SL exceeds the 90th percentile.

**Hypsometric Curves** -- Area-elevation relationship for 2 km x 2 km windows
around fault trace centroids. HI values > 0.5 suggest youthful, tectonically
active landscapes. Reference curves (convex = young, concave = mature) are overlaid.

#### 2.4 Uncertainty and Confidence Layers

**Confidence Heatmap** -- Composite data-coverage map showing resolution availability:
- White = no data
- Light blue = SRTM 30 m only
- Medium blue = GA 10 m + SRTM
- Dark blue = LiDAR 0.5 m + 10 m + SRTM

For large LiDAR files, coverage is approximated from footprint bounds rather than
pixel-by-pixel reading.

**Multi-Resolution Comparison** -- Side-by-side 2 km x 2 km tiles at 30 m, 10 m,
and 0.5 m resolution at the same fault location. Shows how resolution affects
scarp detectability. Tile selection prioritises fault-proximate locations.

#### 3.3 Confidence Scoring

Each fault segment receives a composite confidence score (0-10):
- Scarp evidence: 0-3 points (based on height relative to minimum detectable)
- Detection consistency: 0-2 points (detection rate)
- Drainage corroboration: 0-2 points (number of anomalies)
- Slope/SVF anomaly: 0-1 each
- DEM resolution bonus: 0-1 points
- LiDAR confirmation: 0-1 bonus

Uncertainty classification:
- **Aleatory** -- Inherent randomness (scarp below detection threshold)
- **Epistemic** -- Knowledge gap (inconsistent detection, needs higher-res data)
- **Low** -- Multiple concordant indicators

#### 3.4 Cross-Validation

Benchmark against published analogue scarp heights:
- **Yallourn Monocline**: 1-2 m scarps (same region, similar geology)
- **Korean scarps**: 2-4 m scarps (different tectonic setting, similar scale)

Resolution sufficiency assessment:
- "sufficient" -- scarp >= 3 m (detectable by SRTM)
- "need_10m" -- scarp 1-3 m (requires GA 10 m DEM)
- "need_LiDAR" -- scarp < 1 m (requires LiDAR)

Where LiDAR Phase 1 results are available, they are preferred for cross-validation.

### Figures (02-VisualAudit.py)

| Figure | What It Shows | How to Interpret |
|--------|-------------|-----------------|
| `{dem_key}_hillshade_composite.png` | Mean grayscale + RGB false-colour hillshade | RGB highlights features visible from specific azimuths |
| `{dem_key}_svf_openness_comparison.png` | Hillshade vs SVF vs openness | SVF reveals structures invisible in single-azimuth hillshade |
| `{dem_key}_lineament_density.png` | KDE density of extracted lineaments on hillshade | Hot spots = structural corridors with multiple lineaments |
| `{dem_key}_drainage_anomaly_map.png` | Anomaly locations on hillshade with fault traces | Co-located anomalies + faults = stronger evidence |
| `{dem_key}_expression_quality_4class_{source}.png` | Per-source fault traces by A/B/C/D class | Red=A (best), Orange=B, Yellow=C, Grey=D (no expression). One figure per fault source. |
| `radar_trace_XXXX.png` | Spider chart of normalised geomorphic indices | Larger area = more tectonically active signature |
| `river_profile_XXXX.png` | Elevation + SL index across fault | SL spikes (red dots) = knickpoints at or near fault |
| `hypsometric_XXXX.png` | Area-elevation curve with HI value | Convex + high HI = young/active; concave + low HI = mature |
| `data_coverage_confidence_heatmap.png` | Resolution availability map | Darker blue = higher resolution data available |
| `multires_XX_*.png` | Same location at 30 m / 10 m / 0.5 m | Compare scarp visibility across resolutions |
| `{dem_key}_confidence_scores.csv` | Per-segment confidence score + uncertainty tier | Higher score = more reliable fault expression |
| `{dem_key}_geomorphic_indices.csv` | Per-segment geomorphic index values | Smf, Vf (proxy), HI (proxy), SL, AF (proxy) |
| `{dem_key}_cross_validation.csv` | Cross-validation against published analogues | Scarp height comparison and resolution sufficiency |
| `lineaments.csv` | Extracted lineament records (azimuth, length, centroid) | Lineaments >= 500 m retained |
| `{dem_key}_river_valley_bed.png` | River valley/bed detection masks on hillshade | Cyan = river valley, magenta = river bed |
| `{dem_key}_hand.tif` | HAND (Height Above Nearest Drainage) raster | <= 0 m = water body |
| `{dem_key}_rem.tif` | REM (Relative Elevation Model) raster | Elevation normalised to river surface |
| `{dem_key}_river_surface.tif` | River surface elevation raster | Interpolated river surface |
| `{dem_key}_river_bed.tif` | Binary river bed mask | Where REM is very low |
| `{dem_key}_river_valley.tif` | Binary river valley mask | Broader valley-floor mask |

---

## 03-ResolutionSensitivity.py

### What It Does

This script implements Phase 3 of the audit plan: a numerical resolution sensitivity
analysis. It uses the reprojected LiDAR DEM (0.5 m) as ground truth, downsamples it
to a series of coarser resolutions, recomputes derivatives and fault metrics at each
resolution, and quantifies how detectability degrades with decreasing resolution.

It imports computation functions from `01-DerivativeGen.py` (the legacy monolith,
loaded via `SourceFileLoader`) and reads Phase 1 expression quality CSVs produced
by 01b-TileSampling.py.

### Methods

#### 1. Downsampling Protocol (Section 1.1)

The reprojected LiDAR DEM (0.5 m, EPSG:28355) is downsampled to five target
resolutions (0.5, 2, 5, 10, 30 m). LiDAR-derived resolutions (0.5 m, 2 m, 5 m)
use three algorithms; 10 m and 30 m are read from independent external DEM `.tif`
files (`dem10m_epsg28355.tif`, `srtm1s_epsg28355.tif`) with algorithm tag `"native"`.

- **Mean aggregation** -- Read native-resolution block, take the mean of valid pixels.
  Preserves scarp relief best for discrete scarps. Applied to LiDAR-derived resolutions.
- **Bilinear** -- `rasterio.warp.reproject` with `Resampling.bilinear`. Smooth
  interpolation; good compromise. Applied to LiDAR-derived resolutions.
- **Cubic** -- `rasterio.warp.reproject` with `Resampling.cubic`. Sharper but may
  introduce ringing artefacts near scarps. Applied to LiDAR-derived resolutions.
- **Native** -- 10 m and 30 m resolutions are read directly from external DEM `.tif`
  files (GA 10 m and SRTM 30 m), not downsampled from LiDAR.

Before downsampling, a **noise filter** removes residual LiDAR spikes exceeding
0.5 m from the 5-pixel neighbourhood mean.

Downsampling is performed per **test site** (1 km x 1 km window centred on the site).

**Test site selection** -- Five representative sites are chosen from the fault trace
inventory with the following priorities:
1. At least one Class 1 trace (well-expressed scarp)
2. One Class 2 trace (moderate)
3. One monocline/fold trace
4. One Class 3 / flat terrain trace
5. Remaining filled with spatially distributed traces (minimum 2 km apart)

Output: `data/derivatives/phase2_sensitivity/downsampled/site_{N}_res{R}m_{alg}.npy` + `.json`

#### 2. Derivative Generation at Each Resolution (Section 1.2)

For each downsampled DEM, the full derivative stack is computed:
- Hillshade at 4 azimuths (315/45/135/225 deg, altitude 8 deg)
- SVF and positive/negative openness (16 directions, max radius capped at 200 cells)
- Slope (degrees)
- Profile curvature
- TPI at 3 physical radii (50 m, 200 m, 1000 m; skipped when radius < 1 cell)

All derivatives are saved as `.npy` + `.json` + `.png` (300 dpi with easting/northing
axes and colorbar), consistent with the 01a-DerivativeGen.py format.

Output: `data/derivatives/phase2_sensitivity/derivatives/site_{N}_res{R}m_{alg}_{deriv}.npy`

#### 3. Quantitative Metric Evaluation (Section 1.3)

Six metrics quantify feature detectability at each resolution:

| Metric | Definition | Interpretation |
|--------|-----------|---------------|
| SNR | Mean scarp relief / std of background variation | >2 = detectable; <1 = below noise |
| Cohen's Kappa | Interpreter agreement (simulated) | >0.8 = almost perfect; >0.6 = substantial |
| PABAK | Prevalence-adjusted bias-adjusted kappa | Corrects kappa for high-prevalence bias |
| RMSE_h | RMSE of scarp height vs. 0.5 m reference | Measures systematic height bias |
| SLPR | Scarp Length Preservation Ratio | <1 = scarps lost at coarser resolution |
| LFI | Lineament Fragmentation Index | >1 = more fragments at coarser resolution |
| MDD | Minimum Detectable Displacement = 1.5 x vertical RMSE | Floor of detectability |

**SNR computation**: The fault buffer is defined as pixels within 100 m of any fault
trace. The control zone uses pixels outside the buffer with slope < 5 deg (to exclude
non-tectonic steep terrain). SNR = |mean(fault zone) - mean(control zone)| / std(control zone).

**Simulated interpreter agreement**: Two virtual interpreters with different detection
thresholds (2 x resolution vs. 2.5 x resolution for scarps; 20 x vs. 30 x for
drainage features; 50 x vs. 70 x for lineaments) plus 5% random disagreement noise.
This models the effect of resolution on subjective feature identification.

**Detection matrix**: Each (resolution, feature type) cell is classified as:
- **Green** (reliably detected): kappa >= 0.8 AND SNR >= 2.0
- **Amber** (marginally detected): kappa >= 0.6 OR SNR >= 2.0
- **Red** (not detected): below both thresholds

Output: `data/derivatives/phase2_sensitivity/metrics/all_metrics.csv`, `detection_matrix.csv`

#### 4. Core Sensitivity Visualizations (Section 2.1)

**Multi-Resolution Hillshade** -- 2 x 3 (or fewer) panel grid showing the same fault
scarp at all resolutions under identical illumination (azimuth 315 deg, altitude 8 deg).
Easting/northing axes, 300 dpi.

**SNR vs Resolution** -- Log-scale x-axis plot of SNR vs. resolution for each algorithm
and test site. Horizontal dashed line at SNR = 2.0. Theoretical curves for 1 m, 2 m,
and 4 m scarp heights overlaid.

**Kappa vs Resolution** -- Grouped bar chart per feature type (scarp, drainage,
lineament). Reference lines at kappa = 0.6 (substantial) and kappa = 0.8 (almost
perfect agreement).

**Scarp Cross-Sections** -- Overlaid elevation transects at each resolution on a
single graph. Shows progressive smearing of the scarp inflection with coarser
resolution. Vertical exaggeration = 15x.

**Detection Matrix Heatmap** -- Colour-coded grid: feature type vs. resolution.
Green = reliably detected, amber = marginal, red = not detected. Cell labels
show status text.

#### 5. Supplementary Spatial Products (Section 2.2)

**SNR Spatial Variability Map** -- Raster of local SNR values computed in a 200 m x
200 m moving window. Identifies "difficult zones" where SNR is consistently below
the detection threshold. Plotted for selected resolutions (0.5 m, 2 m, 10 m) at
the first two test sites.

**Lineament Fragmentation Map** -- Overlays lineaments detected at a given resolution
against the 0.5 m reference, colour-coded:
- Green = preserved (length >= 80% of reference)
- Amber = degraded (length 50-80%)
- Red = lost (length < 50% or no match within 500 m)

#### 6. Uncertainty Assessment (Section 3.1-3.4)

**3.1 Resampling Algorithm Uncertainty** -- Compares mean vs. bilinear vs. cubic at
2 m and 5 m resolution. Reports the standard deviation of SNR and RMSE_h across
algorithms. Recommends the algorithm with the highest SNR most frequently.

**3.2 Reference Data Uncertainty** -- Documents LiDAR vertical RMSE (0.15 m open
ground, 0.50 m under vegetation). Propagates error: MDD_corrected =
sqrt(MDD^2 + lidar_rmse^2). Reports corrected MDD for each resolution.
Notes temporal mismatch risk if LiDAR and SRTM acquisition dates differ.

**3.3 Bootstrap Confidence Intervals** -- Resamples the metric population 1000 times
(with replacement) and reports the 95% CI for SNR and kappa at each (resolution,
feature type, algorithm) combination.

**3.4 Control Zone Sensitivity** -- Recomputes SNR at three independently shifted
control zones (+/-100 m offsets). Reports the coefficient of variation (CV) and
flags sites where SNR varies by >20% across zones (indicating unstable detection).

Output: `data/derivatives/phase2_sensitivity/uncertainty/` (resampling_comparison.csv,
recommended_algorithm.txt, reference_uncertainty_mdd.csv, reference_uncertainty.json,
bootstrap_ci.csv, control_zone_sensitivity.csv)

### Figures (03-ResolutionSensitivity.py)

| Figure | What It Shows | How to Interpret |
|--------|-------------|-----------------|
| `site_{N}_multires_hillshade.png` | Same scarp at all resolutions, identical illumination | Progressive loss of detail and scarp sharpness at coarser resolutions |
| `site_{N}_snr_vs_resolution.png` | SNR curves per algorithm, with theoretical model | Curves crossing SNR=2 indicate the resolution threshold for detection |
| `snr_vs_resolution.png` | Combined SNR plot across all sites (mean algorithm) | Sites with higher-quality scarps maintain detectability to coarser resolutions |
| `kappa_vs_resolution.png` | Kappa bar charts per feature type | Kappa dropping below 0.6 indicates unreliable feature identification |
| `site_{N}_scarp_cross_sections.png` | Overlaid elevation profiles at each resolution | Scarp inflection becomes progressively smeared; vertical offset indicates height bias |
| `detection_matrix_heatmap.png` | Green/amber/red grid: feature vs. resolution | Clear visual summary of which features are detectable at which resolution |
| `site_snr_spatial_res{R}m.png` | SNR raster map at resolution R | Low-SNR patches = zones where fault detection is unreliable |
| `site_{N}_lineament_fragmentation_res{R}m.png` | Lineament overlap: green=preserved, amber=degraded, red=lost | Quantifies how much of the fault network is preserved at this resolution |
| `site_{N}_res{R}m_tile_panel.png` | 2x3 multi-panel: hillshade, slope, SVF, openness+, profile curvature, TPI | Per-site, per-resolution derivative overview for visual quality check |

### Data Outputs (03-ResolutionSensitivity.py)

| File | Description |
|------|-------------|
| `test_sites.csv` | Selected test site centres and metadata (output by `select_scarp_tiles`) |
| `downsampled_results.csv` | Per-resolution, per-algorithm downsampling summary (stems, shapes, validity) |
| `downsampled/{stem}.npy` + `.json` | Downsampled DEM arrays and metadata (transform, resolution, nodata) |
| `downsampled/site_{N}_res{R}m_{alg}.npy` + `.json` | Per-site, per-resolution, per-algorithm DEM windows |
| `downsampled/{stem}_hillshade.npy`, `_slope.npy`, `_svf.npy`, `_openness_pos.npy`, `_curv.npy`, `_tpi.npy` | Derivative arrays at each resolution |
| `metrics/all_metrics.csv` | Full metrics table (SNR, kappa, RMSE_h, SLPR, LFI, MDD per resolution/algorithm/site/feature) |
| `metrics/detection_matrix.csv` | Feature detectability matrix (detected/marginal/not detected) |
| `uncertainty/resampling_comparison.csv` | Algorithm comparison (mean vs. bilinear vs. cubic) |
| `uncertainty/recommended_algorithm.txt` | Best algorithm recommendation per metric |
| `uncertainty/reference_uncertainty_mdd.csv` + `.json` | LiDAR RMSE-propagated MDD correction |
| `uncertainty/bootstrap_ci.csv` | 95% bootstrap CIs for SNR and kappa |
| `uncertainty/control_zone_sensitivity.csv` | SNR CV across shifted control zones |

### Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| TARGET_RESOLUTIONS | [0.5, 2, 5, 10, 30] | Resolutions to test (5 values) |
| RESAMPLING_ALGORITHMS | ["mean", "bilinear", "cubic", "native"] | Downsampling methods; "native" reads external DEMs for 10/30 m |
| N_TEST_SITES | 10 | Number of representative fault sites |
| SITE_RADIUS_M | 500 | Half-width of each test site (1 km x 1 km) |
| SNR_THRESHOLD | 2.0 | SNR below which features are not detectable |
| KAPPA_THRESHOLD_SUBSTANTIAL | 0.6 | Minimum kappa for "substantial" agreement |
| KAPPA_THRESHOLD_ALMOST | 0.8 | Kappa for "almost perfect" agreement |
| LIDAR_VERTICAL_RMSE_OPEN | 0.15 m | LiDAR vertical RMSE on open ground |
| LIDAR_VERTICAL_RMSE_VEG | 0.50 m | LiDAR vertical RMSE under vegetation |
| N_BOOTSTRAP | 1000 | Number of bootstrap resamples |
| BOOTSTRAP_CONFIDENCE | 0.95 | Confidence level for bootstrap CIs |

---

## 05-MLTraceReconstruction.py (archived in code_old/)

> **Archived**: This script and its dependencies (04-LineamentML.py) have been
> moved to `code_old/`. The active workflow now uses 05a + 05b instead.

### What It Does

This script closes the loop between unsupervised latent-space discovery (Script 04)
and the geomorphic ground-truthing approach from Scripts 01a/01b/02. Script 04 identifies
which tiles look like scarps in latent space; Script 05 tests whether those tiles form
geomorphically coherent, laterally continuous fault traces --- the criterion geologists
use to distinguish real scarps from road cuts, terrace risers, or autoencoder false
positives (Limitation #6).

The script runs per DEM source (`--dem-source dem10m | lidar_0_5m | srtm1s`) and writes
all outputs to `ml_trace_reconstruction_{dem_source}/` so that each resolution is processed
independently and compared later in Scripts 07 / 07a.

Five steps, each independently resumable via `--step`:

1. **Candidate Aggregation, Corridor Detection & Fault Validation** -- Retrieves tiles
   cosine-similar to Class 1 reference tiles (well-constrained: height >= 1 m, detection
   rate >= 50%, >= 3 profiles), filters out known-scarp tiles, and computes a composite
   ML score. A 2D KDE on candidate centroids identifies contiguous corridor hotspots.
   Corridor formation is augmented with five enhancements:

   - **Class 1 reference weight (Script 02 × Script 08 cross-referencing)**: Each
     Class 1 reference tile is cross-referenced against Script 02's per-trace Smf
     (sinuosity) and Script 08's geological plausibility. Because the three scripts
     use different trace_id numbering schemes, matching is done spatially: a cKDTree
     on Script 02 trace centroids provides Smf for each Class 1 tile, and a cKDTree
     on Script 08's 10k+ point-level geological samples provides geological
     plausibility and unit age class. Tiles whose trace has high Smf (> 5.0,
     river-like) or falls in `no_lithology_contrast` / `insufficient_evidence` /
     Quaternary-only context are down-weighted:

     | Condition | Weight | Effect |
     |-----------|--------|--------|
     | High Smf AND poor geology | 0.0 | Dropped entirely |
     | High Smf only | 0.2 | Strongly down-weighted |
     | Poor geology only | 0.5 | Moderately down-weighted |
     | Neither flag | 1.0 | Full weight |

     The weight is applied as a multiplier to `ml_score` before corridor detection,
     so down-weighted reference tiles contribute less to KDE density and to
     downstream confidence scoring (Step 4). The original unweighted score is
     preserved in `ml_score_original`.

   - **Sinuosity / meander flagging**: Each candidate chain is NN-ordered and its
     sinuosity (Smf = path_length / straight_distance) and turning-angle periodicity
     are computed. Corridors whose chain Smf exceeds 2.0 or whose Script 02 trace-level
     `ref_smf` exceeds 5.0 are flagged as meander-suspect (real faults have Smf 3.0-3.7;
     river meanders typically > 5.0). High-sinuosity KDE corridors are split into
     straighter sub-corridors.
   - **Low-density fallback corridor detection**: Candidate chains too sparse to trigger
     a KDE hotspot are recovered via DBSCAN-style chain linking (200 m max gap, 3-200
     tiles per chain). Fallback corridors are deprioritised relative to KDE corridors
     but retained so short or isolated lineaments are not lost.
   - **FSD1 proximity prioritisation**: Corridors within 1000 m of an FSD1 (neotectonic)
     structural point are flagged `near_fsd1` and visually highlighted in the density map.
   - **Known-fault trace validation**: All four fault databases (FSD1, CWvic1,
     GeoStru2501, GeoStru501) are reconstructed from structural points, and for each
     trace the script searches for ML candidate tiles and section picks within a 200 m
     relocation tolerance. Per-trace offset metrics, a relocation-confidence rating
     (high / moderate / low / none / no_evidence), and a combined summary figure are
     produced.

2. **Cross-Section Grid per Corridor (bidirectional)** -- Generates section lines in
   two orientations through each corridor:

   - **North (E-W sections, looking north)**: lines spaced in northing at 25 m,
     half-length 350 m, sampled from the native DEM via rasterio windowed single-row reads.
   - **East (N-S sections, looking east)**: lines spaced in easting at 25 m, half-length
     350 m, sampled via single-column windowed reads.

   **v2 multi-scale 1D scarp detection**: The Phase 1 steepest-gradient routine has been
   replaced by a multi-scale curvature-based detector that reuses the same TV denoising
   weights as the 2D LSM (`SCARP_TV_WEIGHTS = LSM_TV_WEIGHTS`). For each elevation
   profile:

   1. TV-denoise at 6 scales (weights 0.005--0.16) using
      `skimage.restoration.denoise_tv_chambolle` (1D mode).
   2. Compute first derivative (slope S_n) and second derivative (curvature C_n) at
      each scale.
   3. Find the steepest-gradient point `d_max` (same as v1) within the central search
      window.
   4. Walk downslope from `d_max` until curvature crosses zero or drops below the
      background noise threshold (mean +/- 2 sigma of curvature in the flat reference
      segments at both profile ends, using the same control-zone logic as Script 03's
      SNR). That crossing point, refined by parabolic interpolation for sub-pixel
      precision, is the **base-of-scarp** position. Walk upslope to record the
      **crest** position.
   5. Compute base-of-scarp position at all 6 scales. Find the longest range of scales
      where the position is stable (within `SCARP_STABILITY_PIX = 2` pixels). Use the
      coarsest scale in that stable range (maximally denoised while still tracking the
      real base). If no stable range exists, flag the pick as low-confidence.
   6. The v1 scarp midpoint, height, width, slope, type, and throw sense are still
      computed at the finest scale for backward compatibility.

   Corridors whose dominant lineament azimuth is within 30 deg of E-W are flagged
   `poor_geometry_flag` (N-looking sections sub-parallel); corridors within 30 deg
   of N-S are flagged `poor_geometry_flag_east` (E-looking sections sub-parallel).
   Picks are saved per orientation as `picks_north.csv` and `picks_east.csv`, now
   including the new fields: `crest_position`, `base_of_scarp_position`,
   `scale_used`, `position_stability_range`, `pick_confidence`, and
   `lsm_agreement_distance_m` (populated in Step 4).

   **v2 cache invalidation**: Step 2 checks existing picks CSVs for the
   `base_of_scarp_position` column. If absent (v1 picks), the picks are regenerated
   automatically with the new multi-scale detector.

3. **Continuity Analysis (both orientations)** -- For each corridor and orientation,
   evaluates detection rate, lateral coherence (offset between adjacent picks along the
   section-perpendicular axis), throw-sense consistency, and run length (longest unbroken
   chain of coherent picks). Produces fence diagrams (stacked cross-sections ordered by
   the spacing axis) and pick-track plots (scarp-coordinate vs. spacing-axis) for visual
   QC. A combined `continuity_summary.csv` records per-corridor, per-orientation metrics.

4. **Trace Reconstruction & Confidence Scoring** -- Chains coherent pick runs from both
   orientations into unified (easting, northing) trace polylines via nearest-neighbour
   ordering (5 m deduplication). **v2 change**: trace chaining now uses
   `base_of_scarp_position` coordinates instead of the v1 scarp midpoint, providing a
   geomorphically better-justified trace-defining point. Falls back to the scarp
   midpoint if `base_of_scarp_position` is null.

   **v2 LSM cross-validation**: Before scoring, Step 4 loads the 2D LSM skeleton
   (from Step 5's GeoTIFF) and builds a cKDTree on all nonzero LSM pixels. For each
   detected pick, the perpendicular distance from the base-of-scarp point to the
   nearest LSM skeleton pixel is computed and written back to the picks CSV as
   `lsm_agreement_distance_m`. Close agreement (within a few metres) between the
   independent 1D and 2D derivations is strong corroboration; large disagreement
   flags the pick for review.

   Each trace is scored on a 0-12 composite scale (expanded from v1's 0-10) using
   continuous linear ramps between thresholds:

   | Component | Max | Ramp |
   |-----------|-----|------|
   | Scarp evidence (mean height) | 3 | 0.5 m -> 0, 2.0 m -> 1.0 |
   | Lateral continuity (detection rate + run length) | 2 | 0.5/0.95 and 1x/3x MIN_RUN |
   | Throw consistency (within each orientation) | 1 | binary |
   | ML latent evidence (mean ml_score of nearby tiles) | 2 | 0.1 -> 0, 0.6 -> 1.0 |
   | Drainage/slope corroboration (mean max_slope_deg) | 1 | 10 deg -> 0, 35 deg -> 1.0 |
   | Database proximity (nearest structural point) | 1 | 1000 m -> 0, 100 m -> 1.0 |
   | **Pick stability** (base-of-scarp stable across scales) | 1 | fraction of picks with high/medium confidence |
   | **LSM agreement** (median distance to LSM skeleton) | 1 | 50 m -> 0, 5 m -> 1.0 |

   Traces farther than 500 m from any cataloged structural point are flagged as novel
   discoveries. A final plan-view map plots all trace points coloured by confidence
   (discrete RdYlGn, only levels present in the data), with novel traces drawn larger
   than corroborating traces.

5. **Lineament Strength Map (LSM)** -- Computes a multi-scale curvature-persistence
   raster over the full DEM, independent of the ML latent space. The LSM serves two
   purposes: (a) it provides a geomorphologically grounded independent lineament
   detection for cross-validation against ML results, and (b) its saved GeoTIFF is
   consumed by Script 04 v2 as the source of pseudo-labels for contrastive training.

   Pipeline:
   1. Load the DEM and compute TV-denoised versions at 6 scales
      (`skimage.restoration.denoise_tv_chambolle`, weights 0.005--0.16).
   2. For each scale, compute the Laplacian curvature of the smoothed DEM.
   3. Apply Lindeberg scale-space normalisation (sigma^0.5) and take the
      maximum curvature that persists across >= 3 consecutive scales. This
      is the raw LSM --- genuine fault scarps produce persistent curvature
      spikes across multiple smoothing scales, while noise and small-scale
      artifacts do not.
   4. Build a river mask from the HAND raster (HAND < threshold) plus river
      bed/surface/valley rasters from Script 02, buffer it by 20 m, and zero
      out LSM within the mask. This is applied after curvature computation so
      that river valley morphology is not smoothed away before detection.
   5. Compute background statistics (mean, std) in a control zone > 200 m from
      known faults, then apply hysteresis thresholding: low threshold =
      mean + 2 sigma, high threshold = mean + 3 sigma.
   6. Skeletonise the hysteresis mask, extract connected components, and
      filter by minimum length (200 m) and maximum width (200 m).
   7. Save the LSM as a GeoTIFF and a lineaments CSV.

   The LSM is cached at `validation/lsm/{dem_source}_lineament_strength_map.tif`
   and loaded by `load_or_compute_lsm()` on subsequent runs. Script 04 v2 reads
   this file to derive per-tile mean LSM values for pseudo-label construction
   (see Script 04 v2 documentation above).

### Key Design Decisions

- **Class 1 only seeding**: Only Class 1 reference tiles (well-constrained scarps)
  are used as query anchors for candidate retrieval. Class 3 references are themselves
  uncertain, and using them as anchors would propagate that uncertainty into the
  candidate pool.
- **Class 1 weight via Script 02 × Script 08 cross-referencing**: Not all Class 1
  reference tiles are equally reliable. Some trace river banks (high Smf) or sit in
  Quaternary-only / no-lithology-contrast geological contexts that are unlikely to
  host bedrock faults. Each Class 1 tile is spatially cross-referenced against
  Script 02's Smf and Script 08's geological plausibility, then down-weighted or
  dropped. The weight multiplies `ml_score` before corridor detection, so
  river-like or geologically unsupported references contribute less to KDE density
  and to downstream confidence scoring. This mitigates the source-dominance bias
  noted in Limitation #13.
- **No HDBSCAN cluster requirement**: A tile cosine-similar to a Class 1 reference is
  evidence enough on its own. Requiring cluster co-membership would double-count the
  same latent-space signal and needlessly shrink recall.
- **Bidirectional sections**: Fixed E-W and N-S sections are the DEM analogue of
  seismic inlines and crosslines. Two orientations ensure that faults of any trend are
  crossed near-perpendicularly by at least one set; the `poor_geometry_flag` /
  `poor_geometry_flag_east` pair indicates which orientation is sub-parallel for each
  corridor.
- **Sinuosity at corridor stage, not just trace stage**: Flagging meander-suspect
  corridors before section extraction prevents wasting LiDAR profiling on river
  meanders. The Script 02 trace-level `ref_smf` is the primary indicator; the NN-chain
  Smf is a fallback when no Script 02 trace matches.
- **Source dominance awareness**: GeoStru501 (377 traces) dominates candidate seeding
  by default, since FSD1 has only 5 traces. The script reports per-source retrieval
  counts so users can assess whether corridors are broadly representative or an
  artifact of one source dominating the reference pool.

### Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| CLASS1_ONLY | True | Restrict candidate seeding to Class 1 reference tiles |
| COSIM_TOP_K | 200 | Top-k neighbours per reference tile |
| CORRIDOR_KDE_BANDWIDTH_M | 2000 | KDE bandwidth for corridor hotspot detection |
| CORRIDOR_MIN_DENSITY_PCTILE | 75 | Percentile threshold on KDE surface |
| CORRIDOR_BUFFER_M | 500 | Buffer around corridor bounding box |
| FALLBACK_CHAIN_DIST_M | 200 | Max gap between tiles in a fallback chain |
| FALLBACK_MIN_CHAIN_TILES | 3 | Min tiles to form a fallback corridor |
| CORRIDOR_SINUOSITY_FLAG_THRESH | 2.0 | NN-chain Smf above this flags meander |
| CORRIDOR_REF_SMF_FLAG_THRESH | 5.0 | Script 02 ref_smf above this flags meander |
| CORRIDOR_PERIODICITY_FLAG_THRESH | 0.35 | Turning-angle autocorrelation threshold |
| CLASS1_SMF_DOWNGRADE_THRESH | 5.0 | Smf above this down-weights Class 1 ref tiles |
| CLASS1_GEO_DOWNGRADE_LABELS | no_lithology_contrast, insufficient_evidence | Script 08 plausibility labels that trigger down-weighting |
| CLASS1_GEO_DOWNGRADE_AGES | Quaternary | Unit age classes that trigger down-weighting |
| CLASS1_FULL_WEIGHT | 1.0 | Weight for clean Class 1 refs (no flags) |
| CLASS1_PARTIAL_WEIGHT | 0.5 | Weight for poor-geology-only refs |
| CLASS1_LOW_WEIGHT | 0.2 | Weight for high-Smf-only refs |
| CLASS1_DROP_WEIGHT | 0.0 | Weight for refs with both high Smf and poor geology |
| FSD1_PROXIMITY_THRESHOLD_M | 1000 | Tiles within this distance flagged FSD1-adjacent |
| FAULT_VALIDATION_TOLERANCE_M | 200 | Allowed relocation from known fault position |
| ORIENTATIONS | ["north", "east"] | E-W and N-S section orientations |
| SECTION_SPACING_M | 25 | Spacing between sections |
| SECTION_HALF_LENGTH_M | 350 | Half-length of each section |
| SECTION_STRIKE_FLAG_THRESHOLD_DEG | 30 | Azimuth threshold for poor-geometry flag |
| MIN_SCARP_HEIGHT_M | 0.5 | Minimum detectable scarp height |
| SCARP_WIDTH_DISCRETE_MAX_M | 200 | Discrete vs. monocline threshold |
| PICK_LATERAL_JUMP_MAX_M | 40 | Max offset between adjacent coherent picks |
| PICK_GAP_TOLERANCE | 2 | Consecutive non-detections before run breaks |
| MIN_RUN_LENGTH_SECTIONS | 5 | Min coherent picks to promote to trace |
| THROW_SENSE_TOLERANCE | 1 | Max vergence flips before run splits |
| LSM_TV_WEIGHTS | [0.005, 0.01, 0.02, 0.04, 0.08, 0.16] | TV denoising strengths (multi-scale) |
| LSM_PERSISTENCE_K | 3 | Min consecutive scales for a pixel to survive |
| LSM_SCALE_NORM_GAMMA | 0.5 | Lindeberg scale-space normalisation exponent |
| LSM_HAND_BUFFER_M | 20 | Buffer around HAND-flagged river pixels before masking |
| LSM_HYSTERESIS_LOW_SIGMA | 2.0 | Low threshold = mean + 2 sigma of background |
| LSM_HYSTERESIS_HIGH_SIGMA | 3.0 | High threshold = mean + 3 sigma of background |
| LSM_MIN_LENGTH_M | 200 | Min lineament length (m) |
| LSM_MAX_SCARP_WIDTH_M | 200 | Max discrete scarp width (m) |
| LSM_FAULT_BUFFER_M | 200 | Buffer around known faults for control-zone exclusion |
| SCARP_TV_WEIGHTS | [0.005, 0.01, 0.02, 0.04, 0.08, 0.16] | TV denoising strengths for 1D scarp detection (reuses LSM_TV_WEIGHTS) |
| SCARP_STABILITY_PIX | 2 | Max position drift (pixels) for base-of-scarp to be considered stable across scales |
| SCARP_BG_MARGIN_FRAC | 0.1 | Fraction of profile ends used as background curvature reference |
| CONF_WEIGHT_PICK_STABILITY | 1 | Weight for pick stability in composite confidence score |
| CONF_WEIGHT_LSM_AGREEMENT | 1 | Weight for LSM agreement in composite confidence score |
| CONF_MAX_TOTAL | 12 | Maximum composite confidence score (v2: expanded from 10) |

### Command-Line Usage

```bash
# Full run for default DEM source (dem10m)
python 05-MLTraceReconstruction.py

# Run for a specific DEM source
python 05-MLTraceReconstruction.py --dem-source lidar_0_5m
python 05-MLTraceReconstruction.py --dem-source srtm1s

# Resume from a specific step (1=candidates, 2=sections, 3=continuity, 4=traces, 5=LSM)
python 05-MLTraceReconstruction.py --step 2

# Use Script 04 v2 outputs (10-channel autoencoder, versioned tile index)
python 05-MLTraceReconstruction.py --dem-source srtm1s --model-version v2
```

Prerequisites: Script 04 must have completed for the matching `--dem-source`
(requires `tile_index.parquet` [v1] or `tile_index_v2.parquet` [v2],
`latent_vectors.npy` from `autoencoder/` [v1] or `autoencoder_v2/` [v2], and the
quality CSV from Phase 1 audit). The native DEM for the chosen source is used for
section sampling at full morphological resolution.

The script is idempotent: Step 1 caches `candidate_tiles.parquet` and skips cosine
similarity recomputation on re-run. If the cached parquet lacks `class1_weight` or
`best_ref_tile_idx` (pre-weighting versions), it is automatically rebuilt. Step 2
caches `picks_north.csv` / `picks_east.csv` per corridor and skips section extraction
if found; **v2 cache check**: if existing picks lack the `base_of_scarp_position`
column (v1 picks), they are automatically regenerated with the new multi-scale
detector. Steps 3-4 always re-run from cached picks (fast).

**`--model-version` flag**: Controls which Script 04 outputs are loaded. `v1`
(default) reads `tile_index.parquet` and `autoencoder/latent_vectors.npy`. `v2`
reads `tile_index_v2.parquet` and `autoencoder_v2/latent_vectors.npy`. The v2
multi-scale 1D scarp detection and LSM cross-validation features are active
regardless of model version (they operate on the raw DEM, not the ML stack), but
using `--model-version v2` ensures the candidate tiles and latent-space scores
come from the 10-channel model.

### Output

All outputs go to `data/derivatives/ml_trace_reconstruction_{dem_source}/`.

| Directory / File | Content |
|------------------|---------|
| `candidate_tiles.parquet` | Consolidated candidate tile table with ml_score, ml_score_original, class1_weight, class1_weight_reason, n_class1_retrievals |
| `candidate_tiles_by_source.csv` | Per-source retrieval breakdown |
| `source_dominance_summary.csv` | Source dominance in candidate pool |
| `corridors/corridor_index.csv` | corridor_id, bbox, area, azimuth, sinuosity, meander_flag, near_fsd1, is_fallback, poor_geometry_flag, poor_geometry_flag_east |
| `corridors/continuity_summary.csv` | Per-corridor, per-orientation continuity metrics |
| `corridors/figures/candidate_tiles_planview.png` | Candidates on hillshade (sized by retrievals) |
| `corridors/figures/corridor_density_map.png` | KDE hotspots with corridor bboxes (colour-coded) |
| `sections/corridor_{N}/picks_north.csv` | E-W section picks (northing, scarp_easting, height, width, class, throw_sense, **crest_position, base_of_scarp_position, scale_used, position_stability_range, pick_confidence, lsm_agreement_distance_m**) |
| `sections/corridor_{N}/picks_east.csv` | N-S section picks (easting, scarp_northing, height, width, class, throw_sense, **crest_position, base_of_scarp_position, scale_used, position_stability_range, pick_confidence, lsm_agreement_distance_m**) |
| `sections/corridor_{N}/section_{NNNNNN}N.png` | Elevation profile with scarp zone annotated (north orientation) |
| `sections/corridor_{N}/section_{NNNNNN}E.png` | Elevation profile with scarp zone annotated (east orientation) |
| `sections/corridor_{N}/corridor_{N}_fence_north.png` | Stacked E-W cross-sections (fence diagram, north) |
| `sections/corridor_{N}/corridor_{N}_fence_east.png` | Stacked N-S cross-sections (fence diagram, east) |
| `sections/corridor_{N}/corridor_{N}_pick_track_north.png` | Scarp easting vs. northing (north) |
| `sections/corridor_{N}/corridor_{N}_pick_track_east.png` | Scarp northing vs. easting (east) |
| `traces/ml_inferred_fault_traces.csv` | trace_id, length, azimuth, confidence_score, novel_flag, nearest_db_source, score_breakdown |
| `traces/ml_inferred_trace_coords.csv` | Per-point coordinates (trace_id, point_idx, easting, northing) |
| `traces/ml_inferred_fault_map.png` | Plan-view hillshade with ML traces coloured by confidence |
| `validation/corridor_qc_flags.csv` | poor_geometry_north/east, meander_suspect, fallback_corridor, source_dominance_warning, low_class1_coverage |
| `validation/fault_trace_overlay/fault_trace_validation.csv` | Per-trace validation: n_candidates, n_picks, mean/median offset, has_ml_evidence |
| `validation/fault_trace_overlay/relocation_assessment.csv` | Relocation confidence rating per trace |
| `validation/fault_trace_overlay/fault_validation_{source}.png` | Per-database overlay figure (4 files) |
| `validation/fault_trace_overlay/fault_validation_summary.png` | Combined summary figure (all 4 databases) |
| `validation/fault_trace_overlay/fault_trace_detail/{source}_{trace_id}.png` | Per-trace 3-panel detail: (left) map view with trace + candidates, (middle) geological units, (right) offset histogram (traces with ML evidence only) |
| `validation/fault_trace_overlay/fault_trace_detail/{source}_overview.png` | Per-source overview of all traces with ML evidence |
| `validation/class1_downweighted_samples.png` | 4 sample Class 1 ref tiles that were down-weighted (high Smf or poor geology), shown with hillshade context and nearby candidates |
| `validation/class1_fullweight_samples.png` | 4 sample Class 1 ref tiles that retained full weight (fault-like Smf, structurally supported), shown with hillshade context and nearby candidates |
| `validation/lsm/{dem_source}_lineament_strength_map.tif` | LSM GeoTIFF (multi-scale curvature persistence raster, river-masked, hysteresis-thresholded) |
| `validation/lsm/{dem_source}_lineament_strength_map_lineaments.csv` | Extracted lineament records (length, azimuth, width, bounding box) from the LSM |

### Figures (05-MLTraceReconstruction.py)

| Figure | What It Shows | Axes | How to Interpret |
|--------|-------------|------|-----------------|
| `candidate_tiles_planview.png` | Candidate tiles on hillshade, sized by number of Class 1 retrievals | x = easting (m, EPSG:28355), y = northing (m, EPSG:28355) | Larger markers = tiles retrieved by more reference tiles = stronger latent-space signal. Clusters of large markers indicate promising corridors. |
| `corridor_density_map.png` | KDE surface of candidate density with corridor bounding boxes overlaid | x = easting (m), y = northing (m), colour = KDE density (hot cmap) | Lime boxes = main KDE corridors, orange dotted = fallback corridors, magenta = meander-flagged, green = near FSD1. Labels show corridor ID, azimuth, Smf, and flags. High-density hotspots are primary profiling targets. |
| `section_{NNNNNN}N.png` | Single E-W elevation profile with detected scarp zone annotated | x = distance from section centre (m, E), y = elevation (m AHD) | Red shaded zone = scarp location; annotation gives height, width, max slope, throw sense (west_up / east_up), and pick confidence (high/medium/low). Green dotted line = crest position; orange dash-dot line = base-of-scarp position. Blue dashed line = section centre. "discrete" = sharp scarp; "monocline" = broad flexure. |
| `section_{NNNNNN}E.png` | Single N-S elevation profile with detected scarp zone annotated | x = distance from section centre (m, N), y = elevation (m AHD) | Same as north profile but throw sense is north_up / south_up. Compare with north profiles to confirm scarp geometry from both directions. |
| `corridor_{N}_fence_{orientation}.png` | Fence diagram: stacked cross-sections ordered by spacing axis | x = relative offset (m), y = section index (ordered by northing for north, easting for east) | Red bars = detected scarp positions per section. Vertically aligned bars = laterally continuous scarp. Gaps or jumps indicate discontinuities. Title shows detection rate, number of runs, longest run. |
| `corridor_{N}_pick_track_{orientation}.png` | Scarp coordinate vs. spacing axis for all picks | x = scarp easting (north) or scarp northing (east) (m), y = northing (north) or easting (east) (m) | Red dots = detected scarp positions; blue lines connect coherent runs. A straight vertical alignment of dots indicates a linear, continuous fault. Scattered dots = poor continuity. |
| `ml_inferred_fault_map.png` | Plan-view hillshade with all ML-inferred trace points coloured by confidence | x = easting (m, EPSG:28355), y = northing (m, EPSG:28355), colour = confidence score (RdYlGn, discrete) | Green = high confidence, red = low confidence. Large markers = novel traces (far from known faults), small squares = corroborating traces. Grey dots = known fault points (subsampled). |
| `fault_validation_{source}.png` | Known fault traces from one database overlaid on hillshade with ML candidates and picks | x = easting (m), y = northing (m) | Source-coloured lines = original fault traces; cyan dots = ML candidate tiles within 200 m; yellow triangles = ML section picks within 200 m. Cyan/yellow clustering near a trace confirms the ML pipeline recovers the known fault. |
| `fault_validation_summary.png` | All four databases' traces overlaid together with ML evidence | x = easting (m), y = northing (m) | Circles around trace centroids show 200 m relocation tolerance. FSD1 = green, CWvic1 = magenta, GeoStru2501 = blue, GeoStru501 = orange. Overlap between ML candidates (cyan) and trace circles indicates where ML could refine fault locations. |
| `fault_trace_detail/{source}_{trace_id}.png` | Three-panel zoomed figure per trace: (left) map view with trace, candidates coloured by ml_score, ML picks; (middle) geological units from GeoUnit 250 m grid points coloured by ABBREVIATE code, with trace overlay; (right) offset histogram | Left: x = easting (m), y = northing (m). Middle: x = easting (m), y = northing (m). Right: x = offset from trace (m), y = count | Left panel: candidate colour (YlOrRd) = ml_score, size = n_class1_retrievals. Middle panel: geological unit points coloured by tab20 cmap (no hillshade background); coloured legend (ABBREVIATE + count, up to 25 units) placed to the right of the panel via bbox_to_anchor; if > 25 units, shows total count only. Right panel: cyan histogram = candidate offsets, yellow = pick offsets; red dashed line = 200 m tolerance. Use the geology panel to check whether the trace follows a lithological contact or cuts through a single unit. |
| `class1_downweighted_samples.png` | 4 sample Class 1 reference tiles that were down-weighted due to high Smf or poor geological context | x = easting (m), y = northing (m), 4 panels | Each panel: hillshade background, cyan dots = candidate tiles, coloured star = Class 1 ref tile (by source), small circles = other refs of same source. Title shows source, trace_id, weight, and reason (e.g. "high_Smf(7.5)" or "poor_geo(no_lithology_contrast,Pliocene)"). Use to verify that down-weighted tiles are indeed river-like or geologically unsupported. |
| `class1_fullweight_samples.png` | 4 sample Class 1 reference tiles that retained full weight (fault-like Smf, structurally supported) | x = easting (m), y = northing (m), 4 panels | Same layout as down-weighted figure. Title shows source, trace_id, weight=1.0. Use to verify that full-weight tiles are genuinely fault-like with good geological context. |

---

## 05a-KnownTraceScarpValidation (dem10m / hs)

> **Note:** This script exists as two active variants in the project root:
> `05a-KnownTraceScarpValidation_dem10m.py` and `05a-KnownTraceScarpValidation_hs.py`.
> They share the same pipeline structure but use **different threshold presets**
> calibrated to their respective DEM sources. A legacy monolithic version
> (`05a-KnownTraceScarpValidation.py`) and an SRTM variant are archived in `code_old/`.

### What It Does

This script is a known-trace, non-ML validation pass that asks the inverse of
Script 05: along already catalogued fault traces, where is there independent,
pixel-scale geomorphic evidence of a scarp base in the DEM itself?

Key scope and intent:
- Uses only known structural-point databases (FSD1, CWvic1, GeoStru2501,
  GeoStru501) plus DEM morphology.
- Reuses Script 05-style multi-scale TV + Laplacian curvature logic and river/HAND
  exclusion.
- Does not import autoencoder outputs, cosine-similarity candidates, or ML picks.

### Methods

#### 1. Known-Trace Reconstruction and Prior Attachment

Structural points are loaded from `structural_points_inventory.csv`, filtered to the
four known databases, then reconstructed into ordered traces (cluster + nearest-
neighbour ordering). Phase-1 priors are attached when available:
- Expression quality class and detection rate
- Prior scarp profile seeds (`*_scarp_analysis.csv`)
- Nearby drainage anomaly context

Optional runtime controls for testing are included:
- DEM-extent clipping of points before reconstruction (`--clip-points-to-dem`)
- Point clipping buffer (`--point-buffer-m`)
- Trace and per-trace window caps (`--max-traces`, `--max-windows-per-trace`)

#### 2. Local Windowing Around Traces

For each reconstructed known trace, the script builds one or more local analysis
windows constrained by:
- DEM extent
- Prior quality class (smaller buffers for lower-prior traces)
- Maximum pixel budget per window (`MAX_LOCAL_WINDOW_PIXELS`)

Seeded windows (from prior scarp-evidence profiles) are merged with trace-following
windows and deduplicated.

#### 3. Multi-Scale Curvature Lineament Detection

Per window, the script computes a local lineament-strength signal:
- TV denoising at multiple scales (`LOCAL_TV_WEIGHTS`)
- Laplacian curvature per scale with scale normalisation
- Persistence filtering across scales (`LSM_PERSISTENCE_K`)
- Hysteresis thresholding (low/high sigma controls)
- River/HAND masking with dilation buffer to suppress channel artefacts

Candidate selection is sign-aware and tiered:
- Strongest-scale signed curvature is used to split convex/base candidates
  (positive sign) from concave/crest candidates (negative sign)
- Tier thresholds are calibrated as:
  - Tier 1: `lsm >= (low + high) / 2`
  - Tier 2: `lsm >= low`

This yields persistent candidate pixels that preserve both candidate strength and
geomorphic polarity (base vs crest intent).

#### 4. Scarp Base/Crest Slope-Break Tracing

From each candidate pixel, tracing behavior depends on curvature sign:
- Positive (base-intent) candidates: current pixel is treated as base, then traced
  uphill to find crest
- Negative (crest-intent) candidates: current pixel is treated as crest, then traced
  downhill to find base

Tracing targets geomorphic slope breaks rather than curvature-background return:
- Local slope is sampled in 2-pixel steps along the trace direction
- A break is accepted when slope persistently drops to a fraction of the starting
  slope (with a short look-ahead check for stability)
- A fallback break is used when slope re-steepens after flattening

Scarp geometry is retained only if it meets physical filters (height and width,
including minimum detectable height and maximum discrete width).

#### 5. Segment Building and Scoring

Detected scarp-base pixels are skeletonised and grouped into connected segments.
Segments are filtered by:
- Minimum length
- Sinuosity and turning periodicity
- River clearance

Each retained segment receives a 0-10 confidence score from seven components:
height/width ratio, scale persistence, sinuosity, proximity to known trace,
distance from river mask, mean asymmetry score, and candidate-tier strength.

#### 6. Prior Comparison and Coverage Summary

Each segment is cross-referenced to nearest prior scarp profile and drainage anomaly,
then trace-level coverage metrics are calculated:
- Corroborated fraction of known-trace length
- Evidence delta (confirmed/refined/discovered/unconfirmed)
- Per-database summary statistics

#### 7. Key Thresholds (Variant-Specific)

| Parameter | dem10m variant | hs variant | Purpose |
|-----------|---------------|------------|---------|
| `MIN_SCARP_HEIGHT_M` | 1.0 | 0.2 | Minimum scarp height to retain |
| `LSM_HYSTERESIS_LOW_SIGMA` | 1.5 | 0.5 (DEM>=5m: 1.0) | Low hysteresis threshold |
| `LSM_HYSTERESIS_HIGH_SIGMA` | 2.0 | 4.0 (DEM>=5m: 1.5) | High hysteresis threshold |
| `LSM_PERSISTENCE_K` | 2 | 2 | Scales a candidate must survive |
| `MAX_LOCAL_WINDOW_PIXELS` | 16,000,000 | 16,000,000 | Pixel budget per local window |
| `TARGET_TV_SIGMAS_M` | [20, 28, 40, 56, 80, 113, 160] | [20, 28, 40, 56, 80, 113, 160] | TV denoising scales (m) |

### Output

All outputs are written to:
`data/derivatives/known_trace_scarp_validation/{dem_source}/`

| File / Directory | Content |
|------------------|---------|
| `known_trace_scarp_segments.csv` | Segment-level attributes (source, known_trace_id, geometry WKT, length, morphology, confidence, prior-comparison fields) |
| `known_trace_scarp_summary.csv` | Per-trace summary (segment count, corroborated length fraction, prior class/rate, evidence delta) |
| `known_trace_scarp_summary_by_database.csv` | Source-level aggregation (number of traces, traces with scarps, mean corroborated length %) |
| `all_candidate_points.csv` | All candidate pixels (x, y) detected during lineament scanning |
| `all_inferred_points.csv` | Inferred scarp points (x, y, tier) after geometry filtering; consumed by 05b |
| `known_trace_points.csv` | Reconstructed known-trace point coordinates (for 05b overlay) |
| `window_diagnostics.csv` | Per-window diagnostic metadata (when `--write-diagnostics` enabled) |
| `classical_ml_known_trace_comparison.csv` | Optional outer join to Script 05 validation table when present |
| `known_trace_scarp_summary.png` | DEM-wide overview: known traces as scatter points and validated scarp segments as lines |
| `local_reconstruction_plots/trace_{trace_id}_chunk_{chunk}.png` | Two-panel local figures: revised scarp segments (panel 1) and original known-trace points (panel 2), each clipped to DEM extent with 1 km buffer; chunks split to <= 4 km scarp length per figure |
| `local_lsm/*.tif` | Optional per-window local LSM rasters when `--write-local-lsm` is enabled |

### Command-Line Usage

```bash
# Standard run for a selected DEM source (choose the variant matching your DEM)
python 05a-KnownTraceScarpValidation_dem10m.py --dem-source dem10m
python 05a-KnownTraceScarpValidation_hs.py --dem-source lidar_0_5m

# Quick test run (fastest path): lower-resolution DEM + strict caps
python 05a-KnownTraceScarpValidation_hs.py --dem-source srtm1s --max-traces 2 --max-windows-per-trace 2

# Optional speed-up by clipping points to DEM extent (with trace-id caveat)
python 05a-KnownTraceScarpValidation_hs.py --dem-source srtm1s --clip-points-to-dem --point-buffer-m 200 --max-traces 2 --max-windows-per-trace 2
```

Notes:
- `--clip-points-to-dem` can significantly reduce runtime but may change reconstructed
  trace IDs and reduce one-to-one joining with phase1 prior trace tables.
- The local reconstruction figures are generated automatically at the end of the run.

---

## 05b-DEMWideScarpInference.py

### What It Does

This script is an ML-based wide-area scarp inference pipeline with iterative
active learning. It reads the pixel-level candidate points produced by Script 05a
(`all_inferred_points.csv`), enriches each point with local geometric features
sampled from the DEM and HAND rasters, trains a gradient-boosted-tree classifier
on HighCon (positive) vs confirmed-removed (negative) labels, and scores the
unlabeled pool to produce probability-ranked scarp candidates.

The pipeline runs as an **iterative active-learning loop** over multiple rounds.
Each round k trains on Trace1..k as positives (plus accumulated negatives),
scores the unlabeled pool, then exports top candidates for human review. A
shape-based filter (I/L vs U/O morphology) zeros out points belonging to
folded-back or closed-loop clusters that are unlikely to be real fault scarps.
The loop supports up to 4 HighCon traces (`Trace1.xlsx` through `Trace4.xlsx`).

Key capabilities:
- **Iterative label expansion**: Round k uses Trace1..k as positives; each round
  adds the next trace and accumulates explicit negatives from prior rounds.
- **Shape/topology filtering**: Connected-component clustering of candidate
  points followed by straightness-based classification separates elongated
  I/L-shaped traces (likely faults) from U/O-shaped clusters (likely rivers or
  artefacts). Applied both as a model feature and as a post-hoc hard filter.
- **Versioned audit trail**: Each round writes to `HighCon/round_{N}/` with a
  labeled-set audit CSV, round metadata JSON, and explicit negatives CSV, so
  any round can be inspected or rolled back.
- **Spatial visualisation**: Per-round spatial score maps (ensemble score with
  HighCon trace overlay), shape filter maps, and cross-round comparison plots.

### Methods

#### Stage 0 -- Rule-Based Pre-Filters

Before any ML, the unlabeled pool is cleaned with two deterministic filters:

1. **Border exclusion** -- Erodes the valid-data mask by 20 m
   (`EDGE_BUFFER_M`) to remove edge artefacts where DEM derivatives are
   unreliable near the tile boundary.

2. **Water/flat exclusion** -- Builds a water mask from the HAND raster
   (HAND <= 0 m = river/lake bed) when available. If no HAND raster exists,
   falls back to a DEM-fill approach: areas with < 0.05 m relief in a 70 m
   window, filtered to connected components >= 1 ha to catch only genuine
   water bodies. Points inside the water mask are excluded.

The same pre-filters are applied to the negative set to ensure consistency
between training and scoring.

#### Stage 1 -- Labelled Dataset Assembly (Round-Aware)

Three data groups are assembled each round:

- **HighCon positives** -- Loaded from `data/HighCon/Trace{1..k}.xlsx` where k
  is the current round number. Each point is cross-referenced against this
  DEM source's `all_inferred_points.csv` to (a) filter out points from the
  wrong DEM (the HighCon traces blend both sources) and (b) look up the correct
  `break_type` from the CSV to avoid label leaks. Traces with no overlap to the
  current DEM source's CSV are silently skipped. Label = 1.

- **Confirmed-removed negatives** -- The set difference between the full
  `all_inferred_points.csv` and the filtered TXT file
  (`data/FilteredOutput/DEM10.txt` or `DEM05.txt`). These are points that 05a
  detected but that did not survive to the filtered output. Label = 0.

- **Explicit negatives** -- Accumulated across rounds from
  `HighCon/round_{N}/explicit_negatives.csv`. Currently auto-generated from
  the bottom 10% of the scored pool (lowest probability, capped at 2000 points
  per round); in a full workflow these would be human-reviewed rejections. They
  sharpen the negative class over successive rounds.

- **Unlabeled pool** -- The TXT file minus HighCon points. HighCon points are
  explicitly removed to prevent positive contamination of the scoring pool.
  Label = -1. The pool and its computed features are cached across rounds
  (explicit parameter pass, not `DataFrame.attrs`) to avoid recomputation.

Negatives are stratified by `tier x break_type` combination and sampled
proportionally (target ~3x positive count) to ensure the model sees different
failure modes rather than a random subsample.

#### Stage 2 -- Feature Engineering

All features are computed per point from the DEM and HAND rasters. A set of
DEM-derived rasters is precomputed once and reused across all point groups and
all rounds (saving ~7 TV denoising passes per group per round).

| Feature Group | Features | Method |
|---------------|----------|--------|
| Pull-through | `tier`, `break_type`, `break_type_p` | Inherited from 05a CSV |
| HAND | `hand_value` | Bilinear sample of HAND raster |
| Local terrain | `local_slope_deg`, `local_relief_m`, `tri` | Gradient, max-min in 3x3 window, mean abs deviation |
| Multi-scale curvature | `curv_scale_0`--`curv_scale_6`, `curv_consistency`, `curv_dominant_sign` | TV denoise at 7 scales (weights 0.0025--0.16), Laplacian curvature per scale, sign-agreement across scales |
| Perpendicular profile | `profile_max_slope`, `profile_before_slope`, `profile_after_slope`, `profile_slope_ratio` | Perpendicular profile (half-width 50 m, 5 samples) oriented along local DEM gradient; max/before/after slope along profile |
| Local geometry | `local_linearity`, `local_density` | PCA eigenvalue ratio on 50 m kNN neighbourhood (1.0 = linear string, 0.5 = isotropic cloud); neighbour count |
| **Shape/topology** | `shape_straightness`, `shape_has_cycle`, `shape_cluster_size` | Proximity graph clustering + MST backbone + straightness ratio (see below) |

Location and geology proxy columns (`x`, `y`, `trace_id`, `dist_to_trace`,
`median_distance_to_known_m`, etc.) are explicitly excluded from training to
prevent the model from learning spatial location rather than morphology.

#### Shape/Topology Feature: I/L vs U/O Classification

The unit of analysis for shape filtering is not a single point but the
connected cluster of candidate points it belongs to. The pipeline:

1. **Builds a proximity graph** over candidate points using a cKDTree with a
   connectivity radius of 20 m (`CONNECTIVITY_RADIUS_M`), comparable to point
   spacing along a trace. Edges connect points within this radius.

2. **Finds connected components** -- each component is a candidate "trace."

3. **Computes a straightness descriptor** per component using the minimum
   spanning tree (MST) as the backbone path:
   - Two-pass Dijkstra (graph diameter) finds the two farthest-apart endpoints.
   - `straightness = euclidean_distance(endpoint_A, endpoint_B) / path_length_along_MST`
   - **I shape**: straightness ~ 0.9--1.0 (endpoints far apart, path nearly straight)
   - **L shape**: straightness ~ 0.6--0.85 (one bend, endpoints still well separated)
   - **U shape**: straightness ~ 0.0--0.3 (path folds back, endpoints close together)
   - **O shape (closed loop)**: straightness ~ 0 and the component's graph
     contains a cycle (edges >= nodes), detected directly without the distance ratio.

4. **Broadcasts** the straightness, has_cycle, and cluster_size back onto
   every point in the component as model features.

This feature is used in two ways (not mutually exclusive):

- **(a) As a model feature** -- XGBoost learns the association naturally,
  since HighCon traces (I/L by construction) show high `shape_straightness`.

- **(b) As a post-hoc hard filter** in Stage 5 -- Points in clusters with
  `straightness < STRAIGHTNESS_FLOOR (0.4)` or `has_cycle = True` have their
  probability zeroed out. The raw pre-filter score is retained as
  `prob_xgb_raw` for audit purposes.

#### Stage 3 -- Spatial Leakage-Safe Validation

Standard random train/test split would leak spatially autocorrelated
information. Instead, points are binned into along-trace chunks of ~2 km
(`CHUNK_LENGTH_M`), and whole chunks are assigned to train (70%) or
validation (30%). This ensures the model is evaluated on spatially held out
segments, not just random pixels from the same trace.

#### Stage 4 -- Model Training

Two models are trained:

- **XGBoost** (primary scorer) -- 400 trees, max depth 6, learning rate 0.05,
  with `scale_pos_weight` set to the negative/positive ratio for class
  imbalance. Early stopping on validation logloss.

- **Logistic Regression** (baseline) -- StandardScaler + balanced class
  weights, used as a sanity floor.

#### Stage 5 -- Score, Shape-Filter, and Ensemble

The trained XGBoost model is applied to the unlabeled pool, producing
`prob_xgb_raw` (pre-filter) per point. The Logistic Regression model produces
`prob_lr`. An **ensemble score** is computed as the average of the two raw
probabilities: `prob_ensemble_raw = (prob_xgb_raw + prob_lr) / 2`.

The shape filter is then applied:
- Points passing the shape test (`shape_straightness >= 0.4` and no cycle)
  retain their score in `prob_xgb` and `prob_ensemble`.
- Points failing the shape test are zeroed in both `prob_xgb` and
  `prob_ensemble`, while `prob_xgb_raw` and `prob_ensemble_raw` are preserved
  for audit.

A HighCon sanity check scores all HighCon positives to verify the model
assigns high probabilities to known scarps. Score distributions are reported at
multiple thresholds (0.3, 0.5, 0.7, 0.9). If the current round is not the
final round, the top 5000 candidates are exported for human review (to be
added as HighCon positives in the next round).

#### Iterative Active-Learning Loop

The pipeline wraps the above stages in an outer loop:

```
Round k:
  1. Train XGBoost + LR on current labelled set
     (Trace1..k as positives + confirmed-removed + explicit negatives)
  2. Score unlabeled pool
  3. Apply shape filter (I/L pass, U/O zeroed)
  4. Export top candidates for human review (if not final round)
  5. Auto-generate explicit negatives from bottom 10% of scored pool
  6. Re-run from step 1 with expanded label set
```

DEM and HAND rasters are loaded once; precomputed DEM-derived rasters and
unlabeled pool features are cached and reused across rounds. After all rounds
complete, a cross-round comparison plot and summary CSV are generated.

#### Versioned Audit Directories

Each round writes to `data/HighCon/round_{N}/`:

- `labeled_set_audit.csv` -- All points in the labelled set for this round
  (x, y, tier, break_type, trace_id, round, dem_source).
- `round_meta.json` -- Round metadata (traces used, counts, AUC, feature list).
- `explicit_negatives.csv` -- Auto-generated explicit negatives for this round.

### Output

All outputs are written to `data/FilteredOutput/` (per-round) and
`data/HighCon/round_{N}/` (audit):

**Per-round outputs** (suffixed `_r{N}`):

| File | Content |
|------|---------|
| `{dem_source}_r{N}_scored_points.csv` | Full unlabeled pool with x, y, tier, break_type, prob_xgb_raw, shape_straightness, shape_has_cycle, prob_xgb, prob_lr, prob_ensemble_raw, prob_ensemble |
| `{dem_source}_r{N}_high_conf_candidates.csv` | Subset where prob_xgb >= 0.5 |
| `{dem_source}_r{N}_top_candidates.csv` | Top 5000 candidates by prob_xgb (for human review; not generated on final round) |
| `{dem_source}_r{N}_xgb_model.json` | Saved XGBoost model |
| `{dem_source}_r{N}_feature_columns.txt` | Ordered list of feature columns |

**Per-round figures** (suffixed `_r{N}`):

| Figure | What It Shows | How to Interpret |
|--------|-------------|-----------------|
| `{dem_source}_r{N}_spatial_score_map.png` | Spatial scatter of unlabeled pool coloured by `prob_ensemble` (shape-failed points excluded), with HighCon trace points overlaid as gold scatter | Red = high scarp probability, blue = low. Gold points = HighCon traces used this round. Clusters of red near gold = model agrees with known traces. |
| `{dem_source}_r{N}_shape_filter_map.png` | Spatial map of shape filter results: green = I/L survived, red = U/O filtered | Green clusters should align with known fault trends; red clusters near rivers or circular features confirm the filter is catching non-fault morphology. |
| `{dem_source}_r{N}_shap_summary.png` | SHAP feature attribution (beeswarm plot) | Features at top = most influential; red/blue = high/low feature values; rightward = pushes prediction toward positive (scarp) |
| `{dem_source}_r{N}_feature_importance.png` | XGBoost feature importance by weight and gain (side-by-side) | Weight = how often a feature is used in splits; gain = average improvement when used |
| `{dem_source}_r{N}_precision_recall.png` | Precision-recall curve on held-out spatial chunks | AP (average precision) near 1.0 = strong model |
| `{dem_source}_r{N}_score_distribution.png` | Histogram of unlabeled vs HighCon probability scores (log-scale) | Clear separation between HighCon (red, should peak high) and unlabeled (blue, should peak low) = model has learned meaningful signal |

**Cross-round outputs** (generated after all rounds complete):

| File | Content |
|------|---------|
| `{dem_source}_round_summary.csv` | Per-round: round, n_positive, n_negative, n_high_conf, auc_xgb, auc_lr |
| `{dem_source}_cross_round_comparison.png` | Left: bar chart of positive/negative/high-conf counts per round (log-scale); Right: AUC per round |

**Audit directory** (`data/HighCon/round_{N}/`):

| File | Content |
|------|---------|
| `{dem_source}_labeled_set_audit.csv` | All labelled points for this round (x, y, tier, break_type, trace_id, round, dem_source) |
| `{dem_source}_round_meta.json` | Round metadata: traces used, n_positive, n_negative, n_unlabeled, AUC, feature_cols |
| `explicit_negatives.csv` | Auto-generated explicit negatives (x, y, tier, break_type) |

### Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NUM_TRACES` | 4 | Number of HighCon traces available (Trace1--4); determines auto round count |
| `CONNECTIVITY_RADIUS_M` | 20.0 | Proximity graph radius for shape clustering (tune to point spacing) |
| `MIN_CLUSTER_SIZE` | 5 | Minimum points in a cluster to compute shape descriptors |
| `STRAIGHTNESS_FLOOR` | 0.4 | Clusters below this straightness are U/O-like; points zeroed |
| `EDGE_BUFFER_M` | 20.0 | Border exclusion buffer (reuses 05a's LSM_EDGE_BUFFER_M) |
| `HAND_WATER_THRESHOLD_M` | 0.0 | HAND <= this = water body |
| `HAND_INFLUENCE_THRESHOLD_M` | 2.0 | Low-HAND zone threshold (from 05a) |
| `PROFILE_HALF_WIDTH_M` | 50.0 | Half-width of perpendicular profile for feature extraction |
| `PROFILE_SAMPLE_COUNT` | 5 | Number of samples along perpendicular profile |
| `LOCAL_RELIEF_WINDOW_PX` | 3 | Window size for local relief / TRI (3x3) |
| `KNN_RADIUS_M` | 50.0 | Radius for local linearity / density computation |
| `KNN_MIN_NEIGHBORS` | 3 | Minimum neighbours for linearity computation |
| `TV_WEIGHTS` | [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16] | TV denoising strengths (7 scales) |
| `TARGET_TV_SIGMAS_M` | [20, 28, 40, 56, 80, 113, 160] | Target physical smoothing scales (m) |
| `CHUNK_LENGTH_M` | 2000.0 | Along-trace chunk size for spatial holdout |
| `TRAIN_FRACTION` | 0.7 | Fraction of chunks for training |
| `XGB_PARAMS.n_estimators` | 400 | Number of boosting rounds |
| `XGB_PARAMS.max_depth` | 6 | Maximum tree depth |
| `XGB_PARAMS.learning_rate` | 0.05 | Step size shrinkage |
| `XGB_PARAMS.subsample` | 0.8 | Row subsampling ratio |
| `XGB_PARAMS.colsample_bytree` | 0.8 | Column subsampling ratio |
| `XGB_PARAMS.min_child_weight` | 5 | Minimum sum of instance weight in child |

### Command-Line Usage

```bash
# Full run for 10 m DEM with auto rounds (one per trace, up to 4)
python 05b-DEMWideScarpInference.py

# Run for LiDAR 0.5 m DEM source
python 05b-DEMWideScarpInference.py --dem-source lidar_0_5m

# Quick test run: subsample unlabeled pool to 5000 points, single round
python 05b-DEMWideScarpInference.py --max-unlabeled 5000 --rounds 1

# Specify number of rounds explicitly
python 05b-DEMWideScarpInference.py --dem-source dem10m --rounds 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dem-source` | `dem10m` | DEM source to process (`dem10m` or `lidar_0_5m`) |
| `--max-unlabeled` | 0 | Subsample unlabeled pool to N points (0 = all). Useful for quick test runs. |
| `--rounds` | 0 | Number of iterative active-learning rounds (0 = auto, one per available trace). Round k uses Trace1..k as positives. |

Prerequisites: Script 05a must have completed for the matching `--dem-source`
(requires `all_inferred_points.csv` in
`data/derivatives/known_trace_scarp_validation/{dem_source}/`). The
`data/HighCon/Trace1.xlsx` through `Trace4.xlsx` files must exist (positive
labels). The filtered TXT files (`data/FilteredOutput/DEM10.txt` or
`DEM05.txt`) must exist (unlabeled pool source). The reprojected DEM and HAND
raster for the chosen source are loaded for feature engineering.

---

## 05c-MLTunedDEMWideInference.py

### What It Does

This script extends 05a-KnownTraceScarpValidation_hs.py to perform **DEM-wide
scarp-point inference** on the 0.5 m LiDAR surface. Unlike 05a (which only
processes local windows around known traces) and 05b (which trains an ML
classifier on the candidate pool), 05c tiles the entire LiDAR DEM into
overlapping 800-px tiles, runs the full curvature-LSM + slope-break classify
pipeline on each tile, and filters the resulting inferred points by proximity
to smoothed ground-truth trace lines.

The pipeline runs in four phases:

1. **Phase 1 -- DEM characteristic analysis**: Compares terrain metrics
   (relief, slope, curvature, LSM persistence) at FSD1 survey points vs random
   non-scarp locations to establish whether the DEM physically resolves the
   scarp signal.
2. **Phase 2 -- Parameter tuning**: Uses Optuna (or grid search fallback) to
   maximise the fraction of FSD1 points that have an LSM candidate within 30 m
   (candidate-coverage objective), with a density penalty to prevent the
   candidate mask from drowning the classify step.
3. **Phase 3 -- Alternative candidate sources**: Merges 18 independent
   candidate-detection methods (multi-scale curvature, edge detection,
   topographic openness, aspect change, adaptive slope, TPI, TRI, DoG,
   geomorphon, Hough, orientation consistency, phase congruency, ridge crest,
   supervised ML probability, subtle slope break, convexity profile, gentle
   slope anomaly, multi-scale relief anomaly, and slope-profile curvature)
   into the LSM candidate mask via logical OR before classification.
4. **DEM-wide inference**: Runs the full 05a classify pipeline (`_local_lsm` +
   `_classify_scarp_pixels`) on every tile with the tuned parameters, then
   deduplicates, applies a per-trace proximity filter, and evaluates quality.

Ground truth: HighCon Trace1.xlsx + Trace2.xlsx (both filtered to 0.5 m
LiDAR CSV). Trace2 is replaced by FSD1 structural points
(`structural_points_inventory.csv`, source = `FSD1`) for smoothing, because
FSD1 points are field-surveyed and far cleaner (81% PCA linearity vs <50%
for HighCon Trace2). The original HighCon Trace2 points are retained for
evaluation.

### Methods

#### Step 1 -- Smoothed Trace Line Construction

Scattered HighCon (and FSD1) points are ordered and smoothed into continuous
guide traces:

1. **PCA-based monotonic ordering** (`_order_points_monotonic`): Points are
   centred, the first principal component is computed via SVD, and all points
   are projected onto this axis and sorted. For N-S trending fault traces
   this produces a single north-to-south pass without the back-tracking
   caused by greedy nearest-neighbour chaining.

2. **B-spline smoothing** (`_fit_smoothing_spline`): Cumulative arc length is
   used as the parameter. `scipy.interpolate.splprep` fits a B-spline (k=3 or
   lower) with a smoothing factor (`s=50`). The spline is resampled at 50 m
   intervals. Requires >= 10 unique points; otherwise falls back to raw
   ordered points.

3. **Trace 2 FSD1 replacement**: When `TRACE2_REPLACE_WITH_FSD1 = True`
   (default), Trace2's HighCon points are replaced by FSD1 points within
   `FSD1_TRACE2_BUFFER_M = 2000 m` of the HighCon Trace2 coordinates. The
   smoothed line is built from FSD1; the original HighCon points are kept in
   `raw_points_highcon` for fair quality evaluation.

4. **Per-trace buffer assignment**: Trace1 gets `TRACE_BUFFER_M = 100 m`
   (tight, points must be on the trace). Trace2 (FSD1) gets
   `TRACE2_BUFFER_M = 300 m` (wider, to account for FSD1's positional
   uncertainty vs the smoothed line).

#### Phase 1 -- DEM Characteristic Analysis

`run_dem_characteristic_analysis` extracts terrain metrics at FSD1 survey
points and at random non-scarp points (> 500 m from any FSD1 point):

- **DEM profiles** (`_extract_dem_profiles`): For each point, a 200 m
  transect is extracted along the local DEM gradient. Metrics: local relief
  (max - min), slope max/mean, profile curvature, plan curvature.
- **LSM persistence** (`_extract_persistence_at_points`): The full 05a LSM
  pipeline (TV denoise + Laplacian + persistence + hysteresis) is run on a
  300 m window around each point. Metrics: LSM value at the point, candidate
  status (boolean), persistence count.
- **Comparison**: FSD1 vs random distributions are compared via summary
  statistics (mean/median/std) and visualised as violin/box plots. The key
  metric is candidate coverage: what percentage of FSD1 points have an LSM
  candidate within the window vs what percentage of random points do.

#### Phase 2 -- Candidate-Coverage Parameter Tuning

`tune_candidate_coverage` optimises LSM parameters to maximise the fraction
of FSD1 survey points that have an LSM candidate pixel within 30 m, with a
penalty for excessive candidate density.

- **Objective function** (`_evaluate_candidate_coverage`): For each candidate
  parameter set, the LSM pipeline is run on 4-6 sampled windows near the
  traces. Coverage = (FSD1 points with candidate within 30 m) / (total FSD1
  points). Density penalty = max(0, candidate_density - 5%) * 10. Score =
  coverage - penalty.
- **Optimiser**: Optuna TPE sampler with 30 trials (default). Falls back to
  grid search if Optuna is not installed. The search space covers 8
  parameters: `MIN_SCARP_HEIGHT_M`, `LSM_HYSTERESIS_LOW_SIGMA`,
  `LSM_HYSTERESIS_HIGH_SIGMA`, `LSM_PERSISTENCE_K`, `TV_WEIGHT_SCALE`,
  `BASE_RAY_LENGTH_M`, `MIN_SCARP_ASYMMETRY_SCORE`, `MIN_SCARP_WIDTH_M`.
- **Output**: Best parameters are saved to `tuned_parameters.json` with the
  full trial log in `tuning_results_log.csv`.

#### Phase 3 -- Alternative Candidate Sources

18 independent detection methods are merged into the LSM candidate mask
via logical OR before classification. Each method is independently toggled
via command-line flags (`--no-curvature`, `--no-edge`, etc.).

| Source | Function | Key Parameters | What It Detects |
|--------|----------|---------------|-----------------|
| Multi-scale curvature | `_compute_multiscale_curvature_candidates` | sigma 1/3/5, p90 | High-curvature pixels across 3 Gaussian scales |
| Edge detection | `_compute_edge_detection_candidates` | sigma 2.0, p88 | Sobel/Scharr edge magnitude |
| Topographic openness | `_compute_topographic_openness_candidates` | 30/60/100 m annulus, p88 | Differential openness (ridges vs valleys) |
| Aspect change | `_compute_aspect_change_candidates` | sigma 3.0, p90 | Sharp aspect discontinuities |
| Adaptive slope | `_compute_adaptive_slope_candidates` | 150 m window, k=2, p90 | Local slope anomalies vs neighbourhood |
| TPI | `_compute_tpi_candidates` | 50/100/200 m windows, k=2, p90 | Topographic position index (elev vs neighbourhood mean) |
| TRI | `_compute_tri_candidates` | p92 | Terrain ruggedness index (mean abs deviation) |
| DoG | `_compute_dog_candidates` | (1,3)/(2,5)/(3,7), p90 | Difference of Gaussians band-pass |
| Geomorphon | `_compute_geomorphon_candidates` | flat=0.5 | 10-bin terrain classification (disabled by default) |
| Hough | `_compute_hough_candidates` | 20 deg, 50 m, 20 m gap | Straight-line segment voting |
| Orientation consistency | `_compute_orientation_consistency_candidates` | p90 | Coherent gradient direction clusters |
| Phase congruency | `_compute_phase_congruency_candidates` | phasepack | Invariant edge detection (disabled by default) |
| Ridge crest | `_compute_ridge_crest_candidates` | p90 | Local maxima of profile curvature |
| ML probability | `_compute_ml_probability_candidates` | XGBoost, threshold 0.5 | Supervised classifier on 18 DEM features, trained on FSD1 vs random |
| Subtle slope break | `_compute_subtle_slope_break_candidates` | min_relief 0.5, max_relief 12, width 50-400, p80 | Gentle scarps with low relief over wide spans |
| Convexity profile | `_compute_convexity_profile_candidates` | p88 | Cross-profile convexity anomalies |
| Gentle slope anomaly | `_compute_gentle_slope_anomaly_candidates` | base 0.3-6%, anomaly 1.1, p75 | Slope deviations from gentle background |
| Multi-scale relief | `_compute_multiscale_relief_anomaly_candidates` | 50/100/200 m, p90 | Relief anomalies at multiple neighbourhood scales |
| Slope-profile curvature | `_compute_slope_profile_curvature_candidates` | inflection delta 0.002, p88 | Inflection points detected by slope-direction reversal |

**Auto-relax**: When >= 2 new sources are enabled, the script automatically
halves `MIN_SCARP_ASYMMETRY_SCORE`, `MIN_SCARP_WIDTH_M`, and
`MIN_SCARP_HEIGHT_M` to let more candidates through the classify step.

#### DEM-Wide Inference (`run_dem_wide_inference`)

1. **Tile generation**: The DEM is tiled into 800-px tiles with 100-px
   overlap. When `--limit-to-trace2-buffer` is set, tiles are restricted to
   a rectangle around Trace2.

2. **Per-tile processing** (for each tile):
   - Read the DEM window. Skip if < 100 valid pixels.
   - Compute the LSM pipeline (`_05a._local_lsm`) with tuned parameters.
     This produces: TV-denoised DEM, multi-scale Laplacian curvature,
     persistence map, hysteresis-thresholded candidate mask, signed
     curvature, and gradient field.
   - **FSD1 seed injection** (if `--no-fsd1-seeds` is not set, which is
     the default): FSD1 survey points within the tile are injected into the
     candidate mask as additional seeds, so the classify step evaluates them
     even where LSM hysteresis did not produce a candidate. Default is
     **disabled** (`--no-fsd1-seeds` defaults to `True`) for a blind method.
   - **Alternative source merge**: All 18 alternative candidate sources are
     computed on the tile and merged via logical OR into the candidate mask.
   - **Classification** (`_05a._classify_scarp_pixels`): Each candidate pixel
     is classified by:
     - **Hard filters** (reject before ray tracing): river mask, low gradient
       (< 1e-6), ambiguous curvature sign.
     - **Ray tracing**: Walks uphill/downhill along the gradient direction to
       find slope breaks (crest and base). A slope break is accepted when
       slope persistently drops to a fraction of the starting slope, with a
       short look-ahead for stability. An inflection-based detection
       (slope-direction reversal) is also used for gentle scarps.
     - **Physical filters** (reject after ray tracing): height < minimum,
       width > maximum or <= 0, out-of-bounds.
     - **Soft fallback** (B1): When ray tracing fails to find crest or base,
       but the candidate has strong LSM curvature (>= medium threshold) and
       a minimum elevation drop (>= 0.05 m over 50-150 m walk), it is
       accepted as a lower-confidence point (`classify_method =
       "curvature_fallback"`). Otherwise rejected with reason
       `no_crest_fallback_rejected` or `no_base_fallback_rejected`.
     - **Success**: When both crest and base are found and pass all
       physical filters, the point is recorded with
       `classify_method = "ray_traced"`.
   - **ML probability filtering** (if enabled): If a trained ML model exists,
     inferred pixels are filtered by ML probability > threshold (default 0.5).
   - **Diagnostics**: Per-candidate diagnostics (rejection reason, classify
     method, height, width, asymmetry, etc.) are collected into
     `classify_diagnostics.csv`.

3. **Post-processing**:
   - **Deduplication**: Overlap zones produce duplicate points; dedup by
     rounded (x, y) key.
   - **Proximity filter**: Points are filtered by distance to the nearest
     smoothed trace line, using the per-trace buffer (100 m for Trace1,
     300 m for Trace2). Points outside all trace buffers are removed.
   - Outputs are saved as `dem_wide_inferred_points.csv` (pre-proximity)
     and `T1T2_inferred_points.csv` (post-proximity).

#### Step 4 -- Quality Evaluation

`_evaluate_inference_quality` compares inferred points against ground truth:

- **Recall**: Fraction of ground-truth (HighCon) points that have an inferred
  point within `match_radius = 100 m`.
- **Precision**: Fraction of inferred points that have a ground-truth point
  within `match_radius = 100 m`.
- **F1**: Harmonic mean of precision and recall.
- **Per-trace evaluation**: For Trace2 (FSD1), both the original HighCon
  Trace2 points and the FSD1 survey points are evaluated separately.
- **Point-level masks**: Each HighCon point is flagged as recovered/missed,
  and each inferred point is flagged as on-trace/off-trace, for visualisation.

#### Step 5 -- Visualisation

A single-panel overview plot is generated showing:
- Hillshade background
- Smoothed trace lines (coloured per trace)
- HighCon raw points coloured by recovery status (green = recovered,
  grey = missed)
- Inferred points coloured by precision status (red = on-trace, blue =
  off-trace)

### Output

All outputs are written to:
`data/derivatives/known_trace_scarp_validation/lidar_0_5m/`

| File / Directory | Content |
|------------------|---------|
| `dem_wide_inferred_points.csv` | All inferred scarp points (x, y, tier, break_type, classify_method, ml_probability) **before** trace-proximity filtering. This is the full raw output of the classify pipeline after dedup. |
| `T1T2_inferred_points.csv` | Inferred points **after** trace-proximity filtering (within 100 m of Trace1 or 300 m of Trace2). This is the final filtered output. |
| `dem_wide_candidate_points.csv` | All LSM candidate pixels (x, y) before classification. These are the raw candidates from the LSM + alt-source merge. |
| `classify_diagnostics.csv` | Per-candidate diagnostics: row, col, x, y, tier, local_slope, grad_mag, curvature_sign, ray_length_m, base_found, crest_found, height_m, width_m, asymmetry, touches_river, rejection_reason, classify_method, tile_idx. One row per candidate pixel across all tiles. |
| `tuned_parameters.json` | Best-found parameters from Phase 2 tuning, with default params and search space for comparison. |
| `tuning_results_log.csv` | Per-trial parameter set and objective score from tuning. |
| `inference_quality.json` | Quality metrics: recall, precision, F1, n_inferred, n_highcon_total, n_highcon_recovered, per-trace stats. |
| `tile_diagnostics.csv` | Per-tile metadata: tile_idx, col_off, row_off, valid_pixels, persistent_pixels, n_pixels, status. |
| `dem_characteristics.csv` | Per-point terrain metrics at FSD1 and random locations (Phase 1). |
| `dem_characteristics_summary.json` | Summary statistics comparing FSD1 vs random distributions. |
| `dem_characteristics_comparison.png` | Violin/box plots comparing FSD1 vs random terrain metrics. |
| `smoothed_traces_overview.png` | Raw HighCon points + smoothed B-spline lines overlaid on hillshade. |
| `dem_wide_inference_overview.png` | Single-panel overview: hillshade + smoothed traces + HighCon recovery + inferred on-trace/off-trace. |
| `tile_lsm/*.tif` | Per-tile LSM rasters (only when `--write-tiles` enabled). |
| `ml_candidate_model.pkl` | Trained XGBoost candidate model (when ML probability source is enabled). |

**How to read `classify_diagnostics.csv`**:

The `rejection_reason` column is a **diagnostic label**, not exclusively a
rejection. It records the outcome of each candidate pixel's classification
attempt:

| Rejection Reason | Meaning | Became Inferred? |
|------------------|---------|-----------------|
| `success` | Ray tracing found both crest and base; all physical filters passed (primary break, `break_type="p"`) | Yes (`ray_traced`) |
| `success_intermediate` | A secondary slope break was also found on the same candidate pixel (`break_type="i"`); this is a *separate* inferred point derived from the same candidate | Yes (`ray_traced`) |
| `fallback_accepted` | Ray tracing failed but curvature fallback passed (strong LSM + min height) | Yes (`curvature_fallback`) |
| `river_masked` | Pixel is inside the river/water mask | No |
| `low_gradient` | Gradient magnitude < 1e-6 (flat terrain) | No |
| `ambiguous_sign` | Curvature sign is 0 (neither convex nor concave) | No |
| `no_crest_fallback_rejected` | Ray tracing found no crest, and fallback criteria were not met | No |
| `no_base_fallback_rejected` | Ray tracing found no base, and fallback criteria were not met | No |
| `height_fail` | Scarp height < `MIN_SCARP_HEIGHT_M` | No |
| `width_fail` | Scarp width > `SCARP_WIDTH_DISCRETE_MAX_M` or <= 0 | No |
| `out_of_bounds` | Crest or base pixel is outside the DEM window | No |

> **Note on 1:1 correspondence**: Each candidate pixel produces exactly one
> primary diagnostic row (`success`, `fallback_accepted`, or a rejection
> reason). If the primary break succeeds and a secondary intermediate break is
> also found, the intermediate break appends an *additional* inferred point
> (`break_type="i"`) and an *additional* diagnostic row
> (`rejection_reason="success_intermediate"`). This ensures the diagnostics
> CSV and the inferred-points CSV have a 1:1 row correspondence.

The `classify_method` column records how the point was classified:
`ray_traced` (primary), `curvature_fallback` (soft fallback), or `none`
(rejected).

**How to read `dem_wide_inference_overview.png`**:

- **Green scatter**: HighCon ground-truth points recovered (have an inferred
  point within 100 m). Should cluster along the smoothed trace lines.
- **Grey scatter**: HighCon points missed. Investigate these to understand
  why the pipeline failed to detect the scarp at those locations.
- **Red scatter**: Inferred points that are on-trace (within 100 m of a
  HighCon point). These are true positives.
- **Blue scatter**: Inferred points that are off-trace (no HighCon point
  within 100 m). These are either false positives or real discoveries beyond
  the known-trace database. Check spatial context: if they cluster along a
  plausible geomorphic trend, they may be real; if they scatter randomly or
  near rivers, they are likely artefacts.

### Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `TILE_SIZE_PX` | 800 | Tile size for DEM-wide inference (~8 km at 10 m/px) |
| `TILE_OVERLAP_PX` | 100 | Tile overlap to avoid edge effects (~1 km) |
| `TRACE_BUFFER_M` | 100.0 | Per-trace proximity buffer for Trace1 (HighCon) |
| `TRACE2_BUFFER_M` | 300.0 | Per-trace proximity buffer for Trace2 (FSD1) |
| `SMOOTH_SPLINE_S_FACTOR` | 50.0 | B-spline smoothing factor (higher = smoother) |
| `SMOOTH_RESAMPLE_INTERVAL_M` | 50.0 | Point spacing on resampled smoothed trace |
| `MIN_POINTS_FOR_SPLINE` | 10 | Minimum scattered points to fit a spline |
| `FSD1_TRACE2_BUFFER_M` | 2000.0 | Buffer for filtering FSD1 points to Trace2 proximity |
| `TUNING_N_TRIALS` | 30 | Number of Optuna trials (or grid size per dimension) |
| `TUNING_SAMPLE_WINDOWS` | 6 | Windows to sample per tuning trial |
| `MATCH_RADIUS_M` | 100.0 | Radius for quality evaluation (recall/precision) |

### Command-Line Usage

```bash
# Full run: all phases (1=analysis, 2=tuning, 3=alt candidates, inference)
python 05c-MLTunedDEMWideInference.py

# Run only the inference step (skip tuning and analysis, use defaults)
python 05c-MLTunedDEMWideInference.py --phase inference

# Run phases 1 and 2 only (analysis + tuning, no inference)
python 05c-MLTunedDEMWideInference.py --phase 2

# Restrict inference to a rectangle around Trace2 (faster for testing)
python 05c-MLTunedDEMWideInference.py --limit-to-trace2-buffer --trace2-search-buffer-m 2000

# Disable a specific alternative candidate source
python 05c-MLTunedDEMWideInference.py --no-hough --no-ml-probability

# Use F1-based tuning objective instead of candidate coverage
python 05c-MLTunedDEMWideInference.py --tune-objective f2

# Enable FSD1 seed injection (diagnostic mode, not blind)
python 05c-MLTunedDEMWideInference.py --no-fsd1-seeds
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dem-source` | `lidar_0_5m` | DEM source (only `lidar_0_5m` supported) |
| `--phase` | `all` | Which phase to run: `1` (analysis), `2` (tuning), `3` (alt candidates), `inference` (DEM-wide only), `all` (1+2+3+inference) |
| `--n-trials` | 30 | Number of Optuna trials for parameter tuning |
| `--skip-tuning` | False | Skip tuning, use default parameters |
| `--tune-objective` | `candidate_coverage` | Tuning objective: `candidate_coverage` (Phase 2) or `f2` (legacy F1-based) |
| `--trace-buffer-m` | 100.0 | Buffer around Trace1 for filtering inferred points |
| `--trace2-buffer-m` | 300.0 | Buffer around Trace2 for filtering inferred points |
| `--no-fsd1-seeds` | True (disabled) | Disable FSD1 seed injection into candidate mask (default: disabled for blind method) |
| `--fsd1-seed-accept-radius-m` | 30.0 | Radius around each FSD1 point to seed as candidate |
| `--limit-to-trace2-buffer` | False | Restrict DEM-wide inference to a rectangle around Trace2 |
| `--trace2-search-buffer-m` | 2000.0 | Buffer to define the restricted search area |
| `--plot-trace2-buffer-m` | 2000.0 | Buffer around Trace2 for plotting (0 = full extent) |
| `--write-tiles` | False | Save per-tile LSM rasters (for debugging) |
| `--sensitivity` | False | Run criterion sensitivity analysis |
| `--ml-model-path` | None | Path to save/load the trained ML candidate model |
| `--ml-probability-threshold` | 0.5 | ML probability threshold for filtering |
| `--ml-filter-mode` | `or` | How to use ML: `filter` (AND, deprecated) or `or` (OR, default) |
| `--no-curvature` | False | Disable multi-scale curvature source |
| `--no-edge` | False | Disable edge detection source |
| `--no-openness` | False | Disable topographic openness source |
| `--no-aspect` | False | Disable aspect change source |
| `--no-adaptive-slope` | False | Disable adaptive slope source |
| `--no-tpi` | False | Disable TPI source |
| `--no-tri` | False | Disable TRI source |
| `--no-dog` | False | Disable DoG source |
| `--geomorphon` | False | Enable geomorphon source (disabled by default) |
| `--no-hough` | False | Disable Hough lineament source |
| `--no-orientation` | False | Disable orientation consistency source |
| `--phase-congruency` | False | Enable phase congruency source (disabled by default) |
| `--no-ridge-crest` | False | Disable ridge crest source |
| `--no-ml-probability` | False | Disable supervised ML probability source |
| `--no-subtle-slope-break` | False | Disable subtle slope break source |
| `--no-convexity-profile` | False | Disable convexity profile source |
| `--no-gentle-slope-anomaly` | False | Disable gentle slope anomaly source |
| `--no-multiscale-relief` | False | Disable multi-scale relief anomaly source |
| `--no-slope-profile-curvature` | False | Disable slope-profile curvature (inflection) source |

Prerequisites: Script 05a must have completed for `lidar_0_5m` (requires
`all_inferred_points.csv` in
`data/derivatives/known_trace_scarp_validation/lidar_0_5m/`). The reprojected
LiDAR DEM (`lidar_0_5m_epsg28355_10m_smoke.tif`) must exist in
`data/derivatives/`. The HighCon trace files (`Trace1.xlsx`, `Trace2.xlsx`)
must exist in `data/HighCon/`. The structural points inventory
(`structural_points_inventory.csv`) must exist in `data/`.

---

## 06a-ScarpTraceCleaning.py

### What It Does

This script is a preprocessing pipeline that converts raw, unordered scarp trace
point clouds (`data/HighCon/Trace*.xlsx`) into cleaned, ordered polylines stored
as one CSV per segment (`data/HighCon/cleaned_traces/trace_XX.csv`). The output
is consumed directly by `06b-ScarpParameterProfiling.py`.

The raw Excel files contain tens of thousands of points per trace, including
exact duplicates, isolated noise, and disordered sequences. The pipeline is
order-independent and handles branching/splay structures via DBSCAN clustering.

**Break-type awareness**: Each input point carries a `break_type` label
(`p` = primary, `i` = inferred, `s` = secondary, or `unknown`), sourced from the
`tier` column in the Excel file or looked up from 05a's `all_inferred_points.csv`.
Break types are preserved through every filtering stage, used to separate
parallel traces of different types within a cluster, and scored for consistency.
Each output CSV includes a `break_type` column.

### Methods

#### Stage 0 -- Diagnose before you clean

Each raw point cloud is plotted as a density-coloured scatter (kNN distance,
`YlOrRd` colourmap) overlaid on a DEM hillshade. Console output reports the
median, MAD, and 95th-percentile of k-th nearest-neighbour distances, plus the
count and locations of sequential gaps. This informs the downstream `eps` and
filtering thresholds rather than guessing.

#### Stage 1 -- Outlier rejection (density + terrain)

Two complementary filters applied sequentially:

1. **Deduplication**: exact duplicate coordinate pairs (rounded to 0.1 m) are
   removed first -- the raw files contain many thousands of identical points
   (e.g. Trace1: 49 261 -> 8 046 unique; Trace2: 16 877 -> 3 006 unique).
2. **Density filter** (Stage 1a): for each point, the distance to its k-th
   nearest neighbour (`KNN_K = 8`) is computed. Points whose kNN distance
   exceeds `median + DENSITY_MAD_FACTOR * MAD` (`factor = 5.0`) are dropped as
   isolated noise.
3. **Terrain filter** (Stage 1b): the full DEM slope grid is computed once
   (vectorised `np.gradient`), then sampled at each surviving point via
   `scipy.ndimage.map_coordinates`. Points sitting on near-flat terrain
   (slope < `TERRAIN_SLOPE_MIN_DEG = 0.5` deg) or impossibly steep terrain
   (> 70 deg) are dropped. The 0.5 deg threshold is deliberately low because the
   10 m DEM resolves scarp slopes as only ~1 deg median.

#### Stage 2 -- Segment into real sub-traces (DBSCAN)

DBSCAN spatial clustering replaces the previous sequential-gap splitting (which
assumed input order). `eps` is set to `max(median_spacing * 2.0, 5.0)` m,
`min_samples = 5`. Each cluster represents a candidate scarp segment. Clusters
below `MIN_CLUSTER_POINTS = 10` or with spatial extent < 50 m are dropped as
residual noise blobs.

#### Stage 3 -- Dense point extraction per cluster

Each cluster is a band of points around the true trace. The main pipeline
(`simplified_pipeline_for_trace`) extracts an ordered dense trace:

1. **PCA direction**: The cluster's principal axis is computed via covariance
   eigendecomposition to determine the dominant trace direction.
2. **Projection sorting**: Points are projected onto the principal axis and
   sorted by projection value, giving a spatial ordering independent of input
   sequence.
3. **Thinning**: Sorted points are thinned to near-uniform spacing
   (`TARGET_POINT_SPACING_M = 10.0` m) by accumulating distance and keeping
   points at intervals; a maximum gap check (`MAX_POINT_SPACING_M = 30.0` m)
   ensures no large jumps.
4. **Break-type consistency**: Each extracted trace is scored by the
   dominant break type and its consistency ratio. Traces are ranked by a
   composite score (length + point count + consistency) and the top
   `MAX_TRACES_PER_FILE = 5` per input file are retained.
5. **Direction splitting**: Each trace is checked for direction reversals
   and split if the heading changes significantly.

A legacy `stage3_centerline` function (rasterise -> skeletonize -> graph
longest-path) exists in the code but is not called by the main pipeline.

#### Stage 4 -- Smooth and simplify

Skeleton pixel paths are converted back to real-world coordinates, then:

1. **Arc-length resampling** to uniform spacing.
2. **Savitzky-Golay filter** (`window = 11`, `polyorder = 3`) applied along
   arc-length to remove pixel-level jaggedness while preserving the measured
   point geometry.
3. **Douglas-Peucker simplification** (`tolerance = 5.0` m) via
   `shapely.simplify` to cut redundant vertices.

#### Stage 5 -- QC pass

Each cleaned centerline is overlaid on the hillshade alongside the original
cluster points. Metrics reported:

- **close_frac**: fraction of original cluster points within
  `2 * RASTER_CELL_SIZE_M` (10 m) of the centerline. Segments with
  `close_frac < 0.6` are flagged `REVIEW`; otherwise `OK`.
- **sinuosity**: arc-length / straight-line distance.
- **length_m**: total centerline length.
- **median_dist_m**: median distance from original points to centerline.

### Output

All outputs are written to `data/HighCon/cleaned_traces/`:

| File / Directory | Content |
|------------------|---------|
| `trace_XX.csv` | One cleaned ordered polyline per file (zero-padded 2-digit ID). Columns: `x` (easting, EPSG:28355), `y` (northing, EPSG:28355), `break_type` (`p`/`i`/`s`/`unknown`). One row per ordered vertex. |
| `manifest.csv` | Per-trace metadata for 06b consumption. See column table below. |
| `qc/stage0_{TraceName}.png` | Stage 0 diagnostic figure: raw point cloud coloured by kNN density on hillshade. |
| `qc/stage5_trace_{XX}.png` | Stage 5 QC figure: cleaned centerline (red line) overlaid on original cluster points (blue scatter) on hillshade. |

**`manifest.csv` columns:**

| Column | Description |
|--------|-------------|
| `trace_id` | Integer segment ID (zero-padded in filename) |
| `source_file` | Original `Trace*.xlsx` filename |
| `n_points` | Vertices in the final smoothed centerline |
| `length_m` | Centerline length (m) |
| `break_type` | Dominant break type (`p`, `i`, `s`, or `unknown`) |
| `break_type_consistency` | Fraction of points with the dominant type |
| `score` | Composite quality score (length + point count + consistency) |
| `qc_flag` | `OK` if close_frac > 0.6, otherwise `REVIEW` |
| `csv_path` | Relative filename of the corresponding `trace_XX.csv` |

### Figures (06a-ScarpTraceCleaning.py)

| Figure | Axes | Interpretation |
|--------|------|----------------|
| `qc/stage0_{TraceName}.png` | x = Easting (m), y = Northing (m); colour = k-NN distance (m) | Use to assess scatter width (how far points sit off the linear trend), gap locations (real breaks vs. digitising dropout), and whether strands branch/overlap. Yellow/red points are isolated noise candidates. Hillshade provides terrain context. |
| `qc/stage5_trace_{XX}.png` | x = Easting (m), y = Northing (m) | Blue scatter = original cluster points after Stage 1 filtering; red line = cleaned centerline after Stage 4 smoothing. Title shows trace ID, length, sinuosity, close_frac, and QC flag. Use to verify the centerline faithfully follows the point band. REVIEW-flagged segments may need manual inspection. |

### Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `KNN_K` | 8 | k-th nearest neighbour for density computation |
| `DENSITY_MAD_FACTOR` | 5.0 | Reject points with kNN dist > median + factor * MAD |
| `TERRAIN_SLOPE_MIN_DEG` | 0.5 | Drop points on near-flat terrain (low for 10 m DEM) |
| `TERRAIN_SLOPE_MAX_DEG` | 70.0 | Drop points on impossibly steep terrain |
| `DBSCAN_MIN_SAMPLES` | 5 | Minimum points per DBSCAN cluster core |
| `MIN_CLUSTER_POINTS` | 10 | Minimum points to keep a cluster |
| `MIN_CLUSTER_EXTENT_M` | 50.0 | Minimum spatial extent to keep a cluster (m) |
| `RASTER_CELL_SIZE_M` | 5.0 | Grid cell size for rasterisation (m) |
| `SKELETON_SIGMA` | 0.8 | Gaussian pre-smoothing of density mask before skeletonizing |
| `SPUR_LENGTH_THRESHOLD` | 4 | Prune skeleton branches shorter than this (pixels) |
| `SAVGOL_WINDOW` | 11 | Savitzky-Golay smoothing window (must be odd; small for measured points) |
| `SAVGOL_POLYORDER` | 3 | Savitzky-Golay polynomial order |
| `DP_TOLERANCE_M` | 5.0 | Douglas-Peucker simplification tolerance (m) |
| `TARGET_POINT_SPACING_M` | 10.0 | Target spacing between output vertices (m) |
| `MAX_POINT_SPACING_M` | 30.0 | Maximum allowed gap before forcing a point (m) |
| `MAX_TRACES_PER_FILE` | 5 | Maximum traces retained per input file |
| `MIN_TRACE_LENGTH_M` | 100.0 | Minimum trace length to keep (m) |
| `MIN_POINTS_PER_TRACE` | 10 | Minimum points per trace |
| `MERGE_CLUSTER_GAP_M` | 200.0 | Max gap for merging adjacent DBSCAN clusters (m) |
| `MERGE_DIRECTION_TOLERANCE_DEG` | 30.0 | Direction tolerance for cluster merge (deg) |
| `BREAK_TYPE_CONSISTENCY_THRESHOLD` | 0.7 | Minimum consistency for type assignment |
| `VALID_BREAK_TYPES` | {p, i, s, unknown} | Recognised break type labels |

### Command-Line Usage

```bash
# Standard run (uses default 10 m DEM)
python 06a-ScarpTraceCleaning.py

# Specify a different DEM
python 06a-ScarpTraceCleaning.py --dem-source path/to/dem.tif

# Skip Stage 0 diagnostic plots (faster)
python 06a-ScarpTraceCleaning.py --skip-diagnose

# Skip Stage 1b terrain filter (density filter only)
python 06a-ScarpTraceCleaning.py --skip-terrain
```

---

## 06b-ScarpParameterProfiling.py

### What It Does

This script calculates quantitative scarp parameters at 10 m intervals along
cleaned scarp traces. For each station along a trace, it extracts a 1 km
perpendicular DEM transect, identifies the scarp zone via cumulative relief
analysis, estimates hanging-wall and footwall regional slopes, decomposes
scarp height/throw, estimates strike/dip of the scarp surface from adjacent
crest-base point pairs, fits the Nash (1980) linear diffusion model to
extract the diffusivity kappa (or the kappa * t product when no age constraint
is available), and estimates folded-fault dip using a V-rule geometric model
that classifies dip into three categories (<45, 45-55, >55 deg).

The script reads cleaned polylines from `data/HighCon/cleaned_traces/`
(produced by 06a) via a `manifest.csv` index. If the cleaned directory is
absent, it falls back to legacy inline dedup/split/NN-order processing of raw
`Trace*.xlsx` files. Only traces longer than 50 m are processed.

### Methods

#### 1. Trace Loading and Resampling

Cleaned trace CSVs are loaded from `data/HighCon/cleaned_traces/trace_XX.csv`
(columns: `x`, `y`). Each polyline is resampled at uniform 10 m intervals along
arc-length (`resample_trace()`), producing station coordinates, cumulative
distances, and local azimuths (deg from north, derived from segment direction).

#### 2. Perpendicular Profile Extraction

At each station, a 1 km perpendicular transect (`PROFILE_HALF_LENGTH_M = 500`
m each side) is extracted from the DEM using bilinear interpolation
(`scipy.ndimage.map_coordinates`). The transect is oriented at azimuth + 90 deg
to the local scarp strike. Sampling resolution is 10 m along the transect.

#### 3. Scarp Parameter Calculation (`calculate_reverse_fault_parameters`)

The main pipeline uses a cumulative relief analysis approach, not the
multi-scale slope-break detection that exists as legacy functions. For each
perpendicular profile:

1. The profile is smoothed with an adaptive Gaussian filter
   (`sigma = len(profile) / 50`).
2. Far-field elevations (outer 20% on each side) determine which side is the
   hanging wall (higher) vs. footwall (lower).
3. Smoothed slopes (`dz/dx`) are computed; the maximum slope location
   identifies the main scarp face.
4. A slope threshold (25% of max slope) defines the crest (where slope
   first exceeds threshold when leaving the HW) and base (where slope drops
   below threshold when entering the FW).
5. Regional slopes are estimated from far-field points on each side.
6. The scarp zone is identified as the region containing the majority of total
   relief change. Vertical separation (VS) is computed as the elevation
   difference between HW and FW regional surfaces projected to the scarp
   centre.

> **Note**: Functions `identify_slope_breaks`, `fit_regional_slopes`, and
> `calculate_scarp_throw` exist in the code but are **not called** by the main
> pipeline. They represent an earlier multi-scale derivative approach kept for
> reference and potential future use.

#### 4. Strike / Dip Estimation (`calculate_scarp_geometry`)

A 5-station sliding window (`window_radius = 5`, ~100 m at 10 m station spacing)
collects 3-D crest and base points `(x, y, z)` reconstructed from station
coordinates, profile distances, and local azimuth. A least-squares plane is
fitted to all points in the window. The plane normal is converted to geological
strike (deg from north), dip (deg from horizontal), and dip direction. Plane fit
quality is reported as RMS residual.

If a dip > 1 deg is detected, the throw calculation is re-run with the fitted
dip to update net_slip and heave.

#### 5. Diffusion Constant Fitting (`fit_diffusion_constant`)

The Nash (1980) linear diffusion model is fitted to each profile:

```
z(x) = (H/2) * erfc(x / (2 * sqrt(kappa * t)))
```

The profile is centred on the crest position, normalised, and trimmed to
+/- 200 m. `scipy.optimize.curve_fit` is used with bounds
`H in [0.01, 500]`, `kappa_t in [0.1, 1e6]`. If a scarp age (`--age-yr`) is
provided, `kappa = kappa_t / age`; otherwise only `kappa_t` is reported.
Goodness-of-fit is reported as R-squared.

#### 6. Profile Quality

A composite quality score: `valid_fraction` (proportion of non-NaN pixels in
the transect). If valid fraction < 0.5, the station is skipped entirely.

#### 7. Re-estimation with Dip

After all stations are processed, strike/dip is computed in batch. If a
station has dip > 1 deg, `calculate_reverse_fault_parameters` is re-called
with the fitted dip to update net_slip, heave, and vertical_displacement.

#### 8. Folded-Fault Dip Estimation (`estimate_folded_fault_dip`)

This function implements a "Rule of V" geometric model for estimating fault
dip from the plan-view curvature of the fault trace on a near-planar terrain.
It is called after all per-station parameters have been computed and provides
an independent dip estimate that cross-checks the plane-fit dip from
`calculate_scarp_geometry`.

**Mathematical framework.** The terrain is modelled as a gently sloping plane
`h_terrain(x, y) = slope_s * s + const`, where `s` is the along-trace
coordinate (nearly flat in the perpendicular `n` direction). The fault is
modelled as a folded surface consisting of two planar wings meeting at an
apex (the V-shape in plan view). Each wing is a plane:

```
h_fault = a_i * s + b_i * n + c_i   (i = 1, 2)
```

At the outcrop line (where the fault intersects the terrain), `h_fault =
h_terrain`, which yields the constraint:

```
a_i + b_i * (dn/ds)_i = slope_s
```

The true dip on each wing is:

```
tan(alpha) = sqrt(a_i^2 + b_i^2)
```

An independent V-rule dip is computed from the outcrop geometry:

```
sin(alpha) = H / W
```

where `H` is the vertical relief (median primary throw) and `W` is the scarp
width. This provides a geometric cross-check on the plane-fit dip.

**Algorithm.** For each sliding window (~1 km, `window_half = 500` m parameter
of `estimate_folded_fault_dip`) along the trace:

1. **PCA projection.** Station coordinates are projected onto along-trace
   (`s`) and perpendicular (`n`) axes using principal component analysis.
2. **V-shape detection.** A V-shape is identified when the perpendicular
   projection `n` has a clear interior extremum (apex) with the perpendicular
   displacement reversing direction across it. The threshold requires
   `n_range > 0.05 * s_range`.
3. **Two-wing plane fit.** If a V-shape is detected, two separate planes are
   fitted (wing 1: stations before the apex; wing 2: stations after). The
   reported dip is the mean of the two wing dips.
4. **Single-plane fit.** If no V-shape is detected, a single plane is fitted
   to the entire window.
5. **V-rule constrained dip.** The median primary throw and scarp width in the
   window give `sin(alpha) = H / W`, clipped to `(0, 1)`.
6. **Dip category classification.** The final dip category is determined by
   the mean of the plane-fit dip and the V-rule dip (when available):
   - Category 0: < 45 deg (low dip)
   - Category 1: 45-55 deg (target interval for folded faults)
   - Category 2: > 55 deg (steep)
   - Category -1: unknown (no valid estimate)

Each station retains the highest-confidence (best R-squared) estimate across
overlapping windows.

### Output

All outputs are written to `data/derivatives/scarp_parameter_analysis/`:

| File / Directory | Content |
|------------------|---------|
| `parameter_database.csv` | All parameters at 10 m intervals (one row per station). See column table below. |
| `scarp_segment_summaries.csv` | Per-trace (segment) averaged parameters. See column table below. |
| `diffusion_analysis/diffusion_fits.csv` | Diffusion fit details (trace_id, distance_along, x, y, primary_throw, kappa, kappa_t, H_fit, r_squared). |
| `cross_sections/segment_{trace_id}/panel_{NNN}.png` | Cross-section panel plots. |
| `along_scarp_profiles/segment_{trace_id}_parameters.png` | 4-subplot along-scarp parameter profiles (throw, strike/dip, regional slopes, quality). |
| `per_trace_parameter_profiles/{trace}_parameter_profiles.png` | Per-original-trace parameter profiles (northing on x-axis): throw, azimuth, dip, and folded-fault dip category. |
| `3d_visualizations/segment_{trace_id}/` | 3D terrain renderings (3 camera angles per chunk). |
| `profiles_cache/trace_{trace_id}.npz` | Cached profile arrays (distances + elevations matrix) for plot regeneration without reprocessing. |

**`parameter_database.csv` columns:**

| Column | Description |
|--------|-------------|
| `trace_id` | Segment ID from 06a manifest |
| `distance_along` | Cumulative distance along trace (m) |
| `x`, `y` | Station coordinates (EPSG:28355) |
| `azimuth` | Local scarp strike at this station (deg from north) |
| `HW_regional_slope_deg` | Hanging-wall regional slope (deg) |
| `FW_regional_slope_deg` | Footwall regional slope (deg) |
| `HW_regional_slope` | Hanging-wall regional slope (dimensionless, dz/dx) |
| `FW_regional_slope` | Footwall regional slope (dimensionless, dz/dx) |
| `local_relief` | Elevation range across the 1 km transect (VS_max - VS_min) (m) |
| `z_crest_primary` | Elevation of detected scarp crest (m) |
| `z_base_primary` | Elevation of detected scarp base (m) |
| `z_hw_projected` | HW regional surface projected to scarp centre (m) |
| `z_fw_projected` | FW regional surface projected to scarp centre (m) |
| `primary_throw` | Apparent throw: z_crest - z_base (m) |
| `true_vertical_displacement` | True VD: z_hw_projected - z_fw_projected (m) |
| `vertical_separation` | Same as true_vertical_displacement (m) |
| `VS_max` | Maximum vertical separation in profile (m) |
| `VS_min` | Minimum vertical separation in profile (m) |
| `intermediate_throw` | Throw from crest to first intermediate break (m) |
| `total_cumulative_throw` | Max of true VD and cumulative intermediate throws (m) |
| `net_slip` | Net slip = true_VD / sin(dip) (m, NaN if no dip) |
| `heave` | Heave = true_VD / tan(dip) (m, NaN if no dip) |
| `vertical_displacement` | Updated VD after dip re-estimation (m) |
| `scarp_width_m` | Horizontal distance between crest and base (m) |
| `scarp_slope_deg` | Scarp face slope: arctan(throw / width) (deg) |
| `scarp_slope` | Scarp slope (dimensionless, dz/dx) |
| `is_reverse_scarp` | Boolean: whether the scarp was classified as reverse |
| `n_intermediate_breaks` | Number of intermediate slope breaks between crest and base |
| `break_hierarchy` | Always `"none"` in current pipeline (legacy field) |
| `kappa` | Diffusivity (m^2/yr); NaN if no age provided |
| `kappa_t` | Diffusion product kappa * t (m^2) |
| `diffusion_H_fit` | Fitted scarp height H from diffusion model (m) |
| `diffusion_r_squared` | Goodness-of-fit R^2 for diffusion model |
| `profile_quality` | Composite quality score (0-1) |
| `strike` | Scarp surface strike from plane fit (deg from north) |
| `dip` | Scarp surface dip from plane fit (deg) |
| `dip_direction` | Dip direction (deg from north) |
| `plane_quality_rms` | RMS residual of plane fit (m) |

> **Note**: The folded-fault dip estimates (folded_dip_deg, v_rule_dip_deg,
> dip_category, cross_slope_b, wing_label, confidence) are computed in-memory
> by `estimate_folded_fault_dip` and passed directly to
> `plot_per_trace_parameter_profiles` for visualisation. They are not persisted
> to `parameter_database.csv` in the current implementation.

**`scarp_segment_summaries.csv` columns:**

| Column | Description |
|--------|-------------|
| `trace_id` | Segment ID |
| `n_stations` | Number of 10 m stations processed |
| `trace_length_m` | Total trace length (m) |
| `mean_primary_throw` | Mean apparent throw (m) |
| `max_primary_throw` | Maximum apparent throw (m) |
| `mean_total_cumulative_throw` | Mean cumulative throw (m) |
| `mean_HW_slope_deg` | Mean HW regional slope (deg) |
| `mean_FW_slope_deg` | Mean FW regional slope (deg) |
| `mean_strike` | Mean scarp strike (deg) |
| `mean_dip` | Mean scarp dip (deg) |
| `mean_n_breaks` | Mean number of intermediate breaks per station |
| `mean_kappa_t` | Mean diffusion product (m^2) |
| `mean_diffusion_r_squared` | Mean diffusion fit R^2 |
| `mean_profile_quality` | Mean profile quality score |
| `mean_local_relief` | Mean local relief (m) |

### Figures (06b-ScarpParameterProfiling.py)

| Figure | Axes | Interpretation |
|--------|------|----------------|
| `cross_sections/segment_{id}/panel_{NNN}.png` | x = Distance along transect (m), y = Elevation (m) | Black line = DEM elevation profile; red dashed = HW regional slope projection; blue dashed = FW regional slope projection; red triangle = detected crest; blue triangle = detected base; green square = projected HW/FW elevations at scarp centre; orange shaded band = scarp zone. Title shows distance along trace, apparent/true throw, net slip, HW/FW slopes, dip, and break count. Panels are generated every 100 m along the trace, 4 profiles per panel. |
| `along_scarp_profiles/segment_{id}_parameters.png` | x = Distance along trace (m), shared across 4 subplots | Subplot 1: primary throw (red) and total cumulative throw (blue) -- y = Throw (m). Subplot 2: strike (black, left axis) and dip (red, right axis) -- y = deg. Subplot 3: HW slope (red) and FW slope (blue) -- y = Regional slope (deg). Subplot 4: profile quality (black) -- y = Quality (0-1). |
| `per_trace_parameter_profiles/{trace}_parameter_profiles.png` | x = Northing (m), shared across 4 subplots | Subplot 1: primary throw (solid) and total cumulative throw (dashed), stepped average -- y = Throw (m). Subplot 2: azimuth (circular statistics) -- y = Azimuth (deg). Subplot 3: dip -- y = Dip (deg). Subplot 4: folded-fault dip category (V-rule) -- y = Category, fixed 4-row axis (unknown, <45, 45-55, >55). Coloured horizontal segments per station; trace IDs annotated. Segments sorted N-to-S. Reference lines show mean and max (panels 1-3 only). |
| `3d_visualizations/segment_{id}/view_{angle}_chunk{NN}.png` | x = Easting (m), y = Northing (m), z = Elevation (m x 5 vertical exaggeration) | Terrain surface coloured by `plt.cm.terrain` elevation colormap. Red spheres = scarp crest points; blue spheres = scarp base points; purple dashed lines = crest-base links. Three camera angles per chunk: plan view (elev=90, azim=-90), oblique NE (elev=35, azim=45), oblique SW (elev=35, azim=225). Chunks are ~2 km long. A north arrow is drawn for orientation. |

### Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `PROFILE_HALF_LENGTH_M` | 500.0 | Half-length of perpendicular transect (m); total = 1 km |
| `PROFILE_RESOLUTION_M` | 10.0 | Sampling interval along transect (m) |
| `STATION_SPACING_M` | 10.0 | Distance along trace between stations (m) |
| `REGIONAL_FARFIELD_DIST_M` | 200.0 | Min distance from scarp centre for regional slope fit (m) |
| `SCARP_SEARCH_HALF_WIDTH_M` | 100.0 | Search window for crest/base detection (m) |
| `BREAK_SMOOTHING_WINDOWS` | [5, 11, 21] | Multi-scale Gaussian derivative windows (pixels) |
| `MIN_BREAK_SEPARATION_M` | 20.0 | Minimum distance between detected breaks (m) |
| `BREAK_PRIMARY_THRESHOLD` | 0.60 | Relief fraction for primary break classification |
| `BREAK_SECONDARY_THRESHOLD` | 0.80 | Relief fraction for secondary break classification |
| `VERTICAL_EXAGGERATION` | 5.0 | Vertical exaggeration for 3D plots |
| `CROSS_SECTION_PANEL_SIZE` | 4 | Number of cross-section profiles per figure |
| `CROSS_SECTION_INTERVAL_M` | 100.0 | Plot a cross-section every N m along trace |
| `CHUNK_3D_LENGTH_M` | 2000.0 | 3D visualisation chunk size (m) |
| `PLOT_DPI` | 200 | Output figure resolution (dpi) |

### Command-Line Usage

```bash
# Standard run (processes all traces from cleaned_traces/)
python 06b-ScarpParameterProfiling.py

# Specify a different DEM
python 06b-ScarpParameterProfiling.py --dem-source path/to/dem.tif

# Limit traces for testing
python 06b-ScarpParameterProfiling.py --max-traces 5

# Process a specific range of traces
python 06b-ScarpParameterProfiling.py --start-trace 10 --end-trace 20

# Provide a scarp age to compute kappa (instead of kappa_t only)
python 06b-ScarpParameterProfiling.py --age-yr 60000

# CSV output only, skip all plots
python 06b-ScarpParameterProfiling.py --no-plots

# Skip 3D visualisations (keep cross-sections and along-scarp profiles)
python 06b-ScarpParameterProfiling.py --skip-3d

# Resume: skip traces already in parameter_database.csv
python 06b-ScarpParameterProfiling.py --resume

# Regenerate all plots from saved CSV + profile cache (no reprocessing)
python 06b-ScarpParameterProfiling.py --plots-only
```

Notes:
- Run `06a-ScarpTraceCleaning.py` first to produce `data/HighCon/cleaned_traces/`.
  If the cleaned directory is absent, 06b falls back to legacy inline processing
  of raw `Trace*.xlsx` files (with a warning).
- The `--resume` flag appends to the existing `parameter_database.csv` without
  reprocessing completed traces. Use `--start-trace 0` without `--resume` to
  start fresh (clears the old CSV).
- The `--plots-only` mode requires a prior full run (needs both
  `parameter_database.csv` and `profiles_cache/*.npz`).

---





## How to Run

```bash
# All scripts should be run with the project venv:
#   .venv/Scripts/python.exe <script>.py

# Pre-pipeline: Exploratory data analysis (no output files)
.venv/Scripts/python.exe 00-DataEDA.py

# Phase 1a: DEM reprojection, alignment, and derivative generation
.venv/Scripts/python.exe 01a-DerivativeGen.py

# Phase 1b: Tile sampling and Phase 1 surface expression audit
.venv/Scripts/python.exe 01b-TileSampling.py

# Phase 2: Visual audit and uncertainty assessment
.venv/Scripts/python.exe 02-VisualAudit.py

# Phase 3: Resolution sensitivity analysis (requires 01a output)
.venv/Scripts/python.exe 03-ResolutionSensitivity.py

# Phase 5a: Known-trace scarp validation (requires 01a DEM + structural points)
.venv/Scripts/python.exe 05a-KnownTraceScarpValidation_dem10m.py --dem-source dem10m
.venv/Scripts/python.exe 05a-KnownTraceScarpValidation_hs.py --dem-source lidar_0_5m

# Phase 5b: ML-based wide-area scarp inference (requires 05a + HighCon + filtered TXT)
.venv/Scripts/python.exe 05b-DEMWideScarpInference.py            # default dem10m
.venv/Scripts/python.exe 05b-DEMWideScarpInference.py --dem-source lidar_0_5m
.venv/Scripts/python.exe 05b-DEMWideScarpInference.py --max-unlabeled 50000  # quick test

# Phase 6a: Scarp trace cleaning (requires 01a DEM output)
.venv/Scripts/python.exe 06a-ScarpTraceCleaning.py               # default 10 m DEM
.venv/Scripts/python.exe 06a-ScarpTraceCleaning.py --skip-diagnose  # skip Stage 0 plots

# Phase 6b: Scarp parameter profiling (requires 06a output + 01a DEM)
.venv/Scripts/python.exe 06b-ScarpParameterProfiling.py          # full run
.venv/Scripts/python.exe 06b-ScarpParameterProfiling.py --max-traces 5  # quick test
.venv/Scripts/python.exe 06b-ScarpParameterProfiling.py --no-plots      # CSV only
.venv/Scripts/python.exe 06b-ScarpParameterProfiling.py --plots-only    # regenerate plots
```

All scripts are idempotent -- they skip outputs that already exist.
To regenerate, delete the relevant output files or subdirectories.

**Important**: Scripts must be run with the project venv (`.venv/Scripts/python.exe`),
not the system `python`, because pandas and other dependencies are installed only
in the venv.

**Important**: If upgrading from a previous version that produced .tif derivatives,
the new scripts will automatically delete old .tif derivative files and regenerate
as .npy. The reprojected DEM .tif files are always kept.

**LiDAR processing** is very slow due to the 42.8 GB file size. Hillshade generation
takes several hours. SVF/openness can take >24 hours. Plan accordingly.

---

## Expected Results

### Expression Quality Distribution

Based on the flat-lying alluvial terrain of the Axedale region, we expect:
- Most fault traces will be **Class 3 / Class C** (inferred) at 30 m resolution
  because SRTM cannot resolve scarps < 3 m
- 10 m DEM will upgrade many traces to **Class 2 / Class B** where scarps are 1-3 m
- LiDAR will provide the best classification, potentially detecting scarps as
  small as 0.2 m
- Some traces may be confirmed as **Class D** (no surface expression) even at
  LiDAR resolution, indicating they are either very old, deeply weathered, or
  miscoded in the source database

### Key Outputs for Decision-Making

1. **Expression quality map** -- Identifies which faults have geomorphic expression
   vs. which are database-only. Prioritises targets for field verification.

2. **Confidence scores** -- Quantifies reliability of each fault classification.
   Low-confidence segments need higher-resolution data or field checks.

3. **Multi-resolution comparison** -- Demonstrates the value of higher-resolution
   DEMs for scarp detection. Essential for justifying LiDAR acquisition costs.

4. **Cross-validation** -- Places local results in the context of published analogue
   studies. If Axedale scarps fall within the Yallourn (1-2 m) range, they are
   consistent with known slow-deformation fault scarps in similar geology.

5. **ML lineament discovery** (Phase 4, archived) -- Identifies candidate tiles in scarp
   clusters that are not near any known fault. These represent unmapped lineaments
   worth field-checking. The correlation between embedding distance and real scarp
   distance quantifies how well the autoencoder has learned fault morphology.

6. **Cosine-similarity retrieval** (Phase 4, archived) -- For each source and quality class,
   retrieves tiles that are latent-space-similar to known-scarp reference tiles. The
   similarity-score violin plots quantify retrieval quality; the UMAP overlays show
   spatial distribution of retrieved tiles; and the example grids provide visual
   confirmation that similar tiles share scarp morphology.

7. **ML-inferred fault traces** (Phase 5, archived) -- Reconstructs fault traces from
   cross-section picks through candidate corridors and scores them on a composite
   0-10 confidence scale. Traces far from any cataloged structural point are flagged
   as novel discoveries. The fence diagrams and pick-track plots allow visual QC
   of lateral continuity the way a geologist would read a seismic fence diagram ---
   directly addressing Limitation #6 (the autoencoder cannot distinguish fault
   scarps from artificial cuts, but continuity analysis can).

8. **Channel importance** (Phase 6, archived) -- Quantifies which terrain-derivative channels
   the autoencoder relies on for fault-scarp retrieval. The ablation-vs-saliency
   scatter plots and radar charts let a researcher diagnose whether the model is
   learning genuine geomorphic structure or relying on illumination artifacts. The
   interpretation flags provide automated warnings for illumination dominance and
   cross-source divergence.

9. **Multi-resolution summary** (Phase 7, archived) -- Overlays ML-inferred traces from all
   three DEM resolutions on a single map. Traces detected at all resolutions are
   the most robust; traces unique to LiDAR represent subtle features that justify
   high-resolution data acquisition. The confidence-coloured version highlights
   which traces are most reliable for database inclusion.

10. **Known-trace scarp validation** (Phase 5a, active) -- Validates that known fault
    traces have pixel-scale geomorphic evidence of a scarp in the DEM, producing
    segment-level confidence scores (0-10) and evidence deltas. The `all_inferred_points.csv`
    output seeds the 05b ML inference pipeline.

11. **ML-based wide-area scarp inference** (Phase 5b, active) -- Trains XGBoost on
    HighCon traces (positive) vs confirmed-removed negatives, scores the unlabeled
    candidate pool, and applies a shape-based I/L vs U/O filter. The iterative
    active-learning loop expands labels across rounds, and spatial plots show
    score distributions and shape filtering results.

12. **Scarp trace cleaning** (Phase 6a, active) -- Converts raw point clouds into
    cleaned, ordered polylines with break-type preservation. The `manifest.csv`
    provides per-trace QC metadata (length, sinuosity, close_frac, break_type
    consistency) for 06b consumption.

13. **Scarp parameter profiling** (Phase 6b, active) -- Produces a parameter database
    at 10 m station intervals along cleaned traces, including throw decomposition,
    strike/dip estimation, and diffusion constant fitting. Cross-section panels,
    along-scarp parameter profiles, and 3D visualisations support geological
    interpretation.

### Resolution Sensitivity (Phase 3)

Based on the scarp heights expected in the Axedale region (0.5-4 m) and the
quantitative metrics computed by 03-ResolutionSensitivity.py:

- **0.5 m (LiDAR)** -- All scarps >= 0.2 m detectable. Kappa > 0.8 for all feature
  types. SNR >> 2 for scarps >= 1 m. This is the gold standard.

- **2 m** -- Scarps >= 1 m remain detectable (SNR > 2, kappa > 0.6). Sub-metre
  scarps and subtle drainage deflections become marginal. LFI increases (more
  lineament fragmentation).

- **5 m** -- Only scarps >= 2 m are reliably detected. Monoclines and broad folds
  may still be visible. Drainage anomalies largely lost. SLPR drops significantly.

- **10 m** -- Scarps >= 3 m detectable. Most 1-2 m scarps from the Yallourn analogue
  range are not detected. Kappa drops below 0.6 for scarp and drainage features.

- **30 m (SRTM)** -- Only the largest scarps (>= 5 m) are detectable. Most fault
  traces in the region fall below the MDD. The detection matrix should show red
  (not detected) for most feature types.

The **detection matrix heatmap** provides a single-page summary of these thresholds.
The **recommended algorithm** (typically mean aggregation for discrete scarps) is
output in `phase2_sensitivity/uncertainty/recommended_algorithm.txt`.

### ML Lineament Discovery (Phase 4) -- archived

> The scripts referenced in this section (04-LineamentML.py, 05-MLTraceReconstruction.py,
> 06-ChannelImportance.py, 07/07a/08) are archived in `code_old/`. The active workflow
> uses 05a/05b/06a/06b instead. The following expectations apply only if the archived
> scripts are run.

Based on the unsupervised pipeline in 04-LineamentML.py:

- **Autoencoder** should converge within 30-50 epochs (MSE loss plateau). The
  reconstruction quality is less important than the structure of the latent space ---
  even imperfect reconstruction can yield informative embeddings.

- **UMAP embedding** of known-scarp tiles should form a recognisable cluster or dense
  region. If scarp tiles are scattered uniformly, the autoencoder has not learned
  fault-specific morphology (this may indicate the tile resolution or channel
  selection needs adjustment).

- **HDBSCAN** is expected to find 5-20 clusters at the default parameters
  (min_cluster_size=50). The scarp cluster may be a small, tight group or a
  sub-cluster within a larger morphological class. A high noise fraction (>80%)
  is expected since most tiles lack fault features.

- **KMeans** provides a complementary partition of the same latent space into 10
  clusters. Unlike HDBSCAN, every tile is assigned to a cluster. Comparing KMeans
  and HDBSCAN labels identifies tiles that are consistently grouped vs. those at
  the boundary of cluster definitions.

- **Cosine-similarity retrieval** extends the clustering approach by directly
  measuring latent-space similarity to known-scarp reference tiles. For each of
  the 9 source x quality groups, the retrieval should show:
  - Class 1 (well-constrained) traces retrieving tight, high-scoring neighbours
    (violin peak near cosine sim = 0.8--1.0), with similar tiles clustering tightly
    around reference tiles in UMAP space.
  - Class 3 (inferred) traces retrieving diffuse, lower-scoring neighbours
    (violin spread, lower median), with similar tiles scattered more broadly.
  - GeoStru501 (377 traces) should produce the most robust retrieval due to sample
    size; FSD1 (5 traces) will have limited reference coverage.

- **Embedding-vs-real-distance correlation**: A Pearson r > 0.3 is a positive sign
  that the embedding captures real spatial structure. Stronger correlations (r > 0.5)
  suggest that candidate tiles near the scarp cluster in embedding space are
  genuinely worth field-checking. If r < 0.1, the autoencoder may be capturing
  elevation or vegetation patterns instead of fault morphology.

- **k-NN purity**: For known-scarp tiles, purity > 0.3 (i.e. at least 6 of 20
  nearest latent-space neighbours are also scarp tiles) indicates meaningful
  clustering. Random expectation with ~3 % scarp tiles would be purity ~0.03.

- **Saliency tiles** should show visible terrain features (slope breaks, linear
  valleys, SVF anomalies) in candidate tiles. If tiles appear featureless or
  dominated by artefacts, the channel selection or normalisation may need adjustment.

- **Similar tile examples** should show reference tiles with clear scarp morphology
  (visible slope break, structural point overlay) and retrieved similar tiles that
  share the same morphology. If similar tiles show no geomorphic features, the
  cosine similarity is capturing non-scarp structure (e.g. vegetation or elevation).

### Channel Importance Analysis (Phase 6) -- archived

Based on the explainability pipeline in 06-ChannelImportance.py (archived):

- **Ablation and saliency agreement** is the key signal. When both methods rank the
  same channel as important, the result is trustworthy. Disagreement (Spearman
  |rho| < 0.3) means one method may be misleading --- ablation is coarse but
  task-grounded; saliency is fine-grained but can be unreliable in saturated
  activation regions.

- **Expected channel ranking**: Slope and SVF are expected to be the most important
  channels, as they capture scarp gradient and sky-view obstruction --- the two
  strongest morphological indicators of fault scarps. Curvature channels (plan,
  profile) should be moderately important for detecting sharp crest/valley
  transitions. Hillshade channels may be important but could indicate illumination-
  artifact dependence if they dominate over SVF + curvature.

- **Illumination-artifact flag**: If any single hillshade channel's importance
  exceeds 2x the mean of (curvature + SVF) importance, the autoencoder may be
  relying on illumination direction rather than intrinsic terrain morphology. This
  would suggest the model generalises poorly to DEMs with different illumination
  or to real-world field conditions.

- **Cross-source divergence**: If the Kruskal-Wallis test flags a channel as
  divergent across sources (p < 0.05), the channel's importance depends on which
  fault database is used. This is expected for sources with very different trace
  densities (GeoStru501 has 377 traces vs FSD1's 5) but may also indicate that
  different sources encode different fault morphologies.

- **Radar chart overlap**: If all three sources produce similar polygon shapes on
  the radar chart, channel importance is robust across databases. Divergent shapes
  suggest source-specific bias in the reference tile pool.

### Multi-Resolution Summary (Phase 7) -- archived

Based on the summary overlay in 07-MultiResolutionSummary.py (archived):

- **Traces detected at all three resolutions** are the most robust --- they
  represent large, unambiguous scarps visible even at SRTM 30 m. These should
  align with the highest-confidence traces from Script 05.

- **Traces unique to LiDAR (0.5 m)** represent small or subtle scarps invisible
  at coarser resolutions. These are the traces that justify LiDAR acquisition
  and are the most likely candidates for novel fault discoveries.

- **Traces unique to 10 m DEM** but absent from LiDAR may indicate features that
  are resolvable at 10 m but lost in LiDAR noise, or artifacts of the tiling/
  autoencoder pipeline at different resolutions.

- **Confidence-coloured overlay**: If high-confidence traces (green) cluster in
  areas where all three sources agree, the pipeline is internally consistent.
  Low-confidence traces (red) appearing only in one source warrant field
  verification before inclusion in a fault database.

### Multi-Resolution Reframe and Neotectonic Validation (Phase 7a) -- archived

Based on the synthesis in 07a-MultiResolutionReframe.py (archived):

- **Expression class distribution** should show that all fault DB sources
  (FSD1, CWvic1, GeoStru2501, GeoStru501) have a mix of A/B/C expression
  classes. If the distribution is roughly similar across sources, it confirms
  that the ML pipeline measures topographic expression, not fault type ---
  supporting the reframe from "fault confidence" to "lineament expression
  confidence."

- **Resolution recovery table** should show decreasing recovery rate at
  coarser resolutions, with LiDAR 0.5 m as the reference. Typical expectations:
  dem10m should recover >70% of LiDAR Class-1 traces (within 100 m), while
  SRTM 30 m may recover only 30-50%. The class-1 retention rate will be lower
  than the raw recovery rate because coarser DEMs downgrade some Class-1
  traces to Class-2 or Class-3.

- **ROC/PR curves of ML confidence vs. FSD1 neotectonic label**: AUC is
  expected to be near 0.5 (no predictive power) because the autoencoder
  operates on topographic morphology and has no access to fault databases.
  An AUC significantly above 0.7 would suggest the model has inadvertently
  learned fault-type-specific signals, which would contradict the reframe
  narrative and warrant investigation.

- **Basement fault candidates** should include traces from CWvic1,
  GeoStru2501, and GeoStru501 with strong topographic expression (Class A
  or B) but no spatial proximity to FSD1 neotectonic points. These
  represent potentially unmapped basement faults that are topographically
  expressed but not classified as neotectonic. Traces in areas with
  GeoStru501 source dominance >75% should be treated with caution as
  potential database artifacts.

### Geological Context and Composite Confidence (Phase 8) -- archived

Based on the integration in 08-GeologicalContext.py (archived):

- **Geological plausibility distribution**: Most ML-inferred traces in the
  Axedale region are expected to fall into `structurally_controlled` or
  `no_lithology_contrast` categories. The flat-lying alluvial terrain means
  many traces cross the same Quaternary unit on both sides, producing
  `no_lithology_contrast`. Traces that juxtapose different bedrock units
  (e.g. Ordovician vs. Silurian) are the strongest candidates for genuine
  faults.

- **Composite confidence tiers**: The weighting scheme (geological 0.35,
  scarp morphology 0.20, ML confidence 0.20, geomorphic expression 0.15,
  channel illumination 0.10) means geological plausibility dominates the
  score. High-confidence traces (>= 0.65) should be rare and represent
  traces with strong agreement across ML, geomorphic, scarp, and geological
  evidence. Moderate-confidence traces (>= 0.40) are the main working set
  for field verification. Low-confidence traces (< 0.40) lack corroborating
  evidence and should be deprioritised.

- **Targeted LiDAR profiles** (if LiDAR available): The targeted detection
  rate should be >= 0.5 for traces that are real scarps, with mean heights
  of 0.5-3 m. Traces with `targeted_dominant_type == 'discrete'` are more
  likely to be fault scarps than monoclines, which can represent folds or
  compaction features.

- **Composite confidence map**: High-confidence traces (green) should
  cluster in areas with known structural complexity and lithology contrast.
  Low-confidence traces (red) appearing in areas of uniform Quaternary
  cover may represent drainage features or artifacts rather than faults.

---

## Dependencies

- Python 3.11+
- numpy, pandas, matplotlib, scipy, scikit-image, scikit-learn
- rasterio (1.4+), pyproj
- psutil, tqdm
- **01a-DerivativeGen.py / 01b-TileSampling.py / 02-VisualAudit.py / 03-ResolutionSensitivity.py:**
  - `derivative_common.py` (shared module; imported by 01a directly, by 03 via `01-DerivativeGen.py`)
  - cupy (optional, GPU-accelerated array operations in derivative_common)
- **05a-KnownTraceScarpValidation additionally requires:**
  - scikit-image (morphology), networkx (graph operations, legacy centerline)
  - scipy (cKDTree, gaussian_filter)
- **05b-DEMWideScarpInference.py additionally requires:**
  - xgboost (gradient-boosted-tree classifier)
  - shap (optional, for SHAP summary plot; gracefully skipped if not installed)
  - rasterio (DEM and HAND raster reads)
  - scipy (cKDTree for local linearity and shape clustering, map_coordinates for raster sampling)
  - scikit-learn (LogisticRegression baseline, StandardScaler, metrics)
- **06a-ScarpTraceCleaning.py additionally requires:**
  - scikit-learn (DBSCAN clustering)
  - scikit-image (skeletonize, legacy centerline function)
  - scipy (cKDTree, gaussian_filter)
- **06b-ScarpParameterProfiling.py additionally requires:**
  - scipy (gaussian_filter1d, curve_fit, optimize)
  - matplotlib 3D toolkit (mpl_toolkits.mplot3d)
- **Archived scripts (code_old/) dependencies:**
  - 04-LineamentML.py: torch, hdbscan, glass-box-umap, shapely, geopandas
  - 05-MLTraceReconstruction.py: torch, rasterio, scikit-learn
  - 06-ChannelImportance.py: torch, scipy (Kruskal-Wallis, Spearman)
  - 07/07a/08: matplotlib, pandas, numpy, scipy, rasterio, pyproj

## Known Limitations

1. The structural point inventory has no fault names or IDs -- traces are
   reconstructed purely from spatial clustering. Some traces may be
   incorrectly split or merged.

2. Geomorphic indices (Smf, Vf, HI, AF) are approximated because true
   watershed delineation is not performed. HI uses a 2 km x 2 km window
   instead of catchment boundaries.

3. LiDAR SVF computation is capped at 200 cells radius (100 m instead of
   the desired 1000 m) for performance reasons.

4. Drainage anomaly detection relies on slope and curvature from the DEM
   rather than a real drainage network. It may miss anomalies in very flat
   terrain where slope is below the 1 deg threshold.

5. **04-LineamentML.py** (archived) uses a block-wise derivative computation approach
   that computes SVF per 4000-px block. The overlap (64 px) may be
   insufficient for SVF at the block edges --- edge tiles in the stack may
   have slightly inconsistent SVF values. This is mitigated by the stride
   being typically much smaller than the block size.

6. The autoencoder in 04-LineamentML.py (archived) learns shape-based features
   (slope, curvature, SVF) rather than absolute elevation. This means it
   cannot distinguish between a fault scarp and a road cut or artificial
   embankment with similar morphology. Field verification of candidate
   tiles is essential.

7. Glass Box UMAP is stochastic despite fixed random_state. Embedding
   stability should be checked by running twice and comparing results. The
   contribution scores are only meaningful when features are on the same
   scale (hence StandardScaler preprocessing).

8. HDBSCAN clustering is sensitive to min_cluster_size and min_samples.
   The defaults (50 and 10) may over-segment or under-segment depending
   on the tile count and resolution. Adjust and re-run if the scarp
   cluster is too large (includes non-scarp morphology) or too small
   (excludes real scarp tiles).

9. KMeans with a fixed n_clusters=10 may not match the true number of
   morphological classes in the data. Unlike HDBSCAN, it cannot identify
   noise/outlier tiles and forces every tile into a cluster. The choice
   of 10 clusters is a compromise between granularity and interpretability.

10. Cosine-similarity retrieval depends on the quality and spatial coverage
    of reference tiles (those with `is_near_scarp=True`). Sources with few
    traces (e.g. FSD1 with only 5 traces in the Phase 1 audit) will have
    limited reference coverage, producing fewer and potentially noisier
    retrieval results. The retrieval also assumes that latent-space similarity
    corresponds to geomorphic similarity --- this may not hold for tiles with
    similar morphology but different origins (e.g. road cuts vs. fault scarps).

11. The structural point overlay in similar-tile example panels uses a
    +/-200 m search radius followed by +/-64 m tile-extent filtering.
    For tiles at the edge of the LiDAR survey, structural points may be
    sparse or absent, giving a misleading zero-count even if the tile
    contains a fault scarp that was not included in the source databases.

12. **05-MLTraceReconstruction.py** (archived) uses fixed north-south cross-sections as
    the DEM analogue of seismic inlines. For corridors whose dominant
    lineament trend is within 30 deg of east-west, these sections may run
    near-parallel to the feature and miss or distort it. Such corridors are
    flagged as "poor geometry" in the output, but strike-adaptive rotation
    is not yet implemented --- the flag serves as a coverage-gap indicator.

13. The candidate pool is seeded exclusively from Class 1 reference tiles
    (well-constrained scarps). GeoStru501 (377 traces) dominates the
    reference pool by default, so the resulting candidate map is implicitly
    weighted toward whatever morphology GeoStru501's Class 1 traces represent.
    **Mitigation**: Each Class 1 reference tile is now cross-referenced against
    Script 02's Smf and Script 08's geological plausibility, and river-like or
    geologically unsupported tiles are down-weighted (0.0--0.5 multiplier on
    `ml_score`). Users should still check `source_dominance_summary.csv` before
    interpreting corridor results, as corridors found may reflect one source's
    bias rather than the full morphological spectrum.

14. The 2D KDE for corridor detection uses a 2000 m bandwidth (matching
    Script 02's lineament KDE) and a p75 density threshold. This identifies
    broad corridors but may merge spatially adjacent but geomorphically
    distinct features into a single corridor. The p75 threshold is a starting
    point --- lower values produce more/smaller corridors, higher values
    produce fewer/larger ones.

15. Trace reconstruction chains picks via nearest-neighbor logic inherited
    from Phase 1. Short runs (< 5 consecutive coherent picks) are filtered
    out, but longer runs may still include segments where the pick jumped
    between two parallel features. The throw-consistency check (vergence
    flips) catches the most egregious miscorrelations but is not a substitute
    for manual QC of the fence diagrams.

16. **06-ChannelImportance.py** (archived) uses two explainability methods (ablation and
    gradient x input saliency) that can disagree. Ablation is coarse ---
    replacing a channel with its global mean is a blunt instrument that may
    mask nonlinear interactions. Saliency can be misleading in saturated
    activation regions (e.g. ReLU dead neurons). The script treats agreement
    as the trustworthy signal, but disagreement does not mean a channel is
    unimportant --- only that the evidence is conflicting.

17. **06-ChannelImportance.py** (archived) ablation uses a `mean_replace` strategy
    (replacing the channel with its global mean). Alternative ablation
    strategies (zero-replace, noise-replace, permutation) could produce
    different importance rankings. The choice of mean-replace preserves the
    channel's overall distribution but removes its spatial structure, which
    is the most relevant aspect for terrain morphology.

18. **07-MultiResolutionSummary.py** (archived) assumes that Script 05 has been run
    separately for each DEM source (`--dem-source dem10m`, `--dem-source
    lidar_0_5m`, `--dem-source srtm1s`). If trace coordinate files are
    missing for one or more sources, the summary will be incomplete. The
    script reports which sources were found but does not flag the absence
    of a source as an error.

19. **07-MultiResolutionSummary.py** (archived) uses a fixed subsample of 5000 known
     structural points for the background overlay. In areas with dense fault
     networks, this subsample may under-represent the true fault density,
     giving a misleading impression of how many known faults are in the area.

20. **07a-MultiResolutionReframe.py** (archived) uses FSD1 as the sole neotectonic
     reference standard. FSD1 contains only ~5 traces in the Axedale study
     area, so the neotectonic validation has very limited statistical power.
     The ROC/PR curves are informative about ML-neotectonic independence but
     should not be over-interpreted as a definitive test of fault-type
     discrimination. The 500 m spatial join radius is a compromise between
     GPS uncertainty and false-positive rate; larger radii inflate recall
     but reduce precision.

21. **07a-MultiResolutionReframe.py** (archived) spatially matches ML-inferred traces
     to expression-quality (EQ) trace centroids with a 2 km radius. Because
     ML traces are model discoveries at candidate-tile centroids and EQ
     traces are known fault traces, they occupy systematically different
     spatial locations and typically few matches occur. This is expected and
     informative (ML extends beyond the known database), but means the
     ROC/PR analysis has a small matched sample size.

22. **07a-MultiResolutionReframe.py** (archived) basement fault candidates rely on
     GeoStru501 (377 traces) as a major non-neotectonic source. GeoStru501
     is a regional-scale geological map with its own biases and
     generalisations; some "basement faults" identified here may reflect
     GeoStru501 map artefacts rather than genuine structures. The
     `source_dominance_summary.csv` flag (>75% GeoStru501) is a warning but
     not a definitive filter.

23. **08-GeologicalContext.py** (archived) geological sampling uses a 250 m grid
     (GeoUnit(Point).txt) and a 100 m corridor half-width. Traces that pass
     through areas with sparse geological mapping (e.g. outside the survey
     extent) will have `insufficient_evidence` classification regardless of
     actual lithology. The lithology contrast flag is binary and does not
     capture subtle gradational contacts or fault zones wider than 250 m.

24. **08-GeologicalContext.py** (archived) targeted scarp profiles use the LiDAR 0.5 m
     DEM. For traces outside the LiDAR survey extent, targeted profiles
     cannot be extracted and the scarp morphology component falls back to
     Script 05's Phase 5 metrics (if available) or defaults to 0. The
     `--no-lidar` flag forces this fallback for all traces.

25. **08-GeologicalContext.py** (archived) composite scoring weights are heuristic and
     reflect the project's prioritisation of independent geological evidence
     over DEM-derived metrics. Alternative weightings (e.g. higher ML
     confidence weight) would produce different tier assignments. The
     thresholds (High >= 0.65, Moderate >= 0.40) are calibrated for the
     Axedale region and may need adjustment for areas with different
     geological complexity or data quality.

26. **08-GeologicalContext.py** (archived) geological plausibility classifier is
     rule-based and cannot detect faults that occur within a single mapped
     unit (e.g. intra-formational faults, or faults with throw less than
     the geological map resolution). Such faults will be classified as
     `no_lithology_contrast` even if they are real and active.

27. **05a-KnownTraceScarpValidation** thresholds are variant-specific: the `_hs`
     variant uses lower `MIN_SCARP_HEIGHT_M` (0.2 m vs 1.0 m) and wider
     hysteresis sigma range to suit high-resolution LiDAR data. Results from
     the two variants are not directly comparable without normalising for DEM
     resolution.

28. **05b-DEMWideScarpInference.py** shape filter uses `STRAIGHTNESS_FLOOR = 0.4`
     and `CONNECTIVITY_RADIUS_M = 20.0` as empirical defaults. These should be
     tuned against actual Trace1-4 data: a 90 deg L-shape with short arms can
     score lower than expected. The `prob_xgb_raw` column is retained for
     auditing points killed by the shape veto.

29. **06a-ScarpTraceCleaning.py** legacy `stage3_centerline` function
     (rasterise -> skeletonize -> graph longest-path) is defined but not called
     by the main `simplified_pipeline_for_trace`. The active pipeline uses
     PCA-based dense point extraction instead, which is faster but may produce
     slightly different centerlines than the skeletonization approach for
     complex branching clusters.

30. **06b-ScarpParameterProfiling.py** functions `identify_slope_breaks`,
     `fit_regional_slopes`, and `calculate_scarp_throw` are defined but not
     called by the main pipeline. The active pipeline uses
     `calculate_reverse_fault_parameters`, which identifies the scarp zone via
     cumulative relief analysis rather than multi-scale derivative break
     detection. The `break_hierarchy` column is always `"none"` in current
     output.
