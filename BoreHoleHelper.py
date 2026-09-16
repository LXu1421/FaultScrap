"""
BoreHoleHelper.py

Process borehole lithology records containing the keywords "fault", "shear",
and "breccia".  For each matching record:

1. Read the collar coordinates (GDA94 MGA) from the header file and transform
   them to the DEM coordinate system (EPSG:3111 - Vicgrid94).
2. Use the downhole survey (inclination / azimuth) together with the collar
   inclination / azimuth to compute the horizontal offset of the observation
   point from the collar via the average-angle trajectory method.
3. Sample the DEM (Dem10m_axe.tif) at the collar location to obtain the ground
   surface elevation.
4. Observation elevation = DEM surface elevation - borehole depth.
5. Keep only observations that fall within the DEM extent.
6. Write easting, northing, elevation (plus metadata) to a CSV.

Author: DuMate
"""

import csv
import math
import os

import numpy as np
import rasterio
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "BoreHole")
HEADER_FILE = os.path.join(DATA_DIR, "boreholeheader.txt")
LITHOLOGY_FILE = os.path.join(DATA_DIR, "bhlithology.txt")
DOWNHOLE_FILE = os.path.join(DATA_DIR, "bhdownhole.txt")
DEM_FILE = os.path.join(SCRIPT_DIR, "data", "Dem10m_axe.tif")
OUTPUT_CSV = os.path.join(DATA_DIR, "fault_shear_breccia_observations.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KEYWORDS = ["fault", "shear", "breccia"]

# DEM CRS is EPSG:3111 (Vicgrid94, GDA94 Lambert Conformal Conic).
DEM_CRS = "EPSG:3111"

# Borehole coordinates are GDA94 MGA (EPSG:28354 for zone 54, EPSG:28355 for zone 55).
TRANSFORMERS = {
    54: Transformer.from_crs("EPSG:28354", DEM_CRS, always_xy=True),
    55: Transformer.from_crs("EPSG:28355", DEM_CRS, always_xy=True),
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_header():
    """Load borehole header data indexed by siteId.

    Returns a dict: siteId -> {
        easting_vic, northing_vic,        # collar coords in DEM CRS
        measured_depth,                   # total measured depth of hole
        collar_inclination, collar_azimuth,  # from header
        elevation_gl,                     # ground-level elevation if recorded
    }
    """
    headers = {}
    with open(HEADER_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            site_id = row["siteId"]
            try:
                zone = int(row["calcZone"])
                easting = float(row["calcEasting"])
                northing = float(row["calcNorthing"])
            except (ValueError, KeyError):
                continue
            if zone not in TRANSFORMERS:
                continue

            x, y = TRANSFORMERS[zone].transform(easting, northing)

            def _float(val):
                v = (val or "").strip()
                return float(v) if v else None

            headers[site_id] = {
                "easting_vic": x,
                "northing_vic": y,
                "measured_depth": _float(row.get("measuredDepth")),
                "collar_inclination": _float(row.get("collarInclination")),
                "collar_azimuth": _float(row.get("collarAzimuth")),
                "elevation_gl": _float(row.get("elevation_gl")),
            }
    return headers


def load_downhole():
    """Load downhole survey data indexed by siteId.

    Returns a dict: siteId -> list of {depth, inclination, azimuth} sorted by depth.
    """
    downhole = {}
    with open(DOWNHOLE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            site_id = row["siteId"]
            try:
                depth = float(row["depth"])
            except ValueError:
                continue

            def _float(val):
                v = (val or "").strip()
                return float(v) if v else None

            downhole.setdefault(site_id, []).append(
                {
                    "depth": depth,
                    "inclination": _float(row.get("inclination")),
                    "azimuth": _float(row.get("azimuth")),
                }
            )

    for site_id in downhole:
        downhole[site_id].sort(key=lambda r: r["depth"])

    return downhole


def load_lithology_keywords():
    """Load lithology records whose description contains fault/shear/breccia.

    Returns a list of dicts: {siteId, from_depth, to_depth, description, material}.
    """
    records = []
    with open(LITHOLOGY_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            desc = (row.get("lithoDescription") or "").lower()
            if any(kw in desc for kw in KEYWORDS):
                try:
                    from_depth = float(row["fromDepth"])
                except ValueError:
                    continue
                try:
                    to_depth = float(row["toDepth"]) if row.get("toDepth", "").strip() else from_depth
                except ValueError:
                    to_depth = from_depth

                records.append(
                    {
                        "siteId": row["siteId"],
                        "from_depth": from_depth,
                        "to_depth": to_depth,
                        "description": row.get("lithoDescription", ""),
                        "material": row.get("lithoMaterial", ""),
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Trajectory calculation
# ---------------------------------------------------------------------------
def calculate_horizontal_offset(survey_points, target_depth):
    """Compute (east_offset, north_offset) from collar to *target_depth*.

    Uses the average-angle (tangential) method:

    For each segment between consecutive survey stations i and i+1:
        avg_inc = (inc_i + inc_{i+1}) / 2
        avg_az  = (az_i  + az_{i+1})  / 2      (with wraparound handling)
        horiz   = seg_len * cos(|avg_inc|)     # inc is angle from horizontal
        dNorth  = horiz * cos(avg_az)
        dEast   = horiz * sin(avg_az)

    Inclination convention (Victorian Geological Survey):
        -90  = straight down (vertical)
          0  = horizontal

    So |inclination| is the angle from horizontal, and:
        horizontal displacement = depth * cos(|inc|)
        vertical   displacement = depth * sin(|inc|)

    *survey_points* is a sorted list of {depth, inclination, azimuth}.
    """
    if not survey_points:
        return 0.0, 0.0

    # If target is at or above the first survey station, no offset.
    if target_depth <= survey_points[0]["depth"]:
        return 0.0, 0.0

    east_off = 0.0
    north_off = 0.0

    for i in range(1, len(survey_points)):
        p1 = survey_points[i - 1]
        p2 = survey_points[i]

        seg_top = p1["depth"]
        seg_bot = min(p2["depth"], target_depth)
        if seg_bot <= seg_top:
            continue

        seg_len = seg_bot - seg_top

        inc1 = p1["inclination"]
        inc2 = p2["inclination"]
        az1 = p1.get("azimuth")
        az2 = p2.get("azimuth")

        if inc1 is None:
            inc1 = inc2 if inc2 is not None else -90.0
        if inc2 is None:
            inc2 = inc1
        if az1 is None:
            az1 = az2 if az2 is not None else 0.0
        if az2 is None:
            az2 = az1

        avg_inc = (inc1 + inc2) / 2.0
        avg_az = _average_azimuth(az1, az2)

        inc_rad = math.radians(abs(avg_inc))
        az_rad = math.radians(avg_az)

        horiz = seg_len * math.cos(inc_rad)
        north_off += horiz * math.cos(az_rad)
        east_off += horiz * math.sin(az_rad)

        if p2["depth"] >= target_depth:
            break

    # Extend beyond last survey station if needed.
    last = survey_points[-1]
    if target_depth > last["depth"]:
        extra = target_depth - last["depth"]
        inc = last["inclination"]
        az = last.get("azimuth") or 0.0
        if inc is None:
            inc = -90.0
        inc_rad = math.radians(abs(inc))
        az_rad = math.radians(az)
        horiz = extra * math.cos(inc_rad)
        north_off += horiz * math.cos(az_rad)
        east_off += horiz * math.sin(az_rad)

    return east_off, north_off


def _average_azimuth(az1, az2):
    """Average two azimuth values handling 0/360 wraparound."""
    diff = az2 - az1
    if diff > 180:
        az1 += 360
    elif diff < -180:
        az2 += 360
    avg = (az1 + az2) / 2.0
    return avg % 360


def build_survey_points(downhole_records, collar_inc, collar_az):
    """Build a sorted list of survey stations for a borehole.

    Inserts a synthetic station at depth 0 from the collar values when the
    first downhole station is deeper than 0.
    """
    points = []

    # Collect valid downhole stations (must have inclination).
    for r in downhole_records:
        if r["inclination"] is not None:
            points.append(dict(r))

    # Prepend collar station at depth 0 if we have collar values.
    if collar_inc is not None:
        collar_point = {
            "depth": 0.0,
            "inclination": collar_inc,
            "azimuth": collar_az if collar_az is not None else 0.0,
        }
        if not points or points[0]["depth"] > 0:
            points.insert(0, collar_point)
        elif points[0]["depth"] == 0:
            # Replace with collar data if azimuth was missing.
            if points[0].get("azimuth") is None and collar_az is not None:
                points[0]["azimuth"] = collar_az

    points.sort(key=lambda p: p["depth"])
    return points


# ---------------------------------------------------------------------------
# DEM utilities
# ---------------------------------------------------------------------------
def sample_dem(dem_data, transform, nodata, x, y):
    """Return the DEM value at (x, y) in DEM CRS coordinates, or None if nodata."""
    col, row = rasterio.transform.TransformMethodsMixin.index(
        None, transform, x, y, offset="center"
    ) if False else _index(transform, x, y)
    if row is None or col is None:
        return None
    if row < 0 or col < 0:
        return None
    val = dem_data[row, col]
    if nodata is not None and abs(float(val) - float(nodata)) < 1.0:
        return None
    return float(val)


def _index(transform, x, y):
    """Convert map coordinates to row/col indices."""
    inv = ~transform
    colf, rowf = inv * (x, y)
    col = int(math.floor(colf))
    row = int(math.floor(rowf))
    return col, row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading borehole header data ...")
    headers = load_header()
    print(f"  Loaded {len(headers):,} borehole headers with valid coordinates")

    print("Loading downhole survey data ...")
    downhole = load_downhole()
    print(f"  Loaded downhole data for {len(downhole):,} boreholes")

    print("Loading lithology records (fault / shear / breccia) ...")
    litho_records = load_lithology_keywords()
    print(f"  Found {len(litho_records):,} matching lithology records")

    print("Opening DEM ...")
    with rasterio.open(DEM_FILE) as dem:
        dem_bounds = dem.bounds
        dem_transform = dem.transform
        dem_nodata = dem.nodata
        dem_data = dem.read(1)

        print(f"  CRS:     {dem.crs}")
        print(f"  Bounds:  L={dem_bounds.left:.1f} B={dem_bounds.bottom:.1f}"
              f" R={dem_bounds.right:.1f} T={dem_bounds.top:.1f}")
        print(f"  Pixel:   {abs(dem_transform[0]):.1f}m x {abs(dem_transform[5]):.1f}m")

        results = []

        for rec in litho_records:
            site_id = rec["siteId"]
            if site_id not in headers:
                continue

            hdr = headers[site_id]
            cx = hdr["easting_vic"]
            cy = hdr["northing_vic"]

            # --- Check collar within DEM -----------------------------------
            if not (dem_bounds.left <= cx <= dem_bounds.right
                    and dem_bounds.bottom <= cy <= dem_bounds.top):
                continue

            # --- DEM surface elevation at collar ---------------------------
            col, row = _index(dem_transform, cx, cy)
            if not (0 <= row < dem.height and 0 <= col < dem.width):
                continue
            dem_elev = float(dem_data[row, col])
            if dem_nodata is not None and abs(dem_elev - float(dem_nodata)) < 1.0:
                continue

            # --- Observation depth (midpoint of from-to) -------------------
            obs_depth = (rec["from_depth"] + rec["to_depth"]) / 2.0

            # --- Build survey & calculate horizontal offset ----------------
            survey_pts = build_survey_points(
                downhole.get(site_id, []),
                hdr["collar_inclination"],
                hdr["collar_azimuth"],
            )
            east_off, north_off = calculate_horizontal_offset(survey_pts, obs_depth)

            obs_e = cx + east_off
            obs_n = cy + north_off

            # --- Check observation point within DEM ------------------------
            if not (dem_bounds.left <= obs_e <= dem_bounds.right
                    and dem_bounds.bottom <= obs_n <= dem_bounds.top):
                continue

            # --- Elevation = surface - depth -------------------------------
            obs_elev = dem_elev - obs_depth

            results.append(
                {
                    "siteId": site_id,
                    "easting": round(obs_e, 2),
                    "northing": round(obs_n, 2),
                    "elevation": round(obs_elev, 2),
                    "dem_surface_elev": round(dem_elev, 2),
                    "obs_depth": round(obs_depth, 2),
                    "from_depth": rec["from_depth"],
                    "to_depth": rec["to_depth"],
                    "collar_inclination": hdr["collar_inclination"] or "",
                    "collar_azimuth": hdr["collar_azimuth"] or "",
                    "description": rec["description"][:200],
                }
            )

    print(f"\n{len(results)} observations within DEM extent")

    # --- Write CSV --------------------------------------------------------
    fieldnames = [
        "siteId", "easting", "northing", "elevation",
        "dem_surface_elev", "obs_depth",
        "from_depth", "to_depth",
        "collar_inclination", "collar_azimuth",
        "description",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV saved to: {OUTPUT_CSV}")

    if results:
        print("\n--- Results ---")
        for r in results:
            print(
                f"  siteId={r['siteId']:<10} "
                f"E={r['easting']:<14.2f} "
                f"N={r['northing']:<14.2f} "
                f"elev={r['elevation']:<10.2f} "
                f"depth={r['obs_depth']:<8.2f} "
                f"desc={r['description'][:60]}"
            )
    else:
        print("\nNo results within DEM extent.")


if __name__ == "__main__":
    main()
