import gc
import time
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

STRUCTURAL_POINTS_DIR = Path(r"data\Extracted_Fault_Feature_Attribute")
STRUCTURAL_POINT_FILES = [
    STRUCTURAL_POINTS_DIR / "CWvic1.txt",
    STRUCTURAL_POINTS_DIR / "FSD1.txt",
    STRUCTURAL_POINTS_DIR / "GeoStru2501.txt",
    STRUCTURAL_POINTS_DIR / "GeoStru501.txt"]

OUTPUT_DIR = Path(r"data")

# 5) Chunked structural-point ingestion
MAX_POINTS_PER_SOURCE = 1_000_000_000

def load_structural_points_chunked(
    txt_path: Path,
    chunksize: int = 500000,
    max_points: Optional[int] = MAX_POINTS_PER_SOURCE,
 ) -> pd.DataFrame:
    if not txt_path.exists():
        return pd.DataFrame(columns=["XCoord", "YCoord", "source"])

    rows = []
    n_kept = 0
    usecols = ["XCoord", "YCoord"]
    for chunk in pd.read_csv(txt_path, sep=";", usecols=usecols, chunksize=chunksize):
        c = chunk.copy()
        c["XCoord"] = pd.to_numeric(c["XCoord"], errors="coerce")
        c["YCoord"] = pd.to_numeric(c["YCoord"], errors="coerce")
        c = c.dropna(subset=["XCoord", "YCoord"])

        if max_points is not None:
            remaining = max_points - n_kept
            if remaining <= 0:
                break
            if len(c) > remaining:
                c = c.sample(n=remaining, random_state=RANDOM_SEED)

        n_kept += len(c)
        c["source"] = txt_path.stem
        rows.append(c[["XCoord", "YCoord", "source"]])
        del chunk
        gc.collect()

    if not rows:
        return pd.DataFrame(columns=["XCoord", "YCoord", "source"])

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["XCoord", "YCoord", "source"]).reset_index(drop=True)
    return out

def load_all_structural_points(files: List[Path]) -> pd.DataFrame:
    dfs = []
    inv_rows = []
    for fp in files:
        t0 = time.time()
        df = load_structural_points_chunked(fp)
        if not df.empty:
            dfs.append(df)
        inv_rows.append({
            "source": fp.stem,
            "path": str(fp),
            "n_points_loaded": int(len(df)),
            "load_seconds": float(time.time() - t0),
        })
        print(f"{fp.name}: {len(df):,} points")

    inv = pd.DataFrame(inv_rows)
    inv.to_csv(OUTPUT_DIR / "structural_points_load_inventory.csv", index=False)
    if not dfs:
        return pd.DataFrame(columns=["XCoord", "YCoord", "source"])

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(OUTPUT_DIR / "structural_points_inventory.csv", index=False)
    return out

structural_points_df = load_all_structural_points(STRUCTURAL_POINT_FILES)
print(f"Total structural points loaded: {len(structural_points_df):,}")
