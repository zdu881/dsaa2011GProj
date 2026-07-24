"""Prepare the Covertype dataset for the DSAA2011 project.

The primary source is the UCI Machine Learning Repository raw compressed file.
The script preserves that raw file and writes a headered, compressed CSV used by
the analysis pipeline.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "covtype.data.gz"
CSV_PATH = DATA_DIR / "covertype.csv.gz"
SOURCE_PATH = DATA_DIR / "SOURCE.txt"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_FEATURES = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_FEATURES = [f"Soil_Type_{i}" for i in range(1, 41)]
FEATURE_NAMES = CONTINUOUS_FEATURES + WILDERNESS_FEATURES + SOIL_FEATURES
ALL_COLUMNS = FEATURE_NAMES + ["Cover_Type"]
DTYPE_MAP = {
    **{col: "float32" for col in CONTINUOUS_FEATURES},
    **{col: "uint8" for col in WILDERNESS_FEATURES + SOIL_FEATURES + ["Cover_Type"]},
}


def download_raw_file() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1_000_000:
        return

    request = urllib.request.Request(
        UCI_URL,
        headers={"User-Agent": "Mozilla/5.0 DSAA2011 Covertype project"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with RAW_PATH.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)


def validate_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as handle:
        handle.read(1024)


def prepare_covtype_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 1_000_000:
        return pd.read_csv(CSV_PATH, dtype=DTYPE_MAP)

    download_raw_file()
    validate_gzip(RAW_PATH)

    df = pd.read_csv(RAW_PATH, header=None, names=ALL_COLUMNS)
    df = df.astype(DTYPE_MAP)
    df.to_csv(CSV_PATH, index=False, compression="gzip")

    SOURCE_PATH.write_text(
        "\n".join(
            [
                "Dataset: Covertype / Forest Cover Type",
                f"Source: {UCI_URL}",
                "Rows: 581012",
                "Features: 54 cartographic variables plus Cover_Type target",
                "Target labels: 1 Spruce/Fir, 2 Lodgepole Pine, 3 Ponderosa Pine,",
                "  4 Cottonwood/Willow, 5 Aspen, 6 Douglas-fir, 7 Krummholz",
                "Prepared file: data/covertype.csv.gz",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return df


def main() -> None:
    df = prepare_covtype_data()
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df["Cover_Type"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
