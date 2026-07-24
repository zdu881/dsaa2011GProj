"""Download and normalize the Student Dropout dataset.

The course PDF describes the UCI Student Dropout dataset. The official UCI
archive is semicolon-delimited, while the project statement assumes a normal
`pd.read_csv("student_dropout.csv")` call. This script preserves the raw file
and writes a comma-separated, cleaned copy at the repository root.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "student_dropout_uci.zip"
RAW_CSV_PATH = DATA_DIR / "student_dropout_raw_semicolon.csv"
CLEAN_CSV_PATH = ROOT / "student_dropout.csv"
SOURCE_NOTE_PATH = DATA_DIR / "SOURCE.txt"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)


def download_if_needed() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        return

    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)


def normalize_csv() -> pd.DataFrame:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        raw_bytes = zf.read("data.csv")

    RAW_CSV_PATH.write_bytes(raw_bytes)
    df = pd.read_csv(RAW_CSV_PATH, sep=";")
    df.columns = [col.strip() for col in df.columns]
    df.to_csv(CLEAN_CSV_PATH, index=False)

    SOURCE_NOTE_PATH.write_text(
        "Dataset: Predict Students' Dropout and Academic Success\n"
        f"Source: {UCI_URL}\n"
        "Original file: data.csv in the downloaded UCI zip archive\n"
        "Local normalized file: ../student_dropout.csv\n",
        encoding="utf-8",
    )
    return df


def main() -> None:
    download_if_needed()
    df = normalize_csv()
    print(f"Wrote {CLEAN_CSV_PATH}")
    print(f"Shape: {df.shape}")
    print(df["Target"].value_counts().to_string())


if __name__ == "__main__":
    main()
