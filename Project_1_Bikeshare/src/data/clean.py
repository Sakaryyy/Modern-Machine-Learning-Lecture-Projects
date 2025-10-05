from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are leaking target information and/or should be no features
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "casual",
    "registered",
)

# Not necessary columns
DROP_IF_PRESENT: tuple[str, ...] = (
    "instant",
)


def _infer_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Infer a datetime index.

    Returns a new DataFrame with a 'timestamp' column and a DatetimeIndex.
    """
    df = df.copy()

    # Normalize column names just in case
    cols = {c.lower(): c for c in df.columns}
    if "dteday" in cols:
        date_col = cols["dteday"]
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if "hr" in cols:
            hour_col = cols["hr"]
            dt = dt + pd.to_timedelta(df[hour_col].astype(int), unit="h")
    else:
        # Fallback: try to find something date-like
        candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if not candidates:
            raise ValueError("Could not locate a datetime column in the dataset.")
        dt = pd.to_datetime(df[candidates[0]], errors="coerce")

    df["timestamp"] = dt
    df = df.set_index("timestamp").sort_index()
    return df

def clean_bike_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type the UCI bike sharing dataset.

    Steps
    -----
    1) Create a proper DatetimeIndex (hourly if 'hr' is present).
    2) Remove target columns ('casual', 'registered') if present.
    3) Drop non-feature index-like columns ('instant') if present.
    4) Ensure target column 'cnt' exists and is numeric.
    5) Coerce common feature dtypes (category for 'season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit') if present.
    6) Drop rows with missing critical values (timestamp, cnt).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by timestamp with a 'cnt' column.
    """
    df = _infer_datetime(df)

    for c in LEAKAGE_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=[c])

    for c in DROP_IF_PRESENT:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "cnt" not in df.columns:
        raise ValueError("Target column 'cnt' not found in dataset.")

    # Coerce target
    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")

    # Common categorical columns (if present)
    for cat in ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"):
        if cat in df.columns:
            df[cat] = df[cat].astype("Int64").astype("category")

    # Ensure numeric weather columns are numeric
    for num in ("temp", "atemp", "hum", "windspeed"):
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce" )

    # Drop rows with missing timestamp or cnt
    df = df.dropna(subset=["cnt"]).copy()

    return df

def save_processed(df: pd.DataFrame, processed_dir: Path, basename: str = "bike_clean") -> dict[str, Path]:
    """Save processed dataset to csv and binary parquet for downstream reproducibility."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = processed_dir / f"{basename}.parquet"
    out_csv = processed_dir / f"{basename}.csv"
    df.to_parquet(out_parquet, index=True)
    df.to_csv(out_csv, index=True)
    return {"parquet": out_parquet, "csv": out_csv}
