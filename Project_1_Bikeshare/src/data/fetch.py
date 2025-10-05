from __future__ import annotations

from pathlib import Path
import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

RAW_CACHE_BASENAME: Final[str] = "bike_sharing_original.csv"


def fetch_uci_bike(cache_dir: Path, force: bool = False) -> pd.DataFrame:
    """Fetch the UCI Bike Sharing dataset (id=275) and cache as CSV.

    Parameters
    ----------
    cache_dir : Path
        Directory where the raw CSV cache is located.
    force : bool
        If True, refetch from UCI even if a cache exists.

    Returns
    -------
    pd.DataFrame
        The original "day" or "hour" dataset (as delivered by ucimlrepo.data.original).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / RAW_CACHE_BASENAME

    if cache_path.exists() and not force:
        logger.info(f"Loading raw dataset from cache: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("Fetching dataset from UCI (id=275) via ucimlrepo...")
    try:
        from ucimlrepo import fetch_ucirepo  # lazy import
    except ImportError as exc:
        raise RuntimeError(f"ucimlrepo is required to fetch the dataset. Install via `pip install ucimlrepo`.\n"
                           f"Original import error: {exc}")

    bike_sharing = fetch_ucirepo(id=275)
    df = bike_sharing.data.original  # pandas.DataFrame

    logger.info(f"Writing raw cache to {cache_path}")
    df.to_csv(cache_path, index=False)
    return df
