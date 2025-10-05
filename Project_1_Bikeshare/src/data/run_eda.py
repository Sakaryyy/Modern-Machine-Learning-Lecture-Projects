from __future__ import annotations

import logging

from src.visualization import eda_plotting
from src.config.experiment_config import ExperimentConfig
from src.data.fetch import fetch_uci_bike
from src.data.clean import clean_bike_df, save_processed

logger = logging.getLogger(__name__)


def run_eda(cfg: ExperimentConfig, force_fetch: bool) -> None:
    """
    Run the full EDA suite and save all outputs.

    Parameters
    ----------
    cfg : ExperimentConfig
        Top-level configuration (paths, splits, features).
    force_fetch : bool
        If True, re-download raw data even if a cache exists.

    Notes
    ------------
    - Saves raw cache if needed.
    - Saves cleaned processed data to `data/processed`.
    - Saves figures to `outputs/figures` and tables to `outputs/tables`.
    """
    cfg.paths.ensure_exists()
    raw_df = fetch_uci_bike(cache_dir=cfg.paths.raw_dir, force=force_fetch)
    clean_df = clean_bike_df(raw_df)
    save_processed(clean_df, cfg.paths.processed_dir)

    results = eda.run_all(clean_df, cfg.paths.eda_figures_dir)
    for name, path in results.items():
        logger.info(f"Saved {name} -> {path}")