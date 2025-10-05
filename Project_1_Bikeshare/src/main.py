from __future__ import annotations

"""
Entry point for the Bike Sharing project.

This module provides three high-level commands:

1) EDA mode:
   - Fetches/caches the UCI dataset.
   - Cleans and saves a processed copy.
   - Produces comprehensive figures and tables for exploratory analysis.

2) Train mode:
   - Splits the data chronologically into train/holdout/test.
   - Runs forward ablation over interpretable feature groups to find a minimal subset
     that achieves (1 + epsilon) times the best validation RMSE across a ridge lambda grid.
   - Fits a ridge regression (JAX) on train+holdout and evaluates on test.
   - Compares against blind baselines.

3) Test mode:
   - Not Implemented yet.
   
Notes
--------------
- Device / backend:
  We rely on JAXs automatic backend selection and log the chosen device.
  If a GPU is available, JAX will pick it by default, otherwise the CPU is used.
  We further ensure that arrays used for training are placed on the chosen device
  with jax.device_put.
"""

import argparse
import logging
from typing import Literal


from src.config.experiment_config import ExperimentConfig
from src.utils.logging import setup_logging
from src.data.run_eda import run_eda
from src.training.regression_training import run_train

logger = logging.getLogger(__name__)
YMode = Literal["none", "log1p", "sqrt"]

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the program.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields:
        - mode: "eda" or "train"
        - force_fetch: bool to re-pull raw data
        - lam_grid: list of floats for ridge lambda search
        - epsilon: float tolerance for minimal-subset selection
    """
    parser = argparse.ArgumentParser(
        description="Bike Sharing: EDA and Linear Regression (JAX)"
    )
    parser.add_argument(
        "--mode",
        choices=["eda", "train"],
        default="eda",
        help="eda: fetch/clean/visualize, train: split, ablation, ridge fit, metrics",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force re-download of raw data (otherwise use cache if present).",
    )
    parser.add_argument(
        "--lam-grid",
        type=float,
        nargs="+",
        default=[1e-6,9e-5,8e-5,7e-5,6e-5,5e-5,4e-5,3e-5,2e-5,1e-5,8e-4,6e-4,4e-4,2e-4,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3],
        help="Grid of ridge lambda values for selection on the holdout set.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Relative tolerance against best validation RMSE to pick a minimal feature set.",
    )
    parser.add_argument(
        "--y-transform",
        choices=["none", "log1p", "sqrt"],
        default="none",
        help="Target transform used for fitting.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Program entry point.

    - Configures logging.
    - Parses arguments.
    - Dispatches to EDA or training.
    """
    setup_logging()
    cfg = ExperimentConfig.default()
    args = parse_args()

    if args.mode == "eda":
        run_eda(cfg, force_fetch=args.force_fetch)
    elif args.mode == "train":
        run_train(cfg, lam_grid=args.lam_grid, epsilon=args.epsilon, y_transform=args.y_transform)
    else:
        # Defensive programming: argparse should prevent this branch.
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()