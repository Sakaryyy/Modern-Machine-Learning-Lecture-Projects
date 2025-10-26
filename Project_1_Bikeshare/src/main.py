from __future__ import annotations

"""
Entry point for the Bike Sharing project.

This module provides three high-level commands:

1) EDA mode:
   - Fetches/caches the UCI dataset.
   - Cleans and saves a processed copy.
   - Produces comprehensive figures and tables for exploratory analysis.

2) Train mode:
   - Splits the data chronologically into train/validation/test.
   - Runs forward ablation over interpretable feature groups to find a minimal subset
     that achieves (1 + epsilon) times the best validation RMSE across a ridge lambda grid.
   - Fits a ridge regression (JAX) on train+validation and evaluates on test.
   - Compares against blind baselines.

3) Classify mode:
   - Runs multinomial logistic regression (JAX) to predict the hour-of-day.
   - Mirrors the regression pipeline by performing feature ablation and saving
     tables/figures under outputs/.
   
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
from src.training.regression_training import run_regression
from src.training.classification_training import run_classification

logger = logging.getLogger(__name__)
YMode = Literal["none", "log1p", "sqrt"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the program.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields:
        - mode: "eda", "train", or "classify"
        - force_fetch: bool to re-pull raw data
        - lam_grid: list of floats for ridge lambda search
        - epsilon: float tolerance for minimal-subset selection
        - classify_reg_grid / classify_learning_rate / classify_max_iter / classify_tol
          for the classification experiment
    """
    parser = argparse.ArgumentParser(
        description="Bike Sharing: EDA, Regression, and Classification in JAX"
    )
    parser.add_argument(
        "--mode",
        choices=["eda", "regression", "classify"],
        default="eda",
        help=(
            "eda: fetch/clean/visualize; regression: ridge regression; "
            "classify: multinomial logistic regression"
        ),
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
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        help="Grid of ridge lambda values for selection on the validation set.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00001,
        help="Relative tolerance against best validation RMSE to pick a minimal feature set.",
    )
    parser.add_argument(
        "--classify-reg-grid",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help="Grid of L2 penalties to evaluate during classification ablation.",
    )
    parser.add_argument(
        "--classify-learning-rate",
        type=float,
        default=1.0,
        help="Learning rate for gradient descent in softmax regression.",
    )
    parser.add_argument(
        "--classify-max-iter",
        type=int,
        default=5000,
        help="Maximum number of gradient steps for classification mode.",
    )
    parser.add_argument(
        "--classify-tol",
        type=float,
        default=1e-6,
        help="Gradient-norm convergence tolerance for classification mode.",
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
    - Dispatches to EDA, regression training, or classification training.
    """
    setup_logging()
    cfg = ExperimentConfig.default()
    args = parse_args()

    if args.mode == "eda":
        run_eda(cfg, force_fetch=args.force_fetch)
    elif args.mode == "regression":
        run_regression(cfg, lam_grid=args.lam_grid, epsilon=args.epsilon, y_transform=args.y_transform)
    elif args.mode == "classify":
        run_classification(
            cfg,
            reg_grid=args.classify_reg_grid,
            epsilon=args.epsilon,
            learning_rate=args.classify_learning_rate,
            max_iter=args.classify_max_iter,
            tol=args.classify_tol,
        )
    else:
        # Defensive programming: argparse should prevent this branch.
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()