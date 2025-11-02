"""Command line entry point for the image classification project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .data_loading.data_load_and_save import CIFAR10DataManager, PreparedDataset
from .utils.logging import LoggingConfig, LoggingManager, get_logger


class CLIApplication:
    """The command line interface of the project."""

    DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse arguments, prepare data and dispatch to the selected command."""

        parser = self._build_parser()
        args = parser.parse_args(argv)

        if not 0.0 < args.val_split < 1.0:
            parser.error("--val-split must be strictly between 0 and 1.")

        log_level = self._resolve_log_level(args.log_level)
        logging_config = LoggingConfig(
            level=log_level,
            log_to_file=args.log_dir is not None,
            log_directory=args.log_dir,
            filename=args.log_file,
        )
        LoggingManager(logging_config).configure()

        self._logger.info("Launching application with command '%s'.", args.command)

        data_manager = CIFAR10DataManager(
            data_root=args.data_dir,
            val_split=args.val_split,
            seed=args.random_seed,
        )
        dataset = data_manager.prepare_data()

        if args.command == "training":
            return self._run_training(args, dataset)
        if args.command == "classification":
            return self._run_classification(args, dataset)

        self._logger.error("Unknown command '%s'.", args.command)
        return 1

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------
    def _run_training(self, args: argparse.Namespace, dataset: PreparedDataset) -> int:
        """Handle the ``training`` sub-command.
        """

        split_info = {name: split.images.shape[0] for name, split in dataset.splits.items()}
        self._logger.info(
            "Training routine invoked (model=%s, config=%s). Dataset sizes: %s",
            args.model,
            args.config,
            split_info,
        )
        self._logger.warning("Training routine is not yet implemented.")
        return 0

    def _run_classification(self, args: argparse.Namespace, dataset: PreparedDataset) -> int:
        """Handle the ``classification`` sub-command.
        """

        self._logger.info(
            "Classification routine invoked with checkpoint=%s and input=%s.",
            args.checkpoint,
            args.input_path,
        )
        self._logger.warning("Classification routine is not yet implemented.")
        return 0

    # ------------------------------------------------------------------
    # Argument parsing helpers
    # ------------------------------------------------------------------
    def _build_parser(self) -> argparse.ArgumentParser:
        """Create the command line argument parser."""

        parser = argparse.ArgumentParser(
            description="Deep learning experiments for CIFAR-10 image classification.",
        )
        parser.add_argument(
            "--data-dir",
            type=Path,
            default=self.DEFAULT_DATA_DIR,
            help="Directory where raw and processed datasets are stored.",
        )
        parser.add_argument(
            "--val-split",
            type=float,
            default=0.1,
            help="Fraction of the training data to use for validation.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=10,
            help="Seed controlling the train/validation split.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
        )
        parser.add_argument(
            "--log-dir",
            type=Path,
            default=None,
            help="Optional directory where log files should be saved.",
        )
        parser.add_argument(
            "--log-file",
            type=str,
            default="image_classification.log",
            help="Filename for the optional log file.",
        )

        subparsers = parser.add_subparsers(dest="command", required=True)

        training_parser = subparsers.add_parser(
            "training",
            help="Train a neural network for CIFAR-10 classification.",
        )
        training_parser.add_argument(
            "--model",
            type=str,
            default="baseline",
            help="Name of the model architecture to train.",
        )
        training_parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="Optional configuration file describing hyperparameters.",
        )

        classification_parser = subparsers.add_parser(
            "classification",
            help="Run inference using a previously trained model.",
        )
        classification_parser.add_argument(
            "--checkpoint",
            type=Path,
            required=True,
            help="Path to the trained model checkpoint to load.",
        )
        classification_parser.add_argument(
            "--input-path",
            type=Path,
            required=True,
            help="Directory or file containing the images to classify.",
        )

        return parser

    def _resolve_log_level(self, level: str) -> int:
        """Translate a textual log level into the corresponding constant."""

        normalized = level.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            self._logger.warning(
                "Log level '%s' not recognised. Falling back to INFO.",
                level,
            )
            return logging.INFO
        return getattr(logging, normalized)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""

    application = CLIApplication()
    return application.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
