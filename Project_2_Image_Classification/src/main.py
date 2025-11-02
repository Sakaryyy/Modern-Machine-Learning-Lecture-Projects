"""Command line entry point for the image classification project."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import yaml

from Project_2_Image_Classification.src.data_analysis.analysis import AnalysisConfig, CIFAR10DatasetAnalyzer
from Project_2_Image_Classification.src.data_loading.data_load_and_save import CIFAR10DataManager, PreparedDataset
from Project_2_Image_Classification.src.models import (
    BaselineModelConfig,
    ImageClassifierConfig,
    create_baseline_model,
    create_image_classifier,
)
from Project_2_Image_Classification.src.models.building_blocks import ConvBlockConfig, DenseBlockConfig
from Project_2_Image_Classification.src.training_routines import Trainer, TrainerConfig
from Project_2_Image_Classification.src.utils.logging import LoggingConfig, LoggingManager, get_logger


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

        dataset: PreparedDataset | None = None

        if args.command == "training":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_training(args, dataset)
        if args.command == "classification":
            return self._run_classification(args)
        if args.command == "analysis":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_analysis(args, dataset)

        self._logger.error("Unknown command '%s'.", args.command)
        return 1

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------
    def _run_training(self, args: argparse.Namespace, dataset: PreparedDataset) -> int:
        """Handle the ``training`` sub-command."""

        split_info = {name: split.images.shape[0] for name, split in dataset.splits.items()}
        self._logger.info(
            "Training routine invoked (model=%s, config=%s). Dataset sizes: %s",
            args.model,
            args.config,
            split_info,
        )

        config_data = self._load_config_file(args.config)
        trainer_config = self._build_trainer_config(args, config_data)
        model_config = self._extract_model_config(config_data)
        try:
            model = self._build_model(args.model, dataset, model_config)
        except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
            self._logger.error("Failed to construct model '%s': %s", args.model, exc)
            return 1

        trainer = Trainer(model, trainer_config)
        result = trainer.train(dataset)

        self._logger.info("Training completed. Artefacts stored in %s", trainer_config.output_dir)
        self._logger.info("Training history saved to %s", result.history_path)
        return 0

    def _run_classification(self, args: argparse.Namespace) -> int:
        """Handle the ``classification`` sub-command."""

        self._logger.info(
            "Classification routine invoked with checkpoint=%s and input=%s.",
            args.checkpoint,
            args.input_path,
        )
        self._logger.warning("Classification routine is not yet implemented.")
        return 0

    def _run_analysis(self, args: argparse.Namespace, dataset: PreparedDataset | None) -> int:
        """Handle the ``analysis`` sub-command."""

        output_dir = args.output_dir or (args.data_dir / "analysis")
        analysis_config = AnalysisConfig(
            data_root=args.data_dir,
            output_dir=output_dir,
            val_split=args.val_split,
            seed=args.random_seed,
            sample_seed=args.sample_seed,
        )
        analyzer = CIFAR10DatasetAnalyzer(analysis_config)

        self._logger.info(
            "Running dataset analysis with output directory %s.",
            output_dir,
        )
        analyzer.run(dataset)
        self._logger.info("Dataset analysis completed successfully.")
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
        training_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory where training artefacts will be stored.",
        )
        training_parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Number of epochs to train the model for.",
        )
        training_parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Mini-batch size used during optimisation.",
        )
        training_parser.add_argument(
            "--eval-batch-size",
            type=int,
            default=None,
            help="Batch size employed during evaluation passes.",
        )
        training_parser.add_argument(
            "--learning-rate",
            type=float,
            default=None,
            help="Base learning rate supplied to the optimiser or scheduler.",
        )
        training_parser.add_argument(
            "--optimizer",
            type=str,
            default=None,
            help="Identifier of the optimiser to employ (e.g. adamw, sgd).",
        )
        training_parser.add_argument(
            "--scheduler",
            type=str,
            default=None,
            help="Learning-rate scheduler to use (e.g. constant, cosine_decay).",
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

        analysis_parser = subparsers.add_parser(
            "analysis",
            help="Perform descriptive analysis of the CIFAR-10 dataset.",
        )
        analysis_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory where analysis artefacts will be stored.",
        )
        analysis_parser.add_argument(
            "--sample-seed",
            type=int,
            default=1234,
            help="Random seed for sampling images in visualisations.",
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

    def _prepare_dataset(self, args: argparse.Namespace) -> PreparedDataset:
        """Download (if required) and prepare the CIFAR-10 dataset."""

        data_manager = CIFAR10DataManager(
            data_root=args.data_dir,
            val_split=args.val_split,
            seed=args.random_seed,
        )
        return data_manager.prepare_data()

    def _load_config_file(self, path: Path | None) -> Mapping[str, Any]:
        """Load an optional YAML configuration file."""

        if path is None:
            return {}

        if not path.exists():
            raise FileNotFoundError(f"Configuration file '{path}' does not exist.")

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, Mapping):
            raise TypeError("The configuration file must contain a mapping at the top level.")

        return data

    def _build_trainer_config(
            self,
            args: argparse.Namespace,
            config_data: Mapping[str, Any],
    ) -> TrainerConfig:
        """Merge CLI arguments and YAML configuration into a :class:`TrainerConfig`."""

        trainer_section = config_data.get("trainer") if config_data else {}
        if trainer_section is None:
            trainer_section = {}
        if trainer_section and not isinstance(trainer_section, Mapping):
            raise TypeError("The 'trainer' section of the configuration must be a mapping.")

        trainer_dict: dict[str, Any] = dict(trainer_section)

        default_output = trainer_dict.get("output_dir")
        if args.output_dir is not None:
            trainer_dict["output_dir"] = args.output_dir
        elif default_output is None:
            trainer_dict["output_dir"] = self._default_training_output_dir(args.model, args.data_dir)

        if args.epochs is not None:
            trainer_dict["num_epochs"] = args.epochs
        else:
            trainer_dict.setdefault("num_epochs", 20)

        if args.batch_size is not None:
            trainer_dict["batch_size"] = args.batch_size
        else:
            trainer_dict.setdefault("batch_size", 128)

        if args.eval_batch_size is not None:
            trainer_dict["eval_batch_size"] = args.eval_batch_size

        trainer_dict["seed"] = args.random_seed

        optimizer_dict = dict(trainer_dict.get("optimizer", {}))
        if args.optimizer is not None:
            optimizer_dict["name"] = args.optimizer
        optimizer_dict.setdefault("name", "adamw")
        trainer_dict["optimizer"] = optimizer_dict

        scheduler_dict = dict(trainer_dict.get("scheduler", {}))
        if args.scheduler is not None:
            scheduler_dict["name"] = args.scheduler
        scheduler_dict.setdefault("name", "constant")
        if args.learning_rate is not None:
            scheduler_dict["learning_rate"] = args.learning_rate
        scheduler_dict.setdefault("learning_rate", 1e-3)
        trainer_dict["scheduler"] = scheduler_dict

        loss_dict = dict(trainer_dict.get("loss", {}))
        trainer_dict["loss"] = loss_dict

        return TrainerConfig.from_dict(trainer_dict)

    def _extract_model_config(self, config_data: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the model configuration section extracted from ``config_data``."""

        if not config_data:
            return {}

        model_section = config_data.get("model")
        if model_section is None:
            return {}
        if not isinstance(model_section, Mapping):
            raise TypeError("The 'model' section of the configuration must be a mapping.")
        return model_section

    def _build_model(
            self,
            model_name: str,
            dataset: PreparedDataset,
            model_config: Mapping[str, Any],
    ):
        """Instantiate the neural network specified by ``model_name``."""

        train_split = dataset.splits["train"]
        input_shape = tuple(int(dim) for dim in train_split.images.shape[1:])
        class_names = dataset.metadata.get("class_names") if dataset.metadata else None
        if class_names:
            num_classes = len(class_names)
        else:
            num_classes = int(jnp.max(train_split.labels).item()) + 1

        normalized_name = model_name.lower()
        if normalized_name == "baseline":
            config_dict = dict(model_config)
            config = BaselineModelConfig(
                input_shape=input_shape,
                hidden_units=int(config_dict.get("hidden_units", 512)),
                num_classes=num_classes,
                activation=config_dict.get("activation", "relu"),
                dropout_rate=float(config_dict.get("dropout_rate", 0.2)),
                use_bias=bool(config_dict.get("use_bias", True)),
                kernel_init=config_dict.get("kernel_init", "he_normal"),
                bias_init=config_dict.get("bias_init", "zeros"),
            )
            self._logger.info("Constructed baseline model with %d hidden units.", config.hidden_units)
            return create_baseline_model(config)

        if normalized_name in {"cnn", "image_classifier"}:
            config_dict = dict(model_config)
            conv_blocks_cfg = config_dict.get("conv_blocks")
            if not conv_blocks_cfg:
                conv_blocks_cfg = [
                    {"features": 32, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.1},
                    {"features": 64, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.2},
                    {"features": 128, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.3},
                ]
            conv_blocks: list[ConvBlockConfig] = []
            for block in conv_blocks_cfg:
                block_settings = {"activation": "relu", "batch_norm": True}
                block_settings.update(dict(block))
                conv_blocks.append(ConvBlockConfig(**block_settings))

            dense_blocks_cfg = config_dict.get("dense_blocks")
            if dense_blocks_cfg is None:
                dense_blocks_cfg = [{"features": 256, "dropout_rate": 0.5}]
            dense_blocks: list[DenseBlockConfig] = []
            for block in dense_blocks_cfg:
                block_settings = {"activation": "relu"}
                block_settings.update(dict(block))
                dense_blocks.append(DenseBlockConfig(**block_settings))

            image_classifier_config = ImageClassifierConfig(
                input_shape=input_shape,
                num_classes=num_classes,
                conv_blocks=conv_blocks,
                dense_blocks=dense_blocks,
                classifier_dropout=float(config_dict.get("classifier_dropout", 0.5)),
                global_average_pooling=bool(config_dict.get("global_average_pooling", True)),
                classifier_kernel_init=config_dict.get("classifier_kernel_init", "he_normal"),
                classifier_bias_init=config_dict.get("classifier_bias_init", "zeros"),
                classifier_use_bias=bool(config_dict.get("classifier_use_bias", True)),
            )
            self._logger.info(
                "Constructed CNN with %d convolutional blocks and %d dense blocks.",
                len(conv_blocks),
                len(dense_blocks),
            )
            return create_image_classifier(image_classifier_config)

        raise ValueError(f"Unknown model architecture '{model_name}'.")

    def _default_training_output_dir(self, model_name: str, data_dir: Path) -> Path:
        """Return the default directory where training artefacts should be saved."""

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return (data_dir / "training_runs" / f"{model_name}_{timestamp}").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""

    application = CLIApplication()
    return application.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
