"""Command line entry point for the image classification project."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import yaml

from Project_2_Image_Classification.src.ablation_routines.hyperparameter_search import (
    HyperparameterExperimentManager,
)
from Project_2_Image_Classification.src.classification_routines.evaluation import (
    ClassificationConfig,
    ClassificationRunner,
)
from Project_2_Image_Classification.src.config.config import ConfigManager, ProjectConfig
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
from Project_2_Image_Classification.src.utils.helper import log_jax_runtime_info
from Project_2_Image_Classification.src.utils.logging import LoggingConfig, LoggingManager, get_logger


class CLIApplication:
    """The command line interface of the project."""

    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._config_manager = ConfigManager()
        self._project_config: ProjectConfig = self._config_manager.load()

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
        log_jax_runtime_info()

        self._synchronise_paths(args)

        dataset: PreparedDataset | None = None

        if args.command == "training":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_training(args, dataset)
        if args.command == "classification":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_classification(args, dataset)
        if args.command == "analysis":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_analysis(args, dataset)
        if args.command == "experiments":
            dataset = dataset or self._prepare_dataset(args)
            return self._run_experiments(args, dataset)

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
            model, resolved_model_config = self._build_model(args.model, dataset, model_config)
        except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
            self._logger.error("Failed to construct model '%s': %s", args.model, exc)
            return 1

        trainer = Trainer(model, trainer_config)
        result = trainer.train(dataset)
        self._persist_model_definition(
            trainer_config.output_dir,
            args.model,
            resolved_model_config,
            trainer_config,
        )

        self._logger.info("Training completed. Artefacts stored in %s", trainer_config.output_dir)
        self._logger.info("Training history saved to %s", result.history_path)
        if result.best_validation_metrics:
            for name, value in result.best_validation_metrics.items():
                self._logger.info("Best validation %s=%.4f", name, value)
        return 0

    def _run_classification(self, args: argparse.Namespace, dataset: PreparedDataset) -> int:
        """Handle the ``classification`` sub-command."""

        self._logger.info(
            "Classification routine invoked with checkpoint=%s and input=%s.",
            args.checkpoint,
            args.input_path,
        )
        run_directory = args.checkpoint
        if run_directory.is_file():
            if run_directory.parent.name == "checkpoints":
                run_directory = run_directory.parent.parent
            else:
                run_directory = run_directory.parent

        if args.input_path is not None:
            self._logger.warning(
                "Custom input classification is not yet implemented; evaluating the test split instead."
            )

        config = ClassificationConfig(
            run_directory=run_directory,
            batch_size=args.batch_size or self._project_config.training.eval_batch_size,
            output_directory=args.output_dir,
            save_predictions=not args.no_predictions,
        )
        runner = ClassificationRunner(config)
        metrics = runner.run(dataset)
        for name, value in metrics.items():
            self._logger.info("Classification metric %s=%.4f", name, value)
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

    def _run_experiments(self, args: argparse.Namespace, dataset: PreparedDataset) -> int:
        """Execute the configured ablation study and/or hyper-parameter search."""

        config_data = self._load_config_file(args.config)
        trainer_config = self._build_trainer_config(args, config_data)
        model_overrides = self._extract_model_config(config_data)
        base_model_config = self._resolve_model_config(args.model, dataset, model_overrides)
        base_overrides = dict(model_overrides)

        def model_builder(overrides: Mapping[str, Any]) -> tuple[Any, Any]:
            merged = dict(base_overrides)
            merged.update(overrides)
            return self._build_model(args.model, dataset, merged)

        manager = HyperparameterExperimentManager(
            self._project_config,
            dataset,
            args.model,
            model_builder,
            base_model_config,
            trainer_config,
        )

        if args.mode in {"ablation", "both"}:
            self._logger.info("Starting ablation study for model %s.", args.model)
            manager.run_ablation()
        if args.mode in {"hyperparameter", "both"}:
            self._logger.info("Starting hyper-parameter search for model %s.", args.model)
            manager.run_hyperparameter_search()

        self._logger.info("Experiment workflow completed successfully.")
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
            default=self._project_config.paths.data_dir,
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
            default=None,
            help="Optional directory containing additional images to classify.",
        )
        classification_parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Batch size used during evaluation (defaults to training configuration).",
        )
        classification_parser.add_argument(
            "--no-predictions",
            action="store_true",
            help="Disable saving individual prediction CSV files.",
        )
        classification_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory where classification artefacts will be stored.",
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

        experiments_parser = subparsers.add_parser(
            "experiments",
            help="Run ablation studies and hyper-parameter searches.",
        )
        experiments_parser.add_argument(
            "--mode",
            type=str,
            choices=("ablation", "hyperparameter", "both"),
            default="both",
            help="Select which experimental procedure to execute.",
        )
        experiments_parser.add_argument(
            "--model",
            type=str,
            default="baseline",
            help="Model architecture used as the starting point for experiments.",
        )
        experiments_parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="Optional configuration file with experiment overrides.",
        )
        experiments_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory where training artefacts will be stored.",
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

    def _synchronise_paths(self, args: argparse.Namespace) -> None:
        """Update the persisted configuration with the latest CLI arguments."""

        paths = self._project_config.paths
        changed = False
        if args.data_dir != paths.data_dir:
            paths.data_dir = args.data_dir
            changed = True
        if changed:
            paths.ensure_directories()
            self._config_manager.save(self._project_config)

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
        defaults = self._project_config.training

        default_output = trainer_dict.get("output_dir")
        if args.output_dir:
            trainer_dict["output_dir"] = args.output_dir
        elif default_output is None:
            trainer_dict["output_dir"] = self._default_training_output_dir(args.model, args.data_dir)

        if args.epochs:
            trainer_dict["num_epochs"] = args.epochs
        else:
            trainer_dict.setdefault("num_epochs", defaults.num_epochs)

        if args.batch_size:
            trainer_dict["batch_size"] = args.batch_size
        else:
            trainer_dict.setdefault("batch_size", defaults.batch_size)

        if args.eval_batch_size:
            trainer_dict["eval_batch_size"] = args.eval_batch_size
        else:
            trainer_dict.setdefault("eval_batch_size", defaults.eval_batch_size)

        trainer_dict.setdefault("log_every", defaults.log_every)

        trainer_dict["seed"] = args.random_seed

        optimizer_dict = dict(trainer_dict.get("optimizer", {}))
        if args.optimizer is not None:
            optimizer_dict["name"] = args.optimizer
        optimizer_dict.setdefault("name", defaults.optimizer)
        optimizer_dict.setdefault("weight_decay", defaults.weight_decay)
        trainer_dict["optimizer"] = optimizer_dict

        scheduler_dict = dict(trainer_dict.get("scheduler", {}))
        if args.scheduler is not None:
            scheduler_dict["name"] = args.scheduler
        scheduler_dict.setdefault("name", defaults.scheduler)
        if args.learning_rate is not None:
            scheduler_dict["learning_rate"] = args.learning_rate
        scheduler_dict.setdefault("learning_rate", defaults.learning_rate)
        for key, value in defaults.scheduler_kwargs.items():
            scheduler_dict.setdefault(key, value)
        trainer_dict["scheduler"] = scheduler_dict

        loss_dict = dict(trainer_dict.get("loss", {}))
        loss_dict.setdefault("name", defaults.loss)

        trainer_dict["loss"] = loss_dict

        if "metrics" not in trainer_dict and defaults.metrics:
            trainer_dict["metrics"] = list(defaults.metrics)

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

    def _resolve_model_config(
            self,
            model_name: str,
            dataset: PreparedDataset,
            overrides: Mapping[str, Any],
    ) -> BaselineModelConfig | ImageClassifierConfig:
        """Create a model configuration dataclass from overrides and defaults."""

        train_split = dataset.splits["train"]
        input_shape = tuple(int(dim) for dim in train_split.images.shape[1:])
        class_names = dataset.metadata.get("class_names") if dataset.metadata else None
        if class_names:
            num_classes = len(class_names)
        else:
            num_classes = int(jnp.max(train_split.labels).item()) + 1

        normalized_name = model_name.lower()
        overrides_dict = dict(overrides)
        if normalized_name == "baseline":
            defaults = asdict(self._project_config.baseline_defaults)
            config_dict = {**defaults, **overrides_dict}
            config = BaselineModelConfig(
                input_shape=input_shape,
                hidden_units=int(config_dict.get("hidden_units", defaults["hidden_units"])),
                num_classes=num_classes,
                activation=config_dict.get("activation", defaults["activation"]),
                dropout_rate=float(config_dict.get("dropout_rate", defaults["dropout_rate"])),
                use_bias=bool(config_dict.get("use_bias", defaults["use_bias"])),
                kernel_init=config_dict.get("kernel_init", defaults["kernel_init"]),
                bias_init=config_dict.get("bias_init", defaults["bias_init"]),
            )
            self._logger.info("Constructed baseline model with %d hidden units.", config.hidden_units)
            return config

        if normalized_name in {"cnn", "image_classifier"}:
            defaults = asdict(self._project_config.cnn_defaults)
            config_dict = {**defaults, **overrides_dict}

            conv_blocks_cfg = config_dict.get("conv_blocks") or defaults["conv_blocks"]
            conv_blocks: list[ConvBlockConfig] = []
            for block in conv_blocks_cfg:
                block_settings = {"activation": "relu", "batch_norm": True}
                block_settings.update(dict(block))
                conv_blocks.append(ConvBlockConfig(**block_settings))

            dense_blocks_cfg = config_dict.get("dense_blocks") or defaults["dense_blocks"]
            dense_blocks: list[DenseBlockConfig] = []
            for block in dense_blocks_cfg:
                block_settings = {"activation": "relu"}
                block_settings.update(dict(block))
                dense_blocks.append(DenseBlockConfig(**block_settings))

            return ImageClassifierConfig(
                input_shape=input_shape,
                num_classes=num_classes,
                conv_blocks=conv_blocks,
                dense_blocks=dense_blocks,
                classifier_dropout=float(config_dict.get("classifier_dropout", defaults["classifier_dropout"])),
                global_average_pooling=bool(
                    config_dict.get("global_average_pooling", defaults["global_average_pooling"])),
                classifier_kernel_init=config_dict.get("classifier_kernel_init", defaults["classifier_kernel_init"]),
                classifier_bias_init=config_dict.get("classifier_bias_init", defaults["classifier_bias_init"]),
                classifier_use_bias=bool(config_dict.get("classifier_use_bias", defaults["classifier_use_bias"])),
            )

        raise ValueError(f"Unknown model architecture '{model_name}'.")

    def _build_model(
            self,
            model_name: str,
            dataset: PreparedDataset,
            model_config: Mapping[str, Any],
    ):
        """Instantiate the neural network specified by ``model_name``."""

        normalized_name = model_name.lower()
        config = self._resolve_model_config(model_name, dataset, model_config)

        if normalized_name == "baseline":
            self._logger.info("Constructed baseline model with %d hidden units.", config.hidden_units)
            return create_baseline_model(config), config

        if normalized_name in {"cnn", "image_classifier"}:

            self._logger.info(
                "Constructed CNN with %d convolutional blocks and %d dense blocks.",
                len(config.conv_blocks),
                len(config.dense_blocks),
            )
            return create_image_classifier(config), config

        raise ValueError(f"Unknown model architecture '{model_name}'.")

    def _default_training_output_dir(self, model_name: str, data_dir: Path) -> Path:
        """Return the default directory where training artefacts should be saved."""

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = self._project_config.paths.outputs_dir / "training_runs"
        return (base_dir / f"{model_name}_{timestamp}").resolve()

    def _persist_model_definition(
            self,
            output_dir: Path,
            model_name: str,
            model_config: Any,
            trainer_config: TrainerConfig,
    ) -> None:
        """Store a YAML file describing the model and training configuration."""

        definition_path = output_dir / "model_definition.yaml"
        config_dict = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else dict(model_config)
        payload = {
            "model_name": model_name,
            "config": config_dict,
            "trainer": trainer_config.to_dict(),
            "loss": asdict(trainer_config.loss),
        }
        definition_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self._logger.info("Persisted model definition to %s", definition_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""

    application = CLIApplication()
    return application.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
