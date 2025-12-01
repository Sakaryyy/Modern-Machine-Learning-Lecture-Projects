"""Command line entrypoint for training and generation."""

import argparse
from argparse import BooleanOptionalAction
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.data_pipelines import sample_random_grid
from Project_3_Conways_Game_Of_Life_Transformer.src.training.training_routine import (
    analyse_rule_adherence,
    generate_predictions,
    load_run_artifact_configs,
    train_and_evaluate,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import LoggingConfig, LoggingManager, get_logger
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.rule_analysis import compute_rule_categories
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import plot_grid_triplet


def default_config_dict() -> Dict[str, Dict[str, Any]]:
    """Return a dictionary with sensible default configuration values."""

    data_defaults = asdict(DataConfig(height=16, width=16, num_samples=20000))
    data_defaults["cache_dir"] = "cache"
    if isinstance(data_defaults.get("density_range"), tuple):
        data_defaults["density_range"] = list(data_defaults["density_range"])
    model_defaults = asdict(TransformerConfig())
    training_defaults = asdict(TrainingConfig())
    logging_defaults = {"output_dir": "outputs", "log_to_file": True, "filename": "training.log"}
    return {
        "data": data_defaults,
        "model": model_defaults,
        "training": training_defaults,
        "logging": logging_defaults,
    }


def load_or_create_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a YAML configuration file or create one from defaults.

    Parameters
    ----------
    config_path : pathlib.Path
        Location of the configuration file.

    Returns
    -------
    dict
        Nested dictionary with ``data``, ``model``, ``training`` and
        ``logging`` sections populated.
    """

    config_path = config_path.resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = default_config_dict()

    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(defaults, sort_keys=False))
        print(f"Created default configuration at {config_path}")
        return defaults

    loaded = yaml.safe_load(config_path.read_text()) or {}
    merged: Dict[str, Dict[str, Any]] = {}
    for section, default_values in defaults.items():
        merged_section = {**default_values, **(loaded.get(section, {}) or {})}
        merged[section] = merged_section

    return merged


def configure_logging(output_dir: Path, filename: str, log_to_file: bool) -> None:
    """Configure project logging based on the provided options."""

    log_cfg = LoggingConfig(log_directory=output_dir, log_to_file=log_to_file, filename=filename)
    LoggingManager(log_cfg).configure()


def _coerce_density_range(values: Tuple[float, float] | list | None) -> Tuple[float, float] | None:
    if values is None:
        return None
    return float(values[0]), float(values[1])


def build_data_config(section: Dict[str, Any], args: argparse.Namespace) -> DataConfig:
    """Construct a :class:`DataConfig` by combining config file and CLI args."""

    cfg = dict(section)
    overrides = {
        "height": args.height,
        "width": args.width,
        "num_samples": args.num_samples,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "p_stochastic": args.p_stochastic,
        "p_anomaly": args.p_anomaly,
        "anomaly_fraction": args.anomaly_fraction,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if args.density_range is not None:
        cfg["density_range"] = _coerce_density_range(args.density_range)
    elif args.density is not None:
        cfg["density"] = args.density
        cfg["density_range"] = None

    if args.stochastic is not None:
        cfg["stochastic"] = args.stochastic
    if args.anomaly is not None:
        cfg["anomaly_detection"] = args.anomaly
    if args.use_cache is not None:
        cfg["use_cache"] = args.use_cache
    if args.overwrite_cache is not None:
        cfg["overwrite_cache"] = args.overwrite_cache

    cfg["cache_dir"] = Path(cfg.get("cache_dir", "cache"))
    return DataConfig(**cfg)


def build_model_config(section: Dict[str, Any], args: argparse.Namespace) -> TransformerConfig:
    """Construct :class:`TransformerConfig` from config file and CLI args."""

    cfg = dict(section)
    overrides = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_dim": args.mlp_dim,
        "dropout_rate": args.dropout,
        "window_radius": args.window_radius,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if args.local is not None:
        cfg["use_local_attention"] = args.local
    if args.coord_features is not None:
        cfg["use_coord_features"] = args.coord_features

    return TransformerConfig(**cfg)


def build_training_config(section: Dict[str, Any], args: argparse.Namespace) -> TrainingConfig:
    """Construct :class:`TrainingConfig` from config and optional CLI overrides."""

    cfg = dict(section)
    overrides = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "decay_steps": args.decay_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "max_grad_norm": args.max_grad_norm,
        "l2_reg": args.l2_reg,
        "eval_larger_lattice": args.eval_larger_lattice,
        "larger_height": args.larger_height,
        "larger_width": args.larger_width,
        "num_generalization_samples": args.num_generalization_samples,
        "generalization_density": args.generalization_density,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    return TrainingConfig(**cfg)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with config awareness."""

    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the Conway Game of Life transformer. Values are "
            "loaded from config.yaml when available and can be overridden via CLI."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file. If missing, it will be created with defaults.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Training mode
    train_parser = subparsers.add_parser("train", help="Train a transformer model")
    train_parser.add_argument("--height", type=int, help="Grid height; overrides config file")
    train_parser.add_argument("--width", type=int, help="Grid width; overrides config file")
    train_parser.add_argument("--num-samples", type=int, help="Number of snapshot pairs to generate")
    train_parser.add_argument("--train-fraction", type=float, help="Training split fraction")
    train_parser.add_argument("--val-fraction", type=float, help="Validation split fraction")
    train_parser.add_argument("--density", type=float, help="Fixed Bernoulli density for initialization")
    train_parser.add_argument(
        "--density-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Range [low, high] for sampling densities; prevents bias toward a single density.",
    )
    train_parser.add_argument("--seed", type=int, help="Random seed for data generation and training")
    train_parser.add_argument(
        "--stochastic",
        action=BooleanOptionalAction,
        default=None,
        help="Enable stochastic rule mixture for data generation.",
    )
    train_parser.add_argument("--p-stochastic", type=float, help="Probability of using rule 23/3 in stochastic mode")
    train_parser.add_argument(
        "--anomaly",
        action=BooleanOptionalAction,
        default=None,
        help="Enable anomaly detection dataset construction.",
    )
    train_parser.add_argument("--p-anomaly", type=float, help="Probability for anomalous samples in anomaly mode")
    train_parser.add_argument("--anomaly-fraction", type=float, help="Fraction of anomalies in the dataset")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Mini-batch size for optimization")
    train_parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adam", "sgd"],
        help="Optimizer to use during training",
    )
    train_parser.add_argument("--weight-decay", type=float, help="Weight decay for regularisation")
    train_parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["constant", "cosine", "linear"],
        help="Learning rate schedule type",
    )
    train_parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps for LR schedule")
    train_parser.add_argument("--decay-steps", type=int, help="Number of decay steps for LR schedule")
    train_parser.add_argument(
        "--min-lr-ratio",
        type=float,
        help="Final LR as fraction of peak learning rate for schedulers",
    )
    train_parser.add_argument("--max-grad-norm", type=float, help="Gradient clipping threshold")
    train_parser.add_argument("--l2-reg", type=float, help="L2 regularisation weight added to the loss")
    train_parser.add_argument("--d-model", type=int, help="Transformer hidden dimension")
    train_parser.add_argument("--num-heads", type=int, help="Number of attention heads")
    train_parser.add_argument("--num-layers", type=int, help="Number of transformer blocks")
    train_parser.add_argument("--mlp-dim", type=int, help="Hidden dimension in the feedforward sub-layer")
    train_parser.add_argument("--dropout", type=float, help="Dropout rate")
    train_parser.add_argument(
        "--local",
        action=BooleanOptionalAction,
        default=None,
        help="Use local attention instead of global attention.",
    )
    train_parser.add_argument("--window-radius", type=int, help="Radius of the local attention window")
    train_parser.add_argument(
        "--coord-features",
        action=BooleanOptionalAction,
        default=None,
        help="Concatenate coordinate embeddings to the input tokens.",
    )
    train_parser.add_argument(
        "--use-cache",
        action=BooleanOptionalAction,
        default=None,
        help="Load from or persist dataset caches under the run directory.",
    )
    train_parser.add_argument(
        "--overwrite-cache",
        action=BooleanOptionalAction,
        default=None,
        help="Force regeneration of cached datasets even if present.",
    )
    train_parser.add_argument(
        "--eval-larger-lattice",
        action=BooleanOptionalAction,
        default=None,
        help="Evaluate on a larger lattice after training",
    )
    train_parser.add_argument("--larger-height", type=int, help="Height for larger lattice evaluation")
    train_parser.add_argument("--larger-width", type=int, help="Width for larger lattice evaluation")
    train_parser.add_argument(
        "--num-generalization-samples",
        type=int,
        help="Number of samples for the larger lattice generalisation eval",
    )
    train_parser.add_argument(
        "--generalization-density",
        type=float,
        help="Optional fixed density for larger lattice evaluation",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for training artefacts. Defaults to logging.output_dir from the config file.",
    )

    # Generation mode
    gen_parser = subparsers.add_parser("generate", help="Use a trained model for inference")
    gen_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    gen_parser.add_argument("--height", type=int, help="Height of generated grids; defaults to config file")
    gen_parser.add_argument("--width", type=int, help="Width of generated grids; defaults to config file")
    gen_parser.add_argument("--batch-size", type=int, default=64)
    gen_parser.add_argument("--num-samples", type=int, default=16, help="Number of random grids to generate")
    gen_parser.add_argument("--density", type=float, help="Fixed density for generation if density-range is absent")
    gen_parser.add_argument(
        "--density-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Range [low, high] for densities when generating inputs.",
    )
    gen_parser.add_argument("--seed", type=int, help="Seed for input sampling during generation")

    return parser.parse_args()


def run_training(args: argparse.Namespace, config_data: Dict[str, Dict[str, Any]]) -> None:
    """Entry point for the training subcommand."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_section = config_data.get("logging", {})
    output_root = Path(args.output_dir) if args.output_dir is not None else Path(
        logging_section.get("output_dir", "outputs"))
    run_dir = (output_root / f"run_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir, filename=logging_section.get("filename", "training.log"),
                      log_to_file=bool(logging_section.get("log_to_file", True)))
    logger = get_logger(__name__)

    data_cfg = build_data_config(config_data.get("data", {}), args)
    model_cfg = build_model_config(config_data.get("model", {}), args)
    train_cfg = build_training_config(config_data.get("training", {}), args)

    cache_dir = Path(data_cfg.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (run_dir / cache_dir).resolve()
    data_cfg.cache_dir = cache_dir
    logger.info("Caching datasets under %s", data_cfg.cache_dir)

    logger.info("Starting training with output directory %s", run_dir)
    train_and_evaluate(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        output_root=run_dir,
        seed=data_cfg.seed,
    )


def run_generation(args: argparse.Namespace, config_data: Dict[str, Dict[str, Any]]) -> None:
    """Entry point for the generation subcommand."""
    logging_section = config_data.get("logging", {})
    output_dir = Path(logging_section.get("output_dir", "outputs")) / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, filename=logging_section.get("filename", "generation.log"),
                      log_to_file=bool(logging_section.get("log_to_file", True)))
    logger = get_logger(__name__)

    run_dir = args.checkpoint.resolve().parent.parent
    saved_data_cfg, saved_model_cfg, _ = load_run_artifact_configs(run_dir)

    data_section = asdict(saved_data_cfg) if saved_data_cfg is not None else config_data.get("data", {})
    height = args.height or int(data_section.get("height", 16))
    width = args.width or int(data_section.get("width", 16))
    density_range = _coerce_density_range(
        args.density_range) if args.density_range is not None else _coerce_density_range(
        data_section.get("density_range"))
    density = args.density if args.density is not None else float(data_section.get("density", 0.5))
    seed = args.seed if args.seed is not None else int(data_section.get("seed", 0))

    model_cfg = saved_model_cfg or TransformerConfig()
    rng = np.random.default_rng(seed)
    densities = (
        rng.uniform(low=density_range[0], high=density_range[1], size=args.num_samples)
        if density_range is not None
        else np.full((args.num_samples,), density)
    )
    inputs = np.empty((args.num_samples, height, width), dtype=np.int32)
    for i, dens in enumerate(densities):
        inputs[i] = sample_random_grid(height, width, float(dens), rng)

    preds = generate_predictions(args.checkpoint, model_cfg, inputs, args.batch_size)
    deterministic_targets = np.empty_like(inputs, dtype=int)
    for idx, grid in enumerate(inputs):
        deterministic_targets[idx], _ = compute_rule_categories(grid)

    rule_dir = output_dir / "rule_diagnostics"
    rule_dir.mkdir(parents=True, exist_ok=True)
    analyse_rule_adherence(
        inputs=inputs,
        targets=deterministic_targets,
        probs=preds,
        output_dir=rule_dir,
        prefix="generation",
    )
    plot_grid_triplet(
        inputs[0],
        deterministic_targets[0],
        preds[0],
        save_path=rule_dir / "generation_example.png",
        title="Generation example with deterministic target",
    )

    np.save(output_dir / "inputs.npy", inputs)
    np.save(output_dir / "densities.npy", densities)
    np.save(output_dir / "predictions.npy", preds)
    logger.info("Saved generation inputs, densities, and predictions to %s", output_dir)


def main() -> None:
    args = parse_args()
    config_data = load_or_create_config(args.config)
    if args.mode == "train":
        run_training(args, config_data)
    elif args.mode == "generate":
        run_generation(args, config_data)


if __name__ == "__main__":
    main()
