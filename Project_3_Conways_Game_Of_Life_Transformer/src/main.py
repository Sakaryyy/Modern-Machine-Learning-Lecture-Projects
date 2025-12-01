"""Command line entrypoint for training and generation."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.training.training_routine import (
    generate_predictions,
    train_and_evaluate,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import LoggingConfig, LoggingManager, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conway Game of Life Transformer")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Training mode
    train_parser = subparsers.add_parser("train", help="Train a transformer model")
    train_parser.add_argument("--height", type=int, default=16)
    train_parser.add_argument("--width", type=int, default=16)
    train_parser.add_argument("--num-samples", type=int, default=20000)
    train_parser.add_argument("--train-fraction", type=float, default=0.7)
    train_parser.add_argument("--val-fraction", type=float, default=0.15)
    train_parser.add_argument("--density", type=float, default=0.5)
    train_parser.add_argument("--seed", type=int, default=12)
    train_parser.add_argument("--stochastic", action="store_true", help="Use stochastic rule mixture")
    train_parser.add_argument("--p-stochastic", type=float, default=0.8)
    train_parser.add_argument("--anomaly", action="store_true", help="Enable anomaly detection dataset")
    train_parser.add_argument("--p-anomaly", type=float, default=0.6)
    train_parser.add_argument("--anomaly-fraction", type=float, default=0.1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--d-model", type=int, default=64)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=3)
    train_parser.add_argument("--mlp-dim", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--local", action="store_true", help="Use local attention")
    train_parser.add_argument("--window-radius", type=int, default=1)
    train_parser.add_argument("--coord-features", action="store_true", help="Include coordinate features")
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    # Generation mode
    gen_parser = subparsers.add_parser("generate", help="Use a trained model for inference")
    gen_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    gen_parser.add_argument("--height", type=int, required=True)
    gen_parser.add_argument("--width", type=int, required=True)
    gen_parser.add_argument("--batch-size", type=int, default=32)
    gen_parser.add_argument("--num-samples", type=int, default=16, help="Number of random grids to generate")
    gen_parser.add_argument("--density", type=float, default=0.5)

    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    log_cfg = LoggingConfig(log_directory=output_dir, log_to_file=True, filename="training.log")
    LoggingManager(log_cfg).configure()


def run_training(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_dir / f"run_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    logger = get_logger(__name__)

    data_cfg = DataConfig(
        height=args.height,
        width=args.width,
        num_samples=args.num_samples,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        density=args.density,
        seed=args.seed,
        stochastic=args.stochastic or args.anomaly,
        p_stochastic=args.p_stochastic,
        anomaly_detection=args.anomaly,
        p_anomaly=args.p_anomaly,
        anomaly_fraction=args.anomaly_fraction,
        cache_dir=run_dir / "cache",
        use_cache=False,
    )

    model_cfg = TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        dropout_rate=args.dropout,
        use_local_attention=args.local,
        window_radius=args.window_radius,
        use_coord_features=args.coord_features,
    )

    train_cfg = TrainingConfig(learning_rate=args.learning_rate, num_epochs=args.epochs, batch_size=args.batch_size)

    logger.info("Starting training with output directory %s", run_dir)
    train_and_evaluate(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, output_root=run_dir, seed=args.seed)


def run_generation(args: argparse.Namespace) -> None:
    output_dir = Path("outputs/generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    logger = get_logger(__name__)

    model_cfg = TransformerConfig()
    rng = np.random.default_rng(0)
    inputs = (rng.random((args.num_samples, args.height, args.width)) < args.density).astype(np.int32)
    preds = generate_predictions(args.checkpoint, model_cfg, inputs, args.batch_size)
    np.save(output_dir / "inputs.npy", inputs)
    np.save(output_dir / "predictions.npy", preds)
    logger.info("Saved generation inputs and predictions to %s", output_dir)


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "generate":
        run_generation(args)


if __name__ == "__main__":
    main()
