"""Helpers for anomaly detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.config.anomaly_config import AnomalyExperimentConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import ModelConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.training.training_routine import train_and_evaluate
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import plot_multiple_roc_curves


def run_anomaly_experiment_for_size(
        model_config: ModelConfig,
        exp_config: AnomalyExperimentConfig,
        output_root: Path,
        seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train and evaluate anomaly detection for a single lattice size."""

    data_cfg = DataConfig(
        height=exp_config.height,
        width=exp_config.width,
        num_samples=exp_config.num_train + exp_config.num_val + exp_config.num_test,
        train_fraction=exp_config.num_train / (exp_config.num_train + exp_config.num_val + exp_config.num_test),
        val_fraction=exp_config.num_val / (exp_config.num_train + exp_config.num_val + exp_config.num_test),
        density=0.5,
        seed=seed,
        stochastic=True,
        p_stochastic=exp_config.p_train,
        anomaly_detection=True,
        p_anomaly=exp_config.p_anomaly,
        anomaly_fraction=exp_config.anomaly_fraction,
        cache_prefix=f"anom_{exp_config.height}x{exp_config.width}",
    )

    train_cfg = TrainingConfig(
        learning_rate=exp_config.learning_rate,
        num_epochs=exp_config.num_epochs,
        batch_size=exp_config.batch_size,
    )

    _, _, run_dir = train_and_evaluate(
        data_cfg=data_cfg,
        model_cfg=model_config,
        train_cfg=train_cfg,
        output_root=output_root,
        seed=seed,
    )

    roc_path = run_dir / "roc_curve.csv"
    roc_data = np.loadtxt(roc_path, delimiter=",", skiprows=1)
    fpr, tpr = roc_data[:, 0], roc_data[:, 1]
    return fpr, tpr


def run_all_anomaly_experiments(output_root: Path) -> None:
    """Execute anomaly experiments for the canonical three grid sizes."""
    # Shared model configuration; you can tune this further

    model_cfg = ModelConfig()

    sizes = [(16, 16), (32, 32), (64, 64)]

    roc_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for h, w in sizes:
        exp_cfg = AnomalyExperimentConfig(height=h, width=w)
        fpr, tpr = run_anomaly_experiment_for_size(
            model_config=model_cfg, exp_config=exp_cfg, output_root=output_root / f"anomaly_{h}x{w}"
        )
        roc_results[f"{h}x{w}"] = (fpr, tpr)

    plot_multiple_roc_curves(roc_results, save_path=output_root / "roc_curves_overview.png")
