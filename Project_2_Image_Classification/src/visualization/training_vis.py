"""Visualisations describing the optimisation progress during training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig, PlotStyler

__all__ = [
    "TrainingVisualizerConfig",
    "TrainingVisualizer",
]


@dataclass(slots=True)
class TrainingVisualizerConfig:
    """Configuration options for :class:`TrainingVisualizer`.

    Parameters
    ----------
    output_directory:
        Directory where generated figures are stored.
    style_config:
        Optional plotting style overriding the project defaults.
    """

    output_directory: Path
    style_config: PlotStyleConfig | None = None


class TrainingVisualizer:
    """Create publication-ready plots summarising training dynamics."""

    def __init__(self, config: TrainingVisualizerConfig) -> None:
        self._config = config
        self._styler = PlotStyler(config.style_config)
        self._output_dir = config.output_directory
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(self.__class__.__name__)

    def save_learning_curves(
            self,
            history: pd.DataFrame,
            metrics: Sequence[str],
    ) -> Dict[str, Path]:
        """Persist learning curves for each metric listed in ``metrics``."""

        paths: Dict[str, Path] = {}
        for metric in metrics:
            paths[metric] = self.save_metric_curve(history, metric)
        return paths

    def save_metric_curve(
            self,
            history: pd.DataFrame,
            metric: str,
            *,
            filename: str | None = None,
    ) -> Path:
        """Save a line plot comparing training and validation ``metric``."""

        df = history.copy()
        if "epoch" not in df.columns:
            df.insert(0, "epoch", range(1, len(df) + 1))

        train_column = f"train_{metric}"
        val_column = f"validation_{metric}"
        filename = filename or f"{metric}_curve.png"

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(6, 4))
            if train_column in df:
                ax.plot(df["epoch"], df[train_column], marker="o", label="Train")
            if val_column in df:
                ax.plot(df["epoch"], df[val_column], marker="o", label="Validation")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} over epochs")
            if train_column in df or val_column in df:
                ax.legend(frameon=False)
            sns.despine()

            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved %s curve to %s", metric, output_path)
        return output_path

    def save_learning_rate_curve(
            self,
            history: pd.DataFrame,
            *,
            filename: str = "learning_rate_schedule.png",
    ) -> Path:
        """Persist a plot visualising the learning-rate schedule."""

        if "learning_rate" not in history:
            raise ValueError("'history' must contain a 'learning_rate' column to plot the schedule.")

        df = history[["epoch", "learning_rate"]].copy()

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["epoch"], df["learning_rate"], marker="o", color="#4C72B0")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning rate")
            ax.set_title("Learning-rate schedule")
            sns.despine()
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved learning-rate schedule to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self._output_dir / filename
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return output_path
