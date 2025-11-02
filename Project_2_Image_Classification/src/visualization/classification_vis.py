"""Visualisation helpers for model evaluation and classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig, scientific_style


@dataclass(slots=True)
class ClassificationVisualizerConfig:
    """Configuration for :class:`ClassificationVisualizer`."""

    output_directory: Path
    style_config: PlotStyleConfig | None = None


class ClassificationVisualizer:
    """Generate plots for classification experiments."""

    def __init__(self, config: ClassificationVisualizerConfig) -> None:
        self._config = config
        self._config.output_directory.mkdir(parents=True, exist_ok=True)

    def save_confusion_matrix(
            self,
            confusion_matrix: np.ndarray,
            class_names: Sequence[str],
    ) -> Path:
        """Render and persist a confusion matrix heatmap."""

        if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
            msg = "Confusion matrix must be square."
            raise ValueError(msg)

        if len(class_names) != confusion_matrix.shape[0]:
            raise ValueError("Number of class names must match confusion matrix size.")

        figure_path = self._config.output_directory / "confusion_matrix.png"
        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt=".0f",
                cmap="viridis",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            fig.savefig(figure_path)
            plt.close(fig)
        return figure_path

    def save_metrics_table(self, metrics: Mapping[str, float]) -> Path:
        """Persist metrics as a CSV table for downstream analysis."""

        frame = pd.DataFrame([{name: metrics[name] for name in sorted(metrics)}])
        table_path = self._config.output_directory / "classification_metrics.csv"
        frame.to_csv(table_path, index=False)
        return table_path
