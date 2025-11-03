"""Visualisation helpers for model evaluation and classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig, PlotStyler, scientific_style


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
        self._styler = PlotStyler(config.style_config)
        self._logger = get_logger(self.__class__.__name__)

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

    def save_prediction_gallery(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            predictions: np.ndarray,
            class_names: Sequence[str],
            probabilities: np.ndarray | None = None,
            *,
            max_images: int = 25,
            filename: str = "prediction_gallery.png",
    ) -> Path:
        """Save a grid of predictions with true labels for qualitative inspection."""

        num_samples = min(max_images, images.shape[0])
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        with scientific_style(self._config.style_config):
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            axes = np.atleast_2d(axes)
            for index in range(grid_size * grid_size):
                row, col = divmod(index, grid_size)
                ax = axes[row, col]
                ax.axis("off")
                if index >= num_samples:
                    continue
                image = images[index]
                ax.imshow(np.clip(image, 0.0, 1.0))
                pred_idx = int(predictions[index])
                true_idx = int(labels[index])
                pred_name = class_names[pred_idx]
                true_name = class_names[true_idx]
                if probabilities is not None:
                    confidence = float(probabilities[index, pred_idx])
                    subtitle = f"Pred: {pred_name}\nTrue: {true_name}\nConf: {confidence:.2f}"
                else:
                    subtitle = f"Pred: {pred_name}\nTrue: {true_name}"
                ax.set_title(subtitle, fontsize=8)

            fig.suptitle("Model predictions on evaluation samples", fontsize=12)
            fig.tight_layout()
            output_path = self._config.output_directory / filename
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved qualitative prediction gallery to %s", output_path)
        return output_path

    def save_per_class_accuracy(
            self,
            labels: np.ndarray,
            predictions: np.ndarray,
            class_names: Sequence[str],
            *,
            filename: str = "per_class_accuracy.png",
    ) -> Path:
        """Save a bar chart visualising per-class accuracy statistics."""

        frame = pd.DataFrame({
            "true": labels.astype(int),
            "correct": (labels == predictions).astype(int),
        })
        grouped = frame.groupby("true")["correct"].agg(["mean", "count"]).reset_index()
        grouped["class_name"] = grouped["true"].apply(lambda idx: class_names[int(idx)])

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=grouped, x="class_name", y="mean", ax=ax, palette="viridis")
            ax.set_xlabel("Class")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.set_title("Per-class accuracy")
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            output_path = self._config.output_directory / filename
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved per-class accuracy breakdown to %s", output_path)
        return output_path

    def save_confidence_histogram(
            self,
            probabilities: np.ndarray,
            labels: np.ndarray,
            predictions: np.ndarray,
            *,
            filename: str = "confidence_histogram.png",
    ) -> Path:
        """Store a histogram contrasting confidences for correct and incorrect predictions."""

        confidences = probabilities[np.arange(probabilities.shape[0]), predictions.astype(int)]
        correctness = predictions == labels
        frame = pd.DataFrame({
            "confidence": confidences,
            "correct": np.where(correctness, "Correct", "Incorrect"),
        })

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=frame, x="confidence", hue="correct", bins=20, ax=ax, kde=True, stat="density")
            ax.set_xlabel("Predicted class confidence")
            ax.set_ylabel("Density")
            ax.set_title("Confidence distribution across predictions")
            output_path = self._config.output_directory / filename
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved confidence histogram to %s", output_path)
        return output_path
