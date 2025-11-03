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
    figures_directory: Path | None = None
    tables_directory: Path | None = None
    style_config: PlotStyleConfig | None = None


class ClassificationVisualizer:
    """Generate plots for classification experiments."""

    def __init__(self, config: ClassificationVisualizerConfig) -> None:
        self._config = config
        self._base_dir = config.output_directory
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = config.figures_directory or (self._base_dir / "figures")
        self._tables_dir = config.tables_directory or (self._base_dir / "tables")
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._tables_dir.mkdir(parents=True, exist_ok=True)
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

        figure_path = self._figures_dir / "confusion_matrix.png"
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
        table_path = self._tables_dir / "classification_metrics.csv"
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
            rng: np.random.Generator | None = None,
    ) -> Path:
        """Save a grid of predictions with true labels for qualitative inspection."""

        num_available = images.shape[0]
        if num_available == 0:
            raise ValueError("At least one image is required to create a prediction gallery.")

        num_samples = min(max_images, num_available)
        indices = np.arange(num_available)
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
        indices = indices[:num_samples]

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
                image = images[indices[index]]
                ax.imshow(np.clip(image, 0.0, 1.0))
                pred_idx = int(predictions[indices[index]])
                true_idx = int(labels[indices[index]])
                pred_name = class_names[pred_idx]
                true_name = class_names[true_idx]
                if probabilities is not None:
                    confidence = float(probabilities[indices[index], pred_idx])
                    subtitle = f"Pred: {pred_name}\nTrue: {true_name}\nConf: {confidence:.2f}"
                else:
                    subtitle = f"Pred: {pred_name}\nTrue: {true_name}"
                ax.set_title(subtitle, fontsize=8)

            fig.suptitle("Model predictions on evaluation samples", fontsize=12)
            fig.tight_layout()
            output_path = self._figures_dir / filename
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
            output_path = self._figures_dir / filename
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
            output_path = self._figures_dir / filename
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved confidence histogram to %s", output_path)
        return output_path

    def save_activation_overview(
            self,
            image: np.ndarray,
            conv_activations: Sequence[tuple[str, np.ndarray]],
            *,
            label_name: str,
            prediction_name: str,
            confidence: float | None = None,
            filename: str = "activation_overview.png",
    ) -> Path:
        """Render the original image alongside aggregated convolutional activations."""

        columns = max(2, 1 + len(conv_activations))
        with scientific_style(self._config.style_config):
            fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
            axes = np.atleast_1d(axes)

            axes[0].imshow(np.clip(image, 0.0, 1.0))
            subtitle = f"True: {label_name}\nPred: {prediction_name}"
            if confidence is not None:
                subtitle += f"\nConf: {confidence:.2f}"
            axes[0].set_title(subtitle, fontsize=10)
            axes[0].axis("off")

            for axis_index, (layer_name, activation) in enumerate(conv_activations, start=1):
                aggregated = activation.mean(axis=-1)
                ax = axes[axis_index]
                ax.imshow(aggregated, cmap="magma")
                ax.set_title(layer_name.replace("_", " "), fontsize=9)
                ax.axis("off")

            for axis_index in range(len(conv_activations) + 1, columns):
                axes[axis_index].axis("off")

            fig.suptitle("Activation overview", fontsize=12)
            fig.tight_layout()
            output_path = self._figures_dir / filename
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved activation overview to %s", output_path)
        return output_path

    def save_feature_map_grid(
            self,
            layer_name: str,
            activation: np.ndarray,
            *,
            max_maps: int = 6,
            filename: str | None = None,
    ) -> Path:
        """Visualise the most informative feature maps of a convolutional block."""

        if activation.ndim != 3:
            raise ValueError("Activation must be a 3D tensor (height, width, channels) to visualise feature maps.")
        num_channels = activation.shape[-1]
        if num_channels == 0:
            raise ValueError("Activation tensor contains no channels to visualise.")

        maps_to_show = min(max_maps, num_channels)
        flattened = activation.reshape(-1, num_channels)
        importance = flattened.var(axis=0)
        top_indices = np.argsort(importance)[::-1][:maps_to_show]

        cols = int(np.ceil(np.sqrt(maps_to_show)))
        rows = int(np.ceil(maps_to_show / cols))

        with scientific_style(self._config.style_config):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
            axes = np.atleast_2d(axes)
            for position in range(rows * cols):
                row, col = divmod(position, cols)
                ax = axes[row, col]
                if position >= maps_to_show:
                    ax.axis("off")
                    continue
                channel_index = top_indices[position]
                ax.imshow(activation[:, :, channel_index], cmap="magma")
                ax.set_title(f"Ch {channel_index}", fontsize=8)
                ax.axis("off")

            fig.suptitle(layer_name.replace("_", " "), fontsize=12)
            fig.tight_layout()
            safe_layer = layer_name.replace("/", "_")
            output_name = filename or f"{safe_layer}_featuremaps.png"
            output_path = self._figures_dir / output_name
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved feature map grid for %s to %s", layer_name, output_path)
        return output_path

    def save_dense_activation_profile(
            self,
            layer_name: str,
            activation: np.ndarray,
            *,
            top_k: int = 20,
            filename: str | None = None,
    ) -> Path:
        """Plot the strongest dense-layer activations for a given sample."""

        if activation.ndim != 1:
            raise ValueError("Dense activations must be a 1D tensor for visualisation.")

        top_k = min(top_k, activation.shape[0])
        if top_k <= 0:
            raise ValueError("'top_k' must be positive to visualise dense activations.")

        indices = np.argsort(np.abs(activation))[::-1][:top_k]
        values = activation[indices]

        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(max(6, top_k * 0.3), 3.5))
            sns.barplot(x=np.arange(top_k), y=values, ax=ax, palette="viridis")
            ax.set_xlabel("Activation rank")
            ax.set_ylabel("Activation value")
            ax.set_title(layer_name.replace("_", " "))
            ax.set_xticks(np.arange(top_k))
            ax.set_xticklabels([str(idx) for idx in indices], rotation=45, ha="right")
            fig.tight_layout()
            safe_layer = layer_name.replace("/", "_")
            output_name = filename or f"{safe_layer}_dense.png"
            output_path = self._figures_dir / output_name
            fig.savefig(output_path)
            plt.close(fig)

        self._logger.info("Saved dense activation profile for %s to %s", layer_name, output_path)
        return output_path
