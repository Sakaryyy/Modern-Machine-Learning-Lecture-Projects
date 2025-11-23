"""High level plotting primitives shared by the dataset analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.style import (
    PlotStyleConfig,
    PlotStyler,
    place_legend_below,
)

__all__ = [
    "VisualizerConfig",
    "DataVisualizer",
]


@dataclass(slots=True)
class VisualizerConfig:
    """Configuration options for :class:`DataVisualizer`.

    Parameters
    ----------
    output_directory:
        Directory where all visualisations will be persisted.  The directory is
        created eagerly to guarantee that figures can be written without
        further checks.
    style_config:
        Optional plotting style configuration.  When omitted the project wide
        defaults are used.
    sample_grid_shape:
        ``(rows, cols)`` tuple describing the layout of the random image grid.
    """

    output_directory: Path
    style_config: PlotStyleConfig | None = None
    sample_grid_shape: tuple[int, int] = (4, 8)


class DataVisualizer:
    """Utility class that generates dataset plots."""

    def __init__(self, config: VisualizerConfig) -> None:
        self._config = config
        self._output_dir = config.output_directory
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._styler = PlotStyler(config.style_config)
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public plotting interface
    # ------------------------------------------------------------------
    def save_sample_grid(
            self,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            class_names: Sequence[str],
            *,
            filename: str = "sample_grid.png",
            seed: int | None = None,
    ) -> Path:
        """Persist a random grid of dataset examples.

        Parameters
        ----------
        images:
            Image batch of shape ``(N, H, W, C)``.
        labels:
            Associated labels with shape ``(N,)``.
        class_names:
            Iterable mapping each label index to its textual description.
        filename:
            Name of the file created within :attr:`output_directory`.
        seed:
            Optional seed controlling which images are selected.
        """

        num_images = images.shape[0]
        rows, cols = self._config.sample_grid_shape
        grid_size = rows * cols
        rng = np.random.default_rng(seed)
        indices = rng.choice(num_images, size=min(grid_size, num_images), replace=False)

        with self._styler.context():
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            axes = np.atleast_2d(axes)

            for ax, idx in zip(axes.flat, indices):
                image = np.asarray(images[idx])
                label_idx = int(labels[idx])
                title = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

                ax.imshow(image)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in axes.flat[indices.size:]:
                ax.axis("off")

            fig.suptitle("Random CIFAR-10 samples")
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved sample grid to %s", output_path)
        return output_path

    def save_split_overview(
            self,
            split_sizes: Mapping[str, int],
            *,
            filename: str = "split_sizes.png",
    ) -> Path:
        """Plot the number of samples contained in each dataset split."""

        labels = list(split_sizes.keys())
        counts = list(split_sizes.values())

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.barplot(x=labels, y=counts, ax=ax, palette="colorblind", hue="label")
            ax.set_ylabel("Number of samples")
            ax.set_xlabel("Split")
            ax.set_title("CIFAR-10 split sizes")
            for bar, count in zip(ax.patches, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}",
                        ha="center", va="bottom", fontsize=8)
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved split overview to %s", output_path)
        return output_path

    def save_label_distribution(
            self,
            label_counts: Mapping[str, int],
            *,
            title: str,
            filename: str,
    ) -> Path:
        """Plot the class distribution for a single dataset split."""

        labels = list(label_counts.keys())
        counts = np.array(list(label_counts.values()))
        order = np.argsort(labels)
        labels = [labels[i] for i in order]
        counts = counts[order]

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=labels, y=counts, ax=ax, palette="colorblind")
            ax.set_ylabel("Number of samples")
            ax.set_xlabel("Class")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved label distribution to %s", output_path)
        return output_path

    def save_pixel_statistics(
            self,
            pixel_values: np.ndarray,
            *,
            filename: str = "pixel_statistics.png",
            title: str = "Pixel intensity distribution",
    ) -> Path:
        """Create a histogram that visualises pixel intensities."""

        flattened = pixel_values.reshape(-1)

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(flattened, bins=50, kde=True, ax=ax, color="#4C72B0")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            ax.set_title(title)
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved pixel statistics to %s", output_path)
        return output_path

    def save_channel_statistics(
            self,
            per_channel_stats: Mapping[str, Sequence[float]],
            *,
            filename: str = "channel_statistics.png",
    ) -> Path:
        """Visualise basic statistics for each colour channel."""

        channels = list(per_channel_stats.keys())
        stats = np.asarray(list(per_channel_stats.values()))
        stat_names = ("min", "max", "mean", "std")

        with self._styler.context():
            fig, ax = plt.subplots(figsize=(6, 4))
            for idx, name in enumerate(stat_names):
                ax.plot(channels, stats[:, idx], marker="o", label=name)
            ax.set_xlabel("Channel")
            ax.set_ylabel("Value")
            ax.set_title("Per-channel summary statistics")
            place_legend_below(fig, ax)
            output_path = self._save_figure(fig, filename)

        self._logger.info("Saved channel statistics to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        """Persist ``figure`` in :attr:`output_directory` and close it."""

        output_path = self._output_dir / filename
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return output_path


def normalise_label_counts(
        labels: Iterable[int],
        class_names: Sequence[str] | None,
) -> dict[str, int]:
    """Return a mapping from class name to label count."""

    integer_labels = [int(label) for label in labels]
    if not class_names:
        unique_labels = sorted(set(integer_labels))
        class_names = [f"class_{label}" for label in unique_labels]

    counts = {name: 0 for name in class_names}
    for label in integer_labels:
        if 0 <= label < len(class_names):
            counts[class_names[label]] += 1
    return counts
