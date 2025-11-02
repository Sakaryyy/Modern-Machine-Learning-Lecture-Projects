"""Comprehensive descriptive analysis for the CIFAR-10 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from ..data_loading.data_load_and_save import CIFAR10DataManager, DatasetSplit, PreparedDataset
from ..utils.logging import get_logger
from ..visualization.data_vis import DataVisualizer, VisualizerConfig, normalise_label_counts
from ..visualization.style import PlotStyleConfig

__all__ = [
    "AnalysisConfig",
    "CIFAR10DatasetAnalyzer",
]


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration describing how the dataset analysis should be executed.

    Parameters
    ----------
    data_root:
        Directory containing the raw and processed dataset artefacts.
    output_dir:
        Directory where all analysis artefacts will be stored.  The directory is
        created automatically when the analysis starts.
    val_split:
        Fraction of training data held out for validation when the dataset needs
        to be prepared from scratch.
    seed:
        Random seed forwarded to :class:`CIFAR10DataManager` to ensure
        reproducibility of the train/validation split.
    sample_seed:
        Random seed used for selecting random samples when generating the image
        grid visualisation.
    style_config:
        Optional plotting style configuration overriding the global defaults.
    """

    data_root: Path
    output_dir: Path
    val_split: float = 0.1
    seed: int = 10
    sample_seed: int = 1234
    style_config: PlotStyleConfig | None = None


class CIFAR10DatasetAnalyzer:
    """Perform data quality checks and produce descriptive visualisations."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._logger = get_logger(self.__class__.__name__)

        figures_dir = self._config.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        self._visualizer = DataVisualizer(
            VisualizerConfig(
                output_directory=figures_dir,
                style_config=self._config.style_config,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, dataset: PreparedDataset | None = None) -> Dict[str, Any]:
        """Execute the full data analysis pipeline.

        Parameters
        ----------
        dataset:
            Optional pre-loaded dataset.  When ``None`` the analyzer loads the
            dataset from disk using :class:`CIFAR10DataManager`.
        """

        self._logger.info("Starting CIFAR-10 dataset analysis.")
        if dataset is None:
            dataset = self._prepare_dataset()
        else:
            self._logger.debug("Dataset provided externally; skipping loading step.")

        summary = self._summarise_dataset(dataset)
        self._save_summary(summary)
        self._generate_visualisations(dataset, summary)

        self._logger.info("Completed dataset analysis. Artefacts saved to %s", self._config.output_dir)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_dataset(self) -> PreparedDataset:
        """Load the dataset from disk, downloading it when necessary."""

        data_manager = CIFAR10DataManager(
            data_root=self._config.data_root,
            val_split=self._config.val_split,
            seed=self._config.seed,
        )
        dataset = data_manager.prepare_data()
        self._logger.info("Dataset ready with splits: %s", {k: v.images.shape[0] for k, v in dataset.splits.items()})
        return dataset

    def _summarise_dataset(self, dataset: PreparedDataset) -> Dict[str, Any]:
        """Compute descriptive statistics for each split and overall."""

        class_names = self._resolve_class_names(dataset)
        summary: Dict[str, Any] = {
            "class_names": class_names,
            "splits": {},
        }

        accumulated_images = []
        accumulated_labels = []

        for split_name, split in dataset.splits.items():
            split_stats = self._analyse_split(split, class_names)
            summary["splits"][split_name] = split_stats
            accumulated_images.append(np.asarray(split.images))
            accumulated_labels.append(np.asarray(split.labels))

        all_images = np.concatenate(accumulated_images, axis=0)
        all_labels = np.concatenate(accumulated_labels, axis=0)
        summary["overall"] = self._create_statistics(all_images, all_labels, class_names)

        return summary

    def _analyse_split(self, split: DatasetSplit, class_names: Sequence[str]) -> Dict[str, Any]:
        """Return statistics for a single dataset split."""

        images_np = np.asarray(split.images)
        labels_np = np.asarray(split.labels)

        split_stats = self._create_statistics(images_np, labels_np, class_names)
        split_stats["num_samples"] = int(images_np.shape[0])
        split_stats["image_shape"] = list(images_np.shape[1:])
        return split_stats

    def _create_statistics(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            class_names: Sequence[str],
    ) -> Dict[str, Any]:
        """Assemble descriptive statistics for the provided arrays."""

        flattened = images.reshape(images.shape[0], -1)
        per_channel = self._channel_statistics(images)

        stats: Dict[str, Any] = {
            "label_distribution": normalise_label_counts(labels, class_names),
            "pixel_statistics": {
                "min": float(np.min(flattened)),
                "max": float(np.max(flattened)),
                "mean": float(np.mean(flattened)),
                "std": float(np.std(flattened)),
            },
            "channel_statistics": per_channel,
        }
        return stats

    def _channel_statistics(self, images: np.ndarray) -> Dict[str, Sequence[float]]:
        """Compute per-channel summary statistics."""

        channel_names = ("red", "green", "blue")
        if images.ndim != 4 or images.shape[-1] != len(channel_names):
            channel_names = tuple(f"channel_{idx}" for idx in range(images.shape[-1]))

        stats: Dict[str, Sequence[float]] = {}
        for idx, name in enumerate(channel_names):
            channel = images[..., idx].reshape(-1)
            stats[name] = (
                float(np.min(channel)),
                float(np.max(channel)),
                float(np.mean(channel)),
                float(np.std(channel)),
            )
        return stats

    def _generate_visualisations(self, dataset: PreparedDataset, summary: Mapping[str, Any]) -> None:
        """Create and persist all figures required for the analysis report."""

        class_names = summary["class_names"]
        split_sizes = {name: int(split.images.shape[0]) for name, split in dataset.splits.items()}
        self._visualizer.save_split_overview(split_sizes)

        # Random sample grid from training split (or first available split)
        if dataset.splits:
            first_split_name = next(iter(dataset.splits))
            first_split = dataset.splits[first_split_name]
            self._visualizer.save_sample_grid(
                first_split.images,
                first_split.labels,
                class_names,
                seed=self._config.sample_seed,
            )

        for split_name, split_stats in summary["splits"].items():
            self._visualizer.save_label_distribution(
                split_stats["label_distribution"],
                title=f"{split_name.title()} label distribution",
                filename=f"label_distribution_{split_name}.png",
            )

        self._visualizer.save_label_distribution(
            summary["overall"]["label_distribution"],
            title="Overall label distribution",
            filename="label_distribution_overall.png",
        )

        all_images = np.concatenate([np.asarray(split.images) for split in dataset.splits.values()], axis=0)
        self._visualizer.save_pixel_statistics(all_images, filename="pixel_intensity_overall.png")
        self._visualizer.save_channel_statistics(
            summary["overall"]["channel_statistics"],
            filename="channel_statistics_overall.png",
        )

    def _save_summary(self, summary: Mapping[str, Any]) -> None:
        """Persist the computed summary statistics to disk."""

        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self._config.output_dir / "dataset_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        self._logger.info("Saved summary statistics to %s", summary_path)

    @staticmethod
    def _resolve_class_names(dataset: PreparedDataset) -> Sequence[str]:
        """Return class names stored in the dataset metadata."""

        class_names = dataset.metadata.get("class_names")
        if not class_names:
            first_split = next(iter(dataset.splits.values()))
            num_classes = int(jnp.max(first_split.labels).item()) + 1
            return [f"class_{idx}" for idx in range(num_classes)]
        return list(class_names)
