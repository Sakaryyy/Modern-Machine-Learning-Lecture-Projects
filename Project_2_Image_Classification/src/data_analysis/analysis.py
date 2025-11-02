"""Comprehensive analysis pipeline for the CIFAR-10 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loading.data_load_and_save import CIFAR10DataManager, PreparedDataset
from ..utils.logging import get_logger
from ..visualization.data_vis import DataVisualiser, PixelSamplingConfig, SampleGridConfig
from ..visualization.style import PlotStyleConfig, PlotStyleManager

__all__ = ["AnalysisConfig", "CIFAR10DatasetAnalyser", "analyse_cifar10_dataset"]


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration describing how the analysis routine should behave.

    Parameters
    ----------
    data_dir:
        Location where the CIFAR-10 dataset should be stored.  If the processed
        files are not available the downloader persists them inside this
        directory.
    output_dir:
        Directory receiving generated reports.  When ``None`` a project-wide
        default under ``outputs/data_analysis`` is used.
    val_split:
        Fraction of the training images that should be moved to the validation
        set when downloading the dataset.
    random_seed:
        Seed governing stochastic components such as train/validation splitting
        and sampling for visualisations.
    sample_grid:
        Configuration controlling how the grid of random sample images is
        produced.
    pixel_sampling:
        Configuration that determines how many pixel intensities are considered
        when generating histograms.
    style:
        Global plotting style configuration reused across all figures.
    """

    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    output_dir: Path | None = None
    val_split: float = 0.1
    random_seed: int = 10
    sample_grid: SampleGridConfig = SampleGridConfig()
    pixel_sampling: PixelSamplingConfig = PixelSamplingConfig()
    style: PlotStyleConfig = PlotStyleConfig()


class CIFAR10DatasetAnalyser:
    """Perform descriptive statistics and visualisations for CIFAR-10."""

    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "data_analysis"

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Instantiate the analyser with the desired configuration.

        Parameters
        ----------
        config:
            Optional configuration object customising paths and visualisation
            behaviour.
        """
        self._config = config or AnalysisConfig()
        self._logger = get_logger(self.__class__.__name__)
        self._style_manager = PlotStyleManager(self._config.style)
        self._visualiser = DataVisualiser(self._style_manager)
        self._data_manager = CIFAR10DataManager(
            data_root=self._config.data_dir,
            val_split=self._config.val_split,
            seed=self._config.random_seed,
        )

        output_dir = self._config.output_dir or self.DEFAULT_OUTPUT_DIR
        self._output_dir = Path(output_dir).resolve()
        self._figures_dir = self._output_dir / "figures"
        self._tables_dir = self._output_dir / "tables"
        for directory in (self._output_dir, self._figures_dir, self._tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def run(self) -> PreparedDataset:
        """Execute the full data analysis workflow.

        Returns
        -------
        PreparedDataset
            Bundle containing the processed dataset splits and metadata.  The
            object is returned to allow downstream steps to reuse the cached
            data without repeating the download.
        """

        dataset = self._data_manager.prepare_data()
        self._logger.info("Prepared CIFAR-10 dataset for analysis.")

        statistics = self._compute_statistics(dataset)
        self._save_statistics(statistics)
        self._generate_visualisations(dataset, statistics)

        self._logger.info("Completed data analysis. Results stored in %s", self._output_dir)
        return dataset

    def _compute_statistics(self, dataset: PreparedDataset) -> Dict[str, object]:
        """Compute summary statistics required for analysis and visualisation.

        Parameters
        ----------
        dataset:
            Prepared dataset bundle returned by :class:`CIFAR10DataManager`.

        Returns
        -------
        dict
            Dictionary containing numeric summaries of all dataset splits as well
            as the label distribution.
        """

        class_names: Iterable[str] = dataset.metadata.get("class_names", [])
        class_names = tuple(class_names)
        if not class_names:
            inferred_classes = self._infer_num_classes(dataset)
            class_names = tuple(str(index) for index in range(inferred_classes))
            self._logger.warning("Class names missing in metadata. Falling back to indices.")

        num_classes = len(class_names)
        split_statistics: Dict[str, Dict[str, object]] = {}
        label_counts: Dict[str, list[int]] = {}
        split_sizes: Dict[str, int] = {}
        overall_counts = np.zeros(num_classes, dtype=np.int64)
        pixel_range_min = np.inf
        pixel_range_max = -np.inf

        for split_name, split in dataset.splits.items():
            np_images = np.asarray(split.images)
            np_labels = np.asarray(split.labels)

            if np_images.ndim != 4:
                msg = f"Split '{split_name}' images do not have expected shape (N, H, W, C)."
                raise ValueError(msg)

            num_samples = int(np_images.shape[0])
            height, width, channels = (int(dim) for dim in np_images.shape[1:])
            dtype = str(np_images.dtype)
            pixel_min = float(np_images.min())
            pixel_max = float(np_images.max())
            overall_mean = float(np_images.mean())
            overall_std = float(np_images.std())
            channel_means = np_images.mean(axis=(0, 1, 2)).tolist()
            channel_stds = np_images.std(axis=(0, 1, 2)).tolist()

            split_statistics[split_name] = {
                "num_samples": num_samples,
                "height": height,
                "width": width,
                "channels": channels,
                "dtype": dtype,
                "pixel_min": pixel_min,
                "pixel_max": pixel_max,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "channel_mean": channel_means,
                "channel_std": channel_stds,
            }

            counts = np.bincount(np_labels, minlength=num_classes)
            label_counts[split_name] = counts.astype(int).tolist()
            overall_counts += counts
            split_sizes[split_name] = num_samples

            pixel_range_min = min(pixel_range_min, pixel_min)
            pixel_range_max = max(pixel_range_max, pixel_max)

        label_counts["overall"] = overall_counts.astype(int).tolist()

        summary: Dict[str, object] = {
            "class_names": list(class_names),
            "num_classes": num_classes,
            "split_statistics": split_statistics,
            "label_distribution": label_counts,
            "split_sizes": split_sizes,
            "pixel_value_range": {"min": float(pixel_range_min), "max": float(pixel_range_max)},
        }
        self._logger.debug(
            "Computed dataset statistics for splits: %s", ", ".join(split_statistics.keys())
        )
        return summary

    def _save_statistics(self, statistics: Mapping[str, object]) -> None:
        """Persist computed summary statistics as CSV and JSON files.

        Parameters
        ----------
        statistics:
            Dictionary returned by :meth:`_compute_statistics` holding all
            pre-computed measures that should be written to disk.
        """

        split_stats = statistics["split_statistics"]
        class_names = statistics["class_names"]
        label_distribution = statistics["label_distribution"]

        table_rows = []
        for split_name, info in split_stats.items():
            row = {
                "split": split_name,
                "num_samples": info["num_samples"],
                "height": info["height"],
                "width": info["width"],
                "channels": info["channels"],
                "dtype": info["dtype"],
                "pixel_min": info["pixel_min"],
                "pixel_max": info["pixel_max"],
                "overall_mean": info["overall_mean"],
                "overall_std": info["overall_std"],
            }

            for index, value in enumerate(info["channel_mean"], start=1):
                row[f"channel_{index}_mean"] = value
            for index, value in enumerate(info["channel_std"], start=1):
                row[f"channel_{index}_std"] = value
            table_rows.append(row)

        summary_df = pd.DataFrame(table_rows).set_index("split")
        summary_path = self._tables_dir / "split_summary.csv"
        summary_df.to_csv(summary_path)
        self._logger.info("Saved split summary table to %s", summary_path)

        label_df = pd.DataFrame(label_distribution, index=class_names)
        label_path = self._tables_dir / "label_distribution.csv"
        label_df.to_csv(label_path)
        self._logger.info("Saved label distribution table to %s", label_path)

        json_path = self._output_dir / "dataset_statistics.json"
        json_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
        self._logger.info("Persisted dataset statistics JSON to %s", json_path)

    def _generate_visualisations(
            self,
            dataset: PreparedDataset,
            statistics: Mapping[str, object],
    ) -> None:
        """Create and persist the figures accompanying the analysis.

        Parameters
        ----------
        dataset:
            Dataset bundle providing access to all processed splits.
        statistics:
            Statistics dictionary previously created by :meth:`_compute_statistics`.
        """

        class_names = statistics["class_names"]
        split_sizes = statistics["split_sizes"]
        label_distribution = statistics["label_distribution"]

        train_split = dataset.splits.get("train")
        if train_split is None:
            msg = "Training split missing from dataset."
            raise KeyError(msg)

        sample_figure = self._visualiser.sample_grid(
            images=train_split.images,
            labels=train_split.labels,
            class_names=class_names,
            config=self._config.sample_grid,
        )
        self._save_figure("cifar10_sample_grid.png", sample_figure)

        split_sizes_figure = self._visualiser.split_sizes(split_sizes)
        self._save_figure("cifar10_split_sizes.png", split_sizes_figure)

        label_figure = self._visualiser.label_distribution(label_distribution, class_names)
        self._save_figure("cifar10_label_distribution.png", label_figure)

        images_by_split = {name: split.images for name, split in dataset.splits.items()}
        pixel_figure = self._visualiser.pixel_intensity_distribution(
            images_by_split=images_by_split,
            config=self._config.pixel_sampling,
        )
        self._save_figure("cifar10_pixel_intensity.png", pixel_figure)

    def _save_figure(self, filename: str, figure: plt.Figure) -> None:
        """Save a plot.

        Parameters
        ----------
        filename:
            Name of the file that should be created under the figures directory.
        figure:
            Matplotlib figure instance to persist.
        """

        path = self._figures_dir / filename
        try:
            figure.savefig(path, bbox_inches="tight")
            self._logger.info("Saved figure to %s", path)
        except OSError as exc:
            self._logger.error("Failed to save figure %s: %s", path, exc)
            raise
        finally:
            plt.close(figure)

    def _infer_num_classes(self, dataset: PreparedDataset) -> int:
        """Infer the number of classes from the dataset labels.

        Parameters
        ----------
        dataset:
            Dataset bundle which potentially contains label information for all
            splits.

        Returns
        -------
        int
            Inferred number of distinct labels present in the dataset.
        """

        max_label = -1
        for split in dataset.splits.values():
            labels = np.asarray(split.labels)
            if labels.size == 0:
                continue
            max_label = max(max_label, int(labels.max()))
        if max_label < 0:
            msg = "Unable to infer the number of classes from the dataset."
            raise ValueError(msg)
        return max_label + 1


def analyse_cifar10_dataset(config: AnalysisConfig | None = None) -> PreparedDataset:
    """Convenience wrapper executing the dataset analysis workflow.

    Parameters
    ----------
    config:
        Optional analysis configuration.  When omitted the defaults defined in
        :class:`AnalysisConfig` are used.

    Returns
    -------
    PreparedDataset
        Dataset returned by :meth:`CIFAR10DatasetAnalyser.run` to facilitate
        chaining of subsequent experiments.
    """

    analyser = CIFAR10DatasetAnalyser(config)
    return analyser.run()
