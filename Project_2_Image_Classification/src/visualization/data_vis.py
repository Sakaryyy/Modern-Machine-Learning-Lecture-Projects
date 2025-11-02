"""High-level plotting helpers dedicated to dataset visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import PlotStyleManager
from ..utils.logging import get_logger

__all__ = ["DataVisualiser", "SampleGridConfig", "PixelSamplingConfig"]


@dataclass(frozen=True, slots=True)
class SampleGridConfig:
    """Configuration describing how sample image grids are generated."""

    rows: int = 4
    columns: int = 4
    random_seed: int = 11
    figsize: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class PixelSamplingConfig:
    """Configuration controlling histogram generation for pixel intensities."""

    bins: int = 40
    samples_per_split: int = 50_000
    random_seed: int = 12


class DataVisualiser:
    """Factory creating plots summarising dataset characteristics."""

    CHANNEL_NAMES = ("Red", "Green", "Blue")

    def __init__(
            self,
            style_manager: PlotStyleManager | None = None,
    ) -> None:
        """Initialise the visualiser.

        Parameters
        ----------
        style_manager:
            Optional plot style manager used to enforce a consistent look across
            all generated figures.  When omitted, a new instance with default
            configuration is created.
        """
        self._style_manager = style_manager or PlotStyleManager()
        self._logger = get_logger(self.__class__.__name__)

    def sample_grid(
            self,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            class_names: Sequence[str],
            config: SampleGridConfig | None = None,
    ) -> plt.Figure:
        """Return a figure showing randomly selected dataset samples.

        Parameters
        ----------
        images:
            Image tensor of shape ``(N, H, W, C)`` with floating point intensities
            in the range ``[0, 1]``.
        labels:
            Integer encoded labels matching the image tensor.
        class_names:
            Sequence translating numeric labels to human readable names.
        config:
            Optional configuration controlling the grid layout.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object ready to be persisted by the caller.
        """

        cfg = config or SampleGridConfig()
        self._style_manager.apply()

        if images.ndim != 4:
            msg = "Images must be four dimensional (N, H, W, C)."
            raise ValueError(msg)

        num_samples = images.shape[0]
        grid_size = cfg.rows * cfg.columns
        if num_samples < grid_size:
            msg = "Not enough images available to create the sample grid."
            raise ValueError(msg)

        rng = np.random.default_rng(cfg.random_seed)
        indices = rng.choice(num_samples, size=grid_size, replace=False)

        np_images = np.asarray(images)[indices]
        np_labels = np.asarray(labels)[indices]

        figsize = cfg.figsize or (
            self._style_manager.config.figure_size[0],
            self._style_manager.config.figure_size[0],
        )
        fig, axes = plt.subplots(cfg.rows, cfg.columns, figsize=figsize)
        axes_array = np.atleast_1d(np.array(axes, dtype=object))

        for ax, image, label in zip(axes_array.flatten(), np_images, np_labels):
            ax.imshow(np.clip(image, 0.0, 1.0))
            ax.set_title(class_names[int(label)], fontsize=self._style_manager.config.tick_size)
            ax.axis("off")

        fig.suptitle("Random CIFAR-10 samples", fontsize=self._style_manager.config.title_size)
        fig.tight_layout()
        self._logger.debug(
            "Generated sample grid with %d rows and %d columns.",
            cfg.rows,
            cfg.columns,
        )
        return fig

    def split_sizes(self, split_sizes: Mapping[str, int]) -> plt.Figure:
        """Create a bar plot visualising how many samples each split contains.

        Parameters
        ----------
        split_sizes:
            Mapping linking split names to the number of samples contained in the
            respective subset.

        Returns
        -------
        matplotlib.figure.Figure
            Rendered figure displaying the sample counts per dataset split.
        """

        self._style_manager.apply()
        data = pd.DataFrame(
            {"split": list(split_sizes.keys()), "num_samples": list(split_sizes.values())}
        )

        fig, ax = plt.subplots(figsize=self._style_manager.config.figure_size)
        sns.barplot(data=data, x="split", y="num_samples", ax=ax, palette=self._style_manager.config.palette)
        ax.set_title("Dataset split sizes")
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of samples")
        fig.tight_layout()
        self._logger.debug("Generated dataset split size visualisation.")
        return fig

    def label_distribution(
            self,
            label_counts: Mapping[str, Sequence[int]],
            class_names: Sequence[str],
    ) -> plt.Figure:
        """Plot class frequency histograms for every dataset split.

        Parameters
        ----------
        label_counts:
            Mapping assigning to every split the number of samples per class. The
            mapping may additionally contain an "overall" key with the combined
            counts across all splits.
        class_names:
            Sequence translating label indices into human readable class names.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing a subplot per split visualising the counts.
        """

        self._style_manager.apply()
        records = []
        for split_name, counts in label_counts.items():
            for class_index, count in enumerate(counts):
                records.append(
                    {
                        "split": split_name,
                        "class_name": class_names[class_index],
                        "count": int(count),
                    }
                )
        data = pd.DataFrame.from_records(records)

        splits = list(label_counts.keys())
        fig_height = max(4, 3 * len(splits))
        fig, axes = plt.subplots(len(splits), 1, figsize=(self._style_manager.config.figure_size[0], fig_height),
                                 sharex=True)
        if len(splits) == 1:
            axes = [axes]

        for ax, split_name in zip(axes, splits):
            split_df = data[data["split"] == split_name]
            sns.barplot(data=split_df, x="class_name", y="count", ax=ax)
            ax.set_title(f"Label distribution: {split_name.title()}")
            ax.set_xlabel("Class")
            ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        self._logger.debug("Generated label distribution plots for splits: %s", ", ".join(splits))
        return fig

    def pixel_intensity_distribution(
            self,
            images_by_split: Mapping[str, jnp.ndarray],
            config: PixelSamplingConfig | None = None,
    ) -> plt.Figure:
        """Visualise the pixel intensity distribution for every split.

        Parameters
        ----------
        images_by_split:
            Mapping that associates each split name with the corresponding image
            tensor of shape ``(N, H, W, C)``.
        config:
            Optional configuration defining histogram binning and sampling
            behaviour.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with one subplot per image channel illustrating the intensity
            distribution across splits.
        """

        cfg = config or PixelSamplingConfig()
        self._style_manager.apply()

        rng = np.random.default_rng(cfg.random_seed)
        records: list[dict[str, object]] = []

        for split_name, images in images_by_split.items():
            np_images = np.asarray(images)
            if np_images.ndim != 4:
                msg = "Images must be four dimensional (N, H, W, C)."
                raise ValueError(msg)

            flattened = np_images.reshape(-1, np_images.shape[-1])
            if cfg.samples_per_split < flattened.shape[0]:
                indices = rng.choice(flattened.shape[0], size=cfg.samples_per_split, replace=False)
                flattened = flattened[indices]

            channel_count = flattened.shape[-1]
            channel_names = self.CHANNEL_NAMES[:channel_count]
            for channel_index, channel_name in enumerate(channel_names):
                for value in flattened[:, channel_index]:
                    records.append(
                        {
                            "split": split_name,
                            "channel": channel_name,
                            "intensity": float(value),
                        }
                    )

        data = pd.DataFrame.from_records(records)
        if data.empty:
            msg = "No data available to plot pixel intensity distribution."
            raise ValueError(msg)

        channel_names = sorted(data["channel"].unique())
        fig, axes = plt.subplots(len(channel_names), 1,
                                 figsize=(self._style_manager.config.figure_size[0], 4 * len(channel_names)),
                                 sharex=True)
        if len(channel_names) == 1:
            axes = [axes]

        for ax, channel_name in zip(axes, channel_names):
            channel_df = data[data["channel"] == channel_name]
            sns.histplot(
                data=channel_df,
                x="intensity",
                hue="split",
                element="step",
                stat="density",
                common_norm=False,
                bins=cfg.bins,
                ax=ax,
            )
            ax.set_title(f"Pixel intensity distribution - {channel_name} channel")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Density")

        fig.tight_layout()
        self._logger.debug("Generated pixel intensity histograms for channels: %s", ", ".join(channel_names))
        return fig
