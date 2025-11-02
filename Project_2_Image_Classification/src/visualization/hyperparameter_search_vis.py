"""Visualisations for hyper-parameter search summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig, scientific_style


@dataclass(slots=True)
class HyperparameterSearchVisualizerConfig:
    """Configuration for the hyper-parameter search visualiser."""

    output_directory: Path
    style_config: PlotStyleConfig | None = None


class HyperparameterSearchVisualizer:
    """Create plots summarising the outcomes of grid searches."""

    def __init__(self, config: HyperparameterSearchVisualizerConfig) -> None:
        self._config = config
        self._config.output_directory.mkdir(parents=True, exist_ok=True)

    def save_metric_heatmap(self, summary: pd.DataFrame, metric: str, x: str, y: str) -> Path:
        """Save a heatmap showing metric variation across two parameters."""

        pivot = summary.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
        figure_path = self._config.output_directory / f"hyperparameter_{metric}_heatmap.png"
        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
            ax.set_title(f"{metric.replace('_', ' ').title()} across {x} and {y}")
            fig.tight_layout()
            fig.savefig(figure_path)
            plt.close(fig)
        return figure_path

    def save_ranked_results(self, summary: pd.DataFrame, metric: str) -> Path:
        """Persist the sorted search results to CSV."""

        ranked = summary.sort_values(metric, ascending=False).reset_index(drop=True)
        path = self._config.output_directory / "hyperparameter_search_results.csv"
        ranked.to_csv(path, index=False)
        return path
