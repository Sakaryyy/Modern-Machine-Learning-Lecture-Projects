"""Visualisations tailored to ablation study summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig, scientific_style


@dataclass(slots=True)
class AblationVisualizerConfig:
    """Configuration for :class:`AblationVisualizer`."""

    output_directory: Path
    style_config: PlotStyleConfig | None = None


class AblationVisualizer:
    """Create publication-ready plots for ablation studies."""

    def __init__(self, config: AblationVisualizerConfig) -> None:
        self._config = config
        self._config.output_directory.mkdir(parents=True, exist_ok=True)

    def save_metric_overview(self, summary: pd.DataFrame, metric: str) -> Path:
        """Plot the aggregated metric per ablated hyper-parameter."""

        figure_path = self._config.output_directory / f"ablation_{metric}.png"
        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(
                data=summary,
                x="parameter",
                y=metric,
                hue="value",
                ax=ax,
                palette="viridis",
            )
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel("Hyper-parameter")
            ax.set_title(f"Ablation impact on {metric}")
            ax.legend(title="Value", bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(figure_path)
            plt.close(fig)
        return figure_path

    def save_summary_table(self, summary: pd.DataFrame) -> Path:
        """Persist the ablation results as a CSV file."""

        path = self._config.output_directory / "ablation_summary.csv"
        summary.to_csv(path, index=False)
        return path
