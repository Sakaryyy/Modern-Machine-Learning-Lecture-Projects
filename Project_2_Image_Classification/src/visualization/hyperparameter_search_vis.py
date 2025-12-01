"""Visualisations for hyper-parameter search summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Project_2_Image_Classification.src.visualization.style import (
    PlotStyleConfig,
    place_legend_below,
    scientific_style,
)


@dataclass(slots=True)
class HyperparameterSearchVisualizerConfig:
    """Configuration for the hyper-parameter search visualiser."""

    output_directory: Path
    figures_directory: Path | None = None
    tables_directory: Path | None = None
    style_config: PlotStyleConfig | None = None


class HyperparameterSearchVisualizer:
    """Create plots summarising the outcomes of grid searches."""

    def __init__(self, config: HyperparameterSearchVisualizerConfig) -> None:
        self._config = config
        self._base_dir = config.output_directory
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = config.figures_directory or (self._base_dir / "figures")
        self.tables_dir = config.tables_directory or (self._base_dir / "tables")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def save_ranked_results(self, summary: pd.DataFrame, metric: str) -> Path:
        """Persist the sorted search results to CSV."""

        ranked = summary.sort_values(metric, ascending=False).reset_index(drop=True)
        path = self.tables_dir / "hyperparameter_search_results.csv"
        ranked.to_csv(path, index=False)
        return path

    def save_metric_distribution(self, summary: pd.DataFrame, metric: str) -> Path:
        """Save a histogram visualising the dispersion of ``metric`` across runs."""

        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(summary[metric], bins=20, kde=True, ax=ax, color="#4C72B0")
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {metric.replace('_', ' ')} across search runs")
            figure_path = self.figures_dir / f"hyperparameter_{metric}_distribution.png"
            fig.tight_layout()
            fig.savefig(figure_path)
            plt.close(fig)
        return figure_path

    def save_numeric_pairplot(self, summary: pd.DataFrame, metric: str) -> Path:
        """Create a pairplot for all numeric features, highlighting ``metric`` relationships."""

        numeric_frame = summary.select_dtypes(include=[np.number])
        if metric not in numeric_frame.columns:
            numeric_frame = numeric_frame.assign(**{metric: summary[metric].to_numpy(dtype=float)})

        if numeric_frame.shape[1] < 2:
            raise ValueError("At least two numeric columns are required to create a pairplot.")

        with scientific_style(self._config.style_config):
            grid = sns.pairplot(numeric_frame, corner=True, diag_kind="hist")
            grid.fig.suptitle("Pairwise relationships between numeric hyper-parameters", fontsize=12)
            figure_path = self.figures_dir / "hyperparameter_pairplot.png"
            grid.fig.savefig(figure_path)
            plt.close(grid.fig)
        return figure_path

    def save_top_configurations(self, summary: pd.DataFrame, metric: str, top_k: int = 10) -> Path:
        """Plot the top ``top_k`` configurations ranked by ``metric``."""

        ranked = summary.sort_values(metric, ascending=False).head(top_k)
        ranked = ranked.copy()
        ranked["run"] = range(1, len(ranked) + 1)
        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=ranked, x="run", y=metric, ax=ax, palette="viridis")
            ax.set_xlabel("Rank")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Top {len(ranked)} configurations")
            figure_path = self.figures_dir / "hyperparameter_topk.png"
            fig.tight_layout()
            fig.savefig(figure_path)
            plt.close(fig)
        return figure_path

    def save_parameter_effects(self, summary: pd.DataFrame, metric: str, *, metric_label: str | None = None) -> Path:
        """Plot how ``metric`` varies with each numeric hyper-parameter on a single grid."""

        numeric_columns = [
            column
            for column in summary.columns
            if column != metric and pd.api.types.is_numeric_dtype(summary[column])
        ]

        usable_columns: list[str] = []
        column_data: list[pd.DataFrame] = []
        for column in numeric_columns:
            data = summary[[column, metric]].dropna()
            if data.empty or data[column].nunique() <= 1:
                continue
            usable_columns.append(column)
            column_data.append(data.sort_values(column))

        if not usable_columns:
            raise ValueError("No numeric hyper-parameters available to plot effects.")

        n_params = len(usable_columns)
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        figsize = (5.0 * ncols, 4.0 * nrows)

        with scientific_style(self._config.style_config):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=False)
            axes_iter = list(axes.flat)
            legend_ax: plt.Axes | None = None

            for ax, column, data in zip(axes_iter, usable_columns, column_data):
                line_label = "Metric trend" if legend_ax is None else None
                sns.lineplot(data=data, x=column, y=metric, marker="o", ax=ax, label=line_label)

                best_idx = data[metric].idxmax()
                best_row = data.loc[best_idx]
                scatter_label = "Best" if legend_ax is None else None
                ax.scatter([best_row[column]], [best_row[metric]], color="red", s=70, label=scatter_label)

                ax.set_xlabel(column.replace("_", " ").title())
                y_label = metric_label or metric.replace("_", " ").title()
                ax.set_ylabel(y_label)
                ax.set_title(f"{y_label} vs {column.replace('_', ' ').title()}")

                if metric.endswith("relative_to_best") or metric.endswith("normalized_vs_best"):
                    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)

                if legend_ax is None:
                    legend_ax = ax

            for ax in axes_iter[len(usable_columns):]:
                ax.axis("off")

            if legend_ax is not None:
                place_legend_below(fig, legend_ax)

            figure_path = self.figures_dir / f"hyperparameter_parameter_effects_{metric}.png"
            fig.tight_layout()
            fig.savefig(figure_path, bbox_inches="tight")
            plt.close(fig)

        return figure_path
