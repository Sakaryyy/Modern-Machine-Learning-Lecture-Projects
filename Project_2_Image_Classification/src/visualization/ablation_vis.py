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
    figures_directory: Path | None = None
    tables_directory: Path | None = None
    style_config: PlotStyleConfig | None = None


class AblationVisualizer:
    """Create publication-ready plots for ablation studies."""

    def __init__(self, config: AblationVisualizerConfig) -> None:
        self._config = config
        self._base_dir = config.output_directory
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = config.figures_directory or (self._base_dir / "figures")
        self.tables_dir = config.tables_directory or (self._base_dir / "tables")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_figsize(n_params: int, n_values: int) -> tuple[float, float]:
        """
        Pick a sensible figsize based on number of parameters on x and legend items.

        This avoids tiny 7x4 figures with 20 legend entries.
        """
        width = max(6.5, min(16.0, 3.0 + 0.7 * n_params))
        # extra height if legend gets large
        height = max(4.0, min(10.0, 4.0 + 0.15 * n_values))
        return width, height

    @staticmethod
    def _place_legend_below(fig: plt.Figure, ax: plt.Axes, title: str = "Value") -> None:
        """
        Put the legend below the plot and auto-select the number of columns.

        This is much more robust when there are many runs / values.
        """
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        n_items = len(labels)
        # heuristics: try to keep labels readable
        if n_items <= 6:
            ncol = 1
        elif n_items <= 12:
            ncol = 2
        elif n_items <= 20:
            ncol = 3
        else:
            # at this point there are *a lot* of runs; spreading them horizontally
            # is the only way to prevent overflow
            ncol = 4

        # legend below, centered
        fig.legend(
            handles,
            [str(l) for l in labels],  # ensure full string representation
            title=title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.025),
            ncol=ncol,
            frameon=True,
        )
        # leave room at bottom for legend
        fig.subplots_adjust(bottom=0.18)

    @staticmethod
    def _rotate_xticks(ax: plt.Axes, rotation: float = 30.0) -> None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
            tick.set_ha("right")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def save_metric_overview(self, summary: pd.DataFrame, metric: str) -> Path:
        """Plot the aggregated metric per ablated hyper-parameter."""

        figure_path = self.figures_dir / f"ablation_{metric}.png"

        # make sure we have what we need
        if "parameter" not in summary.columns or "value" not in summary.columns:
            raise ValueError("summary must contain 'parameter' and 'value' columns.")

        n_params = summary["parameter"].nunique()
        n_values = summary["value"].nunique()

        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(figsize=self._auto_figsize(n_params, n_values))

            plot_data = summary.copy()
            plot_data["value"] = plot_data["value"].astype(str)

            sns.barplot(
                data=plot_data,
                x="parameter",
                y=metric,
                hue="value",
                ax=ax,
                palette="viridis",
            )

            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel("Hyper-parameter")
            ax.set_title(f"Ablation impact on {metric}")

            # long parameter names -> rotate
            if n_params > 4:
                self._rotate_xticks(ax, rotation=30)

            self._place_legend_below(fig, ax, title="Value")

            fig.tight_layout()
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return figure_path

    def save_delta_overview(self, summary: pd.DataFrame, metric: str) -> Path:
        """Visualise the performance delta relative to the baseline configuration."""

        if "delta_vs_baseline" not in summary.columns:
            raise ValueError("Summary must contain a 'delta_vs_baseline' column.")

        plot_data = summary[summary["parameter"] != "baseline"].copy()
        if plot_data.empty:
            raise ValueError("No ablation entries beyond the baseline configuration were provided.")

        plot_data["value"] = plot_data["value"].astype(str)
        figure_path = self.figures_dir / f"ablation_{metric}_delta.png"

        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(
                figsize=self._auto_figsize(plot_data["parameter"].nunique(), plot_data["value"].nunique())
            )
            sns.barplot(
                data=plot_data,
                x="parameter",
                y="delta_vs_baseline",
                hue="value",
                palette="coolwarm",
                ax=ax,
            )
            ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
            ax.set_ylabel(f"Δ {metric.replace('_', ' ')} vs baseline")
            ax.set_xlabel("Hyper-parameter")
            ax.set_title(f"Change in {metric.replace('_', ' ')} relative to baseline")
            if plot_data["parameter"].nunique() > 4:
                self._rotate_xticks(ax)
            self._place_legend_below(fig, ax, title="Value")
            fig.tight_layout()
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return figure_path

    def save_normalized_overview(self, summary: pd.DataFrame, metric: str) -> Path:
        """Plot the metric normalized against the baseline for easier comparison."""

        if "normalized_vs_baseline" not in summary.columns:
            raise ValueError("Summary must include a 'normalized_vs_baseline' column.")

        normalized = summary[summary["parameter"] != "baseline"].copy()
        if normalized.empty:
            raise ValueError("No ablation entries beyond the baseline configuration were provided.")

        normalized["value"] = normalized["value"].astype(str)
        figure_path = self.figures_dir / f"ablation_{metric}_normalized.png"

        with scientific_style(self._config.style_config):
            fig, ax = plt.subplots(
                figsize=self._auto_figsize(normalized["parameter"].nunique(), normalized["value"].nunique())
            )
            sns.barplot(
                data=normalized,
                x="parameter",
                y="normalized_vs_baseline",
                hue="value",
                palette="crest",
                ax=ax,
            )
            ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_ylabel(f"{metric.replace('_', ' ')} / baseline")
            ax.set_xlabel("Hyper-parameter")
            ax.set_title(f"Normalised {metric.replace('_', ' ')} relative to baseline")
            if normalized["parameter"].nunique() > 4:
                self._rotate_xticks(ax)
            self._place_legend_below(fig, ax, title="Value")
            fig.tight_layout()
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return figure_path

    def save_summary_table(self, summary: pd.DataFrame) -> Path:
        """Persist the ablation results as a CSV file."""

        path = self.tables_dir / "ablation_summary.csv"
        summary.to_csv(path, index=False)
        return path

    def save_parameter_boxplots(self, raw_results: pd.DataFrame, metric: str,
                                baseline_value: float | None = None) -> Path:
        """Generate a grid of boxplots showing ``metric`` per parameter value."""

        if raw_results.empty:
            raise ValueError("Raw ablation results are required to visualise parameter distributions.")

        parameters = list(raw_results["parameter"].unique())
        n_params = len(parameters)
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols

        figsize = (4.5 * ncols, 4.5 * nrows)
        with scientific_style(self._config.style_config):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)
            axes_iter = list(axes.flat)
            legend_anchor = None

            for ax, parameter in zip(axes_iter, parameters):
                group = raw_results[raw_results["parameter"] == parameter].copy()
                if group.empty:
                    ax.axis("off")
                    continue

                group["value"] = group["value"].astype(str)
                sns.boxplot(data=group, x="value", y=metric, ax=ax, color="#4C72B0")

                if len(group) <= 300:
                    sns.swarmplot(
                        data=group,
                        x="value",
                        y=metric,
                        ax=ax,
                        color="#DD8452",
                        size=4,
                    )

                if baseline_value is not None and baseline_value == baseline_value:  # filter NaN
                    ax.axhline(baseline_value, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Baseline")

                value_means = group.groupby("value")[metric].mean()
                if not value_means.empty:
                    best_value = value_means.idxmax()
                    best_score = value_means.max()
                    label = "Best mean" if legend_anchor is None else None
                    ax.scatter(
                        [best_value],
                        [best_score],
                        color="red",
                        s=70,
                        marker="*",
                        zorder=5,
                        label=label,
                    )
                    if label:
                        legend_anchor = ax

                ax.set_xlabel("Parameter value")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(f"'{parameter}' impact")

                if group["value"].dtype == object or group["value"].nunique() > 5:
                    self._rotate_xticks(ax, rotation=30)

            # hide any unused axes
            for ax in axes_iter:
                if ax.has_data():
                    continue
                ax.axis("off")

            if legend_anchor is not None:
                self._place_legend_below(fig, legend_anchor, title=None)

            figure_path = self.figures_dir / "ablation_parameter_distributions.png"
            fig.tight_layout()
            fig.savefig(figure_path, bbox_inches="tight")
            plt.close(fig)

        return figure_path
