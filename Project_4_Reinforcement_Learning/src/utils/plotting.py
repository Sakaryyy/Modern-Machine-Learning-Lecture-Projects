"""Plotting utilities for reinforcement learning experiment reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = [
    "PlotSpec",
    "PlotStyle",
    "RLPlotter",
]


@dataclass(slots=True)
class PlotStyle:
    """Styling parameters for all plots.

    Parameters
    ----------
    context:
        Seaborn context setting (e.g., ``"paper"`` or ``"talk"``).
    palette:
        Color palette name to use for plots.
    style:
        Seaborn style name. Use ``"white"`` for a clean background.
    dpi:
        Resolution of saved plots.
    """

    context: str = "paper"
    palette: str = "deep"
    style: str = "white"
    dpi: int = 300


@dataclass(slots=True)
class PlotSpec:
    """Specification for a multi-series line plot.

    Parameters
    ----------
    name:
        Base filename for the plot.
    title:
        Title displayed at the top of the plot.
    y_columns:
        Columns from the dataframe to visualize.
    y_label:
        Label for the y-axis.
    """

    name: str
    title: str
    y_columns: Sequence[str]
    y_label: str


class RLPlotter:
    """Generate paper-ready plots for reinforcement learning runs."""

    def __init__(self, style: PlotStyle | None = None) -> None:
        self._style = style or PlotStyle()

    def apply_style(self) -> None:
        """Apply the universal seaborn styling to matplotlib."""

        sns.set_theme(
            context=self._style.context,
            style=self._style.style,
            palette=self._style.palette,
            rc={
                "axes.grid": False,
                "figure.dpi": self._style.dpi,
                "savefig.dpi": self._style.dpi,
            },
        )

    def plot_episode_metrics(self, frame: pd.DataFrame, output_dir: Path) -> None:
        """Generate the default set of episode plots.

        Parameters
        ----------
        frame:
            Dataframe containing step-level metrics.
        output_dir:
            Directory where plots should be saved.
        """

        if frame.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        # The default specification covers rewards, environment dynamics, and actions.
        specs = self._default_specs()
        for spec in specs:
            self._plot_lines(
                frame=frame,
                output_dir=output_dir,
                name=spec.name,
                title=spec.title,
                y_columns=spec.y_columns,
                y_label=spec.y_label,
            )

        self._plot_stack(
            frame=frame,
            output_dir=output_dir,
            name="energy_allocation_stack",
            title="Energy Allocation by Source",
            y_columns=[
                "solar_to_demand",
                "solar_to_battery",
                "battery_to_demand",
                "battery_to_grid",
                "grid_to_battery",
                "grid_to_demand",
                "solar_sold",
            ],
            y_label="Energy units",
        )

    def plot_training_metrics(self, frame: pd.DataFrame, output_dir: Path) -> None:
        """Plot training curves across episodes.

        Parameters
        ----------
        frame:
            Dataframe containing episode-level metrics. Expected columns include
            ``episode``, ``total_reward``, and optionally ``avg_reward``.
        output_dir:
            Directory where plots should be saved.
        """

        if frame.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        columns = ["total_reward"]
        if "avg_reward" in frame.columns:
            columns.append("avg_reward")

        self._plot_lines(
            frame=frame,
            output_dir=output_dir,
            name="training_rewards",
            title="Training Reward Progression",
            y_columns=columns,
            y_label="Reward",
            x_column="episode",
        )

    def plot_hyperparameter_sweep(self, frame: pd.DataFrame, output_dir: Path) -> None:
        """Plot sensitivity curves for hyperparameter sweeps.

        Parameters
        ----------
        frame:
            Long-format dataframe containing ``hyperparameter``, ``value``, and
            ``mean_reward`` columns.
        output_dir:
            Directory where plots should be saved.
        """

        if frame.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        for hyperparameter in frame["hyperparameter"].unique():
            subset = frame[frame["hyperparameter"] == hyperparameter]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(
                data=subset,
                x="value",
                y="mean_reward",
                marker="o",
                ax=ax,
            )
            ax.set_title(f"Hyperparameter Sensitivity: {hyperparameter}")
            ax.set_xlabel(hyperparameter)
            ax.set_ylabel("Mean evaluation reward")
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(output_dir / f"sweep_{hyperparameter}.png", bbox_inches="tight")
            plt.close(fig)

    def plot_generation_summary(self, frame: pd.DataFrame, output_dir: Path) -> None:
        """Plot reward statistics across training generations.

        Parameters
        ----------
        frame:
            Dataframe containing ``generation``, ``mean_reward``, and ``std_reward`` columns.
        output_dir:
            Directory where plots should be saved.
        """

        if frame.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=frame, x="generation", y="mean_reward", marker="o", ax=ax)
        ax.fill_between(
            frame["generation"],
            frame["mean_reward"] - frame["std_reward"],
            frame["mean_reward"] + frame["std_reward"],
            alpha=0.2,
        )
        ax.set_title("Generation Performance Summary")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean evaluation reward")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_dir / "generation_summary.png", bbox_inches="tight")
        plt.close(fig)

    def _default_specs(self) -> Sequence[PlotSpec]:
        """Return the default plotting specifications."""

        return [
            PlotSpec(
                name="reward_trace",
                title="Reward and Cumulative Reward",
                y_columns=["reward", "cumulative_reward"],
                y_label="Reward",
            ),
            PlotSpec(
                name="battery_energy",
                title="Battery Energy Over Time",
                y_columns=["battery_energy", "battery_capacity", "battery_health"],
                y_label="Energy units",
            ),
            PlotSpec(
                name="solar_and_demand",
                title="Solar Production and Demand",
                y_columns=["solar_production", "demand"],
                y_label="Energy units",
            ),
            PlotSpec(
                name="price_signal",
                title="Market, Buying, and Selling Prices",
                y_columns=["market_price", "buying_price", "selling_price"],
                y_label="Price",
            ),
            PlotSpec(
                name="solar_intensity",
                title="Solar Intensity",
                y_columns=["solar_intensity"],
                y_label="Intensity",
            ),
            PlotSpec(
                name="forecast_signals",
                title="Next-Hour Forecast Signals",
                y_columns=[
                    "forecast_solar_intensity_1",
                    "forecast_market_price_1",
                    "forecast_demand_1",
                ],
                y_label="Forecast value",
            ),
            PlotSpec(
                name="battery_degradation",
                title="Battery Degradation and Throughput",
                y_columns=["battery_throughput", "degradation_amount", "self_discharge_loss"],
                y_label="Energy / Capacity loss",
            ),
            PlotSpec(
                name="actions",
                title="Policy Actions",
                y_columns=[
                    "action_solar_to_demand",
                    "action_solar_to_battery",
                    "action_battery_to_demand",
                    "action_battery_to_grid",
                    "action_grid_to_battery",
                ],
                y_label="Planned energy units",
            ),
        ]

    def _plot_lines(
            self,
            frame: pd.DataFrame,
            output_dir: Path,
            name: str,
            title: str,
            y_columns: Iterable[str],
            y_label: str,
            x_column: str = "time_step",
    ) -> None:
        """Plot multiple line series against a common x-axis."""

        fig, ax = plt.subplots(figsize=(8, 4))
        # Skip missing columns to keep plotting robust across varied experiments.
        for column in y_columns:
            if column not in frame.columns:
                continue
            sns.lineplot(data=frame, x=x_column, y=column, ax=ax, label=column)
        ax.set_title(title)
        ax.set_xlabel(x_column.replace("_", " ").title())
        ax.set_ylabel(y_label)
        ax.grid(False)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_stack(
            self,
            frame: pd.DataFrame,
            output_dir: Path,
            name: str,
            title: str,
            y_columns: Sequence[str],
            y_label: str,
            x_column: str = "time_step",
    ) -> None:
        """Plot a stacked area chart for energy allocation."""

        # Ensure all required series are present before generating stacked plots.
        if any(column not in frame.columns for column in y_columns):
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        x_values = frame[x_column].to_numpy()
        y_values = [frame[column].to_numpy() for column in y_columns]

        ax.stackplot(x_values, y_values, labels=y_columns)
        ax.set_title(title)
        ax.set_xlabel(x_column.replace("_", " ").title())
        ax.set_ylabel(y_label)
        ax.grid(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
