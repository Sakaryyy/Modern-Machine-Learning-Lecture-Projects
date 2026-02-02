"""Plotting utilities for reinforcement learning experiment reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
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

        self._plot_action_usage_by_hour(frame, output_dir)
        self._plot_battery_health_distribution(frame, output_dir)


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

    def plot_evaluation_comparison(
            self,
            agent_summary: pd.DataFrame,
            baseline_summary: pd.DataFrame,
            output_dir: Path,
    ) -> None:
        """Compare evaluation reward distributions for agent vs baseline."""

        if agent_summary.empty or baseline_summary.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        agent_frame = agent_summary[["total_reward"]].copy()
        agent_frame["policy"] = "Agent"
        baseline_frame = baseline_summary[["total_reward"]].copy()
        baseline_frame["policy"] = "Baseline"
        combined = pd.concat([agent_frame, baseline_frame], ignore_index=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=combined, x="policy", y="total_reward", ax=ax)
        sns.stripplot(
            data=combined,
            x="policy",
            y="total_reward",
            ax=ax,
            color="black",
            alpha=0.35,
        )
        ax.set_title("Evaluation Reward Distribution")
        ax.set_xlabel("Policy")
        ax.set_ylabel("Total reward per episode")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_dir / "evaluation_reward_distribution.png", bbox_inches="tight")
        plt.close(fig)

    def plot_strategy_comparison(
            self,
            agent_strategy: Mapping[str, float],
            baseline_strategy: Mapping[str, float],
            output_dir: Path,
    ) -> None:
        """Compare high-level strategy summaries for agent vs baseline."""

        if not agent_strategy or not baseline_strategy:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        performance_metrics = [
            ("avg_reward", "Avg reward"),
            ("avg_battery_energy", "Avg battery energy"),
            ("avg_battery_health", "Avg battery health"),
        ]
        share_metrics = [
            ("solar_to_demand_share", "Solar → demand share"),
            ("battery_to_demand_share", "Battery → demand share"),
            ("grid_to_demand_share", "Grid → demand share"),
            ("solar_sold_share", "Solar sold share"),
        ]

        def _build_frame(metrics: Sequence[tuple[str, str]]) -> pd.DataFrame:
            rows = []
            for key, label in metrics:
                if key not in agent_strategy or key not in baseline_strategy:
                    continue
                rows.append({"metric": label, "value": agent_strategy[key], "policy": "Agent"})
                rows.append({"metric": label, "value": baseline_strategy[key], "policy": "Baseline"})
            return pd.DataFrame(rows)

        performance_frame = _build_frame(performance_metrics)
        if not performance_frame.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=performance_frame, x="metric", y="value", hue="policy", ax=ax)
            ax.set_title("Strategy Performance Summary")
            ax.set_xlabel("")
            ax.set_ylabel("Value")
            ax.grid(False)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "strategy_performance_summary.png", bbox_inches="tight")
            plt.close(fig)

        share_frame = _build_frame(share_metrics)
        if not share_frame.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=share_frame, x="metric", y="value", hue="policy", ax=ax)
            ax.set_title("Energy Allocation Share Comparison")
            ax.set_xlabel("")
            ax.set_ylabel("Share")
            ax.set_ylim(0.0, 1.0)
            ax.grid(False)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "strategy_energy_shares.png", bbox_inches="tight")
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
                name="price_demand_actions",
                title="Price, Demand, and Charging Actions",
                y_columns=["buying_price", "demand", "battery_to_grid", "grid_to_battery"],
                y_label="Value",
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

    def _plot_action_usage_by_hour(self, frame: pd.DataFrame, output_dir: Path) -> None:
        if "time_of_day" not in frame.columns:
            return

        columns = [
            "solar_to_demand",
            "solar_to_battery",
            "battery_to_demand",
            "battery_to_grid",
            "grid_to_battery",
        ]
        if any(column not in frame.columns for column in columns):
            return

        grouped = frame.groupby("time_of_day")[columns].mean().reset_index()
        self._plot_lines(
            frame=grouped,
            output_dir=output_dir,
            name="action_usage_by_hour",
            title="Average Action Usage by Hour",
            y_columns=columns,
            y_label="Average energy units",
            x_column="time_of_day",
        )

    def _plot_battery_health_distribution(self, frame: pd.DataFrame, output_dir: Path) -> None:
        if "battery_health" not in frame.columns or "battery_capacity" not in frame.columns:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(frame["battery_health"], bins=20, kde=True, ax=ax, color="tab:blue", label="Health")
        sns.histplot(frame["battery_capacity"], bins=20, kde=True, ax=ax, color="tab:orange", label="Capacity")
        ax.set_title("Battery Health and Capacity Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(False)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "battery_health_capacity_distribution.png", bbox_inches="tight")
        plt.close(fig)

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

    # ==================== COMPREHENSIVE RL DIAGNOSTICS ====================

    def plot_detailed_training_diagnostics(
            self,
            detailed_metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot comprehensive training diagnostics from DetailedTrainingDiagnosticsCallback.

        Parameters
        ----------
        detailed_metrics:
            Dictionary containing all training metrics from the callback.
        output_dir:
            Directory where plots should be saved.
        """
        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot learning curves with rolling averages
        self._plot_learning_curves(detailed_metrics, output_dir)

        # Plot value function estimates
        self._plot_value_estimates(detailed_metrics, output_dir)

        # Plot advantage statistics
        self._plot_advantage_stats(detailed_metrics, output_dir)

        # Plot reward distributions
        self._plot_reward_distributions(detailed_metrics, output_dir)

        # Plot environment state distributions
        self._plot_env_state_distributions(detailed_metrics, output_dir)

        # Plot PPO-specific metrics
        self._plot_ppo_metrics(detailed_metrics, output_dir)

        # Create summary dashboard
        self._plot_training_dashboard(detailed_metrics, output_dir)

    def _plot_learning_curves(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot detailed learning curves with rolling averages."""
        episode_rewards = metrics.get("episode_rewards", [])
        if not episode_rewards:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Raw rewards
        ax = axes[0, 0]
        episodes = np.arange(1, len(episode_rewards) + 1)
        ax.plot(episodes, episode_rewards, alpha=0.3, label="Raw rewards")

        # Rolling averages at different windows
        for window in [10, 50, 100]:
            if len(episode_rewards) >= window:
                rolling = pd.Series(episode_rewards).rolling(window=window).mean()
                ax.plot(episodes, rolling, label=f"{window}-episode avg", linewidth=2)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Learning Curve with Rolling Averages")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reward improvement over time (split into segments)
        ax = axes[0, 1]
        n_segments = min(10, len(episode_rewards) // 10)
        if n_segments >= 2:
            segment_size = len(episode_rewards) // n_segments
            segment_means = []
            segment_stds = []
            segment_labels = []
            for i in range(n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < n_segments - 1 else len(episode_rewards)
                segment = episode_rewards[start:end]
                segment_means.append(np.mean(segment))
                segment_stds.append(np.std(segment))
                segment_labels.append(f"{start + 1}-{end}")

            x_pos = np.arange(n_segments)
            ax.bar(x_pos, segment_means, yerr=segment_stds, capsize=3)
            ax.set_xlabel("Training Segment")
            ax.set_ylabel("Mean Reward")
            ax.set_title("Reward by Training Segment")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(segment_labels, rotation=45, ha="right")

        # Cumulative reward
        ax = axes[1, 0]
        cumulative = np.cumsum(episode_rewards)
        ax.plot(episodes, cumulative)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Cumulative Reward Over Training")
        ax.grid(True, alpha=0.3)

        # Episode lengths
        ax = axes[1, 1]
        episode_lengths = metrics.get("episode_lengths", [])
        if episode_lengths:
            ax.plot(np.arange(1, len(episode_lengths) + 1), episode_lengths, alpha=0.5)
            if len(episode_lengths) >= 10:
                rolling = pd.Series(episode_lengths).rolling(window=10).mean()
                ax.plot(np.arange(1, len(episode_lengths) + 1), rolling, label="10-ep avg", linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Length")
            ax.set_title("Episode Lengths Over Training")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "learning_curves_detailed.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_value_estimates(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot value function estimates over training."""
        value_estimates = metrics.get("value_estimates", [])
        if not value_estimates:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Value estimates over rollouts
        ax = axes[0]
        rollouts = np.arange(1, len(value_estimates) + 1)
        ax.plot(rollouts, value_estimates)
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Mean Value Estimate")
        ax.set_title("Value Function Estimates Over Training")
        ax.grid(True, alpha=0.3)

        # Value estimate distribution
        ax = axes[1]
        ax.hist(value_estimates, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(value_estimates), color="red", linestyle="--", label=f"Mean: {np.mean(value_estimates):.2f}")
        ax.set_xlabel("Value Estimate")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Value Estimates")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "value_estimates.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_advantage_stats(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot advantage statistics over training."""
        advantage_means = metrics.get("advantage_means", [])
        advantage_stds = metrics.get("advantage_stds", [])
        if not advantage_means:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        rollouts = np.arange(1, len(advantage_means) + 1)

        # Advantage mean with std bands
        ax = axes[0]
        ax.plot(rollouts, advantage_means, label="Mean", color="blue")
        if advantage_stds and len(advantage_stds) == len(advantage_means):
            lower = np.array(advantage_means) - np.array(advantage_stds)
            upper = np.array(advantage_means) + np.array(advantage_stds)
            ax.fill_between(rollouts, lower, upper, alpha=0.3, color="blue")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Advantage")
        ax.set_title("Advantage Statistics Over Training")
        ax.grid(True, alpha=0.3)

        # Advantage std over time
        ax = axes[1]
        if advantage_stds:
            ax.plot(rollouts, advantage_stds, color="orange")
            ax.set_xlabel("Rollout")
            ax.set_ylabel("Advantage Std")
            ax.set_title("Advantage Standard Deviation Over Training")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "advantage_statistics.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_reward_distributions(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot reward distributions."""
        episode_rewards = metrics.get("episode_rewards", [])
        per_step_rewards = metrics.get("per_env_rewards", [])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Episode reward distribution
        ax = axes[0]
        if episode_rewards:
            ax.hist(episode_rewards, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(np.mean(episode_rewards), color="red", linestyle="--",
                       label=f"Mean: {np.mean(episode_rewards):.2f}")
            ax.axvline(np.median(episode_rewards), color="green", linestyle="--",
                       label=f"Median: {np.median(episode_rewards):.2f}")
            ax.set_xlabel("Episode Reward")
            ax.set_ylabel("Frequency")
            ax.set_title("Episode Reward Distribution")
            ax.legend()

        # Per-step reward distribution (sampled if too large)
        ax = axes[1]
        if per_step_rewards:
            sample_size = min(10000, len(per_step_rewards))
            sampled = np.random.choice(per_step_rewards, sample_size, replace=False) if len(
                per_step_rewards) > sample_size else per_step_rewards
            ax.hist(sampled, bins=50, edgecolor="black", alpha=0.7, color="coral")
            ax.set_xlabel("Step Reward")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Per-Step Reward Distribution (n={sample_size})")

        # Reward percentiles over training
        ax = axes[2]
        if episode_rewards and len(episode_rewards) >= 20:
            window = max(10, len(episode_rewards) // 20)
            p25, p50, p75 = [], [], []
            for i in range(0, len(episode_rewards), window):
                chunk = episode_rewards[i:i + window]
                if chunk:
                    p25.append(np.percentile(chunk, 25))
                    p50.append(np.percentile(chunk, 50))
                    p75.append(np.percentile(chunk, 75))
            x = np.arange(len(p50)) * window
            ax.fill_between(x, p25, p75, alpha=0.3, label="25-75 percentile")
            ax.plot(x, p50, label="Median", linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Reward Percentiles Over Training")
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "reward_distributions.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_env_state_distributions(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot environment state distributions observed during training."""
        battery_states = metrics.get("battery_states", [])
        buying_prices = metrics.get("buying_prices", [])
        solar_productions = metrics.get("solar_productions", [])
        demands = metrics.get("demands", [])

        if not any([battery_states, buying_prices, solar_productions, demands]):
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Battery state distribution
        ax = axes[0, 0]
        if battery_states:
            ax.hist(battery_states, bins=30, edgecolor="black", alpha=0.7, color="green")
            ax.set_xlabel("Battery Energy")
            ax.set_ylabel("Frequency")
            ax.set_title("Battery Energy Distribution During Training")

        # Buying price distribution
        ax = axes[0, 1]
        if buying_prices:
            ax.hist(buying_prices, bins=30, edgecolor="black", alpha=0.7, color="red")
            ax.set_xlabel("Buying Price")
            ax.set_ylabel("Frequency")
            ax.set_title("Buying Price Distribution During Training")

        # Solar production distribution
        ax = axes[1, 0]
        if solar_productions:
            ax.hist(solar_productions, bins=30, edgecolor="black", alpha=0.7, color="orange")
            ax.set_xlabel("Solar Production")
            ax.set_ylabel("Frequency")
            ax.set_title("Solar Production Distribution During Training")

        # Demand distribution
        ax = axes[1, 1]
        if demands:
            ax.hist(demands, bins=30, edgecolor="black", alpha=0.7, color="blue")
            ax.set_xlabel("Demand")
            ax.set_ylabel("Frequency")
            ax.set_title("Demand Distribution During Training")

        fig.tight_layout()
        fig.savefig(output_dir / "environment_state_distributions.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_ppo_metrics(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Plot PPO-specific training metrics."""
        clip_fractions = metrics.get("clip_fractions", [])
        explained_variances = metrics.get("explained_variances", [])
        policy_entropies = metrics.get("policy_entropies", [])
        learning_rates = metrics.get("learning_rates", [])

        if not any([clip_fractions, explained_variances, policy_entropies, learning_rates]):
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Clip fraction
        ax = axes[0, 0]
        if clip_fractions:
            ax.plot(clip_fractions)
            ax.set_xlabel("Update")
            ax.set_ylabel("Clip Fraction")
            ax.set_title("PPO Clip Fraction Over Training")
            ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, label="Typical threshold")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Explained variance
        ax = axes[0, 1]
        if explained_variances:
            ax.plot(explained_variances)
            ax.set_xlabel("Update")
            ax.set_ylabel("Explained Variance")
            ax.set_title("Value Function Explained Variance")
            ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect prediction")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Policy entropy
        ax = axes[1, 0]
        if policy_entropies:
            ax.plot(policy_entropies)
            ax.set_xlabel("Update")
            ax.set_ylabel("Policy Entropy")
            ax.set_title("Policy Entropy Over Training (Exploration)")
            ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        if learning_rates:
            ax.plot(learning_rates)
            ax.set_xlabel("Update")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "ppo_training_metrics.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_training_dashboard(
            self,
            metrics: Dict[str, Any],
            output_dir: Path,
    ) -> None:
        """Create a comprehensive training dashboard."""
        episode_rewards = metrics.get("episode_rewards", [])
        if not episode_rewards:
            return

        fig = plt.figure(figsize=(16, 12))

        # Main learning curve
        ax1 = fig.add_subplot(2, 2, 1)
        episodes = np.arange(1, len(episode_rewards) + 1)
        ax1.plot(episodes, episode_rewards, alpha=0.3, color="blue")
        if len(episode_rewards) >= 50:
            rolling = pd.Series(episode_rewards).rolling(window=50).mean()
            ax1.plot(episodes, rolling, color="blue", linewidth=2, label="50-ep moving avg")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Progress")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Reward distribution comparison (first vs last quarter)
        ax2 = fig.add_subplot(2, 2, 2)
        if len(episode_rewards) >= 20:
            quarter = len(episode_rewards) // 4
            first_quarter = episode_rewards[:quarter]
            last_quarter = episode_rewards[-quarter:]
            ax2.hist(first_quarter, bins=20, alpha=0.5, label=f"First {quarter} episodes", color="red")
            ax2.hist(last_quarter, bins=20, alpha=0.5, label=f"Last {quarter} episodes", color="green")
            ax2.axvline(np.mean(first_quarter), color="red", linestyle="--")
            ax2.axvline(np.mean(last_quarter), color="green", linestyle="--")
            ax2.set_xlabel("Episode Reward")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Reward Distribution: Early vs Late Training")
            ax2.legend()

        # Value estimates if available
        ax3 = fig.add_subplot(2, 2, 3)
        value_estimates = metrics.get("value_estimates", [])
        if value_estimates:
            ax3.plot(value_estimates, color="purple")
            ax3.set_xlabel("Rollout")
            ax3.set_ylabel("Mean Value Estimate")
            ax3.set_title("Value Function Learning")
            ax3.grid(True, alpha=0.3)

        # Summary statistics
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        stats_text = [
            f"Total Episodes: {len(episode_rewards)}",
            f"Mean Reward: {np.mean(episode_rewards):.2f}",
            f"Std Reward: {np.std(episode_rewards):.2f}",
            f"Min Reward: {np.min(episode_rewards):.2f}",
            f"Max Reward: {np.max(episode_rewards):.2f}",
        ]
        if len(episode_rewards) >= 20:
            first_10 = np.mean(episode_rewards[:10])
            last_10 = np.mean(episode_rewards[-10:])
            improvement = last_10 - first_10
            stats_text.extend([
                "",
                f"First 10 Mean: {first_10:.2f}",
                f"Last 10 Mean: {last_10:.2f}",
                f"Improvement: {improvement:+.2f}",
            ])
        ax4.text(0.1, 0.9, "\n".join(stats_text), transform=ax4.transAxes,
                 fontsize=12, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax4.set_title("Training Summary Statistics")

        fig.tight_layout()
        fig.savefig(output_dir / "training_dashboard.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def plot_agent_vs_baseline_analysis(
            self,
            agent_step_frame: pd.DataFrame,
            baseline_step_frame: pd.DataFrame,
            output_dir: Path,
    ) -> None:
        """Create detailed comparison plots between agent and baseline."""
        if agent_step_frame.empty or baseline_step_frame.empty:
            return

        self.apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))

        # Reward comparison by hour
        ax = axes[0, 0]
        if "time_of_day" in agent_step_frame.columns and "reward" in agent_step_frame.columns:
            agent_by_hour = agent_step_frame.groupby("time_of_day")["reward"].mean()
            baseline_by_hour = baseline_step_frame.groupby("time_of_day")["reward"].mean()
            hours = np.arange(24)
            width = 0.35
            ax.bar(hours - width / 2, agent_by_hour.reindex(hours, fill_value=0), width, label="Agent", alpha=0.8)
            ax.bar(hours + width / 2, baseline_by_hour.reindex(hours, fill_value=0), width, label="Baseline", alpha=0.8)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Mean Reward")
            ax.set_title("Reward by Hour of Day")
            ax.legend()
            ax.set_xticks(hours[::2])

        # Battery usage comparison
        ax = axes[0, 1]
        if "battery_energy" in agent_step_frame.columns:
            ax.hist(agent_step_frame["battery_energy"], bins=20, alpha=0.5, label="Agent", color="blue")
            ax.hist(baseline_step_frame["battery_energy"], bins=20, alpha=0.5, label="Baseline", color="orange")
            ax.set_xlabel("Battery Energy")
            ax.set_ylabel("Frequency")
            ax.set_title("Battery Energy Distribution")
            ax.legend()

        # Solar utilization comparison
        ax = axes[1, 0]
        if all(col in agent_step_frame.columns for col in ["solar_to_demand", "solar_to_battery", "solar_sold"]):
            agent_solar_used = agent_step_frame["solar_to_demand"].sum() + agent_step_frame["solar_to_battery"].sum()
            agent_solar_total = agent_solar_used + agent_step_frame["solar_sold"].sum()
            baseline_solar_used = baseline_step_frame["solar_to_demand"].sum() + baseline_step_frame[
                "solar_to_battery"].sum()
            baseline_solar_total = baseline_solar_used + baseline_step_frame["solar_sold"].sum()

            categories = ["Used\n(demand+battery)", "Sold"]
            agent_vals = [agent_solar_used, agent_step_frame["solar_sold"].sum()]
            baseline_vals = [baseline_solar_used, baseline_step_frame["solar_sold"].sum()]

            x = np.arange(len(categories))
            width = 0.35
            ax.bar(x - width / 2, agent_vals, width, label="Agent")
            ax.bar(x + width / 2, baseline_vals, width, label="Baseline")
            ax.set_ylabel("Energy Units")
            ax.set_title("Solar Energy Allocation")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

        # Grid interaction comparison
        ax = axes[1, 1]
        if all(col in agent_step_frame.columns for col in ["grid_to_battery", "grid_to_demand"]):
            categories = ["Grid→Battery", "Grid→Demand", "Battery→Grid"]
            agent_vals = [
                agent_step_frame["grid_to_battery"].sum(),
                agent_step_frame["grid_to_demand"].sum(),
                agent_step_frame.get("battery_to_grid", pd.Series([0])).sum()
            ]
            baseline_vals = [
                baseline_step_frame["grid_to_battery"].sum(),
                baseline_step_frame["grid_to_demand"].sum(),
                baseline_step_frame.get("battery_to_grid", pd.Series([0])).sum()
            ]

            x = np.arange(len(categories))
            width = 0.35
            ax.bar(x - width / 2, agent_vals, width, label="Agent")
            ax.bar(x + width / 2, baseline_vals, width, label="Baseline")
            ax.set_ylabel("Energy Units")
            ax.set_title("Grid Interaction")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

        # Cumulative reward comparison
        ax = axes[2, 0]
        if "reward" in agent_step_frame.columns:
            agent_cumulative = agent_step_frame["reward"].cumsum()
            baseline_cumulative = baseline_step_frame["reward"].cumsum()
            ax.plot(agent_cumulative.values, label="Agent")
            ax.plot(baseline_cumulative.values, label="Baseline")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative Reward")
            ax.set_title("Cumulative Reward Over Episode")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Action distribution comparison
        ax = axes[2, 1]
        action_cols = ["solar_to_demand", "solar_to_battery", "battery_to_demand", "battery_to_grid", "grid_to_battery"]
        if all(col in agent_step_frame.columns for col in action_cols):
            agent_actions = [agent_step_frame[col].mean() for col in action_cols]
            baseline_actions = [baseline_step_frame[col].mean() for col in action_cols]

            x = np.arange(len(action_cols))
            width = 0.35
            ax.bar(x - width / 2, agent_actions, width, label="Agent")
            ax.bar(x + width / 2, baseline_actions, width, label="Baseline")
            ax.set_ylabel("Mean Action Value")
            ax.set_title("Mean Action Values")
            ax.set_xticks(x)
            ax.set_xticklabels([col.replace("_", "\n") for col in action_cols], fontsize=8)
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "agent_vs_baseline_analysis.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
