"""Training pipeline for Stable Baselines3 agents in the energy budgeting task."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from Project_4_Reinforcement_Learning.src.config import EnvironmentConfig
from Project_4_Reinforcement_Learning.src.environment import EnergyBudgetEnv, StepMetrics
from Project_4_Reinforcement_Learning.src.logger import get_logger
from Project_4_Reinforcement_Learning.src.policies import BaselinePolicy
from Project_4_Reinforcement_Learning.src.utils import EpisodeRecorder, RLPlotter, RunArtifactManager

__all__ = [
    "AgentTrainingConfig",
    "HyperparameterSearchConfig",
    "TrainingSummary",
    "run_training_sweep",
]


@dataclass(slots=True)
class AgentTrainingConfig:
    """Configuration for training a Stable Baselines3 agent.

    Parameters
    ----------
    total_timesteps:
        Total number of environment steps used for training.
    n_envs:
        Number of vectorized environments for parallel rollouts.
    eval_episodes:
        Number of evaluation episodes run after training each model.
    seed:
        Optional random seed for reproducibility.
    algorithm:
        Stable Baselines3 algorithm identifier.
    """

    total_timesteps: int = 40_000
    n_envs: int = 4
    eval_episodes: int = 12
    seed: int | None = None
    algorithm: str = "PPO"


@dataclass(slots=True)
class HyperparameterSearchConfig:
    """Hyperparameter sweep specification for training.

    Parameters
    ----------
    grid:
        Mapping of hyperparameter names to candidate values. A cartesian product
        is evaluated across all entries.
    max_configs:
        Optional cap on the number of configurations to evaluate.
    """

    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    max_configs: int | None = None


@dataclass(slots=True)
class TrainingSummary:
    """Container for the key outputs of a training sweep.

    Parameters
    ----------
    best_hyperparameters:
        Hyperparameter dictionary with the best evaluation performance.
    best_mean_reward:
        Mean reward achieved by the best configuration.
    best_std_reward:
        Standard deviation of rewards for the best configuration.
    """

    best_hyperparameters: Mapping[str, Any]
    best_mean_reward: float
    best_std_reward: float


class EpisodeRewardCallback(BaseCallback):
    """Collect episode rewards during Stable Baselines3 training.

    Notes
    -----
    The callback expects the environment to be wrapped with a Monitor so that
    episode statistics are injected into the ``info`` dictionary. The recorded
    data are later used to visualize training progress.
    """

    def __init__(self) -> None:
        super().__init__()
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        """Record episode statistics when available."""

        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._episode_rewards.append(float(episode_info.get("r", 0.0)))
            self._episode_lengths.append(int(episode_info.get("l", 0)))
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Return the captured episode metrics as a dataframe.

        Returns
        -------
        pandas.DataFrame
            Dataframe with columns for episode index, total reward, and length.
        """

        frame = pd.DataFrame(
            {
                "episode": np.arange(1, len(self._episode_rewards) + 1),
                "total_reward": self._episode_rewards,
                "episode_length": self._episode_lengths,
            }
        )
        if not frame.empty:
            frame["avg_reward"] = frame["total_reward"].expanding().mean()
        return frame


def default_hyperparameter_grid() -> Dict[str, Sequence[Any]]:
    """Provide a curated PPO hyperparameter grid for the environment.

    Returns
    -------
    dict
        Mapping of hyperparameter names to candidate values.
    """

    return {
        "learning_rate": [3e-4, 1e-3],
        "gamma": [0.95, 0.99],
        "gae_lambda": [0.9, 0.95],
        "ent_coef": [0.0, 0.01],
        "clip_range": [0.2, 0.3],
    }


def generate_hyperparameter_grid(
        search_config: HyperparameterSearchConfig,
) -> List[Dict[str, Any]]:
    """Expand the hyperparameter grid into a list of configurations.

    Parameters
    ----------
    search_config:
        Hyperparameter sweep specification.

    Returns
    -------
    list[dict]
        List of hyperparameter dictionaries ready for model construction.
    """

    if not search_config.grid:
        grid = default_hyperparameter_grid()
    else:
        grid = dict(search_config.grid)

    keys = list(grid.keys())
    values = [grid[key] for key in keys]

    configs: List[Dict[str, Any]] = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))

    if search_config.max_configs is not None:
        return configs[: search_config.max_configs]
    return configs


def _make_env(config: EnvironmentConfig, seed: int | None) -> EnergyBudgetEnv:
    """Instantiate the energy budgeting environment for rollouts.

    Parameters
    ----------
    config:
        Environment configuration.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    EnergyBudgetEnv
        Instantiated environment.
    """

    env = EnergyBudgetEnv(config)
    if seed is not None:
        env.seed(seed)
    return env


def _build_model(
        algorithm: str,
        env,
        hyperparameters: Mapping[str, Any],
        seed: int | None,
) -> PPO:
    """Construct a Stable Baselines3 model with the requested settings.

    Parameters
    ----------
    algorithm:
        Stable Baselines3 algorithm identifier.
    env:
        Vectorized environment for training.
    hyperparameters:
        Hyperparameters passed to the Stable Baselines3 constructor.
    seed:
        Optional random seed for deterministic behavior.

    Returns
    -------
    stable_baselines3.PPO
        Configured model instance.

    Raises
    ------
    ValueError
        If an unsupported algorithm is requested.
    """

    if algorithm.upper() != "PPO":
        raise ValueError("Only PPO is currently supported for the MultiDiscrete action space.")

    return PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=0,
        **hyperparameters,
    )


def _evaluate_policy(
        policy: Any,
        config: EnvironmentConfig,
        eval_episodes: int,
        seed: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a policy over multiple episodes and record diagnostics.

    Parameters
    ----------
    policy:
        Policy object providing a ``select_action`` method or a
        Stable Baselines3 ``predict`` method.
    config:
        Environment configuration.
    eval_episodes:
        Number of episodes to run.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Episode-level summary dataframe and step-level dataframe.
    """

    summaries: List[Dict[str, Any]] = []
    step_frames: List[pd.DataFrame] = []

    for episode_idx in range(eval_episodes):
        env = _make_env(config, seed=None if seed is None else seed + episode_idx)
        recorder = EpisodeRecorder()
        observation, _ = env.reset(seed=None if seed is None else seed + episode_idx)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            current_observation = observation
            if hasattr(policy, "predict"):
                action, _ = policy.predict(current_observation, deterministic=True)
            else:
                action = policy.select_action(current_observation)
            observation, reward, terminated, truncated, info = env.step(action)
            metrics: StepMetrics = info["metrics"]
            recorder.record_step(
                observation=np.asarray(current_observation),
                action=np.asarray(action),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                metrics=metrics,
            )

        summary = recorder.summary()
        summary.update({"episode": episode_idx + 1})
        summaries.append(summary)

        episode_frame = recorder.to_dataframe()
        episode_frame["episode"] = episode_idx + 1
        step_frames.append(episode_frame)

    summary_frame = pd.DataFrame(summaries)
    step_frame = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    return summary_frame, step_frame


def _summarize_strategy(step_frame: pd.DataFrame) -> Dict[str, float]:
    """Summarize policy behavior from step-level data.

    Parameters
    ----------
    step_frame:
        Dataframe containing detailed environment metrics.

    Returns
    -------
    dict
        Dictionary with aggregate statistics describing the policy behavior.
    """

    if step_frame.empty:
        return {}

    total_demand = float(step_frame["demand"].sum())
    total_solar = float(step_frame["solar_production"].sum())
    demand_covered = float(step_frame["solar_to_demand"].sum())
    battery_covered = float(step_frame["battery_to_demand"].sum())
    grid_covered = float(step_frame["grid_to_demand"].sum())

    return {
        "avg_reward": float(step_frame["reward"].mean()),
        "avg_battery_energy": float(step_frame["battery_energy"].mean()),
        "solar_to_demand_share": demand_covered / total_demand if total_demand else 0.0,
        "battery_to_demand_share": battery_covered / total_demand if total_demand else 0.0,
        "grid_to_demand_share": grid_covered / total_demand if total_demand else 0.0,
        "solar_sold_share": float(step_frame["solar_sold"].sum() / total_solar) if total_solar else 0.0,
        "avg_grid_to_battery": float(step_frame["grid_to_battery"].mean()),
        "avg_solar_to_battery": float(step_frame["solar_to_battery"].mean()),
    }


def _compile_analysis_report(
        best_hyperparameters: Mapping[str, Any],
        agent_summary: Mapping[str, float],
        baseline_summary: Mapping[str, float],
) -> str:
    """Create a textual report describing the learned strategy.

    Parameters
    ----------
    best_hyperparameters:
        Best-performing hyperparameters for the trained agent.
    agent_summary:
        Aggregated statistics for the learned agent.
    baseline_summary:
        Aggregated statistics for the baseline policy.

    Returns
    -------
    str
        Narrative report describing the findings.
    """

    improvement = agent_summary.get("avg_reward", 0.0) - baseline_summary.get("avg_reward", 0.0)

    report_lines = [
        "Strategy Investigation Report",
        "============================",
        "",
        "Algorithm choice:",
        "- PPO is selected because the environment has a low-dimensional observation space and a",
        "  MultiDiscrete action space that PPO supports directly without action discretization.",
        "- The clipped surrogate objective and entropy bonus provide stable training under",
        "  stochastic price/solar/demand dynamics.",
        "",
        "Best hyperparameters:",
    ]
    for key, value in best_hyperparameters.items():
        report_lines.append(f"- {key}: {value}")

    report_lines.extend(
        [
            "",
            "Learned strategy highlights (agent):",
            f"- Average reward per step: {agent_summary.get('avg_reward', 0.0):.3f}",
            f"- Demand coverage (solar/battery/grid):",
            f"  {agent_summary.get('solar_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('battery_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('grid_to_demand_share', 0.0):.2%}",
            f"- Solar sold share: {agent_summary.get('solar_sold_share', 0.0):.2%}",
            f"- Average grid-to-battery charging: {agent_summary.get('avg_grid_to_battery', 0.0):.2f}",
            "",
            "Baseline policy comparison:",
            f"- Baseline average reward per step: {baseline_summary.get('avg_reward', 0.0):.3f}",
            f"- Agent improvement in avg reward per step: {improvement:.3f}",
        ]
    )

    return "\n".join(report_lines)


def _build_sweep_long_frame(
        sweep_frame: pd.DataFrame,
        hyperparameters: Iterable[str],
) -> pd.DataFrame:
    """Convert sweep results into a long-format dataframe for plotting.

    Parameters
    ----------
    sweep_frame:
        Dataframe with one row per hyperparameter configuration.
    hyperparameters:
        Names of hyperparameters to pivot into long format.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe for plotting sensitivity curves.
    """

    long_frame = sweep_frame.melt(
        id_vars=["config_id", "mean_reward", "std_reward"],
        value_vars=list(hyperparameters),
        var_name="hyperparameter",
        value_name="value",
    )
    return long_frame


def run_training_sweep(
        output_dir: Path,
        run_name: str | None,
        training_config: AgentTrainingConfig | None = None,
        search_config: HyperparameterSearchConfig | None = None,
) -> TrainingSummary:
    """Run a hyperparameter sweep and evaluate the best PPO configuration.

    Parameters
    ----------
    output_dir:
        Root directory where experiment artifacts should be written.
    run_name:
        Optional descriptive name for the run folder.
    training_config:
        Training configuration describing the overall training budget.
    search_config:
        Hyperparameter sweep configuration.

    Returns
    -------
    TrainingSummary
        Summary of the best-performing configuration.
    """

    logger = get_logger("Training")
    training_config = training_config or AgentTrainingConfig()
    search_config = search_config or HyperparameterSearchConfig()

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    config = EnvironmentConfig()

    artifact_manager.save_config(config, filename="environment_config.json")
    artifact_manager.save_config(training_config, filename="training_config.json")

    hyperparameter_grid = generate_hyperparameter_grid(search_config)
    artifact_manager.save_json(
        {
            "grid": {k: list(v) for k, v in (search_config.grid or default_hyperparameter_grid()).items()},
            "max_configs": search_config.max_configs,
        },
        filename="hyperparameter_grid.json",
    )

    logger.info("Starting sweep with %s configurations.", len(hyperparameter_grid))

    sweep_records: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_model: PPO | None = None
    best_hyperparameters: Mapping[str, Any] = {}
    best_training_frame: pd.DataFrame | None = None

    for idx, hyperparameters in enumerate(hyperparameter_grid, start=1):
        logger.info("Training configuration %s/%s: %s", idx, len(hyperparameter_grid), hyperparameters)
        env = make_vec_env(
            lambda: _make_env(config, seed=training_config.seed),
            n_envs=training_config.n_envs,
            seed=training_config.seed,
            monitor_dir=str(run_metadata.root_dir / "data"),
        )

        callback = EpisodeRewardCallback()
        model = _build_model(
            algorithm=training_config.algorithm,
            env=env,
            hyperparameters=hyperparameters,
            seed=training_config.seed,
        )
        model.learn(total_timesteps=training_config.total_timesteps, callback=callback)
        env.close()

        training_frame = callback.to_dataframe()
        if not training_frame.empty:
            artifact_manager.save_dataframe(
                training_frame,
                filename=f"training_metrics_config_{idx}",
                subdir="data",
            )

        summary_frame, step_frame = _evaluate_policy(
            model,
            config,
            training_config.eval_episodes,
            training_config.seed,
        )

        mean_reward = float(summary_frame["total_reward"].mean()) if not summary_frame.empty else -np.inf
        std_reward = float(summary_frame["total_reward"].std(ddof=0)) if not summary_frame.empty else 0.0
        sweep_record = {
            "config_id": idx,
            **hyperparameters,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
        sweep_records.append(sweep_record)

        artifact_manager.save_dataframe(
            summary_frame,
            filename=f"evaluation_summary_config_{idx}",
            subdir="data",
        )
        artifact_manager.save_dataframe(
            step_frame,
            filename=f"evaluation_steps_config_{idx}",
            subdir="environment_state",
        )

        if mean_reward > best_score:
            best_score = mean_reward
            best_model = model
            best_hyperparameters = hyperparameters
            best_training_frame = training_frame

    sweep_frame = pd.DataFrame(sweep_records)
    artifact_manager.save_dataframe(sweep_frame, filename="hyperparameter_sweep", subdir="data")

    plotter = RLPlotter()
    if best_training_frame is not None:
        plotter.plot_training_metrics(best_training_frame, output_dir=run_metadata.root_dir / "plots")

    if not sweep_frame.empty:
        long_frame = _build_sweep_long_frame(sweep_frame, best_hyperparameters.keys())
        plotter.plot_hyperparameter_sweep(long_frame, output_dir=run_metadata.root_dir / "plots")

    if best_model is None:
        raise RuntimeError("No valid model was trained during the sweep.")

    artifact_manager.save_policy(best_model, filename="best_ppo_agent.zip")

    agent_summary_frame, agent_step_frame = _evaluate_policy(
        best_model,
        config,
        training_config.eval_episodes,
        training_config.seed,
    )
    baseline_policy = BaselinePolicy(config)
    baseline_summary_frame, baseline_step_frame = _evaluate_policy(
        baseline_policy,
        config,
        training_config.eval_episodes,
        training_config.seed,
    )

    artifact_manager.save_dataframe(agent_summary_frame, filename="best_agent_summary", subdir="data")
    artifact_manager.save_dataframe(agent_step_frame, filename="best_agent_steps", subdir="environment_state")
    artifact_manager.save_dataframe(baseline_summary_frame, filename="baseline_summary", subdir="data")
    artifact_manager.save_dataframe(baseline_step_frame, filename="baseline_steps", subdir="environment_state")

    if not agent_step_frame.empty:
        plotter.plot_episode_metrics(
            agent_step_frame[agent_step_frame["episode"] == 1],
            output_dir=run_metadata.root_dir / "plots",
        )
    if not baseline_step_frame.empty:
        plotter.plot_episode_metrics(
            baseline_step_frame[baseline_step_frame["episode"] == 1],
            output_dir=run_metadata.root_dir / "plots",
        )

    agent_strategy = _summarize_strategy(agent_step_frame)
    baseline_strategy = _summarize_strategy(baseline_step_frame)
    strategy_report = _compile_analysis_report(best_hyperparameters, agent_strategy, baseline_strategy)

    artifact_manager.save_json(agent_strategy, filename="agent_strategy_summary.json", subdir="data")
    artifact_manager.save_json(baseline_strategy, filename="baseline_strategy_summary.json", subdir="data")
    artifact_manager.save_text(strategy_report, filename="strategy_report.txt", subdir="data")

    logger.info("Best hyperparameters: %s", best_hyperparameters)
    logger.info("Best mean reward: %.3f", best_score)

    return TrainingSummary(
        best_hyperparameters=best_hyperparameters,
        best_mean_reward=best_score,
        best_std_reward=float(sweep_frame.loc[sweep_frame["mean_reward"].idxmax(), "std_reward"])
        if not sweep_frame.empty
        else 0.0,
    )
