"""Training pipeline for Stable Baselines3 agents in the energy budgeting task."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
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
    "RECOMMENDED_ALGORITHM",
    "TrainingSummary",
    "TrainingSummaryHyperparameter",
    "run_training_sweep",
    "run_training"
]

RECOMMENDED_ALGORITHM = "PPO"

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
        Stable Baselines3 algorithm identifier. PPO is the default choice for
        this project due to its stable updates in noisy environments.
    generations:
        Number of independent training runs (with different seeds).
    ppo_hyperparameters:
        PPO hyperparameters used for all generations.
    """

    total_timesteps: int = 200_000
    n_envs: int = 10
    eval_episodes: int = 20
    seed: int | None = None
    algorithm: str = RECOMMENDED_ALGORITHM
    generations: int = 10
    ppo_hyperparameters: Mapping[str, Any] = field(default_factory=dict)


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
class TrainingSummaryHyperparameter:
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


@dataclass(slots=True)
class TrainingSummary:
    """Container for the key outputs of training runs.

    Parameters
    ----------
    best_generation:
        Index of the best-performing generation.
    best_mean_reward:
        Mean reward achieved by the best configuration.
    best_std_reward:
        Standard deviation of rewards for the best configuration.
    best_hyperparameters:
        Hyperparameter dictionary used for training.
    """
    best_generation: int
    best_mean_reward: float
    best_std_reward: float
    best_hyperparameters: Mapping[str, Any]

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
        "learning_rate": [3e-4, 5e-4, 1e-3],
        "n_steps": [128, 256, 512],
        "batch_size": [64, 128],
        "n_epochs": [5, 10],
        "gamma": [0.95, 0.99],
        "gae_lambda": [0.9, 0.95],
        "clip_range": [0.1, 0.2, 0.3],
        "ent_coef": [0.0, 0.01],
        "vf_coef": [0.5, 1.0],
        "max_grad_norm": [0.5, 1.0],
        "normalize_advantage": [True, False],
    }


def default_ppo_hyperparameters() -> Dict[str, Any]:
    """Provide a curated PPO hyperparameter set for the refined environment.

    Returns
    -------
    dict
        Mapping of PPO hyperparameters to use for training.
    """

    return {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "n_steps": 1024,
        "batch_size": 256,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "n_epochs": 5,
    }


def default_algorithm_hyperparameter_grid(algorithm: str) -> Dict[str, Sequence[Any]]:
    """Provide an algorithm-specific hyperparameter grid.

    Parameters
    ----------
    algorithm:
        Stable Baselines3 algorithm identifier.

    Returns
    -------
    dict
        Mapping of hyperparameter names to candidate values.
    """

    algorithm_key = algorithm.upper()
    if algorithm_key == "PPO":
        return default_hyperparameter_grid()
    if algorithm_key == "A2C":
        return {
            "learning_rate": [3e-4, 7e-4],
            "n_steps": [5, 10, 20],
            "gamma": [0.95, 0.99],
            "gae_lambda": [0.9, 0.95],
            "ent_coef": [0.0, 0.01],
            "vf_coef": [0.5, 1.0],
            "max_grad_norm": [0.5, 1.0],
            "rms_prop_eps": [1e-5, 1e-4],
            "use_rms_prop": [True, False],
            "normalize_advantage": [True, False],
        }
    raise ValueError(f"Unsupported algorithm for default hyperparameters: {algorithm}")


def generate_hyperparameter_grid(
        search_config: HyperparameterSearchConfig,
        algorithm: str,
) -> List[Dict[str, Any]]:
    """Expand the hyperparameter grid into a list of configurations.

    Parameters
    ----------
    search_config:
        Hyperparameter sweep specification.
    algorithm:
        Algorithm identifier used to pick a default grid when none is supplied.

    Returns
    -------
    list[dict]
        List of hyperparameter dictionaries ready for model construction.
    """

    if not search_config.grid:
        grid = default_algorithm_hyperparameter_grid(algorithm)
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
) -> BaseAlgorithm:
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
    stable_baselines3.common.base_class.BaseAlgorithm
        Configured model instance.

    Raises
    ------
    ValueError
        If an unsupported algorithm is requested.
    """

    # PPO and A2C are on-policy algorithms that natively support MultiDiscrete action spaces
    # and provide stable learning under stochastic dynamics. PPO is the default choice because
    # its clipped objective tends to be more robust to noisy rewards while keeping the policy
    # updates conservative. A2C remains available for faster iterations and comparison.
    algorithm_registry = {
        "PPO": PPO,
        "A2C": A2C,
    }
    algorithm_key = algorithm.upper()
    if algorithm_key not in algorithm_registry:
        raise ValueError(
            "Only PPO and A2C are supported for the MultiDiscrete action space in this project."
        )

    model_kwargs = dict(hyperparameters)
    # Stable Baselines3 warns when using PPO/A2C with MLP policies on GPU; force CPU unless
    # the user explicitly overrides the device.
    model_kwargs.setdefault("device", "cpu")

    return algorithm_registry[algorithm_key](
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=0,
        **model_kwargs,
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
    battery_sold = float(step_frame["battery_to_grid"].sum())

    return {
        "avg_reward": float(step_frame["reward"].mean()),
        "avg_battery_energy": float(step_frame["battery_energy"].mean()),
        "avg_battery_health": float(step_frame["battery_health"].mean()),
        "solar_to_demand_share": demand_covered / total_demand if total_demand else 0.0,
        "battery_to_demand_share": battery_covered / total_demand if total_demand else 0.0,
        "grid_to_demand_share": grid_covered / total_demand if total_demand else 0.0,
        "solar_sold_share": float(step_frame["solar_sold"].sum() / total_solar) if total_solar else 0.0,
        "avg_grid_to_battery": float(step_frame["grid_to_battery"].mean()),
        "avg_solar_to_battery": float(step_frame["solar_to_battery"].mean()),
        "avg_battery_to_grid": float(step_frame["battery_to_grid"].mean()),
    }


def _compile_analysis_report(
        algorithm: str,
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
        f"- Selected algorithm: {algorithm.upper()}.",
        "- PPO/A2C are chosen because the environment has a low-dimensional observation space and",
        "  a MultiDiscrete action space that both algorithms support directly without action",
        "  discretization.",
        "- PPO is the default because its clipped surrogate objective yields more stable learning",
        "  under stochastic price/solar/demand dynamics, while A2C provides a faster baseline for",
        "  comparison.",
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
            f"- Average battery health: {agent_summary.get('avg_battery_health', 0.0):.3f}",
            f"- Demand coverage (solar/battery/grid):",
            f"  {agent_summary.get('solar_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('battery_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('grid_to_demand_share', 0.0):.2%}",
            f"- Solar sold share: {agent_summary.get('solar_sold_share', 0.0):.2%}",
            f"- Average grid-to-battery charging: {agent_summary.get('avg_grid_to_battery', 0.0):.2f}",
            f"- Average battery-to-grid selling: {agent_summary.get('avg_battery_to_grid', 0.0):.2f}",
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
    if long_frame.empty:
        return long_frame

    def _coerce_numeric(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, np.number)):
            return float(value)
        return None

    long_frame["numeric_value"] = long_frame["value"].apply(_coerce_numeric)
    long_frame = long_frame.dropna(subset=["numeric_value"]).copy()
    long_frame["value"] = long_frame["numeric_value"]
    long_frame = long_frame.drop(columns=["numeric_value"])
    return long_frame


def run_training_sweep(
        output_dir: Path,
        run_name: str | None,
        training_config: AgentTrainingConfig | None = None,
        search_config: HyperparameterSearchConfig | None = None,
        env_config: EnvironmentConfig | None = None,
) -> TrainingSummaryHyperparameter:
    """Run a hyperparameter sweep and evaluate the best configuration.

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
    env_config:
        Optional environment configuration override (e.g., long-horizon episodes).

    Returns
    -------
    TrainingSummaryHyperparameter
        Summary of the best-performing configuration.
    """

    logger = get_logger("Training")
    training_config = training_config or AgentTrainingConfig()
    search_config = search_config or HyperparameterSearchConfig()

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    config = env_config or EnvironmentConfig()

    artifact_manager.save_config(config, filename="environment_config.json")
    artifact_manager.save_config(training_config, filename="training_config.json")

    hyperparameter_grid = generate_hyperparameter_grid(search_config, training_config.algorithm)
    artifact_manager.save_json(
        {
            "grid": {
                k: list(v)
                for k, v in (
                        search_config.grid
                        or default_algorithm_hyperparameter_grid(training_config.algorithm)
                ).items()
            },
            "max_configs": search_config.max_configs,
        },
        filename="hyperparameter_grid.json",
    )

    logger.info("Starting sweep with %s configurations.", len(hyperparameter_grid))

    sweep_records: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_model: BaseAlgorithm | None = None
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

    artifact_manager.save_policy(
        best_model,
        filename=f"best_{training_config.algorithm.lower()}_agent.zip",
    )

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

    plotter.plot_evaluation_comparison(
        agent_summary_frame,
        baseline_summary_frame,
        output_dir=run_metadata.root_dir / "plots",
    )

    agent_strategy = _summarize_strategy(agent_step_frame)
    baseline_strategy = _summarize_strategy(baseline_step_frame)
    strategy_report = _compile_analysis_report(
        training_config.algorithm,
        best_hyperparameters,
        agent_strategy,
        baseline_strategy,
    )

    artifact_manager.save_json(agent_strategy, filename="agent_strategy_summary.json", subdir="data")
    artifact_manager.save_json(baseline_strategy, filename="baseline_strategy_summary.json", subdir="data")
    artifact_manager.save_text(strategy_report, filename="strategy_report.txt", subdir="data")
    plotter.plot_strategy_comparison(
        agent_strategy,
        baseline_strategy,
        output_dir=run_metadata.root_dir / "plots",
    )

    logger.info("Best hyperparameters: %s", best_hyperparameters)
    logger.info("Best mean reward: %.3f", best_score)

    return TrainingSummaryHyperparameter(
        best_hyperparameters=best_hyperparameters,
        best_mean_reward=best_score,
        best_std_reward=float(sweep_frame.loc[sweep_frame["mean_reward"].idxmax(), "std_reward"])
        if not sweep_frame.empty
        else 0.0,
    )


def run_training(
        output_dir: Path,
        run_name: str | None,
        training_config: AgentTrainingConfig | None = None,
        env_config: EnvironmentConfig | None = None,
) -> TrainingSummary:
    """Run multiple generations of PPO training and evaluate the best agent.

    Parameters
    ----------
    output_dir:
        Root directory where experiment artifacts should be written.
    run_name:
        Optional descriptive name for the run folder.
    training_config:
        Training configuration describing the overall training budget.
    env_config:
        Optional environment configuration override (e.g., long-horizon episodes).

    Returns
    -------
    TrainingSummary
        Summary of the best-performing generation.
    """

    logger = get_logger("Training")
    training_config = training_config or AgentTrainingConfig()
    hyperparameters = dict(training_config.ppo_hyperparameters) or default_ppo_hyperparameters()

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    config = env_config or EnvironmentConfig()

    artifact_manager.save_config(config, filename="environment_config.json")
    artifact_manager.save_config(training_config, filename="training_config.json")
    artifact_manager.save_json(hyperparameters, filename="ppo_hyperparameters.json")

    logger.info("Starting PPO training with %s generations.", training_config.generations)
    generation_records: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_model: PPO | BaseAlgorithm | None = None
    best_generation = 0
    best_training_frame: pd.DataFrame | None = None

    for generation in range(1, training_config.generations + 1):
        seed = None if training_config.seed is None else training_config.seed + generation
        logger.info("Training generation %s/%s with seed=%s", generation, training_config.generations, seed)

        env = make_vec_env(
            lambda: _make_env(config, seed=seed),
            n_envs=training_config.n_envs,
            seed=seed,
            monitor_dir=str(run_metadata.root_dir / "data"),
        )

        callback = EpisodeRewardCallback()
        model = _build_model(
            algorithm=training_config.algorithm,
            env=env,
            hyperparameters=hyperparameters,
            seed=seed,
        )

        model.learn(total_timesteps=training_config.total_timesteps, callback=callback)
        env.close()

        training_frame = callback.to_dataframe()
        if not training_frame.empty:
            artifact_manager.save_dataframe(
                training_frame,
                filename=f"training_metrics_generation_{generation}",
                subdir="data",
            )

        summary_frame, step_frame = _evaluate_policy(
            model,
            config,
            training_config.eval_episodes,
            seed,
        )

        mean_reward = float(summary_frame["total_reward"].mean()) if not summary_frame.empty else -np.inf
        std_reward = float(summary_frame["total_reward"].std(ddof=0)) if not summary_frame.empty else 0.0

        generation_records.append(
            {
                "generation": generation,
                "seed": seed,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
            }
        )

        artifact_manager.save_dataframe(
            summary_frame,
            filename=f"evaluation_summary_generation_{generation}",
            subdir="data",
        )
        artifact_manager.save_dataframe(
            step_frame,
            filename=f"evaluation_steps_generation_{generation}",
            subdir="environment_state",
        )

        if mean_reward > best_score:
            best_score = mean_reward
            best_model = model
            best_generation = generation
            best_training_frame = training_frame

    sweep_frame = pd.DataFrame(generation_records)
    artifact_manager.save_dataframe(sweep_frame, filename="generation_summary", subdir="data")

    plotter = RLPlotter()
    if best_training_frame is not None:
        plotter.plot_training_metrics(best_training_frame, output_dir=run_metadata.root_dir / "plots")

    if not sweep_frame.empty:
        plotter.plot_generation_summary(sweep_frame, output_dir=run_metadata.root_dir / "plots")

    if best_model is None:
        raise RuntimeError("No valid model was trained during the generation loop.")

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
    plotter.plot_evaluation_comparison(
        agent_summary_frame,
        baseline_summary_frame,
        output_dir=run_metadata.root_dir / "plots",
    )

    agent_strategy = _summarize_strategy(agent_step_frame)
    baseline_strategy = _summarize_strategy(baseline_step_frame)
    strategy_report = _compile_analysis_report(training_config.algorithm, hyperparameters, agent_strategy,
                                               baseline_strategy)

    artifact_manager.save_json(agent_strategy, filename="agent_strategy_summary.json", subdir="data")
    artifact_manager.save_json(baseline_strategy, filename="baseline_strategy_summary.json", subdir="data")
    artifact_manager.save_text(strategy_report, filename="strategy_report.txt", subdir="data")
    plotter.plot_strategy_comparison(
        agent_strategy,
        baseline_strategy,
        output_dir=run_metadata.root_dir / "plots",
    )

    logger.info("Best generation: %s", best_generation)
    logger.info("Best mean reward: %.3f", best_score)

    best_std_reward = float(
        sweep_frame.loc[sweep_frame["mean_reward"].idxmax(), "std_reward"]
    ) if not sweep_frame.empty else 0.0

    return TrainingSummary(
        best_generation=best_generation,
        best_mean_reward=best_score,
        best_std_reward=best_std_reward,
        best_hyperparameters=hyperparameters,
    )
