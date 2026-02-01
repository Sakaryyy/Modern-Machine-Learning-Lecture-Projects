"""Training pipeline for Stable Baselines3 agents in the energy budgeting task."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

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
    "compute_gamma_for_horizon",
    "default_ppo_hyperparameters",
    "long_horizon_ppo_hyperparameters",
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
    use_action_masking:
        Whether to enable MaskablePPO with action masks.
    use_vec_normalize:
        Whether to normalize observations (and optionally rewards).
    normalize_reward:
        Whether to normalize rewards when using VecNormalize.
    resume_best_model:
        Whether to resume training from the best model across generations.
    tensorboard_log:
        Whether to write TensorBoard logs to the run directory.
    show_progress:
        Whether to show tqdm progress bars for training and evaluation.
    """

    total_timesteps: int = 200_000
    n_envs: int = 10
    eval_episodes: int = 20
    seed: int | None = None
    algorithm: str = RECOMMENDED_ALGORITHM
    generations: int = 10
    ppo_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    use_action_masking: bool = True
    use_vec_normalize: bool = True
    normalize_reward: bool = True
    resume_best_model: bool = True
    tensorboard_log: bool = True
    show_progress: bool = True


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
        self._action_totals: List[Dict[str, float]] = []
        self._degradation_events: List[int] = []
        self._per_env_action_totals: List[Dict[str, float]] = []
        self._per_env_degradation_events: List[int] = []

    def _on_step(self) -> bool:
        """Record episode statistics when available."""

        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])
        if not self._per_env_action_totals and len(actions) > 0:
            self._per_env_action_totals = [
                {
                    "solar_to_demand": 0.0,
                    "solar_to_battery": 0.0,
                    "battery_to_demand": 0.0,
                    "battery_to_grid": 0.0,
                    "grid_to_battery": 0.0,
                }
                for _ in range(len(actions))
            ]
            self._per_env_degradation_events = [0 for _ in range(len(actions))]

        if len(actions) > 0:
            for env_idx, action in enumerate(np.asarray(actions)):
                if action is None:
                    continue
                action_values = np.asarray(action).reshape(-1)
                if action_values.size != 5:
                    continue
                totals = self._per_env_action_totals[env_idx]
                totals["solar_to_demand"] += float(action_values[0])
                totals["solar_to_battery"] += float(action_values[1])
                totals["battery_to_demand"] += float(action_values[2])
                totals["battery_to_grid"] += float(action_values[3])
                totals["grid_to_battery"] += float(action_values[4])

        for env_idx, info in enumerate(infos):
            metrics = info.get("metrics")
            if metrics is not None and getattr(metrics, "degradation_event", False):
                self._per_env_degradation_events[env_idx] += 1
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._episode_rewards.append(float(episode_info.get("r", 0.0)))
            self._episode_lengths.append(int(episode_info.get("l", 0)))
            self._action_totals.append(dict(self._per_env_action_totals[env_idx]))
            self._degradation_events.append(int(self._per_env_degradation_events[env_idx]))
            self._per_env_action_totals[env_idx] = {
                "solar_to_demand": 0.0,
                "solar_to_battery": 0.0,
                "battery_to_demand": 0.0,
                "battery_to_grid": 0.0,
                "grid_to_battery": 0.0,
            }
            self._per_env_degradation_events[env_idx] = 0
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
                "action_solar_to_demand": [totals.get("solar_to_demand", 0.0) for totals in self._action_totals],
                "action_solar_to_battery": [totals.get("solar_to_battery", 0.0) for totals in self._action_totals],
                "action_battery_to_demand": [totals.get("battery_to_demand", 0.0) for totals in self._action_totals],
                "action_battery_to_grid": [totals.get("battery_to_grid", 0.0) for totals in self._action_totals],
                "action_grid_to_battery": [totals.get("grid_to_battery", 0.0) for totals in self._action_totals],
                "degradation_events": self._degradation_events,
            }
        )
        if not frame.empty:
            frame["avg_reward"] = frame["total_reward"].expanding().mean()
        return frame


class TrainingProgressCallback(BaseCallback):
    """Display per-step training progress and timing."""

    def __init__(
            self,
            total_timesteps: int,
            update_interval: int = 250,
    ) -> None:
        super().__init__()
        self._total_timesteps = total_timesteps
        self._update_interval = max(1, update_interval)
        self._progress: tqdm | None = None
        self._start_time: float | None = None
        self._last_update_steps = 0
        self._initial_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()
        self._initial_timesteps = self.num_timesteps
        self._last_update_steps = 0
        self._progress = tqdm(
            total=self._total_timesteps,
            desc="Training steps",
            leave=True,
            position=0,
        )

    def _on_step(self) -> bool:
        if self._progress is None or self._start_time is None:
            return True
        steps_done = self.num_timesteps - self._initial_timesteps
        if steps_done <= 0:
            return True
        if steps_done - self._last_update_steps < self._update_interval and steps_done < self._total_timesteps:
            return True
        elapsed = time.perf_counter() - self._start_time
        steps_delta = steps_done - self._last_update_steps
        if steps_delta > 0:
            self._progress.update(steps_delta)
        steps_per_second = steps_done / elapsed if elapsed > 0 else 0.0
        ms_per_step = (elapsed / steps_done) * 1000.0 if steps_done > 0 else 0.0
        self._progress.set_postfix(
            {
                "steps/s": f"{steps_per_second:.1f}",
                "ms/step": f"{ms_per_step:.2f}",
            }
        )
        self._last_update_steps = steps_done
        return True

    def _on_training_end(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None


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

    Notes
    -----
    Key optimizations:
    - learning_rate: 3e-4 is more stable than 1e-3 for PPO
    - ent_coef: 0.02 encourages more exploration in the large action space
    - n_steps: 2048 allows for better advantage estimation
    - batch_size: 128 for more frequent updates
    - gamma: 0.99 is suitable for 24-hour episodes; for longer episodes,
      use compute_gamma_for_horizon() to adjust appropriately
    - policy_kwargs: Larger networks for the complex action space
    """

    return {
        "learning_rate": 3e-4,  # More stable than 1e-3
        "gamma": 0.99,  # Suitable for 24-hour episodes
        "gae_lambda": 0.95,  # Higher for better advantage estimation
        "ent_coef": 0.02,  # More exploration for complex action space
        "clip_range": 0.2,
        "n_steps": 2048,  # More samples before update
        "batch_size": 128,  # More frequent gradient updates
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "n_epochs": 10,  # More epochs per update
        # Larger network for complex MultiDiscrete action space (12x12x11x11x11 = 191,664 actions)
        "policy_kwargs": {
            "net_arch": {
                "pi": [256, 256, 128],  # Policy network: deeper for action complexity
                "vf": [256, 256, 128],  # Value network: matches policy depth
            },
        },
    }


def long_horizon_ppo_hyperparameters(episode_length: int) -> Dict[str, Any]:
    """Provide PPO hyperparameters optimized for long-horizon training.

    Parameters
    ----------
    episode_length:
        Number of steps per episode (e.g., 8760 for a year).

    Returns
    -------
    dict
        PPO hyperparameters tuned for long-horizon learning.

    Notes
    -----
    Long-horizon training requires:
    - Higher gamma to not discount away future rewards
    - More n_steps to capture longer-term patterns
    - Lower learning rate for stability over long rollouts
    - Higher entropy for sustained exploration
    """

    gamma = compute_gamma_for_horizon(episode_length, end_discount=0.1)

    return {
        "learning_rate": 1e-4,  # Lower for stability in long rollouts
        "gamma": gamma,  # Auto-computed for episode length
        "gae_lambda": 0.98,  # Higher for long-horizon advantage estimation
        "ent_coef": 0.03,  # More exploration for seasonal patterns
        "clip_range": 0.15,  # Smaller for more conservative updates
        "n_steps": min(4096, episode_length // 2),  # Longer rollouts, capped
        "batch_size": 256,  # Larger batches for variance reduction
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "n_epochs": 10,
        "policy_kwargs": {
            "net_arch": {
                "pi": [512, 256, 256],  # Larger network for complex patterns
                "vf": [512, 256, 256],
            },
        },
    }


def compute_gamma_for_horizon(episode_length: int, end_discount: float = 0.1) -> float:
    """Compute appropriate gamma for a given episode length.

    For long-horizon training, we need a higher gamma to ensure rewards
    at the end of the episode are not discounted to near-zero.

    Parameters
    ----------
    episode_length:
        Number of steps in an episode.
    end_discount:
        Target discount factor at the end of the episode.
        Default 0.1 means rewards at the final step are worth 10% of immediate rewards.

    Returns
    -------
    float
        Appropriate gamma value.

    Examples
    --------
    >>> compute_gamma_for_horizon(24)    # 24-hour episode
    0.9090...
    >>> compute_gamma_for_horizon(8760)  # Year-long episode
    0.9997...
    """
    if episode_length <= 1:
        return 0.99
    # gamma^episode_length = end_discount
    # gamma = end_discount^(1/episode_length)
    gamma = end_discount ** (1.0 / episode_length)
    # Clamp to reasonable range
    return float(np.clip(gamma, 0.9, 0.9999))


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


def _make_env(
        config: EnvironmentConfig,
        seed: int | None,
        use_action_masking: bool = False,
) -> EnergyBudgetEnv:
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
    if use_action_masking:
        env = ActionMasker(env, lambda inner_env: inner_env.get_action_mask())
    return env


def _make_vec_env(
        config: EnvironmentConfig,
        seed: int | None,
        n_envs: int,
        monitor_dir: str | None,
        use_action_masking: bool,
        use_vec_normalize: bool,
        normalize_reward: bool,
):
    env = make_vec_env(
        lambda: _make_env(config, seed=seed, use_action_masking=use_action_masking),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
    )
    if use_vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=normalize_reward)
    return env


def _build_model(
        algorithm: str,
        env,
        hyperparameters: Mapping[str, Any],
        seed: int | None,
        use_action_masking: bool,
        tensorboard_log: Path | None,
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
        "PPO": MaskablePPO if use_action_masking else PPO,
        "A2C": A2C,
    }
    algorithm_key = algorithm.upper()
    if algorithm_key not in algorithm_registry:
        raise ValueError(
            "Only PPO and A2C are supported for the MultiDiscrete action space in this project."
        )
    if use_action_masking and algorithm_key != "PPO":
        raise ValueError("Action masking is only supported with PPO/MaskablePPO.")

    model_kwargs = dict(hyperparameters)
    # Stable Baselines3 warns when using PPO/A2C with MLP policies on GPU; force CPU unless
    # the user explicitly overrides the device.
    model_kwargs.setdefault("device", "cpu")

    return algorithm_registry[algorithm_key](
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        **model_kwargs,
    )


def _evaluate_policy(
        policy: Any,
        config: EnvironmentConfig,
        eval_episodes: int,
        seed: int | None,
        vec_normalize_path: Path | None = None,
        use_action_masking: bool = False,
        show_progress: bool = True,
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

    progress_bar = (
        tqdm(range(eval_episodes), desc="Evaluating episodes", leave=True, position=0)
        if show_progress
        else range(eval_episodes)
    )

    if vec_normalize_path is None:
        env = _make_env(config, seed=seed, use_action_masking=use_action_masking)
    else:
        vec_env = _make_vec_env(
            config=config,
            seed=seed,
            n_envs=1,
            monitor_dir=None,
            use_action_masking=use_action_masking,
            use_vec_normalize=True,
            normalize_reward=False,
        )
        env = VecNormalize.load(str(vec_normalize_path), vec_env)
        env.training = False
        env.norm_reward = False

    for episode_idx in progress_bar:
        episode_start = time.perf_counter()
        recorder = EpisodeRecorder()
        if vec_normalize_path is None:
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
        else:
            observation = env.reset()

            done = False
            while not done:
                current_observation = observation
                if hasattr(policy, "predict"):
                    action, _ = policy.predict(current_observation, deterministic=True)
                else:
                    action = policy.select_action(current_observation)
                observation, reward, done, infos = env.step(action)
                info = infos[0] if infos else {}
                metrics: StepMetrics = info["metrics"]
                recorder.record_step(
                    observation=np.asarray(current_observation[0]),
                    action=np.asarray(action[0]),
                    reward=float(reward[0]),
                    terminated=bool(done),
                    truncated=bool(info.get("TimeLimit.truncated", False)),
                    metrics=metrics,
                )

        summary = recorder.summary()
        summary.update({"episode": episode_idx + 1})
        summaries.append(summary)

        episode_frame = recorder.to_dataframe()
        episode_frame["episode"] = episode_idx + 1
        step_frames.append(episode_frame)
        if show_progress and isinstance(progress_bar, tqdm):
            episode_duration = time.perf_counter() - episode_start
            progress_bar.set_postfix({"episode_s": f"{episode_duration:.2f}"})

    env.close()
    summary_frame = pd.DataFrame(summaries)
    step_frame = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    return summary_frame, step_frame


def _iterate_with_progress(
        iterable: Iterable[Any],
        total: int | None,
        description: str,
        show_progress: bool,
) -> Iterable[Any]:
    if not show_progress:
        return iterable
    return tqdm(iterable, total=total, desc=description)


def _build_training_callbacks(
        total_timesteps: int,
        show_progress: bool,
) -> BaseCallback:
    reward_callback = EpisodeRewardCallback()
    if not show_progress:
        return reward_callback
    return CallbackList(
        [
            reward_callback,
            TrainingProgressCallback(total_timesteps=total_timesteps),
        ]
    )


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
    degradation_events = float(step_frame["degradation_event"].sum()) if "degradation_event" in step_frame else 0.0
    degradation_amount = float(step_frame["degradation_amount"].sum()) if "degradation_amount" in step_frame else 0.0
    total_steps = float(len(step_frame))

    return {
        "avg_reward": float(step_frame["reward"].mean()),
        "avg_battery_energy": float(step_frame["battery_energy"].mean()),
        "avg_battery_health": float(step_frame["battery_health"].mean()),
        "min_battery_health": float(step_frame["battery_health"].min()),
        "avg_battery_capacity": float(step_frame["battery_capacity"].mean()),
        "solar_to_demand_share": demand_covered / total_demand if total_demand else 0.0,
        "battery_to_demand_share": battery_covered / total_demand if total_demand else 0.0,
        "grid_to_demand_share": grid_covered / total_demand if total_demand else 0.0,
        "solar_sold_share": float(step_frame["solar_sold"].sum() / total_solar) if total_solar else 0.0,
        "avg_grid_to_battery": float(step_frame["grid_to_battery"].mean()),
        "avg_solar_to_battery": float(step_frame["solar_to_battery"].mean()),
        "avg_battery_to_grid": float(step_frame["battery_to_grid"].mean()),
        "action_solar_to_demand_share": float(
            step_frame["solar_to_demand"].sum() / total_steps) if total_steps else 0.0,
        "action_solar_to_battery_share": float(step_frame["solar_to_battery"].sum() / total_steps)
        if total_steps
        else 0.0,
        "action_battery_to_demand_share": float(step_frame["battery_to_demand"].sum() / total_steps)
        if total_steps
        else 0.0,
        "action_battery_to_grid_share": float(
            step_frame["battery_to_grid"].sum() / total_steps) if total_steps else 0.0,
        "action_grid_to_battery_share": float(step_frame["grid_to_battery"].sum() / total_steps)
        if total_steps
        else 0.0,
        "degradation_event_rate": degradation_events / total_steps if total_steps else 0.0,
        "avg_degradation_amount": degradation_amount / total_steps if total_steps else 0.0,
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
            f"- Minimum battery health: {agent_summary.get('min_battery_health', 0.0):.3f}",
            f"- Average battery capacity: {agent_summary.get('avg_battery_capacity', 0.0):.3f}",
            f"- Demand coverage (solar/battery/grid):",
            f"  {agent_summary.get('solar_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('battery_to_demand_share', 0.0):.2%} /",
            f"  {agent_summary.get('grid_to_demand_share', 0.0):.2%}",
            f"- Solar sold share: {agent_summary.get('solar_sold_share', 0.0):.2%}",
            f"- Average grid-to-battery charging: {agent_summary.get('avg_grid_to_battery', 0.0):.2f}",
            f"- Average battery-to-grid selling: {agent_summary.get('avg_battery_to_grid', 0.0):.2f}",
            f"- Degradation event rate: {agent_summary.get('degradation_event_rate', 0.0):.2%}",
            f"- Avg degradation per step: {agent_summary.get('avg_degradation_amount', 0.0):.4f}",
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
    best_vec_normalize_path: Path | None = None

    sweep_iter = _iterate_with_progress(
        enumerate(hyperparameter_grid, start=1),
        total=len(hyperparameter_grid),
        description="Training sweep",
        show_progress=training_config.show_progress,
    )
    for idx, hyperparameters in sweep_iter:
        logger.info("Training configuration %s/%s: %s", idx, len(hyperparameter_grid), hyperparameters)
        training_start = time.perf_counter()
        env = _make_vec_env(
            config=config,
            seed=training_config.seed,
            n_envs=training_config.n_envs,
            monitor_dir=str(run_metadata.root_dir / "data"),
            use_action_masking=training_config.use_action_masking,
            use_vec_normalize=training_config.use_vec_normalize,
            normalize_reward=training_config.normalize_reward,
        )

        callback = _build_training_callbacks(
            total_timesteps=training_config.total_timesteps,
            show_progress=training_config.show_progress,
        )
        model = _build_model(
            algorithm=training_config.algorithm,
            env=env,
            hyperparameters=hyperparameters,
            seed=training_config.seed,
            use_action_masking=training_config.use_action_masking,
            tensorboard_log=run_metadata.root_dir / "tensorboard" if training_config.tensorboard_log else None,
        )
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        training_duration = time.perf_counter() - training_start

        vec_normalize_path = None
        if training_config.use_vec_normalize:
            vec_normalize_path = artifact_manager.save_policy(env, filename=f"vec_normalize_config_{idx}.pkl")
        env.close()

        if isinstance(callback, CallbackList):
            training_frame = callback.callbacks[0].to_dataframe()
        else:
            training_frame = callback.to_dataframe()
        if not training_frame.empty:
            artifact_manager.save_dataframe(
                training_frame,
                filename=f"training_metrics_config_{idx}",
                subdir="data",
            )

        eval_start = time.perf_counter()
        summary_frame, step_frame = _evaluate_policy(
            model,
            config,
            training_config.eval_episodes,
            training_config.seed,
            vec_normalize_path=vec_normalize_path,
            use_action_masking=training_config.use_action_masking,
            show_progress=training_config.show_progress,
        )
        eval_duration = time.perf_counter() - eval_start

        mean_reward = float(summary_frame["total_reward"].mean()) if not summary_frame.empty else -np.inf
        std_reward = float(summary_frame["total_reward"].std(ddof=0)) if not summary_frame.empty else 0.0
        sweep_record = {
            "config_id": idx,
            **hyperparameters,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "training_seconds": training_duration,
            "evaluation_seconds": eval_duration,
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
            best_vec_normalize_path = vec_normalize_path
        logger.info(
            "Config %s done in %.2fs (train=%.2fs, eval=%.2fs).",
            idx,
            training_duration + eval_duration,
            training_duration,
            eval_duration,
        )

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
    if training_config.use_vec_normalize and best_vec_normalize_path is not None:
        target_path = run_metadata.root_dir / "models" / "vec_normalize_best.pkl"
        shutil.copyfile(best_vec_normalize_path, target_path)

    agent_summary_frame, agent_step_frame = _evaluate_policy(
        best_model,
        config,
        training_config.eval_episodes,
        training_config.seed,
        vec_normalize_path=(
            run_metadata.root_dir / "models" / "vec_normalize_best.pkl"
            if training_config.use_vec_normalize
            else None
        ),
        use_action_masking=training_config.use_action_masking,
        show_progress=training_config.show_progress,
    )
    baseline_policy = BaselinePolicy(config)
    # Baseline policy expects raw (unnormalized) observations - no vec_normalize
    baseline_summary_frame, baseline_step_frame = _evaluate_policy(
        baseline_policy,
        config,
        training_config.eval_episodes,
        training_config.seed,
        vec_normalize_path=None,  # Baseline uses absolute price thresholds
        use_action_masking=False,
        show_progress=training_config.show_progress,
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
    hyperparameters = dict(training_config.ppo_hyperparameters) if training_config.ppo_hyperparameters else default_ppo_hyperparameters()

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    config = env_config or EnvironmentConfig()

    # Auto-compute gamma for long-horizon training if not explicitly set
    if "gamma" not in training_config.ppo_hyperparameters:
        episode_length = config.episode_length
        if episode_length > 24:
            computed_gamma = compute_gamma_for_horizon(episode_length, end_discount=0.1)
            hyperparameters["gamma"] = computed_gamma
            logger.info(
                "Long-horizon episode detected (%d steps). Auto-computed gamma=%.6f to ensure "
                "end-of-episode rewards maintain ~10%% value.",
                episode_length,
                computed_gamma,
            )

    artifact_manager.save_config(config, filename="environment_config.json")
    artifact_manager.save_config(training_config, filename="training_config.json")
    artifact_manager.save_json(hyperparameters, filename="ppo_hyperparameters.json")

    logger.info("Starting PPO training with %s generations.", training_config.generations)
    generation_records: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_model: PPO | BaseAlgorithm | None = None
    best_generation = 0
    best_training_frame: pd.DataFrame | None = None
    best_vec_normalize_path: Path | None = None
    best_model_path = run_metadata.root_dir / "models" / "best_ppo_agent.zip"
    best_vecnormalize_path = run_metadata.root_dir / "models" / "vec_normalize_best.pkl"

    generation_iter = _iterate_with_progress(
        range(1, training_config.generations + 1),
        total=training_config.generations,
        description="Training generations",
        show_progress=training_config.show_progress,
    )
    for generation in generation_iter:
        seed = None if training_config.seed is None else training_config.seed + generation
        logger.info("Training generation %s/%s with seed=%s", generation, training_config.generations, seed)
        training_start = time.perf_counter()

        load_vecnormalize = (
                training_config.use_vec_normalize
                and training_config.resume_best_model
                and best_vecnormalize_path.exists()
        )
        env = _make_vec_env(
            config=config,
            seed=seed,
            n_envs=training_config.n_envs,
            monitor_dir=str(run_metadata.root_dir / "data"),
            use_action_masking=training_config.use_action_masking,
            use_vec_normalize=training_config.use_vec_normalize and not load_vecnormalize,
            normalize_reward=training_config.normalize_reward,
        )

        callback = _build_training_callbacks(
            total_timesteps=training_config.total_timesteps,
            show_progress=training_config.show_progress,
        )
        continue_training = False
        model: BaseAlgorithm
        if training_config.resume_best_model and best_model_path.exists():
            if load_vecnormalize:
                env = VecNormalize.load(str(best_vecnormalize_path), env)
                env.training = True
                env.norm_reward = training_config.normalize_reward
            model_class = MaskablePPO if training_config.use_action_masking else PPO
            model = model_class.load(best_model_path, env=env)
            continue_training = True
        else:
            model = _build_model(
                algorithm=training_config.algorithm,
                env=env,
                hyperparameters=hyperparameters,
                seed=seed,
                use_action_masking=training_config.use_action_masking,
                tensorboard_log=run_metadata.root_dir / "tensorboard" if training_config.tensorboard_log else None,
            )

        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback,
            reset_num_timesteps=not continue_training,
            progress_bar=training_config.show_progress,
        )
        training_duration = time.perf_counter() - training_start

        vec_normalize_path = None
        if training_config.use_vec_normalize:
            vec_normalize_path = artifact_manager.save_policy(
                env, filename=f"vec_normalize_generation_{generation}.pkl"
            )
        env.close()

        if isinstance(callback, CallbackList):
            training_frame = callback.callbacks[0].to_dataframe()
        else:
            training_frame = callback.to_dataframe()
        if not training_frame.empty:
            artifact_manager.save_dataframe(
                training_frame,
                filename=f"training_metrics_generation_{generation}",
                subdir="data",
            )

        eval_start = time.perf_counter()
        summary_frame, step_frame = _evaluate_policy(
            model,
            config,
            training_config.eval_episodes,
            seed,
            vec_normalize_path=vec_normalize_path,
            use_action_masking=training_config.use_action_masking,
            show_progress=training_config.show_progress,
        )
        eval_duration = time.perf_counter() - eval_start

        mean_reward = float(summary_frame["total_reward"].mean()) if not summary_frame.empty else -np.inf
        std_reward = float(summary_frame["total_reward"].std(ddof=0)) if not summary_frame.empty else 0.0

        generation_records.append(
            {
                "generation": generation,
                "seed": seed,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "training_seconds": training_duration,
                "evaluation_seconds": eval_duration,
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
            best_vec_normalize_path = vec_normalize_path
            artifact_manager.save_policy(best_model, filename="best_ppo_agent.zip")
            if best_vec_normalize_path is not None:
                shutil.copyfile(best_vec_normalize_path, best_vecnormalize_path)
        logger.info(
            "Generation %s done in %.2fs (train=%.2fs, eval=%.2fs).",
            generation,
            training_duration + eval_duration,
            training_duration,
            eval_duration,
        )

    sweep_frame = pd.DataFrame(generation_records)
    artifact_manager.save_dataframe(sweep_frame, filename="generation_summary", subdir="data")

    plotter = RLPlotter()
    if best_training_frame is not None:
        plotter.plot_training_metrics(best_training_frame, output_dir=run_metadata.root_dir / "plots")

    if not sweep_frame.empty:
        plotter.plot_generation_summary(sweep_frame, output_dir=run_metadata.root_dir / "plots")

    if best_model is None:
        raise RuntimeError("No valid model was trained during the generation loop.")

    if not best_model_path.exists():
        artifact_manager.save_policy(best_model, filename="best_ppo_agent.zip")
    if training_config.use_vec_normalize and best_vec_normalize_path is not None:
        shutil.copyfile(best_vec_normalize_path, best_vecnormalize_path)

    agent_summary_frame, agent_step_frame = _evaluate_policy(
        best_model,
        config,
        training_config.eval_episodes,
        training_config.seed,
        vec_normalize_path=best_vecnormalize_path if training_config.use_vec_normalize else None,
        use_action_masking=training_config.use_action_masking,
        show_progress=training_config.show_progress,
    )
    baseline_policy = BaselinePolicy(config)
    # IMPORTANT: Baseline policy expects raw (unnormalized) observations, so we do NOT
    # pass vec_normalize_path here. The baseline policy uses absolute values like
    # buying_price thresholds, which would break with normalized observations.
    baseline_summary_frame, baseline_step_frame = _evaluate_policy(
        baseline_policy,
        config,
        training_config.eval_episodes,
        training_config.seed,
        vec_normalize_path=None,  # Baseline expects raw observations
        use_action_masking=False,  # Baseline doesn't use action masking
        show_progress=training_config.show_progress,
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
