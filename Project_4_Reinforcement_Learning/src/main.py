"""Command line interface for Project 4 reinforcement learning environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from Project_4_Reinforcement_Learning.src.config.env_config import EnvironmentConfig
from Project_4_Reinforcement_Learning.src.diagnostics.interactive_checks import run_interactive_diagnostics
from Project_4_Reinforcement_Learning.src.environment.energy_budget_env import EnergyBudgetEnv
from Project_4_Reinforcement_Learning.src.logger import (
    LoggingManager,
    LoggingConfig,
    get_logger
)
from Project_4_Reinforcement_Learning.src.policies.baseline_policy import BaselinePolicy
from Project_4_Reinforcement_Learning.src.training import (
    AgentTrainingConfig,
    RECOMMENDED_ALGORITHM,
    HyperparameterSearchConfig,
    long_horizon_ppo_hyperparameters,
    run_training_sweep,
    run_training,
    focused_hyperparameter_grid,
)
from Project_4_Reinforcement_Learning.src.utils import (
    EpisodeRecorder,
    RLPlotter,
    RunArtifactManager
)


def _simulate_baseline(
        steps: int,
        seed: int | None,
        output_dir: Path,
        run_name: str | None,
        config: EnvironmentConfig | None = None,
) -> None:
    """Simulate the environment using the baseline policy.

    Parameters
    ----------
    steps:
        Number of steps to simulate.
    seed:
        Optional random seed.
    output_dir:
        Root directory where runs should be saved.
    run_name:
        Optional descriptive name for the run folder.
    config:
        Optional environment configuration override.
    """

    logger = get_logger("Simulation")
    env_config = config or EnvironmentConfig()
    env = EnergyBudgetEnv(env_config)
    policy = BaselinePolicy(env_config)

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    artifact_manager.save_config(env_config, filename="environment_config.json")

    recorder = EpisodeRecorder()

    observation, info = env.reset(seed=seed)
    logger.info("Initial state: %s", info)

    total_reward = 0.0
    step_idx = None
    for step_idx in range(steps):
        action = policy.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        metrics = info["metrics"]
        recorder.record_step(
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )
        logger.info(
            "Step %s | reward=%.3f | battery=%.2f | capacity=%.2f | health=%.2f | sold=%s | bought=%s",
            step_idx,
            reward,
            metrics.battery_energy,
            metrics.battery_capacity,
            metrics.battery_health,
            metrics.solar_sold,
            metrics.grid_to_battery + metrics.grid_to_demand,
        )

        observation = next_observation
        if terminated or truncated:
            break

    logger.info("Total reward after %s steps: %.3f", step_idx + 1, total_reward)

    step_frame = recorder.to_dataframe()
    summary = recorder.summary()
    summary.update(
        {
            "run_id": run_metadata.run_id,
            "run_created_at": run_metadata.created_at,
            "seed": seed,
            "num_steps": step_idx + 1,
        }
    )
    artifact_manager.save_json(summary, filename="run_summary.json", subdir="data")
    artifact_manager.save_dataframe(step_frame, filename="episode_steps", subdir="data")
    artifact_manager.save_dataframe(
        step_frame,
        filename="environment_state",
        subdir="environment_state",
    )
    artifact_manager.save_policy(policy, filename="baseline_policy.pkl")

    plotter = RLPlotter()
    plotter.plot_episode_metrics(step_frame, output_dir=run_metadata.root_dir / "plots")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Reinforcement learning project utilities.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "diagnostics", "train"],
        default="baseline",
        help="Execution mode.",
    )
    parser.add_argument("--steps", type=int, default=120, help="Number of steps to run (default: 120 for 5-day episodes).")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=120,
        help="Episode length in hours (default: 120 = 5 days for varied conditions).",
    )
    parser.add_argument(
        "--long-horizon-days",
        type=int,
        default=None,
        help="Optional number of days for a long-horizon episode (overrides --episode-length).",
    )
    parser.add_argument(
        "--persist-battery-state",
        action="store_true",
        help="Persist battery energy/health across episode resets.",
    )
    parser.add_argument(
        "--disable-randomize-start-day",
        action="store_false",
        dest="randomize_start_day",
        help="Disable randomizing the start day-of-year on reset.",
    )
    parser.add_argument(
        "--year-length-days",
        type=int,
        default=365,
        help="Number of days in the seasonal cycle.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where run artifacts are saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name to append to the run folder.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause after each step in diagnostics mode.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps for RL mode (default: 100k for single-day training).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=10,
        help="Number of parallel environments for RL training.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Evaluation episodes for each configuration.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=RECOMMENDED_ALGORITHM,
        help="Stable Baselines3 algorithm identifier.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of independent training generations (default: 5 for single-day training).",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of hyperparameter configurations.",
    )
    parser.add_argument(
        "--hyperparameter-search-enabled",
        type=bool,
        default=False,
        help="Whether to use hyperparameter search enabled.",
    )
    parser.add_argument(
        "--disable-action-masking",
        action="store_false",
        dest="use_action_masking",
        help="Disable MaskablePPO action masking.",
    )
    parser.add_argument(
        "--disable-vec-normalize",
        action="store_false",
        dest="use_vec_normalize",
        help="Disable VecNormalize for observations and rewards.",
    )
    parser.add_argument(
        "--disable-resume-best-model",
        action="store_false",
        dest="resume_best_model",
        help="Disable resuming from the best model across generations.",
    )
    parser.add_argument(
        "--disable-tensorboard-log",
        action="store_false",
        dest="tensorboard_log",
        help="Disable TensorBoard logging for training runs.",
    )
    parser.set_defaults(
        randomize_start_day=True,
        use_action_masking=True,
        use_vec_normalize=True,
        resume_best_model=True,
        tensorboard_log=True,
        persist_battery_state=False,  # False for single-day training (each episode is independent)
    )
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> EnvironmentConfig:
    """Construct an environment configuration from CLI arguments."""

    episode_length = args.episode_length
    if args.long_horizon_days is not None:
        episode_length = max(1, args.long_horizon_days) * 24
    persist_battery_state = args.persist_battery_state or (args.long_horizon_days or 0) > 1
    return EnvironmentConfig(
        episode_length=episode_length,
        persist_battery_state=persist_battery_state,
        randomize_start_day=args.randomize_start_day,
        year_length_days=args.year_length_days,
    )


def main() -> None:
    """Run the selected project mode."""

    config = LoggingConfig()
    LoggingManager(config).configure()

    args = parse_args()
    logger = get_logger("Main")
    logger.info("Starting Project 4 with mode '%s'", args.mode)

    env_config = _build_env_config(args)

    if args.mode == "baseline":
        _simulate_baseline(
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            run_name=args.run_name,
            config=env_config,
        )
    elif args.mode == "diagnostics":
        run_interactive_diagnostics(
            config=env_config,
            num_steps=args.steps,
            interactive=args.interactive,
            seed=args.seed,
        )
    else:
        # Use optimized hyperparameters for long-horizon training
        ppo_hyperparameters = {}
        if args.long_horizon_days is not None and args.long_horizon_days > 1:
            episode_length = args.long_horizon_days * 24
            ppo_hyperparameters = long_horizon_ppo_hyperparameters(episode_length)
            logger.info(
                "Using long-horizon PPO hyperparameters for %d-day episodes (gamma=%.6f)",
                args.long_horizon_days,
                ppo_hyperparameters["gamma"],
            )

        training_config = AgentTrainingConfig(
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            algorithm=args.algorithm,
            generations=args.generations,
            ppo_hyperparameters=ppo_hyperparameters,
            use_action_masking=args.use_action_masking,
            use_vec_normalize=args.use_vec_normalize,
            resume_best_model=args.resume_best_model,
            tensorboard_log=args.tensorboard_log,
        )
        if args.hyperparameter_search_enabled:
            search_config = HyperparameterSearchConfig(
                grid=focused_hyperparameter_grid(),
                max_configs=args.max_configs
            )
            run_training_sweep(
                output_dir=args.output_dir,
                run_name=args.run_name,
                training_config=training_config,
                search_config=search_config,
                env_config=env_config,
            )
        else:
            run_training(
                output_dir=args.output_dir,
                run_name=args.run_name,
                training_config=training_config,
                env_config=env_config,
            )


if __name__ == "__main__":
    main()
