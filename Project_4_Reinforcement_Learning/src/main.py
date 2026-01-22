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
from Project_4_Reinforcement_Learning.src.utils import (
    EpisodeRecorder,
    RLPlotter,
    RunArtifactManager
)


def _simulate_baseline(steps: int, seed: int | None, output_dir: Path, run_name: str | None) -> None:
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
    """

    logger = get_logger("Simulation")
    config = EnvironmentConfig()
    env = EnergyBudgetEnv(config)
    policy = BaselinePolicy(config)

    artifact_manager = RunArtifactManager(output_dir, run_name=run_name)
    run_metadata = artifact_manager.initialize()
    artifact_manager.save_config(config, filename="environment_config.json")

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
            "Step %s | reward=%.3f | battery=%s | sold=%s | bought=%s",
            step_idx,
            reward,
            metrics.battery_energy,
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
        choices=["baseline", "diagnostics"],
        default="baseline",
        help="Execution mode.",
    )
    parser.add_argument("--steps", type=int, default=24, help="Number of steps to run.")
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
    return parser.parse_args()


def main() -> None:
    """Run the selected project mode."""

    config = LoggingConfig()
    LoggingManager(config).configure()

    args = parse_args()
    logger = get_logger("Main")
    logger.info("Starting Project 4 with mode '%s'", args.mode)

    if args.mode == "baseline":
        _simulate_baseline(steps=args.steps, seed=args.seed, output_dir=args.output_dir, run_name=args.run_name)
    else:
        run_interactive_diagnostics(
            config=EnvironmentConfig(),
            num_steps=args.steps,
            interactive=args.interactive,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
