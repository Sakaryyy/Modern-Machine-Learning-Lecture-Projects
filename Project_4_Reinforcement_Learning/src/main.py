"""Command line interface for Project 4 reinforcement learning environment."""

from __future__ import annotations

import argparse

from Project_4_Reinforcement_Learning.src.config.env_config import EnvironmentConfig
from Project_4_Reinforcement_Learning.src.diagnostics.interactive_checks import run_interactive_diagnostics
from Project_4_Reinforcement_Learning.src.environment.energy_budget_env import EnergyBudgetEnv
from Project_4_Reinforcement_Learning.src.policies.baseline_policy import BaselinePolicy
from Project_4_Reinforcement_Learning.src.utils.logging import LoggingConfig, LoggingManager, get_logger


def _simulate_baseline(steps: int, seed: int | None) -> None:
    """Simulate the environment using the baseline policy.

    Parameters
    ----------
    steps:
        Number of steps to simulate.
    seed:
        Optional random seed.
    """

    logger = get_logger("Simulation")
    config = EnvironmentConfig()
    env = EnergyBudgetEnv(config)
    policy = BaselinePolicy(config)

    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    step_idx = None

    for step_idx in range(steps):
        action = policy.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        metrics = info["metrics"]
        logger.info(
            "Step %s | reward=%.3f | battery=%s | sold=%s | bought=%s",
            step_idx,
            reward,
            metrics.battery_energy,
            metrics.solar_sold,
            metrics.grid_to_battery + metrics.grid_to_demand,
        )

        if terminated or truncated:
            break

    logger.info("Total reward after %s steps: %.3f", step_idx + 1, total_reward)


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
        _simulate_baseline(steps=args.steps, seed=args.seed)
    else:
        run_interactive_diagnostics(
            config=EnvironmentConfig(),
            num_steps=args.steps,
            interactive=args.interactive,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
