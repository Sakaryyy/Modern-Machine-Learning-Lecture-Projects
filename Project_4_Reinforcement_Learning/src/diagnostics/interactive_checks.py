"""Interactive diagnostics for the energy budgeting environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, List

from Project_4_Reinforcement_Learning.src.config.env_config import EnvironmentConfig
from Project_4_Reinforcement_Learning.src.environment.energy_budget_env import EnergyBudgetEnv, StepMetrics
from Project_4_Reinforcement_Learning.src.policies.baseline_policy import BaselinePolicy
from Project_4_Reinforcement_Learning.src.utils.logging import get_logger


def _validate_metrics(metrics: StepMetrics, config: EnvironmentConfig) -> List[str]:
    """Validate environment invariants.

    Parameters
    ----------
    metrics:
        Step metrics emitted by the environment.
    config:
        Environment configuration for bound checks.

    Returns
    -------
    list[str]
        Collection of validation issues, empty when all checks pass.
    """

    issues: List[str] = []

    if not (0 <= metrics.battery_energy <= config.max_battery_energy):
        issues.append("Battery energy outside valid bounds.")

    solar_balance = (
            metrics.solar_to_demand + metrics.solar_to_battery + metrics.solar_sold
    )
    if solar_balance != metrics.solar_production:
        issues.append("Solar energy allocation does not conserve production.")

    demand_coverage = metrics.solar_to_demand + metrics.battery_to_demand + metrics.grid_to_demand
    if demand_coverage < metrics.demand:
        issues.append("Demand not fully covered by energy sources.")

    if metrics.grid_to_battery < 0 or metrics.grid_to_demand < 0:
        issues.append("Grid purchase values must be non-negative.")

    if metrics.solar_sold < 0:
        issues.append("Solar sales must be non-negative.")

    return issues


def run_interactive_diagnostics(
        config: EnvironmentConfig,
        num_steps: int,
        interactive: bool,
        seed: int | None,
) -> Dict[str, int]:
    """Run an interactive diagnostic session.

    Parameters
    ----------
    config:
        Environment configuration.
    num_steps:
        Number of steps to simulate.
    interactive:
        If ``True`` wait for user input after each step.
    seed:
        Optional random seed.

    Returns
    -------
    dict
        Summary counts for passed/failed checks.
    """

    logger = get_logger("Diagnostics")
    env = EnergyBudgetEnv(config)
    policy = BaselinePolicy(config)

    observation, info = env.reset(seed=seed)
    logger.info("Initial state: %s", info)

    passed_checks = 0
    failed_checks = 0

    for step_idx in range(num_steps):
        action = policy.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics: StepMetrics = info["metrics"]
        issues = _validate_metrics(metrics, config)

        logger.info(
            "Step %s | reward=%.3f | issues=%s",
            step_idx,
            reward,
            "none" if not issues else issues,
        )
        logger.debug("Metrics: %s", asdict(metrics))

        if issues:
            failed_checks += 1
        else:
            passed_checks += 1

        if interactive:
            input("Press Enter to continue to the next step...")

        if terminated or truncated:
            logger.info("Episode ended (terminated=%s, truncated=%s).", terminated, truncated)
            break

    summary = {"passed": passed_checks, "failed": failed_checks}
    logger.info("Diagnostics summary: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for diagnostics."""

    parser = argparse.ArgumentParser(description="Run interactive environment diagnostics.")
    parser.add_argument("--steps", type=int, default=24, help="Number of steps to simulate.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause after each step for manual inspection.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Entry point for diagnostics."""

    args = parse_args()
    config = EnvironmentConfig()
    run_interactive_diagnostics(
        config=config,
        num_steps=args.steps,
        interactive=args.interactive,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
