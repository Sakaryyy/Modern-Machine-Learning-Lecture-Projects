"""Learning-rate schedule utilities for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import optax

Schedule = Callable[[int], float]

__all__ = [
    "LRSchedulerConfig",
    "Schedule",
    "create_learning_rate_schedule",
]


@dataclass(slots=True)
class LRSchedulerConfig:
    """Describe the learning-rate schedule applied during optimisation.

    Parameters
    ----------
    name:
        Identifier of the scheduler.  Supported values are ``"constant"``,
        ``"cosine_decay"``, ``"linear_warmup_cosine_decay"`` and
        ``"exponential_decay"``.
    learning_rate:
        Target learning rate reached after the warmup phase.
    warmup_steps:
        Number of optimisation steps spent in the warmup regime.
    warmup_init_value:
        Initial learning rate used at the beginning of warmup.
    decay_rate:
        Multiplicative factor applied during exponential decay.
    transition_steps:
        Step interval controlling how often the exponential scheduler decays.
        When omitted, the value is inferred from the total training steps.
    alpha:
        Final-to-initial ratio used by the cosine decay schedule.  A value of
        ``0.0`` decays all the way to zero.
    end_learning_rate:
        Final learning rate reached by the ``linear_warmup_cosine_decay``
        schedule.  Defaults to zero.
    """

    name: str = "constant"
    learning_rate: float = 1e-3
    warmup_steps: int = 0
    warmup_init_value: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int | None = None
    alpha: float = 0.0
    end_learning_rate: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("'learning_rate' must be strictly positive.")
        if self.warmup_steps < 0:
            raise ValueError("'warmup_steps' must be non-negative.")
        if self.transition_steps is not None and self.transition_steps <= 0:
            raise ValueError("'transition_steps' must be positive when provided.")
        if self.decay_rate <= 0.0:
            raise ValueError("'decay_rate' must be strictly positive.")
        if self.alpha < 0.0:
            raise ValueError("'alpha' must be non-negative.")


def create_learning_rate_schedule(
        config: LRSchedulerConfig,
        *,
        total_steps: int,
) -> Schedule:
    """Instantiate the schedule described by ``config``."""

    if total_steps <= 0:
        raise ValueError("'total_steps' must be strictly positive.")

    name = config.name.lower()
    if name == "constant":
        schedule = optax.constant_schedule(config.learning_rate)
        return _with_optional_warmup(schedule, config, total_steps)

    if name == "cosine_decay":
        decay_steps = max(1, total_steps - config.warmup_steps)
        cosine = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=decay_steps,
            alpha=config.alpha,
        )
        return _with_optional_warmup(cosine, config, total_steps)

    if name == "linear_warmup_cosine_decay":
        decay_steps = abs(max(1, total_steps - config.warmup_steps))
        if decay_steps - config.warmup_steps < 0:
            config.warmup_steps = config.warmup_steps // 2
        end_value = config.end_learning_rate if config.end_learning_rate is not None else 0.0
        return optax.warmup_cosine_decay_schedule(
            init_value=config.warmup_init_value,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )

    if name == "exponential_decay":
        transition_steps = config.transition_steps or max(1, total_steps // 10)
        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=transition_steps,
            decay_rate=config.decay_rate,
            staircase=False,
        )
        return _with_optional_warmup(schedule, config, total_steps)

    raise ValueError(f"Unknown learning-rate scheduler '{config.name}'.")


def _with_optional_warmup(schedule: Schedule, config: LRSchedulerConfig, total_steps: int) -> Schedule:
    """Wrap ``schedule`` with a linear warmup if requested in ``config``."""

    if config.warmup_steps <= 0:
        return schedule

    warmup = optax.linear_schedule(
        init_value=config.warmup_init_value,
        end_value=config.learning_rate,
        transition_steps=min(config.warmup_steps, total_steps),
    )
    return optax.join_schedules(
        schedules=(warmup, schedule),
        boundaries=(config.warmup_steps,),
    )
