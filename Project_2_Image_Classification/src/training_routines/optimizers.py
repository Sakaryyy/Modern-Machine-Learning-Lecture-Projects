"""Factory functions creating Optax optimisers for the trainer."""

from __future__ import annotations

from dataclasses import dataclass

import optax

from Project_2_Image_Classification.src.training_routines.learning_rate_schedulers import Schedule

__all__ = [
    "OptimizerConfig",
    "create_optimizer",
]


@dataclass(slots=True)
class OptimizerConfig:
    """Describe which optimiser should be used during training.

    Parameters
    ----------
    name:
        Identifier of the optimiser.  Supported values are ``"adamw"``,
        ``"adam"``, ``"sgd"`` and ``"rmsprop"``.
    weight_decay:
        Strength of decoupled weight decay for optimisers that support it.
    momentum:
        Momentum factor used by SGD and RMSProp variants.
    nesterov:
        Whether Nesterov momentum should be employed when using SGD.
    beta1:
        First momentum coefficient used by Adam-style optimisers.
    beta2:
        Second momentum coefficient used by Adam-style optimisers.
    eps:
        Numerical stability constant.
    centered:
        Whether RMSProp should maintain centered gradients.
    """

    name: str = "adamw"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    centered: bool = False

    def __post_init__(self) -> None:
        if self.weight_decay < 0.0:
            raise ValueError("'weight_decay' must be non-negative.")
        if self.momentum < 0.0:
            raise ValueError("'momentum' must be non-negative.")
        if not 0.0 < self.beta1 < 1.0:
            raise ValueError("'beta1' must lie in (0, 1).")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError("'beta2' must lie in (0, 1).")
        if self.eps <= 0.0:
            raise ValueError("'eps' must be strictly positive.")


def create_optimizer(config: OptimizerConfig, schedule: Schedule) -> optax.GradientTransformation:
    """Return an Optax optimiser according to ``config``."""

    name = config.name.lower()
    if name == "adamw":
        return optax.adamw(
            learning_rate=schedule,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if name == "adam":
        return optax.adam(
            learning_rate=schedule,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    if name == "sgd":
        return optax.sgd(
            learning_rate=schedule,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    if name == "rmsprop":
        return optax.rmsprop(
            learning_rate=schedule,
            momentum=config.momentum,
            decay=0.9,
            eps=config.eps,
            centered=config.centered,
        )

    raise ValueError(f"Unsupported optimizer '{config.name}'.")
