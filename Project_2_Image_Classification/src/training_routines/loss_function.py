"""Loss functions used during model optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.nn as jnn
import jax.numpy as jnp
import optax

LossFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

__all__ = [
    "LossConfig",
    "LossFunction",
    "resolve_loss_function",
]


@dataclass(slots=True)
class LossConfig:
    """Configuration describing how the training loss should be computed.

    Parameters
    ----------
    name:
        Identifier of the loss function.  Supported values are
        ``"cross_entropy"`` and ``"mse"``.
    label_smoothing:
        Optional amount of label smoothing applied to one-hot encoded targets
        when using the cross-entropy loss.  The value must lie within
        ``[0, 1)``.
    """

    name: str = "cross_entropy"
    label_smoothing: float = 0.05

    def __post_init__(self) -> None:
        if self.label_smoothing < 0.0 or self.label_smoothing >= 1.0:
            raise ValueError("'label_smoothing' must lie within the interval [0, 1).")


def resolve_loss_function(config: LossConfig) -> LossFunction:
    """Return the callable implementing the configured loss function."""

    normalized_name = config.name.lower()
    if normalized_name in {"cross_entropy", "categorical_cross_entropy"}:
        return lambda logits, labels: _softmax_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=config.label_smoothing,
        )
    if normalized_name in {"mse", "mean_squared_error"}:
        return _mean_squared_error_loss

    raise ValueError(f"Unsupported loss function '{config.name}'.")


def _softmax_cross_entropy_loss(
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        label_smoothing: float = 0.0,
) -> jnp.ndarray:
    """Cross-entropy between ``logits`` and ``labels`` with optional smoothing."""

    num_classes = logits.shape[-1]
    one_hot = _to_one_hot(labels, num_classes)
    if label_smoothing > 0.0:
        smoothing = jnp.asarray(label_smoothing, dtype=logits.dtype)
        one_hot = (1.0 - smoothing) * one_hot + smoothing / num_classes

    losses = optax.softmax_cross_entropy(logits, one_hot)
    return jnp.mean(losses)


def _mean_squared_error_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Return the mean squared error between ``logits`` and ``labels``."""

    predictions = jnn.softmax(logits)
    squared_error = jnp.square(predictions - labels)
    return jnp.mean(squared_error)


def _to_one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Convert ``labels`` to one-hot encodings when necessary."""

    if labels.ndim == 1:
        return jnn.one_hot(labels, num_classes)
    return labels
