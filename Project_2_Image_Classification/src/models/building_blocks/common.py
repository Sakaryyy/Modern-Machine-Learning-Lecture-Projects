"""Common utilities for neural network building blocks."""

from __future__ import annotations

from typing import Callable, Dict

import jax.numpy as jnp
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]

_ACTIVATION_REGISTRY: Dict[str, ActivationFn] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.swish,
    "elu": nn.elu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
}


def resolve_activation(name: str) -> ActivationFn:
    """Return the activation function associated with ``name``.

    Parameters
    ----------
    name:
        Identifier of the activation function.  The value is
        matched case-insensitively against a curated registry of common
        activations.

    Returns
    -------
    Callable[[jax.numpy.ndarray], jax.numpy.ndarray]
        Activation function that can be applied to tensors produced by JAX.

    Raises
    ------
    ValueError
        If ``name`` is not known to the registry.
    """

    try:
        return _ACTIVATION_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation function '{name}'.") from exc


__all__ = [
    "ActivationFn",
    "resolve_activation",
]
