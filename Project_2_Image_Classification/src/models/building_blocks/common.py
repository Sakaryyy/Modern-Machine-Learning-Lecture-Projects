"""Common utilities for neural network building blocks."""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers as flax_initializers

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
InitializerFn = Callable[[jax.Array, tuple[int, ...], jnp.dtype], jnp.ndarray]

_ACTIVATION_REGISTRY: Dict[str, ActivationFn] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.swish,
    "elu": nn.elu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
}


def _build_initializer_registry() -> Dict[str, InitializerFn]:
    """Create the registry mapping textual identifiers to initializers."""

    return {
        "lecun_normal": flax_initializers.lecun_normal(),
        "lecun_uniform": flax_initializers.lecun_uniform(),
        "xavier_normal": flax_initializers.xavier_normal(),
        "xavier_uniform": flax_initializers.xavier_uniform(),
        "kaiming_normal": flax_initializers.variance_scaling(2.0, "fan_in", "truncated_normal"),
        "kaiming_uniform": flax_initializers.variance_scaling(2.0, "fan_in", "uniform"),
        "he_normal": flax_initializers.variance_scaling(2.0, "fan_in", "truncated_normal"),
        "he_uniform": flax_initializers.variance_scaling(2.0, "fan_in", "uniform"),
        "orthogonal": flax_initializers.orthogonal(),
        "normal": flax_initializers.normal(),
        "truncated_normal": flax_initializers.truncated_normal(),
        "uniform": flax_initializers.uniform(),
        "zeros": flax_initializers.zeros,
        "ones": flax_initializers.ones,
    }


_INITIALIZER_REGISTRY: Dict[str, InitializerFn] = _build_initializer_registry()


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


def resolve_initializer(initializer: str | InitializerFn) -> InitializerFn:
    """Return a Flax-compatible initializer based on ``initializer``.

    Parameters
    ----------
    initializer:
        Either a callable that already implements the initializer protocol or a
        string identifying a preconfigured initializer.  Recognised strings
        include ``"he_normal"``, ``"lecun_normal"``, ``"xavier_uniform"`` and
        ``"orthogonal"`` among others.

    Returns
    -------
    Callable[[jax.random.KeyArray, tuple[int, ...], jax.numpy.dtype], jax.numpy.ndarray]
        Initializer function compatible with Flax layers.

    Raises
    ------
    ValueError
        If a string identifier is provided that is not registered.
    """

    if callable(initializer):
        return initializer

    key = initializer.lower()
    try:
        return _INITIALIZER_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown initializer '{initializer}'.") from exc


__all__ = [
    "ActivationFn",
    "InitializerFn",
    "resolve_activation",
    "resolve_initializer",
]
