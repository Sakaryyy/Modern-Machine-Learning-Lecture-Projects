from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

Array = jax.Array
YMode = Literal["none", "log1p", "sqrt"]


def forward_transform(y: Array, mode: YMode) -> Array:
    """
    Transform the target for regression.

    Parameters
    ----------
    y : jax.Array
        Non-negative counts as a vector.
    mode : {"none", "log1p", "sqrt"}
        - "none": return y unchanged.
        - "log1p": return log(1 + y).
        - "sqrt": return sqrt(y).

    Returns
    -------
    jax.Array
        Transformed target.
    """
    if mode == "none":
        return y
    if mode == "log1p":
        return jnp.log1p(jnp.maximum(y, 0.0))
    if mode == "sqrt":
        return jnp.sqrt(jnp.maximum(y, 0.0))
    raise ValueError(f"Unknown mode: {mode}")


def inverse_transform(y_hat: Array, mode: YMode, smear: float | None = None) -> Array:
    """
    Inverse of `forward_transform`.

    Parameters
    ----------
    y_hat : jax.Array
        Predictions in transformed space.
    mode : {"none", "log1p", "sqrt"}
    smear : float or None
        Optional smearing factor for "log1p". If provided, multiplies exp(.) - 1.

    Returns
    -------
    jax.Array
        Predictions on the original count scale.
    """
    if mode == "none":
        return y_hat
    if mode == "log1p":
        base = jnp.expm1(y_hat)
        return base * (smear if smear is not None else 1.0)
    if mode == "sqrt":
        return jnp.square(jnp.maximum(y_hat, 0.0))
    raise ValueError(f"Unknown mode: {mode}")


def smearing_factor(residuals_in_log_space: Array) -> float:
    """
    Smearing estimate for log-space regressions.

    If z = log(1 + y), and z_hat are predictions, residuals e = z - z_hat.
    The unbiased back-transform for E[y | x] is approx E[exp(z_hat + e) - 1]
    = (exp(z_hat) * E[exp(e)]) - 1. E[exp(e)] is estimated by the mean of exp(e).

    Parameters
    ----------
    residuals_in_log_space : jax.Array
        Residuals z - z_hat.

    Returns
    -------
    float
        Estimated E[exp(e)].
    """
    return float(jnp.mean(jnp.exp(residuals_in_log_space)))
