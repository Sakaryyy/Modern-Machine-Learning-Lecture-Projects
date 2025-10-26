from __future__ import annotations

from typing import Tuple, Optional

import jax
from jax import numpy as jnp

Array = jax.Array


def standardize_design(
    X_tr: Array,
        X_va: Array,
    X_te: Array,
    *,
        std_floor: float = 1e-6,
) -> Tuple[Array, Array, Array, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standardize non-binary columns by training mean/std and leave binary columns unchanged.

    Binary detection heuristic
    --------------------------
    A column is treated as binary if every element is either 0 or 1 in X_tr.
    This is robust for one-hot indicators.

    Parameters
    ----------
    X_tr, X_va, X_te : jax.Array
        Design matrices with identical number of columns.
    std_floor : float
        Numerical floor added to std to avoid division by zero.

    Returns
    -------
    Xz_tr, Xz_val, Xz_te : jax.Array
        Standardized designs.
    mu, sd, is_binary : jax.Array
        Training means, training stds (with eps), and a boolean mask indicating
        which columns were preserved (either because they were detected as
        binary or because they were supplied via `preserve_mask`).
    """
    if X_tr.shape[1] == 0:
        m = X_tr.shape[1]
        zero = jnp.zeros((m,), dtype=X_tr.dtype)
        return X_tr, X_va, X_te, zero, jnp.ones((m,), dtype=X_tr.dtype), jnp.ones((m,), dtype=bool)

    mu = jnp.mean(X_tr, axis=0)
    sd = jnp.std(X_tr, axis=0, ddof=0)

    # Identify binary columns on train split
    minv = jnp.min(X_tr, axis=0)
    maxv = jnp.max(X_tr, axis=0)
    is_binary = jnp.logical_and(minv >= 0.0,
                                jnp.logical_and(maxv <= 1.0, jnp.all(jnp.isin(X_tr, jnp.array([0.0, 1.0])), axis=0)))

    # Near-constant columns (could be artifacts)
    near_const = sd < std_floor

    scale = jnp.where(jnp.logical_or(is_binary, near_const), 1.0, jnp.maximum(sd, std_floor))

    def _z(x): return (x - mu) / scale

    X_tr_s = _z(X_tr)
    X_va_s = _z(X_va)
    X_te_s = _z(X_te)

    return X_tr_s, X_va_s, X_te_s, mu, sd, is_binary


def can_select_group(selected_cols: set[str], candidate_group: list[str]) -> bool:
    """
    Enforce hierarchical selection for interaction groups.

    Rules:
    - For atemp:hr_* interactions:
        require both 'atemp' (the linear main effect) AND at least one 'hr_*'
        feature to already be present among selected_cols.

    - For all other groups: no constraint.

    Notes
    -----
    We intentionally require the *linear* 'atemp' column to exist, even if
    polynomial terms like 'atemp^2' are present. This enforces a standard
    hierarchical principle: lower-order main effects accompany higher-order
    interactions.
    """
    # Detect if candidate is the atemp x hour-onehot interaction block
    is_atemp_hr_interaction = all(name.startswith("atemp:hr_") for name in candidate_group)
    if not is_atemp_hr_interaction:
        return True

    has_atemp_main = "atemp" in selected_cols
    has_hour_onehot = any(name.startswith("hr_") for name in selected_cols)

    return has_atemp_main and has_hour_onehot


def to_device_array(
    values,
    *,
    dtype: jnp.dtype,
    device: Optional[jax.Device],
    check_finite: bool,
) -> Array:
    """
    Convert input values to a JAX device array with the requested dtype and device.

    Parameters
    ----------
    values : array-like
        Input values, typically a NumPy ndarray produced by pandas.
    dtype : jax.numpy dtype
        Desired dtype, for example jnp.float32 or jnp.float64.
    device : jax.Device or None
        Target device. If None, JAX uses the default device.
    check_finite : bool
        If True, raise ValueError when NaN or infinite values are detected.

    Returns
    -------
    jax.Array
        Array on the requested device and dtype.
    """
    x = jnp.asarray(values, dtype=dtype)
    if check_finite:
        if not bool(jnp.isfinite(x).all()):
            raise ValueError("Non-finite values detected in feature matrix.")
    return jax.device_put(x, device=device) if device is not None else x
