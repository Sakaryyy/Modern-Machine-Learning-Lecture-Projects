from __future__ import annotations

from typing import Tuple, Optional

import jax
from jax import numpy as jnp

Array = jax.Array


def standardize_design(
    X_tr: Array,
    X_val: Array,
    X_te: Array,
    *,
    eps: float = 1e-8,
    preserve_mask: Optional[Array] = None,
) -> Tuple[Array, Array, Array, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standardize non-binary columns by training mean/std and leave binary columns unchanged.

    Binary detection heuristic
    --------------------------
    A column is treated as binary if every element is either 0 or 1 in X_tr.
    This is robust for one-hot indicators.

    Parameters
    ----------
    X_tr, X_val, X_te : jax.Array
        Design matrices with identical number of columns.
    eps : float
        Numerical floor added to std to avoid division by zero.
    preserve_mask : jax.Array of bool, optional
        Boolean mask with shape (n_features,) marking columns that should remain
        untouched (mean 0, std 1) during scaling. This is useful for columns that
        have already been normalized to lie within a specific range (like [0, 1])
        and should not be z-scored. When provided, the mask is combined with the
        automatically detected binary mask so that any column marked True is left
        unchanged.

    Returns
    -------
    Xz_tr, Xz_val, Xz_te : jax.Array
        Standardized designs.
    mu, sd, preserved : jax.Array
        Training means, training stds (with eps), and a boolean mask indicating
        which columns were preserved (either because they were detected as
        binary or because they were supplied via `preserve_mask`).
    """
    if X_tr.shape[1] == 0:
        zero = jnp.array([], dtype=X_tr.dtype)
        return X_tr, X_val, X_te, zero, zero, zero

    is_binary = jnp.all((X_tr == 0.0) | (X_tr == 1.0), axis=0)
    if preserve_mask is not None:
        preserve_mask = jnp.asarray(preserve_mask, dtype=bool)
        if preserve_mask.shape != (X_tr.shape[1],):
            raise ValueError(
                "preserve_mask must have shape (n_features,) matching the design matrices"
            )
        preserved = jnp.logical_or(is_binary, preserve_mask)
    else:
        preserved = is_binary

    mu = jnp.where(preserved, 0.0, jnp.mean(X_tr, axis=0))
    sd = jnp.where(preserved, 1.0, jnp.std(X_tr, axis=0) + eps)

    def zscore(X: Array) -> Array:
        return (X - mu) / sd

    return zscore(X_tr), zscore(X_val), zscore(X_te), mu, sd, preserved


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
