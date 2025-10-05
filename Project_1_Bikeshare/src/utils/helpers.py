from __future__ import annotations

from typing import Sequence
import logging
import pandas as pd
import jax

logger = logging.getLogger(__file__)


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """
    Validate that all required columns exist in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    cols : sequence of str
        Required column names.

    Raises
    ------
    KeyError
        If any requested column is missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")


def log_jax_runtime_info() -> jax.Device:
    """
    Log JAX backend and device information and return the preferred device.

    Returns
    -------
    jax.Device
        The device we will use for training. Preference order is:
        first GPU if available, otherwise the first CPU device.
    """
    backend = jax.default_backend()
    gpus = jax.devices("gpu")
    cpus = jax.devices("cpu")

    if gpus:
        dev = gpus[0]
        chosen = f"GPU: {dev.device_kind}"
    else:
        dev = cpus[0]
        chosen = f"CPU: {dev.device_kind}"

    logger.info(f"JAX default backend: {backend}")
    logger.info(f"JAX devices (gpu={len(gpus)}, cpu={len(cpus)}). Using {chosen}.")
    return dev
