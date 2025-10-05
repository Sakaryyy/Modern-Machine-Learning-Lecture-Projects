from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    """
    Container for common regression metrics.

    Parameters
    ----------
    rmse_tr : float
        Root-mean-squared error on the training data (or train-like split).
    rmse_val : float
        Root-mean-squared error on the validation data.
    rmse_te : float or None
        Root-mean-squared error on the test data. Optional if not evaluated.
    mae_te : float or None
        Mean absolute error on the test data. Optional if not evaluated.
    r2_te : float or None
        Coefficient of determination on the test data. Optional if not evaluated.
    """

    rmse_tr: float
    rmse_val: float
    rmse_te: Optional[float]
    mae_te: Optional[float]
    r2_te: Optional[float]

    def as_dataframe(self) -> pd.DataFrame:
        """
        Convert the stored metrics to a single-row pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A one-row frame with columns: rmse_tr, rmse_val, rmse_te, mae_te, r2_te.
        """
        return pd.DataFrame(
            [
                {
                    "rmse_tr": self.rmse_tr,
                    "rmse_val": self.rmse_val,
                    "rmse_te": self.rmse_te,
                    "mae_te": self.mae_te,
                    "r2_te": self.r2_te,
                }
            ]
        )


# Internal helpers
def _assert_same_shape(y_true: jax.Array, y_pred: jax.Array) -> None:
    """
    Ensure y_true and y_pred have identical shapes.

    Parameters
    ----------
    y_true : jax.Array
        Ground truth targets.
    y_pred : jax.Array
        Predicted targets.

    Raises
    ------
    ValueError
        If shapes do not match.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true.shape={y_true.shape} vs y_pred.shape={y_pred.shape}"
        )


def _to_py_float(x: jax.Array) -> float:
    """
    Convert a JAX scalar array to a Python float.

    Parameters
    ----------
    x : jax.Array
        Scalar array.

    Returns
    -------
    float
        Python float value.
    """
    return float(jnp.asarray(x).item())


# Public metric functions
def rmse(y_true: jax.Array, y_pred: jax.Array) -> float:
    """
    Compute the root-mean-squared error (RMSE).

    Definition
    ----------
    RMSE(y, yhat) = sqrt( mean( (y - yhat)^2 ) )

    Parameters
    ----------
    y_true : jax.Array
        Ground truth targets.
    y_pred : jax.Array
        Predicted targets, same shape and dtype as `y_true`.

    Returns
    -------
    float
        RMSE as a Python float.
    """
    _assert_same_shape(y_true, y_pred)
    diff = y_true - y_pred
    mse = jnp.mean(jnp.square(diff))
    return _to_py_float(jnp.sqrt(mse))


def mae(y_true: jax.Array, y_pred: jax.Array) -> float:
    """
    Compute the mean absolute error (MAE).

    Definition
    ----------
    MAE(y, yhat) = mean( |y - yhat| )

    Parameters
    ----------
    y_true : jax.Array
        Ground truth targets.
    y_pred : jax.Array
        Predicted targets, same shape and dtype as `y_true`.

    Returns
    -------
    float
        MAE as a Python float.
    """
    _assert_same_shape(y_true, y_pred)
    return _to_py_float(jnp.mean(jnp.abs(y_true - y_pred)))


def r2(y_true: jax.Array, y_pred: jax.Array) -> float:
    """
    Compute the coefficient of determination (R^2).

    Definition
    ----------
    R^2(y, yhat) = 1 - SS_res / SS_tot

    where
      SS_res = sum( (y - yhat)^2 )
      SS_tot = sum( (y - mean(y))^2 )

    Parameters
    ----------
    y_true : jax.Array
        Ground truth targets, one-dimensional or any shape that can be reduced.
    y_pred : jax.Array
        Predicted targets, same shape as `y_true`.

    Returns
    -------
    float
        R^2 as a Python float. If `y_true` is constant (SS_tot == 0), this
        function returns 0.0 by convention to avoid division-by-zero and NaN.

    Notes
    -----
    - R^2 can be negative if the model performs worse than predicting the mean.
    - When y_true is constant, R^2 is not well-defined -> we return 0.0.
    """
    _assert_same_shape(y_true, y_pred)
    y_mean = jnp.mean(y_true)
    ss_res = jnp.sum(jnp.square(y_true - y_pred))
    ss_tot = jnp.sum(jnp.square(y_true - y_mean))
    r2_val = jnp.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return _to_py_float(r2_val)
