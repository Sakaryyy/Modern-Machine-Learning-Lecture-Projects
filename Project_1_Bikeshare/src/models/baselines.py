from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import pandas as pd

from src.utils.preprocess import to_device_array
from src.utils.helpers import require_columns
from src.utils.feature_mapping import ensure_hour_column

Array = jax.Array


@dataclass(frozen=True)
class MeanBaseline:
    """
    Blind baseline that always predicts the training mean of the target.

    Definition
    ----------
    Let y_train be the training targets and let
        mu = mean(y_train).
    For any input example in any split, the prediction is
        y_hat = mu.

    Parameters
    ----------
    mean_y : float
        Training-set mean of the target. Stored as a Python float for stable serialization.
    dtype : jax.numpy dtype, default jnp.float32
        Dtype used when generating prediction arrays.
    device : jax.Device or None, default None
        Device on which to place the prediction arrays. If None, JAX default device is used.
    """

    mean_y: float
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None

    @classmethod
    def from_train(
        cls,
        y: Array,
        *,
        dtype: jnp.dtype = jnp.float32,
        device: Optional[jax.Device] = None,
    ) -> MeanBaseline:
        """
        Fit the baseline by computing the training mean.

        Parameters
        ----------
        y : jax.Array
            Training targets as a JAX array on any device and shape (n,) or compatible.
        dtype : jax.numpy dtype, default jnp.float32
            Dtype for predictions emitted by this baseline.
        device : jax.Device or None, default None
            Target device for predictions.

        Returns
        -------
        MeanBaseline
            Fitted baseline with mean_y populated.
        """
        y = jnp.asarray(y, dtype=dtype)
        mu = float(jnp.mean(y))
        return cls(mean_y=mu, dtype=dtype, device=device)

    def predict(self, n: int) -> Array:
        """
        Generate n predictions equal to the stored training mean.

        Parameters
        ----------
        n : int
            Number of predictions to produce.

        Returns
        -------
        jax.Array
            Array of shape (n,) filled with mean_y, on the configured device and dtype.
        """
        preds = jnp.full((n,), self.mean_y, dtype=self.dtype)
        return to_device_array(preds, dtype=self.dtype, device=self.device, check_finite=True)


@dataclass(frozen=True)
class HourOfDayBaseline:
    """
    Semi-blind baseline that predicts by the mean target for the hour-of-day.

    Definition
    ----------
    Let h in {0, 1, ..., 23} be the hour-of-day for an observation and let
        m_h = mean( y_i | hour_i = h )  computed on the training data.
    The prediction for an input with hour h is
        y_hat = m_h.

    Parameters
    ----------
    hour_means : jax.Array
        Length-24 vector where hour_means[h] is the training mean for hour h.
        Stored on a JAX device for direct indexing during prediction.
    global_mean : float
        Global training mean of the target. Used to fill missing hours and as a fallback.
    dtype : jax.numpy dtype, default jnp.float32
        Dtype used for the hour_means vector and prediction outputs.
    device : jax.Device or None, default None
        Device on which hour_means lives and predictions are placed.
    """

    hour_means: Array
    global_mean: float
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None

    @classmethod
    def from_train(
        cls,
        df_train: pd.DataFrame,
        *,
        target_col: str = "cnt",
        dtype: jnp.dtype = jnp.float32,
        device: Optional[jax.Device] = None,
    ) -> HourOfDayBaseline:
        """
        Fit the baseline by computing per-hour means on the training set.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training DataFrame. Must include the target column and either a column
            "hr" with hour-of-day or a DatetimeIndex from which hour can be derived.
        target_col : str, default "cnt"
            Name of the target column.
        dtype : jax.numpy dtype, default jnp.float32
            Dtype for the internal hour_means vector and predictions.
        device : jax.Device or None, default None
            Device on which to store hour_means and place predictions.

        Returns
        -------
        HourOfDayBaseline
            Fitted baseline with hour_means and global_mean populated.

        Raises
        ------
        KeyError
            If the target column is missing.
        ValueError
            If the DataFrame is empty.
        """
        if df_train.empty:
            raise ValueError("Training DataFrame is empty.")
        require_columns(df_train, [target_col])

        # Ensure we have an 'hr' column. If not present, derive from index.
        dfx = ensure_hour_column(df_train)
        hours = dfx["hr"].astype(int)

        # Compute global mean and per-hour means on training data.
        global_mean = float(df_train[target_col].mean())

        grouped = df_train.assign(hr=hours).groupby("hr")[target_col].mean()
        # Reindex to all 24 hours and fill missing with the global mean.
        grouped = grouped.reindex(range(24), fill_value=global_mean)

        # Convert to JAX device array.
        hour_means_vec = to_device_array(
            grouped.to_numpy(dtype=float),
            dtype=dtype,
            device=device,
            check_finite=True,
        )

        return cls(
            hour_means=hour_means_vec,
            global_mean=global_mean,
            dtype=dtype,
            device=device,
        )

    def predict(self, df: pd.DataFrame, *, hour_col: str = "hr") -> Array:
        """
        Predict using the stored per-hour means.

        Parameters
        ----------
        df : pandas.DataFrame
            Data for which to predict. Must include
            - an hour-of-day column named `hour_col`, or
            - a DatetimeIndex from which hour can be derived.
        hour_col : str, default "hr"
            Name of the hour-of-day column if present.

        Returns
        -------
        jax.Array
            Predictions of shape (n,) on the baseline's configured device and dtype.
        """
        if df.empty:
            # Return an empty vector on the correct device and dtype.
            empty = jnp.empty((0,), dtype=self.dtype)
            return to_device_array(empty, dtype=self.dtype, device=self.device, check_finite=True)

        dfx = ensure_hour_column(df)
        hours_np = dfx[hour_col].to_numpy(dtype=int)
        # Clip for robustness, then index into hour_means.
        hours = jnp.asarray(hours_np)
        hours = jnp.clip(hours, 0, 23)
        preds = jnp.take(self.hour_means, hours, mode="clip")
        # Ensure dtype is consistent (hour_means already has the right dtype).
        preds = jnp.asarray(preds, dtype=self.dtype)
        return to_device_array(preds, dtype=self.dtype, device=self.device, check_finite=True)
