from __future__ import annotations

"""
Foundational feature-transform interfaces and simple selectors.

This module defines a minimal protocol for feature transformers that take a
pandas DataFrame and return a JAX design matrix plus column names. It also
provides a typed ColumnSelector for selecting raw columns.
"""

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.utils.preprocess import to_device_array
from src.utils.helpers import require_columns

Array = jax.Array


class Transformer(Protocol):
    """
    Protocol for feature transformers.

    A transformer accepts a pandas DataFrame and returns a pair:
    a JAX design matrix and the corresponding feature names.

    Functions
    ---------
    fit(df) -> Transformer
        Optional stateful fitting step. Stateless transformers may return self.
    transform(df) -> (Array, list[str])
        Build the design matrix for df. The matrix must have shape
        (n_samples, n_features) and use jax.numpy semantics.
    """

    def fit(self, df: pd.DataFrame) -> Transformer:
        ...

    def transform(self, df: pd.DataFrame) -> Tuple[Array, List[str]]:
        ...


@dataclass
class ColumnSelector:
    """
    Select a set of columns and return them as a JAX design matrix.

    Parameters
    ----------
    columns : sequence of str
        Columns to select from the DataFrame, in order.
    dtype : jax.numpy dtype, default jnp.float32
        Floating point dtype of the resulting design matrix.
    device : jax.Device or None, default None
        Target device on which to place the resulting JAX array. If None,
        the JAX default device is used.
    check_finite : bool, default True
        If True, raise a ValueError when NaN or infinite values are detected
        in the selected columns.
    """

    columns: Sequence[str]
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None
    check_finite: bool = True

    def fit(self, df: pd.DataFrame) -> ColumnSelector:
        """
        Fit the selector. Stateless, so this returns self.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data

        Returns
        -------
        ColumnSelector
            The same instance, unchanged.
        """
        require_columns(df, self.columns)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[Array, List[str]]:
        """
        Extract columns and return a JAX device array and column names.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data indexed by samples.

        Returns
        -------
        X : jax.Array
            Design matrix of shape (n_samples, n_features) on the configured device.
        names : list of str
            Column names in the same order as in X.

        Raises
        ------
        KeyError
            If any requested column is missing.
        ValueError
            If check_finite is True and NaN or infinite values are present.
        """
        require_columns(df, self.columns)

        # Uses pandas to extract raw values as a dense float ndarray.
        # We request float to avoid object or extension dtypes.
        values_np = df.loc[:, list(self.columns)].to_numpy(dtype=float)

        if self.check_finite:
            if not np.isfinite(values_np).all():
                raise ValueError(
                    "Non-finite values detected in selected columns. "
                    "Clean the data or disable check_finite to proceed."
                )

        X = to_device_array(values_np, dtype=self.dtype, device=self.device, check_finite=True)
        names = list(self.columns)
        return X, names
