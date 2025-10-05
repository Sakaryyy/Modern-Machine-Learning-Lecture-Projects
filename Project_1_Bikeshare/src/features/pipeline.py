from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from src.utils.feature_mapping import ensure_hour_column
from utils.preprocess import to_device_array
from src.utils.helpers import require_columns

Array = jax.Array
StepFn = Callable[[pd.DataFrame], Tuple[Array, List[str]]]


@dataclass
class FeaturePipeline:
    """
    Simple feature pipeline that composes stateless step functions.

    Each step consumes a pandas DataFrame and produces a pair:
    a JAX design matrix and the corresponding feature names.

    Parameters
    ----------
    steps : list of callables
        Each callable has signature step(df) -> (X, names) where X is a
        jax.Array with shape (n_samples, n_features) and names is a list[str].
    dtype : jax.numpy dtype, default jnp.float32
        Floating dtype used by every step and by the concatenated design matrix.
    device : jax.Device or None, default None
        Target device on which to place arrays. If None, JAX default device is used.
    check_finite : bool, default True
        If True, fail fast when a step generates non-finite values.
    """

    steps: List[StepFn] = field(default_factory=list)
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None
    check_finite: bool = True

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        """
        Fit the pipeline. Stateless by design, so this returns self.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data. Unused in the base pipeline.

        Returns
        -------
        FeaturePipeline
            This instance, unchanged.
        """
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[Array, List[str]]:
        """
        Apply all steps and horizontally concatenate their outputs.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data indexed by samples.

        Returns
        -------
        X : jax.Array
            Concatenated design matrix of shape (n_samples, n_features_total).
        names : list of str
            Concatenated feature names, in column order.

        Notes
        -----
        - If no steps are present, returns a zero-column matrix with shape (n, 0).
        - Concatenation is along axis 1 (column-wise).
        """
        matrices: List[Array] = []
        names: List[str] = []

        for step in self.steps:
            X_step, cols = step(df)
            # Enforce dtype/device and sanity checks at the pipeline boundary.
            X_step = to_device_array(
                X_step, dtype=self.dtype, device=self.device, check_finite=self.check_finite
            )
            if X_step.ndim != 2:
                raise ValueError("Each step must return a 2D design matrix of shape (n, m).")
            if X_step.shape[0] != len(df):
                raise ValueError(
                    f"Row count mismatch: step returned {X_step.shape[0]} rows, "
                    f"but DataFrame has {len(df)} rows."
                )
            matrices.append(X_step)
            names.extend(cols)

        if not matrices:
            empty = jnp.empty((len(df), 0), dtype=self.dtype)
            empty = jax.device_put(empty, device=self.device) if self.device is not None else empty
            return empty, []

        X = jnp.concatenate(matrices, axis=1)
        return X, names

    def fit_transform(self, df: pd.DataFrame) -> Tuple[Array, List[str]]:
        """
        Convenience method that calls fit followed by transform.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data.

        Returns
        -------
        X : jax.Array
            Concatenated design matrix from all steps.
        names : list of str
            Matching feature names.
        """
        self.fit(df)
        return self.transform(df)


# Step factories
def hour_cyclical_step(
    *,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Build a step that encodes hour-of-day as sin and cos of the angle on a 24-hour circle.

    Mapping
    -------
    Let h be the hour in {0, 1, ..., 23}. Define
        theta = 2 * pi * h / 24
        features = [sin(theta), cos(theta)]
    This encodes periodicity and removes the artificial discontinuity between 23 and 0.

    Returns
    -------
    step : callable
        Function step(df) -> (X, names), where X has two columns: sin_hour, cos_hour.
    """
    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        dfx = ensure_hour_column(df)
        hours = dfx["hr"].to_numpy(dtype=float)
        theta = 2.0 * jnp.pi * jnp.asarray(hours, dtype=dtype) / 24.0
        X = jnp.stack([jnp.sin(theta), jnp.cos(theta)], axis=1)
        return X, ["sin_hour", "cos_hour"]

    return step


def numeric_step(
    col: str,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> StepFn:
    """
    Build a step that selects a single numeric column.

    Parameters
    ----------
    col : str
        Column to extract.

    Returns
    -------
    step : callable
        Function step(df) -> (X, names) with shape (n, 1) and name [col].
    """
    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        require_columns(df, [col])
        values = df.loc[:, [col]].to_numpy(dtype=float)
        X = jnp.asarray(values, dtype=dtype)
        return X, [col]

    return step


def polynomial_step(
    col: str,
    degree: int,
    *,
    include_linear: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Build a step that maps a scalar column to polynomial basis functions.

    Mapping
    -------
    Let x be the selected column as a vector. For degree d >= 2, we construct
    features [x, x^2, ..., x^d] if include_linear is True, or [x^2, ..., x^d]
    otherwise. This introduces simple nonlinearities.

    Parameters
    ----------
    col : str
        Column name to transform.
    degree : int
        Maximum power to include. Must be >= 2 to introduce a nonlinear term.
    include_linear : bool, default True
        Whether to include the linear term x.
    dtype : jax.numpy dtype, default jnp.float32
        Floating dtype for the resulting features.
    device : jax.Device or None, default None
        Target device. The pipeline will still enforce placement and checks.
    check_finite : bool, default True
        If True, raise ValueError on non-finite values.

    Returns
    -------
    step : callable
        Function step(df) -> (X, names) where X has k columns:
        k = degree if include_linear else degree - 1.

    Raises
    ------
    ValueError
        If degree < 1 or if degree < 2 and include_linear is False.

    Notes
    -----
    - For stability, consider standardizing x upstream if the dynamic range is large.
    - For degree 1, this reduces to the linear term only.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if not include_linear and degree < 2:
        raise ValueError("degree must be >= 2 when include_linear is False")

    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        require_columns(df, [col])
        x_np = df[col].to_numpy(dtype=float).reshape(-1, 1)
        x = jnp.asarray(x_np, dtype=dtype)

        cols: List[Array] = []
        names: List[str] = []

        start_pow = 1 if include_linear else 2
        for p in range(start_pow, degree + 1):
            cols.append(jnp.power(x, p))
            names.append(col if p == 1 else f"{col}^{p}")

        X = jnp.concatenate(cols, axis=1) if cols else jnp.empty((len(df), 0), dtype=dtype)
        return X, names

    return step


def hour_onehot_step(
    *,
    hour_col: str = "hr",
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Encode hour-of-day as one-hot indicators.

    Parameters
    ----------
    hour_col : str, default "hr"
        Column containing hour-of-day if present.
    drop_first : bool, default True
        If True, produce 23 columns (hours 1..23). If False, produce 24 columns (0..23).

    Returns
    -------
    StepFn
        A function mapping df -> (X, names) where X has shape (n, 23 or 24).
    """
    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        dfx = ensure_hour_column(df)
        h_np = dfx[hour_col].to_numpy(dtype=int)
        h = jnp.clip(jnp.asarray(h_np), 0, 23)

        eye = jnp.eye(24, dtype=dtype)
        H = eye[h]  # shape (n, 24)

        if drop_first:
            X = H[:, 1:]  # drop hour 0
            names = [f"hr_{k}" for k in range(1, 24)]
        else:
            X = H
            names = [f"hr_{k}" for k in range(24)]

        return X, names
    return step


def hour_fourier_step(
    n_harmonics: int = 2,
    *,
    hour_col: str = "hr",
    dtype: jnp.dtype = jnp.float32,
) -> StepFn:
    """
    Encode hour-of-day with multiple Fourier harmonics.

    Features are [sin(k*2*pi*h/24), cos(k*2*pi*h/24)] for k=1..n_harmonics.

    Parameters
    ----------
    n_harmonics : int, default 2
        Number of harmonics to include. n_harmonics=1 reduces to `hour_cyclical_step`.
    hour_col : str, default "hr"
        Name of the hour column if present.

    Returns
    -------
    StepFn
        Produces a design matrix with 2*n_harmonics columns.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        dfx = ensure_hour_column(df)
        hours = dfx[hour_col].to_numpy(dtype=float)
        theta = 2.0 * jnp.pi * jnp.asarray(hours, dtype=dtype) / 24.0

        cols: List[Array] = []
        names: List[str] = []
        for k in range(1, n_harmonics + 1):
            cols.append(jnp.sin(k * theta))
            cols.append(jnp.cos(k * theta))
            names.append(f"sin_hour_{k}")
            names.append(f"cos_hour_{k}")
        X = jnp.stack(cols, axis=1) if len(cols) == 2 else jnp.column_stack(cols)
        return X, names

    return step


def interaction_onehot_numeric_step(
    numeric_col: str,
    *,
    hour_col: str = "hr",
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> StepFn:
    """
    Build hour-by-numeric interactions: one slope per hour.

    Mapping
    -------
    Let H be the hour one-hot matrix with k columns (k=23 if drop_first else 24),
    and let x be the scalar column as a vector. The interaction block is H .* x,
    yielding k columns: [x*1{h=cat}] for each hour category.

    This allows the effect of `numeric_col` to vary across hours, while the
    overall model remains linear in parameters.

    Returns
    -------
    StepFn
        Produces a design matrix with k columns named like f"{numeric_col}:hr_i".
    """
    base_onehot = hour_onehot_step( hour_col=hour_col, drop_first=drop_first, dtype=dtype)
    numeric = numeric_step(numeric_col, dtype=dtype)

    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        H, hr_names = base_onehot(df)
        x, _ = numeric(df)  # shape (n, 1)
        X = H * x  # broadcast multiply: per-hour slope
        names = [f"{numeric_col}:{name}" for name in hr_names]
        return X, names

    return step


def weekday_onehot_step(
    *,
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Encode day-of-week as one-hot indicators.

    Parameters
    ----------
    drop_first : bool, default True
        If True, produce 6 columns (Mon..Sun minus one). If False, 7 columns.
    dtype, device, check_finite : see other step factories.

    Returns
    -------
    StepFn
        Maps df -> (X, names) with X shape (n, 6 or 7).
    """
    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("weekday_onehot_step requires a DatetimeIndex.")
        wd = df.index.weekday.to_numpy(dtype=int)  # 0=Mon..6=Sun
        eye = jnp.eye(7, dtype=dtype)
        H = eye[wd]
        if drop_first:
            X = H[:, 1:]  # drop Monday
            names = [f"wd_{k}" for k in range(1, 7)]
        else:
            X = H
            names = [f"wd_{k}" for k in range(7)]
        return X, names
    return step


def month_onehot_step(
    *,
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Encode month as one-hot indicators (Jan..Dec).

    Parameters
    ----------
    drop_first : bool, default True
        If True, produce 11 columns (1..12 minus one). If False, 12 columns.

    Returns
    -------
    StepFn
        Maps df -> (X, names) with X shape (n, 11 or 12).
    """
    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("month_onehot_step requires a DatetimeIndex.")
        m0 = df.index.month.to_numpy(dtype=int) - 1  # 0..11
        eye = jnp.eye(12, dtype=dtype)
        H = eye[m0]
        if drop_first:
            X = H[:, 1:]  # drop January
            names = [f"mon_{k}" for k in range(1, 12)]
        else:
            X = H
            names = [f"mon_{k}" for k in range(12)]
        return X, names
    return step


def weathersit_onehot_step(
    *,
    col: str = "weathersit",
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Encode the 'weathersit' categorical code (1..4) as one-hot indicators.

    The UCI metadata documents:
      1: clear / few clouds / partly cloudy
      2: mist + cloudy / mist + broken clouds / mist
      3: light snow / light rain + thunderstorm / scattered clouds
      4: heavy rain + ice pellets + thunderstorm + mist / snow + fog

    Parameters
    ----------
    col : str, default "weathersit"
        Column with integer codes 1..4.
    drop_first : bool, default True
        If True, produce 3 columns (codes 2..4). If False, 4 columns.
    dtype, device, check_finite : see other step factories.

    Returns
    -------
    StepFn
        Maps df -> (X, names); X has shape (n, 3) if drop_first else (n, 4).
    """
    names_all = [
        "ws_clear",
        "ws_mist",
        "ws_light",
        "ws_heavy",
    ]

    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        require_columns(df, [col])
        codes = df[col].to_numpy(dtype=int)
        if not ((codes >= 1) & (codes <= 4)).all():
            raise ValueError("weathersit codes must be in {1,2,3,4}.")
        idx = jnp.asarray(codes - 1, dtype=jnp.int32)  # 0..3
        eye = jnp.eye(4, dtype=dtype)
        H = eye[idx]  # (n, 4)
        if drop_first:
            X = H[:, 1:]
            names = names_all[1:]
        else:
            X = H
            names = names_all
        return X, names

    return step


def season_onehot_step(
    *,
    col: str = "season",
    drop_first: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> StepFn:
    """
    Encode the 'season' categorical code (1..4) as one-hot indicators.

      1: winter, 2: spring, 3: summer, 4: fall

    Parameters
    ----------
    col : str, default "season"
        Column with integer codes 1..4.
    drop_first : bool, default True
        If True, produce 3 columns (codes 2..4). If False, 4 columns.

    Returns
    -------
    StepFn
        Maps df -> (X, names); X has shape (n, 3) if drop_first else (n, 4).
    """
    names_all = [
        "season_winter",
        "season_spring",
        "season_summer",
        "season_fall",
    ]

    def step(df: pd.DataFrame) -> Tuple[Array, List[str]]:
        require_columns(df, [col])
        codes = df[col].to_numpy(dtype=int)
        if not ((codes >= 1) & (codes <= 4)).all():
            raise ValueError("season codes must be in {1,2,3,4}.")
        idx = jnp.asarray(codes - 1, dtype=jnp.int32)  # 0..3
        eye = jnp.eye(4, dtype=dtype)
        H = eye[idx]  # (n, 4)
        if drop_first:
            X = H[:, 1:]
            names = names_all[1:]
        else:
            X = H
            names = names_all
        return X, names

    return step


def build_minimal_candidate_pipeline(
    use_atemp: bool = True,
    use_workingday: bool = True,
    use_humidity: bool = False,
    use_windspeed: bool = False,
    add_hour_cyclical: bool = True,
    poly_temp_degree: int = 1,
    *,
    dtype: jnp.dtype = jnp.float32,
    device: Optional[jax.Device] = None,
    check_finite: bool = True,
) -> FeaturePipeline:
    """
    Construct a small, interpretable pipeline.

    Groups included
    ---------------
    - Hour-of-day cyclical encoding: sin and cos of 2*pi*h/24
    - Apparent temperature: linear term and optionally polynomial powers up to `poly_temp_degree`
    - Optional scalars: workingday, hum, windspeed

    Parameters
    ----------
    use_atemp : bool, default True
        Include apparent temperature features.
    use_workingday : bool, default True
        Include workingday indicator.
    use_humidity : bool, default False
        Include humidity scalar feature.
    use_windspeed : bool, default False
        Include windspeed scalar feature.
    add_hour_cyclical : bool, default True
        Include hour cyclical encoding.
    poly_temp_degree : int, default 1
        If >= 2, adds nonlinear powers of atemp up to the specified degree.
    dtype : jax.numpy dtype, default jnp.float32
        Floating dtype used across steps.
    device : jax.Device or None, default None
        Target device for arrays.
    check_finite : bool, default True
        If True, validate finiteness in each step at the pipeline boundary.

    Returns
    -------
    FeaturePipeline
        Configured pipeline whose steps produce jax.Arrays suitable for modeling.
    """
    steps: List[StepFn] = []

    if add_hour_cyclical:
        steps.append(hour_cyclical_step(dtype=dtype))

    if use_atemp:
        steps.append(numeric_step("atemp", dtype=dtype))
        if poly_temp_degree >= 2:
            steps.append(
                polynomial_step("atemp", degree=poly_temp_degree, include_linear=False, dtype=dtype)
            )

    if use_workingday:
        steps.append(numeric_step("workingday", dtype=dtype))

    if use_humidity:
        steps.append(numeric_step("hum", dtype=dtype))

    if use_windspeed:
        steps.append(numeric_step("windspeed", dtype=dtype))

    return FeaturePipeline(steps=steps, dtype=dtype, device=device, check_finite=check_finite)
