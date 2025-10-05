# src/data/split.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """
    Configuration for chronological data splits.

    Parameters
    ----------
    train_end : str or pandas.Timestamp
        Inclusive right endpoint of the training set. The training slice is
        all rows with timestamp <= train_end.
    validation_end : str or pandas.Timestamp
        Inclusive right endpoint of the validation set. The validation
        slice is all rows with train_end < timestamp <= validation_end. The test
        slice is all rows with timestamp > validation_end.
    """
    train_end: pd.Timestamp | str
    validation_end: pd.Timestamp | str


def data_split(
    df: pd.DataFrame,
    splits: DataSplits,
    *,
    sort_index: bool = True,
    validate_nonempty: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into train, validation, and test sets.

    The split is purely chronological and uses the following half-open intervals:

    - Train:        index <= train_end
    - Validation:   train_end < index <= validation_end
    - Test:         index  > validation_end

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame indexed by a pandas.DatetimeIndex. If the index is not a
        DatetimeIndex, a ValueError is raised.
    splits : DataSplits
        Dataclass specifying the two boundary timestamps.
    sort_index : bool, default True
        If True, the DataFrame is sorted by index before slicing.
    validate_nonempty : bool, default True
        If True, raise a ValueError when any of the three splits is empty.

    Returns
    -------
    (train, validate, test) : tuple of pandas.DataFrame
        The three disjoint slices as defined above.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, if the boundaries are not ordered
        (train_end < validate_end), if the boundaries fall outside the index
        range, or if any split is empty and validate_nonempty is True.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by a pandas.DatetimeIndex.")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot perform chronological split.")

    idx = df.index
    if sort_index and not idx.is_monotonic_increasing:
        df = df.sort_index()
        idx = df.index

    # Normalize boundaries to timestamps with the same timezone as the index.
    train_end = _coerce_timestamp(splits.train_end, idx)
    validate_end = _coerce_timestamp(splits.validation_end, idx)

    if not train_end < validate_end:
        raise ValueError(
            f"Split boundaries must satisfy train_end < validate_end. "
            f"Received train_end={train_end}, validate_end={validate_end}."
        )

    # Check boundaries against the observed index range.
    idx_min = idx.min()
    idx_max = idx.max()
    if train_end < idx_min or validate_end > idx_max:
        raise ValueError(
            "Split boundaries fall outside the DataFrame index range. "
            f"Index range is [{idx_min}, {idx_max}], but received "
            f"train_end={train_end}, validate_end={validate_end}."
        )

    # Boolean masks implement the interval logic exactly.
    train_mask = idx <= train_end
    validation_mask = (idx > train_end) & (idx <= validate_end)
    test_mask = idx > validate_end

    train = df.loc[train_mask]
    validation = df.loc[validation_mask]
    test = df.loc[test_mask]

    if validate_nonempty:
        n_tr = len(train)
        n_ho = len(validation)
        n_te = len(test)
        if n_tr == 0 or n_ho == 0 or n_te == 0:
            raise ValueError(
                "One or more splits are empty after applying the boundaries. "
                f"Sizes are train={n_tr}, validation={n_ho}, test={n_te}. "
                "Adjust train_end and validate_end to produce non-empty splits."
            )

    return train, validation, test


def _coerce_timestamp(ts_like: pd.Timestamp | str, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Convert a timestamp-like value to a pandas.Timestamp aligned to the index timezone.

    Parameters
    ----------
    ts_like : pandas.Timestamp or str
        The timestamp to coerce.
    idx : pandas.DatetimeIndex
        The target index whose timezone and resolution guide the coercion.

    Returns
    -------
    pandas.Timestamp
        Timestamp aligned to the index timezone.
    """
    ts = pd.Timestamp(ts_like)

    # If index is naive, return naive timestamp.
    if idx.tz is None:
        if ts.tz is not None:
            # Convert to naive in the wall time of ts's timezone by dropping tz info after conversion
            ts = ts.tz_convert(None) if ts.tz is not None else ts
            ts = pd.Timestamp(ts.replace(tzinfo=None))
        return ts

    # Index is timezone-aware.
    if ts.tz is None:
        # Localize naive timestamp to the index timezone without shifting clock time.
        return ts.tz_localize(idx.tz)

    # Convert to index timezone if necessary.
    if ts.tz != idx.tz:
        return ts.tz_convert(idx.tz)

    return ts
