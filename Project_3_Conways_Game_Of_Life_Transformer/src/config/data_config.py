from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DataConfig:
    """Configuration for Conway data generation and caching.

    Attributes
    ----------
    height : int
        Lattice height in cells.
    width : int
        Lattice width in cells.
    num_samples : int
        Total number of pairs (x, x') to generate before splitting.
    train_fraction : float
        Fraction of the total data to use for training. Must be in
        (0, 1). The validation and test fractions are derived from this
        and ``val_fraction``.
    val_fraction : float
        Fraction of the total data to use for validation. The test
        fraction is computed as 1 - train_fraction - val_fraction.
    density : float
        Default Bernoulli parameter for the random initial
        configurations. Used when ``density_range`` is ``None``.
    density_range : tuple[float, float] or None
        Inclusive bounds ``(low, high)`` for sampling the Bernoulli
        parameter of every snapshot independently. Using a range avoids
        biasing the model toward a single average density.
    seed : int
        Global random seed used for numpy RNG in data generation and
        splitting.
    stochastic : bool
        If False use deterministic Conway rule 23/3 for data generation.
        If True use the stochastic mixed rules 23/3 and 35/3 as stated
        in the project, with bias parameter ``p_stochastic``.
    p_stochastic : float
        Bias parameter p for the stochastic rule when ``stochastic`` is
        True. Each cell independently applies rule 23/3 with probability
        p and rule 35/3 with probability 1 - p.
    anomaly_detection : bool
        If True generate an anomaly detection dataset based on the
        stochastic dynamics. In this case ``stochastic`` must be True.
        Normal samples are generated with ``p_stochastic`` and anomalous
        samples with ``p_anomaly`` and mixed according to
        ``anomaly_fraction``.
    p_anomaly : float
        Bias parameter p for anomalous samples in anomaly detection
        mode.
    anomaly_fraction : float
        Fraction of total samples that are anomalous in anomaly
        detection mode. The rest are normal.
    cache_dir : str or Path
        Directory where cached datasets are stored. The directory is
        created if it does not exist.
    cache_prefix : str or None
        Optional custom prefix for the cache file name. If None a
        prefix is constructed automatically from the configuration.
    use_cache : bool
        If True attempt to load a cached dataset from ``cache_dir``
        before generating new data.
    overwrite_cache : bool
        If True ignore any existing cache file and regenerate the data
        even if a matching cache file is found.
    """

    height: int
    width: int
    num_samples: int

    train_fraction: float = 0.7
    val_fraction: float = 0.15

    density: float = 0.5
    density_range: Optional[tuple[float, float]] = (0.2, 0.8)
    seed: int = 0

    stochastic: bool = False
    p_stochastic: float = 0.8

    anomaly_detection: bool = False
    p_anomaly: float = 0.6
    anomaly_fraction: float = 0.1

    cache_dir: Path | str = Path("data")
    cache_prefix: Optional[str] = None
    use_cache: bool = True
    overwrite_cache: bool = False

    def __post_init__(self) -> None:
        if self.density_range is not None:
            low, high = self.density_range
            if not (0.0 <= low < high <= 1.0):
                raise ValueError("density_range must satisfy 0 <= low < high <= 1")
        elif not (0.0 <= self.density <= 1.0):
            raise ValueError("density must be in [0, 1]")


@dataclass
class DatasetSplits:
    """Container for train, validation and test splits.

    Attributes
    ----------
    x_train : np.ndarray
        Training inputs of shape (N_train, H, W).
    y_train : np.ndarray
        Training targets of shape (N_train, H, W).
    x_val : np.ndarray
        Validation inputs of shape (N_val, H, W).
    y_val : np.ndarray
        Validation targets of shape (N_val, H, W).
    x_test : np.ndarray
        Test inputs of shape (N_test, H, W).
    y_test : np.ndarray
        Test targets of shape (N_test, H, W).
    labels_train : np.ndarray or None
        Optional labels for anomaly detection datasets. One dimensional
        integer array of shape (N_train,) with entries 0 for normal and
        1 for anomalies, or None for deterministic and purely stochastic
        training data.
    labels_val : np.ndarray or None
        Optional labels for the validation split in anomaly mode.
    labels_test : np.ndarray or None
        Optional labels for the test split in anomaly mode.
    densities_train : np.ndarray or None
        Optional per-sample Bernoulli parameters used to generate the
        training split.
    densities_val : np.ndarray or None
        Optional per-sample Bernoulli parameters used to generate the
        validation split.
    densities_test : np.ndarray or None
        Optional per-sample Bernoulli parameters used to generate the
        test split.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels_train: Optional[np.ndarray] = None
    labels_val: Optional[np.ndarray] = None
    labels_test: Optional[np.ndarray] = None
    densities_train: Optional[np.ndarray] = None
    densities_val: Optional[np.ndarray] = None
    densities_test: Optional[np.ndarray] = None
