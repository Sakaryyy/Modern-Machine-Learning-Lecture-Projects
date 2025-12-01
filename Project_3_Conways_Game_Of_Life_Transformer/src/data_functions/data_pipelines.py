from pathlib import Path
from typing import Tuple

import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import (
    DataConfig,
    DatasetSplits,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.conway_rules import (
    conway_step_periodic,
    stochastic_step_mixed_rules,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import get_logger

Logger = get_logger(__name__)


def sample_random_grid(
        height: int,
        width: int,
        density: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random binary configuration with given density.

    Each cell is independently set to 1 with probability ``density``
    and to 0 otherwise.

    Parameters
    ----------
    height : int
        Lattice height.
    width : int
        Lattice width.
    density : float
        Bernoulli parameter in [0, 1] for cell occupancy.
    rng : np.random.Generator
        Numpy random generator.

    Returns
    -------
    grid : np.ndarray
        Binary array of shape (height, width) with entries in {0, 1}.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1]")

    grid = rng.random(size=(height, width)) < density
    return grid.astype(np.int32)


def generate_deterministic_pairs(
        num_samples: int,
        height: int,
        width: int,
        density: float,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic Conway pairs (x, x').

    Parameters
    ----------
    num_samples : int
        Number of pairs to generate.
    height : int
        Lattice height.
    width : int
        Lattice width.
    density : float
        Bernoulli parameter for random initial states.
    rng : np.random.Generator
        Numpy random generator.

    Returns
    -------
    inputs : np.ndarray
        Array of shape (num_samples, height, width) with initial states.
    targets : np.ndarray
        Array of shape (num_samples, height, width) with next states
        under the deterministic Conway rule 23/3.
    """
    inputs = np.empty((num_samples, height, width), dtype=np.int32)
    targets = np.empty_like(inputs)

    for i in range(num_samples):
        x = sample_random_grid(height, width, density, rng)
        y = conway_step_periodic(x)
        inputs[i] = x
        targets[i] = y

    return inputs, targets


def generate_stochastic_pairs(
        num_samples: int,
        height: int,
        width: int,
        density: float,
        p: float,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stochastic Game of Life pairs (x, x') at fixed p.

    Each pair is generated as follows. A random initial state x is drawn
    from a Bernoulli distribution with parameter ``density``. The next
    state x' is obtained by applying the stochastic mixed rule where
    each cell independently uses rule 23/3 with probability p and rule
    35/3 with probability 1 - p.

    Parameters
    ----------
    num_samples : int
        Number of pairs to generate.
    height : int
        Lattice height.
    width : int
        Lattice width.
    density : float
        Bernoulli parameter for random initial states.
    p : float
        Bias parameter for the stochastic rule. Must be in [0, 1].
    rng : np.random.Generator
        Numpy random generator.

    Returns
    -------
    inputs : np.ndarray
        Array of shape (num_samples, height, width) with initial states.
    targets : np.ndarray
        Array of shape (num_samples, height, width) with next states
        under the stochastic rule.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    inputs = np.empty((num_samples, height, width), dtype=np.int32)
    targets = np.empty_like(inputs)

    for i in range(num_samples):
        x = sample_random_grid(height, width, density, rng)
        y = stochastic_step_mixed_rules(x, p=p, rng=rng)
        inputs[i] = x
        targets[i] = y

    return inputs, targets


def generate_anomaly_pairs(
        num_samples: int,
        height: int,
        width: int,
        density: float,
        p_normal: float,
        p_anomaly: float,
        anomaly_fraction: float,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pairs (x, x') and labels for anomaly detection.

    A fraction of the samples are generated with a "normal" stochastic
    rule parameter p_normal and labelled as 0. The remaining fraction is
    generated with p_anomaly and labelled as 1.

    Parameters
    ----------
    num_samples : int
        Total number of pairs to generate.
    height : int
        Lattice height.
    width : int
        Lattice width.
    density : float
        Bernoulli parameter for initial states.
    p_normal : float
        Bias parameter p for normal data, for example 0.8.
    p_anomaly : float
        Bias parameter p for anomalous data, for example 0.6.
    anomaly_fraction : float
        Fraction of samples that are anomalous. Must be in [0, 1].
    rng : np.random.Generator
        Numpy random generator.

    Returns
    -------
    inputs : np.ndarray
        Array of shape (num_samples, height, width) with initial states.
    targets : np.ndarray
        Array of shape (num_samples, height, width) with next states
        under the corresponding stochastic rule.
    labels : np.ndarray
        Integer array of shape (num_samples,) with 0 for normal samples
        and 1 for anomalies.
    """
    if not (0.0 <= anomaly_fraction <= 1.0):
        raise ValueError("anomaly_fraction must be in [0, 1]")

    num_anom = int(round(num_samples * anomaly_fraction))
    num_normal = num_samples - num_anom

    inputs = np.empty((num_samples, height, width), dtype=np.int32)
    targets = np.empty_like(inputs)
    labels = np.empty((num_samples,), dtype=np.int32)

    all_indices = np.arange(num_samples)
    rng.shuffle(all_indices)
    anom_idx = all_indices[:num_anom]
    normal_idx = all_indices[num_anom:]

    # Anomalous samples
    for idx in anom_idx:
        x = sample_random_grid(height, width, density, rng)
        y = stochastic_step_mixed_rules(x, p=p_anomaly, rng=rng)
        inputs[idx] = x
        targets[idx] = y
        labels[idx] = 1

    # Normal samples
    for idx in normal_idx:
        x = sample_random_grid(height, width, density, rng)
        y = stochastic_step_mixed_rules(x, p=p_normal, rng=rng)
        inputs[idx] = x
        targets[idx] = y
        labels[idx] = 0

    assert labels.sum() == num_anom
    assert (labels == 0).sum() == num_normal

    return inputs, targets, labels


def _sanitize_float_for_name(x: float) -> str:
    """Convert a float to a short string safe for file names."""
    return f"{x:.3f}".replace(".", "p")


def _build_cache_path(config: DataConfig) -> Path:
    """Construct a cache file path from a data configuration.

    Parameters
    ----------
    config : DataConfig
        Data generation configuration.

    Returns
    -------
    path : pathlib.Path
        Path to the npz cache file inside ``config.cache_dir``.
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if config.cache_prefix is not None:
        prefix = config.cache_prefix
    else:
        mode = "det" if not config.stochastic else (
            "anom" if config.anomaly_detection else "stoch"
        )
        bits = [
            "gol",
            mode,
            f"H{config.height}",
            f"W{config.width}",
            f"N{config.num_samples}",
            f"d{_sanitize_float_for_name(config.density)}",
            f"seed{config.seed}",
        ]
        if config.stochastic:
            bits.append(f"p{_sanitize_float_for_name(config.p_stochastic)}")
        if config.anomaly_detection:
            bits.append(f"pa{_sanitize_float_for_name(config.p_anomaly)}")
            bits.append(f"fa{_sanitize_float_for_name(config.anomaly_fraction)}")
        prefix = "_".join(bits)

    file_name = prefix + ".npz"
    return cache_dir / file_name


def _load_cached_dataset(path: Path) -> DatasetSplits:
    """Load dataset splits from a cached npz file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the npz file.

    Returns
    -------
    splits : DatasetSplits
        Loaded dataset splits.
    """
    data = np.load(path, allow_pickle=False)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    labels_train = data["labels_train"] if "labels_train" in data.files else None
    labels_val = data["labels_val"] if "labels_val" in data.files else None
    labels_test = data["labels_test"] if "labels_test" in data.files else None

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        labels_train=labels_train,
        labels_val=labels_val,
        labels_test=labels_test,
    )


def _save_cached_dataset(path: Path, splits: DatasetSplits) -> None:
    """Save dataset splits to a compressed npz file.

    Parameters
    ----------
    path : pathlib.Path
        Path where the npz file will be written.
    splits : DatasetSplits
        Dataset splits to save.
    """
    arrays = {
        "x_train": splits.x_train,
        "y_train": splits.y_train,
        "x_val": splits.x_val,
        "y_val": splits.y_val,
        "x_test": splits.x_test,
        "y_test": splits.y_test,
    }
    if splits.labels_train is not None:
        arrays["labels_train"] = splits.labels_train
    if splits.labels_val is not None:
        arrays["labels_val"] = splits.labels_val
    if splits.labels_test is not None:
        arrays["labels_test"] = splits.labels_test

    np.savez_compressed(path, **arrays)


def _compute_split_sizes(
        num_samples: int,
        train_fraction: float,
        val_fraction: float,
) -> Tuple[int, int, int]:
    """Compute train, validation and test sizes from fractions.

    Parameters
    ----------
    num_samples : int
        Total number of samples.
    train_fraction : float
        Fraction for training in (0, 1).
    val_fraction : float
        Fraction for validation in (0, 1).

    Returns
    -------
    num_train : int
        Number of training samples.
    num_val : int
        Number of validation samples.
    num_test : int
        Number of test samples.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction > 1.0:
        raise ValueError("train_fraction + val_fraction must not exceed 1.0")

    num_train = int(round(num_samples * train_fraction))
    num_val = int(round(num_samples * val_fraction))
    num_test = num_samples - num_train - num_val

    if num_test <= 0:
        raise ValueError("resulting test set size is not positive")

    return num_train, num_val, num_test


def prepare_gol_dataset(config: DataConfig) -> DatasetSplits:
    """Prepare Conway Game of Life data with caching and splitting.

    This function either loads a cached dataset from disk or generates a
    new one according to the configuration. It supports three modes:

    1. Deterministic mode, if stochastic is False:
       Data is generated with the standard deterministic Conway rule.

    2. Stochastic mode, if stochastic is True and anomaly_detection is
       False:
       Data is generated with the stochastic mixed rule at a fixed bias
       parameter p_stochastic.

    3. Anomaly detection mode, if stochastic and anomaly_detection are
       both True:
       Data is generated with a mixture of p_stochastic (normal) and
       p_anomaly (anomaly) according to anomaly_fraction and split into
       train, validation and test sets. Labels indicate normal or
       anomalous samples in all splits.

    Parameters
    ----------
    config : DataConfig
        Data generation and caching configuration.

    Returns
    -------
    splits : DatasetSplits
        Train, validation and test splits as numpy arrays.
    """
    if config.anomaly_detection and not config.stochastic:
        raise ValueError(
            "anomaly_detection=True requires stochastic=True, "
            "since anomalies are defined via stochastic parameters."
        )

    path = _build_cache_path(config)

    if config.use_cache and path.exists() and not config.overwrite_cache:
        Logger.info("Loading cached dataset from %s", path)
        return _load_cached_dataset(path)

    Logger.info("No valid cache at %s, generating new dataset", path)
    np_rng = np.random.default_rng(config.seed)

    # Generate full dataset
    if not config.stochastic:
        inputs, targets = generate_deterministic_pairs(
            num_samples=config.num_samples,
            height=config.height,
            width=config.width,
            density=config.density,
            rng=np_rng,
        )
        labels = None

    elif config.stochastic and not config.anomaly_detection:
        inputs, targets = generate_stochastic_pairs(
            num_samples=config.num_samples,
            height=config.height,
            width=config.width,
            density=config.density,
            p=config.p_stochastic,
            rng=np_rng,
        )
        labels = None

    else:
        inputs, targets, labels = generate_anomaly_pairs(
            num_samples=config.num_samples,
            height=config.height,
            width=config.width,
            density=config.density,
            p_normal=config.p_stochastic,
            p_anomaly=config.p_anomaly,
            anomaly_fraction=config.anomaly_fraction,
            rng=np_rng,
        )

    num_samples = inputs.shape[0]
    num_train, num_val, num_test = _compute_split_sizes(
        num_samples=num_samples,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )

    # Shuffle indices once for all splits
    indices = np.arange(num_samples)
    np_rng.shuffle(indices)

    idx_train = indices[:num_train]
    idx_val = indices[num_train:num_train + num_val]
    idx_test = indices[num_train + num_val:]

    x_train = inputs[idx_train]
    y_train = targets[idx_train]
    x_val = inputs[idx_val]
    y_val = targets[idx_val]
    x_test = inputs[idx_test]
    y_test = targets[idx_test]

    if labels is not None:
        labels_train = labels[idx_train]
        labels_val = labels[idx_val]
        labels_test = labels[idx_test]
    else:
        labels_train = labels_val = labels_test = None

    splits = DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        labels_train=labels_train,
        labels_val=labels_val,
        labels_test=labels_test,
    )

    # Persist to cache
    _save_cached_dataset(path, splits)
    Logger.info("Saved dataset to %s", path)
    return splits
