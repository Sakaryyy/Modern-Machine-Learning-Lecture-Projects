"""
Tools for downloading and persisting the CIFAR-10 dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import numpy as np

try:
    from torchvision import datasets, transforms
except ImportError as exc:
    raise ImportError(
        "torchvision is required for downloading the CIFAR-10 dataset. "
        "Please install it via 'pip install torchvision'.",
    ) from exc

from Project_2_Image_Classification.src.utils.logging import get_logger


@dataclass(frozen=True)
class DatasetSplit:
    """Container bundling together images and labels for one split."""

    images: jnp.ndarray
    labels: jnp.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    """Bundle storing all dataset splits alongside metadata."""

    splits: Dict[str, DatasetSplit]
    metadata: Dict[str, object]


class CIFAR10DataManager:
    """Download, normalise and persist the CIFAR-10 dataset.

    Parameters
    ----------
    data_root:
        Base directory where raw downloads and processed arrays will be stored.
    val_split:
        Fraction of the original training data that will be moved to the
        validation set.  Must lie strictly between 0 and 1.
    seed:
        Seed that controls the random permutation used for the train/validation
        split to ensure reproducibility across runs.
    """

    RAW_SUBDIR = "raw"
    PROCESSED_SUBDIR = "processed"

    def __init__(self, data_root: Path | str, val_split: float = 0.1, seed: int = 10) -> None:
        if not 0.0 < val_split < 1.0:
            msg = "The validation split needs to be within (0, 1)."
            raise ValueError(msg)

        self.data_root = Path(data_root)
        self.val_split = val_split
        self.seed = seed
        self.logger = get_logger(self.__class__.__name__)

        self.raw_dir = self.data_root / self.RAW_SUBDIR
        self.processed_dir = self.data_root / self.PROCESSED_SUBDIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.processed_dir / "metadata.json"

    def prepare_data(self) -> PreparedDataset:
        """Ensure CIFAR-10 is available locally and return all splits.

        Returns
        -------
        PreparedDataset
            Wrapper containing the processed dataset splits and metadata such as
            class names.
        """

        if not self._is_data_cached():
            self.logger.info("Local CIFAR-10 dataset not found, downloading now.")
            dataset = self._download_and_process()
            self._save_processed_data(dataset)
        else:
            self.logger.info("Using cached CIFAR-10 dataset located in %s", self.processed_dir)

        return self._load_processed_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_data_cached(self) -> bool:
        """Return ``True`` if all processed splits are stored on disk."""

        expected_files = [
            self.processed_dir / "cifar10_train.npz",
            self.processed_dir / "cifar10_validation.npz",
            self.processed_dir / "cifar10_test.npz",
        ]

        if not all(path.exists() for path in expected_files):
            return False

        if not self.metadata_path.exists():
            return False

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return (
                np.isclose(metadata.get("val_split", -1.0), self.val_split)
                and metadata.get("seed") == self.seed
        )

    def _download_and_process(self) -> PreparedDataset:
        """Download CIFAR-10 and split it into train/validation/test."""

        train_dataset = datasets.CIFAR10(root=self.raw_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=self.raw_dir, train=False, download=True, transform=transforms.ToTensor())

        self.logger.info("Downloaded CIFAR-10 with %d training and %d test samples.", len(train_dataset),
                         len(test_dataset))

        train_images = self._normalise_images(train_dataset.data)
        train_labels = jnp.array(train_dataset.targets, dtype=jnp.int32)

        test_images = self._normalise_images(test_dataset.data)
        test_labels = jnp.array(test_dataset.targets, dtype=jnp.int32)

        train_split, val_split = self._split_train_validation(train_images, train_labels)
        test_split = DatasetSplit(images=test_images, labels=test_labels)

        splits = {
            "train": train_split,
            "validation": val_split,
            "test": test_split,
        }

        metadata: Dict[str, object] = {
            "class_names": tuple(train_dataset.classes),
        }

        return PreparedDataset(splits=splits, metadata=metadata)

    def _split_train_validation(self, images: jnp.ndarray, labels: jnp.ndarray) -> tuple[DatasetSplit, DatasetSplit]:
        """Split the training data into training and validation subsets."""

        num_samples = images.shape[0]
        validation_size = int(num_samples * self.val_split)

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(num_samples)

        val_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]

        val_images = images[val_indices]
        val_labels = labels[val_indices]
        train_images = images[train_indices]
        train_labels = labels[train_indices]

        self.logger.info(
            "Created split with %d training and %d validation samples.",
            train_images.shape[0],
            val_images.shape[0],
        )

        return DatasetSplit(images=train_images, labels=train_labels), DatasetSplit(images=val_images,
                                                                                    labels=val_labels)

    def _normalise_images(self, images: np.ndarray) -> jnp.ndarray:
        """Normalise image intensities and convert to ``jax`` arrays.

        The original CIFAR-10 images are stored as unsigned 8-bit integers with
        values in the range ``[0, 255]`` (256 discrete colour levels).  We
        convert them to 32-bit floats and scale by ``255`` to obtain a
        numerically stable representation within the unit interval.
        """

        normalised = images.astype(np.float32) / 255.0
        return jnp.array(normalised)

    def _save_processed_data(self, dataset: PreparedDataset) -> None:
        """Persist the processed dataset splits to disk."""

        for split_name in ("train", "validation", "test"):
            split = dataset.splits[split_name]
            np.savez_compressed(
                self.processed_dir / f"cifar10_{split_name}.npz",
                images=np.asarray(split.images),
                labels=np.asarray(split.labels),
            )

        metadata = {
            "val_split": self.val_split,
            "seed": self.seed,
            "class_names": list(dataset.metadata.get("class_names", ())),
            "split_sizes": {
                split_name: {
                    "images": int(dataset.splits[split_name].images.shape[0]),
                    "labels": int(dataset.splits[split_name].labels.shape[0]),
                }
                for split_name in ("train", "validation", "test")
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _load_processed_data(self) -> PreparedDataset:
        """Load the processed data from disk and convert them back to ``jax`` arrays."""

        loaded_splits: Dict[str, DatasetSplit] = {}
        for split_name in ("train", "validation", "test"):
            path = self.processed_dir / f"cifar10_{split_name}.npz"
            with np.load(path) as npz_file:
                images = jnp.array(npz_file["images"], dtype=jnp.float32)
                labels = jnp.array(npz_file["labels"], dtype=jnp.int32)
            loaded_splits[split_name] = DatasetSplit(images=images, labels=labels)

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return PreparedDataset(splits=loaded_splits, metadata=metadata)
