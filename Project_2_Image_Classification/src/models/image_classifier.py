"""Configurable convolutional neural network for image classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from .building_blocks import (
    ConvBlock,
    ConvBlockConfig,
    DenseBlock,
    DenseBlockConfig,
)
from ..utils.logging import get_logger

__all__ = [
    "ImageClassifierConfig",
    "ConfigurableImageClassifier",
    "create_image_classifier",
]

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ImageClassifierConfig:
    """Configuration dataclass describing the CNN architecture.

    Parameters
    ----------
    input_shape:
        Shape of an input sample excluding the batch dimension.
    num_classes:
        Number of output classes that should be predicted.
    conv_blocks:
        Iterable of :class:`ConvBlockConfig` instances describing the
        convolutional part of the architecture.
    dense_blocks:
        Iterable of :class:`DenseBlockConfig` instances forming the classifier
        head.
    classifier_dropout:
        Dropout probability applied before the final classification layer.
    global_average_pooling:
        If ``True`` the spatial dimensions are reduced by global average
        pooling.  Otherwise the tensor is flattened prior to entering the dense
        classifier head.
    """

    input_shape: Tuple[int, int, int]
    num_classes: int
    conv_blocks: Sequence[ConvBlockConfig]
    dense_blocks: Sequence[DenseBlockConfig] = ()
    classifier_dropout: float = 0.0
    global_average_pooling: bool = True

    def __post_init__(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("'input_shape' must contain (height, width, channels).")
        if any(dimension <= 0 for dimension in self.input_shape):
            raise ValueError("All entries of 'input_shape' must be positive integers.")
        if self.num_classes <= 1:
            raise ValueError("'num_classes' must be greater than one.")
        if not isinstance(self.conv_blocks, Sequence) or not self.conv_blocks:
            raise ValueError("'conv_blocks' must be a non-empty sequence of ConvBlockConfig instances.")
        if not all(isinstance(block, ConvBlockConfig) for block in self.conv_blocks):
            raise TypeError("All entries of 'conv_blocks' need to be ConvBlockConfig instances.")
        if not all(isinstance(block, DenseBlockConfig) for block in self.dense_blocks):
            raise TypeError("All entries of 'dense_blocks' need to be DenseBlockConfig instances.")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("'classifier_dropout' must lie within [0, 1).")

        object.__setattr__(self, "conv_blocks", tuple(self.conv_blocks))
        object.__setattr__(self, "dense_blocks", tuple(self.dense_blocks))


class ConfigurableImageClassifier(nn.Module):
    """Flexible CNN architecture assembled from reusable building blocks."""

    config: ImageClassifierConfig = struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """Run a forward pass through the configurable CNN.

        Parameters
        ----------
        inputs:
            Batch of input images with shape ``(batch, height, width, channels)``.
        train:
            Indicates whether the network operates in training mode.

        Returns
        -------
        jax.numpy.ndarray
            Logits predicted for each class.
        """

        x = inputs
        for index, block_config in enumerate(self.config.conv_blocks):
            x = ConvBlock(config=block_config, name=f"conv_block_{index}")(x, train=train)

        if self.config.global_average_pooling:
            x = x.mean(axis=(1, 2))
        else:
            x = x.reshape((x.shape[0], -1))

        for index, block_config in enumerate(self.config.dense_blocks):
            x = DenseBlock(config=block_config, name=f"dense_block_{index}")(x, train=train)

        if self.config.classifier_dropout > 0.0:
            x = nn.Dropout(rate=self.config.classifier_dropout, name="classifier_dropout")(x, deterministic=not train)

        logits = nn.Dense(self.config.num_classes, name="classifier_head")(x)
        return logits


def create_image_classifier(config: ImageClassifierConfig) -> ConfigurableImageClassifier:
    """Create a :class:`ConfigurableImageClassifier` based on ``config``.

    Parameters
    ----------
    config:
        Object containing the architecture description.

    Returns
    -------
    ConfigurableImageClassifier
        CNN ready to be initialized and trained with Flax.
    """

    LOGGER.info(
        "Building ConfigurableImageClassifier with %d conv blocks and %d dense blocks",
        len(config.conv_blocks),
        len(config.dense_blocks),
    )
    return ConfigurableImageClassifier(config=config)
