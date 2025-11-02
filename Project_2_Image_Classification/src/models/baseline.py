"""Baseline neural network models for benchmarking advanced architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from .building_blocks import resolve_activation
from ..utils.logging import get_logger

__all__ = [
    "BaselineModelConfig",
    "BaselineClassifier",
    "create_baseline_model",
]

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class BaselineModelConfig:
    """Configuration dataclass for :class:`BaselineClassifier`.

    Parameters
    ----------
    input_shape:
        Shape of a single image example provided without batch dimension.
    hidden_units:
        Size of the hidden layer in the multi-layer perceptron.
    num_classes:
        Number of output classes to predict.  The classifier returns unnormalized
        logits compatible with cross entropy losses.
    activation:
        String identifying the hidden-layer activation function.
    dropout_rate:
        Dropout probability applied after the hidden layer.  ``0.0`` disables
        dropout entirely.
    use_bias:
        Whether dense layers should learn bias parameters.
    """

    input_shape: Tuple[int, ...]
    hidden_units: int
    num_classes: int
    activation: str = "relu"
    dropout_rate: float = 0.0
    use_bias: bool = True

    def __post_init__(self) -> None:
        if not self.input_shape:
            raise ValueError("'input_shape' must not be empty.")
        if any(dimension <= 0 for dimension in self.input_shape):
            raise ValueError("All entries of 'input_shape' must be positive integers.")
        if self.hidden_units <= 0:
            raise ValueError("'hidden_units' must be a positive integer.")
        if self.num_classes <= 1:
            raise ValueError("'num_classes' must be greater than one.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("'dropout_rate' must lie within the interval [0, 1).")


class BaselineClassifier(nn.Module):
    """A shallow neural network with a single hidden layer.

    The model provides a simple baseline for the image classification task.  The
    design intentionally keeps capacity low such that more involved convolutional
    architectures can be compared against a straightforward reference model.
    """

    config: BaselineModelConfig = struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """Forward pass of the classifier.

        Parameters
        ----------
        inputs:
            Input batch of images with arbitrary trailing dimensions.  The
            tensor is automatically flattened before feeding into the hidden
            layer.
        train:
            Indicates whether the model operates in training mode which affects
            dropout behavior.

        Returns
        -------
        jax.numpy.ndarray
            Unnormalized class logits.
        """

        activation_fn = resolve_activation(self.config.activation)
        x = inputs.reshape((inputs.shape[0], -1))
        x = nn.Dense(self.config.hidden_units, use_bias=self.config.use_bias)(x)
        x = activation_fn(x)

        if self.config.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)

        logits = nn.Dense(self.config.num_classes, use_bias=self.config.use_bias)(x)
        return logits


def create_baseline_model(config: BaselineModelConfig) -> BaselineClassifier:
    """Instantiate a :class:`BaselineClassifier` from ``config``.

    Parameters
    ----------
    config:
        Configuration object describing the baseline architecture.

    Returns
    -------
    BaselineClassifier
        Configured classifier ready for initialization via Flax training
        routines.
    """

    LOGGER.info(
        "Building BaselineClassifier: input_shape=%s, hidden_units=%d, num_classes=%d",
        config.input_shape,
        config.hidden_units,
        config.num_classes,
    )
    return BaselineClassifier(config=config)
