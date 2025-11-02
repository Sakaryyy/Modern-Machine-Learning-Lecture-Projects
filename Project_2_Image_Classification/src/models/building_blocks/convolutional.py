"""Reusable convolutional building blocks for configurable CNN architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from .common import resolve_activation

__all__ = [
    "ConvBlockConfig",
    "ConvBlock",
]


@dataclass(slots=True)
class ConvBlockConfig:
    """Configuration container describing a convolutional block.

    Parameters
    ----------
    features:
        Number of output channels produced by the convolutional layer.
    kernel_size:
        Size of the convolution kernel expressed as ``(height, width)``.
    strides:
        Stride applied by the convolution.
    padding:
        Padding scheme handed to :class:`flax.linen.Conv`.  The value can be
        ``"SAME"`` or ``"VALID"``.
    use_bias:
        Whether the convolution should learn a bias term.
    activation:
        String identifier pointing to an activation function registered in
        :mod:`~src.models.building_blocks.common`.
    batch_norm:
        If ``True`` the block applies batch normalization after the convolution.
    batch_norm_momentum:
        Momentum parameter for :class:`flax.linen.BatchNorm`.
    batch_norm_epsilon:
        Numerical stability constant for :class:`flax.linen.BatchNorm`.
    dropout_rate:
        Dropout probability applied after the activation function.
    pooling_type:
        Optional pooling mode.  Supports ``"max"`` and ``"avg"``.
    pooling_window:
        Spatial window used for the pooling operation.
    pooling_strides:
        Stride applied by the pooling layer.  When omitted, ``pooling_window``
        is used as stride as well.
    """

    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: str = "SAME"
    use_bias: bool = True
    activation: str = "relu"
    batch_norm: bool = True
    batch_norm_momentum: float = 0.9
    batch_norm_epsilon: float = 1e-5
    dropout_rate: float = 0.0
    pooling_type: Optional[str] = None
    pooling_window: Tuple[int, int] = (2, 2)
    pooling_strides: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.features <= 0:
            raise ValueError("'features' must be a positive integer.")
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValueError("'dropout_rate' needs to be in the interval [0, 1).")
        if self.pooling_type not in {None, "max", "avg"}:
            raise ValueError("'pooling_type' must be one of {None, 'max', 'avg'}.")
        if any(dim <= 0 for dim in self.kernel_size):
            raise ValueError("All entries of 'kernel_size' must be positive integers.")
        if any(dim <= 0 for dim in self.strides):
            raise ValueError("All entries of 'strides' must be positive integers.")
        if self.pooling_type is not None:
            if any(dim <= 0 for dim in self.pooling_window):
                raise ValueError("All entries of 'pooling_window' must be positive integers.")
            if self.pooling_strides is not None and any(dim <= 0 for dim in self.pooling_strides):
                raise ValueError("All entries of 'pooling_strides' must be positive integers.")
        padding_upper = self.padding.upper()
        if padding_upper not in {"SAME", "VALID"}:
            raise ValueError("'padding' must be either 'SAME' or 'VALID'.")
        object.__setattr__(self, "padding", padding_upper)


class ConvBlock(nn.Module):
    """A convolutional block composed of conv, normalization, activation and pooling."""

    config: ConvBlockConfig = struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """Apply the configured block to ``inputs``.

        Parameters
        ----------
        inputs:
            Batched image tensor of shape ``(batch, height, width, channels)``.
        train:
            Flag indicating whether the model is in training mode.  Influences
            batch normalization and dropout layers.

        Returns
        -------
        jax.numpy.ndarray
            Tensor containing the transformed activations.
        """

        activation_fn = resolve_activation(self.config.activation)

        x = nn.Conv(
            features=self.config.features,
            kernel_size=self.config.kernel_size,
            strides=self.config.strides,
            padding=self.config.padding,
            use_bias=self.config.use_bias,
        )(inputs)

        if self.config.batch_norm:
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=self.config.batch_norm_momentum,
                epsilon=self.config.batch_norm_epsilon,
            )(x)

        x = activation_fn(x)

        if self.config.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)

        if self.config.pooling_type is not None:
            x = _apply_pooling(
                x,
                pool_type=self.config.pooling_type,
                window_shape=self.config.pooling_window,
                strides=self.config.pooling_strides,
            )

        return x


def _apply_pooling(
        inputs: jnp.ndarray,
        *,
        pool_type: str,
        window_shape: Sequence[int],
        strides: Optional[Sequence[int]],
) -> jnp.ndarray:
    """Apply a pooling operation to ``inputs``.

    Parameters
    ----------
    inputs:
        Input activations following the convolution and non-linearity.
    pool_type:
        Either ``"max"`` or ``"avg"`` describing the pooling mode.
    window_shape:
        Size of the pooling window.
    strides:
        Optional pooling strides.  When ``None`` the window size is reused.

    Returns
    -------
    jax.numpy.ndarray
        Activations after pooling has been applied.

    Raises
    ------
    ValueError
        If an unknown pooling type is provided.
    """

    strides = tuple(strides) if strides is not None else tuple(window_shape)
    window_shape = tuple(window_shape)
    if pool_type == "max":
        return nn.max_pool(inputs, window_shape=window_shape, strides=strides, padding="SAME")
    if pool_type == "avg":
        return nn.avg_pool(inputs, window_shape=window_shape, strides=strides, padding="SAME")
    raise ValueError(f"Unsupported pooling type '{pool_type}'.")
