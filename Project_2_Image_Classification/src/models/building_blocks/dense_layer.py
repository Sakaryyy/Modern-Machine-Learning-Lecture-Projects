"""Dense building blocks shared by model architectures."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from Project_2_Image_Classification.src.models.building_blocks.common import InitializerFn, resolve_activation, \
    resolve_initializer

__all__ = [
    "DenseBlockConfig",
    "DenseBlock",
]


@dataclass(slots=True)
class DenseBlockConfig:
    """Configuration container describing a fully connected block.

    Parameters
    ----------
    features:
        Number of output units produced by the dense layer.
    activation:
        String identifying the non-linearity applied after the dense layer.
    dropout_rate:
        Dropout probability.  A value of ``0.0`` disables dropout entirely.
    use_bias:
        Whether the dense layer should learn a bias term.
    kernel_init:
        Initialisation scheme used for the dense layer weights.  The value can
        be either a callable following the Flax initializer protocol or a
        string referencing a curated registry (e.g. ``"he_normal"`` or
        ``"xavier_uniform"``).
    bias_init:
        Initialisation scheme used for the bias parameter.  Defaults to zeros.
    """

    features: int
    activation: str = "relu"
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_init: str | InitializerFn = "he_normal"
    bias_init: str | InitializerFn = "zeros"

    def __post_init__(self) -> None:
        if self.features <= 0:
            raise ValueError("'features' must be a positive integer.")
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValueError("'dropout_rate' needs to be in the interval [0, 1).")


class DenseBlock(nn.Module):
    """A dense layer followed by an optional activation and dropout."""

    config: DenseBlockConfig = struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """Apply the dense block to ``inputs``.

        Parameters
        ----------
        inputs:
            Activations to be processed by the dense layer.
        train:
            Flag describing whether the block operates in training mode.

        Returns
        -------
        jax.numpy.ndarray
            Tensor after applying the dense transformation and optional dropout.
        """

        activation_fn = resolve_activation(self.config.activation)
        dense_kwargs: dict[str, object] = {
            "use_bias": self.config.use_bias,
            "kernel_init": resolve_initializer(self.config.kernel_init),
        }
        if self.config.use_bias:
            dense_kwargs["bias_init"] = resolve_initializer(self.config.bias_init)

        x = nn.Dense(self.config.features, **dense_kwargs)(inputs)
        x = activation_fn(x)

        if self.config.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)

        return x
