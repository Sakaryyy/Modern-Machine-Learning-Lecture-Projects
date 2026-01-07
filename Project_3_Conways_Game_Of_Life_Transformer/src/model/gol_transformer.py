"""
Game of Life transformer.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig


def get_relative_mesh(height: int, width: int):
    """Generates the relative difference mesh for a torus grid.

    Returns:
        dy: (L, L) matrix of relative Y offsets (wrapped)
        dx: (L, L) matrix of relative X offsets (wrapped)
    """
    length = height * width
    # Use stop_gradient to ensure coordinates are treated as constants
    ys = jax.lax.stop_gradient(jnp.arange(height))
    xs = jax.lax.stop_gradient(jnp.arange(width))
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

    # Shape (L, 2)
    coords = jnp.stack([yy, xx], axis=-1).reshape(length, 2)

    coords_i = coords[:, None, :]  # (L, 1, 2)
    coords_j = coords[None, :, :]  # (1, L, 2)

    # Raw differences
    dy_raw = coords_i[..., 0] - coords_j[..., 0]
    dx_raw = coords_i[..., 1] - coords_j[..., 1]

    # Wrap around for Torus (Shortest signed difference)
    # (delta + half) % size - half
    h_half = height // 2
    w_half = width // 2

    dy = (dy_raw + h_half) % height - h_half
    dx = (dx_raw + w_half) % width - w_half

    return dy, dx


def build_coordinate_features(
        height: int,
        width: int,
) -> jnp.ndarray:
    """Construct normalized coordinate features for a 2D grid.

    The coordinates are scaled to the range [0, 1]. They are meant to be
    concatenated to the raw cell state as additional input features.

    Parameters
    ----------
    height : int
        Height of the lattice (number of rows).
    width : int
        Width of the lattice (number of columns).

    Returns
    -------
    coords : jnp.ndarray
        Array of shape (height, width, 2) containing the normalized
        y and x coordinates for each lattice site.
    """
    ys = jnp.arange(height, dtype=jnp.float32)
    xs = jnp.arange(width, dtype=jnp.float32)

    # Avoid division by zero for degenerate shapes
    y_denom = jnp.maximum(height - 1, 1)
    x_denom = jnp.maximum(width - 1, 1)

    ys = ys / y_denom
    xs = xs / x_denom

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.stack([yy, xx], axis=-1)  # (H, W, 2)
    return coords


class RelativePositionBias(nn.Module):
    """Learned 2D relative positional bias for self attention.

    This module assigns a learned bias to each pair of lattice positions
    based on their shortest periodic offsets in the y and x directions.
    Bucketing on the 2D offsets keeps the encoding invariant to the
    flattened sequence width, enabling the same spatial relationships to
    be reused when evaluating on different lattice sizes.

    Attributes
    ----------
    num_heads : int
        Number of attention heads. A separate bias is learned for each
        head.
    max_distance : int
        Maximum relative offset treated distinctly in each axis. Larger
        absolute offsets are clipped to this value per axis. NOTE: max_distance
        must be small (e.g. 3-5) so that 'far away' on a small grid maps to the
        same bucket as 'far away' on a larger grid map.
    """

    num_heads: int
    max_distance: int

    def _shortest_signed_diff(self, delta: jnp.ndarray, size: int) -> jnp.ndarray:
        """Return the minimal signed difference on a periodic axis."""

        half = size // 2
        return (delta + half) % size - half


    @nn.compact
    def __call__(self, height: int, width: int) -> jnp.ndarray:
        """Compute the relative positional bias for a 2D lattice.

        Parameters
        ----------
        height : int
            Lattice height.
        width : int
            Lattice width.

        Returns
        -------
        bias : jnp.ndarray
            Additive bias of shape (1, num_heads, L, L) for
            L = height * width that can be added to the attention mask.
        """
        dy, dx = get_relative_mesh(height, width)

        # Clip to max distance.
        max_dist = int(self.max_distance)
        dy = jnp.clip(dy, -max_dist, max_dist)
        dx = jnp.clip(dx, -max_dist, max_dist)

        # Shift to positive indices for embedding lookup [0, 2*max_dist]
        offset = max_dist
        dy_bucket = (dy + offset).astype(jnp.int32)
        dx_bucket = (dx + offset).astype(jnp.int32)

        num_buckets = 2 * max_dist + 1

        # Initialize to ZEROS
        rel_emb = self.param(
            "rel_embedding",
            nn.initializers.zeros,
            (self.num_heads, num_buckets, num_buckets),
        )

        # Look up bias
        bias = rel_emb[:, dy_bucket, dx_bucket]  # (H, L, L)
        bias = bias[None, :, :, :]  # (1, H, L, L)
        return bias


class TransformerBlock(nn.Module):
    """Single transformer encoder block.

    This consists of a self attention sub layer followed by a
    positionwise MLP, each wrapped with layer normalization and residual
    connections.

    Attributes
    ----------
    d_model : int
        Dimensionality of the input and output representations.
    num_heads : int
        Number of attention heads.
    mlp_dim : int
        Dimensionality of the hidden layer in the MLP.
    dropout_rate : float
        Dropout rate applied in attention and MLP.
    """

    d_model: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray,
            mask: Optional[jnp.ndarray],
            attn_bias: Optional[jnp.ndarray],
            train: bool,
    ) -> jnp.ndarray:
        """Apply the transformer block.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch, length, d_model).
        mask : jnp.ndarray or None
            Additive attention mask of shape (1, 1, length, length) or
            a broadcastable variant. Entries are 0.0 for allowed pairs
            and large negative values for masked pairs. If None no
            masking is applied.
        attn_bias : jnp.ndarray or None
            Additional additive bias of shape (1, num_heads, length,
            length) or broadcastable to that shape. This is typically
            used for relative positional biases. If not None it is added
            to the mask before attention.
        train : bool
            If True enable training behaviors such as dropout.

        Returns
        -------
        y : jnp.ndarray
            Output tensor of the same shape as `x`.
        """
        # Combine mask and positional bias into a single additive bias
        combined_bias = mask
        if attn_bias is not None:
            if combined_bias is None:
                combined_bias = attn_bias
            else:
                combined_bias = combined_bias + attn_bias

        # Self attention sub layer
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(y, mask=combined_bias)
        x = x + y

        # Feed forward sub layer
        y = nn.LayerNorm()(x)
        y = nn.Dense(
            self.mlp_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)

        return x + y


class GameOfLifeTransformer(nn.Module):
    """Transformer model that predicts the next Game of Life state.

    This model takes a batch of Game of Life grids of shape
    (batch, height, width) with binary entries and predicts per cell
    logits for the next time step. It uses a stack of transformer
    encoder blocks with optional local self attention on a 2D lattice.

    Attributes
    ----------
    config : TransformerConfig
        Configuration for the transformer stack.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray,
            train: bool,
    ) -> jnp.ndarray:
        """Apply the Game of Life transformer.

        Parameters
        ----------
        x : jnp.ndarray
            Input grids of shape (batch, height, width) with values in
            {0, 1}. The model internally casts to float32.
        train : bool
            If True enable training behaviors such as dropout.

        Returns
        -------
        logits : jnp.ndarray
            Logits for the next step of shape (batch, height, width).
            Apply a sigmoid to obtain per cell probabilities for being
            alive in the next time step.
        """
        batch_size, height, width = x.shape
        length = height * width

        # Embedding for state vector
        x_indices = x.astype(jnp.int32)
        features = nn.Embed(
            num_embeddings=2,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )(x_indices)  # (B, H, W, D)


        if self.config.use_coord_features:
            # Raw cell state as float feature
            x_float = x.astype(jnp.float32)[..., None]  # (B, H, W, 1)

            ys = jnp.arange(height, dtype=jnp.float32) / max(height - 1, 1)
            xs = jnp.arange(width, dtype=jnp.float32) / max(width - 1, 1)
            yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
            coords = jnp.stack([yy, xx], axis=-1)  # (H, W, 2)
            coords = jnp.broadcast_to(coords, (batch_size, height, width, 2))

            feature_list = [x_float, coords]
            features_concat = jnp.concatenate(feature_list, axis=-1)
            features = nn.Dense(self.config.d_model)(features_concat)

        features = features.reshape(batch_size, length, self.config.d_model)
        # 3. Setup Attention
        use_local = self.config.use_local_attention
        effective_radius = 1 if use_local else self.config.window_radius

        # Bias
        if self.config.use_relative_position_bias:
            attn_bias = RelativePositionBias(
                num_heads=self.config.num_heads,
                max_distance=effective_radius,
            )(height=height, width=width)
        else:
            attn_bias = None

        # Mask
        if use_local:
            dy, dx = get_relative_mesh(height, width)

            # Chebyshev distance for mask (square radius)
            dist = jnp.maximum(jnp.abs(dy), jnp.abs(dx))
            allowed = dist <= effective_radius

            # -1e9 is safe for float32 softmax
            mask = jnp.where(allowed, 0.0, -1e9)
            mask = mask[None, None, :, :]
        else:
            mask = None

        h = features
        for _ in range(self.config.num_layers):
            h = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                mlp_dim=self.config.mlp_dim,
                dropout_rate=self.config.dropout_rate,
            )(h, mask=mask, attn_bias=attn_bias, train=train)

        logits = nn.Dense(
            1,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(h)
        logits = logits.reshape(batch_size, height, width)
        return logits
