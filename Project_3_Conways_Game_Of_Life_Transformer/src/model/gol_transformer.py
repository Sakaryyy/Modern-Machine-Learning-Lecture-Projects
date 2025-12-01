"""
Game of Life transformer.
"""

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig


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


def make_local_attention_mask(
        height: int,
        width: int,
        radius: int,
) -> jnp.ndarray:
    """Build a local additive attention mask for a 2D periodic lattice.

    The mask encodes which sites are allowed to attend to which others.
    Each site is allowed to attend to all sites within a square window
    of given radius in both directions. Mask entries are 0.0 where
    attention is allowed and a large negative value where it is masked.

    Periodic boundary conditions are used, so the lattice is treated as
    a torus.

    Parameters
    ----------
    height : int
        Height of the lattice.
    width : int
        Width of the lattice.
    radius : int
        Window radius. A radius of 1 corresponds to a 3x3 neighborhood.

    Returns
    -------
    mask : jnp.ndarray
        Attention mask of shape (1, 1, L, L) where L = height * width.
        Entries are 0.0 for allowed pairs and a large negative value
        for disallowed pairs.
    """
    h = height
    w = width
    l = h * w

    ys = jnp.arange(h)
    xs = jnp.arange(w)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.stack([yy, xx], axis=-1).reshape(l, 2)  # (L, 2)

    coords_i = coords[:, None, :]
    coords_j = coords[None, :, :]

    dy = jnp.abs(coords_i[..., 0] - coords_j[..., 0])
    dx = jnp.abs(coords_i[..., 1] - coords_j[..., 1])

    dy = jnp.minimum(dy, h - dy)
    dx = jnp.minimum(dx, w - dx)

    allowed = (dy <= radius) & (dx <= radius)
    mask_2d = jnp.where(allowed, 0.0, -1e9)  # additive bias
    mask = mask_2d[None, None, :, :]  # (1, 1, L, L)
    return mask


class RelativePositionBias(nn.Module):
    """Learned relative positional bias for self attention.

    This module assigns a learned bias to each pair of positions based
    on their relative distance in the flattened 1D sequence. The bias is
    shared across layers and is added to the attention logits.

    Attributes
    ----------
    num_heads : int
        Number of attention heads. A separate bias is learned for each
        head.
    max_distance : int
        Maximum relative distance treated distinctly. Distances with
        absolute value larger than this are clipped.
    """

    num_heads: int
    max_distance: int

    @nn.compact
    def __call__(self, length: int) -> jnp.ndarray:
        """Compute the relative positional bias for a sequence.

        Parameters
        ----------
        length : int
            Sequence length L.

        Returns
        -------
        bias : jnp.ndarray
            Additive bias of shape (1, num_heads, L, L) that can be
            added to the attention mask. Entries are real numbers that
            shift the attention logits before the softmax.
        """
        # Relative position indices from i-j
        pos = jnp.arange(length)
        rel = pos[:, None] - pos[None, :]  # (L, L)
        rel = jnp.clip(rel, -self.max_distance, self.max_distance)

        # Map to [0, 2*max_distance]
        offset = self.max_distance
        rel_bucket = (rel + offset).astype(jnp.int32)

        num_buckets = 2 * self.max_distance + 1

        # Parameter shape (num_heads, num_buckets)
        rel_emb = self.param(
            "rel_embedding",
            nn.initializers.zeros,
            (self.num_heads, num_buckets),
        )

        # Look up per head and position pair
        # rel_bucket: (L, L), rel_emb: (H, B)
        bias = rel_emb[:, rel_bucket]  # (H, L, L)
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

        # Raw cell state as float feature
        x_float = x.astype(jnp.float32)[..., None]  # (B, H, W, 1)

        feature_list = [x_float]

        if self.config.use_coord_features:
            coords = build_coordinate_features(height, width)  # (H, W, 2)
            coords = jnp.broadcast_to(coords, (batch_size, height, width, 2))
            feature_list.append(coords)

        features = jnp.concatenate(feature_list, axis=-1)  # (B, H, W, C)
        features = features.reshape(batch_size, length, features.shape[-1])

        h = nn.Dense(
            self.config.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(features)

        # Optional local attention mask
        if self.config.use_local_attention:
            mask = make_local_attention_mask(
                height=height,
                width=width,
                radius=self.config.window_radius,
            )
        else:
            mask = None

        # Optional relative positional bias
        attn_bias = None
        if self.config.use_relative_position_bias:
            attn_bias = RelativePositionBias(
                num_heads=self.config.num_heads,
                max_distance=self.config.max_relative_distance,
            )(length)

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
