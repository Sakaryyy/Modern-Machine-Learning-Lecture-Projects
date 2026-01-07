from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration container for the Game of Life transformer.

    Attributes
    ----------
    d_model : int
        Dimensionality of the token embeddings and hidden representations.
    num_heads : int
        Number of attention heads in each self attention layer.
    num_layers : int
        Number of transformer blocks to stack.
    mlp_dim : int
        Dimensionality of the hidden layer in the MLP sub block.
    dropout_rate : float
        Dropout rate used in attention and MLP.
    use_local_attention : bool
        If True restrict attention to a local window around each cell.
        If False use full global self attention.
    window_radius : int
        Radius of the local attention window in lattice coordinates.
        For Game of Life a value of 1 corresponds to the 3x3 neighborhood.
    use_coord_features : bool
        If True concatenate normalized 2D coordinates to the raw cell
        state as input features.
    use_relative_position_bias : bool
        If True add a learned 2D relative positional bias to attention
        logits based on the shortest periodic offsets in the lattice.
    max_relative_distance : int
        Maximum offset magnitude per axis for which separate relative
        bias parameters are learned. Larger offsets are clipped to this
        value.
    use_convolutional_attention : bool
        If True replace dot-product self attention with a local
        convolutional kernel shared across the lattice, providing a
        stronger inductive bias for local cellular rules.
    """

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 6
    mlp_dim: int = 512
    dropout_rate: float = 0.0
    use_local_attention: bool = True
    window_radius: int = 1
    use_coord_features: bool = False
    use_relative_position_bias: bool = True
    use_convolutional_attention: bool = True
    max_relative_distance: int = 2
