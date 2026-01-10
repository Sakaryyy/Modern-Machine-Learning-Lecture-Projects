"""
Analysis utilities for parameter counts and memory estimates.
"""

from typing import Tuple

import jax
import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import ModelConfig


def count_parameters(params) -> int:
    """Count the total number of scalar parameters in a tree.

    Parameters
    ----------
    params :
        PyTree of JAX arrays such as the `params` field in a Flax state.

    Returns
    -------
    num_params : int
        Total number of scalar entries in the parameter tree.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(leaf.shape) for leaf in leaves))


def estimate_parameter_memory(
        num_params: int,
        bytes_per_param: int = 4,
) -> float:
    """Estimate memory usage of model parameters in megabytes.

    Parameters
    ----------
    num_params : int
        Number of scalar parameters.
    bytes_per_param : int, optional
        Bytes per parameter. Four corresponds to float32.

    Returns
    -------
    memory_mb : float
        Estimated parameter memory in megabytes.
    """
    bytes_total = num_params * bytes_per_param
    return bytes_total / (1024.0 * 1024.0)


def estimate_attention_activation_memory(
        batch_size: int,
        height: int,
        width: int,
        config: ModelConfig,
        bytes_per_float: int = 4,
) -> Tuple[float, float]:
    """Estimate memory usage for attention activations per forward pass.

    This function provides a rough upper bound for the memory consumed
    by Q, K, V, and attention weights for a single forward pass with
    full self attention or local attention. It multiplies by the number
    of transformer layers.

    Parameters
    ----------
    batch_size : int
        Batch size.
    height : int
        Lattice height.
    width : int
        Lattice width.
    config : ModelConfig
        Transformer configuration, used for d_model, num_heads, number
        of layers, and local attention settings.
    bytes_per_float : int, optional
        Number of bytes per floating point value, typically 4 for
        float32.

    Returns
    -------
    memory_per_layer_mb : float
        Estimated memory usage in megabytes for the attention related
        activations of a single layer.
    memory_total_mb : float
        Estimated memory usage in megabytes summed over all layers.
    """
    L = height * width
    d = config.d_model
    h = config.num_heads

    # Q, K, V activations: 3 * B * L * d
    qkv_elems = 3 * batch_size * L * d

    # Attention weights: B * H * L * effective_L
    if config.use_local_attention:
        # Effective length per query is approximately the local window size
        window_side = 2 * config.window_radius + 1
        effective_L = window_side * window_side
    else:
        effective_L = L

    attn_elems = batch_size * h * L * effective_L

    elems_per_layer = qkv_elems + attn_elems
    memory_per_layer_mb = elems_per_layer * bytes_per_float / (1024.0 * 1024.0)
    memory_total_mb = memory_per_layer_mb * config.num_layers

    return memory_per_layer_mb, memory_total_mb


def estimate_attention_flops(
        batch_size: int,
        height: int,
        width: int,
        config: ModelConfig,
) -> float:
    """Estimate the number of floating point operations per forward pass.

    This function returns a rough count of floating point operations for
    the attention mechanism in a single layer of the transformer,
    multiplied by the number of layers. It is not meant to be exact
    wallclock timing but rather a relative complexity indicator.

    Parameters
    ----------
    batch_size : int
        Batch size.
    height : int
        Lattice height.
    width : int
        Lattice width.
    config : ModelConfig
        Transformer configuration.

    Returns
    -------
    flops : float
        Estimated number of floating point operations.
    """
    L = height * width
    d = config.d_model
    h = config.num_heads
    d_head = d // h

    if config.use_local_attention:
        window_side = 2 * config.window_radius + 1
        effective_L = window_side * window_side
    else:
        effective_L = L

    # QK^T: B * H * L * effective_L * d_head multiplications and additions
    flops_qk = 2.0 * batch_size * h * L * effective_L * d_head

    # Attention * V: B * H * L * effective_L * d_head
    flops_av = 2.0 * batch_size * h * L * effective_L * d_head

    # Multiply by number of layers
    flops = (flops_qk + flops_av) * config.num_layers
    return flops
