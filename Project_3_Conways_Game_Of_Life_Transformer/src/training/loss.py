import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state


def binary_cross_entropy_with_logits(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean binary cross entropy given logits and targets.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits of arbitrary shape, for example (batch, H, W).
    targets : jnp.ndarray
        Binary targets of the same shape as `logits`, with values {0, 1}.

    Returns
    -------
    loss : jnp.ndarray
        Scalar mean binary cross entropy.
    """
    targets = targets.astype(jnp.float32)
    # Stable BCE with logits
    # max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    log_exp = jnp.logaddexp(0.0, -jnp.abs(logits))
    loss = jnp.maximum(logits, 0.0) - logits * targets + log_exp
    return loss.mean()


def log_likelihood_from_logits(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per sample log likelihood log P_theta(x' | x).

    This function assumes a factorized Bernoulli model over lattice
    sites, where the probability of a cell being alive is given by
    sigmoid(logit). The log likelihood for each sample is computed as
    the sum over all cells.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits of shape (batch_size, ...) with arbitrary trailing
        spatial dimensions, for example (batch, height, width).
    targets : jnp.ndarray
        Binary targets of the same shape as `logits` with entries in
        {0, 1}.

    Returns
    -------
    log_likelihood : jnp.ndarray
        One-dimensional array of shape (batch_size,) containing the log
        likelihood log P_theta(x' | x) for each sample in the batch.
    """
    targets_f = targets.astype(jnp.float32)

    # Numerically stable computation of log sigmoid and log(1 - sigmoid)
    # log_sigmoid = -softplus(-logit)
    log_p1 = -jax.nn.softplus(-logits)
    # log(1 - sigmoid(logit)) = log_sigmoid(-logit)
    log_p0 = -jax.nn.softplus(logits)

    log_prob_per_site = targets_f * log_p1 + (1.0 - targets_f) * log_p0

    # Sum over all non batch dimensions
    sum_axes = tuple(range(1, log_prob_per_site.ndim))
    log_likelihood = log_prob_per_site.sum(axis=sum_axes)
    return log_likelihood


def compute_log_likelihoods_dataset(
        state: train_state.TrainState,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
) -> np.ndarray:
    """Compute log likelihoods log P_theta(x' | x) for a full dataset.

    Parameters
    ----------
    state : TrainState
        Trained model state containing parameters and an apply function.
    inputs : np.ndarray
        Input grids of shape (num_samples, height, width) with entries
        in {0, 1}.
    targets : np.ndarray
        Target grids of shape (num_samples, height, width) with entries
        in {0, 1}.
    batch_size : int
        Batch size used to process the dataset.

    Returns
    -------
    log_likelihoods : np.ndarray
        One-dimensional array of shape (num_samples,) containing the log
        likelihood for each pair (x, x') under the model.
    """
    num_samples = inputs.shape[0]
    log_likelihoods = np.empty((num_samples,), dtype=np.float64)

    def batch_fn(params, x_batch, y_batch):
        logits = state.apply_fn(
            {"params": params},
            x_batch,
            train=False,
        )
        return log_likelihood_from_logits(logits, y_batch)

    batch_fn_jit = jax.jit(batch_fn)

    start = 0
    while start < num_samples:
        end = min(start + batch_size, num_samples)
        x_batch = jnp.array(inputs[start:end])
        y_batch = jnp.array(targets[start:end])

        ll_batch = batch_fn_jit(state.params, x_batch, y_batch)
        log_likelihoods[start:end] = np.array(ll_batch)

        start = end

    return log_likelihoods
