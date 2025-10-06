from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict, Any

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class ClassificationMetrics:
    """Container storing a set of classification metrics.

    Parameters
    ----------
    accuracy : float
        Proportion of correctly classified observations.
    misclassification : float
        Complement of accuracy, so the empirical error rate.
    log_loss : float
        Average negative log-likelihood (cross entropy) under the predicted
        class probabilities.
    mutual_information : float | None
        Estimated mutual information I(Y; X) between the discrete label and the
        input based on the modelled conditional distribution p(y | x). Expressed
        in natural logarithm. Optional if conditional probabilities are
        unavailable.
    """

    accuracy: float
    misclassification: float
    log_loss: float
    mutual_information: Optional[float]

    def as_dict(self) -> dict[str, float | None]:
        """Represent the metrics as a plain dictionary for DataFrame creation."""

        return {
            "accuracy": self.accuracy,
            "misclassification": self.misclassification,
            "log_loss": self.log_loss,
            "mutual_information_nats": self.mutual_information,
        }


def accuracy(y_true: Array, y_pred: Array) -> float:
    """Compute the proportion of exact matches between predicted and true labels."""

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in accuracy: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )
    correct = jnp.mean((y_true == y_pred).astype(jnp.float32))
    return float(correct)


def misclassification_rate(y_true: Array, y_pred: Array) -> float:
    """Return the empirical misclassification rate 1 - accuracy."""

    return 1.0 - accuracy(y_true, y_pred)


def log_loss(y_true: Array, proba: Array, *, eps: float = 1e-12) -> float:
    """Compute the multiclass log-loss (cross entropy).

    Parameters
    ----------
    y_true : Array of shape (n,)
        Ground-truth labels encoded as integers in {0, ..., K-1}.
    proba : Array of shape (n, K)
        Predicted class probabilities. Rows must sum to one (within tolerance).
    eps : float, default 1e-12
        Clipping threshold applied before taking the logarithm to guarantee
        numerical stability.
    """

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D integer vector of labels")
    if proba.ndim != 2:
        raise ValueError("proba must be a 2D matrix of shape (n_samples, n_classes)")
    if proba.shape[0] != y_true.shape[0]:
        raise ValueError("Number of probability rows must equal number of labels")

    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    proba = jnp.asarray(proba, dtype=jnp.float32)
    proba = jnp.clip(proba, eps, 1.0)
    proba = proba / jnp.sum(proba, axis=1, keepdims=True)

    log_probs = jnp.log(proba)
    nll = -jnp.mean(log_probs[jnp.arange(y_true.shape[0]), y_true])
    return float(nll)


def confusion_matrix(y_true: Array, y_pred: Array, n_classes: int) -> Array:
    """Compute a dense confusion matrix as a JAX array."""

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in confusion_matrix: {y_true.shape=} vs {y_pred.shape=}"
        )
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.int32)

    cm = jnp.zeros((n_classes, n_classes), dtype=jnp.int32)
    cm = jax.lax.fori_loop(
        0,
        y_true.shape[0],
        lambda idx, acc: acc.at[y_true[idx], y_pred[idx]].add(1),
        cm,
    )
    return cm


def mutual_information(
        y_true: Array,
        proba: Array,
        *,
        eps: float = 1e-12,
        base: Literal["nat", "bit"] = "nat",
        return_parts: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Estimate the mutual information I(Y; X) using model probabilities.

    The estimate is

    .. math:: I(Y; X) = H(Y) - H(Y | X),

    where H(Y) is approximated by the empirical distribution of y_true and
    H(Y | X) by averaging the entropy of the predicted conditional probabilities.
    Parameters
    ----------
    y_true : Array
        Integer labels, shape (n_samples,).
    proba : Array
        Predicted class probabilities, shape (n_samples, n_classes).
    eps : float, optional (default=1e-12)
        Numerical stability epsilon for clipping probabilities.
    base : {"nat","bit"}, optional (default="nat")
        Units of information. "nat" uses natural logarithms. "bit" divides by log(2).
    return_parts : bool, optional (default=False)
        If True, return a dict with {"H_y", "H_yx", "mi", "base"} instead of just MI.

    Returns
    -------
    float or dict
        Mutual information I(Y;X) in the requested base (float), or a dict with
        components if return_parts=True.
    """

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if proba.ndim != 2 or proba.shape[0] != y_true.shape[0]:
        raise ValueError("proba must have shape (n_samples, n_classes)")

    # Normalize inputs and dtypes
    proba = jnp.asarray(proba)
    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    n_samples, n_classes = proba.shape

    # Helpers
    def _empirical_label_entropy(y: Array, K: int, *, dtype) -> Array:
        counts = jnp.bincount(y, length=K)
        p_y = counts.astype(dtype) / jnp.maximum(float(n_samples), 1.0)
        p_y = jnp.clip(p_y, eps, 1.0)
        return -jnp.sum(p_y * jnp.log(p_y))

    def _conditional_entropy_from_probs(P: Array, *, dtype) -> Array:
        P = jnp.asarray(P, dtype=dtype)
        P = jnp.clip(P, eps, 1.0)
        P = P / jnp.sum(P, axis=1, keepdims=True)
        ent_rows = -jnp.sum(P * jnp.log(P), axis=1)
        return jnp.mean(ent_rows)

    def _convert_base(x: Array, base: str) -> Array:
        if base == "bit":
            return x / jnp.log(jnp.array(2.0, dtype=x.dtype))
        return x  # nat

    # Compute components in nats first
    dtype = proba.dtype
    H_y_nat = _empirical_label_entropy(y_true, n_classes, dtype=dtype)
    H_yx_nat = _conditional_entropy_from_probs(proba, dtype=dtype)
    mi_nat = jnp.maximum(H_y_nat - H_yx_nat, 0.0)  # clamp tiny negatives

    # Convert to requested base for return
    H_y = _convert_base(H_y_nat, base)
    H_yx = _convert_base(H_yx_nat, base)
    mi = _convert_base(mi_nat, base)

    if return_parts:
        return {
            "H_y": float(H_y),
            "H_yx": float(H_yx),
            "mi": float(mi),
            "base": base,
        }
    return float(mi)
