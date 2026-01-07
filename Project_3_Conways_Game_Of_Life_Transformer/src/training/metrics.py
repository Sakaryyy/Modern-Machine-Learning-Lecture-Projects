"""Metric utilities for evaluating binary Game of Life predictions."""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def accuracy_from_logits(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean cellwise accuracy from logits and binary targets.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits of shape (batch, H, W) or similar.
    targets : jnp.ndarray
        Binary targets of the same shape.

    Returns
    -------
    acc : jnp.ndarray
        Scalar accuracy in [0, 1].
    """
    probs = jax.nn.sigmoid(logits)
    preds = (probs >= 0.5).astype(jnp.int32)
    targets_int = targets.astype(jnp.int32)
    correct = (preds == targets_int).astype(jnp.float32)
    return correct.mean()


def balanced_accuracy_from_logits(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        eps: float = 1e-6,
) -> jnp.ndarray:
    """Compute balanced accuracy across alive/dead classes."""
    probs = jax.nn.sigmoid(logits)
    preds = (probs >= 0.5).astype(jnp.int32)
    targets_int = targets.astype(jnp.int32)

    tp = jnp.sum((preds == 1) & (targets_int == 1))
    tn = jnp.sum((preds == 0) & (targets_int == 0))
    fp = jnp.sum((preds == 1) & (targets_int == 0))
    fn = jnp.sum((preds == 0) & (targets_int == 1))

    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    return 0.5 * (tpr + tnr)


def negative_log_likelihood_scores(
        log_likelihoods: np.ndarray,
) -> np.ndarray:
    """Convert log likelihoods to anomaly scores.

    Anomalies are expected to have lower likelihood under the model,
    hence higher negative log likelihood. This function transforms
    log P_theta(x' | x) to -log P_theta(x' | x) so that high scores
    correspond to more anomalous samples.

    Parameters
    ----------
    log_likelihoods : np.ndarray
        Array of log likelihoods of shape (num_samples,).

    Returns
    -------
    scores : np.ndarray
        Array of anomaly scores of shape (num_samples,), where higher
        values indicate more probable anomalies.
    """
    return -log_likelihoods


def compute_roc_curve(
        scores: np.ndarray,
        labels: np.ndarray,
        num_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an ROC curve from anomaly scores and labels.

    This function treats higher scores as more anomalous. A sample is
    classified as anomaly if ``score >= threshold``. The thresholds are
    swept linearly over the score range and the true positive rate
    (TPR) and false positive rate (FPR) are computed for each threshold.

    Parameters
    ----------
    scores : np.ndarray
        One-dimensional array of shape (num_samples,) containing
        anomaly scores. Higher values should indicate stronger evidence
        for an anomaly, for example negative log likelihoods.
    labels : np.ndarray
        One-dimensional integer array of shape (num_samples,) with
        values 1 for anomalous samples and 0 for normal samples.
    num_thresholds : int, optional
        Number of thresholds to evaluate along the score range.

    Returns
    -------
    thresholds : np.ndarray
        Array of shape (num_thresholds,) with the evaluated thresholds.
    fpr : np.ndarray
        Array of shape (num_thresholds,) with the false positive rates.
    tpr : np.ndarray
        Array of shape (num_thresholds,) with the true positive rates.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")

    if np.unique(labels).tolist() not in ([0, 1], [1, 0]):
        raise ValueError("labels must contain both 0 (normal) and 1 (anomaly)")

    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        thresholds = np.linspace(min_score - 1.0, max_score + 1.0, num_thresholds)
    else:
        thresholds = np.linspace(min_score, max_score, num_thresholds)

    tpr = np.empty_like(thresholds)
    fpr = np.empty_like(thresholds)

    # Small epsilon to avoid division by zero in degenerate cases
    eps = 1e-12

    for i, thr in enumerate(thresholds):
        predicted_anomaly = scores >= thr

        tp = np.sum((predicted_anomaly == 1) & (labels == 1))
        fp = np.sum((predicted_anomaly == 1) & (labels == 0))
        fn = np.sum((predicted_anomaly == 0) & (labels == 1))
        tn = np.sum((predicted_anomaly == 0) & (labels == 0))

        tpr[i] = tp / (tp + fn + eps)
        fpr[i] = fp / (fp + tn + eps)

    return thresholds, fpr, tpr


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute the area under the ROC curve using trapezoidal rule.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates in ascending order.
    tpr : np.ndarray
        Corresponding true positive rates.

    Returns
    -------
    auc : float
        Area under the ROC curve in [0, 1].
    """
    return float(np.trapezoid(tpr, fpr))


def compute_brier_score(
        y_prob: np.ndarray,
        y_true: np.ndarray,
) -> float:
    """Compute the Brier score for binary probabilistic predictions.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    y_true : np.ndarray
        Binary targets with entries in {0, 1}.

    Returns
    -------
    score : float
        Brier score, which is the mean squared error between
        probabilities and targets. Lower is better.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability curve for binary predictions.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    y_true : np.ndarray
        Binary targets with entries in {0, 1}.
    num_bins : int, optional
        Number of bins between 0 and 1.

    Returns
    -------
    bin_centers : np.ndarray
        Centers of the probability bins.
    empirical_freq : np.ndarray
        Empirical frequency of y_true = 1 in each bin. Empty bins are
        filled with NaN.
    bin_counts : np.ndarray
        Number of samples that landed in each bin.
    """
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    digitized = np.digitize(y_prob, bins) - 1
    digitized = np.clip(digitized, 0, num_bins - 1)

    empirical_freq = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        mask = digitized == b
        if np.any(mask):
            empirical_freq[b] = y_true[mask].mean()
            bin_counts[b] = mask.sum()
        else:
            empirical_freq[b] = np.nan

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, empirical_freq, bin_counts
