r"""Multinomial logistic (softmax) regression implemented in JAX.

The implementation mirrors the mathematical development from the lecture notes.
For a design matrix :math:`X \in \mathbb{R}^{n \times p}` and discrete class labels
``y`` taking values in ``{0, …, K-1}``, we model the conditional distribution as

.. math::

   P_\theta(y = k \mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=0}^{K-1} \exp(w_j^\top x + b_j)},

where ``w_k`` is the ``k``-th column of the weight matrix ``W`` and ``b_k`` is the
``k``-th intercept. The loss optimised is the average negative log-likelihood with
an optional :math:`\ell_2` penalty on the weights. Gradient descent is carried out
in full-batch mode which keeps the correspondence with the analytical gradients
derived in class. To aid transparency we record per-iteration diagnostics (loss,
gradient norm, empirical accuracy) and expose them for downstream visualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from src.utils.preprocess import to_device_array

Array = jax.Array


@dataclass(frozen=True)
class SoftmaxTrainingRecord:
    """Diagnostics recorded at a training iteration."""

    iteration: int
    loss: float
    grad_norm: float
    accuracy: float
    log_loss: float


@dataclass(frozen=True)
class FittedSoftmaxRegression:
    """Fitted parameters and diagnostics for a multinomial logistic regression model.

    Attributes
    ----------
    weights, bias : Array
        Learned parameters.
    n_iter : int
        Number of gradient steps executed.
    loss : float
        Final objective value (penalised negative log-likelihood).
    grad_norm : float
        Euclidean norm of the gradient at the stopping iterate.
    history : tuple[SoftmaxTrainingRecord, ...]
        Optional per-iteration diagnostics (loss, gradient norm, accuracy). The
        tuple may be empty when history recording is disabled.
    """

    weights: Array
    bias: Array
    n_iter: int
    loss: float
    grad_norm: float
    dtype: jnp.dtype
    device: Optional[jax.Device]
    history: Tuple[SoftmaxTrainingRecord, ...]


@dataclass
class SoftmaxRegression:
    r"""Multiclass logistic regression solved via full-batch gradient descent.

    Objective
    ---------
    For feature matrix :math:`X \in R^{n \times p}` and class labels
    :math:`y \in {0, ..., K-1}`, the model parameterizes the conditional
    probabilities using weights :math:`W \in R^{p \times K}` and intercept
    :math:`b \in R^{K}`:

        P(y = k | x) = softmax_k(x^T W + b).

    The negative log-likelihood with optional :math:`\ell_2` penalty is

    .. math::
        L(W, b) = - \frac{1}{n} \sum_{i=1}^n \log P(y_i | x_i)
                  + \frac{\lambda}{2} \|W\|_F^2,

    where :math:`\lambda >= 0` is `reg_strength`. The intercept is not
    regularized. Optimization is carried out with deterministic gradient descent
    and a constant step size `learning_rate`.
    """

    n_classes: int
    reg_strength: float = 0.0
    learning_rate: float = 0.1
    max_iter: int = 500
    tol: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None
    use_64bit: bool = False
    logger: Optional[logging.Logger] = None
    log_every: int = 25
    record_history: bool = True

    def _maybe_enable_64bit(self) -> None:
        if self.use_64bit:
            jax.config.update("jax_enable_x64", True)

    def fit(self, X: Array, y: Array) -> FittedSoftmaxRegression:
        """Fit the softmax regression model.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Design matrix. Each row corresponds to an observation.
        y : jax.Array of shape (n_samples,)
            Integer-encoded labels in {0, ..., n_classes - 1}.

        Returns
        -------
        FittedSoftmaxRegression
            Trained parameters together with optimisation diagnostics. When
            ``record_history`` is True the returned object contains a trajectory
            of :class:`SoftmaxTrainingRecord` entries that can be visualised to
            confirm convergence.
        """

        self._maybe_enable_64bit()

        if X.ndim != 2:
            raise ValueError("Design matrix X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("Label vector y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of rows in X ({X.shape[0]}) must match number of labels ({y.shape[0]})"
            )
        if self.n_classes <= 1:
            raise ValueError("n_classes must be at least 2 for multinomial logistic regression")

        X_dev = to_device_array(X, dtype=self.dtype, device=self.device, check_finite=True)
        y_dev = to_device_array(y, dtype=jnp.int32, device=self.device, check_finite=True)

        n_samples, n_features = X_dev.shape
        if n_samples == 0:
            raise ValueError("Cannot fit softmax regression on an empty dataset")

        W = jnp.zeros((n_features, self.n_classes), dtype=self.dtype)
        b = jnp.zeros((self.n_classes,), dtype=self.dtype)

        y_onehot = jax.nn.one_hot(y_dev, self.n_classes, dtype=self.dtype)

        reg = jnp.asarray(self.reg_strength, dtype=self.dtype)
        lr = jnp.asarray(self.learning_rate, dtype=self.dtype)

        def loss_fn(params: Tuple[Array, Array]) -> jnp.ndarray:
            weights, bias = params
            logits = X_dev @ weights + bias
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            nll = -jnp.mean(jnp.sum(y_onehot * log_probs, axis=1))
            reg_term = 0.5 * reg * jnp.sum(jnp.square(weights))
            return nll + reg_term

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        params = (W, b)
        last_grad_norm = jnp.inf
        final_loss = jnp.inf
        n_iter_done = 0
        history: List[SoftmaxTrainingRecord] = []
        last_logged_iter = 0

        def _log_iteration(iteration: int, loss_val: jnp.ndarray, grad_norm_val: jnp.ndarray) -> None:
            nonlocal last_logged_iter
            if self.logger is None and not self.record_history:
                return

            logits = X_dev @ params[0] + params[1]
            proba = jax.nn.softmax(logits, axis=1)
            preds = jnp.argmax(proba, axis=1)
            acc = jnp.mean((preds == y_dev).astype(self.dtype))
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            nll = -jnp.mean(log_probs[jnp.arange(y_dev.shape[0]), y_dev])

            record = SoftmaxTrainingRecord(
                iteration=iteration,
                loss=float(loss_val),
                grad_norm=float(grad_norm_val),
                accuracy=float(acc),
                log_loss=float(nll),
            )
            if self.record_history:
                history.append(record)
            if self.logger is not None:
                self.logger.info(
                    "[softmax] iter=%03d loss=%.6f grad_norm=%.3e accuracy=%.4f log_loss=%.6f",
                    iteration,
                    record.loss,
                    record.grad_norm,
                    record.accuracy,
                    record.log_loss,
                )
            last_logged_iter = iteration

        for it in range(int(self.max_iter)):
            loss_val, (gW, gb) = loss_and_grad(params)
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(gW)) + jnp.sum(jnp.square(gb)))
            params = (params[0] - lr * gW, params[1] - lr * gb)

            n_iter_done = it + 1
            final_loss = loss_val
            last_grad_norm = grad_norm

            if self.log_every > 0 and (
                    (it == 0)
                    or ((it + 1) % int(self.log_every) == 0)
                    or float(grad_norm) <= self.tol
            ):
                _log_iteration(n_iter_done, loss_val, grad_norm)

            if float(grad_norm) <= self.tol:
                break

        if self.log_every <= 0 or last_logged_iter != n_iter_done:
            _log_iteration(n_iter_done, final_loss, last_grad_norm)

        weights_f, bias_f = params
        weights_f = to_device_array(weights_f, dtype=self.dtype, device=self.device, check_finite=False)
        bias_f = to_device_array(bias_f, dtype=self.dtype, device=self.device, check_finite=False)

        return FittedSoftmaxRegression(
            weights=weights_f,
            bias=bias_f,
            n_iter=n_iter_done,
            loss=float(final_loss),
            grad_norm=float(last_grad_norm),
            dtype=self.dtype,
            device=self.device,
            history=tuple(history),
        )

    def predict_proba(self, X: Array, fit: FittedSoftmaxRegression) -> Array:
        """Predict class probabilities for a design matrix."""

        X_dev = to_device_array(X, dtype=self.dtype, device=self.device, check_finite=True)
        logits = X_dev @ fit.weights + fit.bias
        return jax.nn.softmax(logits, axis=1)

    def predict(self, X: Array, fit: FittedSoftmaxRegression) -> Array:
        """Predict the most likely class for each observation."""

        proba = self.predict_proba(X, fit)
        return jnp.argmax(proba, axis=1).astype(jnp.int32)
