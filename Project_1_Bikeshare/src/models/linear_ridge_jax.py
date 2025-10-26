from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from src.utils.preprocess import to_device_array

Array = jax.Array


@dataclass(frozen=True)
class FittedLinear:
    """
    Fitted parameters for a linear model with optional ridge regularization.

    Parameters
    ----------
    w : jax.Array
        Weight vector of shape (n_features,).
    b : float
        Intercept term. Not penalized during fitting when fit_intercept=True.
    sigma2 : float
        Training mean squared error computed on the final fit. This is an
        empirical MSE.
    dtype : jax.numpy dtype
        Floating dtype used for w and for predictions.
    device : jax.Device or None
        Device on which w is.
    """
    w: Array
    b: float
    sigma2: float
    dtype: jnp.dtype
    device: Optional[jax.Device]


@dataclass
class RidgeClosedForm:
    """
    Closed-form ridge regression with optional intercept.

    Objective
    ---------
    We solve for w (and optionally b) by minimizing

        J(w, b) = 0.5 * || y - (Phi w + b) ||^2 + lam * ||w||^2

    where:
      - Phi is the design matrix with shape (n_samples, n_features).
      - w is the weight vector with shape (n_features,).
      - b is an intercept term that is NOT penalized when fit_intercept=True.
      - lam >= 0 is the ridge strength.

    Normal equations
    ----------------
    When fit_intercept=True, we center Phi and y:
        Phi_c = Phi - mean(Phi, axis=0)
        y_c   = y   - mean(y)

    We then solve the ridge system for w using Phi_c and y_c:

        (Phi_c^T Phi_c + 2*lam * I) w = Phi_c^T y_c

    and recover the intercept as:
        b = mean(y) - mean(Phi, axis=0) @ w

    This ensures the intercept is not penalized. When fit_intercept=False,
    the same normal equations are applied directly to Phi and y and b is set to 0.

    Params
    ----------
    lam : float, default 0.0
        Ridge strength. With the 0.5 factor in the loss, the normal equations
        include 2*lam on the diagonal.
    fit_intercept : bool, default True
        Whether to include an unpenalized intercept term via centering.
    solver : {"cholesky", "solve", "auto"}, default "auto"
        Linear system solver:
          - "cholesky": use Cholesky factorization (SPD assumption).
          - "solve": use a generic solver jnp.linalg.solve.
          - "auto": try Cholesky and fall back to solve on failure.
    jitter : float, default 1e-8
        Small positive value added to the diagonal to improve numerical stability.
    dtype : jax.numpy dtype, default jnp.float32
        Floating dtype for all computations.
    device : jax.Device or None, default None
        Device on which to place inputs, parameters, and predictions. If None,
        the JAX default device is used.
    use_64bit : bool, default False
        Whether to enable 64-bit floats in JAX. If True, this flag is set before
        arrays are created.
    """
    lam: float = 0.0
    fit_intercept: bool = True
    solver: Literal["cholesky", "solve", "auto"] = "auto"
    jitter: float = 1e-7
    dtype: jnp.dtype = jnp.float32
    device: Optional[jax.Device] = None
    use_64bit: bool = False

    # Public API
    def fit(self, Phi: Array, y: Array) -> FittedLinear:
        """
        Fit ridge regression parameters.

        Parameters
        ----------
        Phi : jax.Array
            Design matrix with shape (n_samples, n_features).
        y : jax.Array
            Target vector with shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        FittedLinear
            Fitted model with weights w, intercept b, and training MSE sigma2.

        Raises
        ------
        ValueError
            If shapes are inconsistent or inputs are empty.
        """
        self._maybe_enable_64bit()

        Phi_dev = to_device_array(Phi, dtype=self.dtype, device=self.device, check_finite=True)
        y_dev = to_device_array(y, dtype=self.dtype, device=self.device, check_finite=True)

        y_vec = self._ensure_vector(y_dev)

        if self.fit_intercept:
            Phi_mean = jnp.mean(Phi_dev, axis=0)
            y_mean = jnp.mean(y_vec)
            Phi_c = Phi_dev - Phi_mean
            y_c = y_vec - y_mean
            w = self._solve_ridge(Phi_c, y_c, lam=self.lam, jitter=self.jitter)
            b = float(y_mean - Phi_mean @ w)
        else:
            w = self._solve_ridge(Phi_dev, y_vec, lam=self.lam, jitter=self.jitter)
            b = 0.0

        resid = y_vec - (Phi_dev @ w + b)
        sigma2 = float(jnp.mean(jnp.square(resid)))

        # Ensure w lives on the configured device and dtype
        w = to_device_array(w, dtype=self.dtype, device=self.device, check_finite=False)

        return FittedLinear(w=w, b=b, sigma2=sigma2, dtype=self.dtype, device=self.device)

    def predict(self, Phi: Array, fit: FittedLinear) -> Array:
        """
        Predict targets for a given design matrix.

        Parameters
        ----------
        Phi : jax.Array
            Design matrix with shape (n_samples, n_features). Must have the same
            number of columns as the fitted weight vector.
        fit : FittedLinear
            Fitted parameters returned by `fit`.

        Returns
        -------
        jax.Array
            Predicted targets with shape (n_samples,), placed on the model device.
        """
        X = to_device_array(Phi, dtype=fit.dtype, device=fit.device, check_finite=True)
        if X.ndim != 2 or X.shape[1] != fit.w.shape[0]:
            raise ValueError(
                f"Design matrix has shape {X.shape}, but expected (n, {fit.w.shape[0]})."
            )
        yhat = X @ fit.w + fit.b
        return to_device_array(yhat, dtype=fit.dtype, device=fit.device, check_finite=False)

    # Internal helpers
    def _solve_ridge(self, X: Array, y: Array, *, lam: float, jitter: float) -> Array:
        """
        Solve (X^T X + 2*lam * I + jitter * I) w = X^T y.

        Parameters
        ----------
        X : jax.Array
            Centered or raw design matrix with shape (n, m).
        y : jax.Array
            Centered or raw target vector with shape (n,).
        lam : float
            Ridge strength. See class docstring for objective scaling.
        jitter : float
            Additional diagonal regularization to improve numerical stability.

        Returns
        -------
        jax.Array
            Weight vector w with shape (m,).
        """
        n, m = X.shape
        lam_eff = lam
        jitter_eff = jitter

        for attempt in range(3):
            # Try Cholesky
            XtX = X.T @ X
            A = XtX + jnp.diag((2.0 * lam_eff + jitter_eff) * jnp.ones((m,), dtype=self.dtype))
            Xty = X.T @ y
            try:
                L = jnp.linalg.cholesky(A)
                w = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Xty))
                if jnp.isfinite(w).all():
                    return w
            except Exception:
                pass  # fall through

            # Try generic solve
            try:
                w = jnp.linalg.solve(A, Xty)
                if jnp.isfinite(w).all():
                    return w
            except Exception:
                pass

            # Try SVD ridge (most stable)
            try:
                w = self._ridge_via_svd(X, y, lam_eff)
                if jnp.isfinite(w).all():
                    return w
            except Exception:
                pass

            # Escalate regularisation and jitter, then retry
            lam_eff = max(lam_eff * 10.0, lam_eff + 1e-3)
            jitter_eff = max(jitter_eff * 10.0, 1e-6)

        # Last resort: raise with a clear message
        raise ValueError(
            f"Ridge solve produced non-finite coefficients after escalations; "
            f"cond(X) likely huge. Tried lam up to {lam_eff:.3g}."
        )

    def _ridge_via_svd(self, X: Array, y: Array, lam: float) -> Array:
        # X = U Σ V^T ; ridge: w = V diag( Σ / (Σ^2 + 2λ) ) U^T y
        U, s, Vt = jnp.linalg.svd(X, full_matrices=False)
        denom = s * s + 2.0 * lam
        coeff = s / denom
        return Vt.T @ (coeff * (U.T @ y))

    def _validate_design_targets(self, Phi: Array, y: Array) -> Tuple[int, int]:
        if Phi.ndim != 2:
            raise ValueError(f"Phi must be 2D (n, m). Received shape {Phi.shape}.")
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D with a single column. Received shape {y.shape}.")
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"y with 2D shape must have one column. Received shape {y.shape}.")
        if Phi.shape[0] != (y.shape[0] if y.ndim == 1 else y.shape[0]):
            raise ValueError(
                f"Row count mismatch: Phi has {Phi.shape[0]} rows but y has {y.shape[0]}."
            )
        if Phi.shape[0] == 0 or Phi.shape[1] == 0:
            raise ValueError("Phi must have positive number of rows and columns.")
        return Phi.shape

    def _ensure_vector(self, y: Array) -> Array:
        return y.squeeze(-1) if y.ndim == 2 else y

    def _maybe_enable_64bit(self) -> None:
        if self.use_64bit:
            jax.config.update("jax_enable_x64", True)
