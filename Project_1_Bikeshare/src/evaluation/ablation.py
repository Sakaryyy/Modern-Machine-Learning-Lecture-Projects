from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal, Callable
import logging

import jax
import jax.numpy as jnp
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.linear_ridge_jax import RidgeClosedForm
from src.metrics.regression import rmse
from src.utils.targets import forward_transform
from src.utils.preprocess import standardize_design

Array = jax.Array
YMode = Literal["none", "log1p", "sqrt"]


@dataclass(frozen=True)
class AblationResult:
    """
    Result of forward feature-group ablation.

    Parameters
    ----------
    features : list of str
        Column names of the selected feature subset, in order.
    lam : float
        Ridge strength selected on the validation set for the chosen subset.
    rmse_tr : float
        Training RMSE of the chosen subset at `lam`.
    rmse_val : float
        Validation RMSE of the chosen subset at `lam`.
    """
    features: List[str]
    lam: float
    rmse_tr: float
    rmse_val: float


def forward_ablation(
    df_tr: pd.DataFrame,
    y_tr: Array,
    df_val: pd.DataFrame,
    y_val: Array,
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    lam_grid: Sequence[float],
    epsilon: float = 0.01,
    *,
    y_transform: YMode,
    lam_floor: float = 1e-8,
    can_select: Optional[Callable[[set[str], List[str]], bool]] = None,
    record_trace: bool = False,
    logger: Optional[logging.Logger] = None,
) -> AblationResult | Tuple[AblationResult, pd.DataFrame]:
    """
    Greedy forward selection over feature groups with ridge lambda search.

    Procedure
    ---------
    1) Start with no groups selected.
    2) At each iteration, tentatively add each remaining group to the current set,
       build the design matrices, and sweep over `lam_grid`. For each trial subset,
       fit a ridge model on train and score RMSE on validation. Keep the trial
       that yields the smallest validation RMSE at its best lambda.
    3) Append that best group to the selected set and repeat until all groups
       have been considered.
    4) Let best_val be the lowest validation RMSE ever achieved by any trial along
       the greedy path. Return the smallest prefix of the selected groups whose
       validation RMSE is within (1 + epsilon) * best_val (re-sweeping `lam_grid`
       to get the best lambda and to compute the corresponding training RMSE).

    Parameters
    ----------
    df_tr : pandas.DataFrame
        Training split features in tabular form. Each group pipeline will read
        from this frame and output a JAX design matrix.
    y_tr : jax.Array
        Training targets with shape (n_train,) or (n_train, 1). Converted to a
        vector internally.
    df_val : pandas.DataFrame
        Holdout or validation split features in tabular form.
    y_val : jax.Array
        Validation targets with shape (n_val,) or (n_val, 1). Converted to a
        vector internally.
    candidate_groups : list of (group_names, FeaturePipeline)
        Each entry represents a small, interpretable group of features. The
        pipeline is expected to emit a JAX matrix and the list of column names.
    lam_grid : sequence of float
        Search grid for the ridge penalty. Must be non-empty and contain
        non-negative values.
    epsilon : float, default 0.01
        Relative tolerance for the final minimal subset selection:
        choose the smallest subset whose validation RMSE <= (1 + epsilon) * best_val.
    record_trace : bool, default False
        If True, returns a second value: a pandas DataFrame trace with one row
        per step, summarizing choices and scores.
    logger : logging.Logger or None
        If provided, logs per-iteration selections and scores.

    Returns
    -------
    AblationResult
        Selected feature names, chosen lambda, and the training/validation RMSE.

    Raises
    ------
    ValueError
        If inputs are inconsistent, `lam_grid` is empty, or `candidate_groups`
        is empty.
    """
    if df_tr.empty or df_val.empty:
        raise ValueError("Training and validation DataFrames must be non-empty.")
    if len(candidate_groups) == 0:
        raise ValueError("candidate_groups must be non-empty.")
    if len(lam_grid) == 0:
        raise ValueError("lam_grid must be non-empty.")
    if any(l < 0.0 for l in lam_grid):
        raise ValueError("lam_grid must contain non-negative values.")

    y_tr = _as_vector(forward_transform(y_tr, y_transform))
    y_val = _as_vector(forward_transform(y_val, y_transform))

    # Helper to compose the design from a list of pipelines.
    def design(pipes: List[FeaturePipeline], df: pd.DataFrame) -> Tuple[Array, List[str]]:
        if not pipes:
            # Zero-column JAX matrix with n rows
            return jnp.empty((len(df), 0), dtype=y_tr.dtype), []
        mats: List[Array] = []
        cols_all: List[str] = []
        for p in pipes:
            Xp, cols = p.transform(df)
            if not jnp.isfinite(Xp).all():
                raise ValueError("Non-finite values in a feature pipeline output.")
            if Xp.ndim != 2 or Xp.shape[0] != len(df):
                raise ValueError(f"Pipeline must return (n, m). Got {Xp.shape} for n={len(df)}.")
            mats.append(Xp)
            cols_all.extend(cols)
        X = jnp.concatenate(mats, axis=1) if mats else jnp.empty((len(df), 0), dtype=y_tr.dtype)
        return X, cols_all

    # Greedy selection
    selected: List[Tuple[List[str], FeaturePipeline]] = []
    remaining = candidate_groups.copy()

    # Track the best validation RMSE achieved by any greedy trial subset,
    # along with the lambda and the corresponding full column list of that subset.
    best_overall_val = float("inf")
    best_overall_lam = float("nan")
    best_overall_cols: List[str] = []

    trace_rows: List[dict] = []
    step = 0

    while remaining:
        step += 1
        selected_names_set: set[str] = set()
        for names, _ in selected:
            selected_names_set.update(names)

        trial_records: List[
            Tuple[float, float, float, List[str], List[str], FeaturePipeline]
        ] = []
        # tuple fields:
        # (rmse_val, lam, rmse_tr, trial_cols, candidate_group_names, candidate_pipe)

        for group_names, pipe in remaining:
            if can_select is not None:
                try:
                    if not can_select(selected_names_set, group_names):
                        if logger:
                            logger.debug(
                                "Skipping group this round due to dependency: +{%s}",
                                ", ".join(group_names)
                            )
                        continue  # keep it for later rounds
                except Exception as e:
                    if logger:
                        logger.warning("Dependency function failed for +{%s}: %s",
                                       ", ".join(group_names), e)
                    continue

            trial_pipes = [p for _, p in selected] + [pipe]
            try:
                Xtr_raw, trial_cols = design(trial_pipes, df_tr)
                Xva_raw, _ = design(trial_pipes, df_val)
                Xtr, Xva, _Xte_dummy, mu, sd, is_bin = standardize_design(Xtr_raw, Xva_raw, Xva_raw)

                best_for_trial = None
                nan_flag_for_group = False
                for lam in lam_grid:
                    lam_eff = float(max(lam, lam_floor))
                    model = RidgeClosedForm(
                        lam=lam_eff,
                        fit_intercept=True,
                        solver="auto",
                        jitter=1e-6,
                        dtype=Xtr.dtype,
                        use_64bit=(Xtr.dtype == jnp.float64),
                    )
                    fit = model.fit(Xtr, y_tr)
                    y_tr_pred = model.predict(Xtr, fit)
                    y_va_pred = model.predict(Xva, fit)
                    if not (jnp.isfinite(y_tr_pred).all() and jnp.isfinite(y_va_pred).all()):
                        nan_flag_for_group = True
                        continue
                    s_tr = rmse(y_tr, y_tr_pred)
                    s_va = rmse(y_val, y_va_pred)
                    rec = (s_va, float(lam), s_tr, trial_cols, group_names, pipe)
                    if best_for_trial is None or s_va < best_for_trial[0]:
                        best_for_trial = rec

                if best_for_trial is None:
                    # All lambdas non-finite -> mark as +inf so it is never chosen
                    if logger and nan_flag_for_group:
                        logger.warning(
                            "Skipping group due to non-finite scores at all lambdas: +{%s}",
                            ", ".join(group_names),
                        )
                    trial_records.append((float("inf"), float("nan"), float("inf"), trial_cols, group_names, pipe))
                else:
                    trial_records.append(best_for_trial)
            except Exception as e:
                if logger:
                    logger.warning("Trial failed for group +{%s}: %s", ", ".join(group_names), e)
                trial_records.append((float("inf"), float("nan"), float("inf"), [], group_names, pipe))

        if not trial_records:
            if logger:
                logger.info("No feasible groups in this iteration, stopping greedy selection.")
            break

        # Choose the group whose addition gives the lowest validation RMSE.
        trial_records.sort(key=lambda t: t[0])
        best_add = trial_records[0]
        rmse_val_star, lam_star, rmse_tr_star, trial_cols_star, group_names_star, pipe_star = best_add

        # Append only the candidate group itself to the selection.
        selected.append((group_names_star, pipe_star))

        if rmse_val_star < best_overall_val:
            best_overall_val = rmse_val_star
            best_overall_lam = lam_star
            best_overall_cols = trial_cols_star

        cumulative_ncols = len(trial_cols_star)
        if logger is not None:
            logger.info(
                "Ablation step %d: +{%s} (cols=%d) -> val RMSE=%s at lambda=%s "
                "(best overall so far=%.4f)",
                step,
                ", ".join(group_names_star),
                cumulative_ncols,
                f"{rmse_val_star:.4f}" if jnp.isfinite(rmse_val_star) else "inf",
                f"{lam_star:.4g}" if jnp.isfinite(lam_star) else "nan",
                best_overall_val if jnp.isfinite(best_overall_val) else float("inf"),
            )

        if record_trace:
            trace_rows.append(
                {
                    "step": step,
                    "chosen_group": ", ".join(group_names_star),
                    "chosen_group_ncols": len(group_names_star),
                    "chosen_lambda": lam_star,
                    "chosen_rmse_val": rmse_val_star,
                    "chosen_rmse_tr": rmse_tr_star,
                    "best_overall_rmse_val_so_far": best_overall_val,
                    "cumulative_ncols": cumulative_ncols,
                }
            )

        remaining = [g for g in remaining if g[0] != group_names_star]

    # Target threshold for minimal subset within tolerance.
    target = (1.0 + float(epsilon)) * best_overall_val

    # Re-evaluate prefixes of the selected list and return the smallest that meets the target.
    pipes_acc: List[FeaturePipeline] = []
    for k, (group_names, pipe) in enumerate(selected, start=1):
        pipes_acc.append(pipe)
        Xtr_raw, cols = design(pipes_acc, df_tr)
        Xva_raw, _ = design(pipes_acc, df_val)
        Xtr, Xva, _dummy, mu, sd, is_bin = standardize_design(Xtr_raw, Xva_raw, Xva_raw)

        best_subset = None
        for lam in lam_grid:
            lam_eff = float(max(lam, lam_floor))
            model = RidgeClosedForm(
                lam=lam_eff,
                fit_intercept=True,
                solver="auto",
                jitter=1e-6,
                dtype=Xtr.dtype,
                use_64bit=(Xtr.dtype == jnp.float64)
            )
            fit = model.fit(Xtr, y_tr)
            y_tr_pred = model.predict(Xtr, fit)
            y_va_pred = model.predict(Xva, fit)

            if not (jnp.isfinite(y_tr_pred).all() and jnp.isfinite(y_va_pred).all()):
                continue

            s_tr = rmse(y_tr, y_tr_pred)
            s_va = rmse(y_val, y_va_pred)
            rec = (s_va, float(lam), s_tr)
            if best_subset is None or s_va < best_subset[0]:
                best_subset = rec

        if best_subset is None:
            continue

        s_va_star, lam_star, s_tr_star = best_subset

        if logger is not None:
            logger.info(
                "Prefix k=%d (cols=%d) -> val RMSE=%.4f at lambda=%.4g; target (1+eps)*best=%.4f",
                k, len(cols), s_va_star, lam_star, target
            )

        if s_va_star <= target:
            result = AblationResult(features=cols, lam=lam_star, rmse_tr=s_tr_star, rmse_val=s_va_star)
            if record_trace:
                return result, pd.DataFrame(trace_rows)
            return result

    # Fallback: return the best observed subset across all greedy trials.
    # Rebuild the subset that produced best_overall_cols to compute rmse_tr.
    # Fallback: return the global best observed along the greedy path
    result = AblationResult(features=best_overall_cols, lam=best_overall_lam,
                            rmse_tr=float("nan"), rmse_val=best_overall_val)
    if record_trace:
        return result, pd.DataFrame(trace_rows)
    return result


# Utilities
def _as_vector(y: Array) -> Array:
    """
    Ensure target is a one-dimensional vector on whatever device it currently is.

    Parameters
    ----------
    y : jax.Array
        Target with shape (n,) or (n, 1).

    Returns
    -------
    jax.Array
        A vector of shape (n,).

    Raises
    ------
    ValueError
        If y has rank greater than 2 or has more than one column.
    """
    y = jnp.asarray(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError(f"y must be shape (n,) or (n, 1). Received shape {y.shape}.")
