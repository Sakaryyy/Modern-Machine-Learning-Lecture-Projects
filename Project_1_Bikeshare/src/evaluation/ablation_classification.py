from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.metrics.classification import (
    accuracy,
    log_loss,
)
from src.models.softmax_regression_jax import SoftmaxRegression
from src.utils.preprocess import standardize_design

Array = jax.Array


@dataclass(frozen=True)
class ClassificationAblationResult:
    """Summary of the greedy forward ablation outcome."""

    features: List[str]
    reg_strength: float
    step: int
    train_accuracy: float
    val_accuracy: float
    train_misclassification: float
    val_misclassification: float
    train_log_loss: float
    val_log_loss: float


def _build_design(
    pipes: List[FeaturePipeline],
    df: pd.DataFrame,
    *,
    dtype: jnp.dtype,
) -> Tuple[Array, List[str]]:
    if not pipes:
        empty = jnp.empty((len(df), 0), dtype=dtype)
        return empty, []

    mats: List[Array] = []
    names: List[str] = []
    for pipe in pipes:
        X, cols = pipe.transform(df)
        if X.ndim != 2 or X.shape[0] != len(df):
            raise ValueError("Each pipeline step must return a 2D matrix with matching rows")
        mats.append(X)
        names.extend(cols)
    design = jnp.concatenate(mats, axis=1) if mats else jnp.empty((len(df), 0), dtype=dtype)
    return design, names


def forward_ablation_classification(
    *,
    df_tr: pd.DataFrame,
    y_tr: Array,
    df_val: pd.DataFrame,
    y_val: Array,
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    reg_grid: Sequence[float],
    epsilon: float,
    n_classes: int,
    dtype: jnp.dtype,
    device: jax.Device | None,
    preserve_columns: Optional[set[str]] = None,
    max_iter: int = 500,
    learning_rate: float = 0.1,
    tol: float = 1e-6,
    record_trace: bool = False,
    logger: Optional[logging.Logger] = None,
) -> ClassificationAblationResult | Tuple[ClassificationAblationResult, pd.DataFrame]:
    """Run greedy forward ablation tailored to softmax regression."""

    if not candidate_groups:
        raise ValueError("candidate_groups must be non-empty")
    if not reg_grid:
        raise ValueError("reg_grid must be non-empty")
    if df_tr.empty or df_val.empty:
        raise ValueError("Training and validation frames must be non-empty")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")

    remaining = candidate_groups.copy()
    selected: List[Tuple[List[str], FeaturePipeline]] = []
    selected_features_history: List[List[str]] = []

    best_overall = float("inf")
    trace_rows: List[dict] = []

    step = 0
    while remaining:
        step += 1
        best_record = None

        for names, pipe in remaining:
            trial_pipes = [p for _, p in selected] + [pipe]
            X_tr_raw, trial_cols = _build_design(trial_pipes, df_tr, dtype=dtype)
            X_val_raw, _ = _build_design(trial_pipes, df_val, dtype=dtype)

            preserve_mask = None
            if preserve_columns:
                preserve_mask = jnp.array([col in preserve_columns for col in trial_cols], dtype=bool)

            X_tr_std, X_val_std, _, _, _, _ = standardize_design(
                X_tr_raw,
                X_val_raw,
                X_val_raw,
                preserve_mask=preserve_mask,
            )

            for lam in reg_grid:
                logger.info(f"[Ablation] Fitting for pipes: {names} with lam: {lam} out of {reg_grid}")
                lam_eff = float(max(lam, 0.0))
                model = SoftmaxRegression(
                    n_classes=n_classes,
                    reg_strength=lam_eff,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    tol=tol,
                    dtype=dtype,
                    device=device,
                    logger=logger,
                    record_history=False,
                    log_every=0,
                )
                fit = model.fit(X_tr_std, y_tr)
                proba_tr = model.predict_proba(X_tr_std, fit)
                proba_val = model.predict_proba(X_val_std, fit)
                pred_tr = jnp.argmax(proba_tr, axis=1)
                pred_val = jnp.argmax(proba_val, axis=1)

                acc_val = accuracy(y_val, pred_val)
                mis_val = 1.0 - acc_val
                log_val = log_loss(y_val, proba_val)
                acc_tr = accuracy(y_tr, pred_tr)
                mis_tr = 1.0 - acc_tr
                log_tr = log_loss(y_tr, proba_tr)

                record = (
                    mis_val,
                    log_val,
                    lam_eff,
                    mis_tr,
                    log_tr,
                    acc_tr,
                    acc_val,
                    names,
                    pipe,
                )

                if best_record is None:
                    best_record = record
                else:
                    best_mis, best_log, *_ = best_record
                    if (mis_val < best_mis) or (
                        jnp.isclose(mis_val, best_mis) and log_val < best_log
                    ):
                        best_record = record

        if best_record is None:
            raise RuntimeError("Failed to select a candidate group during ablation")

        (
            mis_val,
            log_val,
            lam_eff,
            mis_tr,
            log_tr,
            acc_tr,
            acc_val,
            group_names,
            group_pipe,
        ) = best_record
        selected.append((group_names, group_pipe))
        current_features = [name for names, _ in selected for name in names]
        selected_features_history.append(list(current_features))

        best_overall = min(best_overall, mis_val)

        trace_rows.append(
            {
                "step": step,
                "chosen_group": ", ".join(group_names),
                "selected_features": ", ".join(current_features),
                "best_lambda": lam_eff,
                "val_misclassification": mis_val,
                "val_accuracy": acc_val,
                "val_log_loss": log_val,
                "train_misclassification": mis_tr,
                "train_accuracy": acc_tr,
                "train_log_loss": log_tr,
                "best_overall_val_misclassification_so_far": best_overall,
            }
        )

        if logger is not None:
            logger.info(
                "[ablation] step=%d picked=%s lambda=%.4g val_err=%.4f train_err=%.4f",
                step,
                ", ".join(group_names),
                lam_eff,
                mis_val,
                mis_tr,
            )

        remaining = [g for g in remaining if g[0] != group_names]

    trace_df = pd.DataFrame(trace_rows)

    threshold = (1.0 + float(epsilon)) * float(best_overall)
    eligible = trace_df[trace_df["val_misclassification"] <= threshold]
    if eligible.empty:
        chosen_row = trace_df.iloc[-1]
        chosen_idx = len(trace_df) - 1
    else:
        chosen_row = eligible.iloc[0]
        chosen_idx = int(chosen_row["step"]) - 1

    chosen_features = selected_features_history[chosen_idx]

    result = ClassificationAblationResult(
        features=chosen_features,
        reg_strength=float(chosen_row["best_lambda"]),
        step=int(chosen_row["step"]),
        train_accuracy=float(chosen_row["train_accuracy"]),
        val_accuracy=float(chosen_row["val_accuracy"]),
        train_misclassification=float(chosen_row["train_misclassification"]),
        val_misclassification=float(chosen_row["val_misclassification"]),
        train_log_loss=float(chosen_row["train_log_loss"]),
        val_log_loss=float(chosen_row["val_log_loss"]),
    )

    if record_trace:
        return result, trace_df
    return result