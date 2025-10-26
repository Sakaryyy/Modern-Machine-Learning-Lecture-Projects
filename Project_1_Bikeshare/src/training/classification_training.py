from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from src.config.experiment_config import ExperimentConfig, FeatureConfig
from src.data.clean import clean_bike_df, save_processed
from src.data.fetch import fetch_uci_bike
from src.data.split import DataSplits, data_split
from src.evaluation.ablation_classification import forward_ablation_classification
from src.features.pipeline import (
    FeaturePipeline,
    month_onehot_step,
    normalized_numeric_step,
    numeric_step,
    polynomial_step,
    season_onehot_step,
    weekday_onehot_step,
    weathersit_onehot_step,
)
from src.metrics.classification import (
    ClassificationMetrics,
    accuracy,
    confusion_matrix,
    log_loss,
    misclassification_rate,
    mutual_information,
)
from src.models.softmax_regression_jax import SoftmaxRegression
from src.utils.helpers import log_jax_runtime_info
from src.utils.io import save_table_xlsx
from src.utils.preprocess import standardize_design, to_device_array
from src.visualization import classification_plotting as viz_classif

logger = logging.getLogger(__name__)


def _build_classification_groups(
    feats_cfg: FeatureConfig,
    *,
    cnt_max: float,
    dtype: jnp.dtype,
    device: jax.Device | None,
    available_columns: set[str],
) -> List[Tuple[List[str], FeaturePipeline]]:
    """Construct feature groups that exclude the target hour variable."""

    groups: List[Tuple[List[str], FeaturePipeline]] = []

    if "cnt" not in available_columns:
        raise KeyError("Column 'cnt' is required to build unit-interval features.")

    groups.append(
        (
            ["cnt_unit_interval"],
            FeaturePipeline(
                steps=[normalized_numeric_step("cnt", data_min=0.0, data_max=cnt_max, dtype=dtype)],
                dtype=dtype,
                device=device,
            ),
        )
    )

    if feats_cfg.use_atemp:
        atemp_steps = [numeric_step("atemp", dtype=dtype)]
        atemp_names = ["atemp"]
        if int(feats_cfg.poly_temp_degree) >= 2:
            atemp_steps.append(
                polynomial_step(
                    "atemp",
                    degree=int(feats_cfg.poly_temp_degree),
                    include_linear=False,
                    dtype=dtype,
                )
            )
            atemp_names.extend(
                [f"atemp^{p}" for p in range(2, int(feats_cfg.poly_temp_degree) + 1)]
            )
        groups.append((atemp_names, FeaturePipeline(steps=atemp_steps, dtype=dtype, device=device)))

    if feats_cfg.use_temp and "temp" in available_columns:
        groups.append(
            (
                ["temp"],
                FeaturePipeline(steps=[numeric_step("temp", dtype=dtype)], dtype=dtype, device=device),
            )
        )

    if feats_cfg.use_humidity and "hum" in available_columns:
        groups.append(
            (
                ["hum"],
                FeaturePipeline(steps=[numeric_step("hum", dtype=dtype)], dtype=dtype, device=device),
            )
        )

    if feats_cfg.use_windspeed and "windspeed" in available_columns:
        groups.append(
            (
                ["windspeed"],
                FeaturePipeline(
                    steps=[numeric_step("windspeed", dtype=dtype)], dtype=dtype, device=device
                ),
            )
        )

    if feats_cfg.use_workingday and "workingday" in available_columns:
        groups.append(
            (
                ["workingday"],
                FeaturePipeline(steps=[numeric_step("workingday", dtype=dtype)], dtype=dtype, device=device),
            )
        )

    if feats_cfg.use_holiday and "holiday" in available_columns:
        groups.append(
            (
                ["holiday"],
                FeaturePipeline(steps=[numeric_step("holiday", dtype=dtype)], dtype=dtype, device=device),
            )
        )

    if feats_cfg.use_weekday_onehot:
        groups.append(
            (
                [f"wd_{k}" for k in range(1, 7)],
                FeaturePipeline(
                    steps=[weekday_onehot_step(drop_first=True, dtype=dtype)],
                    dtype=dtype,
                    device=device,
                ),
            )
        )

    if feats_cfg.use_month_onehot:
        groups.append(
            (
                [f"mon_{k}" for k in range(1, 12)],
                FeaturePipeline(
                    steps=[month_onehot_step(drop_first=True, dtype=dtype)],
                    dtype=dtype,
                    device=device,
                ),
            )
        )

    if feats_cfg.use_season_onehot:
        groups.append(
            (
                ["season_spring", "season_summer", "season_fall"],
                FeaturePipeline(
                    steps=[
                        season_onehot_step(
                            drop_first=True,
                            dtype=dtype,
                        )
                    ],
                    dtype=dtype,
                    device=device,
                ),
            )
        )

    if feats_cfg.use_weathersit_onehot and "weathersit" in available_columns:
        groups.append(
            (
                ["ws_mist", "ws_light", "ws_heavy"],
                FeaturePipeline(
                    steps=[weathersit_onehot_step(drop_first=True, dtype=dtype)],
                    dtype=dtype,
                    device=device,
                ),
            )
        )

    return groups


def _build_design_for_columns(
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    selected_cols: List[str],
    df: pd.DataFrame,
    *,
    dtype: jnp.dtype,
) -> Tuple[jax.Array, List[str]]:
    mats: List[jax.Array] = []
    names: List[str] = []

    for group_names, pipe in candidate_groups:
        intersection = [name for name in group_names if name in selected_cols]
        if not intersection:
            continue
        X, cols = pipe.transform(df)
        idx = [i for i, c in enumerate(cols) if c in intersection]
        if idx:
            index_array = jnp.asarray(idx, dtype=jnp.int32)
            mats.append(X[:, index_array])
            names.extend([cols[i] for i in idx])

    if mats:
        design = jnp.concatenate(mats, axis=1)
    else:
        design = jnp.empty((len(df), 0), dtype=dtype)
    return design, names


def _per_hour_accuracy_frame(
    y_true: jax.Array,
    y_pred: jax.Array,
    *,
    split: str,
    n_classes: int,
) -> pd.DataFrame:
    """Summarise accuracy per hour-of-day for a split."""

    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.int32)

    correct = (y_true == y_pred).astype(jnp.float32)
    support_counts = jnp.bincount(y_true, length=n_classes)
    support = support_counts.astype(jnp.float32)
    correct_counts = jnp.bincount(y_true, weights=correct, length=n_classes)
    accuracy_per_hour = jnp.where(support > 0.0, correct_counts / jnp.maximum(support, 1.0), jnp.nan)

    df = pd.DataFrame(
        {
            "split": split,
            "hour": jnp.arange(n_classes).tolist(),
            "support": jnp.asarray(support_counts, dtype=int).tolist(),
            "accuracy": jnp.asarray(accuracy_per_hour, dtype=float).tolist(),
        }
    )
    return df


def _true_class_probability_frame(
    y_true: jax.Array,
    proba: jax.Array,
    *,
    split: str,
) -> pd.DataFrame:
    """Return a DataFrame describing confidence assigned to the true class."""

    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    proba = jnp.asarray(proba, dtype=jnp.float32)
    idx = jnp.arange(y_true.shape[0])
    prob_true = proba[idx, y_true]
    y_pred = jnp.argmax(proba, axis=1)
    correct = jnp.asarray(y_pred == y_true, dtype=bool)
    df = pd.DataFrame(
        {
            "split": split,
            "true_class_probability": jnp.asarray(prob_true, dtype=float).tolist(),
            "correct": jnp.asarray(correct, dtype=bool).tolist(),
        }
    )
    return df


def run_classification(
    cfg: ExperimentConfig,
    *,
    reg_grid: Sequence[float],
    epsilon: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> None:
    """Execute the classification experiment using multinomial logistic regression.

    Workflow
    --------
    1. Load the processed bike-sharing dataset (or fetch/clean if missing).
    2. Chronologically split into train / validation / test partitions.
    3. Compose feature groups mirroring the regression experiment and run greedy
       forward ablation to obtain a minimal subset within ``(1 + epsilon)`` of
       the best validation misclassification rate, searching over ``reg_grid``.
    4. Fit softmax regression on the training split only to obtain clean
       diagnostics for train/validation performance and log the optimisation trace.
    5. Refit on train+validation before evaluating on the untouched test set.
    6. Save rich tabular summaries (metrics, coefficients, per-hour analysis,
       probability calibration, optimisation history) together with figures for
       confusion matrices, ablation trace, convergence diagnostics, and per-hour
       accuracy.
    """

    cfg.paths.ensure_exists()
    device = log_jax_runtime_info()
    dtype = jnp.float32

    processed_csv = cfg.paths.processed_dir / "bike_clean.csv"
    if processed_csv.exists():
        df = pd.read_csv(processed_csv, parse_dates=["timestamp"], index_col="timestamp")
        logger.info("Loaded processed dataset from %s", processed_csv)
    else:
        raw_df = fetch_uci_bike(cache_dir=cfg.paths.raw_dir, force=False)
        df = clean_bike_df(raw_df)
        save_processed(df, cfg.paths.processed_dir)
        logger.info("Fetched and cleaned dataset; saved processed copy.")

    if "hr" not in df.columns:
        raise KeyError("Column 'hr' must be present to serve as the classification target.")

    splits = DataSplits(train_end=cfg.splits.train_end, validation_end=cfg.splits.validation_end)
    train_df, validation_df, test_df = data_split(df, splits)
    logger.info("Split sizes: train=%d, validation=%d, test=%d", len(train_df), len(validation_df), len(test_df))

    feature_cols_df_tr = train_df.drop(columns=["hr"])
    feature_cols_df_va = validation_df.drop(columns=["hr"])
    feature_cols_df_te = test_df.drop(columns=["hr"])

    y_tr = to_device_array(
        train_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )
    y_va = to_device_array(
        validation_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )
    y_te = to_device_array(
        test_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )

    n_classes = int(jnp.max(jnp.concatenate([y_tr, y_va, y_te])) + 1)

    cnt_max = float(train_df["cnt"].max())
    if cnt_max <= 0.0:
        raise ValueError("Training data must contain positive bike counts to scale into [0, 1].")

    candidate_groups = _build_classification_groups(
        cfg.features,
        cnt_max=cnt_max,
        dtype=dtype,
        device=device,
        available_columns=set(train_df.columns),
    )
    logger.info("Constructed %d candidate feature groups for classification", len(candidate_groups))

    preserve_cols = {"cnt_unit_interval"}

    abl_result, abl_trace = forward_ablation_classification(
        df_tr=feature_cols_df_tr,
        y_tr=y_tr,
        df_val=feature_cols_df_va,
        y_val=y_va,
        candidate_groups=candidate_groups,
        reg_grid=reg_grid,
        epsilon=epsilon,
        n_classes=n_classes,
        dtype=dtype,
        device=device,
        preserve_columns=preserve_cols,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        record_trace=True,
        logger=logger,
    )

    logger.info(
        "Ablation selected %d feature columns across %d groups with lambda=%.4g (val acc=%.3f, val err=%.3f)",
        len(abl_result.features),
        abl_result.step,
        abl_result.reg_strength,
        abl_result.val_accuracy,
        abl_result.val_misclassification,
    )

    X_tr, selected_names = _build_design_for_columns(
        candidate_groups, abl_result.features, feature_cols_df_tr, dtype=dtype
    )
    X_va, _ = _build_design_for_columns(
        candidate_groups, abl_result.features, feature_cols_df_va, dtype=dtype
    )
    X_te, _ = _build_design_for_columns(
        candidate_groups, abl_result.features, feature_cols_df_te, dtype=dtype
    )

    preserve_mask = jnp.array([name.endswith("_unit_interval") for name in selected_names], dtype=bool)
    X_tr, X_va, X_te, mu, sd, preserved = standardize_design(
        X_tr,
        X_va,
        X_te
    )
    logger.info(
        "Standardization preserved %d columns and scaled %d columns.",
        int(jnp.sum(preserved)),
        X_tr.shape[1] - int(jnp.sum(preserved)),
    )
    logger.info("Selected feature columns (%d): %s", len(selected_names), ", ".join(selected_names))
    logger.info(
        "Design shapes: train=%s, validation=%s, test=%s",
        X_tr.shape,
        X_va.shape,
        X_te.shape,
    )

    train_log_every = max(1, int(max_iter // 10) if max_iter > 0 else 1)

    model_train_only = SoftmaxRegression(
        n_classes=n_classes,
        reg_strength=float(abl_result.reg_strength),
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        dtype=dtype,
        device=device,
        logger=logger,
        log_every=train_log_every,
        record_history=True,
    )
    fit_train_only = model_train_only.fit(X_tr, y_tr)
    logger.info(
        "Train-only fit finished after %d iterations (grad norm=%.3e)",
        fit_train_only.n_iter,
        fit_train_only.grad_norm,
    )

    proba_tr = model_train_only.predict_proba(X_tr, fit_train_only)
    proba_va = model_train_only.predict_proba(X_va, fit_train_only)
    pred_tr = jnp.argmax(proba_tr, axis=1)
    pred_va = jnp.argmax(proba_va, axis=1)

    metrics_tr = ClassificationMetrics(
        accuracy=accuracy(y_tr, pred_tr),
        misclassification=misclassification_rate(y_tr, pred_tr),
        log_loss=log_loss(y_tr, proba_tr),
        mutual_information=mutual_information(y_tr, proba_tr),
    )

    metrics_va = ClassificationMetrics(
        accuracy=accuracy(y_va, pred_va),
        misclassification=misclassification_rate(y_va, pred_va),
        log_loss=log_loss(y_va, proba_va),
        mutual_information=mutual_information(y_va, proba_va),
    )

    logger.info(
        (
            "Train metrics: acc=%.4f err=%.4f logloss=%.4f mi=%.4f | "
            "Validation acc=%.4f err=%.4f logloss=%.4f mi=%.4f"
        ),
        metrics_tr.accuracy,
        metrics_tr.misclassification,
        metrics_tr.log_loss,
        metrics_tr.mutual_information,
        metrics_va.accuracy,
        metrics_va.misclassification,
        metrics_va.log_loss,
        metrics_va.mutual_information,
    )

    conf_mat_validation = confusion_matrix(y_va, pred_va, n_classes=n_classes)

    history_frames: List[pd.DataFrame] = []
    if fit_train_only.history:
        hist_tr = pd.DataFrame(
            [
                {
                    "iteration": rec.iteration,
                    "loss": rec.loss,
                    "grad_norm": rec.grad_norm,
                    "accuracy": rec.accuracy,
                    "log_loss": rec.log_loss,
                    "phase": "train_only",
                }
                for rec in fit_train_only.history
            ]
        )
        history_frames.append(hist_tr)

    per_hour_frames = [
        _per_hour_accuracy_frame(y_tr, pred_tr, split="train", n_classes=n_classes),
        _per_hour_accuracy_frame(y_va, pred_va, split="validation", n_classes=n_classes),
    ]
    probability_frames = [
        _true_class_probability_frame(y_tr, proba_tr, split="train"),
        _true_class_probability_frame(y_va, proba_va, split="validation"),
    ]

    X_trva = jnp.concatenate([X_tr, X_va], axis=0)
    y_trva = jnp.concatenate([y_tr, y_va], axis=0)

    model_final = SoftmaxRegression(
        n_classes=n_classes,
        reg_strength=float(abl_result.reg_strength),
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        dtype=dtype,
        device=device,
        logger=logger,
        log_every=train_log_every,
        record_history=True,
    )
    fit_final = model_final.fit(X_trva, y_trva)
    logger.info(
        "Final fit (train+validation) completed after %d iterations (grad norm=%.3e)",
        fit_final.n_iter,
        fit_final.grad_norm,
    )

    if fit_final.history:
        hist_final = pd.DataFrame(
            [
                {
                    "iteration": rec.iteration,
                    "loss": rec.loss,
                    "grad_norm": rec.grad_norm,
                    "accuracy": rec.accuracy,
                    "log_loss": rec.log_loss,
                    "phase": "train_validation",
                }
                for rec in fit_final.history
            ]
        )
        history_frames.append(hist_final)

    proba_trva = model_final.predict_proba(X_trva, fit_final)
    proba_te = model_final.predict_proba(X_te, fit_final)
    pred_trva = jnp.argmax(proba_trva, axis=1)
    pred_te = jnp.argmax(proba_te, axis=1)

    metrics_trva = ClassificationMetrics(
        accuracy=accuracy(y_trva, pred_trva),
        misclassification=misclassification_rate(y_trva, pred_trva),
        log_loss=log_loss(y_trva, proba_trva),
        mutual_information=mutual_information(y_trva, proba_trva),
    )

    metrics_te = ClassificationMetrics(
        accuracy=accuracy(y_te, pred_te),
        misclassification=misclassification_rate(y_te, pred_te),
        log_loss=log_loss(y_te, proba_te),
        mutual_information=mutual_information(y_te, proba_te),
    )

    uniform_proba = jnp.full((y_te.shape[0], n_classes), 1.0 / float(n_classes), dtype=dtype)
    baseline_metrics = ClassificationMetrics(
        accuracy=1.0 / float(n_classes),
        misclassification=1.0 - 1.0 / float(n_classes),
        log_loss=log_loss(y_te, uniform_proba),
        mutual_information=mutual_information(y_te, uniform_proba),
    )

    logger.info(
        (
            "Test metrics: acc=%.4f err=%.4f logloss=%.4f mi=%.4f | "
            "Blind acc=%.4f err=%.4f mi=%.4f"
        ),
        metrics_te.accuracy,
        metrics_te.misclassification,
        metrics_te.log_loss,
        metrics_te.mutual_information,
        baseline_metrics.accuracy,
        baseline_metrics.misclassification,
        baseline_metrics.mutual_information,
    )

    conf_mat_test = confusion_matrix(y_te, pred_te, n_classes=n_classes)

    per_hour_frames.append(
        _per_hour_accuracy_frame(y_te, pred_te, split="test", n_classes=n_classes)
    )
    probability_frames.append(
        _true_class_probability_frame(y_te, proba_te, split="test")
    )

    coef_df = pd.DataFrame(
        jnp.asarray(fit_final.weights, dtype=float),
        index=selected_names,
        columns=[f"class_{k}" for k in range(n_classes)],
    )
    bias_df = pd.DataFrame(
        [jnp.asarray(fit_final.bias, dtype=float)],
        index=["bias"],
        columns=[f"class_{k}" for k in range(n_classes)],
    )
    coef_full_df = pd.concat([coef_df, bias_df])

    metrics_table = pd.DataFrame(
        [
            {"split": "train", **metrics_tr.as_dict()},
            {"split": "validation", **metrics_va.as_dict()},
            {"split": "train+validation", **metrics_trva.as_dict()},
            {"split": "test", **metrics_te.as_dict()},
        ]
    )

    baseline_table = pd.DataFrame(
        [{"model": "uniform_blind", **baseline_metrics.as_dict()}]
    )

    mi_df = pd.DataFrame(
        [
            {"split": "train", "mutual_information_nats": float(metrics_tr.mutual_information)},
            {"split": "validation", "mutual_information_nats": float(metrics_va.mutual_information)},
            {"split": "train+validation", "mutual_information_nats": float(metrics_trva.mutual_information)},
            {"split": "test", "mutual_information_nats": float(metrics_te.mutual_information)},
            {"split": "blind_uniform", "mutual_information_nats": float(baseline_metrics.mutual_information)},
        ]
    )

    per_hour_df = pd.concat(per_hour_frames, ignore_index=True)
    probability_df = pd.concat(probability_frames, ignore_index=True)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    selected_features_df = pd.DataFrame({"feature": selected_names})
    scaling_df = pd.DataFrame(
        {
            "feature": selected_names,
            "mean": jnp.asarray(mu, dtype=float).tolist(),
            "std": jnp.asarray(sd, dtype=float).tolist(),
            "preserved": jnp.asarray(preserved, dtype=bool).tolist(),
        }
    )

    summary_path = cfg.paths.classification_tables_dir / "classification_summary.xlsx"
    save_table_xlsx(
        {
            "metrics": metrics_table,
            "baseline": baseline_table,
            "ablation_trace": abl_trace,
            "confusion_validation": pd.DataFrame(
                jnp.asarray(conf_mat_validation, dtype=float),
                index=[f"true_{k}" for k in range(n_classes)],
                columns=[f"pred_{k}" for k in range(n_classes)],
            ),
            "confusion_test": pd.DataFrame(
                jnp.asarray(conf_mat_test, dtype=float),
                index=[f"true_{k}" for k in range(n_classes)],
                columns=[f"pred_{k}" for k in range(n_classes)],
            ),
            "coefficients": coef_full_df,
            "per_hour_accuracy": per_hour_df,
            "true_class_probability": probability_df,
            "training_history": history_df,
            "selected_features": selected_features_df,
            "standardization": scaling_df,
            "mutual_information": mi_df,
        },
        summary_path,
    )
    logger.info("Saved classification tables to %s", summary_path)

    fig_conf_validation = cfg.paths.classification_figures_dir / "confusion_matrix_validation.png"
    viz_classif.plot_confusion_matrix(
        conf_mat_validation, fig_conf_validation, title="Confusion matrix (validation)"
    )

    fig_conf_test = cfg.paths.classification_figures_dir / "confusion_matrix_test.png"
    viz_classif.plot_confusion_matrix(
        conf_mat_test, fig_conf_test, title="Confusion matrix (test)"
    )

    fig_trace = cfg.paths.classification_figures_dir / "classification_ablation.png"
    viz_classif.plot_ablation_trace(abl_trace, fig_trace)

    if not history_df.empty:
        fig_history = cfg.paths.classification_figures_dir / "classification_training_history.png"
        viz_classif.plot_training_history(history_df, fig_history)
    else:
        fig_history = None

    fig_per_hour = cfg.paths.classification_figures_dir / "classification_per_hour_accuracy.png"
    viz_classif.plot_hour_accuracy(per_hour_df, fig_per_hour)

    fig_probability = cfg.paths.classification_figures_dir / "classification_true_class_probability.png"
    viz_classif.plot_true_class_probability(probability_df, fig_probability)

    fig_mi = cfg.paths.classification_figures_dir / "classification_mutual_information.png"
    viz_classif.plot_mutual_information(mi_df, fig_mi)

    logger.info(
        "Classification figures saved: %s, %s, %s, %s, %s%s",
        fig_conf_validation,
        fig_conf_test,
        fig_trace,
        fig_per_hour,
        fig_probability,
        f", {fig_history}" if fig_history is not None else "",
    )
