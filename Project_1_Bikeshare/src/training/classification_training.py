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

    if "yr" in available_columns:
        groups.append(
            (
                ["yr"],
                FeaturePipeline(steps=[numeric_step("yr", dtype=dtype)], dtype=dtype, device=device),
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


def run_classification(
    cfg: ExperimentConfig,
    *,
    reg_grid: Sequence[float],
    epsilon: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> None:
    """Execute the classification experiment using logistic regression."""

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
    train_df, holdout_df, test_df = data_split(df, splits)
    logger.info("Split sizes: train=%d, holdout=%d, test=%d", len(train_df), len(holdout_df), len(test_df))

    feature_cols_df_tr = train_df.drop(columns=["hr"])
    feature_cols_df_ho = holdout_df.drop(columns=["hr"])
    feature_cols_df_te = test_df.drop(columns=["hr"])

    y_tr = to_device_array(
        train_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )
    y_ho = to_device_array(
        holdout_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )
    y_te = to_device_array(
        test_df["hr"].astype(int).to_numpy(), dtype=jnp.int32, device=device, check_finite=True
    )

    n_classes = int(jnp.max(jnp.concatenate([y_tr, y_ho, y_te])) + 1)

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
        df_val=feature_cols_df_ho,
        y_val=y_ho,
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
    X_ho, _ = _build_design_for_columns(
        candidate_groups, abl_result.features, feature_cols_df_ho, dtype=dtype
    )
    X_te, _ = _build_design_for_columns(
        candidate_groups, abl_result.features, feature_cols_df_te, dtype=dtype
    )

    preserve_mask = jnp.array([name.endswith("_unit_interval") for name in selected_names], dtype=bool)
    X_tr, X_ho, X_te, _mu, _sd, preserved = standardize_design(
        X_tr,
        X_ho,
        X_te,
        preserve_mask=preserve_mask,
    )
    logger.info(
        "Standardization preserved %d columns and scaled %d columns.",
        int(jnp.sum(preserved)),
        X_tr.shape[1] - int(jnp.sum(preserved)),
    )

    X_trh = jnp.concatenate([X_tr, X_ho], axis=0)
    y_trh = jnp.concatenate([y_tr, y_ho], axis=0)

    model = SoftmaxRegression(
        n_classes=n_classes,
        reg_strength=float(abl_result.reg_strength),
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        dtype=dtype,
        device=device,
    )
    fit = model.fit(X_trh, y_trh)
    logger.info("Fitted softmax regression in %d iterations (final grad norm=%.3e)", fit.n_iter, fit.grad_norm)

    proba_trh = model.predict_proba(X_trh, fit)
    proba_te = model.predict_proba(X_te, fit)
    pred_trh = jnp.argmax(proba_trh, axis=1)
    pred_te = jnp.argmax(proba_te, axis=1)

    metrics_trh = ClassificationMetrics(
        accuracy=accuracy(y_trh, pred_trh),
        misclassification=misclassification_rate(y_trh, pred_trh),
        log_loss=log_loss(y_trh, proba_trh),
        mutual_information=mutual_information(y_trh, proba_trh),
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
        "Test accuracy=%.4f (error=%.4f); blind accuracy=%.4f (error=%.4f)",
        metrics_te.accuracy,
        metrics_te.misclassification,
        baseline_metrics.accuracy,
        baseline_metrics.misclassification,
    )

    conf_mat = confusion_matrix(y_te, pred_te, n_classes=n_classes)
    conf_df = pd.DataFrame(
        jnp.asarray(conf_mat, dtype=float),
        index=[f"true_{k}" for k in range(n_classes)],
        columns=[f"pred_{k}" for k in range(n_classes)],
    )

    coef_df = pd.DataFrame(
        jnp.asarray(fit.weights, dtype=float),
        index=selected_names,
        columns=[f"class_{k}" for k in range(n_classes)],
    )
    bias_df = pd.DataFrame(
        [jnp.asarray(fit.bias, dtype=float)],
        index=["bias"],
        columns=[f"class_{k}" for k in range(n_classes)],
    )
    coef_full_df = pd.concat([coef_df, bias_df])

    metrics_table = pd.DataFrame(
        [
            {"split": "train+holdout", **metrics_trh.as_dict()},
            {"split": "test", **metrics_te.as_dict()},
        ]
    )

    baseline_table = pd.DataFrame(
        [{"model": "uniform_blind", **baseline_metrics.as_dict()}]
    )

    summary_path = cfg.paths.model_tables_dir / "classification_summary.xlsx"
    save_table_xlsx(
        {
            "metrics": metrics_table,
            "baseline": baseline_table,
            "ablation_trace": abl_trace,
            "confusion_matrix": conf_df,
            "coefficients": coef_full_df,
        },
        summary_path,
    )
    logger.info("Saved classification tables to %s", summary_path)

    fig_conf = cfg.paths.model_figures_dir / "confusion_matrix.png"
    viz_classif.plot_confusion_matrix(conf_mat, fig_conf)

    fig_trace = cfg.paths.model_figures_dir / "classification_ablation.png"
    viz_classif.plot_ablation_trace(abl_trace, fig_trace)

    logger.info("Classification figures saved to %s and %s", fig_conf, fig_trace)
