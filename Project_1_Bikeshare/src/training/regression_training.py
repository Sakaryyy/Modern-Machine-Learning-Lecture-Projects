from __future__ import annotations

import logging
from typing import List, Sequence, Tuple, Literal

import jax
import jax.numpy as jnp
import pandas as pd

from src.config.experiment_config import ExperimentConfig, FeatureConfig
from src.data.clean import clean_bike_df, save_processed
from src.data.fetch import fetch_uci_bike
from src.data.split import DataSplits, data_split
from src.evaluation.ablation_regression import forward_ablation
from src.features.pipeline import (
    FeaturePipeline,
    interaction_onehot_numeric_step,
    polynomial_step,
    numeric_step,
    hour_onehot_step,
    hour_fourier_step,
    hour_cyclical_step,
    weekday_onehot_step,
    month_onehot_step,
    season_onehot_step,
    weathersit_onehot_step,
)
from src.metrics.regression import mae, r2, rmse
from src.models.baselines import HourOfDayBaseline, MeanBaseline
from src.models.linear_ridge_jax import RidgeClosedForm
from src.utils.helpers import log_jax_runtime_info
from src.utils.io import save_table_xlsx
from src.utils.preprocess import standardize_design, can_select_group
from src.utils.preprocess import to_device_array
from src.utils.targets import forward_transform, inverse_transform, smearing_factor
from src.visualization import regression_plotting as viz_train

logger = logging.getLogger(__name__)
YMode = Literal["none", "log1p", "sqrt"]


def build_candidate_groups(
        cfg_feats: FeatureConfig,
) -> List[Tuple[List[str], FeaturePipeline]]:
    """
    Create interpretable feature groups for forward ablation.

    Parameters
    ----------
    cfg_feats : FeatureConfig
        Feature-related configuration (whether to include workingday, humidity,
        windspeed, and the polynomial degree for atemp).

    Returns
    -------
    list of (feature_names, FeaturePipeline)
        Each group is a small pipeline of stateless transforms that produces a matrix
        and a list of column names. Groups are:
        - Hour of day cyclical encoding: ["sin_hour", "cos_hour"]
        - Apparent temperature and optional quadratic: ["atemp"] or ["atemp", "atemp^2"]
        - Optional singletons: ["workingday"], ["hum"], ["windspeed"]
    """
    groups: List[Tuple[List[str], FeaturePipeline]] = []

    if cfg_feats.add_hour_cyclical:
        if int(cfg_feats.hour_fourier_harmonics) > 1:
            hour_pipe = FeaturePipeline(
                steps=[hour_fourier_step(n_harmonics=int(cfg_feats.hour_fourier_harmonics))]
            )
            names = []
            for k in range(1, int(cfg_feats.hour_fourier_harmonics) + 1):
                names.extend([f"sin_hour_{k}", f"cos_hour_{k}"])
            groups.append((names, hour_pipe))
        else:
            hour_pipe = FeaturePipeline(
                steps=[hour_cyclical_step()]
            )
            groups.append((["sin_hour", "cos_hour"], hour_pipe))

    if cfg_feats.add_hour_onehot:
        onehot = hour_onehot_step(
            drop_first=bool(cfg_feats.onehot_drop_first)
        )
        names = [f"hr_{k}" for k in (range(1, 24) if cfg_feats.onehot_drop_first else range(24))]
        groups.append((names, FeaturePipeline(steps=[onehot])))

    # Base atemp and optional polynomial
    if cfg_feats.use_atemp:
        atemp_steps = [numeric_step("atemp")]
        names_atemp: List[str] = ["atemp"]
        if int(cfg_feats.poly_temp_degree) >= 2:
            atemp_steps.append(
                polynomial_step(
                    "atemp",
                    degree=int(cfg_feats.poly_temp_degree),
                    include_linear=False,
                )
            )
            names_atemp = ["atemp"] + [f"atemp^{p}" for p in range(2, int(cfg_feats.poly_temp_degree) + 1)]
        groups.append(
            (names_atemp, FeaturePipeline(steps=atemp_steps)))

    # Optional interactions: hour one-hot x atemp
    if cfg_feats.add_hour_interactions_with_atemp:
        if not cfg_feats.add_hour_onehot:
            logger.warning(
                "add_hour_interactions_with_atemp=True but add_hour_onehot=False; enabling one-hot implicitly for interactions.")
        inter_step = interaction_onehot_numeric_step(
            "atemp",
            drop_first=bool(cfg_feats.onehot_drop_first)
        )
        # Names follow the step definition: atemp:hr_i
        names = [f"atemp:hr_{k}" for k in (range(1, 24) if cfg_feats.onehot_drop_first else range(24))]
        groups.append(
            (names, FeaturePipeline(steps=[inter_step])))

    if cfg_feats.use_weekday_onehot:
        wk = weekday_onehot_step(drop_first=True)
        wk_names = [f"wd_{k}" for k in range(1, 7)]
        groups.append((wk_names, FeaturePipeline(steps=[wk])))

    if cfg_feats.use_month_onehot:
        mh = month_onehot_step(drop_first=True)
        mon_names = [f"mon_{k}" for k in range(1, 12)]
        groups.append((mon_names, FeaturePipeline(steps=[mh])))

    if cfg_feats.use_season_onehot:
        season_step = season_onehot_step(
            drop_first=bool(cfg_feats.season_drop_first)
        )
        season_names = (
            ["season_spring", "season_summer", "season_fall"]
            if cfg_feats.season_drop_first else
            ["season_winter", "season_spring", "season_summer", "season_fall"]
        )
        groups.append(
            (season_names, FeaturePipeline(steps=[season_step]))
        )

    if cfg_feats.use_weathersit_onehot:
        ws_step = weathersit_onehot_step(
            drop_first=bool(cfg_feats.weathersit_drop_first)
        )
        ws_names = (
            ["ws_mist", "ws_light", "ws_heavy"]
            if cfg_feats.weathersit_drop_first else
            ["ws_clear", "ws_mist", "ws_light", "ws_heavy"]
        )
        groups.append(
            (ws_names, FeaturePipeline(steps=[ws_step]))
        )

    # Optional singletons
    if cfg_feats.use_workingday:
        groups.append(
            (
                ["workingday"],
                FeaturePipeline(
                    steps=[numeric_step("workingday")],
                ),
            )
        )
    if cfg_feats.use_humidity:
        groups.append(
            (
                ["hum"],
                FeaturePipeline(
                    steps=[numeric_step("hum")],
                ),
            )
        )
    if cfg_feats.use_windspeed:
        groups.append(
            (
                ["windspeed"],
                FeaturePipeline(
                    steps=[numeric_step("windspeed")],
                ),
            )
        )
    # Log group composition
    logger.info("Candidate feature groups:")
    for names, _ in groups:
        logger.info("  - %s (ncols=%d)", ", ".join(names), len(names))
    return groups


def _approx_condition_number(X: jax.Array) -> float:
    """
    Approximate 2-norm condition number of the centered design matrix X
    using singular values. If X has zero columns, returns 0.0.
    """
    if X.shape[1] == 0:
        return 0.0
    Xc = X - jnp.mean(X, axis=0)
    s = jnp.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    smax = jnp.max(s)
    smin = jnp.min(s)
    cond = jnp.where(smin > 0, smax / smin, jnp.inf)
    return float(cond)


def run_regression(
        cfg: ExperimentConfig,
        lam_grid: Sequence[float],
        epsilon: float,
        y_transform: YMode
) -> None:
    """
    Train a ridge regression with forward ablation and evaluate against baselines.

    Parameters
    ----------
    cfg : ExperimentConfig
        Top-level configuration (paths, splits, features).
    lam_grid : sequence of float
        The grid of ridge regularization strengths to search on the validation set.
        The objective minimized is mean squared error with L2 penalty. The final
        metric reported is RMSE on the test set.
    epsilon : float
        Relative tolerance used to choose the minimal feature subset whose
        validation RMSE is within (1 + epsilon) of the best observed across
        all subsets and lambda values.

    Notes
    -----
    1) Load processed data (or fetch+clean if missing).
    2) Chronological split into train, validation, test.
    3) Build interpretable candidate feature groups (hour cyclical, atemp±poly, etc.).
    4) Forward ablation to find a minimal feature subset and best lambda.
    5) Fit ridge on train+validation with the selected lambda.
    6) Evaluate on test, compare to strict blind and semi-blind baselines.
    7) Save a metrics workbook.
    """
    # Device info and selection (log, place arrays on this device for training).
    cfg.paths.ensure_exists()
    device = log_jax_runtime_info()
    dtype = jnp.float32

    # Load processed data or fetch+clean if missing.
    processed_csv = cfg.paths.processed_dir / "bike_clean.csv"
    if processed_csv.exists():
        df = pd.read_csv(processed_csv, parse_dates=["timestamp"], index_col="timestamp")
        logger.info(f"Loaded processed dataset: {processed_csv}")
    else:
        raw_df = fetch_uci_bike(cache_dir=cfg.paths.raw_dir, force=False)
        df = clean_bike_df(raw_df)
        save_processed(df, cfg.paths.processed_dir)
        logger.info("Fetched, cleaned, and saved processed dataset.")

    # Datasplitting via Timeline.
    splits = DataSplits(train_end=cfg.splits.train_end, validation_end=cfg.splits.validation_end)
    train_df, validation_df, test_df = data_split(df, splits)
    logger.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    # Targets to device as JAX vectors.
    y_tr = to_device_array(
        train_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True
    )
    y_va = to_device_array(
        validation_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True
    )
    y_te = to_device_array(
        test_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True
    )

    # Baselines.
    mean_bl = MeanBaseline.from_train(y_tr, dtype=dtype, device=device)
    hod_bl = HourOfDayBaseline.from_train(train_df, dtype=dtype, device=device)

    # Candidate groups and ablation.
    logger.info("Lambda grid: %s", ", ".join(f"{lam:.4g}" for lam in lam_grid))
    candidate_groups = build_candidate_groups(cfg.features)
    abl_result, abl_trace = forward_ablation(
        df_tr=train_df,
        y_tr=y_tr,
        df_val=validation_df,
        y_val=y_va,
        candidate_groups=candidate_groups,
        lam_grid=lam_grid,
        epsilon=epsilon,
        y_transform=y_transform,
        lam_floor=1e-8,
        record_trace=True,
        logger=logger,
        can_select=can_select_group
    )
    steps_completed = (
        int(abl_trace["step"].max()) if hasattr(abl_trace, "empty") and not abl_trace.empty else len(
            abl_result.features)
    )
    logger.info(
    "Ablation selected %d features over %d greedy steps with lambda=%.4g (val rmse=%.3f)",
    len(abl_result.features),
        steps_completed,
        abl_result.lam,
        abl_result.rmse_val,
    )

    # Verify minimality contract
    best_overall = float(abl_trace["best_overall_rmse_val_so_far"].min())
    threshold = (1.0 + float(epsilon)) * best_overall
    logger.info(
        "Minimality threshold: %.6f (epsilon=%.4f, best validation RMSE=%.6f)",
        threshold,
        epsilon,
        best_overall,
    )

    # Helper to compose the JAX design for an arbitrary DataFrame and a chosen column list.
    def build_design(chosen_feats: List[str], df_part: pd.DataFrame) -> jax.Array:
        """Assemble the design matrix restricted to ``chosen_feats`` for ``df_part``."""

        matrices: List[jax.Array] = []
        for group_names, pipe in candidate_groups:
            X, cols = pipe.transform(df_part)
            # Keep only columns present in chosen_feats
            idx = [i for i, col in enumerate(cols) if col in chosen_feats]
            if idx:
                matrices.append(X[:, jnp.array(idx, dtype=jnp.int32)])
        if matrices:
            return jnp.concatenate(matrices, axis=1)
        return jnp.empty((len(df_part), 0), dtype=dtype)

    # Fit ridge on train+validation using selected features and selected lambda.
    X_tr = build_design(abl_result.features, train_df)
    X_va = build_design(abl_result.features, validation_df)
    X_te = build_design(abl_result.features, test_df)

    X_tr, X_va, X_te, mu, sd, is_binary = standardize_design(X_tr, X_va, X_te)
    logger.info(
        "Standardization preserved %d binary columns and scaled %d numeric columns.",
        int(jnp.sum(is_binary)),
        X_tr.shape[1] - int(jnp.sum(is_binary)),
    )

    z_tr = forward_transform(y_tr, y_transform)
    z_va = forward_transform(y_va, y_transform)

    logger.info(
        "Design shapes: train=%s, validation=%s, test=%s",
        X_tr.shape,
        X_va.shape,
        X_te.shape,
    )
    logger.info(
        "Condition numbers: train=%.3e, validation=%.3e",
        _approx_condition_number(X_tr),
        _approx_condition_number(X_va),
    )

    model_train_only = RidgeClosedForm(
        lam=float(abl_result.lam), dtype=dtype, device=device, fit_intercept=True
    )
    fit_train_only = model_train_only.fit(X_tr, z_tr)
    logger.info(
        "Train-only ridge fit: intercept b=%.6f, training sigma²=%.6f",
        fit_train_only.b,
        fit_train_only.sigma2,
    )

    zhat_tr_train = model_train_only.predict(X_tr, fit_train_only)
    zhat_va_train = model_train_only.predict(X_va, fit_train_only)

    smear_train = None
    if y_transform == "log1p":
        resid_train = z_tr - zhat_tr_train
        smear_train = smearing_factor(resid_train)
        logger.info("Smearing factor (train split, log1p): %.6f", smear_train)

    yhat_tr_train = inverse_transform(zhat_tr_train, y_transform, smear=smear_train)
    yhat_va_train = inverse_transform(zhat_va_train, y_transform, smear=smear_train)
    yhat_tr_train = jnp.maximum(yhat_tr_train, 0.0)
    yhat_va_train = jnp.maximum(yhat_va_train, 0.0)

    metrics_train = {
        "rmse": rmse(y_tr, yhat_tr_train),
        "mae": mae(y_tr, yhat_tr_train),
        "r2": r2(y_tr, yhat_tr_train),
    }
    metrics_validation = {
        "rmse": rmse(y_va, yhat_va_train),
        "mae": mae(y_va, yhat_va_train),
        "r2": r2(y_va, yhat_va_train),
    }
    logger.info(
        "Train-only metrics: train RMSE=%.4f MAE=%.4f R2=%.4f | validation RMSE=%.4f MAE=%.4f R2=%.4f",
        metrics_train["rmse"],
        metrics_train["mae"],
        metrics_train["r2"],
        metrics_validation["rmse"],
        metrics_validation["mae"],
        metrics_validation["r2"],
    )

    yhat_mean_tr = jnp.maximum(mean_bl.predict(len(train_df)), 0.0)
    yhat_mean_va = jnp.maximum(mean_bl.predict(len(validation_df)), 0.0)
    yhat_mean_te = jnp.maximum(mean_bl.predict(len(test_df)), 0.0)

    yhat_hod_tr = jnp.maximum(hod_bl.predict(train_df), 0.0)
    yhat_hod_va = jnp.maximum(hod_bl.predict(validation_df), 0.0)
    yhat_hod_te = jnp.maximum(hod_bl.predict(test_df), 0.0)

    X_trva = jnp.concatenate([X_tr, X_va], axis=0)
    z_trh = jnp.concatenate([z_tr, z_va], axis=0)
    y_trh_orig = jnp.concatenate([y_tr, y_va], axis=0)

    logger.info(
        "Combined design shape train+validation=%s with condition≈%.3e",
        X_trva.shape,
        _approx_condition_number(X_trva),
    )

    model_final = RidgeClosedForm(
        lam=float(abl_result.lam), dtype=dtype, device=device, fit_intercept=True
    )
    fit_final = model_final.fit(X_trva, z_trh)
    logger.info(
        "Final ridge fit (train+validation): intercept b=%.6f, sigma²=%.6f",
        fit_final.b,
        fit_final.sigma2,
    )

    # Test design and predictions.
    zhat_trva = model_final.predict(X_trva, fit_final)
    zhat_te = model_final.predict(X_te, fit_final)

    smear_final = None
    if y_transform == "log1p":
        resid_z = z_trh - zhat_trva
        smear_final = smearing_factor(resid_z)
        logger.info("Smearing factor (train+validation, log1p): %.6f", smear_final)

    yhat_trva = inverse_transform(zhat_trva, y_transform, smear=smear_final)
    yhat_te = inverse_transform(zhat_te, y_transform, smear=smear_final)

    yhat_trva = jnp.maximum(yhat_trva, 0.0)
    yhat_te = jnp.maximum(yhat_te, 0.0)

    metrics_trva = {
        "rmse": rmse(y_trh_orig, yhat_trva),
        "mae": mae(y_trh_orig, yhat_trva),
        "r2": r2(y_trh_orig, yhat_trva),
    }
    metrics_test = {
        "rmse": rmse(y_te, yhat_te),
        "mae": mae(y_te, yhat_te),
        "r2": r2(y_te, yhat_te),
    }
    logger.info(
        "Final metrics (train+validation -> test): train+validation RMSE=%.4f MAE=%.4f R2=%.4f | test RMSE=%.4f MAE=%.4f R2=%.4f",
        metrics_trva["rmse"],
        metrics_trva["mae"],
        metrics_trva["r2"],
        metrics_test["rmse"],
        metrics_test["mae"],
        metrics_test["r2"],
    )

    baseline_validation = {
        "mean": {
            "rmse": rmse(y_va, yhat_mean_va),
            "mae": mae(y_va, yhat_mean_va),
            "r2": r2(y_va, yhat_mean_va),
        },
        "hour_of_day": {
            "rmse": rmse(y_va, yhat_hod_va),
            "mae": mae(y_va, yhat_hod_va),
            "r2": r2(y_va, yhat_hod_va),
        },
    }
    baseline_test = {
        "mean": {
            "rmse": rmse(y_te, yhat_mean_te),
            "mae": mae(y_te, yhat_mean_te),
            "r2": r2(y_te, yhat_mean_te),
        },
        "hour_of_day": {
            "rmse": rmse(y_te, yhat_hod_te),
            "mae": mae(y_te, yhat_hod_te),
            "r2": r2(y_te, yhat_hod_te),
        },
    }
    logger.info(
        "Baseline RMSE (validation): mean=%.4f hour-of-day=%.4f | (test): mean=%.4f hour-of-day=%.4f",
        baseline_validation["mean"]["rmse"],
        baseline_validation["hour_of_day"]["rmse"],
        baseline_test["mean"]["rmse"],
        baseline_test["hour_of_day"]["rmse"],
    )

    # Save metrics workbook.
    metrics_table = pd.DataFrame(
        [
            {
                "split": "train",
                "fit": "train_only",
                "y_transform": y_transform,
                "smearing": float(smear_train) if smear_train is not None else float("nan"),
                "rmse": float(metrics_train["rmse"]),
                "mae": float(metrics_train["mae"]),
                "r2": float(metrics_train["r2"]),
            },
            {
                "split": "validation",
                "fit": "train_only",
                "y_transform": y_transform,
                "smearing": float(smear_train) if smear_train is not None else float("nan"),
                "rmse": float(metrics_validation["rmse"]),
                "mae": float(metrics_validation["mae"]),
                "r2": float(metrics_validation["r2"]),
            },
            {
                "split": "train+validation",
                "fit": "final",
                "y_transform": y_transform,
                "smearing": float(smear_final) if smear_final is not None else float("nan"),
                "rmse": float(metrics_trva["rmse"]),
                "mae": float(metrics_trva["mae"]),
                "r2": float(metrics_trva["r2"]),
            },
            {
                "split": "test",
                "fit": "final",
                "y_transform": y_transform,
                "smearing": float(smear_final) if smear_final is not None else float("nan"),
                "rmse": float(metrics_test["rmse"]),
                "mae": float(metrics_test["mae"]),
                "r2": float(metrics_test["r2"]),
            },
        ]
    )
    baseline_rows = []
    for split, metrics_map in ("validation", baseline_validation), ("test", baseline_test):
        for model_name, vals in metrics_map.items():
            baseline_rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "rmse": float(vals["rmse"]),
                    "mae": float(vals["mae"]),
                    "r2": float(vals["r2"]),
                }
            )
    baseline_table = pd.DataFrame(baseline_rows)

    # Training visualizations and tabular diagnostics

    lam_curve_df = viz_train.compute_lambda_curve(
        train_df, validation_df, y_tr, y_va, candidate_groups, abl_result.features, lam_grid, dtype, device
    )

    ablation_path_df = viz_train.compute_ablation_path(
        train_df, validation_df, y_tr, y_va, candidate_groups, lam_grid, dtype, device
    )
    coef_df = viz_train.coefficient_table(fit_final, abl_result.features)
    selected_features_df = pd.DataFrame({"feature": abl_result.features})
    scaling_df = pd.DataFrame(
        {
            "feature": abl_result.features,
            "mean": jnp.asarray(mu, dtype=float).tolist(),
            "std": jnp.asarray(sd, dtype=float).tolist(),
            "is_binary": jnp.asarray(is_binary, dtype=bool).tolist(),
        }
    )

    design_stats_df = pd.DataFrame(
        [
            {"split": "train", "n_rows": X_tr.shape[0], "n_cols": X_tr.shape[1],
             "condition": _approx_condition_number(X_tr)},
            {"split": "validation", "n_rows": X_va.shape[0], "n_cols": X_va.shape[1],
             "condition": _approx_condition_number(X_va)},
            {"split": "train+validation", "n_rows": X_trva.shape[0], "n_cols": X_trva.shape[1],
             "condition": _approx_condition_number(X_trva)},
            {"split": "test", "n_rows": X_te.shape[0], "n_cols": X_te.shape[1],
             "condition": _approx_condition_number(X_te)},
        ]
    )

    preds_train_df = viz_train.predictions_table(
        train_df.index, y_tr, yhat_tr_train, yhat_mean_tr, yhat_hod_tr, split="train"
    )
    preds_validation_df = viz_train.predictions_table(
        validation_df.index, y_va, yhat_va_train, yhat_mean_va, yhat_hod_va, split="validation"
    )
    preds_test_df = viz_train.predictions_table(
        test_df.index, y_te, yhat_te, yhat_mean_te, yhat_hod_te, split="test"
    )

    summary_path = cfg.paths.regression_tables_dir / "regression_summary.xlsx"
    save_table_xlsx(
        {
            "metrics": metrics_table,
            "baselines": baseline_table,
            "ablation_trace": abl_trace,
            "lambda_curve": lam_curve_df,
            "ablation_path": ablation_path_df,
            "coefficients": coef_df,
            "selected_features": selected_features_df,
            "standardization": scaling_df,
            "design_statistics": design_stats_df,
            "predictions_train": preds_train_df,
            "predictions_validation": preds_validation_df,
            "predictions_test": preds_test_df,
        },
        summary_path,
    )
    logger.info("Saved regression tables to %s", summary_path)

    metrics_path = cfg.paths.regression_tables_dir / "metrics.xlsx"
    save_table_xlsx({"metrics": metrics_table}, metrics_path)
    logger.info("Saved compatibility metrics workbook to %s", metrics_path)

    fig_lambda = viz_train.plot_lambda_curve(
        lam_curve_df, cfg.paths.regression_figures_dir / "lambda_curve_selected.png"
    )
    fig_ablation_path = viz_train.plot_ablation_path(
        ablation_path_df, cfg.paths.regression_figures_dir / "ablation_path.png"
    )
    fig_coeff = viz_train.plot_coefficients(
        coef_df, cfg.paths.regression_figures_dir / "coefficients_bar.png"
    )

    fig_parity_validation = viz_train.plot_parity(
        preds_validation_df, cfg.paths.regression_figures_dir / "parity_validation.png", split="validation"
    )
    fig_parity_test = viz_train.plot_parity(
        preds_test_df, cfg.paths.regression_figures_dir / "parity_test.png", split="test"
    )

    fig_resid_hist_validation = viz_train.plot_residual_hist(
        preds_validation_df, cfg.paths.regression_figures_dir / "residual_hist_validation.png", split="validation"
    )
    fig_resid_hist_test = viz_train.plot_residual_hist(
        preds_test_df, cfg.paths.regression_figures_dir / "residual_hist_test.png", split="test"
    )

    fig_resid_vs_pred_validation = viz_train.plot_residual_vs_pred(
        preds_validation_df, cfg.paths.regression_figures_dir / "residual_vs_pred_validation.png", split="validation"
    )
    fig_resid_vs_pred_test = viz_train.plot_residual_vs_pred(
        preds_test_df, cfg.paths.regression_figures_dir / "residual_vs_pred_test.png", split="test"
    )

    fig_timeseries_validation = viz_train.plot_timeseries_overlay(
        preds_validation_df, cfg.paths.regression_figures_dir / "timeseries_overlay_validation.png", split="validation"
    )
    fig_timeseries_test = viz_train.plot_timeseries_overlay(
        preds_test_df, cfg.paths.regression_figures_dir / "timeseries_overlay_test.png", split="test"
    )

    fig_residuals_hour_validation = viz_train.plot_residuals_by_hour(
        preds_validation_df, cfg.paths.regression_figures_dir / "residuals_by_hour_validation.png", split="validation"
    )
    fig_residuals_hour_test = viz_train.plot_residuals_by_hour(
        preds_test_df, cfg.paths.regression_figures_dir / "residuals_by_hour_test.png", split="test"
    )

    test_comparison_df = pd.DataFrame(
        [
            {"model": "ridge", "rmse_test": float(metrics_test["rmse"])},
            {"model": "baseline_mean", "rmse_test": float(baseline_test["mean"]["rmse"])},
            {"model": "baseline_hour_of_day", "rmse_test": float(baseline_test["hour_of_day"]["rmse"])},
        ]
    )
    fig_baseline = viz_train.plot_baseline_comparison(
        test_comparison_df, cfg.paths.regression_figures_dir / "baseline_comparison.png"
    )
    fig_ablation_trace = viz_train.plot_ablation_trace_table(
        abl_trace, cfg.paths.regression_figures_dir / "ablation_trace_table.png"
    )

    logger.info(
        "Saved regression figures: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
        fig_lambda,
        fig_ablation_path,
        fig_coeff,
        fig_parity_validation,
        fig_parity_test,
        fig_resid_hist_validation,
        fig_resid_hist_test,
        fig_resid_vs_pred_validation,
        fig_resid_vs_pred_test,
        fig_timeseries_validation,
        fig_timeseries_test,
        fig_residuals_hour_validation,
        fig_residuals_hour_test,
    )
    logger.info("Saved baseline comparison figure -> %s", fig_baseline)
    logger.info("Saved ablation trace figure -> %s", fig_ablation_trace)

    viz_train.save_checkpoint(
        path=cfg.paths.regression_tables_dir.parent / "regression" / "models" / "ridge_checkpoint.npz",
        fit=fit_final,
        selected_features=abl_result.features,
        lam=abl_result.lam,
    )
    logger.info("Saved ridge model checkpoint.")
