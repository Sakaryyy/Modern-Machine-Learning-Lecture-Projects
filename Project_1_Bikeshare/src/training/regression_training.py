from __future__ import annotations

import logging
from typing import List, Sequence, Tuple, Literal

import pandas as pd

import jax
import jax.numpy as jnp

from src.config.experiment_config import ExperimentConfig
from src.utils.io import save_table_xlsx
from utils.preprocess import to_device_array
from src.utils.targets import forward_transform, inverse_transform, smearing_factor
from src.utils.preprocess import standardize_design, can_select_group
from src.utils.helpers import log_jax_runtime_info
from src.data.fetch import fetch_uci_bike
from src.data.clean import clean_bike_df, save_processed
from src.data.split import DataSplits, data_split
from src.visualization import train as viz_train
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
from src.models.linear_ridge_jax import RidgeClosedForm
from src.models.baselines import HourOfDayBaseline, MeanBaseline
from src.metrics.regression import mae, r2, rmse
from src.evaluation.ablation import forward_ablation

logger = logging.getLogger(__name__)
YMode = Literal["none", "log1p", "sqrt"]


def build_candidate_groups(
        cfg_feats
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


def run_train(cfg: ExperimentConfig, lam_grid: Sequence[float], epsilon: float, y_transform: YMode) -> None:
    """
    Train a ridge regression with forward ablation and evaluate against baselines.

    Parameters
    ----------
    cfg : ExperimentConfig
        Top-level configuration (paths, splits, features).
    lam_grid : sequence of float
        The grid of ridge regularization strengths to search on the holdout set.
        The objective minimized is mean squared error with L2 penalty. The final
        metric reported is RMSE on the test set.
    epsilon : float
        Relative tolerance used to choose the minimal feature subset whose
        validation RMSE is within (1 + epsilon) of the best observed across
        all subsets and lambda values.

    Notes
    -----
    1) Load processed data (or fetch+clean if missing).
    2) Chronological split into train, holdout, test.
    3) Build interpretable candidate feature groups (hour cyclical, atemp±poly, etc.).
    4) Forward ablation to find a minimal feature subset and best lambda.
    5) Fit ridge on train+holdout with the selected lambda.
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
    logger.info(f"Split sizes: train={len(train_df)}, holdout={len(validation_df)}, test={len(test_df)}")

    # Targets to device as JAX vectors.
    y_tr = to_device_array(train_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True)
    y_va = to_device_array(validation_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True)
    y_te = to_device_array(test_df["cnt"].to_numpy(dtype=float), dtype=dtype, device=device, check_finite=True)

    # Baselines.
    mean_bl = MeanBaseline.from_train(y_tr, dtype=dtype, device=device)
    hod_bl = HourOfDayBaseline.from_train(train_df, dtype=dtype, device=device)

    # Candidate groups and ablation.
    logger.info("Lambda grid: %s", ", ".join(f"{l:.4g}" for l in lam_grid))
    candidate_groups = build_candidate_groups(cfg.features)
    abl, abl_trace = forward_ablation(
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
    logger.info(
        f"Ablation choice: features={abl.features}, lambda={abl.lam:.5g}, "
        f"rmse_tr={abl.rmse_tr:.3f}, rmse_val={abl.rmse_val:.3f}",
    )

    # Verify minimality contract
    best_overall = float(abl_trace["best_overall_rmse_val_so_far"].min())
    threshold = (1.0 + float(epsilon)) * best_overall
    logger.info("Minimality threshold: (1+epsilon)*best = %.6f with epsilon=%.4f", threshold, epsilon)
    logger.info("Final chosen validation RMSE: %.6f", abl.rmse_val)

    # Helper to compose the JAX design for an arbitrary DataFrame and a chosen column list.
    def build_design(chosen_feats: List[str], df_part: pd.DataFrame) -> jax.Array:
        matrices: List[jax.Array] = []
        for group_names, pipe in candidate_groups:
            X, cols = pipe.transform(df_part)
            # Keep only columns present in chosen_feats
            idx = [i for i, c in enumerate(cols) if c in chosen_feats]
            if idx:
                X_sel = X[:, jnp.array(idx, dtype=jnp.int32)]
                matrices.append(X_sel)
        if matrices:
            return jnp.concatenate(matrices, axis=1)
        return jnp.empty((len(df_part), 0), dtype=dtype)

    # Fit ridge on train+holdout using selected features and selected lambda.
    X_tr = build_design(abl.features, train_df)
    X_va = build_design(abl.features, validation_df)
    X_te = build_design(abl.features, test_df)

    X_tr, X_va, X_te, mu, sd, is_bin = standardize_design(X_tr, X_va, X_te)
    logger.info("Standardization: %d binary columns left unscaled; %d scaled.",
                int(jnp.sum(is_bin)), X_tr.shape[1] - int(jnp.sum(is_bin)))

    # Transform targets for fitting (log1p or sqrt)
    z_tr = forward_transform(y_tr, y_transform)
    z_va = forward_transform(y_va, y_transform)
    z_te = forward_transform(y_te, y_transform)

    X_trh = jnp.concatenate([X_tr, X_va], axis=0)
    z_trh = jnp.concatenate([z_tr, z_va], axis=0)
    y_trh_orig = jnp.concatenate([y_tr, y_va], axis=0)

    logger.info("Design shapes: X_tr=(%d,%d), X_va=(%d,%d), X_trh=(%d,%d), X_te=(%d,%d)",
                X_tr.shape[0], X_tr.shape[1],
                X_va.shape[0], X_va.shape[1],
                X_trh.shape[0], X_trh.shape[1],
                X_te.shape[0], X_te.shape[1])

    cond_trh = _approx_condition_number(X_trh)
    logger.info(f"Approximate condition number of centered X_trh: {cond_trh:.3e}")

    model = RidgeClosedForm(lam=float(abl.lam), dtype=dtype, device=device, fit_intercept=True)
    fit = model.fit(X_trh, z_trh)
    logger.info(f"Fitted ridge: intercept b={fit.b:.6f}, training MSE sigma2={fit.sigma2:.6f} "
                f"on transformed target '{y_transform}'.")

    # Test design and predictions.
    zhat_trh = model.predict(X_trh, fit)
    zhat_te = model.predict(X_te, fit)

    smear = None
    if y_transform == "log1p":
        resid_z = z_trh - zhat_trh
        smear = smearing_factor(resid_z)
        logger.info("Smearing factor (log1p): %.6f", smear)

    yhat_trh = inverse_transform(zhat_trh, y_transform, smear=smear)
    yhat_te = inverse_transform(zhat_te, y_transform, smear=smear)
    yhat_trh = jnp.maximum(yhat_trh, 0.0)
    yhat_te = jnp.maximum(yhat_te, 0.0)

    # Metrics.
    rmse_trh = rmse(y_trh_orig, yhat_trh)
    rmse_test = rmse(y_te, yhat_te)
    mae_test = mae(y_te, yhat_te)
    r2_test = r2(y_te, yhat_te)
    logger.info(f"Final metrics: RMSE(tr+val)={rmse_trh:.4f}, RMSE(test)={rmse_test:.4f}, "
                f"MAE(test)={mae_test:.4f}, R2(test)={r2_test:.4f}")

    # Baselines on test.
    yhat_mean = jnp.maximum(mean_bl.predict(len(test_df)), 0.0)
    yhat_hod = jnp.maximum(hod_bl.predict(test_df), 0.0)
    rmse_mean = rmse(y_te, yhat_mean)
    rmse_hod = rmse(y_te, yhat_hod)
    logger.info(f"Baselines RMSE(test): mean={rmse_mean:.4f}, hour-of-day={rmse_hod:.4f}")

    # Save metrics workbook.
    metrics_df = pd.DataFrame(
        [
            {
                "model": "ridge",
                "y_transform": y_transform,
                "smearing": float(smear) if smear is not None else float("nan"),
                "features": ", ".join(abl.features),
                "lambda": float(abl.lam),
                "rmse_train_holdout": rmse_trh,
                "rmse_test": rmse_test,
                "mae_test": mae_test,
                "r2_test": r2_test,
            },
            {
                "model": "baseline_mean",
                "y_transform": "none",
                "smearing": float("nan"),
                "features": "-",
                "lambda": float("nan"),
                "rmse_train_holdout": float("nan"),
                "rmse_test": float(rmse_mean),
                "mae_test": float("nan"),
                "r2_test": float("nan"),
            },
            {
                "model": "baseline_hour_of_day",
                "y_transform": "none",
                "smearing": float("nan"),
                "features": "-",
                "lambda": float("nan"),
                "rmse_train_holdout": float("nan"),
                "rmse_test": float(rmse_hod),
                "mae_test": float("nan"),
                "r2_test": float("nan"),
            },
        ]
    )
    save_table_xlsx({"metrics": metrics_df}, cfg.paths.model_tables_dir / "metrics.xlsx")
    logger.info(f"Saved metrics -> {cfg.paths.model_tables_dir / "metrics.xlsx"}")

    # Training visualizations and tabular diagnostics

    # 1) Lambda curve for the selected subset on validation.
    lam_curve_df = viz_train.compute_lambda_curve(
        train_df, validation_df, y_tr, y_va, candidate_groups, abl.features, lam_grid, dtype, device
    )
    viz_train.plot_lambda_curve(lam_curve_df, cfg.paths.model_figures_dir / "lambda_curve_selected.png")
    # Save the curve as a sheet too.
    save_table_xlsx({"lambda_curve_selected": lam_curve_df}, cfg.paths.model_tables_dir / "training_report_lambda_curve.xlsx")

    # 2) Ablation path: best validation RMSE vs number of groups.
    ablation_path_df = viz_train.compute_ablation_path(
        train_df, validation_df, y_tr, y_va, candidate_groups, lam_grid, dtype, device
    )
    viz_train.plot_ablation_path(ablation_path_df, cfg.paths.model_figures_dir / "ablation_path.png")
    save_table_xlsx({"ablation_path": ablation_path_df}, cfg.paths.model_tables_dir / "training_report_ablation_path.xlsx")

    # 3) Coefficients for the final model.
    coef_df = viz_train.coefficient_table(fit, abl.features)
    viz_train.plot_coefficients(coef_df, cfg.paths.model_figures_dir / "coefficients_bar.png")
    save_table_xlsx({"coefficients": coef_df}, cfg.paths.model_tables_dir / "model_coefficients.xlsx")

    # 4) Test-set predictions and residual diagnostics.
    preds_df = viz_train.predictions_table(test_df.index, y_te, yhat_te, yhat_mean, yhat_hod)
    viz_train.plot_parity(preds_df, cfg.paths.model_figures_dir / "parity_test.png")
    viz_train.plot_residual_hist(preds_df, cfg.paths.model_figures_dir / "residual_hist_test.png")
    viz_train.plot_residual_vs_pred(preds_df, cfg.paths.model_figures_dir / "residual_vs_pred_test.png")
    viz_train.plot_timeseries_overlay(preds_df, cfg.paths.model_figures_dir / "timeseries_overlay_test.png")
    viz_train.plot_baseline_comparison(metrics_df, cfg.paths.model_figures_dir / "baseline_comparison.png")
    viz_train.plot_ablation_trace_table(abl_trace, cfg.paths.model_figures_dir / "ablation_trace_table.png")
    viz_train.plot_residuals_by_hour(preds_df, cfg.paths.model_figures_dir / "residuals_by_hour.png")
    save_table_xlsx({"predictions_test": preds_df}, cfg.paths.model_tables_dir / "predictions_test.xlsx")

    # 5) Save a light-weight checkpoint for reproducibility.
    viz_train.save_checkpoint(
        path=cfg.paths.model_tables_dir.parent / "models" / "ridge_checkpoint.npz",
        fit=fit,
        selected_features=abl.features,
        lam=abl.lam,
    )
    logger.info("Saved model checkpoint.")