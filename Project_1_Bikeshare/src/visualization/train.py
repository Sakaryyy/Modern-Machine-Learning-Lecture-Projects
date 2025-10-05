from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io import ensure_dir, save_figure
from src.models.linear_ridge_jax import RidgeClosedForm, FittedLinear
from src.features.pipeline import FeaturePipeline
from src.utils.feature_mapping import ensure_hour_column

Array = jax.Array
sns.set_context("talk")
sns.set_style("whitegrid")


# Computation helpers
def _build_design_from_selected(
    df: pd.DataFrame,
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    selected_cols: List[str],
    dtype: jnp.dtype,
) -> Array:
    mats: List[Array] = []
    for _, pipe in candidate_groups:
        X, cols = pipe.transform(df)
        idx = [i for i, c in enumerate(cols) if c in selected_cols]
        if idx:
            X_sel = X[:, jnp.array(idx, dtype=jnp.int32)]
            mats.append(X_sel)
    if mats:
        return jnp.concatenate(mats, axis=1)
    return jnp.empty((len(df), 0), dtype=dtype)


def compute_lambda_curve(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    y_tr: Array,
    y_val: Array,
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    selected_cols: List[str],
    lam_grid: Sequence[float],
    dtype: jnp.dtype,
    device: jax.Device | None,
) -> pd.DataFrame:
    """
    Compute validation RMSE across a lambda grid for the final selected subset.

    Returns
    -------
    pandas.DataFrame
        Columns: ["lambda", "rmse_val", "rmse_tr"].
    """
    Xtr = _build_design_from_selected(df_tr, candidate_groups, selected_cols, dtype)
    Xva = _build_design_from_selected(df_val, candidate_groups, selected_cols, dtype)

    rows = []
    for lam in lam_grid:
        model = RidgeClosedForm(lam=float(lam), dtype=dtype, device=device, fit_intercept=True)
        fit = model.fit(Xtr, y_tr)
        y_tr_pred = model.predict(Xtr, fit)
        y_va_pred = model.predict(Xva, fit)
        rows.append(
            {
                "lambda": float(lam),
                "rmse_tr": float(jnp.sqrt(jnp.mean((y_tr - y_tr_pred) ** 2))),
                "rmse_val": float(jnp.sqrt(jnp.mean((y_val - y_va_pred) ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values("lambda")


def compute_ablation_path(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    y_tr: Array,
    y_val: Array,
    candidate_groups: List[Tuple[List[str], FeaturePipeline]],
    lam_grid: Sequence[float],
    dtype: jnp.dtype,
    device: jax.Device | None,
) -> pd.DataFrame:
    """
    For k = 1..G, compute the best validation RMSE achievable by the first k groups
    in a greedy-forward order. This mirrors the subset re-evaluation performed
    when choosing a minimal set within tolerance.
    """
    # Build a fixed greedy order by replaying the "forward" logic once.
    remaining = candidate_groups.copy()
    selected: List[Tuple[List[str], FeaturePipeline]] = []
    greedy_order: List[Tuple[List[str], FeaturePipeline]] = []

    # Establish greedy order by repeated look-ahead.
    while remaining:
        best_rec = None
        for names, pipe in remaining:
            trial_pipes = [p for _, p in selected] + [pipe]
            Xtr = jnp.concatenate([p.transform(df_tr)[0] for p in trial_pipes], axis=1) if trial_pipes else jnp.empty((len(df_tr), 0), dtype=dtype)
            Xva = jnp.concatenate([p.transform(df_val)[0] for p in trial_pipes], axis=1) if trial_pipes else jnp.empty((len(df_val), 0), dtype=dtype)
            for lam in lam_grid:
                model = RidgeClosedForm(lam=float(lam), dtype=dtype, device=device, fit_intercept=True)
                fit = model.fit(Xtr, y_tr)
                y_va_pred = model.predict(Xva, fit)
                s_va = float(jnp.sqrt(jnp.mean((y_val - y_va_pred) ** 2)))
                rec = (s_va, names, pipe)
                if best_rec is None or s_va < best_rec[0]:
                    best_rec = rec
        assert best_rec is not None
        _, picked_names, picked_pipe = best_rec
        selected.append((picked_names, picked_pipe))
        greedy_order.append((picked_names, picked_pipe))
        remaining = [g for g in remaining if g[0] != picked_names]

    # For each prefix length k, rescan the lambda grid and record best validation RMSE.
    rows = []
    pipes_acc: List[FeaturePipeline] = []
    for k, (names, pipe) in enumerate(greedy_order, start=1):
        pipes_acc.append(pipe)
        Xtr = jnp.concatenate([p.transform(df_tr)[0] for p in pipes_acc], axis=1) if pipes_acc else jnp.empty((len(df_tr), 0), dtype=dtype)
        Xva = jnp.concatenate([p.transform(df_val)[0] for p in pipes_acc], axis=1) if pipes_acc else jnp.empty((len(df_val), 0), dtype=dtype)

        best_val = float("inf")
        best_lam = float("nan")
        for lam in lam_grid:
            model = RidgeClosedForm(lam=float(lam), dtype=dtype, device=device, fit_intercept=True)
            fit = model.fit(Xtr, y_tr)
            y_va_pred = model.predict(Xva, fit)
            s_va = float(jnp.sqrt(jnp.mean((y_val - y_va_pred) ** 2)))
            if s_va < best_val:
                best_val = s_va
                best_lam = float(lam)

        rows.append(
            {
                "k_groups": k,
                "last_added_group": ", ".join(names),
                "best_lambda": best_lam,
                "best_rmse_val": best_val,
            }
        )

    return pd.DataFrame(rows)


# Plotting helpers
def plot_ablation_trace_table(df_trace: pd.DataFrame, out_path: Path) -> Path:
    """
    Plot the greedy ablation trace as a line plot of best-overall validation RMSE
    against step index, annotated with the chosen group per step.

    Parameters
    ----------
    df_trace : pandas.DataFrame
        Output of forward_ablation(..., record_trace=True).
    out_path : pathlib.Path
        Where to write the PNG.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=df_trace,
        x="step",
        y="best_overall_rmse_val_so_far",
        marker="o",
        ax=ax,
        label="best overall val RMSE so far",
    )
    ax.set_title("Greedy ablation trace")
    ax.set_xlabel("step")
    ax.set_ylabel("best-overall validation RMSE")
    # Annotate last-added group at each step
    for _, row in df_trace.iterrows():
        ax.annotate(
            row["chosen_group"],
            (row["step"], row["best_overall_rmse_val_so_far"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_residuals_by_hour(df_preds: pd.DataFrame, out_path: Path) -> Path:
    """
    Plot mean residual by hour-of-day with 1 std error bars.

    Parameters
    ----------
    df_preds : pandas.DataFrame
        Table with columns y_true, y_hat and a DatetimeIndex. If an 'hr' column
        is present, it will be used instead of deriving hour from the index.
    out_path : pathlib.Path
        Where to write the PNG.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """
    ensure_dir(out_path.parent)
    dfx = ensure_hour_column(df_preds)
    residuals = dfx["y_true"] - dfx["y_hat"]
    tmp = dfx.assign(res=residuals).groupby("hr")["res"].agg(["mean", "std", "count"])
    tmp["stderr"] = tmp["std"] / np.sqrt(tmp["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(tmp.index.values, tmp["mean"].values, yerr=tmp["stderr"].values, fmt="o-")
    ax.set_title("Residuals by hour-of-day (test)")
    ax.set_xlabel("hour")
    ax.set_ylabel("mean residual ± 1 stderr")
    ax.axhline(0.0)
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_lambda_curve(df_curve: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df_curve, x="lambda", y="rmse_val", marker="o", ax=ax, label="validation")
    sns.lineplot(data=df_curve, x="lambda", y="rmse_tr", marker="o", ax=ax, label="train")
    ax.set_xscale("log")
    ax.set_title("Validation and Train RMSE vs lambda")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    return save_figure(fig, out_path)


def plot_ablation_path(df_path: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df_path, x="k_groups", y="best_rmse_val", marker="o", ax=ax)
    ax.set_title("Ablation path: best validation RMSE vs number of groups")
    ax.set_xlabel("number of groups selected")
    ax.set_ylabel("best validation RMSE")
    return save_figure(fig, out_path)


def coefficient_table(fit: FittedLinear, feature_names: List[str]) -> pd.DataFrame:
    w = np.asarray(fit.w)
    return pd.DataFrame(
        {"feature": feature_names, "coef": w, "abs_coef": np.abs(w)}
    ).sort_values("abs_coef", ascending=False)


def plot_coefficients(df_coef: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_coef, x="abs_coef", y="feature", ax=ax, orient="h")
    ax.set_title("Coefficient magnitudes (sorted by absolute value)")
    ax.set_xlabel("|coef|")
    ax.set_ylabel("feature")
    return save_figure(fig, out_path)


def predictions_table(
    index: pd.Index,
    y_true: Array,
    y_hat: Array,
    y_hat_mean: Array,
    y_hat_hod: Array,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_hat": np.asarray(y_hat),
            "y_hat_mean": np.asarray(y_hat_mean),
            "y_hat_hod": np.asarray(y_hat_hod),
        },
        index=index,
    )


def plot_parity(df_preds: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df_preds, x="y_true", y="y_hat", s=12, alpha=0.6, ax=ax)
    lims = [
        min(df_preds["y_true"].min(), df_preds["y_hat"].min()),
        max(df_preds["y_true"].max(), df_preds["y_hat"].max()),
    ]
    ax.plot(lims, lims)
    ax.set_title("Parity plot (test)")
    ax.set_xlabel("true cnt")
    ax.set_ylabel("predicted cnt")
    return save_figure(fig, out_path)


def plot_residual_hist(df_preds: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    residuals = df_preds["y_true"] - df_preds["y_hat"]
    sns.histplot(residuals, bins=40, ax=ax)
    ax.set_title("Residual histogram (test)")
    ax.set_xlabel("residual = y_true - y_hat")
    ax.set_ylabel("count")
    return save_figure(fig, out_path)


def plot_residual_vs_pred(df_preds: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    residuals = df_preds["y_true"] - df_preds["y_hat"]
    sns.scatterplot(x=df_preds["y_hat"], y=residuals, s=12, alpha=0.6, ax=ax)
    ax.axhline(0.0)
    ax.set_title("Residuals vs predictions (test)")
    ax.set_xlabel("predicted cnt")
    ax.set_ylabel("residual")
    return save_figure(fig, out_path)


def plot_timeseries_overlay(df_preds: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    df_preds["y_true"].plot(ax=ax, label="true", linewidth=1.0)
    df_preds["y_hat"].plot(ax=ax, label="ridge", linewidth=1.0)
    ax.set_title("Test time series: true vs predicted")
    ax.set_xlabel("time")
    ax.set_ylabel("cnt")
    ax.legend(loc="best")
    return save_figure(fig, out_path)


def plot_baseline_comparison(metrics_df: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    df = metrics_df[["model", "rmse_test"]].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="model", y="rmse_test", ax=ax)
    ax.set_title("Test RMSE by model")
    ax.set_xlabel("model")
    ax.set_ylabel("RMSE on test")
    return save_figure(fig, out_path)


def save_checkpoint(path: Path, fit: FittedLinear, selected_features: List[str], lam: float) -> None:
    """
    Save a minimal checkpoint with weights, intercept, lambda, and feature names.

    Parameters
    ----------
    path : pathlib.Path
        Output file location. Parent directories are created if needed.
    fit : FittedLinear
        Fitted ridge model.
    selected_features : list of str
        Feature names matching the order in fit.w.
    lam : float
        Ridge lambda used for the final fit.
    """
    ensure_dir(path.parent)
    np.savez_compressed(
        str(path),
        w=np.asarray(fit.w),
        b=np.array([fit.b], dtype=float),
        lam=np.array([lam], dtype=float),
        features=np.array(selected_features, dtype=object),
        dtype=str(fit.dtype),
    )
