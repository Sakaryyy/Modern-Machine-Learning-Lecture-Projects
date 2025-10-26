import math
from pathlib import Path
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator

from src.features.pipeline import FeaturePipeline
from src.models.linear_ridge_jax import RidgeClosedForm, FittedLinear
from src.utils.feature_mapping import ensure_hour_column
from src.utils.io import ensure_dir, save_figure

Array = jax.Array


# ---------------------------------------------------------------------
# Global plotting theme
# ---------------------------------------------------------------------
def _apply_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "font.size": 9,
    })


_apply_paper_style()


def _format_axes(ax: plt.Axes, *, zero_hline: bool = False, integer_xticks: bool = False,
                 integer_yticks: bool = False) -> None:
    if zero_hline:
        ax.axhline(0.0, lw=1.0, alpha=0.6)
    if integer_xticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if integer_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


# =========================
# Computation helpers
# =========================
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
    remaining = candidate_groups.copy()
    selected: List[Tuple[List[str], FeaturePipeline]] = []
    greedy_order: List[Tuple[List[str], FeaturePipeline]] = []

    # Build greedy order via look-ahead
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

    # Prefix evaluation over lambda grid
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


# =========================
# Plotting helpers
# =========================
def plot_ablation_trace_table(df_trace: pd.DataFrame, out_path: Path) -> Path:
    """
    Plot the greedy ablation trace with a clean, non-overlapping layout:
    top panel = best-overall validation RMSE by step,
    bottom panel = a compact table listing the chosen group per step.
    """
    ensure_dir(out_path.parent)

    fig = plt.figure(figsize=(9, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.5, 1.0])
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    sns.lineplot(
        data=df_trace,
        x="step",
        y="best_overall_rmse_val_so_far",
        marker="o",
        ax=ax,
        label="best-overall val RMSE",
    )
    ax.set_title("Greedy ablation trace")
    ax.set_xlabel("Step")
    ax.set_ylabel("Best-overall validation RMSE")
    _format_axes(ax, integer_xticks=True)

    # Highlight current best
    i_min = int(df_trace["best_overall_rmse_val_so_far"].idxmin())
    x_min = df_trace.loc[i_min, "step"]
    y_min = df_trace.loc[i_min, "best_overall_rmse_val_so_far"]
    ax.scatter([x_min], [y_min], zorder=5)
    ax.annotate(f"min @ step {int(x_min)}",
                xy=(x_min, y_min), xytext=(6, 6),
                textcoords="offset points", fontsize=8)

    # Clean table of chosen groups
    ax_table.axis("off")
    tbl_df = df_trace[["step", "chosen_group"]].copy()
    tbl_df.columns = ["Step", "Chosen group"]
    table = ax_table.table(
        cellText=tbl_df.values,
        colLabels=tbl_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.15)

    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_residuals_by_hour(
    df_preds: pd.DataFrame,
    out_path: Path,
    *,
    split: str = "test",
) -> Path:
    """
    Plot mean residual by hour-of-day with 1 std error bars (readable, integer x-ticks).
    """
    ensure_dir(out_path.parent)
    dfx = ensure_hour_column(df_preds)
    residuals = dfx["y_true"] - dfx["y_hat"]
    tmp = dfx.assign(res=residuals).groupby("hr")["res"].agg(["mean", "std", "count"])
    tmp["stderr"] = tmp["std"] / np.sqrt(tmp["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.errorbar(tmp.index.values, tmp["mean"].values, yerr=tmp["stderr"].values, fmt="o-", capsize=2)
    ax.set_title(f"Residuals by hour-of-day ({split})")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean residual ± 1 s.e.")
    _format_axes(ax, zero_hline=True, integer_xticks=True)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_lambda_curve(df_curve: pd.DataFrame, out_path: Path) -> Path:
    """
    Validation and train RMSE vs lambda (log-x), with minimum highlighted.
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    sns.lineplot(data=df_curve, x="lambda", y="rmse_val", marker="o", ax=ax, label="validation")
    sns.lineplot(data=df_curve, x="lambda", y="rmse_tr", marker="s", ax=ax, label="train")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.set_title("RMSE vs λ")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("RMSE")
    _format_axes(ax)

    # Highlight min validation
    idx = df_curve["rmse_val"].idxmin()
    x_star = df_curve.loc[idx, "lambda"]
    y_star = df_curve.loc[idx, "rmse_val"]
    ax.scatter([x_star], [y_star], zorder=5)
    ax.annotate(f"min @ λ={x_star:.2g}", xy=(x_star, y_star), xytext=(6, 6),
                textcoords="offset points", fontsize=8)
    ax.legend(loc="best", frameon=False)
    return save_figure(fig, out_path)


def plot_ablation_path(df_path: pd.DataFrame, out_path: Path) -> Path:
    """
    Best validation RMSE vs number of groups
    """
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(7.5, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    sns.lineplot(data=df_path, x="k_groups", y="best_rmse_val", marker="o", ax=ax)
    ax.set_title("Ablation path")
    ax.set_xlabel("Number of groups selected")
    ax.set_ylabel("Best validation RMSE")
    _format_axes(ax, integer_xticks=True)

    # Min marker
    i_min = int(df_path["best_rmse_val"].idxmin())
    x_min = df_path.loc[i_min, "k_groups"]
    y_min = df_path.loc[i_min, "best_rmse_val"]
    ax.scatter([x_min], [y_min], zorder=5)
    ax.annotate(f"min @ k={int(x_min)}", xy=(x_min, y_min), xytext=(6, 6),
                textcoords="offset points", fontsize=8)

    # Bottom table of last-added groups
    ax_tbl.axis("off")
    show = df_path[["k_groups", "last_added_group", "best_lambda"]].copy()
    show.columns = ["k", "last added group", "best λ"]
    table = ax_tbl.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.15)

    return save_figure(fig, out_path)


def coefficient_table(fit: FittedLinear, feature_names: List[str]) -> pd.DataFrame:
    w = np.asarray(fit.w)
    return pd.DataFrame(
        {"feature": feature_names, "coef": w, "abs_coef": np.abs(w)}
    ).sort_values("abs_coef", ascending=False)


def plot_coefficients(df_coef: pd.DataFrame, out_path: Path) -> Path:
    """
    Horizontal bar chart of coefficient magnitudes (top-N if very long).
    """
    ensure_dir(out_path.parent)
    df = df_coef.copy()
    N = len(df)
    max_show = 40
    if N > max_show:
        df = df.head(max_show)

    fig, ax = plt.subplots(figsize=(8.0, 0.25 + 0.28 * len(df)))
    sns.barplot(data=df, x="abs_coef", y="feature", ax=ax, orient="h")
    ax.set_title("Coefficient magnitudes (sorted by |coef|)")
    ax.set_xlabel("|coef|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return save_figure(fig, out_path)


def predictions_table(
    index: pd.Index,
    y_true: Array,
    y_hat: Array,
    y_hat_mean: Array,
    y_hat_hod: Array,
    *,
    split: str,
) -> pd.DataFrame:
    """Return a table with predictions for a particular data split."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_hat": np.asarray(y_hat),
            "y_hat_mean": np.asarray(y_hat_mean),
            "y_hat_hod": np.asarray(y_hat_hod),
        },
        index=index,
    )
    df.index.name = "timestamp"
    df["split"] = split
    return df


def plot_parity(
        df_preds: pd.DataFrame,
        out_path: Path,
        *,
        split: str = "test"
) -> Path:
    """
    Parity scatter with 1:1 line and inset stats (RMSE, R²).
    Robust to NaN/Inf and degenerate ranges.
    """
    ensure_dir(out_path.parent)

    # Keep only finite pairs
    dfx = df_preds[["y_true", "y_hat"]].replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)

    if dfx.empty:
        # Graceful fallback: informative placeholder plot
        ax.text(0.5, 0.5, "No finite y_true / y_hat to plot", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return save_figure(fig, out_path)

    # Scatter (density-aware)
    sns.scatterplot(data=dfx, x="y_true", y="y_hat", s=10, alpha=0.5, ax=ax, edgecolor=None)

    y_true = dfx["y_true"].to_numpy()
    y_hat = dfx["y_hat"].to_numpy()

    # Compute safe limits
    lo = float(np.min([y_true.min(), y_hat.min()]))
    hi = float(np.max([y_true.max(), y_hat.max()]))

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0  # final safety net

    # Avoid zero-width range
    if hi - lo <= 0:
        pad = 1.0 if hi == 0.0 else 0.05 * max(1.0, abs(hi))
        lo -= pad
        hi += pad

    # 1:1 line and limits
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Stats on filtered data
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_true - y_hat) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

    ax.set_title(f"Parity plot ({split})")
    ax.set_xlabel("True cnt")
    ax.set_ylabel("Predicted cnt")
    ax.text(
        0.02, 0.98, f"RMSE = {rmse:.3g}\nR² = {r2:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    return save_figure(fig, out_path)


def _fd_bins(x: np.ndarray) -> int:
    """
    Robust Freedman–Diaconis bin count.
    Handles NaNs/Inf, empty arrays, and zero-range data gracefully.
    Returns a sensible bin count in [10, 80].
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 10

    rng = x.max() - x.min()
    if not np.isfinite(rng) or rng <= 0:
        return min(40, max(10, int(np.sqrt(n))))

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        return min(40, max(10, int(np.sqrt(n))))

    h = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if not np.isfinite(h) or h <= 0:
        return min(40, max(10, int(np.sqrt(n))))

    bins = (rng / h)
    if not np.isfinite(bins) or bins <= 0:
        return min(40, max(10, int(np.sqrt(n))))

    return int(np.clip(np.ceil(bins), 10, 80))


def plot_residual_hist(
        df_preds: pd.DataFrame,
        out_path: Path,
        *,
        split: str = "test"
) -> Path:
    """
    Residual histogram with mean/±1σ markers.
    Robust to NaN/Inf and empty input.
    """
    ensure_dir(out_path.parent)

    # Keep only finite pairs
    dfx = (
        df_preds[["y_true", "y_hat"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)

    if dfx.empty:
        ax.text(0.5, 0.5, "No finite residuals to plot", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return save_figure(fig, out_path)

    residuals = (dfx["y_true"] - dfx["y_hat"]).to_numpy()
    if residuals.size == 0 or not np.isfinite(residuals).any():
        ax.text(0.5, 0.5, "No finite residuals to plot", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return save_figure(fig, out_path)

    residuals = residuals[np.isfinite(residuals)]
    bins = _fd_bins(residuals)

    sns.histplot(residuals, bins=bins, ax=ax, edgecolor=None)
    mu = float(np.mean(residuals)) if residuals.size else 0.0
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0

    for v, ls in [(mu, "-"), (mu - sigma, ":"), (mu + sigma, ":")]:
        if np.isfinite(v):
            ax.axvline(v, lw=1.0, linestyle=ls, alpha=0.9)

    ax.set_title(f"Residual histogram ({split})")
    ax.set_xlabel("Residual = y_true − y_hat")
    ax.set_ylabel("Count")
    ax.text(
        0.98, 0.98, f"mean={mu:.3g}\nσ={sigma:.3g}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    return save_figure(fig, out_path)


def plot_residual_vs_pred(
    df_preds: pd.DataFrame,
    out_path: Path,
    *,
    split: str = "test",
) -> Path:
    """
    Residuals vs predictions with light scatter and binned mean+/-s.e. trend.
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    residuals = df_preds["y_true"] - df_preds["y_hat"]
    sns.scatterplot(x=df_preds["y_hat"], y=residuals, s=10, alpha=0.35, ax=ax, edgecolor=None)

    # Binned trend (equal-count bins)
    yhat = df_preds["y_hat"].to_numpy()
    res = residuals.to_numpy()
    n = len(yhat)
    if n >= 20:
        q = np.linspace(0, 1, 21)
        edges = np.quantile(yhat, q)
        # de-duplicate edges
        edges = np.unique(edges)
        if len(edges) >= 5:
            idx = np.digitize(yhat, edges[1:-1], right=True)
            means = []
            centers = []
            stderrs = []
            for k in range(len(edges) - 1):
                m = res[idx == k]
                h = yhat[idx == k]
                if m.size > 1:
                    centers.append(np.mean(h))
                    means.append(np.mean(m))
                    stderrs.append(np.std(m, ddof=1) / math.sqrt(m.size))
            if centers:
                ax.errorbar(centers, means, yerr=stderrs, fmt="o-", capsize=2, lw=1.2)

    ax.set_title(f"Residuals vs predictions ({split})")
    ax.set_xlabel("Predicted cnt")
    ax.set_ylabel("Residual")
    _format_axes(ax, zero_hline=True)
    return save_figure(fig, out_path)


def plot_timeseries_overlay(
    df_preds: pd.DataFrame,
    out_path: Path,
    *,
    split: str = "test",
) -> Path:
    """
    Overlay of true vs predicted time series
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
    df_preds["y_true"].plot(ax=ax, label="true", linewidth=1.0)
    df_preds["y_hat"].plot(ax=ax, label="ridge", linewidth=1.0)
    ax.set_title(f"{split.capitalize()} time series: true vs predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cnt")
    ax.legend(loc="best", frameon=False, ncol=2)
    return save_figure(fig, out_path)


def plot_baseline_comparison(metrics_df: pd.DataFrame, out_path: Path) -> Path:
    """
    Test RMSE by model, with value annotations.
    """
    ensure_dir(out_path.parent)
    df = metrics_df[["model", "rmse_test"]].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    g = sns.barplot(data=df, x="model", y="rmse_test", ax=ax)
    ax.set_title("Test RMSE by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (test)")
    for p in g.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3g}", (p.get_x() + p.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    return save_figure(fig, out_path)


def save_checkpoint(path: Path, fit: FittedLinear, selected_features: List[str], lam: float) -> None:
    """
    Save a minimal checkpoint with weights, intercept, lambda, and feature names.
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
