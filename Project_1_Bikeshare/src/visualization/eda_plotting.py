from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir, save_figure, save_table_xlsx
from src.utils.feature_mapping import add_interpretable_columns, ensure_hour_column

logger = logging.getLogger(__name__)

r"""
EDA visualization utilities.

Three common cases for Correlation Analysis:

1) NUMERIC–NUMERIC  (for example cnt vs atemp):
   Pearson r:
       r(x, y) = Cov(x, y) / (sigma_x sigma_y)
   Measures linear association in [-1, 1]. r=1/-1 indicates perfect linear relation
   with positive/negative slope. r=~0 indicates no linear association.

2) CATEGORICAL–NUMERIC (for example cnt vs season, weekday, workingday, hr, ...):
   Correlation ratio (eta, "eta"):
       eta = sqrt( SS_between / SS_total )
   where:
       SS_total  = sum_i (y_i - y_mean)^2
       SS_between = sum_k n_k (y_mean_k - y_mean)^2
       y_mean is the global mean of y, y_mean_k is the mean of y within category k, n_k size of k.
   eta element of [0, 1]. eta = 0 means the category tells us nothing about ys mean. eta close to 1
   means category explains most of the variance of ys mean.

3) CATEGORICAL–CATEGORICAL (like weekday vs season):
   This is not relevant for our data here!
"""

# Styling
sns.set_context("talk")
sns.set_style("whitegrid")

def _figsize() -> tuple[float, float]:
    return 10.0, 4.0


# Correlation computations with clear docstrings
def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute Pearsons r between two numeric series.

    r(x, y) = Cov(x, y) / (sigma_x sigma_y), in [-1, 1].
    Assumes linear relationship. Sensitive to outliers!
    """
    s = pd.concat([x, y], axis=1).dropna()
    if s.shape[0] < 2:
        return np.nan
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))

def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta(categories -> values).

    eta = sqrt(SS_between / SS_total), where:
        SS_total  = sum (y_i - y_mean)^2
        SS_between = sum_k n_k (y_mean_k - y_mean)^2
    Returns eta element of [0, 1].
    """
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty:
        return np.nan
    overall_mean = df["val"].mean()
    ss_total = ((df["val"] - overall_mean) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    grouped = df.groupby("cat", observed=True)["val"]
    means = grouped.mean()
    counts = grouped.size()
    ss_between = (counts * (means - overall_mean) ** 2).sum()
    eta = np.sqrt(float(ss_between / ss_total))
    return float(eta)

# Core numeric plots
def plot_time_series_cnt(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=_figsize())
    df["cnt"].plot(ax=ax)
    ax.set_title("Bike rentals over time (cnt)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count (cnt)")
    out_path = out_dir / "ts_cnt.png"
    path = save_figure(fig, out_path)
    plt.close(fig)
    return path

def plot_hist_cnt(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["cnt"], bins=40, ax=ax)
    ax.set_title("Distribution of cnt")
    ax.set_xlabel("cnt")
    out_path = out_dir / "hist_cnt.png"
    path = save_figure(fig, out_path)
    plt.close(fig)
    return path

def plot_hourly_profile(df: pd.DataFrame, out_dir: Path) -> Path:
    if "hr" in df.columns:
        hour = df["hr"].astype(int)
    else:
        hour = df.index.hour
    prof = df.assign(hour=hour).groupby("hour")["cnt"].mean().reset_index()
    fig, ax = plt.subplots(figsize=_figsize())
    sns.lineplot(data=prof, x="hour", y="cnt", marker="o", ax=ax)
    ax.set_title("Mean cnt by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean cnt")
    out_path = out_dir / "hourly_profile.png"
    path = save_figure(fig, out_path)
    plt.close(fig)
    return path

# Numeric scatter plots
def plot_scatter_features(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    dfi = add_interpretable_columns(df)
    pairs: list[Tuple[str, str]] = [
        ("atemp_C", "Apparent temperature (°C)"),
        ("temp_C", "Temperature (°C)"),
        ("humidity_pct", "Humidity (%)"),
        ("windspeed_0_67", "Wind speed (0..67)"),
    ]
    for col, label in pairs:
        if col in dfi.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=dfi, x=col, y="cnt", s=10, alpha=0.5, ax=ax)
            r = pearson_corr(dfi[col], dfi["cnt"])
            ax.set_title(f"cnt vs {label}  (Pearson r = {r:.3f})")
            ax.set_xlabel(label)
            ax.set_ylabel("cnt")
            out[col] = save_figure(fig, out_dir / f"scatter_cnt_{col}.png")
            plt.close(fig)
    return out

def plot_corr_heatmap(df: pd.DataFrame, out_dir: Path) -> Path | None:
    # Only numeric columns
    dfi = add_interpretable_columns(df)
    num = dfi.select_dtypes(include=[np.number])
    if num.empty:
        return None
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, ax=ax, annot=False, cmap="viridis", square=False)
    ax.set_title("Numeric feature correlation (Pearson r)")
    out_path = out_dir / "corr_heatmap.png"
    path = save_figure(fig, out_path)
    plt.close(fig)
    return path

# Categorical analyses and plots
CAT_VARIANTS = [
    ("season_name", "Season"),
    ("yr_name", "Year"),
    ("mnth_name", "Month"),
    ("weekday_name", "Weekday"),
    ("holiday_name", "Holiday"),
    ("workingday_name", "Workday"),
    ("weathersit_name", "Weather"),
    ("hr", "Hour of day"),
]


def plot_cnt_by_category_means(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Bar plots of mean cnt by categorical feature.

    For each categorical feature, we compute the mean of cnt per category and plot
    with normal approximation.
    """
    out: dict[str, Path] = {}
    dfc = add_interpretable_columns(ensure_hour_column(df))
    for col, label in CAT_VARIANTS:
        if col not in dfc.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=dfc, x=col, y="cnt", errorbar=("ci", 95), ax=ax)
        ax.set_title(f"Mean cnt by {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Mean cnt")
        if dfc[col].dtype.name != "category" and col != "hr":
            dfc[col] = dfc[col].astype("category")
        eta = correlation_ratio(dfc[col], dfc["cnt"])
        ax.set_title(f"Mean cnt by {label}  (Correlation ratio eta = {eta:.3f})")
        out[col] = save_figure(fig, out_dir / f"cat_mean_cnt_{col}.png")
        plt.close(fig)
    return out

def plot_cnt_by_category_box(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Box plots of cnt by categorical feature (distributional view)."""
    out: dict[str, Path] = {}
    dfc = add_interpretable_columns(ensure_hour_column(df))
    for col, label in CAT_VARIANTS:
        if col not in dfc.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=dfc, x=col, y="cnt", ax=ax, showfliers=False)
        ax.set_title(f"cnt distribution by {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("cnt")
        out[col] = save_figure(fig, out_dir / f"cat_box_cnt_{col}.png")
        plt.close(fig)
    return out

def plot_categorical_eta_bar(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Bar chart summarizing correlation ratio eta(cat to cnt) across categorical features."""
    dfc = add_interpretable_columns(ensure_hour_column(df))
    rows = []
    for col, label in CAT_VARIANTS:
        if col in dfc.columns:
            eta = correlation_ratio(dfc[col], dfc["cnt"])
            rows.append({"feature": label, "eta": eta})
    if not rows:
        return None
    eta_df = pd.DataFrame(rows).sort_values("eta", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=eta_df, x="feature", y="eta", ax=ax)
    ax.set_title("Categorical to numeric association: correlation ratio eta with cnt")
    ax.set_xlabel("Categorical feature")
    plt.xticks(rotation=45)
    ax.set_ylabel("eta(cat to cnt)")
    out_path = out_dir / "categorical_eta_bar.png"
    path = save_figure(fig, out_path)
    plt.close(fig)
    return path

# Tables
def export_eda_tables(df: pd.DataFrame, out_dir: Path) -> Path:
    """Export EDA summary tables to an Excel workbook.

    Sheets:
        - describe: pandas describe() including non-numerics (transposed)
        - corr_with_cnt: Pearson r for numeric columns vs cnt
        - categorical_eta: correlation ratio eta(cat to cnt) for selected categorical features
    """
    dfi = add_interpretable_columns(ensure_hour_column(df))

    desc = dfi.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
    frames: dict[str, pd.DataFrame] = {"describe": desc}

    if "cnt" in dfi.columns:
        num = dfi.select_dtypes(include=[np.number])
        if not num.empty:
            corr_cnt = num.corr()[["cnt"]].reset_index().rename(columns={"index": "feature"})
            frames["corr_with_cnt"] = corr_cnt

        # categorical eta
        rows = []
        for col, label in CAT_VARIANTS:
            if col in dfi.columns:
                rows.append(
                    {"feature": label, "eta": correlation_ratio(dfi[col], dfi["cnt"])}
                )
        if rows:
            frames["categorical_eta"] = pd.DataFrame(rows).sort_values("eta", ascending=False)

    out_path = out_dir / "eda_summary.xlsx"
    return save_table_xlsx(frames, out_path)

# Main Run API
def run_all(df: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    """Run the full visualization suite and save outputs.

    Returns
    -------
    dict
        Map of plot/table names to output paths.

    Notes
    -----
    - Numeric–numeric correlation uses Pearsons r (linear association).
    - Categorical–numeric association uses the correlation ratio eta(cat to cnt).
    """
    ensure_dir(out_dir)
    results: dict[str, Path] = {
        "ts_cnt": plot_time_series_cnt(df, out_dir),
        "hist_cnt": plot_hist_cnt(df, out_dir),
        "hourly_profile": plot_hourly_profile(df, out_dir)
    }

    # Numeric relationships (interpretable units)
    results.update({f"scatter_{k}": v for k, v in plot_scatter_features(df, out_dir).items()})
    heat = plot_corr_heatmap(df, out_dir)
    if heat is not None:
        results["corr_heatmap"] = heat

    # Categorical analyses
    for name, path in plot_cnt_by_category_means(df, out_dir).items():
        results[f"cat_mean_{name}"] = path
    for name, path in plot_cnt_by_category_box(df, out_dir).items():
        results[f"cat_box_{name}"] = path
    eta_bar = plot_categorical_eta_bar(df, out_dir)
    if eta_bar is not None:
        results["categorical_eta_bar"] = eta_bar

    # Tables
    results["eda_tables"] = export_eda_tables(df, out_dir.parent / "tables")
    return results
