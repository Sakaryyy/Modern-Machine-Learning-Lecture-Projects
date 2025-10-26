from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.utils.io import ensure_dir, save_figure


# ---------------------------------------------------------------------
# Unified compact style
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


def plot_confusion_matrix(confusion: jnp.ndarray, out_path: Path, *, title: str) -> Path:
    """
    Plot a confusion matrix heatmap
    """
    ensure_dir(out_path.parent)
    data = jnp.asarray(confusion, dtype=jnp.float32)
    n = int(data.shape[0]) if data.ndim == 2 else int(jnp.sqrt(data.size))
    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"shrink": 0.85},
        square=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted hour")
    ax.set_ylabel("True hour")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return save_figure(fig, out_path)


def plot_ablation_trace(trace: pd.DataFrame, out_path: Path) -> Path:
    """
    Greedy feature ablation: accuracy over steps.
    """
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(9.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.5, 1.0])
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    sns.lineplot(data=trace, x="step", y="val_accuracy", marker="o", ax=ax, label="validation")
    sns.lineplot(data=trace, x="step", y="train_accuracy", marker="s", ax=ax, label="train")
    ax.set_ylim(0, 1)
    ax.set_title("Greedy feature ablation: accuracy by step")
    ax.set_xlabel("Step (number of feature groups selected)")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", frameon=False)

    # Mark max validation
    i_max = int(trace["val_accuracy"].idxmax())
    x_max = trace.loc[i_max, "step"]
    y_max = trace.loc[i_max, "val_accuracy"]
    ax.scatter([x_max], [y_max], zorder=5)
    ax.annotate(f"max @ step {int(x_max)}", xy=(x_max, y_max), xytext=(6, 6),
                textcoords="offset points", fontsize=8)

    # Bottom table for groups
    ax_tbl.axis("off")
    tbl = trace[["step", "chosen_group"]].copy()
    tbl.columns = ["Step", "Chosen group"]
    table = ax_tbl.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.15)

    return save_figure(fig, out_path)


def plot_training_history(history: pd.DataFrame, out_path: Path) -> Path:
    """
    Optimisation diagnostics recorded during softmax training (loss & accuracy).
    """
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    sns.lineplot(data=history, x="iteration", y="loss", hue="phase", ax=axes[0])
    axes[0].set_title("Loss trajectory")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective")

    sns.lineplot(data=history, x="iteration", y="accuracy", hue="phase", ax=axes[1])
    axes[1].set_title("In-sample accuracy trajectory")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=len(labels), loc="upper center", frameon=False)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.suptitle("Softmax regression optimisation diagnostics", x=0.01, ha="left", fontsize=10)
    return save_figure(fig, out_path)


def plot_hour_accuracy(per_hour: pd.DataFrame, out_path: Path) -> Path:
    """
    Per-hour accuracy across splits.
    Background bars show total support (all splits), lines show accuracy per split.
    """
    ensure_dir(out_path.parent)

    support_bg = per_hour.groupby("hour", as_index=False)["support"].sum()
    fig, ax1 = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)

    # Background bars
    sns.barplot(data=support_bg, x="hour", y="support", alpha=0.2, ax=ax1)
    ax1.set_ylabel("Support (samples)")
    ax1.set_xlabel("Hour of day")
    ax1.set_title("Per-hour accuracy across splits")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Accuracy lines on twin y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=per_hour, x="hour", y="accuracy", hue="split", marker="o", ax=ax2)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Accuracy")

    # Shared legend for accuracy lines only
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncols=len(labels), frameon=False)

    return save_figure(fig, out_path)


def plot_true_class_probability(prob_df: pd.DataFrame, out_path: Path) -> Path:
    """
    Distribution of predicted probability assigned to the true class.
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    sns.histplot(
        data=prob_df,
        x="true_class_probability",
        hue="correct",
        multiple="stack",
        kde=False,
        ax=ax,
        bins=40,
        edgecolor=None,
    )
    ax.set_xlabel("Predicted probability of the true class")
    ax.set_ylabel("Count")
    ax.set_title("Model confidence for the true class by correctness")

    # Summary box
    vals = prob_df["true_class_probability"].to_numpy()
    mu = float(np.mean(vals))
    med = float(np.median(vals))
    ax.text(0.98, 0.98, f"mean={mu:.3f}\nmedian={med:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    return save_figure(fig, out_path)
