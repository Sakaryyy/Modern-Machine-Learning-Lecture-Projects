from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir, save_figure

sns.set_context("talk")
sns.set_style("whitegrid")


def plot_confusion_matrix(confusion: jnp.ndarray, out_path: Path, *, title: str) -> Path:
    """Plot a confusion matrix heatmap and save it to disk."""

    ensure_dir(out_path.parent)
    data = jnp.asarray(confusion, dtype=jnp.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted hour")
    ax.set_ylabel("True hour")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_ablation_trace(trace: pd.DataFrame, out_path: Path) -> Path:
    """Plot validation accuracy over greedy ablation steps."""

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=trace, x="step", y="val_accuracy", marker="o", ax=ax, label="validation accuracy")
    sns.lineplot(
        data=trace,
        x="step",
        y="train_accuracy",
        marker="s",
        ax=ax,
        label="train accuracy",
    )
    ax.set_ylim(0, 1)
    ax.set_title("Greedy feature ablation: accuracy by step")
    ax.set_xlabel("Step (number of feature groups selected)")
    ax.set_ylabel("Accuracy")

    for _, row in trace.iterrows():
        ax.annotate(
            row["chosen_group"],
            (row["step"], row["val_accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_training_history(history: pd.DataFrame, out_path: Path) -> Path:
    """Visualise optimisation diagnostics recorded during softmax training."""

    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=history, x="iteration", y="loss", hue="phase", ax=axes[0])
    axes[0].set_title("Loss trajectory")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective value")

    sns.lineplot(data=history, x="iteration", y="accuracy", hue="phase", ax=axes[1])
    axes[1].set_title("In-sample accuracy trajectory")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=len(labels))
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.suptitle("Softmax regression optimisation diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_hour_accuracy(per_hour: pd.DataFrame, out_path: Path) -> Path:
    """Plot per-hour accuracy for different dataset splits."""

    ensure_dir(out_path.parent)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=per_hour, x="hour", y="accuracy", hue="split", marker="o", ax=ax1)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Hour of day")
    ax1.set_title("Per-hour accuracy across splits")

    ax2 = ax1.twinx()
    sns.barplot(data=per_hour, x="hour", y="support", hue="split", alpha=0.2, ax=ax2)
    ax2.set_ylabel("Support (samples)")
    max_support = max(float(per_hour["support"].max()), 1.0)
    ax2.set_ylim(0, max_support * 1.2)
    if ax2.legend_ is not None:
        ax2.legend_.remove()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(labels))
    leg1 = ax1.get_legend()
    if leg1 is not None:
        leg1.remove()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_true_class_probability(prob_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot the distribution of the predicted probability assigned to the true class."""

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=prob_df,
        x="true_class_probability",
        hue="correct",
        multiple="stack",
        kde=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted probability of the true hour")
    ax.set_ylabel("Count")
    ax.set_title("Model confidence for the true class by correctness")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path
