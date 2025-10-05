from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir, save_figure

sns.set_context("talk")
sns.set_style("whitegrid")


def plot_confusion_matrix(confusion: jnp.ndarray, out_path: Path) -> Path:
    """Plot a confusion matrix heatmap and save it to disk."""

    ensure_dir(out_path.parent)
    data = jnp.asarray(confusion, dtype=jnp.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix: predicted hour vs. true hour")
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
