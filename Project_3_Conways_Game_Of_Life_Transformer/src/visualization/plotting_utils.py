from typing import Dict, Sequence, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Project_3_Conways_Game_Of_Life_Transformer.src.training.metrics import calibration_curve
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.rule_analysis import (
    RuleCategory,
    compute_rule_categories,
    conway_neighbor_counts,
)


def set_scientific_plot_style() -> None:
    """Configure a global scientific plotting style.

    The style is inspired by seaborn's white theme with a paper
    context. It adjusts font sizes, line widths, and figure DPI to
    produce publication grade figures.
    """
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "figure.figsize": (5.0, 3.5),
        }
    )


# ---------------------------------------------------------------------
# Training history plots
# ---------------------------------------------------------------------

def plot_training_curves(
        history: Dict[str, Sequence[float]],
        title: str = "Training and validation curves",
        save_path: str = None,
) -> None:
    """Plot loss and accuracy curves over epochs.

    Parameters
    ----------
    history : dict
        Dictionary mapping metric names to sequences of per epoch
        values. Expected keys include "train_loss", "val_loss",
        "train_accuracy", and "val_accuracy".
    title : str, optional
        Figure title.
    save_path : str or None, optional
        Optional file path to save the figure as PNG.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    has_lr = "learning_rate" in history
    has_gap = "train_val_gap" in history
    n_cols = 2 + int(has_lr or has_gap)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross entropy")
    ax.set_title("Loss")
    ax.legend()

    if n_cols > 2:
        ax = axes[2]
        if has_lr:
            ax.plot(epochs, history["learning_rate"], label="learning rate")
        if has_gap:
            ax.plot(epochs, history["train_val_gap"], label="train - val acc")
        ax.set_xlabel("Epoch")
        ax.set_title("Optimisation diagnostics")
        ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_accuracy"], label="train")
    ax.plot(epochs, history["val_accuracy"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------
# Grid and probability visualizations
# ---------------------------------------------------------------------

def _imshow_grid(
        ax: plt.Axes,
        grid: np.ndarray,
        title: str,
        cmap: str = "Greys",
        vmin: float = 0.0,
        vmax: float = 1.0,
) -> None:
    """Helper to show a single grid image."""
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_grid_pair_examples(
        inputs: np.ndarray,
        targets: np.ndarray,
        title: str = "Example transitions",
        save_path: str | None = None,
) -> None:
    """Plot a vertical grid of (input, target) pairs from the dataset."""

    num_examples = int(min(inputs.shape[0], targets.shape[0]))
    if num_examples == 0:
        return

    fig, axes = plt.subplots(num_examples, 2, figsize=(6, 2.5 * num_examples))
    axes = np.atleast_2d(axes)

    for idx in range(num_examples):
        _imshow_grid(axes[idx, 0], inputs[idx], title=f"Input t (sample {idx})")
        _imshow_grid(axes[idx, 1], targets[idx], title=f"Target t+1 (sample {idx})")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_grid_triplet(
        x: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        figsize: Tuple[float, float] = (9.0, 3.0),
        save_path: str = None,
        title: str = None,
) -> None:
    """Visualize input, target, and predicted probabilities.

    Parameters
    ----------
    x : np.ndarray
        Input grid of shape (H, W) with entries in {0, 1}.
    y_true : np.ndarray
        Target grid of shape (H, W) with entries in {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities for the next state of shape (H, W) with
        values in [0, 1].
    figsize : tuple of float, optional
        Figure size in inches.
    save_path : str or None, optional
        Optional path to save the figure as PNG.
    title : str, optional
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _imshow_grid(axes[0], x, title="Input $x_t$", cmap="Greys", vmin=0.0, vmax=1.0)
    _imshow_grid(axes[1], y_true, title="Target $x_{t+1}$", cmap="Greys", vmin=0.0, vmax=1.0)
    _imshow_grid(axes[2], y_prob, title="Predicted $p(x_{t+1}=1|x_t)$", cmap="viridis", vmin=0.0, vmax=1.0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_grid_difference(
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
        figsize: Tuple[float, float] = (4.0, 4.0),
        save_path: str = None,
) -> None:
    """Visualize where the model predictions differ from the target.

    Cells are colored according to the type of error. This is useful
    when debugging failure modes.

    Parameters
    ----------
    y_true : np.ndarray
        Target grid of shape (H, W) with entries in {0, 1}.
    y_pred_binary : np.ndarray
        Binary predictions of shape (H, W) with entries in {0, 1}.
    figsize : tuple of float, optional
        Figure size in inches.
    save_path : str or None, optional
        Optional path to save the figure.
    """
    # Encode four states into a single integer value
    # 0 correct dead, 1 false positive, 2 false negative, 3 correct alive
    true = y_true.astype(int)
    pred = y_pred_binary.astype(int)

    correct = (true == pred)
    false_pos = (pred == 1) & (true == 0)
    false_neg = (pred == 0) & (true == 1)

    code = np.zeros_like(true, dtype=int)
    code[false_pos] = 1
    code[false_neg] = 2
    code[(true == 1) & correct] = 3

    cmap = plt.get_cmap("tab10")
    colors = np.zeros(code.shape + (3,), dtype=float)
    colors[code == 0] = cmap(0)[:3]
    colors[code == 1] = cmap(1)[:3]
    colors[code == 2] = cmap(2)[:3]
    colors[code == 3] = cmap(3)[:3]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(colors, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Prediction vs target")

    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=cmap(0), label="True negative"),
        Patch(color=cmap(1), label="False positive"),
        Patch(color=cmap(2), label="False negative"),
        Patch(color=cmap(3), label="True positive"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_score_histograms(
        scores: np.ndarray,
        labels: np.ndarray,
        bins: int = 50,
        save_path: str = None,
) -> None:
    """Plot histograms of anomaly scores for normal and anomalous samples.

    Parameters
    ----------
    scores : np.ndarray
        One dimensional array of anomaly scores. Larger values should
        indicate more anomalous samples, for example negative log
        likelihoods.
    labels : np.ndarray
        One dimensional integer array with values 0 for normal samples
        and 1 for anomalies.
    bins : int, optional
        Number of histogram bins.
    save_path : str or None, optional
        Optional file path to save the figure as PNG.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    normal = scores[labels == 0]
    anomalous = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.hist(
        normal,
        bins=bins,
        density=True,
        histtype="step",
        label="normal",
        linewidth=1.5,
    )
    ax.hist(
        anomalous,
        bins=bins,
        density=True,
        histtype="step",
        label="anomalous",
        linewidth=1.5,
    )
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        label: str,
        ax: plt.Axes,
) -> None:
    """Plot a single ROC curve on the given axes.

    Parameters
    ----------
    fpr : np.ndarray
        Array of false positive rates in [0, 1].
    tpr : np.ndarray
        Array of true positive rates in [0, 1].
    label : str
        Label for the curve, for example including the lattice size.
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which to draw the curve.
    """
    ax.plot(fpr, tpr, marker="", linestyle="-", label=label)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)


def plot_multiple_roc_curves(
        roc_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "ROC curves for anomaly detection in stochastic GoL",
        save_path: str = None,
) -> None:
    """Plot multiple ROC curves for different settings in one figure.

    Parameters
    ----------
    roc_dict : dict
        Mapping from curve labels (str) to pairs (fpr, tpr) where both
        are one-dimensional arrays of the same length.
    title : str, optional
        Title of the figure.
    save_path : str or None, optional
        If not None, path where the figure will be saved as PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for label, (fpr, tpr) in roc_dict.items():
        plot_roc_curve(fpr=fpr, tpr=tpr, label=label, ax=ax)

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="Random")

    ax.set_title(title)
    ax.legend(loc="lower right")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_calibration_curve(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        num_bins: int = 10,
        save_path: str = None,
) -> None:
    """Plot a calibration curve for probabilistic predictions.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    y_true : np.ndarray
        Binary targets with entries in {0, 1}.
    num_bins : int, optional
        Number of bins between 0 and 1.
    save_path : str or None, optional
        Optional file path to save the figure.
    """
    bin_centers, empirical_freq, bin_counts = calibration_curve(
        y_prob=y_prob,
        y_true=y_true,
        num_bins=num_bins,
    )

    mask = ~np.isnan(empirical_freq)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.plot(bin_centers[mask], empirical_freq[mask], marker="o", label="model")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Visualise how many cells populate each bin
    ax_hist = ax.twinx()
    ax_hist.bar(
        bin_centers,
        bin_counts / np.maximum(bin_counts.sum(), 1),
        width=1.0 / num_bins * 0.9,
        alpha=0.2,
        color="C1",
        label="bin mass",
    )
    ax_hist.set_ylabel("Fraction of cells per bin")
    ax_hist.set_ylim(0.0, ax_hist.get_ylim()[1])

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_rule_diagnostics(
        x: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str,
        save_path: str | None = None,
) -> None:
    """Visualize how Conway's rules explain predictions.

    The plot shows the input board, neighbor counts, deterministic rule
    outcome, model probability heatmap, binary prediction, and a rule
    category map that highlights which rule applies at each cell.
    """

    neighbors = conway_neighbor_counts(x)
    deterministic_next, rule_categories = compute_rule_categories(x, neighbors)

    cmap_rules = ListedColormap([
        "#1b9e77",  # survival
        "#d95f02",  # birth
        "#7570b3",  # death
        "#666666",  # stays dead
    ])
    rule_titles = {
        RuleCategory.SURVIVES: "Survival (alive with 2/3 neighbors)",
        RuleCategory.BIRTH: "Birth (dead with 3 neighbors)",
        RuleCategory.DIES: "Death (alive with <2 or >3 neighbors)",
        RuleCategory.STAYS_DEAD: "Stays dead",
    }

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    _imshow_grid(axes[0, 0], x, title="Input $x_t$")
    _imshow_grid(axes[0, 1], neighbors, title="Neighbor counts", cmap="magma", vmin=0, vmax=8)
    _imshow_grid(axes[0, 2], deterministic_next, title="Deterministic rule output", cmap="Greys", vmin=0, vmax=1)
    _imshow_grid(axes[1, 0], y_true, title="Dataset target", cmap="Greys", vmin=0, vmax=1)
    _imshow_grid(axes[1, 1], y_prob, title="Model $p(x_{t+1}=1)$", cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].imshow(rule_categories, cmap=cmap_rules, origin="lower", vmin=0, vmax=len(RuleCategory) - 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title("Rule category map")

    # Custom legend for rule categories
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=cmap_rules(rule.value), label=rule_titles[rule]) for rule in RuleCategory]
    axes[1, 2].legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
