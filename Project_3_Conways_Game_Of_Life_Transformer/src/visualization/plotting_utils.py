from typing import Dict, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "figure.figsize": (5.0, 3.5),
            "figure.autolayout": False,
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
    if "train_balanced_accuracy" in history and "val_balanced_accuracy" in history:
        ax.plot(epochs, history["train_balanced_accuracy"], label="train (balanced)")
        ax.plot(epochs, history["val_balanced_accuracy"], label="validation (balanced)")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("Balanced accuracy")
    else:
        ax.plot(epochs, history["train_accuracy"], label="train")
        ax.plot(epochs, history["val_accuracy"], label="validation")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()

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
        add_colorbar: bool = True,
) -> None:
    """Helper to show a single grid image."""
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    import textwrap
    wrapped_title = "\n".join(textwrap.wrap(title, width=30))
    ax.set_title(wrapped_title)
    if add_colorbar:
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

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

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
    _imshow_grid(axes[0], x, title="Input $x_t$", cmap="Greys")
    _imshow_grid(axes[1], y_true, title="Target $x_{t+1}$", cmap="Greys")
    _imshow_grid(axes[2], y_prob, title="Predicted $p(x_{t+1}=1|x_t)$", cmap="viridis")

    if title:
        fig.suptitle(title)
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
    colors[code == 0] = cmap(0)[:3]  # TN
    colors[code == 1] = cmap(1)[:3]  # FP
    colors[code == 2] = cmap(2)[:3]  # FN
    colors[code == 3] = cmap(3)[:3]  # TP

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(colors, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Prediction Error Map")

    # Custom legend
    legend_patches = [
        Patch(color=cmap(0), label="True negative"),
        Patch(color=cmap(1), label="False positive"),
        Patch(color=cmap(2), label="False negative"),
        Patch(color=cmap(3), label="True positive"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_grid_triplet_array(
        inputs: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
        title: str,
        rule_label: str,
        threshold: float = 0.5,
        max_examples: int | None = None,
        include_difference: bool = True,
        rng: np.random.Generator | None = None,
        save_path: str | None = None,
) -> None:
    """Plot multiple (input, target, prediction) triplets in a single figure.

    Parameters
    ----------
    inputs : np.ndarray
        Array of shape (N, H, W) with input states.
    targets : np.ndarray
        Array of shape (N, H, W) with corresponding target states.
    probs : np.ndarray
        Array of shape (N, H, W) with predicted probabilities.
    title : str
        Title for the full figure.
    rule_label : str
        Human-readable description of which rule generated the targets.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions when
        include_difference is True.
    max_examples : int or None, optional
        Maximum number of examples to include. If None all are shown.
    include_difference : bool, optional
        If True add columns for the binary predictions and error map.
    rng : np.random.Generator or None, optional
        Optional generator to select a random subset/order when limiting the
        number of examples.
    save_path : str or None, optional
        Optional path to save the resulting figure.
    """

    num_available = inputs.shape[0]
    num_examples = num_available if max_examples is None else min(max_examples, num_available)
    if num_examples == 0:
        return

    example_indices = np.arange(num_available) if rng is None else rng.permutation(num_available)
    example_indices = example_indices[:num_examples]

    inputs = inputs[example_indices]
    targets = targets[example_indices]
    probs = probs[example_indices]

    columns = ["Input $x_t$", f"Target $x_{{t+1}}$\n({rule_label})", "Predicted $p(x_{t+1}=1)$"]
    if include_difference:
        columns.extend(["Binary Pred", "Error Map"])

    n_cols = len(columns)
    fig, axes = plt.subplots(num_examples, n_cols, figsize=(3.0 * n_cols, 2.5 * num_examples))
    axes = np.atleast_2d(axes)

    preds = (probs > threshold).astype(int)
    for idx, example_idx in enumerate(example_indices):
        _imshow_grid(axes[idx, 0], inputs[idx], title=columns[0], cmap="Greys", add_colorbar=False)
        _imshow_grid(axes[idx, 1], targets[idx], title=f"{columns[1]} (idx {example_idx})", cmap="Greys",
                     add_colorbar=False)
        _imshow_grid(axes[idx, 2], probs[idx], title=columns[2], cmap="viridis", add_colorbar=False)

        if include_difference:
            _imshow_grid(axes[idx, 3], preds[idx], title="Binary Pred", cmap="Greys", add_colorbar=False)

            diff_true = targets[idx].astype(int)
            diff_pred = preds[idx].astype(int)
            correct = (diff_true == diff_pred)
            false_pos = (diff_pred == 1) & (diff_true == 0)
            false_neg = (diff_pred == 0) & (diff_true == 1)

            code = np.zeros_like(diff_true, dtype=int)
            code[false_pos] = 1
            code[false_neg] = 2
            code[(diff_true == 1) & correct] = 3

            cmap = plt.get_cmap("tab10")
            colors = np.zeros(code.shape + (3,), dtype=float)
            colors[code == 0] = cmap(0)[:3]
            colors[code == 1] = cmap(1)[:3]
            colors[code == 2] = cmap(2)[:3]
            colors[code == 3] = cmap(3)[:3]

            axes[idx, 4].imshow(colors, origin="lower", interpolation="nearest")
            axes[idx, 4].set_xticks([])
            axes[idx, 4].set_yticks([])
            axes[idx, 4].set_title("Error Map")

    # Build a single legend for the error codes
    if include_difference:
        legend_patches = [
            Patch(color=plt.get_cmap("tab10")(0), label="TN"),
            Patch(color=plt.get_cmap("tab10")(1), label="FP"),
            Patch(color=plt.get_cmap("tab10")(2), label="FN"),
            Patch(color=plt.get_cmap("tab10")(3), label="TP"),
        ]
        fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=8)

    fig.suptitle(f"{title}", y=1.01, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_confusion_overview(
        y_true_batch: np.ndarray,
        y_prob_batch: np.ndarray,
        rule_label: str,
        threshold: float = 0.5,
        save_path: str | None = None,
        title: str | None = None,
) -> None:
    """Summarise confusion statistics over a batch of predictions.

    The plot combines aggregated counts with spatial error frequency maps
    to highlight where the model tends to make false positives or false
    negatives.
    """

    preds = (y_prob_batch > threshold).astype(int)
    tp = int(((preds == 1) & (y_true_batch == 1)).sum())
    fp = int(((preds == 1) & (y_true_batch == 0)).sum())
    tn = int(((preds == 0) & (y_true_batch == 0)).sum())
    fn = int(((preds == 0) & (y_true_batch == 1)).sum())

    # Frequency maps across the dataset
    code_maps = []
    for true, pred in zip(y_true_batch, preds):
        correct = (true == pred)
        false_pos = (pred == 1) & (true == 0)
        false_neg = (pred == 0) & (true == 1)
        code = np.zeros_like(true, dtype=int)
        code[false_pos] = 1
        code[false_neg] = 2
        code[(true == 1) & correct] = 3
        code_maps.append(code)

    code_stack = np.stack(code_maps, axis=0)
    freq_maps = np.array([(code_stack == i).mean(axis=0) for i in range(4)])

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    # 1. Confusion Matrix
    ax = axes[0, 0]
    totals = np.array([[tp, fp], [fn, tn]])
    ax.imshow(totals, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], labels=["Pred 1", "Pred 0"])
    ax.set_yticks([0, 1], labels=["True 1", "True 0"])
    ax.set_title("Confusion Matrix")
    for (i, j), val in np.ndenumerate(totals):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black" if val < totals.max() / 2 else "white")

    # 2. Bar Chart
    ax = axes[0, 1]
    bars = ax.bar(["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["C3", "C1", "C0", "C2"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_yscale('log')
    ax.set_title("Error Counts (Log Scale)")

    # 3. Text Summary
    ax = axes[0, 2]
    total_cells = y_true_batch.size
    acc = (tp + tn) / total_cells if total_cells > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    stats_text = (
        f"Rule: {rule_label}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1 Score: {f1:.4f}"
    )
    ax.text(0.1, 0.5, stats_text, va="center", fontsize=11, family="monospace")
    ax.axis("off")
    ax.set_title("Metrics Summary")

    axes[0, 3].axis("off")

    # 4. Spatial Heatmaps
    titles = ["True Negative Rate", "False Positive Rate", "False Negative Rate", "True Positive Rate"]
    cmap_heat = "magma"
    for idx, (freq_map, subplot) in enumerate(zip(freq_maps, axes[1])):
        _imshow_grid(
            subplot,
            freq_map,
            title=titles[idx],
            cmap=cmap_heat,
            vmin=0.0,
            vmax=1.0,
            add_colorbar=True,
        )

    fig.suptitle(title or "Prediction Quality & Spatial Error Analysis", fontsize=14)
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

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(normal, bins=bins, density=True, alpha=0.5, label="Normal", color="C0")
    ax.hist(normal, bins=bins, density=True, histtype="step", linewidth=2, color="C0")

    ax.hist(anomalous, bins=bins, density=True, alpha=0.5, label="Anomalous", color="C1")
    ax.hist(anomalous, bins=bins, density=True, histtype="step", linewidth=2, color="C1")

    ax.set_xlabel("Anomaly Score (Negative Log Likelihood)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        roc_dict: Dict[str, Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, float]],
        title: str = "ROC curves for anomaly detection in stochastic GoL",
        save_path: str = None,
) -> None:
    """Plot multiple ROC curves for different settings in one figure.

    Parameters
    ----------
    roc_dict : dict
        Mapping from curve labels (str) to pairs (fpr, tpr) where both
        are one-dimensional arrays of the same length. Optionally a
        third entry (auc) can be provided to annotate the legend.
    title : str, optional
        Title of the figure.
    save_path : str or None, optional
        If not None, path where the figure will be saved as PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for label, payload in roc_dict.items():
        if len(payload) == 3:
            fpr, tpr, auc = payload  # type: ignore
            label_text = f"{label} (AUC={auc:.3f})"
        else:
            fpr, tpr = payload  # type: ignore
            label_text = label

        ax.plot(fpr, tpr, linewidth=2, label=label_text)

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

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

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly Calibrated")

    # Model calibration
    ax.plot(bin_centers[mask], empirical_freq[mask], "s-", linewidth=2, label="Model", color="C0")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Empirical Frequency (Positive Class)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Histogram of predictions
    ax_hist = ax.twinx()
    ax_hist.bar(
        bin_centers,
        bin_counts / bin_counts.sum(),
        width=1.0 / num_bins * 0.9,
        alpha=0.2,
        color="C1",
        label="Prediction Distribution"
    )
    ax_hist.set_ylabel("Fraction of Predictions")
    ax_hist.set_ylim(0.0, ax_hist.get_ylim()[1] * 3)  # flatten hist to bottom

    # Unified legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_hist.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

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
        "#1b9e77",  # Survive (0)
        "#d95f02",  # Birth (1)
        "#7570b3",  # Death (2)
        "#666666",  # Stays Dead (3)
    ])
    rule_titles = {
        RuleCategory.SURVIVES: "Survives (2-3 neighbors)",
        RuleCategory.BIRTH: "Birth (3 neighbors)",
        RuleCategory.DIES: "Dies (under/overpop)",
        RuleCategory.STAYS_DEAD: "Stays Dead",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    _imshow_grid(axes[0, 0], x, title="Input State")
    _imshow_grid(axes[0, 1], neighbors, title="Neighbor Count", cmap="magma", vmin=0, vmax=8)
    _imshow_grid(axes[0, 2], deterministic_next, title="Deterministic Next", cmap="Greys")

    _imshow_grid(axes[1, 0], y_true, title="Dataset Target", cmap="Greys")
    _imshow_grid(axes[1, 1], y_prob, title="Predicted Probability", cmap="viridis")

    axes[1, 2].imshow(rule_categories, cmap=cmap_rules, origin="lower", vmin=0, vmax=len(RuleCategory) - 1,
                      interpolation="nearest")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title("Rule Category Map")

    legend_patches = [Patch(color=cmap_rules(rule.value), label=rule_titles[rule]) for rule in RuleCategory]
    axes[1, 2].legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_by_neighbor_count(
        inputs: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
        save_path: str = None,
        title: str = "Performance by Neighbor Count"
) -> None:
    """Analyze model accuracy and error based on neighbor density (0-8).

    This is crucial for GoL to see if specific local configurations fail.
    """
    # 1. Flatten everything
    neighbors = np.array([conway_neighbor_counts(grid) for grid in inputs]).flatten()
    targets_flat = targets.flatten()
    probs_flat = probs.flatten()
    preds_flat = (probs_flat > 0.5).astype(int)

    # 2. Create DataFrame
    df = pd.DataFrame({
        "neighbors": neighbors,
        "target": targets_flat,
        "prob": probs_flat,
        "pred": preds_flat,
        "correct": (targets_flat == preds_flat).astype(int),
        "error_abs": np.abs(targets_flat - probs_flat)
    })

    # 3. Group
    stats = df.groupby("neighbors").agg({
        "correct": "mean",
        "error_abs": "mean",
        "prob": ["mean", "std"],
        "target": "count"  # count samples
    }).reset_index()
    stats.columns = ['neighbors', 'accuracy', 'mae', 'prob_mean', 'prob_std', 'count']

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy Bar Chart
    ax = axes[0]
    sns.barplot(data=stats, x="neighbors", y="accuracy", ax=ax, palette="Blues_d")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Accuracy per Neighbor Count")
    ax.set_xlabel("Living Neighbors")
    ax.set_ylabel("Accuracy")
    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.5)

    # MAE Line Chart
    ax = axes[1]
    sns.lineplot(data=stats, x="neighbors", y="mae", ax=ax, marker="o", linewidth=2, color="C3")
    ax.set_title("Mean Absolute Error (Prob Diff)")
    ax.set_xlabel("Living Neighbors")
    ax.set_ylabel("MAE (|y - p|)")
    ax.grid(True)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rule_probability_distributions(
        inputs: np.ndarray,
        probs: np.ndarray,
        save_path: str = None,
        title: str = "Probability Distribution by Logic Rule"
) -> None:
    """Violin plot showing predicted probabilities for each logic case.

    Ideally:
    - Survive/Birth should be concentrated near 1.0
    - Dies/StaysDead should be concentrated near 0.0
    """
    # 1. Compute rule categories
    rule_cats = []
    for x in inputs:
        n = conway_neighbor_counts(x)
        _, cats = compute_rule_categories(x, n)
        rule_cats.append(cats)

    rule_cats_flat = np.concatenate(rule_cats).flatten()
    probs_flat = probs.flatten()

    # Map enum to string
    cat_map = {
        RuleCategory.SURVIVES.value: "Survive\n(Target 1)",
        RuleCategory.BIRTH.value: "Birth\n(Target 1)",
        RuleCategory.DIES.value: "Dies\n(Target 0)",
        RuleCategory.STAYS_DEAD.value: "Dead\n(Target 0)"
    }
    labels = [cat_map[c] for c in rule_cats_flat]

    df = pd.DataFrame({"Rule": labels, "Probability": probs_flat})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="Rule", y="Probability", ax=ax, scale="width", cut=0, inner="box")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis='y', alpha=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_autoregressive_rollout(
        initial_state: np.ndarray,
        model_step_fn: Callable[[np.ndarray], np.ndarray],
        true_step_fn: Callable[[np.ndarray], np.ndarray],
        steps: int = 6,
        save_path: str = None,
        title: str = "Autoregressive Rollout Stability"
) -> None:
    """Simulate the game for N steps and compare model vs truth.

    This visualizes if errors accumulate (e.g. gliders disintegrating).
    """
    current_model = initial_state.copy()
    current_true = initial_state.copy()

    history_model = [current_model]
    history_true = [current_true]

    for _ in range(steps):
        # We need to reshape for the batch-based model function usually
        # Assuming model_step_fn handles single (H,W) or (1,H,W) input and returns probability
        pred_prob = model_step_fn(current_model[None, ...])[0]
        current_model = (pred_prob > 0.5).astype(np.int32)

        current_true = true_step_fn(current_true)

        history_model.append(current_model)
        history_true.append(current_true)

    # Plotting
    cols = steps + 1
    rows = 3  # Model, Truth, Diff
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 5.0))

    for t in range(cols):
        # Truth
        _imshow_grid(axes[0, t], history_true[t], title=f"True t={t}", add_colorbar=False)
        # Model
        _imshow_grid(axes[1, t], history_model[t], title=f"Model t={t}", add_colorbar=False)
        # Diff
        diff = history_true[t] - history_model[t]  # 0 match, 1 miss, -1 ghost
        axes[2, t].imshow(diff, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        axes[2, t].set_xticks([])
        axes[2, t].set_yticks([])
        axes[2, t].set_title("Diff (Blue=Miss, Red=Ghost)")

    axes[0, 0].set_ylabel("Ground Truth")
    axes[1, 0].set_ylabel("Model Prediction")
    axes[2, 0].set_ylabel("Difference")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
