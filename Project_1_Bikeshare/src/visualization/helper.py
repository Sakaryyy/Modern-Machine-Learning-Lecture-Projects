from contextlib import contextmanager
from typing import Iterable, Optional

import matplotlib as mpl
import numpy as np
import seaborn as sns

# -----------------------------------------------------------------------------
# Global style helpers
# -----------------------------------------------------------------------------

_DEFAULT_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 10,
    "ps.fonttype": 10,
}

_COLOR_CYCLE = sns.color_palette("colorblind")


def mm_to_in(millimetres: float) -> float:
    return millimetres / 25.4


@contextmanager
def paper_theme():
    old = mpl.rcParams.copy()
    try:
        mpl.rcParams.update(_DEFAULT_CONTEXT)
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
        yield
    finally:
        mpl.rcParams.update(old)


SINGLE_COL_W_MM = 85
DOUBLE_COL_W_MM = 175


def fig_size(
        *,
        width_mm: float = SINGLE_COL_W_MM,
        height_mm: Optional[float] = None,
        aspect: Optional[float] = None,
) -> tuple[float, float]:
    """Compute figure size (inches) for a given width and aspect/height.

    If `aspect` is provided, height = width / aspect. If neither height nor aspect
    provided, defaults to golden‑ratio height.
    """
    w_in = mm_to_in(width_mm)
    if height_mm is not None:
        h_in = mm_to_in(height_mm)
    else:
        if aspect is None:
            aspect = 1.618
        h_in = w_in / aspect
    return (w_in, h_in)


def _finalise(ax: mpl.axes.Axes, *, square: bool = False) -> None:
    """Apply final small touches common to most plots."""
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)
    ax.minorticks_on()
    if square:
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass


def _fd_bins(x: np.ndarray) -> int:
    """Freedman–Diaconis optimal bin count for histograms."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return int(np.sqrt(x.size))
    h = 2 * iqr * (x.size ** (-1 / 3))
    if h <= 0:
        return int(np.sqrt(x.size))
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(bins, 10)


def _stats_box(ax: mpl.axes.Axes, lines: Iterable[str], loc: str = "upper left") -> None:
    text = "\n".join(lines)
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
    ax.text(0.02 if "left" in loc else 0.98,
            0.98 if "upper" in loc else 0.02,
            text,
            transform=ax.transAxes,
            ha="left" if "left" in loc else "right",
            va="top" if "upper" in loc else "bottom",
            fontsize=8,
            bbox=bbox)
