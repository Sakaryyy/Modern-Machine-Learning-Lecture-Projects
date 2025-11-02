"""Configure the scientific plotting style shared across the project.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping

import matplotlib as mpl
import seaborn as sns

from Project_2_Image_Classification.src.utils.logging import get_logger

__all__ = [
    "PlotStyleConfig",
    "PlotStyler",
    "scientific_style",
]


@dataclass(slots=True)
class PlotStyleConfig:
    """Container describing how the global plotting style should look.

    Parameters
    ----------
    context:
        Named Seaborn context which controls default scaling of figure
        elements.  Typical values are ``"paper"`` and ``"talk"``.
    style:
        Base Seaborn style string defining the background, grid and spines.
    font_scale:
        Global multiplier that scales all font sizes.
    palette:
        Name of a Seaborn colour palette or an iterable of colours that will be
        used as the default cycle.
    rc_params:
        Additional Matplotlib runtime configuration values applied on top of
        the Seaborn context/style.  These settings primarily fine tune line
        widths, grid appearance and DPI.
    """

    context: str = "paper"
    style: str = "whitegrid"
    font_scale: float = 1.0
    palette: str = "colorblind"
    rc_params: Mapping[str, object] = field(
        default_factory=lambda: {
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
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


class PlotStyler:
    """Apply a globally consistent plotting style.

    The styler exposes :meth:`apply` for long running processes and a
    context manager via :meth:`context` for local adjustments.  All operations
    are logged to aid reproducibility.
    """

    def __init__(self, config: PlotStyleConfig | None = None) -> None:
        self._config = config or PlotStyleConfig()
        self._logger = get_logger(self.__class__.__name__)

    def apply(self) -> None:
        """Activate the configured plotting style for the remainder of the process."""

        self._logger.debug(
            "Applying plotting style (context=%s, style=%s, palette=%s).",
            self._config.context,
            self._config.style,
            self._config.palette,
        )

        sns.set_theme(
            context=self._config.context,
            style=self._config.style,
            font_scale=self._config.font_scale,
            palette=self._config.palette,
        )
        mpl.rcParams.update(dict(self._config.rc_params))

    @contextmanager
    def context(self) -> Iterator[None]:
        """Temporarily apply the plotting style within the context block."""

        previous_params: Dict[str, object] = mpl.rcParams.copy()
        previous_palette = sns.color_palette()

        self._logger.debug("Entering plotting style context.")
        try:
            self.apply()
            yield
        finally:
            mpl.rcParams.update(previous_params)
            sns.set_palette(previous_palette)
            self._logger.debug("Restored previous plotting style after context exit.")


@contextmanager
def scientific_style(config: PlotStyleConfig | None = None) -> Iterator[None]:
    """Return a context manager that applies the project wide plotting style."""

    styler = PlotStyler(config)
    with styler.context():
        yield
