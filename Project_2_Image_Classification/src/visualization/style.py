"""Utilities defining a consistent visual style for all project figures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import matplotlib as mpl
import seaborn as sns

from ..utils.logging import get_logger

__all__ = ["PlotStyleConfig", "PlotStyleManager"]


@dataclass(frozen=True, slots=True)
class PlotStyleConfig:
    """Configuration container describing how plots should look.

    Parameters
    ----------
    context:
        Pre-defined seaborn plotting context controlling base font sizes.
    style:
        Base seaborn style (``whitegrid`` by default).
    palette:
        Name of the seaborn colour palette to use for categorical variables.
    font:
        Name of the font family used across all figures.
    figure_size:
        Default size for generated matplotlib figures in inches.
    dpi:
        Resolution for on-screen rendering of figures.
    save_dpi:
        Resolution for persisted figures when saved to disk.
    grid:
        If ``True`` all axes render grid lines to aid interpretation.
    grid_alpha:
        Transparency of the grid lines.
    grid_linestyle:
        Line style used for grid rendering.
    use_tex:
        Toggle LaTeX rendering of text elements.
    title_size:
        Font size used for axis titles.
    label_size:
        Font size used for axis labels.
    tick_size:
        Font size used for tick labels.
    legend_size:
        Font size used for legend entries.
    facecolor:
        Background colour of generated figures.
    extra_rcparams:
        Optional dictionary of additional ``matplotlib`` rc parameters applied on
        top of the defaults.
    """

    context: str = "talk"
    style: str = "whitegrid"
    palette: str = "deep"
    font: str = "DejaVu Sans"
    figure_size: tuple[float, float] = (10.0, 6.0)
    dpi: int = 150
    save_dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.25
    grid_linestyle: str = "--"
    use_tex: bool = False
    title_size: int = 18
    label_size: int = 14
    tick_size: int = 12
    legend_size: int = 12
    facecolor: str = "white"
    extra_rcparams: Mapping[str, object] = field(default_factory=dict)


class PlotStyleManager:
    """Apply and optionally temporarily override the global plotting style."""

    def __init__(self, config: PlotStyleConfig | None = None) -> None:
        self._config = config or PlotStyleConfig()
        self._logger = get_logger(self.__class__.__name__)

    @property
    def config(self) -> PlotStyleConfig:
        """Return the active plot style configuration."""

        return self._config

    def apply(self) -> None:
        """Apply the configured plotting style globally.

        The method configures seaborn and matplotlib to ensure a consistent look
        across all visualisations.
        """

        sns.set_theme(
            context=self._config.context,
            style=self._config.style,
            palette=self._config.palette,
            font=self._config.font,
        )

        rc_params: dict[str, object] = {
            "figure.figsize": self._config.figure_size,
            "figure.dpi": self._config.dpi,
            "savefig.dpi": self._config.save_dpi,
            "axes.titlesize": self._config.title_size,
            "axes.labelsize": self._config.label_size,
            "xtick.labelsize": self._config.tick_size,
            "ytick.labelsize": self._config.tick_size,
            "legend.fontsize": self._config.legend_size,
            "axes.facecolor": self._config.facecolor,
            "figure.facecolor": self._config.facecolor,
            "axes.grid": self._config.grid,
        }

        if self._config.grid:
            rc_params["grid.alpha"] = self._config.grid_alpha
            rc_params["grid.linestyle"] = self._config.grid_linestyle
        if self._config.use_tex:
            rc_params["text.usetex"] = True

        rc_params.update(self._config.extra_rcparams)
        mpl.rcParams.update(rc_params)

        self._logger.debug("Applied plot style configuration: %s", self._config)

    @contextmanager
    def context(self) -> Iterator[None]:
        """Context manager applying the configured style temporarily."""

        original_rc = mpl.rcParams.copy()
        original_style = sns.axes_style()
        original_context = sns.plotting_context()
        original_palette = sns.color_palette()

        try:
            self.apply()
            yield
        finally:
            mpl.rcParams.update(original_rc)
            sns.set_style(original_style)
            sns.set_context(original_context)
            sns.set_palette(original_palette)
            self._logger.debug("Restored previous plotting configuration.")
