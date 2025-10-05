from __future__ import annotations

from pathlib import Path
from typing import Mapping
import pandas as pd
import matplotlib.figure


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def save_figure(fig: matplotlib.figure.Figure, out_path: Path, dpi: int = 150) -> Path:
    """Save a Matplotlib figure to PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to save.
    out_path : Path
        Destination path, ending with .png.
    dpi : int
        Resolution in dots-per-inch.

    Returns
    -------
    Path
        The output path.
    """
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path

def save_table_xlsx(frames: Mapping[str, pd.DataFrame], out_path: Path) -> Path:
    """Save one or more DataFrames into a single Excel workbook.

    Parameters
    ----------
    frames : Mapping[str, pd.DataFrame]
        Dict mapping sheet names to DataFrames.
    out_path : Path
        Destination path, ending with .xlsx.

    Returns
    -------
    Path
        The output path.
    """
    ensure_dir(out_path.parent)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return out_path
