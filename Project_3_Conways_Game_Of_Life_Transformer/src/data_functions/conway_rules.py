"""Conway Game of Life update rules.

This module contains deterministic and stochastic transition functions for
the cellular automaton.  All functions operate on NumPy arrays and assume
periodic boundary conditions so that grids can be tiled seamlessly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import (
    get_logger,
)

LOGGER = get_logger(__name__)


def _convolve_with_periodic_neighbors(grid: np.ndarray) -> np.ndarray:
    """Compute the sum of the eight neighbours using periodic padding."""

    padded = np.pad(grid, 1, mode="wrap")
    h, w = grid.shape

    neighbor_sum = (
            padded[0:h, 0:w]
            + padded[0:h, 1: w + 1]
            + padded[0:h, 2: w + 2]
            + padded[1: h + 1, 0:w]
            + padded[1: h + 1, 2: w + 2]
            + padded[2: h + 2, 0:w]
            + padded[2: h + 2, 1: w + 1]
            + padded[2: h + 2, 2: w + 2]
    )
    return neighbor_sum


def _apply_rule(grid: np.ndarray, neighbor_sum: np.ndarray, survive: str, born: str) -> np.ndarray:
    """Apply a generic Life-like rule."""

    survive_counts = {int(c) for c in survive}
    born_counts = {int(c) for c in born}

    alive = grid == 1
    birth = np.isin(neighbor_sum, list(born_counts)) & (~alive)
    stay_alive = np.isin(neighbor_sum, list(survive_counts)) & alive

    next_grid = np.where(birth | stay_alive, 1, 0).astype(np.int32)
    return next_grid


def conway_step_periodic(grid: np.ndarray) -> np.ndarray:
    """Deterministic Conway 23/3 rule with periodic boundary conditions."""

    neighbor_sum = _convolve_with_periodic_neighbors(grid)
    return _apply_rule(grid, neighbor_sum, survive="23", born="3")


def stochastic_step_mixed_rules(grid: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Stochastic update mixing Conway 23/3 with rule 35/3.

    Each cell independently draws a Bernoulli(p) variable to decide
    whether to apply the standard 23/3 rule or the alternative 35/3 rule
    (survive with 3 or 5 neighbours, born with 3).
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    neighbor_sum = _convolve_with_periodic_neighbors(grid)

    # Draw mask once to keep deterministic for a given rng state
    use_standard = rng.random(size=grid.shape) < p

    next_standard = _apply_rule(grid, neighbor_sum, survive="23", born="3")
    next_alt = _apply_rule(grid, neighbor_sum, survive="35", born="3")

    next_grid = np.where(use_standard, next_standard, next_alt)
    return next_grid.astype(np.int32)


RULES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "23/3": conway_step_periodic,
}
