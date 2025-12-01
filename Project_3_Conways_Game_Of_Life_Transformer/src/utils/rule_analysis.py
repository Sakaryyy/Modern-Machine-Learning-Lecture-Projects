"""Utilities for analysing Conway's Game of Life rule application."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np


class RuleCategory(Enum):
    """Enumeration of Conway rule outcomes."""

    SURVIVES = 0
    BIRTH = 1
    DIES = 2
    STAYS_DEAD = 3


@dataclass
class RuleMetrics:
    """Summary of model performance per Conway rule."""

    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total

    def add_counts(self, total: int, correct: int) -> None:
        self.total += total
        self.correct += correct


def conway_neighbor_counts(grid: np.ndarray) -> np.ndarray:
    """Compute neighbor counts for a 2D Conway grid using periodic padding."""

    shifts = [-1, 0, 1]
    neighbors = np.zeros_like(grid, dtype=int)
    for dx in shifts:
        for dy in shifts:
            if dx == 0 and dy == 0:
                continue
            neighbors += np.roll(np.roll(grid, dx, axis=-2), dy, axis=-1)
    return neighbors


def compute_rule_categories(
        x: np.ndarray, neighbors: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic next state and rule category per cell.

    Parameters
    ----------
    x : np.ndarray
        Input grid (H, W) of 0/1 values.
    neighbors : np.ndarray or None
        Optional pre-computed neighbor counts.
    """

    if neighbors is None:
        neighbors = conway_neighbor_counts(x)

    survives = (x == 1) & ((neighbors == 2) | (neighbors == 3))
    birth = (x == 0) & (neighbors == 3)
    dies = (x == 1) & ~survives

    deterministic_next = np.where(x == 1, survives.astype(int), birth.astype(int))
    categories = np.full_like(x, fill_value=RuleCategory.STAYS_DEAD.value)
    categories[survives] = RuleCategory.SURVIVES.value
    categories[birth] = RuleCategory.BIRTH.value
    categories[dies] = RuleCategory.DIES.value

    return deterministic_next, categories


def summarise_rule_accuracy(
        x: np.ndarray,
        predictions_binary: np.ndarray,
) -> Dict[RuleCategory, RuleMetrics]:
    """Compute per-rule accuracies comparing predictions to deterministic rules."""

    neighbors = conway_neighbor_counts(x)
    deterministic_next, categories = compute_rule_categories(x, neighbors)

    summary: Dict[RuleCategory, RuleMetrics] = {}
    for rule in RuleCategory:
        mask = categories == rule.value
        total = int(mask.sum())
        correct = int((predictions_binary[mask] == deterministic_next[mask]).sum()) if total > 0 else 0
        summary[rule] = RuleMetrics(total=total, correct=correct)
    return summary
