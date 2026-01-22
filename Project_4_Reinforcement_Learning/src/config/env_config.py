"""Configuration for the energy budgeting environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Hyperparameters describing the environment dynamics.

    Parameters
    ----------
    max_battery_energy:
        Maximum battery capacity in energy units.
    max_solar_power:
        Maximum solar panel output per hour at full irradiation.
    base_price:
        Fixed fee added to the market buying price.
    selling_price:
        Revenue per unit of energy sold back to the grid.
    episode_length:
        Number of time steps in a single episode.
    """

    max_battery_energy: int = 10
    max_solar_power: int = 11
    base_price: float = 1.0
    selling_price: float = 1.0
    episode_length: int = 24
