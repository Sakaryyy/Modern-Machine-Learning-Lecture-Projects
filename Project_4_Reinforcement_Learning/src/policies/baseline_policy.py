"""Baseline policy for the energy budgeting environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Project_4_Reinforcement_Learning.src.config import EnvironmentConfig


@dataclass(slots=True)
class BaselinePolicy:
    """Rule-based baseline for the energy management task.

    The policy prioritizes using solar energy for immediate demand, storing
    remaining solar power, and charging from the grid only when the market
    price is low.

    Parameters
    ----------
    config:
        Environment configuration used to align with capacity constraints.
    cheap_price_threshold:
        Buying price below which grid charging is considered attractive.
    """

    config: EnvironmentConfig
    cheap_price_threshold: float = 1.5

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Select an action based on the current observation.

        Parameters
        ----------
        observation:
            Array containing ``[time_of_day, buying_price, battery_energy]``.

        Returns
        -------
        numpy.ndarray
            Action array ``[solar_to_demand, solar_to_battery, battery_to_demand, grid_to_battery]``.
        """

        time_of_day = int(observation[0])
        buying_price = float(observation[1])
        battery_energy = int(observation[2])

        expected_demand = self._expected_demand(time_of_day)

        solar_to_demand = expected_demand
        solar_to_battery = self.config.max_solar_power

        battery_to_demand = min(battery_energy, expected_demand)

        if buying_price < self.cheap_price_threshold:
            grid_to_battery = self.config.max_battery_energy - battery_energy
        else:
            grid_to_battery = 0

        return np.array(
            [solar_to_demand, solar_to_battery, battery_to_demand, grid_to_battery],
            dtype=np.int64,
        )

    def _expected_demand(self, time_of_day: int) -> int:
        """Estimate expected demand based on the deterministic component.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.

        Returns
        -------
        int
            Expected demand rounded to the nearest integer.
        """

        demand_shape = np.sin(2 * np.pi * time_of_day / 24) ** 2
        expected = 2 * self.config.max_battery_energy / np.pi * demand_shape
        return int(round(expected))
