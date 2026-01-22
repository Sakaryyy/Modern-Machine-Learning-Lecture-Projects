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
    charge_price_threshold_day: float = 0.85
    charge_price_threshold_night: float = 1.15
    discharge_price_threshold: float = 2.40
    enable_solar_grid_arbitrage: bool = True
    cheap_price_threshold: float | None = None

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

        p_max = int(self.config.max_solar_power)
        e_max = int(self.config.max_battery_energy)
        selling_price = float(getattr(self.config, "selling_price", 1.0))

        # Optional legacy override
        charge_day = (
            float(self.cheap_price_threshold)
            if self.cheap_price_threshold is not None
            else float(self.charge_price_threshold_day)
        )
        charge_night = float(self.charge_price_threshold_night)
        discharge_th = float(self.discharge_price_threshold)

        is_night = time_of_day <= 5 or time_of_day >= 19
        charge_th = charge_night if is_night else charge_day

        # Case 1: Arbitrage regime (model allows selling solar at R^S while buying at R^B_t).
        if self.enable_solar_grid_arbitrage and buying_price < selling_price:
            solar_to_demand = 0
            solar_to_battery = 0
            battery_to_demand = 0
            grid_to_battery = e_max - battery_energy

            return np.array(
                [solar_to_demand, solar_to_battery, battery_to_demand, grid_to_battery],
                dtype=np.int64,
            )

        # Default solar behaviour: attempt to cover demand first, then store leftover.
        # The environment clamps these attempts to the actually available solar and capacity.
        solar_to_demand = p_max
        solar_to_battery = p_max

        # Battery discharge: only when the current buying price is high.
        battery_to_demand = battery_energy if buying_price >= discharge_th else 0

        # Grid charging: only when buying price is low (more permissive at night).
        grid_to_battery = (e_max - battery_energy) if buying_price <= charge_th else 0

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
