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
            Array containing ``[time_of_day, buying_price, battery_energy, battery_capacity,
            battery_health, forecast_solar, forecast_price, forecast_demand]``.

        Returns
        -------
        numpy.ndarray
            Action array ``[solar_to_demand, solar_to_battery, battery_to_demand, battery_to_grid,
            grid_to_battery]``.
        """

        time_of_day = int(observation[0])
        buying_price = float(observation[1])
        battery_energy = float(observation[2])
        battery_capacity = float(observation[3])
        battery_health = float(observation[4])

        p_max = int(self.config.max_solar_power)
        e_max = float(self.config.max_battery_energy)
        market_price = buying_price - self.config.base_price - self.config.trade_fee_per_unit
        selling_price = float(
            max(
                0.0,
                market_price * self.config.selling_price_multiplier + self.config.selling_price_offset
                - self.config.trade_fee_per_unit,
            )
        )
        forecast_horizon = int(self.config.forecast_horizon)
        forecast_start = 5
        forecast_end = forecast_start + forecast_horizon
        forecast_prices = observation[forecast_start + forecast_horizon: forecast_start + 2 * forecast_horizon]
        forecast_demand = observation[forecast_start + 2 * forecast_horizon: forecast_start + 3 * forecast_horizon]
        max_forecast_price = float(np.max(forecast_prices)) if forecast_prices.size else buying_price
        avg_forecast_demand = float(np.mean(forecast_demand)) if forecast_demand.size else 0.0

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

        # Case 1: Forecasted arbitrage window encourages charging the battery aggressively.
        if self.enable_solar_grid_arbitrage and buying_price < selling_price:
            solar_to_demand = 0
            solar_to_battery = 0
            battery_to_demand = 0
            battery_to_grid = 0
            grid_to_battery = int(round(e_max - battery_energy))

            return np.array(
                [solar_to_demand, solar_to_battery, battery_to_demand, battery_to_grid, grid_to_battery],
                dtype=np.int64,
            )

        # Default solar behaviour: attempt to cover demand first, then store leftover.
        # The environment clamps these attempts to the actually available solar and capacity.
        solar_to_demand = p_max
        solar_to_battery = p_max

        # Battery discharge: only when the current buying price is high or forecast spikes soon.
        should_discharge = buying_price >= discharge_th or max_forecast_price >= discharge_th
        discharge_cap = battery_capacity * max(self.config.battery_degradation_threshold, 0.1)
        battery_to_demand = int(round(battery_energy)) if should_discharge and battery_energy > discharge_cap else 0

        # Battery-to-grid selling: use when prices are exceptionally high and health is strong.
        battery_to_grid = 0
        if max_forecast_price >= discharge_th * 1.1 and battery_health > 0.75:
            buffer = max(0.0, avg_forecast_demand - battery_energy)
            battery_to_grid = int(max(0.0, battery_energy - buffer))

        # Grid charging: only when buying price is low (more permissive at night).
        grid_to_battery = int(round(e_max - battery_energy)) if buying_price <= charge_th else 0

        return np.array(
            [solar_to_demand, solar_to_battery, battery_to_demand, battery_to_grid, grid_to_battery],
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
