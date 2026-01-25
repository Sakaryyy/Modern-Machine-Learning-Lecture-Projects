"""Gymnasium environment for energy budgeting with solar panels and a battery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from Project_4_Reinforcement_Learning.src.config import EnvironmentConfig
from Project_4_Reinforcement_Learning.src.logger import get_logger

__all__ = [
    "StepMetrics",
    "EnergyBudgetEnv",
]


@dataclass(slots=True, frozen=True)
class StepMetrics:
    """Container for step-level metrics.

    Parameters
    ----------
    time_step:
        Absolute time step counter ``t``.
    time_of_day:
        Hour of day in ``[0, 23]``.
    day_of_year:
        Day of year index in ``[0, year_length_days - 1]``.
    solar_intensity:
        Solar irradiation multiplier ``S_t``.
    solar_production:
        Discrete solar production ``P^S_t``.
    demand:
        Household demand ``D_t``.
    market_price:
        Market price component ``R^M_t``.
    buying_price:
        Total buying price ``R^B_t``.
    solar_to_demand:
        Solar units allocated to demand.
    selling_price:
        Selling price per unit of energy.
    solar_to_battery:
        Solar units allocated to charging the battery.
    battery_to_demand:
        Battery units discharged to cover demand.
    battery_to_grid:
        Battery units discharged to sell to the grid.
    grid_to_battery:
        Grid units purchased to charge the battery.
    grid_to_demand:
        Grid units purchased to cover remaining demand.
    solar_sold:
        Solar units sold to the grid.
    balance:
        Monetary balance ``B_t`` after trading and degradation penalties.
    battery_energy:
        Battery energy after the transition.
    battery_capacity:
        Current effective battery capacity after degradation.
    battery_health:
        Battery health fraction in ``[0, 1]``.
    battery_throughput:
        Total energy throughput processed by the battery this step.
    degradation_event:
        Whether a degradation event occurred.
    degradation_amount:
        Capacity fraction lost due to degradation in this step.
    self_discharge_loss:
        Energy lost due to self discharge.
    seasonal_solar_factor:
        Seasonal multiplier applied to solar intensity.
    seasonal_demand_factor:
        Seasonal multiplier applied to demand.
    seasonal_price_factor:
        Seasonal multiplier applied to market price.
    weather_factor:
        Daily weather multiplier for solar intensity.
    daylight_factor:
        Daylight duration multiplier for solar intensity.
    forecast_solar_intensity:
        Forecasted solar intensity for the next horizon.
    forecast_market_price:
        Forecasted market price for the next horizon.
    forecast_demand:
        Forecasted demand for the next horizon.
    """

    time_step: int
    time_of_day: int
    day_of_year: int
    solar_intensity: float
    solar_production: int
    demand: int
    market_price: float
    buying_price: float
    selling_price: float
    solar_to_demand: int
    solar_to_battery: int
    battery_to_demand: int
    battery_to_grid: int
    grid_to_battery: int
    grid_to_demand: int
    solar_sold: int
    balance: float
    battery_energy: float
    battery_capacity: float
    battery_health: float
    battery_throughput: float
    degradation_event: bool
    degradation_amount: float
    self_discharge_loss: float
    seasonal_solar_factor: float
    seasonal_demand_factor: float
    seasonal_price_factor: float
    weather_factor: float
    daylight_factor: float
    forecast_solar_intensity: Tuple[float, ...]
    forecast_market_price: Tuple[float, ...]
    forecast_demand: Tuple[float, ...]


class EnergyBudgetEnv(gym.Env):
    """Energy budgeting environment based on the project specification.

    The agent manages how solar production, battery energy, and grid purchases
    are used to meet household demand while maximizing monetary balance. The
    refined dynamics include battery degradation, trading at market-linked
    prices, and noisy forecasts for solar, price, and demand signals.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self._rng: np.random.Generator = np.random.default_rng()
        self._logger = get_logger(self.__class__.__name__)

        self.action_space = gym.spaces.MultiDiscrete(
            [
                self.config.max_solar_power + 1,
                self.config.max_solar_power + 1,
                self.config.max_battery_energy + 1,
                self.config.max_battery_energy + 1,
                self.config.max_battery_energy + 1,
            ]
        )

        self._buying_price_clip: float = float(self.config.buying_price_clip)
        self._forecast_horizon: int = int(self.config.forecast_horizon)

        observation_low = [0.0, 0.0, -self._buying_price_clip, 0.0, 0.0, 0.0]
        observation_high = [
            23.0,
            float(self.config.year_length_days - 1),
            self._buying_price_clip,
            float(self.config.max_battery_energy),
            float(self.config.max_battery_energy),
            1.0,
        ]
        forecast_low = [0.0] * self._forecast_horizon + [-self._buying_price_clip] * self._forecast_horizon
        forecast_low += [0.0] * self._forecast_horizon
        forecast_high = [1.0] * self._forecast_horizon + [self._buying_price_clip] * self._forecast_horizon
        forecast_high += [float(self.config.max_demand)] * self._forecast_horizon

        self.observation_space = gym.spaces.Box(
            low=np.array(observation_low + forecast_low, dtype=np.float32),
            high=np.array(observation_high + forecast_high, dtype=np.float32),
            dtype=np.float32,
        )

        self._time_step = 0
        self._absolute_time_step = 0
        self._battery_energy = float(self.config.max_battery_energy * self.config.initial_soc_fraction)
        self._battery_health = 1.0
        self._battery_capacity = float(self.config.max_battery_energy)
        self._current_solar_intensity = 0.0
        self._current_market_price = 0.0
        self._current_demand = 0
        self._forecast_solar: Tuple[float, ...] = tuple()
        self._forecast_price: Tuple[float, ...] = tuple()
        self._forecast_demand: Tuple[float, ...] = tuple()
        self._day_offset = 0
        self._daily_weather_factor = 1.0
        self._daily_daylight_factor = 1.0
        self._has_reset = False
        self._last_day_of_year: int | None = None

    def seed(self, seed: int | None = None) -> None:
        """Seed the random number generator.

        Parameters
        ----------
        seed:
            Seed value for reproducibility.
        """

        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment state.

        Parameters
        ----------
        seed:
            Optional seed for the random number generator.
        options:
            Additional reset options (unused).

        Returns
        -------
        numpy.ndarray
            Initial observation of the environment.
        dict
            Diagnostic information.
        """

        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._time_step = 0
        if not self._has_reset or not self.config.persist_battery_state:
            self._absolute_time_step = 0
            if self.config.randomize_start_day:
                self._day_offset = int(self._rng.integers(0, self.config.year_length_days))
            else:
                self._day_offset = 0
            self._battery_health = 1.0
            self._battery_capacity = float(self.config.max_battery_energy)
            self._battery_energy = float(self.config.max_battery_energy * self.config.initial_soc_fraction)
        self._maybe_update_daily_factors(self._current_day_of_year())
        self._sample_exogenous(time_of_day=self._current_time_of_day(), absolute_time=self._absolute_time_step)
        self._has_reset = True

        observation = self._get_observation()
        info = {"battery_energy": self._battery_energy, "battery_capacity": self._battery_capacity}
        self._logger.debug("Environment reset: %s", info)
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action:
            Action array consisting of
            ``[solar_to_demand, solar_to_battery, battery_to_demand, battery_to_grid, grid_to_battery]``.

        Returns
        -------
        numpy.ndarray
            Next observation.
        float
            Reward (monetary balance).
        bool
            Whether the episode has terminated.
        bool
            Whether the episode was truncated.
        dict
            Additional diagnostic information.
        """

        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.shape[0] != 5:
            raise ValueError(f"Expected action of shape (5,), got {action_arr.shape}")

        solar_to_demand, solar_to_battery, battery_to_demand, battery_to_grid, grid_to_battery = (
            int(v) for v in action_arr
        )

        time_of_day = self._current_time_of_day()
        day_of_year = self._current_day_of_year()

        solar_intensity = float(self._current_solar_intensity)
        solar_production = int(np.floor(self.config.max_solar_power * solar_intensity))

        market_price = float(self._current_market_price)
        buying_price = float(self.config.base_price + market_price + self.config.trade_fee_per_unit)
        selling_price = float(
            max(
                0.0,
                market_price * self.config.selling_price_multiplier + self.config.selling_price_offset
                - self.config.trade_fee_per_unit,
            )
        )

        demand = int(self._current_demand)

        self_discharge_loss = self._battery_energy * self.config.battery_self_discharge_rate
        self._battery_energy = max(0.0, self._battery_energy - self_discharge_loss)
        self._battery_energy = min(self._battery_energy, self._battery_capacity)

        # Allocate solar to demand (can intentionally under-allocate to enable selling/arbitrage).
        solar_to_demand = min(solar_to_demand, solar_production, demand)
        remaining_solar = solar_production - solar_to_demand

        # Discharge battery to meet the remaining demand (agent chooses the attempt).
        demand_remaining = max(0, demand - solar_to_demand)
        max_discharge = self._battery_energy * self.config.battery_discharge_efficiency
        max_discharge_int = int(np.floor(max_discharge))
        discharge_used = min(battery_to_demand, max_discharge_int, demand_remaining)
        battery_after_discharge = self._battery_energy - discharge_used / self.config.battery_discharge_efficiency

        # Optional battery discharge to sell back to the grid.
        max_sell_discharge = battery_after_discharge * self.config.battery_discharge_efficiency
        max_sell_discharge_int = int(np.floor(max_sell_discharge))
        battery_to_grid_used = min(battery_to_grid, max_sell_discharge_int)
        battery_after_discharge -= (
            battery_to_grid_used / self.config.battery_discharge_efficiency
            if self.config.battery_discharge_efficiency > 0
            else 0.0
        )

        # Charge battery from remaining solar (agent chooses the attempt).
        capacity_remaining = max(0.0, self._battery_capacity - battery_after_discharge)
        max_solar_charge = int(
            np.floor(capacity_remaining / self.config.battery_charge_efficiency)
            if self.config.battery_charge_efficiency > 0
            else 0
        )
        solar_to_battery = min(solar_to_battery, remaining_solar, max_solar_charge)
        stored_from_solar = solar_to_battery * self.config.battery_charge_efficiency
        remaining_solar -= solar_to_battery

        # Charge battery from grid (agent chooses the attempt).
        capacity_remaining = max(0.0, self._battery_capacity - (battery_after_discharge + stored_from_solar))
        max_grid_charge = int(
            np.floor(capacity_remaining / self.config.battery_charge_efficiency)
            if self.config.battery_charge_efficiency > 0
            else 0
        )
        grid_to_battery = min(grid_to_battery, max_grid_charge)
        stored_from_grid = grid_to_battery * self.config.battery_charge_efficiency

        # Any remaining demand is automatically purchased from the grid.
        grid_to_demand = max(0, demand_remaining - discharge_used)

        # Any remaining solar is sold at the fixed selling price.
        solar_sold = remaining_solar

        bought_energy = grid_to_battery + grid_to_demand
        energy_sold = solar_sold + battery_to_grid_used
        balance = -bought_energy * buying_price + energy_sold * selling_price

        battery_energy_before_degradation = battery_after_discharge + stored_from_solar + stored_from_grid
        self._battery_energy = min(self._battery_capacity, battery_energy_before_degradation)

        battery_throughput = (
                discharge_used + battery_to_grid_used + solar_to_battery + grid_to_battery
        )
        degradation_event, degradation_amount = self._apply_degradation(
            battery_energy=self._battery_energy,
            battery_throughput=battery_throughput,
        )

        if degradation_amount > 0.0:
            balance -= degradation_amount * self.config.degradation_cost_per_unit

        metrics = StepMetrics(
            time_step=self._time_step,
            time_of_day=time_of_day,
            day_of_year=day_of_year,
            solar_intensity=solar_intensity,
            solar_production=solar_production,
            demand=demand,
            market_price=market_price,
            buying_price=buying_price,
            selling_price=selling_price,
            solar_to_demand=solar_to_demand,
            solar_to_battery=int(round(solar_to_battery)),
            battery_to_demand=int(round(discharge_used)),
            battery_to_grid=int(round(battery_to_grid_used)),
            grid_to_battery=int(round(grid_to_battery)),
            grid_to_demand=int(round(grid_to_demand)),
            solar_sold=int(solar_sold),
            balance=float(balance),
            battery_energy=float(self._battery_energy),
            battery_capacity=float(self._battery_capacity),
            battery_health=float(self._battery_health),
            battery_throughput=float(battery_throughput),
            degradation_event=bool(degradation_event),
            degradation_amount=float(degradation_amount),
            self_discharge_loss=float(self_discharge_loss),
            forecast_solar_intensity=self._forecast_solar,
            forecast_market_price=self._forecast_price,
            forecast_demand=self._forecast_demand,
            seasonal_solar_factor=float(self._seasonal_solar_factor(day_of_year)),
            seasonal_demand_factor=float(self._seasonal_demand_factor(day_of_year)),
            seasonal_price_factor=float(self._seasonal_price_factor(day_of_year)),
            weather_factor=float(self._daily_weather_factor),
            daylight_factor=float(self._daily_daylight_factor),
        )

        self._logger.debug("Step metrics: %s", metrics)

        # Advance time.
        self._time_step += 1
        self._absolute_time_step += 1
        self._maybe_update_daily_factors(self._current_day_of_year())
        self._sample_exogenous(time_of_day=self._current_time_of_day(), absolute_time=self._absolute_time_step)

        use_trunc = bool(getattr(self.config, "use_time_limit_truncation", False))
        reached_horizon = self._time_step >= self.config.episode_length
        terminated = False if use_trunc else reached_horizon
        truncated = reached_horizon if use_trunc else False

        observation = self._get_observation()
        info = {"metrics": metrics}
        return observation, float(balance), bool(terminated), bool(truncated), info

    def render(self) -> None:
        """Render the current environment state."""

        self._logger.info(
            "t=%s (hour=%s) | battery=%.2f | capacity=%.2f | health=%.2f",
            self._time_step,
            self._current_time_of_day(),
            self._battery_energy,
            self._battery_capacity,
            self._battery_health,
        )

    def _solar_intensity(self, time_of_day: int, day_of_year: int) -> float:
        """Compute solar irradiation intensity ``S_t``.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.
        day_of_year:
            Day of year index in ``[0, year_length_days - 1]``.

        Returns
        -------
        float
            Solar irradiation intensity in ``[0, 1]``.
        """

        cloud_factor = self._rng.uniform(0.0, 1.0)  # W_t in the handout
        solar_angle = 2 * np.pi * time_of_day / 24
        base_intensity = max(0.0, -cloud_factor * np.cos(solar_angle))
        intensity = (
                base_intensity
                * self._seasonal_solar_factor(day_of_year)
                * self._daily_weather_factor
                * self._daily_daylight_factor
        )
        return float(np.clip(intensity, 0.0, 1.0))

    def _expected_solar_intensity(self, time_of_day: int, day_of_year: int) -> float:
        """Return the expected solar intensity."""

        expected_cloud = 0.5
        solar_angle = 2 * np.pi * time_of_day / 24
        base_intensity = max(0.0, -expected_cloud * np.cos(solar_angle))
        intensity = (
                base_intensity
                * self._seasonal_solar_factor(day_of_year)
                * self._daily_daylight_factor
        )
        return float(np.clip(intensity, 0.0, 1.0))

    def _market_price(self, time_of_day: int, day_of_year: int, solar_intensity: float) -> float:
        """Compute the market price component ``R^M_t``.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.
        solar_intensity:
            Solar irradiation intensity ``S_t``.

        Returns
        -------
        float
            Market price component.
        """

        noise = self._rng.normal(0.0, 1.0)  # xi_t
        price_shape = 2 * np.sin(2 * np.pi * time_of_day / 24) ** 2
        seasonal = self._seasonal_price_factor(day_of_year)
        return float(price_shape * seasonal + 0.5 * (noise - solar_intensity))

    def _expected_market_price(self, time_of_day: int, day_of_year: int, solar_intensity: float) -> float:
        """Return the expected market price component."""

        price_shape = 2 * np.sin(2 * np.pi * time_of_day / 24) ** 2
        seasonal = self._seasonal_price_factor(day_of_year)
        return float(price_shape * seasonal - 0.5 * solar_intensity)

    def _demand(self, time_of_day: int, day_of_year: int) -> int:
        """Compute household demand ``D_t``.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.

        Returns
        -------
        int
            Discrete demand level.

        Notes
        -----
        The project description includes a ``min(0, ...)`` term which would
        always yield non-positive demand. We interpret this as ``max(0, ...)``
        to ensure demand is non-negative.
        """

        noise = self._rng.normal(0.0, 1.0)  # zeta_t
        demand_shape = np.sin(2 * np.pi * time_of_day / 24) ** 2
        seasonal = self._seasonal_demand_factor(day_of_year)
        scaled = 2 * self.config.max_battery_energy / np.pi * (demand_shape + 0.5 * noise) * seasonal
        return int(max(0.0, np.floor(scaled)))

    def _expected_demand(self, time_of_day: int, day_of_year: int) -> float:
        """Return the expected demand."""

        demand_shape = np.sin(2 * np.pi * time_of_day / 24) ** 2
        seasonal = self._seasonal_demand_factor(day_of_year)
        scaled = 2 * self.config.max_battery_energy / np.pi * demand_shape * seasonal
        return float(max(0.0, scaled))

    def _sample_exogenous(self, time_of_day: int, absolute_time: int) -> None:
        """Sample stochastic variables for the current time step.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.
        absolute_time:
            Absolute time step across episodes.
        """

        day_of_year = self._day_of_year(absolute_time)
        self._current_solar_intensity = self._solar_intensity(time_of_day, day_of_year)
        self._current_market_price = self._market_price(time_of_day, day_of_year, self._current_solar_intensity)
        self._current_demand = self._demand(time_of_day, day_of_year)
        self._forecast_solar, self._forecast_price, self._forecast_demand = self._forecast_signals(
            time_of_day,
            absolute_time,
        )

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector.

        Returns
        -------
        numpy.ndarray
            Observation vector of ``[time_of_day, day_of_year, buying_price, battery_energy,
            battery_capacity, battery_health, forecast_solar, forecast_price, forecast_demand]``.
        """

        time_of_day = self._current_time_of_day()
        day_of_year = self._current_day_of_year()
        buying_price_raw = float(
            self.config.base_price + self._current_market_price + self.config.trade_fee_per_unit
        )
        buying_price = float(
            np.clip(buying_price_raw, -self._buying_price_clip, self._buying_price_clip)
        )
        forecast_values = (
                list(self._forecast_solar) + list(self._forecast_price) + list(self._forecast_demand)
        )
        return np.array(
            [
                float(time_of_day),
                float(day_of_year),
                float(buying_price),
                float(self._battery_energy),
                float(self._battery_capacity),
                float(self._battery_health),
                *forecast_values,
            ],
            dtype=np.float32,
        )

    def _forecast_signals(
            self,
            time_of_day: int,
            absolute_time: int,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        """Generate noisy forecasts for solar, market price, and demand.

        Parameters
        ----------
        time_of_day:
            Current hour of day in ``[0, 23]``.
        absolute_time:
            Current absolute time step.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
            Forecasted solar intensity, market price, and demand for the next horizon.
        """

        forecast_solar = []
        forecast_price = []
        forecast_demand = []
        for step_ahead in range(1, self._forecast_horizon + 1):
            future_abs_time = absolute_time + step_ahead
            future_time = future_abs_time % 24
            future_day = self._day_of_year(future_abs_time)
            expected_solar = self._expected_solar_intensity(future_time, future_day)
            solar_noise = self._rng.normal(0.0, self.config.forecast_noise_std)
            solar_forecast = float(np.clip(expected_solar + solar_noise, 0.0, 1.0))

            expected_price = self._expected_market_price(future_time, future_day, solar_forecast)
            price_noise = self._rng.normal(0.0, self.config.forecast_noise_std)
            price_forecast = float(
                np.clip(
                    expected_price + price_noise,
                    -self._buying_price_clip,
                    self._buying_price_clip,
                )
            )

            expected_demand = self._expected_demand(future_time, future_day)
            demand_noise = self._rng.normal(0.0, self.config.forecast_noise_std)
            demand_forecast = float(np.clip(expected_demand + demand_noise, 0.0, self.config.max_demand))

            forecast_solar.append(solar_forecast)
            forecast_price.append(price_forecast)
            forecast_demand.append(demand_forecast)

        return tuple(forecast_solar), tuple(forecast_price), tuple(forecast_demand)

    def _seasonal_solar_factor(self, day_of_year: int) -> float:
        seasonal_phase = 2 * np.pi * day_of_year / self.config.year_length_days
        return float(1.0 + self.config.seasonal_solar_amplitude * np.sin(seasonal_phase - np.pi / 2))

    def _seasonal_demand_factor(self, day_of_year: int) -> float:
        seasonal_phase = 2 * np.pi * day_of_year / self.config.year_length_days
        return float(1.0 + self.config.seasonal_demand_amplitude * np.cos(seasonal_phase))

    def _seasonal_price_factor(self, day_of_year: int) -> float:
        seasonal_phase = 2 * np.pi * day_of_year / self.config.year_length_days
        return float(1.0 + self.config.seasonal_price_amplitude * np.cos(seasonal_phase))

    def _maybe_update_daily_factors(self, day_of_year: int) -> None:
        if self._last_day_of_year == day_of_year:
            return
        self._last_day_of_year = day_of_year
        weather = float(self._rng.normal(1.0, self.config.weather_variability))
        self._daily_weather_factor = float(np.clip(weather, 0.3, 1.5))
        seasonal_phase = 2 * np.pi * day_of_year / self.config.year_length_days
        self._daily_daylight_factor = float(
            np.clip(1.0 + self.config.daylight_variability * np.sin(seasonal_phase), 0.5, 1.5)
        )

    def _current_time_of_day(self) -> int:
        return int(self._absolute_time_step % 24)

    def _day_of_year(self, absolute_time: int) -> int:
        return int((absolute_time // 24 + self._day_offset) % self.config.year_length_days)

    def _current_day_of_year(self) -> int:
        return self._day_of_year(self._absolute_time_step)

    def get_action_mask(self) -> np.ndarray:
        """Return a flattened action mask for MaskablePPO."""

        solar_intensity = float(self._current_solar_intensity)
        solar_production = int(np.floor(self.config.max_solar_power * solar_intensity))
        demand = int(self._current_demand)

        max_solar_to_demand = min(solar_production, demand)
        max_discharge = int(np.floor(self._battery_energy * self.config.battery_discharge_efficiency))
        max_battery_to_demand = min(demand, max_discharge)
        max_battery_to_grid = max_discharge

        capacity_remaining = max(0.0, self._battery_capacity - self._battery_energy)
        max_charge = int(
            np.floor(capacity_remaining / self.config.battery_charge_efficiency)
            if self.config.battery_charge_efficiency > 0
            else 0
        )
        max_solar_to_battery = min(solar_production, max_charge)
        max_grid_to_battery = max_charge

        masks = [
            self._range_mask(self.config.max_solar_power, max_solar_to_demand),
            self._range_mask(self.config.max_solar_power, max_solar_to_battery),
            self._range_mask(self.config.max_battery_energy, max_battery_to_demand),
            self._range_mask(self.config.max_battery_energy, max_battery_to_grid),
            self._range_mask(self.config.max_battery_energy, max_grid_to_battery),
        ]
        return np.concatenate(masks, axis=0)

    @property
    def action_masks(self) -> np.ndarray:
        """Expose action masks for sb3-contrib utilities."""

        return self.get_action_mask()

    @staticmethod
    def _range_mask(max_value: int, allowed_max: int) -> np.ndarray:
        mask = np.zeros(max_value + 1, dtype=bool)
        if allowed_max >= 0:
            mask[: min(max_value, allowed_max) + 1] = True
        return mask

    def _apply_degradation(self, battery_energy: float, battery_throughput: float) -> Tuple[bool, float]:
        """Apply battery degradation based on usage and deep discharge.

        Parameters
        ----------
        battery_energy:
            Current battery energy after applying actions.
        battery_throughput:
            Total energy throughput for this step.

        Returns
        -------
        tuple[bool, float]
            Whether a degradation event occurred and the total capacity loss fraction.
        """

        degradation_amount = 0.0
        degradation_event = False

        throughput_loss = self.config.battery_degradation_per_unit * (
                battery_throughput / self.config.max_battery_energy)
        degradation_amount += throughput_loss

        soc_fraction = battery_energy / self._battery_capacity if self._battery_capacity > 0 else 0.0
        if soc_fraction <= self.config.battery_degradation_threshold:
            if self._rng.random() < self.config.battery_degradation_probability:
                degradation_event = True
                degradation_amount += self.config.battery_degradation_event_drop

        if degradation_amount > 0.0:
            self._battery_health = max(
                self.config.battery_min_health,
                self._battery_health - degradation_amount,
            )
            self._battery_capacity = self.config.max_battery_energy * self._battery_health
            self._battery_energy = min(self._battery_energy, self._battery_capacity)

        return degradation_event, degradation_amount
