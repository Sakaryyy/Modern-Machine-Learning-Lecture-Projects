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
    solar_to_battery:
        Solar units allocated to charging the battery.
    battery_to_demand:
        Battery units discharged to cover demand.
    grid_to_battery:
        Grid units purchased to charge the battery.
    grid_to_demand:
        Grid units purchased to cover remaining demand.
    solar_sold:
        Solar units sold to the grid.
    balance:
        Monetary balance ``B_t``.
    battery_energy:
        Battery energy after the transition.
    """

    time_step: int
    time_of_day: int
    solar_intensity: float
    solar_production: int
    demand: int
    market_price: float
    buying_price: float
    solar_to_demand: int
    solar_to_battery: int
    battery_to_demand: int
    grid_to_battery: int
    grid_to_demand: int
    solar_sold: int
    balance: float
    battery_energy: int


class EnergyBudgetEnv(gym.Env):
    """Energy budgeting environment based on the project specification.

    The agent manages how solar production, battery energy, and grid purchases
    are used to meet household demand while maximizing monetary balance.
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
            ]
        )

        self._buying_price_clip: float = float(getattr(self.config, "buying_price_clip", 10.0))

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -self._buying_price_clip, 0.0], dtype=np.float32),
            high=np.array(
                [23.0, self._buying_price_clip, self.config.max_battery_energy],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._time_step = 0
        self._battery_energy = self.config.max_battery_energy // 2
        self._current_solar_intensity = 0.0
        self._current_market_price = 0.0
        self._current_demand = 0

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
        self._battery_energy = self.config.max_battery_energy // 2

        self._sample_exogenous(time_of_day=self._time_step % 24)

        observation = self._get_observation()
        info = {"battery_energy": self._battery_energy}
        self._logger.debug("Environment reset: %s", info)
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action:
            Action array consisting of
            ``[solar_to_demand, solar_to_battery, battery_to_demand, grid_to_battery]``.

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
        if action_arr.shape[0] != 4:
            raise ValueError(f"Expected action of shape (4,), got {action_arr.shape}")

        solar_to_demand, solar_to_battery, battery_to_demand, grid_to_battery = (
            int(v) for v in action_arr
        )

        time_of_day = self._time_step % 24

        solar_intensity = float(self._current_solar_intensity)
        solar_production = int(np.floor(self.config.max_solar_power * solar_intensity))

        market_price = float(self._current_market_price)
        buying_price = float(self.config.base_price + market_price)

        demand = int(self._current_demand)

        # Allocate solar to demand (can intentionally under-allocate to enable selling/arbitrage).
        solar_to_demand = min(solar_to_demand, solar_production, demand)
        remaining_solar = solar_production - solar_to_demand

        # Discharge battery to meet the remaining demand (agent chooses the attempt).
        demand_remaining = max(0, demand - solar_to_demand)
        discharge_used = min(battery_to_demand, self._battery_energy, demand_remaining)
        battery_after_discharge = self._battery_energy - discharge_used

        # Charge battery from remaining solar (agent chooses the attempt).
        capacity_remaining = self.config.max_battery_energy - battery_after_discharge
        solar_to_battery = min(solar_to_battery, remaining_solar, capacity_remaining)
        remaining_solar -= solar_to_battery

        # Charge battery from grid (agent chooses the attempt).
        grid_to_battery = min(grid_to_battery, capacity_remaining - solar_to_battery)

        # Any remaining demand is automatically purchased from the grid.
        grid_to_demand = max(0, demand_remaining - discharge_used)

        # Any remaining solar is sold at the fixed selling price.
        solar_sold = remaining_solar

        bought_energy = grid_to_battery + grid_to_demand
        balance = -bought_energy * buying_price + solar_sold * self.config.selling_price

        self._battery_energy = min(
            self.config.max_battery_energy,
            battery_after_discharge + solar_to_battery + grid_to_battery,
        )

        metrics = StepMetrics(
            time_step=self._time_step,
            time_of_day=time_of_day,
            solar_intensity=solar_intensity,
            solar_production=solar_production,
            demand=demand,
            market_price=market_price,
            buying_price=buying_price,
            solar_to_demand=solar_to_demand,
            solar_to_battery=solar_to_battery,
            battery_to_demand=discharge_used,
            grid_to_battery=grid_to_battery,
            grid_to_demand=grid_to_demand,
            solar_sold=solar_sold,
            balance=float(balance),
            battery_energy=int(self._battery_energy),
        )

        self._logger.debug("Step metrics: %s", metrics)

        # Advance time.
        self._time_step += 1
        self._sample_exogenous(time_of_day=self._time_step % 24)

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
            "t=%s (hour=%s) | battery=%s",
            self._time_step,
            self._time_step % 24,
            self._battery_energy,
        )

    def _solar_intensity(self, time_of_day: int) -> float:
        """Compute solar irradiation intensity ``S_t``.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.

        Returns
        -------
        float
            Solar irradiation intensity in ``[0, 1]``.
        """

        cloud_factor = self._rng.uniform(0.0, 1.0)  # W_t in the handout
        solar_angle = 2 * np.pi * time_of_day / 24
        intensity = max(0.0, -cloud_factor * np.cos(solar_angle))
        return float(intensity)

    def _market_price(self, time_of_day: int, solar_intensity: float) -> float:
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
        return float(price_shape + 0.5 * (noise - solar_intensity))

    def _demand(self, time_of_day: int) -> int:
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
        scaled = 2 * self.config.max_battery_energy / np.pi * (demand_shape + 0.5 * noise)
        return int(max(0.0, np.floor(scaled)))

    def _sample_exogenous(self, time_of_day: int) -> None:
        """Sample stochastic variables for the current time step.

        Parameters
        ----------
        time_of_day:
            Hour of day in ``[0, 23]``.
        """

        self._current_solar_intensity = self._solar_intensity(time_of_day)
        self._current_market_price = self._market_price(time_of_day, self._current_solar_intensity)
        self._current_demand = self._demand(time_of_day)

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector.

        Returns
        -------
        numpy.ndarray
            Observation vector of ``[time_of_day, buying_price, battery_energy]``.
        """

        time_of_day = self._time_step % 24
        buying_price_raw = float(self.config.base_price + self._current_market_price)
        buying_price = float(
            np.clip(buying_price_raw, -self._buying_price_clip, self._buying_price_clip)
        )
        return np.array([time_of_day, buying_price, self._battery_energy], dtype=np.float32)
