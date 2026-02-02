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
        Legacy fixed revenue per unit of energy sold back to the grid.
    episode_length:
        Number of time steps in a single episode.
    forecast_horizon:
        Number of hours of look-ahead forecasts provided to the agent.
    forecast_noise_std:
        Standard deviation of forecast errors added to solar, demand, and prices.
    buying_price_clip:
        Clip range for buying prices in observations.
    max_demand:
        Upper bound on demand used to define the observation space.
    initial_soc_fraction:
        Initial battery state-of-charge expressed as a fraction of capacity.
    battery_charge_efficiency:
        Efficiency applied when charging the battery (0-1).
    battery_discharge_efficiency:
        Efficiency applied when discharging the battery (0-1).
    battery_self_discharge_rate:
        Fraction of energy lost each time step due to self-discharge.
    battery_degradation_threshold:
        State-of-charge fraction below which degradation risk increases.
    battery_degradation_probability:
        Probability of a degradation event when below the threshold.
    battery_degradation_event_drop:
        Fractional capacity loss when a degradation event occurs.
    battery_degradation_per_unit:
        Capacity loss per unit of energy throughput.
    battery_min_health:
        Minimum allowable battery health (capacity fraction).
    degradation_cost_per_unit:
        Monetary penalty per unit of capacity lost.
    selling_price_multiplier:
        Multiplier applied to the market price when selling to the grid.
    selling_price_offset:
        Offset applied to the selling price after the multiplier.
    trade_fee_per_unit:
        Transaction fee applied to each unit bought or sold.
    persist_battery_state:
        Keep battery energy, health, and capacity across resets.
    randomize_start_day:
        Randomize the starting day-of-year on reset for seasonal variety.
    year_length_days:
        Number of days in the seasonal cycle (default: 365).
    seasonal_solar_amplitude:
        Seasonal amplitude applied to solar production.
    seasonal_demand_amplitude:
        Seasonal amplitude applied to household demand.
    seasonal_price_amplitude:
        Seasonal amplitude applied to market prices.
    weather_variability:
        Daily weather multiplier variability applied to solar intensity.
    daylight_variability:
        Seasonal daylight multiplier applied to solar intensity.
    """

    max_battery_energy: int = 10
    max_solar_power: int = 11
    base_price: float = 1.0
    selling_price: float = 1.0
    episode_length: int = 24
    forecast_horizon: int = 3
    forecast_noise_std: float = 0.15
    buying_price_clip: float = 10.0
    max_demand: int = 20
    initial_soc_fraction: float = 0.5
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.92
    battery_self_discharge_rate: float = 0.002
    battery_degradation_threshold: float = 0.2
    battery_degradation_probability: float = 0.08
    battery_degradation_event_drop: float = 0.01
    battery_degradation_per_unit: float = 0.002
    battery_min_health: float = 0.6
    degradation_cost_per_unit: float = 4.0
    selling_price_multiplier: float = 0.9
    selling_price_offset: float = 0.05
    trade_fee_per_unit: float = 0.02
    persist_battery_state: bool = False  # False for single-day training (each episode independent)
    randomize_start_day: bool = True
    year_length_days: int = 365
    seasonal_solar_amplitude: float = 0.35
    seasonal_demand_amplitude: float = 0.25
    seasonal_price_amplitude: float = 0.15
    weather_variability: float = 0.25
    daylight_variability: float = 0.2

    # Reward shaping parameters (optional bonuses to improve learning signal)
    # These are tuned to provide a strong learning signal to help the agent learn quickly
    enable_reward_shaping: bool = True
    solar_utilization_bonus: float = 0.5  # Strong bonus per unit of solar used (encourages solar usage)
    battery_health_bonus: float = 0.2  # Bonus for maintaining battery health (scaled by health 0-1)
    demand_coverage_penalty: float = 0.0  # Additional penalty for unmet demand (beyond grid cost)

    # Additional reward shaping for improved learning
    low_price_charging_bonus: float = 0.3  # Strong bonus for charging when price is below threshold
    high_price_selling_bonus: float = 0.4  # Strong bonus for selling/discharging when price is high
    price_threshold_low: float = 1.2  # Price below which charging is "smart"
    price_threshold_high: float = 1.3  # Price above which selling/discharging is "smart"

    # Reduce degradation penalty for faster learning
    battery_degradation_penalty_scale: float = 1.0  # Scale factor for degradation penalty (lower = less penalty)
