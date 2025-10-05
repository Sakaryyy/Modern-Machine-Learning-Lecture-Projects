from __future__ import annotations

import pandas as pd

# Label mappings and denormalization helpers (for interpretability)
SEASON_MAP = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
YR_MAP     = {0: "2011", 1: "2012"}
MNTH_MAP   = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}
# UCI docs encode weekday 0=Sunday,...,6=Saturday.
WEEKDAY_MAP = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
WEATHERSIT_MAP = {
    1: "Clear/Few clouds",
    2: "Mist/Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Rain/Snow"
}
BIN_MAP = {0: "No", 1: "Yes"}

def _map_if_present(s: pd.Series, mapping: dict[int, str]) -> pd.Series:
    try:
        return s.astype("Int64").map(mapping).astype("category")
    except Exception:
        # Fallback: leave as is if mapping fails
        return s.astype("category")

def add_interpretable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add labelled columns for interpretability in plots.

    Temperature normalization given in the dataset card of the website:
        temp:   (t - t_min) / (t_max - t_min), with t_min = -8°C,  t_max = +39°C  (hourly)
        atemp:  (t - t_min) / (t_max - t_min), with t_min = -16°C, t_max = +50°C  (hourly)
        hum:    normalized to [0,1] by dividing by 100
        windspeed: normalized by dividing by 67 (max).

    Returns a copy with new columns when source columns are present:
        temp_C, atemp_C, humidity_pct, windspeed_0_67,
        season_name, yr_name, mnth_name, weekday_name, weathersit_name,
        holiday_name, workingday_name
    """
    df2 = df.copy()
    if "temp" in df2.columns:
        df2["temp_C"] = df2["temp"] * (39 - (-8)) + (-8)  # 47°C range
    if "atemp" in df2.columns:
        df2["atemp_C"] = df2["atemp"] * (50 - (-16)) + (-16)  # 66°C range
    if "hum" in df2.columns:
        df2["humidity_pct"] = df2["hum"] * 100.0
    if "windspeed" in df2.columns:
        df2["windspeed_0_67"] = df2["windspeed"] * 67.0  # note: unit is dataset-specific

    # Label-mapped categoricals (if present)
    if "season" in df2.columns:
        df2["season_name"] = _map_if_present(df2["season"], SEASON_MAP)
    if "yr" in df2.columns:
        df2["yr_name"] = _map_if_present(df2["yr"], YR_MAP)
    if "mnth" in df2.columns:
        df2["mnth_name"] = _map_if_present(df2["mnth"], MNTH_MAP)
    if "weekday" in df2.columns:
        df2["weekday_name"] = _map_if_present(df2["weekday"], WEEKDAY_MAP)
    if "weathersit" in df2.columns:
        df2["weathersit_name"] = _map_if_present(df2["weathersit"], WEATHERSIT_MAP)
    if "holiday" in df2.columns:
        df2["holiday_name"] = _map_if_present(df2["holiday"], BIN_MAP)
    if "workingday" in df2.columns:
        df2["workingday_name"] = _map_if_present(df2["workingday"], BIN_MAP)
    return df2

def ensure_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    if "hr" in df.columns:
        return df
    # Derive from index if missing
    dfx = df.copy()
    dfx["hr"] = df.index.hour
    return dfx
