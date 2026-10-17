"""Leakage-aware time, lag, rolling, and exact-horizon label construction."""
import numpy as np
import pandas as pd


def build_supervised(data, config):
    ts = config["timestamp_column"]
    loc = config["location_id_column"]
    horizon = int(config["forecast_horizon_hours"])
    if horizon < 1:
        raise ValueError("forecast_horizon_hours must be >= 1")
    data = data.sort_values([loc, ts]).copy()
    hour = data[ts].dt.hour
    doy = data[ts].dt.dayofyear
    forecast_time = data[ts] + pd.to_timedelta(horizon, unit="h")
    forecast_hour = forecast_time.dt.hour
    forecast_doy = forecast_time.dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    data["forecast_hour_sin"] = np.sin(2 * np.pi * forecast_hour / 24)
    data["forecast_hour_cos"] = np.cos(2 * np.pi * forecast_hour / 24)
    data["forecast_doy_sin"] = np.sin(2 * np.pi * forecast_doy / 365.25)
    data["forecast_doy_cos"] = np.cos(2 * np.pi * forecast_doy / 365.25)
    configured = list(dict.fromkeys(config["feature_columns"]))
    for column in configured:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    # Any configured feature absent in a particular file is ignored, then audited by the caller.
    base_features = [c for c in configured if c in data.columns]
    engineered = []
    derived = {}
    for column in base_features:
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue
        for lag in config.get("lags_hours", []):
            name = f"{column}_lag_{lag}h"
            derived[name] = data.groupby(loc, sort=False)[column].shift(int(lag))
            engineered.append(name)
        for window in config.get("rolling_windows_hours", []):
            name = f"{column}_rolling_mean_{window}h"
            # Shift first so the rolling window contains strictly earlier observations.
            derived[name] = data.groupby(loc, sort=False)[column].transform(lambda s: s.shift(1).rolling(int(window), min_periods=1).mean())
            engineered.append(name)
    if derived:
        data = pd.concat([data, pd.DataFrame(derived, index=data.index)], axis=1)
    feature_columns = [c for c in base_features if pd.api.types.is_numeric_dtype(data[c])]
    feature_columns += ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "forecast_hour_sin", "forecast_hour_cos", "forecast_doy_sin", "forecast_doy_cos"] + engineered
    feature_columns = list(dict.fromkeys(feature_columns))
    # Labels are joined on exact valid time rather than row offset, so data gaps cannot shift targets.
    label_source = data[[loc, ts] + list(config["target_columns"].values())].copy()
    label_source[ts] = label_source[ts] - pd.to_timedelta(horizon, unit="h")
    label_source = label_source.rename(columns={c: f"__future__{name}" for name, c in config["target_columns"].items()})
    supervised = data.merge(label_source, on=[loc, ts], how="left", validate="one_to_one")
    return supervised, feature_columns
