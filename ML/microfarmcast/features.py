"""Leakage-safe issue-time features and one-hour-ahead labels."""
import hashlib
import numpy as np
import pandas as pd

TARGETS = ["temperature_c", "relative_humidity_pct", "wind_speed_mps", "soil_moisture", "rainfall_mm"]
STATIC = ["latitude", "longitude", "elevation_m"]


def _location_code(value: object) -> float:
    return float(int(hashlib.sha1(str(value).encode()).hexdigest()[:8], 16) / 0xffffffff)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.sort_values(["location_id", "timestamp"]).copy()
    g = out.groupby("location_id", sort=False)
    t = out.timestamp.dt
    out["hour_sin"], out["hour_cos"] = np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24)
    out["year_day_sin"], out["year_day_cos"] = np.sin(2*np.pi*t.dayofyear/365.25), np.cos(2*np.pi*t.dayofyear/365.25)
    out["location_code"] = out.location_id.map(_location_code)
    features = STATIC + ["hour_sin", "hour_cos", "year_day_sin", "year_day_cos", "location_code"]
    # Every observation-derived feature ends at issue time or earlier. The target
    # is independently shifted from the next hourly record below.
    for col in TARGETS:
        for lag in (0, 1, 3, 6, 24):
            name = f"{col}_lag_{lag}h"
            out[name] = g[col].shift(lag)
            features.append(name)
        for hours in (3, 6, 24):
            name = f"{col}_prior_{hours}h_mean"
            out[name] = g[col].transform(lambda s, h=hours: s.shift(1).rolling(h, min_periods=1).mean())
            features.append(name)
    # Require an exact next-hour observation at the same location.
    future = out[["location_id", "timestamp", *TARGETS]].copy()
    future["timestamp"] -= pd.Timedelta(hours=1)
    future = future.rename(columns={c: f"target_{c}" for c in TARGETS})
    out = out.merge(future, on=["location_id", "timestamp"], how="left", validate="one_to_one")
    return out, features
