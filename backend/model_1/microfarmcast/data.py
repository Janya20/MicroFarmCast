"""CSV loading, schema checks, and timestamp normalization."""
from pathlib import Path
import pandas as pd

REQUIRED = ["timestamp", "location_id", "latitude", "longitude", "elevation_m",
            "temperature_c", "relative_humidity_pct", "rainfall_mm",
            "wind_speed_mps", "soil_moisture"]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    invalid = parsed.isna()
    if invalid.any():
        examples = frame.loc[invalid, "timestamp"].head(5).tolist()
        raise ValueError(f"Invalid timestamps in column 'timestamp' at {int(invalid.sum())} row(s); examples: {examples}")
    frame = frame.copy()
    frame["timestamp"] = parsed
    for column in REQUIRED[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing or non-numeric values")
    if frame["location_id"].isna().any():
        raise ValueError("Column 'location_id' contains missing values")
    if ((frame["relative_humidity_pct"] < 0) | (frame["relative_humidity_pct"] > 100)).any():
        raise ValueError("relative_humidity_pct must be between 0 and 100")
    if (frame[["rainfall_mm", "wind_speed_mps", "soil_moisture"]] < 0).any().any():
        raise ValueError("rainfall_mm, wind_speed_mps, and soil_moisture must be non-negative")
    if frame.duplicated(["location_id", "timestamp"]).any():
        raise ValueError("Duplicate location_id/timestamp observations found")
    return frame.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
