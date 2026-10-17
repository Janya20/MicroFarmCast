"""Input loading, schema checks, and UTC normalization."""
from pathlib import Path
import pandas as pd


def load_config(path):
    import json
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs(paths, timestamp_column, location_id_column, required_extra=()):
    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}; use CSV or Parquet")
        frame["_source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No input_paths configured")
    data = pd.concat(frames, ignore_index=True)
    required = {timestamp_column, location_id_column, *required_extra}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    # Naive values are interpreted as UTC; local wall-clock timestamps must be localized upstream.
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce", utc=True)
    invalid_times = int(data[timestamp_column].isna().sum())
    data = data.dropna(subset=[timestamp_column, location_id_column]).copy()
    data[location_id_column] = data[location_id_column].astype(str)
    data = data.sort_values([location_id_column, timestamp_column])
    duplicate_count = int(data.duplicated([location_id_column, timestamp_column]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate location/timestamp rows; resolve duplicates explicitly")
    return data, {"rows_loaded": len(frames) and sum(len(f) for f in frames), "invalid_timestamps_removed": invalid_times, "locations": int(data[location_id_column].nunique()), "columns": list(data.columns), "source_files": [str(Path(p)) for p in paths]}


def validate_targets(data, target_columns):
    missing = sorted(set(target_columns.values()) - set(data.columns))
    if missing:
        raise ValueError(f"Missing target columns: {missing}. See README schema.")
    for name, column in target_columns.items():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if name == "relative_humidity" and ((data[column].dropna() < 0).any() or (data[column].dropna() > 100).any()):
            raise ValueError(f"{column} must be percent in [0, 100]")
        if name == "rainfall" and (data[column].dropna() < 0).any():
            raise ValueError(f"{column} must be non-negative mm")
    return data
