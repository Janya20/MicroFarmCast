"""Load saved bundles and forecast one hour ahead for a farm issue time."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import load_csv
from .features import build_features, TARGETS


def predict(input_csv: str, location_id: str, timestamp: str, artifact_dir: str = "artifacts") -> dict:
    history = load_csv(input_csv)
    issue = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(issue): raise ValueError(f"Invalid prediction timestamp: {timestamp!r}")
    match = history[(history.location_id.astype(str) == str(location_id)) & (history.timestamp == issue)]
    if match.empty: raise ValueError("No observation exists for this location and issue timestamp; provide a matching history row.")
    featured, names = build_features(history)
    row = featured[(featured.location_id.astype(str) == str(location_id)) & (featured.timestamp == issue)]
    if row.empty: raise ValueError("Could not construct features for requested farm/time.")
    result = {"location_id":str(location_id), "issue_timestamp":issue.isoformat(), "forecast_timestamp":(issue+pd.Timedelta(hours=1)).isoformat(), "predictions":{}}
    root = Path(artifact_dir)
    for col in TARGETS:
        model_file = root/"models"/("rainfall.joblib" if col == "rainfall_mm" else f"{col}.joblib")
        if not model_file.exists(): raise FileNotFoundError(f"Saved model not found: {model_file}")
        bundle = joblib.load(model_file)
        xraw = row[bundle["features"]].replace([np.inf,-np.inf],np.nan)
        x = bundle["imputer"].transform(xraw)
        if bundle["kind"] == "rainfall":
            probability = float(bundle["classifier"].predict_proba(x)[0,1])
            amount = max(0.0, float(bundle["amount_model"].predict(x)[0])) if probability >= .5 else 0.0
            result["predictions"][col] = {"rain_probability":probability, "rainfall_mm":amount}
        else: result["predictions"][col] = float(bundle["model"].predict(x)[0])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Forecast one hour ahead from supplied farm history")
    p.add_argument("--input", required=True); p.add_argument("--location", required=True)
    p.add_argument("--timestamp", required=True); p.add_argument("--artifacts", default="artifacts")
    a = p.parse_args()
    print(json.dumps(predict(a.input, a.location, a.timestamp, a.artifacts), indent=2))
