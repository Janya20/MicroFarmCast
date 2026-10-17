"""Load saved models and predict from each location's latest history row."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from .features import build_supervised


def predict_from_history(history, artifacts_dir="artifacts", horizon_hours=None):
    artifacts = Path(artifacts_dir).resolve()
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    if horizon_hours is not None:
        config["forecast_horizon_hours"] = int(horizon_hours)
    medians = json.loads((artifacts / "training_medians.json").read_text(encoding="utf-8"))
    data = history.copy()
    ts, loc = config["timestamp_column"], config["location_id_column"]
    data[ts] = pd.to_datetime(data[ts], errors="coerce", utc=True)
    supervised, features = build_supervised(data, config)
    latest = supervised.sort_values(ts).groupby(loc, as_index=False).tail(1).copy()
    results = latest[[loc, ts]].rename(columns={ts: "forecast_origin_utc"}).copy()
    results["valid_time_utc"] = results["forecast_origin_utc"] + pd.to_timedelta(config["forecast_horizon_hours"], unit="h")
    X = latest[features].replace([np.inf, -np.inf], np.nan)
    for name, median in medians.items():
        target = config["target_columns"].get(name)
        if name == "rainfall":
            classifier = XGBClassifier()
            classifier.load_model(bytearray((artifacts / "rainfall_occurrence.json").read_bytes()))
            probability = classifier.predict_proba(X.fillna(pd.Series(median)))[:, 1]
            results["rain_probability"] = probability
            pred_rain = probability >= 0.5
            amount_path = artifacts / "rainfall_amount.json"
            rainfall = np.zeros(len(X))
            if amount_path.exists():
                amount = XGBRegressor(); amount.load_model(bytearray(amount_path.read_bytes()))
                rainfall[pred_rain] = np.maximum(0, np.asarray(amount.predict(X.loc[pred_rain].fillna(pd.Series(median)))).reshape(-1))
            results[target] = rainfall
        elif (artifacts / f"{name}.json").exists():
            model = XGBRegressor(); model.load_model(bytearray((artifacts / f"{name}.json").read_bytes()))
            results[target] = model.predict(X.fillna(pd.Series(median)))
    return results
