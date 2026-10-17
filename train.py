"""Command line entry point."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from microfarmcast.data import load_config
from microfarmcast.train import train


def make_synthetic(config):
    """Create small deterministic example data, explicitly not for performance claims."""
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    frames = []
    for site, lat, lon, elevation in [("demo_a", 12.9, 77.6, 900), ("demo_b", 13.1, 77.8, 850), ("demo_c", 12.7, 77.4, 950)]:
        n = 24 * 40
        time = pd.date_range(start, periods=n, freq="h")
        h = time.hour.to_numpy()
        temp = 24 + 5*np.sin(2*np.pi*(h-6)/24) + rng.normal(0, 0.5, n)
        rain = (rng.random(n) < 0.12) * rng.gamma(1.5, 2.0, n)
        frame = pd.DataFrame({"timestamp": time, "location_id": site, "latitude": lat, "longitude": lon, "elevation_m": elevation, "slope_deg": 3.0, "aspect_deg": 180.0, "temperature_c": temp, "relative_humidity_pct": np.clip(72-temp*0.6+rng.normal(0, 3, n), 20, 100), "rainfall_mm": rain, "wind_speed_m_s": np.maximum(0, 2+rng.normal(0, .7, n)), "soil_moisture_m3_m3": np.clip(.25+np.cumsum(rain)*.0005+rng.normal(0,.01,n), .05,.6), "air_pressure_hpa": 910+rng.normal(0,1,n), "solar_radiation_w_m2": np.maximum(0, 500*np.sin(2*np.pi*(h-6)/24))})
        frames.append(frame)
    demo = pd.concat(frames, ignore_index=True)
    path = Path("demo_data") / "synthetic_observations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(path, index=False)
    config["input_paths"] = [str(path)]
    config["output_dir"] = "demo_artifacts"
    config["xgboost"] = {**config["xgboost"], "n_estimators": 30, "max_depth": 3, "n_jobs": 2}
    return config


def main():
    parser = argparse.ArgumentParser(description="Train MicroFarmCast Model 1")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--demo-synthetic", action="store_true", help="Run only a clearly labelled synthetic smoke check")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.demo_synthetic:
        config = make_synthetic(config)
    metrics, summary = train(config, args.config, synthetic_demo=args.demo_synthetic)
    if args.demo_synthetic:
        Path(config["output_dir"], "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"label": summary["label"], "summary": summary, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
