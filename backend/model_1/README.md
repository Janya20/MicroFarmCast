# MicroFarmCast Model 1 prototype

Runnable one-hour-ahead XGBoost prototype for temperature, relative humidity, wind speed, soil moisture, and rainfall. The project contained no real sensor/weather observations at inspection. The included/generated CSV is synthetic and is only a pipeline demonstration. Synthetic metrics are not evidence of farm-level skill, 100–500 m validation, or real-world accuracy; local observations are required for that.

## Quick start

Python 3.10+ is required. Install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python train.py --config config.yaml
python predict.py --input data/demo_microclimate.csv --location DEMO_A --timestamp 2025-07-15T12:00:00Z --artifacts artifacts
```

Training writes five bundles (`temperature_c`, `relative_humidity_pct`, `wind_speed_mps`, `soil_moisture`, and two-stage `rainfall`) and `artifacts/metrics.json`. Prediction uses one observed issue-time row plus its preceding location history; it forecasts the next hour. Supply a timestamp that exists in the CSV for the requested location.

## Data schema

CSV columns: `timestamp, location_id, latitude, longitude, elevation_m, temperature_c, relative_humidity_pct, rainfall_mm, wind_speed_mps, soil_moisture`. Timestamp accepts ISO-8601 and is normalized to UTC. Duplicate location/time pairs, invalid timestamps, missing schema columns, nonnumeric values, out-of-range humidity, and negative rain/wind/soil values produce clear errors. Units are °C, percent, mm per observation interval, m/s, and volumetric soil moisture (m³/m³). The demo also has a `synthetic_demo` marker column.

`scripts/generate_demo_data.py` creates 45 hourly days for six invented locations at `data/demo_microclimate.csv`. Randomness is seeded. Do not mix synthetic rows with observed data for evaluation. Change `data.input_csv` in `config.yaml` for another CSV that follows the schema.

## Modeling details

For every location, training labels are made by shifting observations back one hour: the row at issue time T receives the matching same-location observation at T+1 as its target. Current values and lags are available at T; rolling means explicitly shift one row before calculating. Features include coordinates, elevation, cyclical time, a stable location encoding, current/lagged observations, and prior rolling means. Nothing from T+1 enters the feature matrix.

Splits are chronological (70% train, next 15% validation, final 15% test), with boundary rows embargoed so next-hour targets remain in their partition. Imputers fit only on training rows. Continuous targets report MAE/RMSE beside persistence (carry the issue-time value forward). Rainfall uses a rain/no-rain classifier followed by an amount regressor trained only on rainy training rows; reports classification scores and all-case amount errors. With real data, validate across time and held-out locations before operational use.

## Project files

- `microfarmcast/data.py`: CSV loading and validation.
- `microfarmcast/features.py`: leakage-safe feature and label creation.
- `microfarmcast/train.py`: fitting, evaluation, and artifact saving.
- `microfarmcast/predict.py`: saved-model inference.
- `scripts/generate_demo_data.py`: reproducible synthetic data generation.
- `config.yaml`: paths and model settings.

## Backend integration status

This folder contains the standalone Model 1 training and batch prediction prototype. It is not registered with the FastAPI prediction endpoints yet: those endpoints currently accept a current weather/features payload, while this model requires a timestamped observation history to build lag features. The included CSV is synthetic and exists only for running the demo. The repository-level datasets are not used by this standalone demo.
