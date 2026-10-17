# MicroFarmCast Model 1

A beginner-friendly XGBoost prototype for next-hour (configurable horizon) farm/study-area microclimate prediction. This repository currently contains **no real observations**. It never downloads data or requests credentials. A generated dataset is available only for a clearly labelled pipeline smoke check; its metrics are not evidence of predictive skill.

## Current evidence and limits

Workspace discovery found the setup notebook and no source datasets. Until local station/farm sensors provide target labels, this cannot be validated as a true farm-level prediction model. ERA5-Land is roughly 9 km scale. Do not claim 100–500 m accuracy by attaching a 30 m DEM to coarse weather data.

## Setup

From this project directory, create an environment and install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.13 is unavailable, use another supported Python version and create the environment with `python -m venv .venv`.

## Input schema

Provide CSV or Parquet files and list them in `config.json` (`input_paths`). Each input must be a flat table with one row per `location_id` and timestamp. Required keys: `timestamp`, `location_id`, `latitude`, `longitude`; all five target columns below are required for this first multi-target trainer. Extra numeric predictor columns may be added to `feature_columns` in the config.

| Column | Meaning | Unit / convention |
|---|---|---|
| `timestamp` | Observation time | ISO 8601 with timezone preferred; internally normalized to UTC |
| `location_id` | Stable station/grid/farm identifier | String |
| `latitude`, `longitude` | WGS84 point coordinate | Decimal degrees |
| `temperature_c` | Air temperature | °C |
| `relative_humidity_pct` | Relative humidity | % in [0, 100] |
| `rainfall_mm` | Accumulated rainfall for the stated observation interval | mm; document interval and align to target interval |
| `wind_speed_m_s` | Wind speed | m/s |
| `soil_moisture_m3_m3` | Volumetric soil water content | m³/m³ |
| `elevation_m` | Elevation | metres |
| `slope_deg`, `aspect_deg` | Optional terrain derivatives | degrees; aspect clockwise from north |
| `air_pressure_hpa`, `solar_radiation_w_m2` | Optional predictors | hPa, W/m² |

Missing target rows are excluded for that target; missing predictors are filled using medians fit on the training partition only. Feature audit counts are saved. Naive timestamps are interpreted as UTC; local-time source data must first be localized correctly, including daylight-saving transitions. Do not mix units or rainfall accumulation intervals in one column.

### Adding source data

1. **ERA5-Land hourly:** use the [Copernicus Climate Data Store ERA5-Land dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land). Select a small area, dates, hourly frequency, and needed single-level weather/soil variables. CDS account/API setup may be required for scripted requests. Review variable names, units, licence, and the dataset's roughly 9 km effective grid. Convert to the schema units and UTC, add stable grid-cell IDs and coordinates, and document the source in your project notes. No automatic request is made here.
2. **Copernicus DEM GLO-30:** obtain only the tiles covering the study area from the [Copernicus DEM collection](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) after checking the current access eligibility/licence. Derive elevation and, if practical, slope/aspect with a geospatial GIS/raster workflow, then join static values by location. DEM detail does not increase weather-label resolution.
3. **Local station/farm sensors:** use timestamped, quality-controlled observations as target labels. Keep station/farm IDs, coordinates, sensor height/depth, units, sampling interval, calibration/QC flags, and missingness metadata. For credible farm-level evaluation, hold out sensors or farms from training where the sample permits.

## Run

Put real CSV/Parquet files under `data/`, edit `input_paths` and `forecast_horizon_hours` in `config.json`, then run:

```powershell
python train.py --config config.json
```

The horizon is in exact hourly steps. The label at time `t + horizon` is joined by location and timestamp; rows without that exact future label are dropped. Evaluation uses chronological train/validation/test periods. Baseline comparisons include persistence and a regional forecast column when present. A deterministic subset of sites is also excluded from the location-holdout training fit when at least three sites and enough rows are available; otherwise that metric is omitted. The validation partition is reserved in this first version for chronological separation and review; models fit on training-period rows only.

Outputs under `artifacts/`: per-target XGBoost JSON model, training-only feature medians, `metrics.json`, `feature_importance.csv`, and `run_summary.json`. Rainfall uses rain/no-rain classification, then amount regression trained on rainy training examples. Classification metrics are precision, recall, F1, and confusion counts; amount metrics are conditional on observed rain, and a combined predicted-rainfall MAE/RMSE is also reported.

### Synthetic smoke check

To generate an explicitly synthetic CSV and run the same pipeline:

```powershell
python train.py --config config.json --demo-synthetic
```

The demo file is written under `demo_data/`, and results are marked `SYNTHETIC_DEMO_NOT_REAL_PERFORMANCE`. Do not use those metrics or feature importances as validation, and do not merge demo outputs with real run artifacts.

### NASA POWER hourly point CSV

For the downloaded 2024 Bengaluru hourly POWER CSV, convert the metadata-prefixed export and run the four available weather targets:

```powershell
python prepare_power_csv.py "C:\Users\sabre\Downloads\POWER_Point_Hourly_20240101_20241231_012d97N_077d59E_UTC.csv"
python train.py --config config_power.json
```

This export contains temperature, RH, wind, and hourly precipitation but no soil-moisture target. Its meteorology is MERRA-2-derived and roughly 0.5° × 0.625°; use the result as a regional reanalysis prototype, not farm-level validation. The `config_power.json` intentionally trains only the four available targets. It has one location, so spatial holdout metrics are unavailable. The original file's timestamps are UTC, and precipitation is mm/hour.

After obtaining matching ERA5-Land `swvl1` for 2024 at the nearest Bengaluru grid cell, merge and train all five reanalysis targets:

```powershell
python prepare_bengaluru_mixed_sources.py "data/power_bengaluru_2024.csv" "C:\Users\sabre\Downloads\ERA5-2\reanalysis-era5-land-timeseries-sfc-soil-waterz7pocehl.csv"
python train.py --config config_bengaluru_5target.json
```

`swvl1` is volumetric soil water layer 1 in m³/m³. This configuration combines MERRA-2 weather targets with ERA5-Land soil water from the nearest 0.1° grid point; timestamps are joined exactly in UTC, but the sources use different grids. Results are reanalysis-to-reanalysis agreement scores, not independent farm-level accuracy.

## Prediction

After training, use the Python API `microfarmcast.predict.predict_from_history(history, artifacts_dir, horizon_hours=None)`. Supply a DataFrame with the same input schema and enough history for lag/rolling features. The function predicts the next target horizon for each location's last timestamp. Predictions are only as trustworthy as the temporal coverage, feature availability, and target-label quality of the training data.

For command-line inference, copy `data/recent_history_template.csv` to a new CSV and fill it with recent, hourly observations for one or more locations. Include at least 25 consecutive hours per location so the 24-hour lags/rolling features can be formed. Keep timestamps timezone-aware when possible and use the schema units above. Do not put forecast-time measurements in rows after the forecast origin. Run:

```powershell
python predict.py --input data/recent_history.csv --artifacts artifacts_bengaluru_5target --output artifacts_bengaluru_5target/prediction.csv
```

The output contains the history's latest timestamp as `forecast_origin_utc`, the predicted valid time, rain probability, and the five predicted targets. The forecast horizon defaults to the value saved with the model; override it only if the models were trained for that horizon (`--horizon-hours 1`). The bundled 2024 files only produce a retrospective prediction for 2025-01-01; they cannot provide a current forecast. Use genuinely recent observations for current inference.

## Assumptions and known limitations

- Hourly regularity is expected for lag columns; missing timestamps are not filled. Exact future-label matching prevents row shifts, and lag features at missing hours will remain missing before imputation.
- Coordinates are assumed WGS84; timestamps are normalized to UTC; all input units must already match the schema.
- XGBoost does not natively use pandas nulls uniformly across all supported operations here, so train-partition medians are applied consistently to validation/test/inference.
- A future regional forecast is only a fair baseline if it was actually available at prediction time and names the same horizon/valid time. Configure it as a feature and baseline column only when that condition holds.
- A one-hour model trained on reanalysis labels is not an independent operational forecast; it mostly learns relationships in the gridded source. Independent local sensors and spatial/temporal holdouts are necessary for meaningful validation.
