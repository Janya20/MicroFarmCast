# MicroFarmCast Backend

FastAPI backend for farm records, Open-Meteo forecasts, weather observations, teammate-provided predictions, risk alerts, and dashboard summaries.

## Setup

Run these commands from the `backend` directory:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `DATABASE_URL` in `.env` to the connection string for the existing PostgreSQL database. Keep `.env` private and do not commit it. The application loads this setting from the backend directory and does not create or modify tables during startup.

## Run

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. The health endpoint is `http://127.0.0.1:8000/health`.

## Tests

Run the isolated tests from the `backend` directory:

```powershell
python -m unittest discover -s tests -v
```

The `test_api.py` tests mock the SQLAlchemy session and Open-Meteo/model
integrations; they do not connect to PostgreSQL. The separate
`test_database_integration.py` suite targets only `microfarmcast_test_db` and
requires its existing schema to be provisioned first. It never creates tables
or uses the main database.

## Model Integration

Member 1's microclimate model and Member 3's crop-risk model should each be exposed as a callable that accepts a dictionary and returns a JSON-compatible dictionary. The service input contains `farm` (ID, latitude, longitude, and crop type), `weather`, and `features`. Register the real callables during application initialization, once their modules are available:

```python
from app.services.prediction_service import (
    register_crop_risk_model,
    register_microclimate_model,
)
from teammate_models import crop_risk, microclimate

register_microclimate_model(microclimate.predict, "member-1-model")
register_crop_risk_model(crop_risk.predict, "member-3-model")
```

The crop-risk callable must include a `risk_level` of `low`, `medium`, `high`, or `critical` in its result. Without a registered model, prediction endpoints return `503`; they do not return fabricated predictions. High and critical results create deduplicated alerts. Explanation and recommended-action text is included only when the model returns it.

Sensor readings require connected hardware or a real sensor integration. Until then, `/api/weather/sensors` returns `501`; manually submitted observations remain separate from Open-Meteo forecasts.

## IBM LinuxONE Note

The current dependency list uses `psycopg2-binary`. Verify that a compatible wheel is available for the target IBM LinuxONE Python/platform combination. If not, use the platform-supported Psycopg/libpq installation path during deployment; no database driver or schema change is made by this backend setup.
