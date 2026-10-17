"""Sensor, observation, and external forecast endpoints."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Farm, WeatherObservation
from app.schemas import ObservationCreate, ObservationResponse, WeatherForecastResponse
from app.services import weather_service

router = APIRouter(prefix="/api/weather", tags=["weather"])


@router.get("/sensors")
def get_sensor_observations():
    """Report the sensor integration as unavailable until hardware is connected."""
    raise HTTPException(
        status_code=501,
        detail="Sensor hardware integration is not configured.",
    )


@router.get("/observations", response_model=list[ObservationResponse])
def get_weather_observations(
    farm_id: int | None = Query(default=None, gt=0),
    db: Session = Depends(get_db),
):
    """List persisted observations, optionally for one farm."""
    statement = select(WeatherObservation).order_by(
        WeatherObservation.observed_at.desc()
    )
    if farm_id is not None:
        if db.get(Farm, farm_id) is None:
            raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
        statement = statement.where(WeatherObservation.farm_id == farm_id)
    return db.scalars(statement).all()


@router.post(
    "/observations/{farm_id}",
    response_model=ObservationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_weather_observation(
    observation: ObservationCreate,
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    """Persist supplied sensor or weather-station data without inventing readings."""
    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
    record = WeatherObservation(
        farm_id=farm_id,
        **observation.model_dump(exclude_none=True),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get(
    "/forecast",
    response_model=WeatherForecastResponse,
    summary="Get an hourly Open-Meteo forecast",
    description=(
        "Returns temperature, relative humidity, precipitation, and wind speed "
        "for the requested coordinates. Forecast data is provided by Open-Meteo."
    ),
    responses={
        422: {"description": "Latitude or longitude is outside the valid range."},
        502: {"description": "The weather provider returned an error or invalid response."},
        504: {"description": "The weather provider timed out."},
    },
)
async def get_weather_forecast(
    latitude: float = Query(ge=-90, le=90, description="Latitude in decimal degrees."),
    longitude: float = Query(ge=-180, le=180, description="Longitude in decimal degrees."),
):
    """Fetch a real forecast; provider failures are never replaced with sample data."""
    try:
        return await weather_service.fetch_forecast(latitude, longitude)
    except weather_service.ForecastTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except (
        weather_service.ForecastConnectionError,
        weather_service.ForecastProviderError,
    ) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get(
    "/farms/{farm_id}/forecast",
    response_model=WeatherForecastResponse,
    summary="Get an hourly forecast for a registered farm",
    responses={
        404: {"description": "The farm was not found."},
        502: {"description": "The weather provider returned an error."},
        504: {"description": "The weather provider timed out."},
    },
)
async def get_farm_weather_forecast(
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    """Fetch a forecast from the farm's saved coordinates."""
    farm = db.get(Farm, farm_id)
    if farm is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
    try:
        return await weather_service.fetch_forecast(farm.latitude, farm.longitude)
    except weather_service.ForecastTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except (
        weather_service.ForecastConnectionError,
        weather_service.ForecastProviderError,
    ) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error