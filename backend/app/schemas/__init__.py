
"""Pydantic schemas for MicroFarmCast API."""

from datetime import datetime
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class FarmCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    crop_type: str | None = Field(default=None, max_length=100)
    location_description: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Farm name cannot be blank.")
        return value


class FarmResponse(ORMResponse):
    id: int
    name: str
    latitude: float
    longitude: float
    crop_type: str | None = None
    location_description: str | None = None
    created_at: datetime


class ObservationCreate(BaseModel):
    temperature_c: float | None = Field(default=None, ge=-100, le=80, allow_inf_nan=False)
    humidity_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    soil_moisture: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    precipitation_mm: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    wind_speed_mps: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source: str | None = Field(default=None, max_length=50)
    observed_at: datetime | None = None

    @model_validator(mode="after")
    def require_measurement(self) -> "ObservationCreate":
        measurements = (
            self.temperature_c,
            self.humidity_percent,
            self.soil_moisture,
            self.precipitation_mm,
            self.wind_speed_mps,
        )
        if all(value is None for value in measurements):
            raise ValueError("At least one measured weather value is required.")
        return self


class ObservationResponse(ORMResponse):
    id: int
    farm_id: int
    temperature_c: float | None = None
    humidity_percent: float | None = None
    soil_moisture: float | None = None
    precipitation_mm: float | None = None
    wind_speed_mps: float | None = None
    source: str | None = None
    observed_at: datetime
    created_at: datetime


class PredictionCreate(BaseModel):
    prediction_type: str = Field(min_length=1, max_length=50)
    predicted_value: dict[str, Any]
    risk_level: str | None = Field(default=None, max_length=30)
    model_name: str | None = Field(default=None, max_length=150)
    predicted_at: datetime | None = None


class PredictionResponse(ORMResponse):
    id: int
    farm_id: int
    prediction_type: str
    predicted_value: dict[str, Any]
    risk_level: str | None = None
    model_name: str | None = None
    predicted_at: datetime
    created_at: datetime


class AlertResponse(ORMResponse):
    id: int
    farm_id: int
    title: str
    message: str
    severity: str | None = None
    status: str
    created_at: datetime
    acknowledged_at: datetime | None = None

class AlertCreate(BaseModel):
    farm_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"] | None = None
    status: Literal["open", "acknowledged", "closed"] = "open"
    acknowledged_at: datetime | None = None


class WeatherForecastHourly(BaseModel):
    time: list[datetime]
    temperature_2m: list[float | None]
    relative_humidity_2m: list[float | None]
    precipitation: list[float | None]
    wind_speed_10m: list[float | None]

    @model_validator(mode="after")
    def validate_hourly_values(self) -> "WeatherForecastHourly":
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.relative_humidity_2m),
            len(self.precipitation),
            len(self.wind_speed_10m),
        }
        if len(lengths) != 1:
            raise ValueError("Hourly forecast arrays must have matching lengths.")
        if any(
            value is not None and (not isfinite(value) or not -100 <= value <= 80)
            for value in self.temperature_2m
        ):
            raise ValueError("Forecast temperatures must be finite and within -100 to 80 C.")
        if any(value is not None and not 0 <= value <= 100 for value in self.relative_humidity_2m):
            raise ValueError("Forecast humidity values must be between 0 and 100 percent.")
        if any(
            value is not None and (not isfinite(value) or value < 0)
            for value in self.precipitation
        ):
            raise ValueError("Forecast precipitation values cannot be negative.")
        if any(
            value is not None and (not isfinite(value) or value < 0)
            for value in self.wind_speed_10m
        ):
            raise ValueError("Forecast wind speeds cannot be negative.")
        return self


class WeatherForecastResponse(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    utc_offset_seconds: int
    hourly_units: dict[str, str]
    hourly: WeatherForecastHourly


class DashboardSummaryResponse(BaseModel):
    farms: list["DashboardFarmSummary"]
    total_farms: int
    total_observations: int
    total_predictions: int
    total_alerts: int
    predictions_by_risk_level: dict[str, int]
    alerts_by_severity: dict[str, int]


class DashboardFarmSummary(BaseModel):
    farm: FarmResponse
    latest_observation: ObservationResponse | None = None
    latest_microclimate_prediction: PredictionResponse | None = None
    latest_crop_risk_prediction: PredictionResponse | None = None
    active_alerts: list[AlertResponse]


class MicroclimatePredictionRequest(BaseModel):
    farm_id: int = Field(gt=0)
    weather: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)


class CropRiskPredictionRequest(BaseModel):
    farm_id: int = Field(gt=0)
    crop_type: str | None = Field(default=None, max_length=100)
    weather: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)


class PredictionResultResponse(BaseModel):
    prediction_id: int
    farm_id: int
    prediction_type: Literal["microclimate", "crop-risk"]
    result: dict[str, Any]
    risk_level: str | None = None
    model_name: str
    predicted_at: datetime


class AlertStatusUpdate(BaseModel):
    status: Literal["acknowledged", "closed"]