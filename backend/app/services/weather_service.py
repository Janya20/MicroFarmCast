"""Asynchronous client for the free Open-Meteo forecast API."""

import httpx
from pydantic import ValidationError

from app.schemas import WeatherForecastResponse

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_TIMEOUT_SECONDS = 10.0


class ForecastTimeoutError(Exception):
    """The Open-Meteo request exceeded its timeout."""


class ForecastConnectionError(Exception):
    """The Open-Meteo service could not be reached."""


class ForecastProviderError(Exception):
    """Open-Meteo returned an error or an unexpected response."""


async def fetch_forecast(
    latitude: float,
    longitude: float,
    client: httpx.AsyncClient | None = None,
) -> WeatherForecastResponse:
    """Fetch and validate hourly weather data for the given coordinates."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        ),
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }

    if client is not None:
        return await _request_forecast(client, params)

    async with httpx.AsyncClient(timeout=FORECAST_TIMEOUT_SECONDS) as async_client:
        return await _request_forecast(async_client, params)


async def _request_forecast(
    client: httpx.AsyncClient,
    params: dict[str, float | str],
) -> WeatherForecastResponse:
    """Request provider data and convert provider failures into clear errors."""
    try:
        response = await client.get(FORECAST_URL, params=params)
    except httpx.TimeoutException as error:
        raise ForecastTimeoutError("Open-Meteo did not respond before the timeout.") from error
    except httpx.RequestError as error:
        raise ForecastConnectionError("Could not connect to Open-Meteo.") from error

    if response.is_error:
        raise ForecastProviderError(
            f"Open-Meteo returned HTTP {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise ForecastProviderError("Open-Meteo returned invalid JSON.") from error

    if not isinstance(payload, dict) or payload.get("error"):
        raise ForecastProviderError("Open-Meteo reported a forecast error.")

    try:
        return WeatherForecastResponse.model_validate(payload)
    except ValidationError as error:
        raise ForecastProviderError(
            "Open-Meteo returned an unexpected forecast format."
        ) from error