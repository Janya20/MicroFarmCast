"""Non-destructive API and service tests using mocked database sessions."""

from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.database import get_db
from app.main import app
from app.models.db_models import Alert, Farm, Prediction, WeatherObservation
from app.schemas import WeatherForecastResponse
from app.services import prediction_service, weather_service

NOW = datetime.now(timezone.utc)


class BackendApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = MagicMock()
        self.db.refresh.side_effect = self.refresh_record
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        app.dependency_overrides.clear()

    @staticmethod
    def refresh_record(record):
        if record.id is None:
            record.id = 7
        if isinstance(record, Farm):
            record.created_at = NOW
        elif isinstance(record, WeatherObservation):
            record.observed_at = record.observed_at or NOW
            record.created_at = NOW
        elif isinstance(record, Prediction):
            record.predicted_at = NOW
            record.created_at = NOW
        elif isinstance(record, Alert):
            record.created_at = NOW

    @staticmethod
    def make_farm():
        return Farm(
            id=7,
            name="Test farm",
            latitude=17.4,
            longitude=78.5,
            crop_type="tomato",
            location_description=None,
            created_at=NOW,
        )

    async def test_health_and_farm_validation(self):
        self.assertEqual((await self.client.get("/health")).json(), {"status": "healthy"})
        self.assertIn("/api/predictions/crop-risk", app.openapi()["paths"])

        self.db.refresh.side_effect = self.refresh_record
        created = await self.client.post(
            "/api/farms",
            json={"name": "Test farm", "latitude": 17.4, "longitude": 78.5},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["id"], 7)

        self.db.scalars.return_value.all.return_value = [self.make_farm()]
        listed = await self.client.get("/api/farms")
        self.assertEqual(len(listed.json()), 1)
        self.db.get.return_value = self.make_farm()
        retrieved = await self.client.get("/api/farms/7")
        self.assertEqual(retrieved.status_code, 200)

        invalid = await self.client.post(
            "/api/farms",
            json={"name": "Bad coordinates", "latitude": 91, "longitude": 0},
        )
        self.assertEqual(invalid.status_code, 422)

        self.db.get.return_value = None
        missing = await self.client.get("/api/farms/999")
        self.assertEqual(missing.status_code, 404)

    async def test_observation_validation_and_persistence_shape(self):
        self.db.get.return_value = self.make_farm()
        invalid = await self.client.post(
            "/api/weather/observations/7", json={"source": "station"}
        )
        self.assertEqual(invalid.status_code, 422)

        created = await self.client.post(
            "/api/weather/observations/7",
            json={"temperature_c": 24.5, "humidity_percent": 62, "source": "station"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["source"], "station")

        self.db.scalars.return_value.all.return_value = [
            WeatherObservation(
                id=11,
                farm_id=7,
                temperature_c=24.5,
                humidity_percent=62,
                source="station",
                observed_at=NOW,
                created_at=NOW,
            )
        ]
        listed = await self.client.get("/api/weather/observations", params={"farm_id": 7})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["farm_id"], 7)

    async def test_forecast_route_maps_provider_errors(self):
        invalid = await self.client.get(
            "/api/weather/forecast", params={"latitude": 91, "longitude": 78}
        )
        self.assertEqual(invalid.status_code, 422)

        with patch(
            "app.routers.weather.weather_service.fetch_forecast",
            new=AsyncMock(side_effect=weather_service.ForecastTimeoutError("timeout")),
        ):
            timeout = await self.client.get(
                "/api/weather/forecast", params={"latitude": 17.4, "longitude": 78.5}
            )
        self.assertEqual(timeout.status_code, 504)

    async def test_farm_forecast_uses_saved_coordinates(self):
        self.db.get.return_value = self.make_farm()
        forecast = WeatherForecastResponse.model_validate(
            {
                "latitude": 17.4,
                "longitude": 78.5,
                "timezone": "UTC",
                "utc_offset_seconds": 0,
                "hourly_units": {},
                "hourly": {
                    "time": [],
                    "temperature_2m": [],
                    "relative_humidity_2m": [],
                    "precipitation": [],
                    "wind_speed_10m": [],
                },
            }
        )
        with patch(
            "app.routers.weather.weather_service.fetch_forecast",
            new=AsyncMock(return_value=forecast),
        ) as fetch:
            response = await self.client.get("/api/weather/farms/7/forecast")
        self.assertEqual(response.status_code, 200)
        fetch.assert_awaited_once_with(17.4, 78.5)

    async def test_prediction_unavailable_and_successful_model(self):
        self.db.get.return_value = self.make_farm()
        self.db.scalar.return_value = None
        with patch(
            "app.routers.predictions.prediction_service.predict_microclimate",
            side_effect=prediction_service.PredictionUnavailableError("model unavailable"),
        ):
            unavailable = await self.client.post(
                "/api/predictions/microclimate",
                json={"farm_id": 7, "weather": {"temperature_c": 24}},
            )
        self.assertEqual(unavailable.status_code, 503)
        self.db.commit.assert_not_called()

        self.db.refresh.side_effect = self.refresh_record
        with patch(
            "app.routers.predictions.prediction_service.predict_microclimate",
            return_value=({"temperature_c": 25}, "member-1-model"),
        ):
            response = await self.client.post(
                "/api/predictions/microclimate",
                json={"farm_id": 7, "features": {"elevation_m": 500}},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "member-1-model")

    async def test_crop_risk_prediction_uses_model_result_for_alert(self):
        self.db.get.return_value = self.make_farm()
        self.db.scalar.return_value = None
        model_result = {
            "risk_level": "high",
            "explanation": "Model-supported explanation.",
            "recommended_actions": ["Inspect affected plants."],
        }
        with (
            patch(
                "app.routers.predictions.prediction_service.predict_crop_risk",
                return_value=(model_result, "member-3-model", "high"),
            ),
            patch("app.routers.predictions.alert_service.create_crop_risk_alert") as create_alert,
        ):
            response = await self.client.post(
                "/api/predictions/crop-risk",
                json={"farm_id": 7, "weather": {"humidity_percent": 90}},
            )
        self.assertEqual(response.status_code, 200)
        create_alert.assert_called_once_with(self.db, 7, model_result, "high")

    async def test_get_latest_prediction(self):
        self.db.get.return_value = self.make_farm()
        self.db.scalar.return_value = Prediction(
            id=9,
            farm_id=7,
            prediction_type="microclimate",
            predicted_value={"temperature_c": 25},
            risk_level=None,
            model_name="member-1-model",
            predicted_at=NOW,
            created_at=NOW,
        )
        response = await self.client.get(
            "/api/predictions/microclimate", params={"farm_id": 7}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction_id"], 9)
        self.assertEqual(response.json()["result"], {"temperature_c": 25})

    async def test_alert_deduplication_and_acknowledgement(self):
        self.db.get.return_value = self.make_farm()
        alert = Alert(
            id=3,
            farm_id=7,
            title="High crop risk",
            message="Model reported risk.",
            severity="high",
            status="open",
            created_at=NOW,
        )
        self.db.scalar.return_value = alert
        duplicate = await self.client.post(
            "/api/alerts",
            json={
                "farm_id": 7,
                "title": "High crop risk",
                "message": "Model reported risk.",
                "severity": "high",
            },
        )
        self.assertEqual(duplicate.status_code, 200)
        self.db.add.assert_not_called()

        self.db.get.return_value = alert
        acknowledged = await self.client.patch(
            "/api/alerts/3", json={"status": "acknowledged"}
        )
        self.assertEqual(acknowledged.status_code, 200)
        self.assertIsNotNone(acknowledged.json()["acknowledged_at"])

    async def test_dashboard_empty_state(self):
        self.db.scalars.return_value.all.return_value = []
        response = await self.client.get("/api/dashboard/summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["farms"], [])
        self.assertEqual(response.json()["total_farms"], 0)

    async def test_dashboard_includes_latest_records_and_active_alerts(self):
        farm = self.make_farm()
        observation = WeatherObservation(
            id=2,
            farm_id=7,
            temperature_c=24.5,
            observed_at=NOW,
            created_at=NOW,
        )
        prediction = Prediction(
            id=3,
            farm_id=7,
            prediction_type="crop-risk",
            predicted_value={"risk_level": "high"},
            risk_level="high",
            model_name="member-3-model",
            predicted_at=NOW,
            created_at=NOW,
        )
        alert = Alert(
            id=4,
            farm_id=7,
            title="High crop risk",
            message="Risk reported by model.",
            severity="high",
            status="open",
            created_at=NOW,
        )
        query_results = ([farm], [observation], [prediction], [alert])
        self.db.scalars.side_effect = [
            MagicMock(all=MagicMock(return_value=records))
            for records in query_results
        ]
        response = await self.client.get("/api/dashboard/summary")
        self.assertEqual(response.status_code, 200)
        summary = response.json()
        self.assertEqual(summary["farms"][0]["latest_observation"]["temperature_c"], 24.5)
        self.assertEqual(summary["farms"][0]["latest_crop_risk_prediction"]["risk_level"], "high")
        self.assertEqual(len(summary["farms"][0]["active_alerts"]), 1)


class OpenMeteoServiceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def forecast_payload():
        return {
            "latitude": 17.4,
            "longitude": 78.5,
            "timezone": "Asia/Kolkata",
            "utc_offset_seconds": 19800,
            "hourly_units": {
                "time": "iso8601",
                "temperature_2m": "C",
                "relative_humidity_2m": "%",
                "precipitation": "mm",
                "wind_speed_10m": "km/h",
            },
            "hourly": {
                "time": ["2026-10-13T10:00"],
                "temperature_2m": [24.5],
                "relative_humidity_2m": [62],
                "precipitation": [1.2],
                "wind_speed_10m": [12.0],
            },
        }

    async def test_provider_success_and_invalid_measurements(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=self.forecast_payload())
            )
        ) as client:
            forecast = await weather_service.fetch_forecast(17.4, 78.5, client=client)
        self.assertEqual(forecast.timezone, "Asia/Kolkata")
        self.assertEqual(forecast.hourly.temperature_2m, [24.5])

        invalid_payload = self.forecast_payload()
        invalid_payload["hourly"]["temperature_2m"] = [81]
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=invalid_payload)
            )
        ) as client:
            with self.assertRaises(weather_service.ForecastProviderError):
                await weather_service.fetch_forecast(17.4, 78.5, client=client)

    async def test_provider_failure(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(503))
        ) as client:
            with self.assertRaises(weather_service.ForecastProviderError):
                await weather_service.fetch_forecast(17.4, 78.5, client=client)


if __name__ == "__main__":
    unittest.main()