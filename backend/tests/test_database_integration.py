"""Real PostgreSQL integration tests, restricted to the isolated test database."""

import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.database import DATABASE_URL, get_db
import app.models.db_models  # Register the SQLAlchemy models.
from app.main import app
from app.models.db_models import Alert, Farm, Prediction, WeatherObservation

TEST_DATABASE = "microfarmcast_test_db"
REQUIRED_TABLES = {
    "farms",
    "weather_observations",
    "predictions",
    "alerts",
}


class FarmDatabaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not configured.")

        test_url = make_url(DATABASE_URL).set(database=TEST_DATABASE)
        if test_url.get_backend_name() != "postgresql":
            raise RuntimeError("Refusing to test: DATABASE_URL must use PostgreSQL.")
        if test_url.database != TEST_DATABASE:
            raise RuntimeError("Refusing to test an unexpected database.")
        if test_url.drivername == "postgresql":
            test_url = test_url.set(drivername="postgresql+psycopg2")

        cls.engine = create_engine(test_url, pool_pre_ping=True)
        try:
            with cls.engine.connect() as connection:
                actual_database = connection.execute(
                    text("SELECT current_database()")
                ).scalar_one()
                if actual_database != TEST_DATABASE:
                    raise RuntimeError(
                        "Refusing to inspect or test an unexpected database: "
                        f"{actual_database!r}"
                    )
                public_tables = set(
                    connection.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = 'public'"
                        )
                    ).scalars()
                )
        except Exception:
            cls.engine.dispose()
            raise

        missing_tables = REQUIRED_TABLES - public_tables
        if missing_tables:
            cls.engine.dispose()
            raise unittest.SkipTest(
                f"{TEST_DATABASE} is missing public tables: "
                f"{', '.join(sorted(missing_tables))}. No schema was created."
            )

        cls.SessionLocal = sessionmaker(
            bind=cls.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    def setUp(self):
        self.farm_names: list[str] = []
        self.original_get_db_override = app.dependency_overrides.get(get_db)

        def override_get_db():
            with self.SessionLocal() as db:
                yield db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        try:
            if self.farm_names:
                with self.SessionLocal() as cleanup_db:
                    farm_ids = cleanup_db.scalars(
                        select(Farm.id).where(Farm.name.in_(self.farm_names))
                    ).all()
                    if farm_ids:
                        for model in (Alert, Prediction, WeatherObservation):
                            cleanup_db.execute(
                                delete(model).where(model.farm_id.in_(farm_ids))
                            )
                        cleanup_db.execute(
                            delete(Farm).where(Farm.id.in_(farm_ids))
                        )
                        cleanup_db.commit()
        finally:
            if self.original_get_db_override is None:
                app.dependency_overrides.pop(get_db, None)
            else:
                app.dependency_overrides[get_db] = self.original_get_db_override

    def _create_farm(self) -> tuple[str, int]:
        farm_name = f"Integration Test Farm {uuid4().hex}"
        self.farm_names.append(farm_name)
        response = self.client.post(
            "/api/farms",
            json={
                "name": farm_name,
                "latitude": 17.4,
                "longitude": 78.5,
                "crop_type": "tomato",
                "location_description": "Integration test",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return farm_name, response.json()["id"]

    def test_create_farm_is_persisted_in_postgresql(self):
        farm_name, farm_id = self._create_farm()

        with self.SessionLocal() as verification_db:
            saved_farm = verification_db.get(Farm, farm_id)
            self.assertIsNotNone(saved_farm)
            self.assertEqual(saved_farm.name, farm_name)
            self.assertAlmostEqual(saved_farm.latitude, 17.4)
            self.assertAlmostEqual(saved_farm.longitude, 78.5)
            self.assertEqual(saved_farm.crop_type, "tomato")

    def test_weather_observation_is_persisted(self):
        _, farm_id = self._create_farm()
        response = self.client.post(
            f"/api/farms/{farm_id}/observations",
            json={
                "temperature_c": 24.5,
                "humidity_percent": 62,
                "source": "integration-test",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

        with self.SessionLocal() as verification_db:
            saved_observation = verification_db.get(
                WeatherObservation, response.json()["id"]
            )
            self.assertIsNotNone(saved_observation)
            self.assertEqual(saved_observation.farm_id, farm_id)
            self.assertEqual(saved_observation.temperature_c, 24.5)
            self.assertEqual(saved_observation.humidity_percent, 62)
            self.assertEqual(saved_observation.source, "integration-test")

    def test_prediction_is_persisted(self):
        _, farm_id = self._create_farm()
        response = self.client.post(
            f"/api/farms/{farm_id}/predictions",
            json={
                "prediction_type": "crop-risk",
                "predicted_value": {"risk_level": "high"},
                "risk_level": "high",
                "model_name": "integration-test-model",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

        with self.SessionLocal() as verification_db:
            saved_prediction = verification_db.get(Prediction, response.json()["id"])
            self.assertIsNotNone(saved_prediction)
            self.assertEqual(saved_prediction.farm_id, farm_id)
            self.assertEqual(saved_prediction.predicted_value, {"risk_level": "high"})
            self.assertEqual(saved_prediction.model_name, "integration-test-model")

    def test_alert_is_persisted(self):
        _, farm_id = self._create_farm()
        response = self.client.post(
            "/api/alerts",
            json={
                "farm_id": farm_id,
                "title": "Integration test alert",
                "message": "Persist this test alert.",
                "severity": "high",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

        with self.SessionLocal() as verification_db:
            saved_alert = verification_db.get(Alert, response.json()["id"])
            self.assertIsNotNone(saved_alert)
            self.assertEqual(saved_alert.farm_id, farm_id)
            self.assertEqual(saved_alert.title, "Integration test alert")
            self.assertEqual(saved_alert.severity, "high")
            self.assertEqual(saved_alert.status, "open")


if __name__ == "__main__":
    unittest.main()
