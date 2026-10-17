"""Temporary process-local storage for demonstrating create endpoints.

Records in this module are lost whenever the application process restarts.
This is not a database and must not be used for permanent data storage.
"""

from typing import Any
from collections import Counter

Record = dict[str, Any]

_farms: dict[int, Record] = {}
_observations: list[Record] = []
_predictions: list[Record] = []
_alerts: list[Record] = []


def clear() -> None:
    """Clear temporary records, including between isolated tests."""
    _farms.clear()
    _observations.clear()
    _predictions.clear()
    _alerts.clear()


def list_farms() -> list[Record]:
    """Return copies of all farms in insertion order."""
    return [farm.copy() for farm in _farms.values()]


def get_farm(farm_id: int) -> Record | None:
    """Return a copy of one farm, if it exists."""
    farm = _farms.get(farm_id)
    return farm.copy() if farm is not None else None


def _list_farm_records(collection: list[Record], farm_id: int) -> list[Record]:
    """Return copies of records associated with one farm."""
    return [record.copy() for record in collection if record["farm_id"] == farm_id]


def list_observations(farm_id: int) -> list[Record]:
    """Return copies of a farm's observations."""
    return _list_farm_records(_observations, farm_id)


def list_predictions(farm_id: int) -> list[Record]:
    """Return copies of a farm's predictions."""
    return _list_farm_records(_predictions, farm_id)


def list_alerts(
    farm_id: int | None = None,
    severity: str | None = None,
) -> list[Record]:
    """Return copies of alerts, optionally filtered by stored fields."""
    return [
        alert.copy()
        for alert in _alerts
        if (farm_id is None or alert["farm_id"] == farm_id)
        and (severity is None or alert["severity"] == severity)
    ]


def dashboard_summary() -> Record:
    """Count current records and their stored risk/severity values."""
    risk_levels = Counter(
        prediction["risk_level"]
        for prediction in _predictions
        if prediction.get("risk_level") is not None
    )
    alert_severities = Counter(alert["severity"] for alert in _alerts)
    return {
        "total_farms": len(_farms),
        "total_observations": len(_observations),
        "total_predictions": len(_predictions),
        "total_alerts": len(_alerts),
        "predictions_by_risk_level": dict(risk_levels),
        "alerts_by_severity": dict(alert_severities),
    }


def add_farm(data: Record) -> Record:
    """Store a farm temporarily and return it with its generated ID."""
    farm_id = len(_farms) + 1
    farm = {"id": farm_id, **data}
    _farms[farm_id] = farm
    return farm


def farm_exists(farm_id: int) -> bool:
    """Check whether a farm exists in this running process."""
    return farm_id in _farms


def _add_related_record(collection: list[Record], farm_id: int, data: Record) -> Record:
    """Store a farm observation or prediction temporarily."""
    record = {"id": len(collection) + 1, "farm_id": farm_id, **data}
    collection.append(record)
    return record


def add_observation(farm_id: int, data: Record) -> Record:
    """Store a weather or sensor observation temporarily."""
    return _add_related_record(_observations, farm_id, data)


def add_prediction(farm_id: int, data: Record) -> Record:
    """Store a prediction temporarily."""
    return _add_related_record(_predictions, farm_id, data)


def add_alert(data: Record) -> Record:
    """Store an alert temporarily and return it with its generated ID."""
    alert = {"id": len(_alerts) + 1, **data}
    _alerts.append(alert)
    return alert