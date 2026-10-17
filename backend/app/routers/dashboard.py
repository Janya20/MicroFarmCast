"""Database-backed dashboard summary endpoint."""

from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Alert, Farm, Prediction, WeatherObservation
from app.schemas import (
    AlertResponse,
    DashboardFarmSummary,
    DashboardSummaryResponse,
    FarmResponse,
    ObservationResponse,
    PredictionResponse,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummaryResponse:
    """Summarize farms and their latest persisted observations, predictions, and alerts."""
    farms = db.scalars(select(Farm).order_by(Farm.id)).all()
    observations = db.scalars(
        select(WeatherObservation).order_by(WeatherObservation.observed_at.desc())
    ).all()
    predictions = db.scalars(
        select(Prediction).order_by(Prediction.predicted_at.desc())
    ).all()
    alerts = db.scalars(select(Alert).order_by(Alert.created_at.desc())).all()

    latest_observation: dict[int, WeatherObservation] = {}
    for observation in observations:
        latest_observation.setdefault(observation.farm_id, observation)

    latest_predictions: dict[tuple[int, str], Prediction] = {}
    for prediction in predictions:
        latest_predictions.setdefault(
            (prediction.farm_id, prediction.prediction_type), prediction
        )

    alerts_by_farm: dict[int, list[Alert]] = {}
    for alert in alerts:
        if alert.status != "closed":
            alerts_by_farm.setdefault(alert.farm_id, []).append(alert)

    farm_summaries = [
        DashboardFarmSummary(
            farm=FarmResponse.model_validate(farm),
            latest_observation=(
                ObservationResponse.model_validate(latest_observation[farm.id])
                if farm.id in latest_observation
                else None
            ),
            latest_microclimate_prediction=(
                PredictionResponse.model_validate(
                    latest_predictions[(farm.id, "microclimate")]
                )
                if (farm.id, "microclimate") in latest_predictions
                else None
            ),
            latest_crop_risk_prediction=(
                PredictionResponse.model_validate(
                    latest_predictions[(farm.id, "crop-risk")]
                )
                if (farm.id, "crop-risk") in latest_predictions
                else None
            ),
            active_alerts=[
                AlertResponse.model_validate(alert)
                for alert in alerts_by_farm.get(farm.id, [])
            ],
        )
        for farm in farms
    ]

    return DashboardSummaryResponse(
        farms=farm_summaries,
        total_farms=len(farms),
        total_observations=len(observations),
        total_predictions=len(predictions),
        total_alerts=len(alerts),
        predictions_by_risk_level=dict(
            Counter(
                prediction.risk_level
                for prediction in predictions
                if prediction.risk_level is not None
            )
        ),
        alerts_by_severity=dict(
            Counter(
                alert.severity
                for alert in alerts
                if alert.status != "closed" and alert.severity is not None
            )
        ),
    )