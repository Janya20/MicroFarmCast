"""API endpoints for teammate-provided prediction models."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Farm, Prediction, WeatherObservation
from app.schemas import (
    CropRiskPredictionRequest,
    MicroclimatePredictionRequest,
    PredictionResultResponse,
)
from app.services import alert_service, prediction_service

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def _farm_model_input(
    farm: Farm,
    weather: dict,
    features: dict,
    db: Session,
) -> dict:
    if not weather:
        observation = db.scalar(
            select(WeatherObservation)
            .where(WeatherObservation.farm_id == farm.id)
            .order_by(WeatherObservation.observed_at.desc())
            .limit(1)
        )
        if observation is not None:
            weather = {
                key: value
                for key, value in {
                    "temperature_c": observation.temperature_c,
                    "humidity_percent": observation.humidity_percent,
                    "soil_moisture": observation.soil_moisture,
                    "precipitation_mm": observation.precipitation_mm,
                    "wind_speed_mps": observation.wind_speed_mps,
                }.items()
                if value is not None
            }

    return {
        "farm": {
            "id": farm.id,
            "latitude": farm.latitude,
            "longitude": farm.longitude,
            "crop_type": farm.crop_type,
        },
        "weather": weather,
        "features": features,
    }


def _save_prediction(
    farm_id: int,
    prediction_type: str,
    result: dict,
    model_name: str,
    risk_level: str | None,
    db: Session,
) -> PredictionResultResponse:
    record = Prediction(
        farm_id=farm_id,
        prediction_type=prediction_type,
        predicted_value=result,
        risk_level=risk_level,
        model_name=model_name,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return PredictionResultResponse(
        prediction_id=record.id,
        farm_id=record.farm_id,
        prediction_type=prediction_type,
        result=record.predicted_value,
        risk_level=record.risk_level,
        model_name=model_name,
        predicted_at=record.predicted_at,
    )


def _get_latest_prediction(
    farm_id: int,
    prediction_type: str,
    db: Session,
) -> PredictionResultResponse:
    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
    record = db.scalar(
        select(Prediction)
        .where(
            Prediction.farm_id == farm_id,
            Prediction.prediction_type == prediction_type,
        )
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {prediction_type} prediction was found for farm {farm_id}.",
        )
    return PredictionResultResponse(
        prediction_id=record.id,
        farm_id=record.farm_id,
        prediction_type=prediction_type,
        result=record.predicted_value,
        risk_level=record.risk_level,
        model_name=record.model_name or "unknown",
        predicted_at=record.predicted_at,
    )


def _raise_prediction_http_error(error: Exception) -> None:
    if isinstance(error, prediction_service.PredictionUnavailableError):
        raise HTTPException(status_code=503, detail=str(error)) from error
    if isinstance(error, prediction_service.PredictionInputError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    if isinstance(error, prediction_service.PredictionExecutionError):
        raise HTTPException(status_code=502, detail=str(error)) from error
    raise error


@router.get("/microclimate", response_model=PredictionResultResponse)
def get_microclimate_prediction(
    farm_id: int = Query(gt=0),
    db: Session = Depends(get_db),
):
    """Return the latest saved microclimate prediction for a farm."""
    return _get_latest_prediction(farm_id, "microclimate", db)


@router.get("/crop-risk", response_model=PredictionResultResponse)
def get_crop_risk_prediction(
    farm_id: int = Query(gt=0),
    db: Session = Depends(get_db),
):
    """Return the latest saved crop-risk prediction for a farm."""
    return _get_latest_prediction(farm_id, "crop-risk", db)


@router.post("/microclimate", response_model=PredictionResultResponse)
def predict_microclimate(
    request: MicroclimatePredictionRequest,
    db: Session = Depends(get_db),
):
    farm = db.get(Farm, request.farm_id)
    if farm is None:
        raise HTTPException(status_code=404, detail=f"Farm {request.farm_id} was not found.")

    model_input = _farm_model_input(farm, request.weather, request.features, db)
    try:
        result, model_name = prediction_service.predict_microclimate(model_input)
    except (
        prediction_service.PredictionUnavailableError,
        prediction_service.PredictionInputError,
        prediction_service.PredictionExecutionError,
    ) as error:
        _raise_prediction_http_error(error)

    return _save_prediction(
        farm.id, "microclimate", result, model_name, None, db
    )


@router.post("/crop-risk", response_model=PredictionResultResponse)
def predict_crop_risk(
    request: CropRiskPredictionRequest,
    db: Session = Depends(get_db),
):
    farm = db.get(Farm, request.farm_id)
    if farm is None:
        raise HTTPException(status_code=404, detail=f"Farm {request.farm_id} was not found.")

    model_input = _farm_model_input(farm, request.weather, request.features, db)
    if request.crop_type:
        model_input["farm"]["crop_type"] = request.crop_type
    try:
        result, model_name, risk_level = prediction_service.predict_crop_risk(model_input)
    except (
        prediction_service.PredictionUnavailableError,
        prediction_service.PredictionInputError,
        prediction_service.PredictionExecutionError,
    ) as error:
        _raise_prediction_http_error(error)

    response = _save_prediction(
        farm.id, "crop-risk", result, model_name, risk_level, db
    )
    alert_service.create_crop_risk_alert(db, farm.id, result, risk_level)
    return response