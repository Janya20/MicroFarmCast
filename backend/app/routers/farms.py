
"""Database-backed farm, observation, and prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Farm, WeatherObservation, Prediction
from app.schemas import (
    AlertResponse,
    FarmCreate,
    FarmResponse,
    ObservationCreate,
    ObservationResponse,
    PredictionCreate,
    PredictionResponse,
)

router = APIRouter(prefix="/api/farms", tags=["farms"])


@router.get("", response_model=list[FarmResponse])
@router.get("/", response_model=list[FarmResponse], include_in_schema=False)
def get_farms(db: Session = Depends(get_db)):
    return db.scalars(select(Farm).order_by(Farm.id)).all()


@router.get("/{farm_id}", response_model=FarmResponse)
def get_farm(
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    farm = db.get(Farm, farm_id)
    if farm is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
    return farm


@router.get("/{farm_id}/observations", response_model=list[ObservationResponse])
def get_farm_observations(
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")

    statement = (
        select(WeatherObservation)
        .where(WeatherObservation.farm_id == farm_id)
        .order_by(WeatherObservation.observed_at.desc())
    )
    return db.scalars(statement).all()


@router.get("/{farm_id}/predictions", response_model=list[PredictionResponse])
def get_farm_predictions(
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")

    statement = (
        select(Prediction)
        .where(Prediction.farm_id == farm_id)
        .order_by(Prediction.predicted_at.desc())
    )
    return db.scalars(statement).all()


@router.get("/{farm_id}/alerts", response_model=list[AlertResponse])
def get_farm_alerts(
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    # Import locally to avoid coupling this router to other model files.
    from app.models.db_models import Alert

    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")

    statement = (
        select(Alert)
        .where(Alert.farm_id == farm_id)
        .order_by(Alert.created_at.desc())
    )
    return db.scalars(statement).all()


@router.post("", response_model=FarmResponse, status_code=status.HTTP_201_CREATED)
def create_farm(
    farm: FarmCreate,
    db: Session = Depends(get_db),
):
    record = Farm(**farm.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post(
    "/{farm_id}/observations",
    response_model=ObservationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_observation(
    observation: ObservationCreate,
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
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


@router.post(
    "/{farm_id}/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_prediction(
    prediction: PredictionCreate,
    farm_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    if db.get(Farm, farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")

    record = Prediction(
        farm_id=farm_id,
        **prediction.model_dump(exclude_none=True),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
