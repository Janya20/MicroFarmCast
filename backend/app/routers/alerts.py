"""Endpoints for listing and updating persisted risk alerts."""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Alert, Farm
from app.schemas import AlertCreate, AlertResponse, AlertStatusUpdate

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertResponse])
@router.get("/", response_model=list[AlertResponse], include_in_schema=False)
def get_alerts(
    farm_id: int | None = Query(default=None, gt=0),
    severity: Literal["low", "medium", "high", "critical"] | None = None,
    alert_status: Literal["open", "acknowledged", "closed"] | None = Query(
        default=None, alias="status"
    ),
    db: Session = Depends(get_db),
):
    """Return persisted alerts with optional farm, severity, and status filters."""
    statement = select(Alert).order_by(Alert.created_at.desc())
    if farm_id is not None:
        if db.get(Farm, farm_id) is None:
            raise HTTPException(status_code=404, detail=f"Farm {farm_id} was not found.")
        statement = statement.where(Alert.farm_id == farm_id)
    if severity is not None:
        statement = statement.where(Alert.severity == severity)
    if alert_status is not None:
        statement = statement.where(Alert.status == alert_status)
    return db.scalars(statement).all()


@router.post(
    "",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"description": "The alert's farm was not found."}},
)
def create_alert(
    alert: AlertCreate,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create an alert, returning an existing matching open alert if present."""
    if db.get(Farm, alert.farm_id) is None:
        raise HTTPException(status_code=404, detail=f"Farm {alert.farm_id} was not found.")

    if alert.status in {"open", "acknowledged"}:
        existing = db.scalar(
            select(Alert)
            .where(
                Alert.farm_id == alert.farm_id,
                Alert.title == alert.title,
                Alert.message == alert.message,
                Alert.severity == alert.severity,
                Alert.status.in_(("open", "acknowledged")),
            )
            .order_by(Alert.created_at.desc())
            .limit(1)
        )
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return existing

    values = alert.model_dump(exclude_none=True)
    if values["status"] == "acknowledged" and "acknowledged_at" not in values:
        values["acknowledged_at"] = datetime.now(timezone.utc)
    record = Alert(**values)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.patch("/{alert_id}", response_model=AlertResponse)
def update_alert_status(
    update: AlertStatusUpdate,
    alert_id: int = Path(gt=0),
    db: Session = Depends(get_db),
):
    """Acknowledge or close an alert."""
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} was not found.")
    alert.status = update.status
    if update.status == "acknowledged":
        alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert