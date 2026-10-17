"""Alert rules derived from real crop-risk model output."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import Alert

ALERT_RISK_LEVELS = {"high", "critical"}


def create_crop_risk_alert(
    db: Session,
    farm_id: int,
    result: dict[str, Any],
    risk_level: str,
) -> Alert | None:
    """Create one open alert for a high-risk model result, if none already exists."""
    if risk_level not in ALERT_RISK_LEVELS:
        return None

    title = f"{risk_level.title()} crop risk"
    message_parts = [f"The crop-risk model reported {risk_level} risk."]
    explanation = result.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        message_parts.append(f"Explanation: {explanation.strip()}")

    actions = result.get("recommended_actions")
    if isinstance(actions, list):
        supported_actions = [
            action.strip()
            for action in actions
            if isinstance(action, str) and action.strip()
        ]
        if supported_actions:
            message_parts.append(
                "Recommended actions: " + "; ".join(supported_actions)
            )
    message = " ".join(message_parts)

    existing = db.scalar(
        select(Alert)
        .where(
            Alert.farm_id == farm_id,
            Alert.title == title,
            Alert.status == "open",
        )
        .limit(1)
    )
    if existing is not None:
        return existing

    alert = Alert(
        farm_id=farm_id,
        title=title,
        message=message,
        severity=risk_level,
        status="open",
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert