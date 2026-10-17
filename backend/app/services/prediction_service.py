"""Adapter boundary for teammate-owned microclimate and crop-risk models."""

import json
from collections.abc import Callable
from typing import Any, Protocol


class PredictionModel(Protocol):
    def __call__(self, model_input: dict[str, Any]) -> dict[str, Any]: ...


class PredictionUnavailableError(Exception):
    """Raised when a teammate has not registered the required model."""


class PredictionInputError(Exception):
    """Raised when there is not enough available input for prediction."""


class PredictionExecutionError(Exception):
    """Raised when a registered model fails or returns invalid output."""


microclimate_model: PredictionModel | None = None
microclimate_model_name: str | None = None
crop_risk_model: PredictionModel | None = None
crop_risk_model_name: str | None = None


def register_microclimate_model(model: PredictionModel, model_name: str) -> None:
    """Register Member 1's callable model and its identifying name."""
    global microclimate_model, microclimate_model_name
    microclimate_model = model
    microclimate_model_name = model_name


def register_crop_risk_model(model: PredictionModel, model_name: str) -> None:
    """Register Member 3's callable model and its identifying name."""
    global crop_risk_model, crop_risk_model_name
    crop_risk_model = model
    crop_risk_model_name = model_name


def predict_microclimate(
    model_input: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    result, model_name = _execute(
        microclimate_model,
        microclimate_model_name,
        model_input,
        "microclimate",
    )
    return result, model_name


def predict_crop_risk(
    model_input: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    result, model_name = _execute(
        crop_risk_model,
        crop_risk_model_name,
        model_input,
        "crop-risk",
    )
    risk_level = result.get("risk_level")
    if risk_level not in {"low", "medium", "high", "critical"}:
        raise PredictionExecutionError(
            "The crop-risk model must return a supported risk_level."
        )
    return result, model_name, risk_level


def _execute(
    model: PredictionModel | None,
    model_name: str | None,
    model_input: dict[str, Any],
    prediction_type: str,
) -> tuple[dict[str, Any], str]:
    if model is None or not model_name:
        raise PredictionUnavailableError(
            f"The {prediction_type} model is not registered."
        )
    if not model_input.get("weather") and not model_input.get("features"):
        raise PredictionInputError(
            f"Weather data or model features are required for {prediction_type}."
        )

    try:
        result = model(model_input)
        if not isinstance(result, dict) or not result:
            raise ValueError("Model output must be a non-empty object.")
        normalized = json.loads(json.dumps(result, allow_nan=False))
    except Exception as error:
        raise PredictionExecutionError(
            f"The {prediction_type} model failed or returned invalid output."
        ) from error

    return normalized, model_name
