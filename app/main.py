"""
FastAPI application for MP291 parcel delivery days prediction.
Exposes POST /predict and GET /health with JSON logging and strict validation.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from mp291.clean import clean_dataframe


class JSONFormatter(logging.Formatter):
    """Custom logging formatter outputting single-line JSON to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_dict.update(record.props)
        return json.dumps(log_dict)


# Configure JSON logger to stdout
logger = logging.getLogger("mp291.api")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)


class PredictionInput(BaseModel):
    """
    Input schema for POST /predict.
    Fields defined in PROBLEM.md:
    distance_km, weight_kg, hub_backlog, pin_code_tier, declared_value_rs, transfers, service, destination.
    """
    distance_km: float = Field(..., description="Distance to destination in km")
    weight_kg: float = Field(..., description="Parcel weight in kg")
    hub_backlog: int = Field(
        ...,
        ge=0,
        description="Parcels waiting at the hub; below 0 is a typing mistake and must be rejected."
    )
    pin_code_tier: int = Field(..., description="Delivery-area tier")
    declared_value_rs: float = Field(
        ...,
        le=35000,
        description="Declared value in rupees; above 35000 is impossible and must be rejected."
    )
    transfers: Union[int, float] = Field(..., description="Hub-to-hub transfers count")
    service: str = Field(..., description="Service chosen (standard, express, economy)")
    destination: str = Field(..., description="Type of destination (Metro, Tier-2, Rural)")


class PredictionOutput(BaseModel):
    """Output schema for POST /predict."""
    prediction: float


class HealthOutput(BaseModel):
    """Output schema for GET /health."""
    status: str


# Global model reference
model = None


def load_model(model_path: Optional[str] = None):
    """Load the trained champion model from file."""
    path = model_path or os.getenv("MODEL_PATH", "models/model.joblib")
    if os.path.exists(path):
        loaded = joblib.load(path)
        logger.info(f"Model successfully loaded from {path}", extra={"props": {"event": "model_loaded", "path": path}})
        return loaded
    logger.warning(f"Model artifact not found at {path}", extra={"props": {"event": "model_missing", "path": path}})
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global model
    model = load_model()
    yield


app = FastAPI(
    title="MP291 Parcel Delivery Prediction API",
    description="Dehradun Speed Parcel Hub Delivery Days Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput)
def health():
    """Health check endpoint required by probes and grading tool."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    """
    Predict delivery_days given parcel and routing features.
    Enforces client validation rules and logs inferences in structured JSON.
    """
    global model
    if model is None:
        model = load_model()

    if model is None:
        logger.error("Model unavailable for prediction", extra={"props": {"event": "error", "reason": "model_not_found"}})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently loaded"
        )

    # Convert request payload to DataFrame
    data_dict = payload.model_dump()
    raw_df = pd.DataFrame([data_dict])

    # Clean the input (sentinels, casing/spelling)
    cleaned_df = clean_dataframe(raw_df, is_train=False)

    feature_cols = [
        "distance_km",
        "weight_kg",
        "hub_backlog",
        "pin_code_tier",
        "declared_value_rs",
        "transfers",
        "service",
        "destination",
    ]
    features_df = cleaned_df[feature_cols]

    # Model inference
    try:
        pred = model.predict(features_df)
        pred_value = round(float(pred[0]), 4)
    except Exception as e:
        logger.error(
            f"Prediction failed: {str(e)}",
            extra={"props": {"event": "prediction_error", "error": str(e)}}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    # Structured JSON log per prediction without PII
    logger.info(
        "Prediction completed",
        extra={
            "props": {
                "event": "prediction",
                "service": payload.service,
                "destination": payload.destination,
                "prediction": pred_value,
            }
        }
    )

    return {"prediction": pred_value}
