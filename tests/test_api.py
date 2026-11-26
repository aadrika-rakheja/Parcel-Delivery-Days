"""
Tests for FastAPI service endpoints and client schema rules using TestClient.
"""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, load_model
import app.main as api_module


@pytest.fixture(autouse=True)
def ensure_model():
    """Ensure a working model is loaded in the app module for testing."""
    if api_module.model is None:
        loaded = load_model()
        if loaded is not None:
            api_module.model = loaded
        else:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([4.25])
            api_module.model = mock_model
    yield


@pytest.fixture
def client():
    """Provide FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Standard valid payload satisfying client schema."""
    return {
        "distance_km": 350.5,
        "weight_kg": 3.2,
        "hub_backlog": 15,
        "pin_code_tier": 2,
        "declared_value_rs": 4500.0,
        "transfers": 1,
        "service": "standard",
        "destination": "Tier-2",
    }


def test_health_endpoint(client):
    """GET /health must return status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(client, valid_payload):
    """POST /predict with valid payload returns 200 and numeric prediction."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0


def test_rule_reject_declared_value_above_35000(client, valid_payload):
    """Rule 1: declared_value_rs above 35000 is impossible and must be rejected with 422."""
    bad_payload = valid_payload.copy()
    bad_payload["declared_value_rs"] = 35001.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_rule_accept_boundary_declared_value_35000(client, valid_payload):
    """declared_value_rs exactly 35000 is acceptable (le=35000)."""
    boundary_payload = valid_payload.copy()
    boundary_payload["declared_value_rs"] = 35000.0
    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200


def test_rule_reject_negative_hub_backlog(client, valid_payload):
    """Rule 2: hub_backlog below 0 is always a typing mistake and must be rejected with 422."""
    bad_payload = valid_payload.copy()
    bad_payload["hub_backlog"] = -1
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_required_field(client, valid_payload):
    """Request missing any required field must get 422."""
    incomplete_payload = valid_payload.copy()
    del incomplete_payload["service"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_json_logging_format(client, valid_payload, caplog):
    """Verify that predictions emit structured JSON log conforming to client rules."""
    import logging
    from app.main import JSONFormatter

    with caplog.at_level(logging.INFO, logger="mp291.api"):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        # Verify structured log record exists
        pred_records = [
            r for r in caplog.records
            if r.name == "mp291.api" and getattr(r, "props", {}).get("event") == "prediction"
        ]
        assert len(pred_records) >= 1
        record = pred_records[-1]

        # Verify JSONFormatter outputs valid JSON with required metadata
        formatter = JSONFormatter()
        formatted_str = formatter.format(record)
        parsed = json.loads(formatted_str)

        assert parsed.get("level") == "INFO"
        assert parsed.get("event") == "prediction"
        assert parsed.get("service") == valid_payload["service"]
        assert parsed.get("destination") == valid_payload["destination"]
        assert "prediction" in parsed
        assert "timestamp" in parsed
