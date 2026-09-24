"""
Tests for the prediction API: health check, valid prediction,
input validation, and unknown-category handling.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

VALID_SESSION = {
    "Administrative": 1, "Administrative_Duration": 14.65,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 19, "ProductRelated_Duration": 283.88,
    "BounceRates": 0.008, "ExitRates": 0.042, "PageValues": 68.58,
    "SpecialDay": 0.0, "Month": "May", "OperatingSystems": 1,
    "Browser": 1, "Region": 1, "TrafficType": 1,
    "VisitorType": "Returning_Visitor", "Weekend": False
}


@pytest.fixture
def client():
    """Runs the app's lifespan (loading the model) for each test."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_session(client):
    response = client.post("/predict", json=VALID_SESSION)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Purchase", "No Purchase"]
    assert 0.0 <= data["purchase_probability"] <= 1.0
    assert data["confidence"] in ["High", "Medium", "Low"]
    assert len(data["top_contributing_features"]) == 5


def test_predict_invalid_bounce_rate(client):
    invalid_session = {**VALID_SESSION, "BounceRates": 1.5}
    response = client.post("/predict", json=invalid_session)
    assert response.status_code == 422


def test_predict_unknown_category(client):
    unknown_month_session = {**VALID_SESSION, "Month": "Zztober"}
    response = client.post("/predict", json=unknown_month_session)
    assert response.status_code == 200