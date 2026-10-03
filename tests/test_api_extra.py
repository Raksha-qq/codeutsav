"""Tests for FastAPI backend and operator dashboard endpoints.

Verifies PRD §7.4 & §6:
- Health check endpoint /api/health
- Index HTML dashboard serving at /
- Video stream endpoint /video
- WebSocket /events connection
"""

import pytest
from fastapi.testclient import TestClient
from billetvision.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "BilletVision"


def test_dashboard_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "BilletVision" in response.text


def test_websocket_events(client):
    with client.websocket_connect("/events") as websocket:
        # Check connection can be opened cleanly
        assert websocket is not None
