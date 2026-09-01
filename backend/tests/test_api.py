from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "sample_digit.png"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200_with_valid_image(client):
    with FIXTURE_IMAGE.open("rb") as f:
        response = client.post("/predict", files={"file": ("sample.png", f, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-3)


def test_predict_rejects_non_image_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"\x00\x01\x02not-a-real-png", "image/png")},
    )

    assert response.status_code == 400


def test_predict_requires_a_file(client):
    response = client.post("/predict")

    assert response.status_code == 422
