"""Integration test against the real model: pytest -m slow"""

import pytest

pytestmark = pytest.mark.slow

LABELS = ["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"]


@pytest.fixture(scope="module")
def real_client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


def test_health_reports_loaded_model(real_client):
    assert real_client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_german_example_picks_technical_problem(real_client):
    from tests.conftest import AUTH

    response = real_client.post(
        "/classify",
        json={
            "text": "Mein Kassensystem druckt seit gestern keine Bons mehr",
            "labels": LABELS,
        },
        headers=AUTH,
    )
    assert response.status_code == 200

    body = response.json()
    assert body["label"] == "Technisches Problem"
    assert body["score"] > 0.5
    assert [item["score"] for item in body["results"]] == sorted(
        (item["score"] for item in body["results"]), reverse=True
    )
    assert body["duration_ms"] > 0


def test_german_invoice_example(real_client):
    from tests.conftest import AUTH

    response = real_client.post(
        "/classify",
        json={
            "text": "Auf der letzten Rechnung wurde mir ein falscher Betrag berechnet.",
            "labels": LABELS,
        },
        headers=AUTH,
    )
    assert response.json()["label"] == "Rechnung"


def test_english_example(real_client):
    from tests.conftest import AUTH

    response = real_client.post(
        "/classify",
        json={
            "text": "Could you please add a dark mode to the dashboard?",
            "labels": LABELS,
            "hypothesis_template": "This message is about {}.",
        },
        headers=AUTH,
    )
    assert response.json()["label"] == "Feature-Wunsch"


def test_spam_example(real_client):
    from tests.conftest import AUTH

    response = real_client.post(
        "/classify",
        json={
            "text": (
                "GRATIS BITCOIN!!! Klicken Sie jetzt hier und verdienen Sie "
                "5000 Euro pro Tag"
            ),
            "labels": LABELS,
        },
        headers=AUTH,
    )
    assert response.json()["label"] == "Spam"


def test_batch_with_real_model(real_client):
    from tests.conftest import AUTH

    response = real_client.post(
        "/classify/batch",
        json={
            "items": [
                {"text": "Der Drucker streikt komplett.", "labels": LABELS},
                {"text": "Bitte schickt mir die Rechnung nochmal.", "labels": LABELS},
            ]
        },
        headers=AUTH,
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["label"] for r in results] == ["Technisches Problem", "Rechnung"]
