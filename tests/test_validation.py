"""Validation and auth - every failure comes back as {"error": "..."}."""

import pytest

from tests.conftest import API_KEY, AUTH, SECOND_API_KEY

BASE = {
    "text": "Mein Kassensystem druckt seit gestern keine Bons mehr",
    "labels": ["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"],
}


def _post(client, **overrides):
    payload = {**BASE, **overrides}
    return client.post("/classify", json=payload, headers=AUTH)


def test_missing_labels(client):
    response = client.post("/classify", json={"text": BASE["text"]}, headers=AUTH)
    assert response.status_code == 422
    assert "labels" in response.json()["error"]


def test_too_few_labels(client):
    response = _post(client, labels=["Nur eins"])
    assert response.status_code == 422
    assert "labels" in response.json()["error"]


def test_too_many_labels(client):
    response = _post(client, labels=[f"Label {i}" for i in range(51)])
    assert response.status_code == 422


def test_duplicate_labels(client):
    response = _post(client, labels=["Rechnung", "Rechnung", "Spam"])
    assert response.status_code == 422
    assert "unique" in response.json()["error"]


def test_empty_label(client):
    response = _post(client, labels=["Rechnung", "   "])
    assert response.status_code == 422
    assert "empty" in response.json()["error"]


def test_missing_text(client):
    response = client.post("/classify", json={"labels": BASE["labels"]}, headers=AUTH)
    assert response.status_code == 422
    assert "text" in response.json()["error"]


def test_empty_text(client):
    assert _post(client, text="").status_code == 422


def test_blank_text(client):
    assert _post(client, text="    ").status_code == 422


def test_text_too_long(client):
    assert _post(client, text="a" * 5001).status_code == 422


def test_text_at_limit_is_accepted(client):
    assert _post(client, text="a" * 5000).status_code == 200


def test_template_without_placeholder(client):
    response = _post(client, hypothesis_template="Diese Nachricht betrifft das Thema.")
    assert response.status_code == 422
    assert "{}" in response.json()["error"]


def test_unknown_field_is_rejected(client):
    response = _post(client, temperature=0.7)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-API-Key": ""},
        {"X-API-Key": "falsch"},
        {"X-API-Key": "test-key-x"},
        {"Authorization": "Bearer falsch"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic test-key"},
        {"Authorization": "test-key"},
    ],
)
def test_wrong_or_missing_api_key(client, headers):
    response = client.post("/classify", json=BASE, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "invalid or missing API key"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": f"Bearer {API_KEY}"},
        {"Authorization": f"Bearer {SECOND_API_KEY}"},
        {"X-API-Key": API_KEY},
        {"X-API-Key": SECOND_API_KEY},
    ],
)
def test_accepted_key_variants(client, headers):
    response = client.post("/classify", json=BASE, headers=headers)
    assert response.status_code == 200


def test_batch_requires_api_key(client):
    response = client.post("/classify/batch", json={"items": [BASE]})
    assert response.status_code == 401


def test_batch_rejects_more_than_32_items(client):
    response = client.post(
        "/classify/batch", json={"items": [BASE] * 33}, headers=AUTH
    )
    assert response.status_code == 422


def test_batch_rejects_empty_items(client):
    response = client.post("/classify/batch", json={"items": []}, headers=AUTH)
    assert response.status_code == 422


def test_batch_validates_each_item(client):
    response = client.post(
        "/classify/batch",
        json={"items": [BASE, {"text": "x", "labels": ["a", "a"]}]},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert "unique" in response.json()["error"]
