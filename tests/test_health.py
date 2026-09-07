from fastapi.testclient import TestClient


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_503_before_model_is_loaded():
    from app import classifier as classifier_module
    from app.main import app

    classifier_module.classifier._pipe = None
    # No context manager -> no lifespan, so the model stays unloaded.
    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading", "model_loaded": False}


def test_health_needs_no_auth(client):
    assert client.get("/health").status_code == 200
