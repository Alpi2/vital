import os
from fastapi.testclient import TestClient
from app.main import app


def setup_module(module):
    # Ensure SECRET_KEY is set for app import/config
    os.environ.setdefault("SECRET_KEY", "testsecret123")


def test_docs_available():
    client = TestClient(app)
    r = client.get("/api/docs")
    assert r.status_code == 200


def test_list_and_create_patient():
    client = TestClient(app)
    # list initially empty
    r = client.get("/api/v1/patients/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # create
    payload = {
        "medical_id": "MID-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "1990-01-01",
        "gender": "female",
    }
    r2 = client.post("/api/v1/patients/", json=payload)
    assert r2.status_code == 200
    body = r2.json()
    assert body["medical_id"] == payload["medical_id"]
    assert body["first_name"] == payload["first_name"]
    assert "id" in body
