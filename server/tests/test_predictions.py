"""
Integration tests for SAM AI prediction endpoints.

Run from the server/ directory:
    pytest tests/ -v

All tests use the Flask test client against an in-memory SQLite database.
A registered+logged-in session is created once per test class.
"""

import json
import pytest
from app import app
from models import db


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_COOKIE_SECURE"] = False

    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.drop_all()


@pytest.fixture(scope="module")
def auth_client(client):
    """Return a test client that has a valid session."""
    credentials = {"email": "pytest@sam.ai", "password": "TestPass123!"}
    client.post("/register", json=credentials)
    client.post("/login", json=credentials)
    return client


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_success(self, client):
        r = client.post("/register", json={"email": "new@sam.ai", "password": "Pass1234!"})
        assert r.status_code == 201
        data = r.get_json()
        assert "id" in data
        assert data["email"] == "new@sam.ai"

    def test_register_duplicate(self, client):
        client.post("/register", json={"email": "dup@sam.ai", "password": "Pass1234!"})
        r = client.post("/register", json={"email": "dup@sam.ai", "password": "Pass1234!"})
        assert r.status_code == 409

    def test_register_short_password(self, client):
        r = client.post("/register", json={"email": "short@sam.ai", "password": "abc"})
        assert r.status_code == 400

    def test_login_invalid_password(self, client):
        client.post("/register", json={"email": "login@sam.ai", "password": "CorrectPass!"})
        r = client.post("/login", json={"email": "login@sam.ai", "password": "WrongPass!"})
        assert r.status_code == 401

    def test_me_unauthenticated(self, client):
        with app.test_client() as c:
            r = c.get("/@me")
            assert r.status_code == 401

    def test_me_authenticated(self, auth_client):
        r = auth_client.get("/@me")
        assert r.status_code == 200
        assert "email" in r.get_json()


# ── Predictions ───────────────────────────────────────────────────────────────

class TestHeartDisease:
    PAYLOAD = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }

    def test_predict(self, auth_client):
        r = auth_client.post("/detect_heart", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("Low", "Moderate", "High")

    def test_requires_auth(self, client):
        with app.test_client() as c:
            r = c.post("/detect_heart", json=self.PAYLOAD)
            assert r.status_code == 401

    def test_missing_fields(self, auth_client):
        r = auth_client.post("/detect_heart", json={"age": 55})
        assert r.status_code == 422


class TestFemaleDiabetes:
    PAYLOAD = {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50,
    }

    def test_predict(self, auth_client):
        r = auth_client.post("/predict", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "probability" in data

    def test_requires_auth(self, client):
        with app.test_client() as c:
            r = c.post("/predict", json=self.PAYLOAD)
            assert r.status_code == 401


class TestMaleDiabetes:
    PAYLOAD = {
        "Age": 45,
        "Gender": "Male", "Polyuria": "Yes", "Polydipsia": "Yes",
        "sudden_weight_loss": "No", "weakness": "Yes", "Polyphagia": "Yes",
        "Genital_thrush": "No", "visual_blurring": "Yes", "Itching": "Yes",
        "Irritability": "No", "delayed_healing": "Yes", "partial_paresis": "No",
        "muscle_stiffness": "No", "Alopecia": "No", "Obesity": "Yes",
    }

    def test_predict(self, auth_client):
        r = auth_client.post("/predict_male", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data

    def test_requires_auth(self, client):
        with app.test_client() as c:
            r = c.post("/predict_male", json=self.PAYLOAD)
            assert r.status_code == 401


class TestLiverHealth:
    PAYLOAD = {"input_data": [45.0, 1.2, 0.4, 300.0, 45.0, 50.0, 6.5, 3.8, 1.1, 1.0]}

    def test_predict(self, auth_client):
        r = auth_client.post("/detect_liver", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "probability_healthy_liver" in data

    def test_requires_auth(self, client):
        with app.test_client() as c:
            r = c.post("/detect_liver", json=self.PAYLOAD)
            assert r.status_code == 401

    def test_wrong_feature_count(self, auth_client):
        r = auth_client.post("/detect_liver", json={"input_data": [1.0, 2.0]})
        assert r.status_code == 422


class TestBreastCancer:
    PAYLOAD = {
        "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
        "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
        "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589,
        "area_error": 153.4, "smoothness_error": 0.006399, "compactness_error": 0.04904,
        "concavity_error": 0.05373, "concave_points_error": 0.01587,
        "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6,
        "worst_area": 2019.0, "worst_smoothness": 0.1622, "worst_compactness": 0.6656,
        "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189,
    }

    def test_predict_returns_charts(self, auth_client):
        r = auth_client.post("/detect_breast_cancer", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "radar_chart" in data
        assert "bar_chart" in data
        assert data["prediction"] in (0, 1)

    def test_requires_auth(self, client):
        with app.test_client() as c:
            r = c.post("/detect_breast_cancer", json=self.PAYLOAD)
            assert r.status_code == 401
