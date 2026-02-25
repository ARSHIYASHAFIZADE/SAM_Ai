import unittest
import json
import os
from app import app
from models import db

class TestPredictionEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # Note: We are using the actual DB/Redis for these tests if configured,
        # or relying on the services' fallback to mock data/URL loading.
        # Since the goal is to test the services' logic and integration.

    def test_hello_world(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "Hello World")

    def test_female_diabetes_prediction(self):
        data = {
            "data": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertIn('prediction', res_data)
        self.assertIn('probability', res_data)

    def test_male_diabetes_prediction(self):
        data = {
            "data": {
                "Glucose": 100,
                "BloodPressure": 80,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 25,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30
            }
        }
        response = self.app.post('/predict_male', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertIn('prediction', res_data)
        self.assertIn('probability', res_data)

    def test_heart_disease_prediction(self):
        data = {
            "data": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }
        response = self.app.post('/detect_heart', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertIn('prediction', res_data)
        self.assertIn('probability', res_data)

    def test_liver_health_prediction(self):
        # The liver service expects an array of 10 values [Age, Gender, BMI, Alcohol, ...]
        # Based on service analysis it uses input_data = request.json.get('input_data')
        data = {
            "input_data": [45, 1, 28.5, 10, 50, 60, 1.2, 0.8, 30, 40]
        }
        response = self.app.post('/detect_liver', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertIn('prediction', res_data)
        self.assertIn('probability_healthy_liver', res_data)

if __name__ == '__main__':
    unittest.main()
