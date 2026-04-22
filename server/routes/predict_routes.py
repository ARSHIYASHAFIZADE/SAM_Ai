from flask import Blueprint, jsonify, request, session
from models import User
from sqlalchemy.orm.exc import NoResultFound
from services.diabetes_service import predict_female_diabetes, predict_male_diabetes, gbc_female, lr_male, feature_names_female
from services.heart_service import predict_heart_disease, model_trained_heart
from services.liver_service import predict_liver_health, gbm_model_liver
from services.cancer_service import predict_breast_cancer, get_breast_cancer_visualizations, model_breast_cancer
from schemas import (
    FemaleDiabetesInput, MaleDiabetesInput, HeartDiseaseInput,
    LiverInput, BreastCancerInput
)
from utils import require_auth, error_response
from ml_models import build_response, get_feature_importance
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)


def _bad(exc: ValidationError):
    return jsonify({"error": "Invalid request body", "details": exc.errors()}), 422


@predict_bp.route('/predict', methods=['POST'])
@require_auth
def predict_female():
    try:
        payload = FemaleDiabetesInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_female_diabetes(payload.model_dump())
        fi = get_feature_importance(gbc_female, feature_names_female)
        return jsonify(build_response(prediction, probability, fi))
    except Exception as e:
        return error_response("Prediction failed", 500, e)


@predict_bp.route('/predict_male', methods=['POST'])
@require_auth
def predict_male():
    try:
        payload = MaleDiabetesInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_male_diabetes(payload.model_dump())
        fi = get_feature_importance(lr_male)
        return jsonify(build_response(prediction, probability, fi))
    except Exception as e:
        return error_response("Prediction failed", 500, e)


@predict_bp.route('/detect_heart', methods=['POST'])
@require_auth
def detect_heart():
    try:
        body = request.get_json(force=True) or {}
        payload = HeartDiseaseInput.model_validate(body.get('data', body))
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_heart_disease(payload.model_dump())
        fi = get_feature_importance(model_trained_heart)
        return jsonify(build_response(prediction, probability, fi))
    except Exception as e:
        return error_response("Prediction failed", 500, e)


@predict_bp.route('/detect_liver', methods=['POST'])
@require_auth
def detect_liver():
    try:
        payload = LiverInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_liver_health(payload.input_data)
        fi = get_feature_importance(gbm_model_liver)
        resp = build_response(prediction, probability, fi)
        resp["probability_healthy_liver"] = resp["probability"]
        return jsonify(resp)
    except Exception as e:
        return error_response("Prediction failed", 500, e)


@predict_bp.route('/detect_breast_cancer', methods=['POST'])
@require_auth
def detect_breast_cancer():
    try:
        payload = BreastCancerInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)
    try:
        input_data = payload.model_dump()
        input_names = list(input_data.keys())
        prediction, probability = predict_breast_cancer(input_data)
        visualizations = get_breast_cancer_visualizations(input_data, input_names)
        fi = get_feature_importance(model_breast_cancer)
        resp = build_response(int(prediction), probability, fi)
        resp["probability_breast_cancer"] = resp["probability"]
        resp["radar_chart"] = visualizations.get('radar_chart')
        resp["bar_chart"] = visualizations.get('bar_chart')
        return jsonify(resp)
    except Exception as e:
        return error_response("Prediction failed", 400, e)
