from flask import Blueprint, jsonify, request, session, current_app
from flask_wtf.csrf import CSRFProtect
from models import User
from sqlalchemy.orm.exc import NoResultFound
from services.diabetes_service import predict_female_diabetes, predict_male_diabetes, gbc_female, lr_male, feature_names_female, X_male_columns
from services.heart_service import predict_heart_disease, model_trained_heart
from services.liver_service import predict_liver_health, gbm_model_liver
from services.cancer_service import predict_breast_cancer, create_pdf_report, send_email, get_breast_cancer_visualizations, model_breast_cancer
from schemas import (
    FemaleDiabetesInput, MaleDiabetesInput, HeartDiseaseInput,
    LiverInput, BreastCancerInput
)
from utils import require_auth, error_response
from ml_models import build_response, get_feature_importance
from pydantic import ValidationError
import tempfile
import os
import threading
import logging

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)


def _bad(exc: ValidationError):
    return jsonify({"error": "Invalid request body", "details": exc.errors()}), 422


# ── Female Diabetes ───────────────────────────────────────────────────────────
@predict_bp.route('/predict', methods=['POST'])
@require_auth  # Fix 3
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
        return error_response("Prediction failed", 500, e)  # Fix 4


# ── Male Diabetes ─────────────────────────────────────────────────────────────
@predict_bp.route('/predict_male', methods=['POST'])
@require_auth  # Fix 3
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
        return error_response("Prediction failed", 500, e)  # Fix 4


# ── Heart Disease ─────────────────────────────────────────────────────────────
@predict_bp.route('/detect_heart', methods=['POST'])
@require_auth  # Fix 3
def detect_heart():
    try:
        body = request.get_json(force=True) or {}
        payload = HeartDiseaseInput.model_validate(body.get('data', {}))
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_heart_disease(payload.model_dump())
        fi = get_feature_importance(model_trained_heart)
        return jsonify(build_response(prediction, probability, fi))
    except Exception as e:
        return error_response("Prediction failed", 500, e)  # Fix 4


# ── Liver Health ──────────────────────────────────────────────────────────────
@predict_bp.route('/detect_liver', methods=['POST'])
@require_auth  # Fix 3
def detect_liver():
    try:
        payload = LiverInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)
    try:
        prediction, probability = predict_liver_health(payload.input_data)
        fi = get_feature_importance(gbm_model_liver)
        resp = build_response(prediction, probability, fi)
        resp["probability_healthy_liver"] = resp["probability"]  # backward compat
        return jsonify(resp)
    except Exception as e:
        return error_response("Prediction failed", 500, e)  # Fix 4


# ── Breast Cancer ─────────────────────────────────────────────────────────────
@predict_bp.route('/detect_breast_cancer', methods=['POST'])
@require_auth  # Fix 3 (also keeps inner auth check for email look-up)
def detect_breast_cancer():
    user_id = session.get('user_id')
    try:
        user = User.query.filter_by(id=user_id).one()
        recipient_email = user.email
    except NoResultFound:
        return jsonify({"error": "User not found"}), 404

    try:
        payload = BreastCancerInput.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return _bad(e)

    try:
        input_data = payload.model_dump()
        prediction, probability = predict_breast_cancer(input_data)
        result = int(prediction)
        input_data_values = list(input_data.values())
        input_names = list(input_data.keys())

        # Fix 7 (from previous fixes): per-request tempfile, guaranteed deletion
        tmp_fd, pdf_filename = tempfile.mkstemp(suffix='.pdf')
        os.close(tmp_fd)

        try:
            create_pdf_report(pdf_filename, result, probability, input_data_values, input_names)

            # Fix 8: send email in a background thread so response isn't blocked
            app_config = dict(current_app.config)  # snapshot config for thread safety
            def _send():
                try:
                    send_email(
                        recipient_email,
                        'Breast Cancer Detection Report',
                        'Please find attached the breast cancer detection report.',
                        pdf_filename, 'pdf', app_config
                    )
                except Exception as ex:
                    logger.error("Background email failed: %s", ex)
                finally:
                    try:
                        os.unlink(pdf_filename)  # clean up after thread finishes
                    except OSError:
                        pass

            threading.Thread(target=_send, daemon=True).start()
            # NOTE: pdf_filename is now owned by the thread; don't unlink here

        except Exception as e:
            # PDF creation failed — clean up synchronously
            try:
                os.unlink(pdf_filename)
            except OSError:
                pass
            return error_response("Report generation failed", 500, e)  # Fix 4

        visualizations = get_breast_cancer_visualizations(input_data, input_names)
        fi = get_feature_importance(model_breast_cancer)
        resp = build_response(result, probability, fi)
        resp["probability_breast_cancer"] = resp["probability"]  # backward compat
        resp["radar_chart"] = visualizations.get('radar_chart')
        resp["bar_chart"] = visualizations.get('bar_chart')
        return jsonify(resp)
    except Exception as e:
        return error_response("Prediction failed", 400, e)  # Fix 4
