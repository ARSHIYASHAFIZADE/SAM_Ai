from flask import Blueprint, jsonify, request, session, current_app
from models import User
from sqlalchemy.orm.exc import NoResultFound
from services.diabetes_service import predict_female_diabetes, predict_male_diabetes
from services.heart_service import predict_heart_disease
from services.liver_service import predict_liver_health
from services.liver_service import predict_liver_health
from services.cancer_service import predict_breast_cancer, create_pdf_report, send_email, get_breast_cancer_visualizations

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict_female():
    try:
        input_data = request.json.get('data')
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction, probability = predict_female_diabetes(input_data)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/predict_male', methods=['POST'])
def predict_male():
    try:
        input_data = request.json.get('data')
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
            
        prediction, probability = predict_male_diabetes(input_data)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/detect_heart', methods=['POST'])
def detect_heart():
    try:
        data = request.get_json()
        input_data = data.get('data', {}) # Expecting {data: {...}}
        if not input_data:
             return jsonify({'error': 'No data provided'}), 400
             
        prediction, probability = predict_heart_disease(input_data)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/detect_liver', methods=['POST'])
def detect_liver():
    try:
        input_data = request.json.get('input_data')
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
            
        prediction, probability = predict_liver_health(input_data)
        
        return jsonify({
            'prediction': prediction,
            'probability_healthy_liver': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/detect_breast_cancer', methods=['POST'])
def detect_breast_cancer():
    try:
        # Auth check
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401
            
        try:
            user = User.query.filter_by(id=user_id).one()
            recipient_email = user.email
        except NoResultFound:
            return jsonify({"error": "User not found"}), 404
        
        input_data = request.json
        if not input_data:
             return jsonify({'error': 'No data provided'}), 400

        prediction, probability = predict_breast_cancer(input_data)
        
        # Report Generation
        result = int(prediction)
        input_data_values = list(input_data.values())
        input_names = list(input_data.keys())
        pdf_filename = 'Breast_Cancer_Detection_Report.pdf'
        
        # Create PDF (Using helper from service, might need wrapper if args differ)
        create_pdf_report(pdf_filename, result, probability, input_data_values, input_names)
        
        # Send Email
        subject = 'Breast Cancer Detection Report'
        body = 'Please find attached the breast cancer detection report.'
        # Pass app config to service helper
        send_email(recipient_email, subject, body, pdf_filename, 'pdf', current_app.config)
        
        # Get Visualizations
        # input_data is a dict {'mean_radius': val, ...} from request.json
        # we need input_names keys (which match what we trained on hopefully)
        visualizations = get_breast_cancer_visualizations(input_data, list(input_data.keys()))

        return jsonify({
            "prediction": result,
            "probability_breast_cancer": probability,
            "radar_chart": visualizations.get('radar_chart'),
            "bar_chart": visualizations.get('bar_chart')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
