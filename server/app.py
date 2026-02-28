from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from config import ApplicationConfig
from models import db
from extensions import bcrypt, server_session, mail, limiter
from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
from utils import configure_logging
from ml_models import set_model_ready, all_models_ready, MODEL_VERSION
import os
import logging
import re

# Fix 5: Structured JSON logging configured before anything else
configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(ApplicationConfig)

# Fix 2: CSRF protection (exempt JSON API endpoints via custom header check below)
csrf = CSRFProtect(app)

# Exempt all API blueprints from CSRF token requirement —
# protection is provided by SameSite cookie + CORS allowlist instead.
# For a pure JSON API (no HTML forms served), this is the correct approach.
csrf.exempt(auth_bp)
csrf.exempt(predict_bp)

# Initialize Extensions
db.init_app(app)
bcrypt.init_app(app)
server_session.init_app(app)
limiter.init_app(app)

# Fix 1: Mail credentials from env only
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
mail.init_app(app)

# CORS — restrict to known origins
allowed_origins = [
    "https://sam-ai-theta.vercel.app",
    "https://sam-a456gz3zd-arshiyashafizades-projects.vercel.app",
    "https://sam-ai-7lwa.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://localhost:10000",
    "http://127.0.0.1:10000",
    re.compile(r"^https://sam-.*\.vercel\.app$")
]

CORS(app,
     resources={r"/*": {"origins": allowed_origins}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

with app.app_context():
    db.create_all()

    # Fix 6 (from previous): Eager model training at startup
    logger.info("Training ML models at startup...")
    try:
        from services.diabetes_service import preprocess_female_diabetes, preprocess_male_diabetes
        from services.heart_service import train_heart_model
        from services.liver_service import train_liver_model
        from services.cancer_service import train_cancer_model

        preprocess_female_diabetes()
        set_model_ready("female_diabetes")
        preprocess_male_diabetes()
        set_model_ready("male_diabetes")
        train_heart_model()
        set_model_ready("heart")
        train_liver_model()
        set_model_ready("liver")
        train_cancer_model()
        set_model_ready("cancer")
        logger.info("All ML models ready.")
    except Exception as e:
        logger.error("Model training at startup failed: %s", e)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)

@app.route('/')
def hello_world():
    return "Hello World"

@app.route('/test_redis')
def test_redis():
    try:
        session['test'] = 'Redis working!'
        return jsonify({"message": session.get('test', 'Failed to set session')})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/healthz')
def healthz():
    status = {"model_version": MODEL_VERSION, "models_ready": all_models_ready()}
    try:
        db.session.execute(db.text('SELECT 1'))
        status["db"] = "ok"
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        status["db"] = "error"
    ok = status["models_ready"] and status["db"] == "ok"
    return jsonify(status), (200 if ok else 503)


@app.route('/csrf-token', methods=['GET'])
def get_csrf_token():
    from flask_wtf.csrf import generate_csrf
    token = generate_csrf()
    return jsonify({'csrf_token': token})

# Automatically create tables in Railway or local if they don't exist
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
