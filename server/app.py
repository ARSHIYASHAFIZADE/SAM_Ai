from flask import Flask, jsonify
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from config import ApplicationConfig
from models import db
from extensions import bcrypt, server_session, mail, limiter
from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
from utils import configure_logging
from ml_models import all_models_ready, MODEL_VERSION
import os
import logging

configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(ApplicationConfig)

# CSRF — exempt JSON-only blueprints; protection is provided by SameSite cookie + CORS
csrf = CSRFProtect(app)
csrf.exempt(auth_bp)
csrf.exempt(predict_bp)

db.init_app(app)
bcrypt.init_app(app)
server_session.init_app(app)
limiter.init_app(app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
mail.init_app(app)

# CORS — localhost by default; add production origins via ALLOWED_ORIGINS env var (comma-separated)
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
]

_extra = os.environ.get("ALLOWED_ORIGINS", "")
if _extra:
    allowed_origins.extend([o.strip() for o in _extra.split(",") if o.strip()])

CORS(app,
     resources={r"/*": {"origins": allowed_origins}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

with app.app_context():
    db.create_all()
    logger.info("Database tables ready.")

app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)


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
    return jsonify({'csrf_token': generate_csrf()})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
