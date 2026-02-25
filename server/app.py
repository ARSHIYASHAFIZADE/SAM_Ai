from flask import Flask, request, jsonify, session
from flask_cors import CORS
from config import ApplicationConfig
from models import db
from extensions import bcrypt, server_session, mail
from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(ApplicationConfig)

# Initialize Extensions
# Important: DB init must happen here with app
db.init_app(app)
bcrypt.init_app(app)
server_session.init_app(app)

# Mail Config (Manual init if config not picked up automatically by init_app?)
# Flask-Mail init_app(app) usually works if config is set on app.
# User's original code set config manually then called Mail(app).
# ApplicationConfig should have the mail settings.
# Let's verify if they are in config.py. If not, we set them here.
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'a6833351@gmail.com'
app.config['MAIL_PASSWORD'] = 'fxfm lfzl gwme oohz'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = 'a6833351@gmail.com'
mail.init_app(app)

# CORS
allowed_origins = [
    "https://sam-ai-theta.vercel.app",
    "https://sam-ai-7lwa.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://localhost:10000",
    "http://127.0.0.1:10000"
]

CORS(app, 
     resources={r"/*": {"origins": allowed_origins}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_response("")
        origin = request.headers.get('Origin')
        allowed_origins = ["https://sam-ai-7lwa.onrender.com", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://localhost:10000", "http://127.0.0.1:10000"]
        if origin in allowed_origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        else:
            response.headers.add("Access-Control-Allow-Origin", "https://sam-ai-theta.vercel.app")
            
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 204

with app.app_context():
    db.create_all()

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)

@app.route('/')
def hello_world():
    return "Hello World"

@app.route('/test_redis')
def test_redis():
    session['test'] = 'Redis working!'
    return jsonify({"message": session.get('test', 'Failed to set session')})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
