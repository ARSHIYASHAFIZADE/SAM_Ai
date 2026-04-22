from flask import Blueprint, jsonify, request, session
from extensions import bcrypt, limiter
from models import User, db

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/@me')
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"id": user.id, "email": user.email, "name": user.name})


@auth_bp.route("/register", methods=["POST"])
@limiter.limit("10 per hour")
def register_user():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(email=email, password=hashed, name=name or None)
    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    return jsonify({"id": user.id, "email": user.email}), 201


@auth_bp.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login_user():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    pass_hash = user.password.encode('utf-8') if isinstance(user.password, str) else user.password
    if not bcrypt.check_password_hash(pass_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user.id
    return jsonify({"id": user.id, "email": user.email})


@auth_bp.route('/logout', methods=['POST'])
def logout_user():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out successfully"})
