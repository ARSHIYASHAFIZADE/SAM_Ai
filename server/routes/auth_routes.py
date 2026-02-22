from flask import Blueprint, jsonify, request, session
from extensions import db, bcrypt
from models import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/@me')
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error":"unauthorized"}), 401
    user = User.query.filter_by(id=user_id).first()
    return jsonify({
        "id":user.id,
        "email":user.email
    })

@auth_bp.route("/register", methods=["POST"])
def register_user():
    email = request.json['email']
    password = request.json['password']
    user_exist = User.query.filter_by(email=email).first() is not None
    if user_exist:
        return jsonify({"error":"user already exist"}), 409
    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({
        "id":new_user.id,
        "email":new_user.email,
    })

@auth_bp.route('/login', methods=['POST'])
def login_user():
    email = request.json['email']
    password = request.json['password']
    user = User.query.filter_by(email=email).first() 
    if user is None:
        return jsonify({"error":"unauthorized"}), 401
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error":"unauthorized"}), 401  
    session["user_id"] = user.id
    return jsonify({
        "id":user.id,
        "email":user.email,
    })

@auth_bp.route('/logout', methods=['POST'])
def logout_user():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out successfully"})
