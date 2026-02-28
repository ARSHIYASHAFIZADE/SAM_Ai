from dotenv import load_dotenv
import os
import redis

load_dotenv()

class ApplicationConfig:
    # Crash immediately if SECRET_KEY is missing
    _secret = os.environ.get("SECRET_KEY")
    if not _secret:
        raise RuntimeError("SECRET_KEY environment variable is not set. Refusing to start.")
    SECRET_KEY = _secret

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = os.environ.get("DEBUG", "false").lower() == "true"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", r"sqlite:///./db.sqlite")

    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB

    # Fix 2: CSRF via Flask-WTF (already in requirements)
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour

    # Fix 6: Redis sessions — fall back to filesystem only if REDIS_URL is absent (local dev)
    _redis_url = os.environ.get("REDIS_URL")
    if _redis_url:
        SESSION_TYPE = "redis"
        SESSION_REDIS = redis.from_url(_redis_url)
    else:
        SESSION_TYPE = "filesystem"

    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_COOKIE_SAMESITE = "None"
    SESSION_COOKIE_SECURE = True
