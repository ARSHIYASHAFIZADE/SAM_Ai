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

    # Fix: Default over to 'filesystem' as the Redis instance running on Railway appears broken
    SESSION_TYPE = "filesystem"
    # _redis_url = os.environ.get("REDIS_URL")
    # if _redis_url:
    #     SESSION_TYPE = "redis"
    #     SESSION_REDIS = redis.from_url(_redis_url)
    
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    _secure = os.environ.get("SESSION_COOKIE_SECURE", "true").lower() != "false"
    SESSION_COOKIE_SECURE = _secure
    SESSION_COOKIE_SAMESITE = "None" if _secure else "Lax"
