from dotenv import load_dotenv
import os
import redis
load_dotenv()

class ApplicationConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = r"sqlite:///./db.sqlite"

    # Session settings for Redis
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_REDIS = redis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))
