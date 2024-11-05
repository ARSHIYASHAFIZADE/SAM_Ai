from dotenv import load_dotenv
import os
import redis

load_dotenv()  # Load environment variables from a .env file if running locally

class ApplicationConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")  # Ensure this is set in Render
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = r"sqlite:///./db.sqlite"  # Change if using a production database

    # Session settings for Redis
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_REDIS = redis.from_url(os.environ.get("REDIS_URL"))  # Connect to Redis using the environment variable
