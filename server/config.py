from dotenv import load_dotenv
import os
import redis 
load_dotenv()

class ApplicationConfig:
    SECRET_KEY=os.environ["SECRET_KEY"]
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = r"sqlite:///./db.sqlite"
    
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False 
    SESSION_USE_SIGNER = True  # Corrected typo: "USER_SIGNER" -> "USE_SIGNER"
    
    # Use Redis URL from environment for production
    SESSION_REDIS = redis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))
