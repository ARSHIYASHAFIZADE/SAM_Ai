from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_session import Session
from flask_mail import Mail
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize extensions (unbound)
db = SQLAlchemy()
bcrypt = Bcrypt()
server_session = Session()
mail = Mail()

# Fix 4: Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)
