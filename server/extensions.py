from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_session import Session
from flask_mail import Mail

# Initialize extensions (unbound)
db = SQLAlchemy()
bcrypt = Bcrypt()
server_session = Session()
mail = Mail()
