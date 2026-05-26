from flask import Flask
from flask_cors import CORS

from .routes import api_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
