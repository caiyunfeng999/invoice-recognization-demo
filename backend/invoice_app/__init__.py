"""Flask application factory.

The app factory keeps initialization independent from route/business modules.
This matches the required modular project style: the main file creates the app,
while API routes and image/OCR functions stay in separate submodules.
"""

from flask import Flask
from flask_cors import CORS

from .routes import api_bp


def create_app() -> Flask:
    """Create and configure the Flask backend application."""
    app = Flask(__name__)

    # The React frontend runs on a different local port, so CORS is required.
    CORS(app)

    # All backend API endpoints are grouped under /api for a clean boundary
    # between the static frontend and the JSON/image processing backend.
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
