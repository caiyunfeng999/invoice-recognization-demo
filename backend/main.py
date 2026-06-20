"""Backend main entry for the invoice OCR web system.

This file is intentionally small.  The course document emphasizes a clear
main entry and modular design, so the Flask application construction lives in
``invoice_app/__init__.py`` and all business logic is delegated to submodules.
Run this file to start the backend API service.
"""

from invoice_app import create_app


# Flask looks for a module-level ``app`` object when the project is launched by
# external tools.  Keeping it here also makes this file easy to read in reports.
app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
