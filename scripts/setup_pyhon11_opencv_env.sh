#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/pyhon11-opencv/bin/python}"
PIP_BIN="${PIP_BIN:-/opt/anaconda3/envs/pyhon11-opencv/bin/pip}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PIP_BIN" install -r "$PROJECT_DIR/backend/requirements.txt"
"$PIP_BIN" install -r "$PROJECT_DIR/backend/requirements-dl.txt"

"$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "flask",
    "flask_cors",
    "cv2",
    "numpy",
    "PIL",
    "pytesseract",
    "paddleocr",
    "paddle",
    "fitz",
    "ultralytics",
    "torch",
    "torchvision",
]

missing = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: NO ({type(exc).__name__}: {exc})")
        missing.append(name)

if missing:
    raise SystemExit(f"Missing modules: {', '.join(missing)}")
PY

echo "pyhon11-opencv environment is ready."

