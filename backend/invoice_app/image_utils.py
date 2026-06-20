"""File and image utility functions.

This module isolates request-file reading, image decoding, PDF-page conversion
and base64 encoding.  Routes call these helpers instead of handling binary file
details directly, keeping controller code simple.
"""

import base64
import io
from typing import Any, Dict

import cv2
import numpy as np
from flask import request
from PIL import Image

from .pdf_utils import pdf_first_page_to_image


def read_file_from_request():
    """Read the uploaded file from Flask ``request.files``.

    Returns:
        filename, MIME type and raw bytes.  The raw bytes can then be routed to
        image decoding or PDF handling according to file type.
    """
    if "file" not in request.files:
        raise ValueError("request must contain a file field")

    upload = request.files["file"]
    raw = upload.read()
    return upload.filename or "", upload.mimetype or "", raw


def is_pdf_file(filename: str, mimetype: str, raw: bytes) -> bool:
    """Detect PDF files using MIME type, filename and file signature."""
    return mimetype == "application/pdf" or filename.lower().endswith(".pdf") or raw.startswith(b"%PDF")


def image_from_bytes(raw: bytes) -> np.ndarray:
    """Decode common image formats into an OpenCV BGR image array."""
    data = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("uploaded file is not a valid image")
    return image


def read_image_from_request() -> np.ndarray:
    """Read an uploaded image or render the selected PDF page as an image."""
    filename, mimetype, raw = read_file_from_request()
    if is_pdf_file(filename, mimetype, raw):
        page_number = int(request.form.get("page", 1))
        image, _ = pdf_first_page_to_image(raw, page_number=page_number)
        return image
    return image_from_bytes(raw)


def image_to_base64(image: np.ndarray) -> str:
    """Encode an OpenCV image as PNG base64 for JSON transport to React."""
    encode_image = image
    if len(image.shape) == 3:
        encode_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(encode_image.astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_metadata(image: np.ndarray) -> Dict[str, Any]:
    """Return image size and channel count for frontend processing history."""
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }
