import base64
import io
from typing import Any, Dict

import cv2
import numpy as np
from flask import request
from PIL import Image


def read_image_from_request() -> np.ndarray:
    if "file" not in request.files:
        raise ValueError("request must contain a file field")

    raw = request.files["file"].read()
    data = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("uploaded file is not a valid image")
    return image


def image_to_base64(image: np.ndarray) -> str:
    encode_image = image
    if len(image.shape) == 3:
        encode_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(encode_image.astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_metadata(image: np.ndarray) -> Dict[str, Any]:
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }
