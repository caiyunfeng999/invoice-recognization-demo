import time
from inspect import signature

from flask import Blueprint, jsonify, request

from .image_utils import image_metadata, image_to_base64, read_image_from_request
from .ocr import recognize_invoice
from .preprocessing import PROCESSOR_LABELS, PROCESSORS


api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "modules": ["preprocessing", "ocr", "parser"]})


@api_bp.get("/methods")
def methods():
    return jsonify(
        {
            "methods": [
                {"key": key, "label": PROCESSOR_LABELS[key]}
                for key in PROCESSORS.keys()
            ]
        }
    )


@api_bp.post("/process/<method>")
def process_image(method: str):
    if method not in PROCESSORS:
        return jsonify({"error": "unknown processing method"}), 400

    try:
        image = read_image_from_request()
        started = time.perf_counter()
        processor = PROCESSORS[method]
        accepted = set(signature(processor).parameters.keys()) - {"image"}
        params = {
            key: value
            for key, value in request.form.to_dict().items()
            if key in accepted
        }
        result = processor(image, **params)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(
            {
                "image": image_to_base64(result.image),
                "method": method,
                "label": PROCESSOR_LABELS[method],
                "description": result.description,
                "elapsed_ms": elapsed_ms,
                "metadata": image_metadata(result.image),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.post("/ocr")
def run_ocr():
    try:
        image = read_image_from_request()
        lang = request.form.get("lang", "chi_sim+eng")
        psm = int(request.form.get("psm", 6))
        started = time.perf_counter()
        result = recognize_invoice(image, lang=lang, psm=psm)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["metadata"] = image_metadata(image)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
