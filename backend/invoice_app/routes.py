"""HTTP API route layer.

This module is the controller layer of the backend.  It receives files and
parameters from React, calls the corresponding image/PDF/OCR/parser modules,
and returns JSON responses.  No heavy image-processing algorithm is implemented
directly here; routes only coordinate submodules.
"""

import time
from inspect import signature

from flask import Blueprint, jsonify, request

from .image_utils import image_from_bytes, image_metadata, image_to_base64, is_pdf_file, read_file_from_request, read_image_from_request
from .ocr import recognize_invoice
from .paddle_ocr import recognize_invoice_paddle
from .parser import completion_score, parse_invoice_text
from .pdf_utils import extract_pdf_text, pdf_first_page_to_image
from .preprocessing import PROCESSOR_LABELS, PROCESSORS
from .validators import validate_fields
from .detectors import detect_fields, detector_status, detectors_status
from .yolo_detector import (
    draw_detections,
    is_yolo_suitable_invoice,
    recognize_invoice_with_yolo,
    yolo_decision_reason,
    yolo_status,
)


api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    """Return a lightweight status response for frontend/server checks."""
    return jsonify({"status": "ok", "modules": ["preprocessing", "ocr", "paddle_ocr", "pdf", "parser", "validators", "detectors"]})


@api_bp.get("/methods")
def methods():
    """Expose available preprocessing methods for UI rendering or debugging."""
    return jsonify(
        {
            "methods": [
                {"key": key, "label": PROCESSOR_LABELS[key]}
                for key in PROCESSORS.keys()
            ]
        }
    )


@api_bp.get("/yolo/status")
def get_yolo_status():
    """Return whether YOLO dependencies and model file are available."""
    return jsonify(yolo_status())


@api_bp.get("/detectors/status")
def get_detectors_status():
    """Return all selectable detector backends and their availability."""
    return jsonify(detectors_status())


@api_bp.post("/yolo/detect")
def yolo_detect():
    """Run selected field-region detector and return an annotated invoice image."""
    try:
        image = read_image_from_request()
        confidence = float(request.form.get("yolo_confidence", 0.35))
        detector = request.form.get("detector", "yolo")
        started = time.perf_counter()
        detections = detect_fields(image, detector=detector, confidence=confidence)
        annotated = draw_detections(image, detections)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(
            {
                "image": image_to_base64(annotated),
                "detector": detector_status(detector if detector in {"yolo", "faster_rcnn", "dfine_l"} else "yolo"),
                "detections": [
                    {
                        "label": item.label,
                        "field": item.field,
                        "confidence": item.confidence,
                        "box": list(item.box),
                    }
                    for item in detections
                ],
                "detection_count": len(detections),
                "elapsed_ms": elapsed_ms,
                "metadata": image_metadata(annotated),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "yolo": yolo_status()}), 500


@api_bp.post("/process/<method>")
def process_image(method: str):
    """Apply one selected preprocessing method to an uploaded image/PDF page."""
    if method not in PROCESSORS:
        return jsonify({"error": "unknown processing method"}), 400

    try:
        image = read_image_from_request()
        started = time.perf_counter()
        processor = PROCESSORS[method]

        # Only pass form parameters accepted by the selected processor.  This
        # lets one frontend parameter panel serve many image-processing methods.
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
    """Run OCR or PDF text extraction and return structured invoice fields."""
    try:
        filename, mimetype, raw = read_file_from_request()
        engine = request.form.get("engine", "tesseract")
        lang = request.form.get("lang", "chi_sim+eng")
        psm = int(request.form.get("psm", 6))
        page_number = max(1, int(request.form.get("page", 1)))
        yolo_mode = request.form.get("use_yolo", "false").lower()
        force_yolo = yolo_mode in {"true", "1", "yes", "yolo", "force"}
        auto_yolo = yolo_mode == "auto"
        yolo_confidence = float(request.form.get("yolo_confidence", 0.35))
        detector = request.form.get("detector", "yolo")
        detector_key = detector if detector in {"yolo", "faster_rcnn", "dfine_l"} else "yolo"
        detector_label = detector_status(detector_key)["label"]
        if engine not in {"paddle", "tesseract"}:
            return jsonify({"error": "unknown OCR engine"}), 400
        started = time.perf_counter()

        def run_standard_ocr(image):
            return recognize_invoice_paddle(image) if engine == "paddle" else recognize_invoice(image, lang=lang, psm=psm)

        def run_image_ocr_with_mode(image):
            if force_yolo:
                detections = detect_fields(image, detector=detector_key, confidence=yolo_confidence)
                result = recognize_invoice_with_yolo(
                    image,
                    confidence=yolo_confidence,
                    detections=detections,
                    detector_label=detector_label,
                )
                result["yolo_mode"] = "force"
                result["detector"] = detector_status(detector_key)
                result["yolo_decision"] = f"手动启用 {detector_label} 辅助定位"
                return result

            standard_result = run_standard_ocr(image)
            if not auto_yolo:
                standard_result["yolo_mode"] = "off"
                standard_result["yolo_decision"] = "未启用 YOLO"
                return standard_result

            reason = yolo_decision_reason(standard_result.get("text", ""))
            if not is_yolo_suitable_invoice(standard_result.get("text", "")):
                standard_result["yolo_mode"] = "auto_skipped"
                standard_result["yolo_decision"] = reason
                return standard_result

            try:
                detections = detect_fields(image, detector=detector_key, confidence=yolo_confidence)
                result = recognize_invoice_with_yolo(
                    image,
                    confidence=yolo_confidence,
                    base_text=standard_result.get("text", ""),
                    detections=detections,
                    detector_label=detector_label,
                )
                result["base_ocr"] = {
                    "engine": standard_result.get("engine"),
                    "average_confidence": standard_result.get("average_confidence"),
                    "completion_score": standard_result.get("completion_score"),
                }
                result["yolo_mode"] = "auto"
                result["detector"] = detector_status(detector_key)
                result["yolo_decision"] = f"{reason}，使用 {detector_label}"
                return result
            except RuntimeError as exc:
                standard_result["yolo_mode"] = "auto_fallback"
                standard_result["yolo_decision"] = f"{reason}；{detector_label} 不可用，已回退普通 OCR：{exc}"
                return standard_result

        if is_pdf_file(filename, mimetype, raw):
            # Electronic PDF invoices often contain real text.  Direct text
            # extraction is faster and more accurate than OCR when available.
            text, page_count = extract_pdf_text(raw, page_number=page_number)
            if len("".join(text.split())) >= 30:
                fields = parse_invoice_text(text)
                result = {
                    "text": text,
                    "fields": fields,
                    "average_confidence": 100,
                    "completion_score": completion_score(fields),
                    "lang": "pdf",
                    "psm": None,
                    "variant": "pdf_text",
                    "engine": "pdf_text",
                    "score": round(completion_score(fields) * 100 + 35, 2),
                    "metadata": {"file_type": "pdf", "pages": page_count, "source_page": min(page_number, page_count)},
                    "yolo_mode": "skipped_pdf_text",
                    "yolo_decision": "PDF 已抽取到可用文本，跳过 YOLO",
                }
            else:
                # Scanned PDFs have little or no embedded text, so the selected
                # page is rendered to an image and sent through the OCR pipeline.
                image, page_count = pdf_first_page_to_image(raw, page_number=page_number)
                result = run_image_ocr_with_mode(image)
                result["metadata"] = {**image_metadata(image), "file_type": "pdf", "pages": page_count, "source_page": min(page_number, page_count)}
        else:
            image = image_from_bytes(raw)
            result = run_image_ocr_with_mode(image)
            result["metadata"] = image_metadata(image)

        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)

        # Validation does not replace manual review; it marks missing or
        # suspicious fields so the frontend can show what needs confirmation.
        result["field_checks"] = validate_fields(result.get("fields", {}))
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
