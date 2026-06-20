"""PaddleOCR recognition module.

PaddleOCR is the primary OCR engine for Chinese invoices.  It is loaded lazily
because model initialization is expensive.  The output is normalized into the
same structure used by the Tesseract module so the route/frontend code can stay
engine-independent.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .layout_parser import OCRLine, line_to_dict, lines_text, refine_fields_with_layout
from .parser import completion_score, parse_invoice_text
from .preprocessing import to_gray


os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

_OCR_ENGINE = None


def get_paddle_engine():
    """Create or reuse a singleton PaddleOCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 paddleocr，请先安装 paddleocr 和 paddlepaddle") from exc

    init_options = (
        # Mobile models are lighter and more suitable for local course demos.
        {
            "lang": "ch",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": "ch",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {"lang": "ch"},
        {},
    )
    last_error = None
    for options in init_options:
        try:
            _OCR_ENGINE = PaddleOCR(**options)
            return _OCR_ENGINE
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"PaddleOCR 初始化失败：{last_error}") from last_error


def prepare_image(image):
    """Prepare invoice image for PaddleOCR by resizing and enhancing contrast."""
    gray = to_gray(image)
    height, width = gray.shape[:2]
    scale = 2 if max(height, width) < 1800 else 1
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    max_side = max(gray.shape[:2])
    if max_side > 2600:
        ratio = 2600 / max_side
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)


def normalize_confidence(value: Any) -> float:
    """Convert PaddleOCR confidence to a 0-100 scale."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score * 100 if score <= 1 else score


def box_to_xyxy(box: Any) -> Optional[Tuple[float, float, float, float]]:
    """Convert PaddleOCR polygon/box formats to xyxy."""
    if box is None:
        return None
    try:
        array = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size < 4:
        return None
    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4].tolist()
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    array = array.reshape(-1, 2)
    xs = array[:, 0]
    ys = array[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def first_present(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return the first non-empty PaddleOCR result value for a set of keys."""
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            if len(value) == 0:
                continue
        except TypeError:
            pass
        return value
    return []


def iter_ocr_items(result: Any) -> Iterable[Tuple[str, float, Optional[Tuple[float, float, float, float]]]]:
    """Yield recognized text, confidence and optional box from PaddleOCR output."""
    if not result:
        return

    if isinstance(result, dict):
        texts = first_present(result, ("rec_texts", "texts"))
        scores = first_present(result, ("rec_scores", "scores"))
        boxes = first_present(result, ("rec_boxes", "rec_polys", "dt_polys", "det_polys", "boxes"))
        for index, (text, score) in enumerate(zip(texts, scores)):
            if text:
                box = boxes[index] if index < len(boxes) else None
                yield str(text), normalize_confidence(score), box_to_xyxy(box)
        single_text = result.get("text")
        if single_text:
            yield str(single_text), normalize_confidence(result.get("score", 0)), box_to_xyxy(result.get("box"))
        return

    if isinstance(result, (list, tuple)):
        if len(result) == 2 and isinstance(result[1], (list, tuple)):
            payload = result[1]
            if payload and isinstance(payload[0], str):
                score = payload[1] if len(payload) > 1 else 0
                yield payload[0], normalize_confidence(score), box_to_xyxy(result[0])
                return
        for item in result:
            yield from iter_ocr_items(item)


def iter_text_scores(result: Any) -> Iterable[Tuple[str, float]]:
    """Yield recognized text and confidence from different PaddleOCR result formats."""
    for text, score, _ in iter_ocr_items(result):
        yield text, score


def recognize_invoice_paddle(image) -> Dict[str, Any]:
    """Run PaddleOCR and parse recognized text into invoice fields."""
    engine = get_paddle_engine()
    rgb_image = prepare_image(image)
    raw_result = engine.ocr(rgb_image)

    lines: List[str] = []
    ocr_lines: List[OCRLine] = []
    scores: List[float] = []
    for text, score, box in iter_ocr_items(raw_result):
        clean_text = text.strip()
        if clean_text:
            lines.append(clean_text)
            if score > 0:
                scores.append(score)
            if box:
                ocr_lines.append(OCRLine(text=clean_text, score=score, box=box))

    text = lines_text(ocr_lines) if ocr_lines else "\n".join(lines)
    fields = parse_invoice_text(text)
    if ocr_lines:
        fields = refine_fields_with_layout(fields, ocr_lines, rgb_image.shape)
    confidence = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {
        "text": text,
        "fields": fields,
        "average_confidence": confidence,
        "completion_score": completion_score(fields),
        "lang": "ch",
        "psm": None,
        "variant": "paddle_clahe",
        "engine": "paddle",
        "score": round(completion_score(fields) * 100 + confidence * 0.35, 2),
        "ocr_lines": [line_to_dict(line) for line in ocr_lines],
    }
