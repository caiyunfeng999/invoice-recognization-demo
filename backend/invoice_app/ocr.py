"""Tesseract OCR module.

Tesseract is kept as a lightweight fallback and comparison engine.  The module
tries multiple preprocessing variants and chooses the candidate with the best
combination of field completeness and OCR confidence.
"""

from typing import Any, Dict

import cv2
import pytesseract

from .parser import completion_score, parse_invoice_text
from .preprocessing import to_gray


def average_confidence(data: Dict[str, Any]) -> float:
    """Calculate average confidence from pytesseract image_to_data output."""
    values = []
    for raw in data.get("conf", []):
        try:
            score = float(raw)
        except ValueError:
            continue
        if score >= 0:
            values.append(score)
    return round(sum(values) / len(values), 2) if values else 0.0


def ocr_variants(image):
    """Yield image variants to improve OCR robustness on different invoices."""
    gray_image = to_gray(image)
    yield "gray", gray_image

    height, width = gray_image.shape[:2]
    scale = 2 if max(height, width) < 1800 else 1
    if scale > 1:
        upscaled = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield "upscaled", upscaled

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_image)
    yield "clahe", clahe

    binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield "clahe_otsu", binary

    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    yield "adaptive", adaptive


def score_result(fields: Dict[str, str], confidence: float, text: str) -> float:
    """Score one OCR result by key field coverage, completeness and confidence."""
    key_fields = ["发票代码", "发票号码", "开票日期", "价税合计", "购买方税号", "购买方名称"]
    key_score = sum(1 for field in key_fields if fields.get(field)) * 18
    return key_score + completion_score(fields) * 60 + confidence * 0.25 + min(len(text), 800) * 0.01


def recognize_single(image, lang: str, psm: int, variant: str) -> Dict[str, Any]:
    """Run Tesseract once on one image variant and parse structured fields."""
    config = f"--psm {int(psm)}"
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    fields = parse_invoice_text(text)
    confidence = average_confidence(data)
    return {
        "text": text,
        "fields": fields,
        "average_confidence": confidence,
        "completion_score": completion_score(fields),
        "lang": lang,
        "psm": int(psm),
        "variant": variant,
        "score": score_result(fields, confidence, text),
    }


def recognize_invoice(image, lang: str = "chi_sim+eng", psm: int = 6) -> Dict[str, Any]:
    """Run all Tesseract variants and return the best structured result."""
    candidates = [
        recognize_single(variant_image, lang, psm, variant)
        for variant, variant_image in ocr_variants(image)
    ]
    best = max(candidates, key=lambda item: item["score"])
    best["candidates"] = [
        {
            "variant": item["variant"],
            "average_confidence": item["average_confidence"],
            "completion_score": item["completion_score"],
            "score": round(item["score"], 2),
        }
        for item in candidates
    ]
    best["score"] = round(best["score"], 2)
    best["engine"] = "tesseract"
    return best
