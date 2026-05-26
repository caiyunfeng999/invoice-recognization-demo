from typing import Any, Dict

import pytesseract

from .parser import completion_score, parse_invoice_text
from .preprocessing import to_gray


def average_confidence(data: Dict[str, Any]) -> float:
    values = []
    for raw in data.get("conf", []):
        try:
            score = float(raw)
        except ValueError:
            continue
        if score >= 0:
            values.append(score)
    return round(sum(values) / len(values), 2) if values else 0.0


def recognize_invoice(image, lang: str = "chi_sim+eng", psm: int = 6) -> Dict[str, Any]:
    gray_image = to_gray(image)
    config = f"--psm {int(psm)}"
    text = pytesseract.image_to_string(gray_image, lang=lang, config=config)
    data = pytesseract.image_to_data(gray_image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    fields = parse_invoice_text(text)
    return {
        "text": text,
        "fields": fields,
        "average_confidence": average_confidence(data),
        "completion_score": completion_score(fields),
        "lang": lang,
        "psm": int(psm),
    }
