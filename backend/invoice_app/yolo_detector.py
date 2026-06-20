"""YOLO field-region detection for invoice OCR.

This module is optional.  The normal PaddleOCR/Tesseract flow still works when
no YOLO model is installed.  After a trained invoice field detector is placed at
``backend/models/invoice_yolo.pt`` (or ``YOLO_INVOICE_MODEL`` is set), these
helpers can locate field regions, crop them, OCR each crop and merge the result
with the regular parser output.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_utils import image_metadata, image_to_base64
from .layout_parser import OCRLine, line_to_dict, lines_text, refine_fields_with_layout
from .paddle_ocr import get_paddle_engine, iter_ocr_items, iter_text_scores, prepare_image
from .parser import completion_score, parse_invoice_text


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "invoice_yolo.pt"
PROJECT_MODEL_PATH = Path(__file__).resolve().parents[2] / "model_results_export" / "yolo" / "best_yolo_mAP708.pt"
YOLO_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".ultralytics"

# Ultralytics writes a settings file during import.  Point it at a project-local
# directory so macOS permission issues in Application Support do not break demos.
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

FIELD_LABELS = {
    "invoice_type": "发票类型",
    "invoice_code": "发票代码",
    "invoice_no": "发票号码",
    "invoice_number": "发票号码",
    "date": "开票日期",
    "invoice_date": "开票日期",
    "checksum": "校验码",
    "check_code": "校验码",
    "buyer_name": "购买方名称",
    "buyer_tax_id": "购买方税号",
    "seller_name": "销售方名称",
    "seller_tax_id": "销售方税号",
    "amount": "金额",
    "tax": "税额",
    "tax_amount": "税额",
    "total": "价税合计",
    "total_amount": "价税合计",
    "drawer": "开票人",
    "issuer": "开票人",
    "发票类型": "发票类型",
    "发票代码": "发票代码",
    "发票号码": "发票号码",
    "开票日期": "开票日期",
    "校验码": "校验码",
    "购买方名称": "购买方名称",
    "购买方税号": "购买方税号",
    "销售方名称": "销售方名称",
    "销售方税号": "销售方税号",
    "金额": "金额",
    "税额": "税额",
    "价税合计": "价税合计",
    "开票人": "开票人",
}

FIELD_OUTPUT_ORDER = [
    "发票类型",
    "发票代码",
    "发票号码",
    "开票日期",
    "校验码",
    "购买方名称",
    "购买方税号",
    "销售方名称",
    "销售方税号",
    "金额",
    "税额",
    "价税合计",
    "开票人",
]

ELECTRONIC_INVOICE_WORDS = ("电子发票", "数电", "全电")
SELLER_NAME_NOISE_WORDS = (
    "开票单位",
    "开票人",
    "收款人",
    "复核",
    "开户行",
    "银行",
    "账号",
    "发票专用章",
    "专用章",
    "印制有限公司",
    "税总函",
)


@dataclass(frozen=True)
class Detection:
    """One YOLO field-region detection."""

    label: str
    field: str
    confidence: float
    box: Tuple[int, int, int, int]


@dataclass
class CropResult:
    """OCR result for one detected invoice field crop."""

    detection: Detection
    text: str
    ocr_confidence: float


_YOLO_MODEL = None
_YOLO_MODEL_PATH = ""


def yolo_model_path() -> str:
    """Return configured YOLO model path."""
    configured = os.environ.get("YOLO_INVOICE_MODEL")
    if configured:
        return configured
    if PROJECT_MODEL_PATH.exists():
        return str(PROJECT_MODEL_PATH)
    return str(DEFAULT_MODEL_PATH)


def get_yolo_model():
    """Load a YOLO model lazily.

    The import is inside the function so the project can run without
    ``ultralytics`` until the YOLO feature is used.
    """
    global _YOLO_MODEL, _YOLO_MODEL_PATH
    model_path = yolo_model_path()
    if _YOLO_MODEL is not None and _YOLO_MODEL_PATH == model_path:
        return _YOLO_MODEL

    if not Path(model_path).exists():
        raise RuntimeError(f"未找到 YOLO 发票字段模型：{model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 ultralytics，请先执行 pip install ultralytics") from exc

    _YOLO_MODEL = YOLO(model_path)
    _YOLO_MODEL_PATH = model_path
    return _YOLO_MODEL


def normalize_label(label: str) -> str:
    """Map YOLO class names to project field names."""
    key = str(label).strip()
    return FIELD_LABELS.get(key, FIELD_LABELS.get(key.lower(), key))


def is_yolo_suitable_invoice(text: str) -> bool:
    """Return whether YOLO should assist this invoice automatically.

    The current training set mainly contains scanned/mobile-shot VAT ordinary
    invoices.  Electronic invoices and text-based PDFs are better handled by
    direct text extraction plus the rule parser, so auto mode leaves them on
    the normal OCR/PDF path.
    """
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    if any(word in compact for word in ELECTRONIC_INVOICE_WORDS):
        return False
    return bool(re.search(r"增.{0,2}税普通发票", compact)) or ("普通发票" in compact and "发票代码" in compact)


def yolo_decision_reason(text: str) -> str:
    """Explain the auto-YOLO routing decision for frontend history/debugging."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return "未识别到足够表头文本，使用普通 OCR"
    if any(word in compact for word in ELECTRONIC_INVOICE_WORDS):
        return "电子发票/数电票，使用普通 OCR 或 PDF 文本直读"
    if is_yolo_suitable_invoice(compact):
        return "识别为扫描/拍照版增值税普通发票，启用 YOLO 辅助定位"
    return "非 YOLO 训练主类型，使用普通 OCR"


def clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clip a floating YOLO xyxy box to image bounds."""
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


def detect_invoice_fields(image: np.ndarray, confidence: float = 0.35) -> List[Detection]:
    """Run YOLO on one invoice image and return normalized detections."""
    model = get_yolo_model()
    height, width = image.shape[:2]
    results = model.predict(image, conf=float(confidence), verbose=False)
    detections: List[Detection] = []

    for result in results:
        names = getattr(result, "names", {}) or getattr(model, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for item in boxes:
            cls_index = int(item.cls[0])
            label = str(names.get(cls_index, cls_index))
            field = normalize_label(label)
            xyxy = item.xyxy[0].tolist()
            score = round(float(item.conf[0]) * 100, 2)
            detections.append(Detection(label=label, field=field, confidence=score, box=clip_box(tuple(xyxy), width, height)))

    detections.sort(key=lambda det: (det.box[1], det.box[0]))
    return detections


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw YOLO field boxes on a copy of the invoice image."""
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(output, (x1, y1), (x2, y2), (68, 118, 157), 2)
        cv2.putText(
            output,
            f"{det.field} {det.confidence:.0f}%",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (68, 118, 157),
            2,
            cv2.LINE_AA,
        )
    return output


def crop_region(image: np.ndarray, box: Tuple[int, int, int, int], padding: int = 6) -> np.ndarray:
    """Crop a detected field region with small padding."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return image[y1:y2, x1:x2]


def ocr_crop_text(crop: np.ndarray) -> Tuple[str, float]:
    """Recognize text in a cropped field region with PaddleOCR."""
    engine = get_paddle_engine()
    raw_result = engine.ocr(prepare_image(crop))
    lines: List[str] = []
    scores: List[float] = []
    for text, score in iter_text_scores(raw_result):
        clean = str(text).strip()
        if clean:
            lines.append(clean)
            if score > 0:
                scores.append(score)
    confidence = round(sum(scores) / len(scores), 2) if scores else 0.0
    return "\n".join(lines), confidence


def ocr_full_page_lines(image: np.ndarray) -> Tuple[str, List[OCRLine], float, Tuple[int, int, int]]:
    """Run full-page PaddleOCR and keep line coordinates for layout refinement."""
    engine = get_paddle_engine()
    prepared = prepare_image(image)
    raw_result = engine.ocr(prepared)
    lines: List[str] = []
    ocr_lines: List[OCRLine] = []
    scores: List[float] = []
    for text, score, box in iter_ocr_items(raw_result):
        clean = str(text).strip()
        if not clean:
            continue
        lines.append(clean)
        if score > 0:
            scores.append(score)
        if box:
            ocr_lines.append(OCRLine(text=clean, score=score, box=box))
    text = lines_text(ocr_lines) if ocr_lines else "\n".join(lines)
    confidence = round(sum(scores) / len(scores), 2) if scores else 0.0
    return text, ocr_lines, confidence, prepared.shape


def center(det: Detection) -> Tuple[float, float]:
    """Return center point of a detection box."""
    x1, y1, x2, y2 = det.box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_height(det: Detection) -> int:
    """Return detection box height."""
    return det.box[3] - det.box[1]


def has_money_text(text: str) -> bool:
    """Check whether OCR crop text contains a money-like value."""
    return bool(re.search(r"[¥￥]?\s*\d{1,12}(?:,\d{3})*(?:\.\d{1,2})", text or ""))


def has_tax_id_text(text: str) -> bool:
    """Check whether OCR crop text contains a taxpayer ID-like value."""
    return bool(re.search(r"(?<![0-9A-Z])[0-9A-Z]{15,20}(?![0-9A-Z])", text or "", flags=re.IGNORECASE))


def has_checksum_text(text: str) -> bool:
    """Check whether OCR crop text contains a checksum-like digit sequence."""
    digits = re.sub(r"\D", "", text or "")
    return 12 <= len(digits) <= 30


def clean_crop_text(text: str) -> str:
    """Normalize OCR crop text before it is sent to the rule parser."""
    lines = [line.strip(" :：,，;；|") for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def is_noisy_seller_name(text: str) -> bool:
    """Reject seal, bank and drawer text as seller names."""
    compact = re.sub(r"\s+", "", text or "")
    return any(word in compact for word in SELLER_NAME_NOISE_WORDS)


def crop_result_dict(item: CropResult) -> Dict[str, Any]:
    """Serialize one crop result for the API response."""
    det = item.detection
    return {
        "label": det.label,
        "field": det.field,
        "confidence": det.confidence,
        "box": list(det.box),
        "text": item.text,
        "ocr_confidence": item.ocr_confidence,
    }


def group_crop_results(crops: List[CropResult]) -> Dict[str, List[CropResult]]:
    """Group crop OCR results by normalized field name."""
    grouped: Dict[str, List[CropResult]] = {}
    for item in crops:
        grouped.setdefault(item.detection.field, []).append(item)
    return grouped


def choose_by_score(candidates: List[CropResult], scorer) -> Optional[CropResult]:
    """Return the best crop according to a scoring function."""
    if not candidates:
        return None
    return max(candidates, key=scorer)


def base_crop_score(item: CropResult) -> float:
    """General crop score combining detector confidence and OCR presence."""
    return item.detection.confidence + (15 if item.text else 0) + item.ocr_confidence * 0.05


def choose_total(candidates: List[CropResult], image_height: int) -> Optional[CropResult]:
    """Choose the total amount box, preferring the lower small-amount row."""
    def score(item: CropResult) -> float:
        _, cy = center(item.detection)
        return base_crop_score(item) + cy / max(image_height, 1) * 35 + (25 if has_money_text(item.text) else 0)

    return choose_by_score(candidates, score)


def choose_money_field(candidates: List[CropResult], total: Optional[CropResult], image_height: int) -> Optional[CropResult]:
    """Choose amount/tax near the total row, not upper detail-line values."""
    if not candidates:
        return None
    total_y = center(total.detection)[1] if total else image_height * 0.75

    def score(item: CropResult) -> float:
        _, cy = center(item.detection)
        above_total = cy < total_y
        distance_score = max(0.0, 30 - abs(total_y - cy) / max(image_height, 1) * 100)
        return (
            base_crop_score(item)
            + distance_score
            + (20 if above_total else -25)
            + (20 if has_money_text(item.text) else 0)
            + cy / max(image_height, 1) * 10
        )

    return choose_by_score(candidates, score)


def choose_positioned_field(
    candidates: List[CropResult],
    *,
    above: Optional[CropResult] = None,
    below: Optional[CropResult] = None,
    prefer_upper: bool = False,
    prefer_lower: bool = False,
    require_tax_id: bool = False,
    reject_seller_noise: bool = False,
    image_height: int = 1,
) -> Optional[CropResult]:
    """Choose one field crop using simple document-layout constraints."""
    filtered = []
    for item in candidates:
        cy = center(item.detection)[1]
        if above and cy >= center(above.detection)[1] + box_height(above.detection) * 0.2:
            continue
        if below and cy <= center(below.detection)[1] - box_height(below.detection) * 0.2:
            continue
        if require_tax_id and not has_tax_id_text(item.text):
            continue
        if reject_seller_noise and is_noisy_seller_name(item.text):
            continue
        filtered.append(item)
    if not filtered:
        filtered = [item for item in candidates if not (reject_seller_noise and is_noisy_seller_name(item.text))]
    if not filtered:
        return None

    def score(item: CropResult) -> float:
        _, cy = center(item.detection)
        value = base_crop_score(item)
        if prefer_upper:
            value += (1 - cy / max(image_height, 1)) * 20
        if prefer_lower:
            value += cy / max(image_height, 1) * 20
        if require_tax_id and has_tax_id_text(item.text):
            value += 30
        return value

    return choose_by_score(filtered, score)


def resolve_layout_fields(crops: List[CropResult], image_shape: Tuple[int, int]) -> List[Tuple[str, CropResult]]:
    """Apply invoice layout constraints to YOLO crop OCR results.

    The detector supplies candidate regions.  This function decides which
    candidate should become each final field line before handing text to the
    existing regex/rule parser.
    """
    image_height, _ = image_shape[:2]
    grouped = group_crop_results(crops)
    chosen: Dict[str, CropResult] = {}

    for field in ("发票类型", "发票代码", "发票号码", "开票日期", "开票人"):
        item = choose_positioned_field(grouped.get(field, []), prefer_upper=field != "开票人", image_height=image_height)
        if item:
            chosen[field] = item

    checksum = choose_positioned_field(grouped.get("校验码", []), image_height=image_height)
    if checksum and has_checksum_text(checksum.text):
        chosen["校验码"] = checksum

    total = choose_total(grouped.get("价税合计", []), image_height)
    if total:
        chosen["价税合计"] = total

    amount = choose_money_field(grouped.get("金额", []), total, image_height)
    tax = choose_money_field(grouped.get("税额", []), total, image_height)
    if amount and tax:
        amount_x, amount_y = center(amount.detection)
        tax_x, tax_y = center(tax.detection)
        same_row = abs(amount_y - tax_y) <= max(box_height(amount.detection), box_height(tax.detection)) * 1.5
        if same_row and amount_x > tax_x:
            # For VAT invoice summary rows, amount is left of tax.  If YOLO
            # class names are crossed but geometry is clear, swap them before
            # generating parser input.
            amount, tax = tax, amount
    if amount:
        chosen["金额"] = amount
    if tax:
        chosen["税额"] = tax

    buyer_tax = choose_positioned_field(grouped.get("购买方税号", []), require_tax_id=True, prefer_upper=True, image_height=image_height)
    buyer_name = choose_positioned_field(grouped.get("购买方名称", []), above=buyer_tax, prefer_upper=True, image_height=image_height)
    if buyer_name:
        chosen["购买方名称"] = buyer_name
    if buyer_tax:
        chosen["购买方税号"] = buyer_tax

    seller_name = choose_positioned_field(
        grouped.get("销售方名称", []),
        below=total,
        prefer_lower=True,
        reject_seller_noise=True,
        image_height=image_height,
    )
    seller_tax = choose_positioned_field(
        grouped.get("销售方税号", []),
        below=seller_name or total,
        require_tax_id=True,
        prefer_lower=True,
        image_height=image_height,
    )
    if seller_name:
        chosen["销售方名称"] = seller_name
    if seller_tax:
        chosen["销售方税号"] = seller_tax

    ordered = [(field, chosen[field]) for field in FIELD_OUTPUT_ORDER if field in chosen and clean_crop_text(chosen[field].text)]
    used_ids = {id(item) for _, item in ordered}
    leftovers = [item for item in crops if id(item) not in used_ids and clean_crop_text(item.text)]
    leftovers.sort(key=lambda item: (item.detection.box[1], item.detection.box[0], -item.detection.confidence))
    return ordered + [(item.detection.field, item) for item in leftovers]


def build_detected_text(crops: List[Tuple[str, CropResult]], base_text: str = "") -> str:
    """Build parser input from layout-resolved YOLO crops plus optional OCR text."""
    lines = []
    for field, item in crops:
        text = clean_crop_text(item.text)
        if text:
            lines.append(f"{field}:{text}")
    if base_text:
        # Keep full-page OCR as fallback evidence after field crops.  Field
        # labels from crops appear first, so parser regexes prefer them.
        lines.append("OCR全文:")
        lines.append(base_text)
    return "\n".join(lines)


def recognize_invoice_with_yolo(
    image: np.ndarray,
    confidence: float = 0.35,
    base_text: str = "",
    detections: Optional[List[Detection]] = None,
    detector_label: str = "YOLO",
) -> Dict[str, Any]:
    """Detect invoice field regions, OCR each crop and return structured fields."""
    detections = detections if detections is not None else detect_invoice_fields(image, confidence=confidence)
    full_text, full_page_lines, full_page_confidence, full_page_shape = ocr_full_page_lines(image)
    crop_ocr_results: List[CropResult] = []
    crop_results: List[Dict[str, Any]] = []
    crop_scores: List[float] = []

    for det in detections:
        text, score = ocr_crop_text(crop_region(image, det.box))
        crop = CropResult(detection=det, text=clean_crop_text(text), ocr_confidence=score)
        crop_ocr_results.append(crop)
        if score > 0:
            crop_scores.append(score)
        crop_results.append(crop_result_dict(crop))

    resolved_crops = resolve_layout_fields(crop_ocr_results, image.shape)
    fallback_text = base_text or full_text
    text = build_detected_text(resolved_crops, base_text=fallback_text)
    fields = parse_invoice_text(text)
    if full_page_lines:
        fields = refine_fields_with_layout(fields, full_page_lines, full_page_shape)
    confidence_value = round(sum(crop_scores) / len(crop_scores), 2) if crop_scores else full_page_confidence
    annotated = draw_detections(image, detections)
    return {
        "text": text,
        "fields": fields,
        "average_confidence": confidence_value,
        "completion_score": completion_score(fields),
        "lang": "ch",
        "psm": None,
        "variant": "detector_field_crop",
        "engine": f"{detector_label}+paddle",
        "score": round(completion_score(fields) * 100 + confidence_value * 0.35, 2),
        "detections": crop_results,
        "ocr_lines": [line_to_dict(line) for line in full_page_lines],
        "resolved_fields": [
            {
                "field": field,
                "source_field": item.detection.field,
                "confidence": item.detection.confidence,
                "box": list(item.detection.box),
                "text": item.text,
            }
            for field, item in resolved_crops
        ],
        "detection_count": len(detections),
        "image": image_to_base64(annotated),
        "metadata": image_metadata(annotated),
    }


def yolo_status() -> Dict[str, Any]:
    """Return YOLO availability information for health/debugging."""
    model_path = yolo_model_path()
    try:
        import ultralytics  # noqa: F401

        has_package = True
    except ImportError:
        has_package = False
    return {
        "model_path": model_path,
        "model_exists": Path(model_path).exists(),
        "ultralytics_installed": has_package,
    }
