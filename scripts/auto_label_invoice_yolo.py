"""Auto pre-label invoice field boxes for YOLO training.

This script creates a first-pass YOLO dataset from invoice images using
PaddleOCR text boxes plus invoice-specific rules.  It is intended for
pre-labeling: review and correct the labels in Roboflow/LabelImg before serious
training.

Usage:
    conda activate pyhon11-opencv
    cd ~/invoice_ocr_web
    python scripts/auto_label_invoice_yolo.py \
        --source test_samples/github_public_20 \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

import sys

sys.path.insert(0, str(BACKEND))

from invoice_app.paddle_ocr import get_paddle_engine, prepare_image  # noqa: E402


CLASS_NAMES = [
    "invoice_type",
    "invoice_code",
    "invoice_no",
    "invoice_date",
    "checksum",
    "buyer_name",
    "buyer_tax_id",
    "seller_name",
    "seller_tax_id",
    "amount",
    "tax",
    "total",
    "drawer",
]
CLASS_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

COMPANY_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()]{2,}"
    r"(?:公司|研究所|研究院|税务局|酒店|宾馆|银行|学校|大学|医院|中心|集团|厂|商行|合作社)"
)
MONEY_RE = re.compile(r"[¥￥]?\s*([0-9]{1,10}(?:\.[0-9]{2})?)")
TAX_TOKEN_RE = re.compile(r"(?<![0-9A-Z])[0-9A-Z]{15,20}(?![0-9A-Z])")
MONEY_VALUE_RE = re.compile(r"\d+\.\d{2}")


@dataclass
class OcrItem:
    text: str
    score: float
    box: Tuple[int, int, int, int]

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


def image_files(source: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in exts)


def poly_to_box(poly) -> Tuple[int, int, int, int]:
    points = np.array(poly).reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def ocr_items(image) -> List[OcrItem]:
    engine = get_paddle_engine()
    result = engine.ocr(prepare_image(image))
    items: List[OcrItem] = []
    for page in result or []:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or page.get("texts") or []
        scores = page.get("rec_scores") or page.get("scores") or []
        polys = page.get("dt_polys") or page.get("rec_polys") or []
        for text, score, poly in zip(texts, scores, polys):
            clean = str(text).strip()
            if not clean:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            items.append(OcrItem(clean, confidence, poly_to_box(poly)))
    return items


def normalize(text: str) -> str:
    return text.replace(" ", "").replace("：", ":").replace("￥", "¥").upper()


def has_money(text: str) -> bool:
    return bool(MONEY_VALUE_RE.search(normalize(text)))


def money_value(text: str) -> Optional[float]:
    match = MONEY_VALUE_RE.search(normalize(text))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if 0 < value < 1_000_000_000 else None


def has_tax_id(text: str) -> bool:
    compact = normalize(text)
    for token in TAX_TOKEN_RE.findall(compact):
        if token.startswith("20") and token.isdigit():
            continue
        if re.search(r"[A-Z]", token) or len(token) in (15, 18, 20):
            return True
    return False


def nearby_value(
    items: List[OcrItem],
    label_words: Iterable[str],
    pattern: re.Pattern,
    y_window: int = 80,
    allow_below: bool = True,
) -> Optional[OcrItem]:
    labels = [item for item in items if any(word in item.text for word in label_words)]
    best = None
    best_distance = 10**9
    for label in labels:
        for item in items:
            if item is label:
                continue
            same_line = abs(item.cy - label.cy) <= y_window and item.cx >= label.cx - 40
            below_label = allow_below and 0 < item.cy - label.cy <= y_window * 2 and abs(item.cx - label.cx) <= 320
            if not same_line and not below_label:
                continue
            if not pattern.search(normalize(item.text)):
                continue
            distance = abs(item.cy - label.cy) + max(0, item.cx - label.cx) + (0 if same_line else 120)
            if distance < best_distance:
                best = item
                best_distance = distance
    return best


def first_by(items: List[OcrItem], predicate, sort_key) -> Optional[OcrItem]:
    matches = [item for item in items if predicate(item)]
    if not matches:
        return None
    return sorted(matches, key=sort_key)[0]


def column_value(items: List[OcrItem], label_predicate, value_predicate, height: int) -> Optional[OcrItem]:
    labels = [item for item in items if label_predicate(item)]
    candidates = [item for item in items if value_predicate(item)]
    best = None
    best_score = 10**9
    for label in labels:
        for candidate in candidates:
            if candidate is label or candidate.cy <= label.cy:
                continue
            if candidate.cy - label.cy > height * 0.42:
                continue
            x_distance = abs(candidate.cx - label.cx)
            if x_distance > 150:
                continue
            score = x_distance * 2 + abs(candidate.cy - label.cy)
            if score < best_score:
                best = candidate
                best_score = score
    return best


def choose_money(items: List[OcrItem], height: int) -> Dict[str, OcrItem]:
    money_items = []
    for item in items:
        text = normalize(item.text)
        match = MONEY_RE.fullmatch(text) or re.search(r"[¥￥]\s*\d", item.text)
        value = money_value(text)
        if match and value is not None:
            money_items.append((value, item))

    result: Dict[str, OcrItem] = {}
    total_inline = first_by(
        items,
        lambda item: ("小写" in item.text or "价税合计" in item.text) and bool(re.search(r"\d+\.\d{2}", normalize(item.text))),
        lambda item: (-item.cy, -item.cx),
    )
    total_near = total_inline or nearby_value(items, ("价税合计", "小写"), re.compile(r"\d+\.\d{2}"), y_window=120)
    if total_near:
        result["total"] = total_near
    elif money_items:
        result["total"] = max(money_items, key=lambda pair: pair[0])[1]

    amount_by_column = column_value(
        items,
        lambda item: "金额" in item.text and "价税合计" not in item.text,
        lambda item: has_money(item.text),
        height,
    )
    if amount_by_column and amount_by_column is not result.get("total"):
        result["amount"] = amount_by_column

    tax_by_column = column_value(
        items,
        lambda item: "税额" in item.text,
        lambda item: has_money(item.text),
        height,
    )
    if tax_by_column and tax_by_column is not result.get("total") and tax_by_column is not result.get("amount"):
        result["tax"] = tax_by_column

    if len(money_items) >= 2:
        candidates = sorted(money_items, key=lambda pair: (pair[1].cy, pair[1].cx))
        lower = [pair for pair in candidates if pair[1].cy > height * 0.35]
        if lower and "amount" not in result:
            amount_candidate = max(lower, key=lambda pair: pair[0])
            if "total" not in result or amount_candidate[1] is not result["total"]:
                result["amount"] = amount_candidate[1]
        if len(lower) >= 2 and "tax" not in result:
            tax_candidate = min(lower, key=lambda pair: pair[0])
            if tax_candidate[1] is not result.get("total"):
                result["tax"] = tax_candidate[1]
    return result


def auto_labels(items: List[OcrItem], width: int, height: int) -> Dict[str, OcrItem]:
    labels: Dict[str, OcrItem] = {}

    title = first_by(
        items,
        lambda item: "发票" in item.text and item.cy < height * 0.22 and len(item.text) >= 4,
        lambda item: (item.cy, abs(item.cx - width / 2)),
    )
    if title:
        labels["invoice_type"] = title

    code = nearby_value(items, ("发票代码", "代码"), re.compile(r"^\d{10,12}$"))
    if not code:
        code = first_by(
            items,
            lambda item: bool(re.fullmatch(r"\d{10,12}", normalize(item.text))) and item.cy < height * 0.28 and not normalize(item.text).startswith("20"),
            lambda item: (item.cy, item.cx),
        )
    if code:
        labels["invoice_code"] = code

    number = first_by(
        items,
        lambda item: bool(re.search(r"(?:NO|No|no)\s*\d{8,20}", item.text)),
        lambda item: (item.cy, -item.cx),
    )
    if not number:
        number = nearby_value(items, ("发票号码", "号码", "NO", "No"), re.compile(r"^\d{8,20}$"))
    if not number:
        number = first_by(
            items,
            lambda item: bool(re.fullmatch(r"\d{8}", normalize(item.text))) and item.cy < height * 0.30 and item is not code,
            lambda item: (item.cy, -item.cx),
        )
    if number:
        labels["invoice_no"] = number

    date = first_by(
        items,
        lambda item: bool(re.search(r"20\d{2}[年\-/\.]?\d{1,2}[月\-/\.]?\d{1,2}", item.text)),
        lambda item: (item.cy, item.cx),
    )
    if date:
        labels["invoice_date"] = date

    checksum = nearby_value(items, ("校验码",), re.compile(r"\d{12,30}"), y_window=120)
    if checksum:
        labels["checksum"] = checksum

    buyer_tax = nearby_value(items, ("购买方", "纳税人识别号", "税号", "统一社会信用代码"), TAX_TOKEN_RE, y_window=110)
    seller_tax = None
    seller_markers = [item for item in items if any(word in item.text for word in ("销售方", "销售信息"))]
    if seller_markers:
        marker_y = min(item.cy for item in seller_markers)
        seller_tax_candidates = [item for item in items if item.cy >= marker_y - 40 and has_tax_id(item.text)]
        if seller_tax_candidates:
            seller_tax = sorted(seller_tax_candidates, key=lambda item: (item.cy, item.cx))[-1]

    tax_ids = [item for item in items if has_tax_id(item.text)]
    if tax_ids:
        tax_ids = sorted(tax_ids, key=lambda item: item.cy)
        upper_tax_ids = [item for item in tax_ids if item.cy < height * 0.55]
        lower_tax_ids = [item for item in tax_ids if item.cy >= height * 0.55]
        if buyer_tax and buyer_tax.cy >= height * 0.55:
            buyer_tax = None
        if buyer_tax:
            labels["buyer_tax_id"] = buyer_tax
        elif upper_tax_ids:
            labels["buyer_tax_id"] = upper_tax_ids[0]
        if seller_tax:
            labels["seller_tax_id"] = seller_tax
        elif lower_tax_ids:
            labels["seller_tax_id"] = lower_tax_ids[-1]
        elif len(tax_ids) > 1:
            labels["seller_tax_id"] = tax_ids[-1]

    companies = [
        item
        for item in items
        if COMPANY_RE.search(item.text)
        and not any(word in item.text for word in ("印制", "发票专用章", "监制", "税总函"))
    ]
    if companies:
        upper = [item for item in companies if height * 0.18 < item.cy < height * 0.55]
        lower = [item for item in companies if item.cy >= height * 0.50]
        if upper:
            labels["buyer_name"] = sorted(upper, key=lambda item: item.cy)[0]
        if seller_markers:
            marker_y = min(item.cy for item in seller_markers)
            seller_companies = [item for item in companies if item.cy >= marker_y - 40]
            if seller_companies:
                labels["seller_name"] = sorted(seller_companies, key=lambda item: (item.cy, item.cx))[0]
        if lower and "seller_name" not in labels:
            labels["seller_name"] = sorted(lower, key=lambda item: item.cy)[-1]
        elif len(companies) > 1:
            labels["seller_name"] = sorted(companies, key=lambda item: item.cy)[-1]

    labels.update({key: value for key, value in choose_money(items, height).items() if key not in labels})

    drawer = nearby_value(items, ("开票人",), re.compile(r"^[\u4e00-\u9fa5]{2,4}$"), y_window=80)
    if drawer:
        labels["drawer"] = drawer

    return labels


def yolo_line(class_name: str, box: Tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_ID[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def copy_and_label(image_path: Path, split: str, labels: Dict[str, OcrItem], output: Path) -> Dict[str, object]:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    image_out = output / "images" / split / image_path.name
    label_out = output / "labels" / split / f"{image_path.stem}.txt"
    preview_out = output / "previews" / f"{image_path.stem}.jpg"
    preview_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)
    lines = [yolo_line(name, item.box, width, height) for name, item in labels.items() if name in CLASS_ID]
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    preview = image.copy()
    for name, item in labels.items():
        x1, y1, x2, y2 = item.box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (61, 118, 157), 2)
        cv2.putText(preview, name, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (61, 118, 157), 2)
    cv2.imwrite(str(preview_out), preview)
    return {
        "image": str(image_path),
        "split": split,
        "label_count": len(lines),
        "missing_classes": [name for name in CLASS_NAMES if name not in labels],
        "preview": str(preview_out),
        "labels": {name: item.text for name, item in labels.items()},
    }


def ensure_dirs(output: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output / rel).mkdir(parents=True, exist_ok=True)


def has_existing_label(output: Path, image_path: Path) -> bool:
    return any((output / "labels" / split / f"{image_path.stem}.txt").exists() for split in ("train", "val"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "test_samples" / "github_public_20"))
    parser.add_argument("--output", default=str(ROOT / "datasets" / "invoice_yolo"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_dirs(output)

    files = image_files(source)
    if args.offset:
        files = files[args.offset :]
    if args.skip_existing:
        files = [path for path in files if not has_existing_label(output, path)]
    files = files[: args.limit]
    report = []
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        try:
            items = ocr_items(image)
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name}: OCR failed: {exc}")
            report.append({"image": str(path), "error": str(exc), "label_count": 0})
            continue
        labels = auto_labels(items, width, height)
        split = "val" if index >= int(len(files) * (1 - args.val_ratio)) else "train"
        item_report = copy_and_label(path, split, labels, output)
        missing = ",".join(item_report["missing_classes"])
        print(f"{path.name}: {item_report['label_count']} labels -> {split}; missing: {missing or 'none'}")
        report.append(item_report)

    report_path = output / "auto_label_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
