"""Coordinate-aware invoice field refinement.

The regex parser works on plain text.  This module uses OCR line boxes to add
layout context: nearby labels, upper/lower invoice regions and left/right
relationships.  It is intentionally conservative and only overrides fields when
the geometric evidence is stronger than the plain-text result.
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .parser import clean_value, likely_tax_number, looks_like_company_name, looks_like_person_name


@dataclass(frozen=True)
class OCRLine:
    """One OCR text line with its page-level box."""

    text: str
    score: float
    box: Tuple[float, float, float, float]

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass(frozen=True)
class MoneyCandidate:
    """A money value recognized by OCR, with its source line position."""

    value: float
    text: str
    line: OCRLine


SELLER_NAME_NOISE = (
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
    "印刷有限公司",
    "安全印刷",
    "税总函",
    "国家税务总局",
    "监制",
)


def compact_text(value: str) -> str:
    """Remove whitespace and normalize common OCR punctuation."""
    return re.sub(r"\s+", "", value or "").replace("￥", "¥").replace("：", ":")


def line_to_dict(line: OCRLine) -> Dict[str, object]:
    """Serialize one OCR line for API debugging."""
    return {"text": line.text, "score": line.score, "box": [round(item, 2) for item in line.box]}


def line_sort_key(line: OCRLine) -> Tuple[float, float]:
    """Sort OCR lines by visual reading order."""
    return line.y1, line.x1


def sorted_lines(lines: Iterable[OCRLine]) -> List[OCRLine]:
    """Return non-empty OCR lines sorted by y then x."""
    return sorted([line for line in lines if clean_value(line.text)], key=line_sort_key)


def lines_text(lines: Iterable[OCRLine]) -> str:
    """Join OCR lines in visual order."""
    return "\n".join(line.text for line in sorted_lines(lines))


def find_first_line(lines: List[OCRLine], keywords: Tuple[str, ...]) -> Optional[OCRLine]:
    """Find the first visual line containing any keyword."""
    for line in lines:
        compact = compact_text(line.text)
        if any(keyword in compact for keyword in keywords):
            return line
    return None


def in_y_range(line: OCRLine, start: float, end: float) -> bool:
    """Return whether a line center is inside a vertical range."""
    return start <= line.cy <= end


def region_lines(lines: List[OCRLine], start: float, end: float) -> List[OCRLine]:
    """Select visual lines in a vertical region."""
    return [line for line in lines if in_y_range(line, start, end)]


def page_size(image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Return page height and width from an image shape tuple."""
    height, width = image_shape[:2]
    return float(height), float(width)


def buyer_region(lines: List[OCRLine], image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Estimate buyer block vertical range."""
    height, _ = page_size(image_shape)
    buyer = find_first_line(lines, ("购买方", "买方"))
    stop = find_first_line(lines, ("销售方", "销售信息", "货物或应税劳务", "项目名称", "价税合计", "合计"))
    start = buyer.y1 if buyer else height * 0.18
    end = stop.y1 if stop and stop.y1 > start else height * 0.50
    return max(0, start - height * 0.02), min(height, end)


def seller_region(lines: List[OCRLine], image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Estimate seller block vertical range."""
    height, _ = page_size(image_shape)
    total = find_first_line(lines, ("价税合计", "小写"))
    seller = find_first_line(lines, ("销售方", "销售信息"))
    start_candidates = [height * 0.58]
    if total:
        start_candidates.append(total.y1)
    if seller:
        start_candidates.append(seller.y1)
    return min(start_candidates), height


def extract_tax_id(text: str) -> str:
    """Extract a likely tax ID from a short OCR line."""
    for item in re.findall(r"(?<![0-9A-Z])[0-9A-Z]{12,20}(?![0-9A-Z])", compact_text(text), flags=re.IGNORECASE):
        if likely_tax_number(item):
            return item
    return ""


def choose_tax_id(lines: List[OCRLine], start: float, end: float, exclude: str = "") -> str:
    """Choose a tax ID in one bounded invoice region."""
    scoped = region_lines(lines, start, end)
    candidates: List[Tuple[int, float, str]] = []
    for index, line in enumerate(scoped):
        text = compact_text(line.text)
        window = "".join(compact_text(item.text) for item in scoped[index : index + 4])
        value = extract_tax_id(window)
        if not value or value == exclude:
            continue
        score = line.score
        if any(keyword in text for keyword in ("纳税人识别号", "税号", "统一社会信用代码")):
            score += 40
        if any(noise in text for noise in ("开户行", "账号", "规格型号")):
            score -= 60
        candidates.append((score, -line.cy, value))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]


def company_from_region(lines: List[OCRLine], start: float, end: float, *, reject_noise: bool = False, exclude: str = "") -> str:
    """Choose a likely company name from a bounded region."""
    candidates: List[Tuple[float, str]] = []
    for line in region_lines(lines, start, end):
        text = clean_value(line.text)
        if any(noise in text for noise in SELLER_NAME_NOISE):
            continue
        if exclude and exclude in text:
            continue
        if not looks_like_company_name(text):
            continue
        score = line.score
        compact = compact_text(text)
        if "名称" in compact or "称" in compact:
            score += 20
        if "有限公司" in compact or "公司" in compact:
            score += 15
        if reject_noise and any(noise in compact for noise in ("银行", "支行", "分行", "开户行", "账号")):
            score -= 60
        candidates.append((score, clean_value(re.sub(r"^(名称|名\s*称|称)[:：]?", "", text))))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def money_values_from_text(text: str) -> List[float]:
    """Extract realistic decimal money values from one OCR line."""
    values = []
    pattern = r"(?<![\d.])[¥￥]?\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{2}))(?!\d)"
    for raw in re.findall(pattern, text or ""):
        value = raw.replace(",", "")
        try:
            numeric = round(float(value), 2)
        except ValueError:
            continue
        if 0 < numeric < 1000000000:
            values.append(numeric)
    return values


def money_candidates(lines: List[OCRLine], image_shape: Tuple[int, int]) -> List[MoneyCandidate]:
    """Collect money candidates with coordinates."""
    height, _ = page_size(image_shape)
    candidates = []
    for line in lines:
        if line.cy < height * 0.35:
            continue
        if any(noise in compact_text(line.text) for noise in ("发票代码", "发票号码", "纳税人识别号", "账号")):
            continue
        for value in money_values_from_text(line.text):
            candidates.append(MoneyCandidate(value=value, text=f"{value:.2f}", line=line))
    return candidates


def choose_amount_tax_total(lines: List[OCRLine], image_shape: Tuple[int, int]) -> Tuple[str, str, str]:
    """Infer amount, tax and total from money positions and arithmetic."""
    height, width = page_size(image_shape)
    candidates = money_candidates(lines, image_shape)
    if not candidates:
        return "", "", ""

    best: Tuple[float, Optional[MoneyCandidate], Optional[MoneyCandidate], Optional[MoneyCandidate]] = (-10**9, None, None, None)
    for total in candidates:
        if total.line.cy < height * 0.55:
            continue
        for amount in candidates:
            if amount is total or amount.line.cy >= total.line.cy:
                continue
            for tax in candidates:
                if tax is total or tax is amount or tax.line.cy >= total.line.cy:
                    continue
                if abs(amount.value + tax.value - total.value) > 0.03:
                    continue
                if amount.value < tax.value:
                    continue
                same_row = abs(amount.line.cy - tax.line.cy) <= max(amount.line.height, tax.line.height) * 2.2
                amount_left = amount.line.cx <= tax.line.cx or abs(amount.line.cx - tax.line.cx) < width * 0.05
                score = 100.0
                score += 40 if same_row else -40
                score += 35 if amount_left else -35
                score += 25 if "价税合计" in compact_text(total.line.text) or "小写" in compact_text(total.line.text) else 0
                score += min(30, (total.line.cy - amount.line.cy) / max(height, 1) * 120)
                score += min(total.line.score + amount.line.score + tax.line.score, 300) * 0.05
                if score > best[0]:
                    best = (score, amount, tax, total)

    _, amount, tax, total = best
    if amount and tax and total:
        return f"{amount.value:.2f}", f"{tax.value:.2f}", f"{total.value:.2f}"

    total_candidates = sorted(
        candidates,
        key=lambda item: (
            "价税合计" in compact_text(item.line.text) or "小写" in compact_text(item.line.text),
            item.line.cy,
            item.value,
        ),
        reverse=True,
    )
    if total_candidates:
        return "", "", f"{total_candidates[0].value:.2f}"
    return "", "", ""


def choose_checksum(lines: List[OCRLine], image_shape: Tuple[int, int]) -> str:
    """Choose checksum near the checksum label, not near tax ID labels."""
    height, _ = page_size(image_shape)
    candidates: List[Tuple[float, str]] = []
    for index, line in enumerate(lines):
        compact = compact_text(line.text)
        if "校验码" not in compact:
            continue
        window = "".join(compact_text(item.text) for item in lines[index : index + 4])
        for raw in re.findall(r"\d[\d\s]{10,35}\d", window):
            digits = re.sub(r"\D", "", raw)
            if 12 <= len(digits) <= 30:
                score = line.score + (1 - line.cy / max(height, 1)) * 20
                candidates.append((score, digits))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def choose_drawer(lines: List[OCRLine], image_shape: Tuple[int, int]) -> str:
    """Choose drawer from bottom area or near the drawer label."""
    height, _ = page_size(image_shape)
    candidates: List[Tuple[float, str]] = []
    for index, line in enumerate(lines):
        compact = compact_text(line.text)
        window = compact + "".join(compact_text(item.text) for item in lines[index + 1 : index + 3])
        if "开票人" not in compact and line.cy < height * 0.78:
            continue
        match = re.search(r"开票人[::]?([\u4e00-\u9fa5]{2,4})", compact)
        if match and looks_like_person_name(match.group(1)):
            candidates.append((line.score + 80, match.group(1)))
            continue
        if "开票人" in compact:
            for offset, nearby in enumerate(lines[index + 1 : index + 5], start=1):
                nearby_compact = compact_text(nearby.text)
                for name in re.findall(r"[\u4e00-\u9fa5]{2,4}", nearby_compact):
                    if looks_like_person_name(name):
                        candidates.append((nearby.score + 70 - offset * 5, name))
        if line.cy > height * 0.82:
            for name in re.findall(r"(?<![\u4e00-\u9fa5])[\u4e00-\u9fa5]{2,4}(?![\u4e00-\u9fa5])", compact):
                if looks_like_person_name(name):
                    candidates.append((line.score, name))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def refine_fields_with_layout(fields: Dict[str, str], ocr_lines: List[OCRLine], image_shape: Tuple[int, int]) -> Dict[str, str]:
    """Refine parsed invoice fields using OCR line coordinates."""
    lines = sorted_lines(ocr_lines)
    if not lines:
        return fields

    refined = dict(fields)
    buyer_start, buyer_end = buyer_region(lines, image_shape)
    seller_start, seller_end = seller_region(lines, image_shape)

    buyer_tax = choose_tax_id(lines, buyer_start, buyer_end)
    seller_tax = choose_tax_id(lines, seller_start, seller_end, exclude=buyer_tax or refined.get("购买方税号", ""))
    if buyer_tax:
        refined["购买方税号"] = buyer_tax
    if seller_tax:
        refined["销售方税号"] = seller_tax
    if refined.get("购买方税号") and refined.get("购买方税号") == refined.get("销售方税号") and "普通发票" in refined.get("发票类型", ""):
        refined["购买方税号"] = ""

    buyer_name = company_from_region(lines, buyer_start, buyer_end)
    seller_name = company_from_region(lines, seller_start, seller_end, reject_noise=True, exclude=buyer_name)
    if buyer_name:
        refined["购买方名称"] = buyer_name
    if seller_name:
        refined["销售方名称"] = seller_name
    if (
        refined.get("购买方名称")
        and refined.get("销售方名称")
        and refined["购买方名称"] == refined["销售方名称"]
    ):
        refined["购买方名称"] = ""

    checksum = choose_checksum(lines, image_shape)
    if checksum:
        refined["校验码"] = checksum
        if refined.get("购买方税号") == checksum:
            refined["购买方税号"] = ""
        if refined.get("销售方税号") == checksum:
            refined["销售方税号"] = ""

    amount, tax, total = choose_amount_tax_total(lines, image_shape)
    if total:
        refined["价税合计"] = total
    if amount and tax:
        refined["金额"] = amount
        refined["税额"] = tax

    drawer = choose_drawer(lines, image_shape)
    if drawer:
        refined["开票人"] = drawer

    return refined
