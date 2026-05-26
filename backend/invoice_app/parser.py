import re
from typing import Dict, Tuple


FIELD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "发票类型": (r"(电子发票|增值税电子普通发票|增值税专用发票|普通发票)",),
    "发票代码": (r"发票代码[:：]?\s*([0-9]{8,20})",),
    "发票号码": (r"发票号码[:：]?\s*([0-9]{6,20})", r"号码[:：]?\s*([0-9]{6,20})"),
    "开票日期": (
        r"开票日期[:：]?\s*([0-9]{4}[年\-/\.][0-9]{1,2}[月\-/\.][0-9]{1,2}日?)",
        r"日期[:：]?\s*([0-9]{4}[年\-/\.][0-9]{1,2}[月\-/\.][0-9]{1,2}日?)",
    ),
    "购买方名称": (r"购买方名称[:：]?\s*([^\n]+)", r"购买方[:：]?\s*([^\n]+)"),
    "购买方税号": (
        r"购买方.*?(?:纳税人识别号|税号)[:：]?\s*([A-Z0-9]{12,20})",
        r"纳税人识别号[:：]?\s*([A-Z0-9]{12,20})",
    ),
    "销售方名称": (r"销售方名称[:：]?\s*([^\n]+)", r"销售方[:：]?\s*([^\n]+)"),
    "销售方税号": (
        r"销售方.*?(?:纳税人识别号|税号)[:：]?\s*([A-Z0-9]{12,20})",
    ),
    "价税合计": (
        r"价税合计.*?[¥￥]?\s*([0-9]+(?:\.[0-9]{1,2})?)",
        r"小写\)?[:：]?\s*[¥￥]?\s*([0-9]+(?:\.[0-9]{1,2})?)",
    ),
    "金额": (r"金额[:：]?\s*[¥￥]?\s*([0-9]+(?:\.[0-9]{1,2})?)",),
    "税额": (r"税额[:：]?\s*[¥￥]?\s*([0-9]+(?:\.[0-9]{1,2})?)",),
    "校验码": (r"校验码[:：]?\s*([0-9 ]{8,})",),
    "开票人": (r"开票人[:：]?\s*([^\s\n]+)",),
}


def normalize_text(text: str) -> str:
    normalized = text.replace(" ", "")
    normalized = normalized.replace("￥", "¥")
    normalized = normalized.replace("：", ":")
    return normalized


def parse_invoice_text(text: str) -> Dict[str, str]:
    compact_text = normalize_text(text)
    result = {}
    for field, patterns in FIELD_PATTERNS.items():
        value = ""
        for pattern in patterns:
            match = re.search(pattern, compact_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                value = value.split("\n")[0].strip()
                break
        result[field] = value
    return result


def completion_score(fields: Dict[str, str]) -> float:
    if not fields:
        return 0.0
    filled = sum(1 for value in fields.values() if value)
    return round(filled / len(fields), 3)
