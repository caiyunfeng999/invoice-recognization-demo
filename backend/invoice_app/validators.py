"""Structured field validation module.

The parser extracts candidate values.  This module checks whether those values
look reasonable, for example whether tax IDs have a valid length and whether
amount + tax is consistent with the total.  Validation results are sent to the
frontend as field status labels.
"""

import re
from typing import Dict


REQUIRED_FIELDS = (
    "发票号码",
    "发票类型",
    "开票日期",
    "购买方名称",
    "销售方名称",
    "销售方税号",
    "价税合计",
)


def money_value(value: str):
    """Safely parse a money string into a rounded float value."""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def check_tax_id(value: str) -> str:
    """Check basic format of a Chinese taxpayer identification number."""
    if not value:
        return "缺失"
    if 15 <= len(value) <= 20 and any(char.isdigit() for char in value) and any(char.isalpha() for char in value):
        return "正常"
    return "需确认：税号格式异常"


def validate_fields(fields: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Return per-field validation status for frontend display."""
    checks: Dict[str, Dict[str, str]] = {}
    for field, value in fields.items():
        if not value:
            level = "missing" if field in REQUIRED_FIELDS else "optional"
            message = "缺失" if field in REQUIRED_FIELDS else "可为空"
        else:
            level = "ok"
            message = "正常"

        checks[field] = {"level": level, "message": message}

    for field in ("购买方税号", "销售方税号"):
        if field in checks:
            if field == "购买方税号" and not fields.get(field, "") and "普通发票" in fields.get("发票类型", ""):
                checks[field] = {"level": "optional", "message": "票面未提供"}
                continue
            message = check_tax_id(fields.get(field, ""))
            checks[field] = {
                "level": "ok" if message == "正常" else "warning",
                "message": message,
            }

    invoice_number = fields.get("发票号码", "")
    if invoice_number:
        valid_invoice_number = bool(re.fullmatch(r"\d{8,20}", invoice_number))
        checks["发票号码"] = {
            "level": "ok" if valid_invoice_number else "warning",
            "message": "正常" if valid_invoice_number else "需确认：发票号码格式异常",
        }

    invoice_date = fields.get("开票日期", "")
    if invoice_date:
        valid_date = bool(re.fullmatch(r"20\d{2}年\d{1,2}月\d{1,2}日", invoice_date))
        checks["开票日期"] = {
            "level": "ok" if valid_date else "warning",
            "message": "正常" if valid_date else "需确认：日期格式异常",
        }

    total = money_value(fields.get("价税合计", ""))
    amount = money_value(fields.get("金额", ""))
    tax = money_value(fields.get("税额", ""))
    if total is not None and amount is not None and tax is not None:
        if abs(amount + tax - total) <= 0.02:
            checks["价税合计"] = {"level": "ok", "message": "金额关系正常"}
            checks["金额"] = {"level": "ok", "message": "金额关系正常"}
            checks["税额"] = {"level": "ok", "message": "金额关系正常"}
        else:
            message = "需确认：金额 + 税额 不等于价税合计"
            checks["价税合计"] = {"level": "warning", "message": message}
            checks["金额"] = {"level": "warning", "message": message}
            checks["税额"] = {"level": "warning", "message": message}

    return checks
