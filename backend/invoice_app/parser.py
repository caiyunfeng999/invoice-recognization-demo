"""Invoice text parsing module.

OCR/PDF extraction returns unstructured text.  This module converts the text into
fixed invoice fields such as invoice number, buyer/seller tax IDs, amount, tax
and drawer.  It combines direct regular-expression extraction with fallback
rules for noisy image OCR and PDF text-order problems.
"""

import re
from typing import Dict, List, Tuple


FIELD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    # Primary patterns: use explicit labels when OCR/PDF text keeps labels and
    # values close together.  Fallback functions below handle noisier cases.
    "发票类型": (r"(增值税电子普通发票|增值税普通发票|增值税专用发票|电子发票|普通发票)",),
    "发票代码": (r"发票代码[:：]?\s*([0-9]{8,12})(?!\d)",),
    "发票号码": (r"发票号码[:：]?\s*([0-9]{8,20})(?!\d)", r"号码[:：]?\s*([0-9]{8,20})(?!\d)"),
    "开票日期": (
        r"开票日期[:：]?\s*([0-9]{4}[年\-/\.][0-9]{1,2}[月\-/\.][0-9]{1,2}日?)",
        r"开票日期[:：]?\s*([0-9]{4}[01][0-9][0-3][0-9])",
    ),
    "购买方名称": (r"购买方名称[:：]?\s*([^\n]+)", r"购买方[:：]?\s*([^\n]+)"),
    # Buyer tax ID is intentionally not extracted from compact full text here:
    # when the buyer field is empty, a broad regex can cross into the seller
    # block and incorrectly copy the seller tax ID.  Region-aware extraction
    # below handles this field.
    "购买方税号": (),
    "销售方名称": (r"销售方名称[:：]?\s*([^\n]+)", r"销售方[:：]?\s*([^\n]+)"),
    "销售方税号": (
        r"销售方.{0,80}?(?:纳税人识别号|税号)[:：]?\s*([A-Z0-9]{12,20})",
        r"代开企业税号[:：]?\s*([A-Z0-9]{12,20})",
    ),
    "价税合计": (
        r"价税合计[^\n]{0,80}?[¥￥]\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"小写\)?[:：]?\s*[¥￥]\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
    ),
    "金额": (r"金额[^\n]{0,20}?[¥￥]\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",),
    "税额": (r"税额[^\n]{0,20}?[¥￥]\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",),
    "校验码": (r"校验码[:：]?\s*([0-9 ]{8,})",),
    "开票人": (r"开票人[:：]?\s*([\u4e00-\u9fa5]{2,4})",),
}

ORG_SUFFIX_PATTERN = (
    r"公司|税务局|合作社|商店|中心|厂|部|研究所|研究院|科学院|学院|大学|学校|医院|"
    r"委员会|管理局|银行|酒店|宾馆|饭店|集团|事务所|经营部|商行"
)

DRAWER_NOISE_WORDS = (
    "销售方",
    "购买方",
    "收款人",
    "复核",
    "开票人",
    "备注",
    "名称",
    "纳税",
    "识别",
    "开户",
    "账号",
    "地址",
    "电话",
    "专用章",
    "专用",
    "爱票",
    "票专",
    "发票",
    "凭证",
    "合计",
    "税额",
    "金额",
    "单位",
    "数量",
    "单价",
)

COMMON_DRAWER_ACCOUNT_NAMES = {
    "管理员",
    "开票员",
    "财务",
    "会计",
    "系统",
}

COMPOUND_SURNAMES = {
    "欧阳",
    "太史",
    "端木",
    "上官",
    "司马",
    "东方",
    "独孤",
    "南宫",
    "万俟",
    "闻人",
    "夏侯",
    "诸葛",
    "尉迟",
    "公羊",
    "赫连",
    "澹台",
    "皇甫",
    "宗政",
    "濮阳",
    "公冶",
    "太叔",
    "申屠",
    "公孙",
    "慕容",
    "仲孙",
    "钟离",
    "长孙",
    "宇文",
    "司徒",
    "鲜于",
    "司空",
    "闾丘",
    "子车",
    "亓官",
    "司寇",
    "巫马",
    "公西",
    "颛孙",
    "壤驷",
    "公良",
    "漆雕",
    "乐正",
    "宰父",
    "谷梁",
    "拓跋",
    "夹谷",
    "轩辕",
    "令狐",
    "段干",
    "百里",
    "呼延",
    "东郭",
    "南门",
    "羊舌",
    "微生",
}

COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费"
    "廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和"
    "穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮"
    "蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支"
    "柯昝管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣"
    "翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓"
    "蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟"
    "薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄"
    "姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀浦尚农温庄晏柴瞿阎充慕连"
    "茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东殴殳"
    "沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关"
    "蒯相查后荆红游竺权逯盖益桓公"
)


def normalize_text(text: str) -> str:
    """Normalize common punctuation and spacing differences before matching."""
    normalized = text.replace(" ", "")
    normalized = normalized.replace("￥", "¥")
    normalized = normalized.replace("：", ":")
    return normalized


def clean_value(value: str) -> str:
    """Clean delimiters, labels and thousands separators from extracted values."""
    value = value.strip()
    value = re.split(r"[|#<>]{2,}", value)[0]
    value = re.sub(r"^(名称|名\s*称)[:：]?", "", value)
    value = value.replace(",", "")
    return value.strip(" :：,，;；|[]()（）")


def looks_like_person_name(value: str) -> bool:
    """Return whether text is a likely drawer name or valid drawer account."""
    compact = re.sub(r"[^\u4e00-\u9fa5]", "", clean_value(value))
    if any(word in compact for word in DRAWER_NOISE_WORDS):
        return False
    if re.search(ORG_SUFFIX_PATTERN, compact):
        return False
    if compact in COMMON_DRAWER_ACCOUNT_NAMES:
        return True
    if not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", compact):
        return False
    if compact[:2] in COMPOUND_SURNAMES:
        return len(compact) >= 3
    if compact[0] in COMMON_SURNAMES:
        return True
    return False


def drawer_from_lines_near_label(raw_text: str) -> str:
    """Extract drawer from lines immediately after the bottom '开票人' label."""
    lines = nonempty_lines(raw_text)
    label_indexes = [index for index, line in enumerate(lines) if "开票人" in line]
    if not label_indexes:
        return ""
    start = label_indexes[-1]
    window = lines[start : start + 5]

    first_line_match = re.search(r"开票人[:：]?\s*([\u4e00-\u9fa5]{2,4})", normalize_text(window[0]))
    if first_line_match and looks_like_person_name(first_line_match.group(1)):
        return first_line_match.group(1)

    for line in window[1:]:
        compact = re.sub(r"[^\u4e00-\u9fa5]", "", line)
        if len(compact) > 4:
            if looks_like_company_name(compact) or is_stamp_or_noise(compact):
                continue
            continue
        if looks_like_person_name(compact):
            return compact
        for name in re.findall(r"[\u4e00-\u9fa5]{2,4}", compact):
            if looks_like_person_name(name):
                return name
    return ""


def is_stamp_or_noise(value: str) -> bool:
    """Filter seal/watermark text that should not be treated as company names."""
    noise_words = (
        "发票专用章",
        "专用章",
        "爱票",
        "监制章",
        "收款人",
        "复核",
        "开票人",
        "发票联",
        "税总函",
        "印制有限公司",
        "印刷有限公司",
        "安全印刷",
        "国家税务总局",
        "监制",
    )
    return any(word in value for word in noise_words)


def looks_like_company_name(value: str) -> bool:
    """Check whether a field value looks like one organization name, not a block."""
    if not value or is_stamp_or_noise(value):
        return False
    if value in ("有限公司", "业有限公司", "责任公司", "股份公司"):
        return False
    if len(value) > 45:
        return False
    if any(word in value for word in ("发票代码", "发票号码", "开票日期", "校验码", "购买方", "销售方", "税率")):
        return False
    return bool(re.search(rf"[\u4e00-\u9fa5A-Za-z0-9（）()]{{2,}}(?:{ORG_SUFFIX_PATTERN})", value))


def company_match_from_text(text: str) -> str:
    """Extract one likely organization name from a noisy line or short block."""
    patterns = (
        rf"(?:名称|名\s*称|称)[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]{{2,}}(?:{ORG_SUFFIX_PATTERN}))",
        rf"([\u4e00-\u9fa5A-Za-z0-9（）()]{{2,}}(?:{ORG_SUFFIX_PATTERN}))",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = clean_value(match.group(1))
        if looks_like_company_name(value):
            return value
    return ""


def first_match(text: str, patterns: Tuple[str, ...]) -> str:
    """Return the first captured value from a list of regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_value(match.group(1).split("\n")[0])
    return ""


def number_candidates(text: str) -> List[str]:
    """Find standalone numeric candidates without taking digits from tax IDs."""
    return re.findall(r"(?<![A-Z0-9])\d{6,20}(?![A-Z0-9])", text, flags=re.IGNORECASE)


def tax_number_candidates(text: str) -> List[str]:
    """Find taxpayer ID candidates from compact text."""
    explicit = re.findall(r"纳税人识别号[:：]?\s*([0-9A-Z]{12,20})", text)
    generic = re.findall(r"(?<![0-9A-Z])(?=[0-9A-Z]*[A-Z])(?=[0-9A-Z]*\d)[0-9A-Z]{15,20}(?![0-9A-Z])", text)
    values = []
    for item in explicit + generic:
        if item not in values:
            values.append(item)
    return values


def tax_number_candidates_from_lines(raw_text: str) -> List[str]:
    """Find tax IDs near tax-number labels, then append generic candidates."""
    values = []
    lines = [normalize_text(line) for line in raw_text.splitlines()]
    for index, line in enumerate(lines):
        if not any(keyword in line for keyword in ("纳税人识别号", "税号")):
            continue
        joined = "".join(lines[index : index + 3])
        for item in re.findall(r"[0-9A-Z]{12,20}", joined):
            if item not in values:
                values.append(item)
    for item in tax_number_candidates(normalize_text(raw_text)):
        if item not in values:
            values.append(item)
    return values


def likely_tax_number(value: str) -> bool:
    """Return whether a string looks like a taxpayer ID, not an item code."""
    if not re.fullmatch(r"[0-9A-Z]{12,20}", value or ""):
        return False
    # Most current Chinese tax IDs are 15-20 chars and may be all digits or
    # alphanumeric.  Shorter numeric strings are usually invoice codes/numbers.
    if value.isdigit() and len(value) < 15:
        return False
    if any(noise in value for noise in ("HHE",)):
        return False
    return True


def tax_number_from_window(lines: List[str], start: int, end: int) -> str:
    """Extract a tax ID only from a bounded visual text window."""
    scoped = lines[start:end]
    for index, line in enumerate(scoped):
        if not any(keyword in line for keyword in ("纳税人识别号", "税号", "统一社会信用代码")):
            continue
        joined = "".join(scoped[index : index + 4])
        for item in re.findall(r"[0-9A-Z]{12,20}", joined):
            if likely_tax_number(item):
                return item
    # Some OCR engines drop the label line but keep the tax ID in the same
    # bounded buyer/seller block.  Use this only inside the already-scoped area.
    for line in scoped:
        if any(noise in line for noise in ("规格型号", "开户行", "账号", "HHE")):
            continue
        for item in re.findall(r"(?<![0-9A-Z])[0-9A-Z]{15,20}(?![0-9A-Z])", line):
            if likely_tax_number(item):
                return item
    return ""


def buyer_tax_number_from_lines(raw_text: str) -> str:
    """Extract buyer tax ID from the buyer block only.

    If the buyer block has no tax ID on the invoice, return an empty string
    instead of falling back to the seller tax ID.
    """
    lines = [normalize_text(line) for line in raw_text.splitlines() if clean_value(line)]
    if not lines:
        return ""

    starts = [index for index, line in enumerate(lines) if "购买方" in line or "买方" in line]
    for start in starts:
        end = min(len(lines), start + 16)
        for index in range(start + 1, min(len(lines), start + 30)):
            line = lines[index]
            if any(marker in line for marker in ("销售方", "销售信息", "货物或应税劳务", "项目名称", "价税合计", "合计")):
                end = index
                break
        value = tax_number_from_window(lines, start, end)
        if value:
            return value
    return ""


def checksum_value(compact_text: str) -> str:
    """Extract invoice checksum when the invoice type contains such a field."""
    match = re.search(r"校验码[:：]?\s*([0-9]{12,30})", compact_text)
    return match.group(1) if match else ""


def normalized_invoice_type(current: str, compact_text: str) -> str:
    """Normalize common OCR variants of Chinese VAT invoice titles."""
    if "电子" in compact_text and "普通发票" in compact_text:
        return "增值税电子普通发票" if "增值税" in compact_text else "电子发票"
    if "专用发票" in compact_text:
        return "增值税专用发票"
    if re.search(r"增.{0,2}税普通发票", compact_text):
        return "增值税普通发票"
    return current


def fallback_seller_tax_number(compact_text: str) -> str:
    """Fallback extraction for seller tax number from seller-related context."""
    patterns = (
        r"代开企业税号[:：]?\s*([0-9A-Z]{12,20})",
        r"销售方.{0,100}?(?:纳税人识别号|税号)[:：]?\s*([0-9A-Z]{12,20})",
    )
    for pattern in patterns:
        match = re.search(pattern, compact_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return ""


def agent_seller_tax_number(compact_text: str) -> str:
    """Extract real seller tax ID from tax-authority-issued invoices."""
    match = re.search(r"代开企业税号[:：]?\s*([0-9A-Z]{12,20})", compact_text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def agent_seller_name_from_lines(raw_text: str) -> str:
    """Extract real seller name from '代开企业名称', including OCR line splits."""
    lines = [clean_value(line) for line in raw_text.splitlines() if clean_value(line)]
    stop_words = ("纳税人识别号", "地址", "电话", "开户行", "账号", "开票人", "复核", "收款人", "完税凭证")
    skip_words = ("销售方", "购买方", "备注", "备", "注", "代开机关")
    for index, line in enumerate(lines):
        if "代开企业名称" not in line:
            continue
        value = clean_value(re.sub(r"^.*?代开企业名称[:：]?", "", line))
        pieces = []
        if value and not any(word in value for word in skip_words + stop_words):
            pieces.append(re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9（）()]", "", value))
        for later in lines[index + 1 : index + 8]:
            compact = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9（）()]", "", later)
            if not compact:
                continue
            if any(word in compact for word in stop_words):
                break
            if any(word == compact for word in skip_words):
                continue
            if looks_like_company_name(compact):
                break
            if re.fullmatch(r"[\u4e00-\u9fa5]{1,6}", compact):
                pieces.append(compact)
                if sum(len(piece) for piece in pieces) >= 2:
                    break
        name = "".join(pieces).strip()
        if 2 <= len(name) <= 20:
            return name
    return ""


def company_candidates(raw_text: str) -> List[str]:
    """Collect likely company/organization names line by line."""
    companies = []
    for line in raw_text.splitlines():
        line = clean_value(line)
        if is_stamp_or_noise(line):
            continue
        company = company_match_from_text(line)
        if company and company not in companies:
            companies.append(company)
    return companies


def buyer_name_from_lines(raw_text: str) -> str:
    """Prefer organization names in the visual block following the '购买方' label."""
    lines = [clean_value(line) for line in raw_text.splitlines() if clean_value(line)]
    for index, line in enumerate(lines):
        if "购买方" not in line:
            continue
        window = []
        start = max(0, index - 6)
        for item in lines[start : index + 12]:
            if "销售方" in item or "货物或应税劳务" in item or "价税合计" in item:
                break
            window.append(item)
        joined = "".join(window)
        company = company_match_from_text(joined)
        if company:
            return company
        for item in window:
            company = company_match_from_text(item)
            if company:
                return company
    return ""


def seller_name_from_lines(raw_text: str, buyer_name: str = "") -> str:
    """Prefer company names appearing near the '销售方' block."""
    lines = [clean_value(line) for line in raw_text.splitlines() if clean_value(line)]
    for index, line in enumerate(lines):
        if not ("价税合计" in line or "小写" in line):
            continue
        window = lines[index : index + 35]
        for item in window:
            value = company_match_from_text(item)
            if (
                value
                and value != buyer_name
                and not any(noise in value for noise in ("银行", "支行", "分行", "印制"))
            ):
                return value
    for index, line in enumerate(lines):
        if "销售方" not in line:
            continue
        window = lines[max(0, index - 5) : index + 8]
        joined = "".join(window)
        for pattern in (
            rf"(?:名称|名\s*称|称)[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9（）()]{{2,}}(?:{ORG_SUFFIX_PATTERN}))",
        ):
            match = re.search(pattern, joined)
            if match:
                value = clean_value(match.group(1))
                if value != buyer_name and looks_like_company_name(value):
                    return value
        for item in window:
            value = company_match_from_text(item)
            if value and value != buyer_name:
                return value
        for item in window:
            value = company_match_from_text(item)
            if value and value != buyer_name:
                return value
    return ""


def seller_tax_number_from_lines(raw_text: str, buyer_tax_id: str = "") -> str:
    """Extract seller tax ID from the lower/seller part of the invoice."""
    lines = [normalize_text(line) for line in raw_text.splitlines() if clean_value(line)]
    start = 0
    for index, line in enumerate(lines):
        if "价税合计" in line or "小写" in line:
            start = index
            break
    candidates = []
    for line in lines[start:]:
        if any(noise in line for noise in ("HHE", "规格型号", "开户行", "账号")):
            continue
        for item in re.findall(r"(?<![0-9A-Z])(?=[0-9A-Z]*[A-Z])(?=[0-9A-Z]*\d)[0-9A-Z]{15,20}(?![0-9A-Z])", line):
            if item != buyer_tax_id and item not in candidates:
                candidates.append(item)
    return candidates[0] if candidates else ""


def fallback_invoice_code(compact_text: str) -> str:
    """Infer invoice code from numeric candidates when label matching fails."""
    candidates = [
        number
        for number in number_candidates(compact_text)
        if 10 <= len(number) <= 12 and not number.startswith("20") and len(number) != 8
    ]
    for index, current in enumerate(candidates):
        for later in candidates[index + 1 :]:
            if len(current) == len(later) and current[:-1] == later[:-1]:
                return later
    if candidates:
        return candidates[0]
    return ""


def fallback_invoice_number(compact_text: str) -> str:
    """Infer invoice number without accidentally slicing taxpayer IDs."""
    candidates = number_candidates(compact_text)
    labeled = re.search(r"(?:发票号码|号码|No)[:：]?\s*([0-9]{8,20})(?!\d)", compact_text, flags=re.IGNORECASE)
    if labeled:
        return labeled.group(1)
    for number in candidates:
        if len(number) == 8 and not number.startswith("20"):
            return number
    for number in candidates:
        if 9 <= len(number) <= 20 and not number.startswith("20"):
            return number
    return ""


def fallback_date(compact_text: str) -> str:
    """Infer billing date only when it appears near the '开票日期' label."""
    match = re.search(r"开票日期[:：]?\s*(20\d{2})\D{0,8}([01]?\d)\D{0,8}([0-3]?\d)", compact_text)
    if not match:
        match = re.search(r"(20\d{2})[年\-/\.]([01]?\d)[月\-/\.]([0-3]?\d)日?", compact_text)
    if not match:
        match = re.search(r"(?<!\d)(20\d{6})(?!\d)", compact_text)
        if match:
            compact = match.group(1)
            return f"{compact[:4]}年{compact[4:6]}月{compact[6:8]}日"
    if not match:
        return ""
    year, month, day = match.groups()
    month_int = int(month)
    day_int = int(day)
    if 1 <= month_int <= 12 and 1 <= day_int <= 31:
        return f"{year}年{month_int:02d}月{day_int:02d}日"
    return ""


def fallback_amount(compact_text: str) -> str:
    """Use the largest realistic money value as a last-resort total amount."""
    money = money_candidates(compact_text)
    if not money:
        return ""
    return max(money, key=lambda item: float(item))


def money_candidates(compact_text: str) -> List[str]:
    """Extract realistic money values while ignoring long serial numbers."""
    values = []
    pattern = r"(?<![\d.])[¥￥]?\s*([0-9]{1,12}(?:,[0-9]{3})*(?:\.[0-9]{2}))(?!\d)"
    for raw in re.findall(pattern, compact_text):
        value = raw.replace(",", "")
        try:
            numeric = float(value)
        except ValueError:
            continue
        if numeric <= 0 or numeric > 1000000000:
            continue
        if value not in values:
            values.append(value)
    return values


def infer_amount_and_tax(compact_text: str, total: str) -> Tuple[str, str]:
    """Infer amount and tax from total using money candidates or tax rate."""
    if not total:
        return "", ""
    try:
        total_value = round(float(total), 2)
    except ValueError:
        return "", ""

    values = [
        round(float(item), 2)
        for item in money_candidates(compact_text)
        if round(float(item), 2) != total_value
    ]
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            if abs(first + second - total_value) < 0.02:
                amount, tax = (first, second) if first >= second else (second, first)
                return f"{amount:.2f}", f"{tax:.2f}"

    rate_matches = re.findall(r"(\d{1,2})%", compact_text)
    rates = [int(item) for item in rate_matches if 0 < int(item) <= 17]
    if rates:
        rate = rates[0] / 100
        amount = round(total_value / (1 + rate), 2)
        tax = round(total_value - amount, 2)
        return f"{amount:.2f}", f"{tax:.2f}"

    return "", ""


def nonempty_lines(raw_text: str) -> List[str]:
    """Return cleaned non-empty text lines for PDF-tail parsing."""
    return [clean_value(line) for line in raw_text.splitlines() if clean_value(line)]


def drawer_from_pdf_tail(tail: List[str]) -> str:
    """Choose drawer from electronic invoice value tail.

    In electronic invoice PDFs, the label "开票人" can appear before all values.
    The actual drawer name often appears after the amount-in-words or total
    amount and before trailing dates/company names.
    """
    after_total_area = False
    for item in tail:
        compact = normalize_text(item)
        if re.search(r"[壹贰叁肆伍陆柒捌玖拾佰仟万亿圆元角分整]", compact) or "价税合计" in compact:
            after_total_area = True
            continue
        if not after_total_area and re.fullmatch(r"[¥￥]?\d+(?:\.\d{2})", compact):
            # Money values usually precede the drawer; continue until the first
            # valid short Chinese name after the amount area.
            after_total_area = True
            continue
        if not after_total_area:
            continue
        if re.fullmatch(r"20\d{6}", compact) or re.fullmatch(r"20\d{2}年[01]?\d月[0-3]?\d日", compact):
            break
        if looks_like_company_name(compact):
            break
        if looks_like_person_name(compact):
            return compact
    return ""


def apply_pdf_tail_values(result: Dict[str, str], raw_text: str) -> None:
    """Handle electronic PDFs whose labels and values are extracted separately.

    Some electronic invoices put all labels first and all values later in the
    extracted text.  After the '开票人' label, values often appear in a stable
    order: invoice number, date, tax IDs, company names, amount, tax, total and
    drawer.  This function reassigns those values to the standard fields.
    """
    lines = nonempty_lines(raw_text)
    tail_start = -1
    for index, line in enumerate(lines):
        if "开票人" in line:
            tail_start = index + 1
    if tail_start < 0 or tail_start >= len(lines):
        return

    tail = lines[tail_start:]
    tax_ids = []
    companies = []
    money_values = []
    dates = []
    invoice_numbers = []
    drawer_candidates = []

    for item in tail:
        compact = normalize_text(item)
        if re.fullmatch(r"\d{16,20}", compact):
            invoice_numbers.append(compact)
        if re.fullmatch(r"20\d{2}年[01]?\d月[0-3]?\d日", compact):
            dates.append(compact)
        if re.fullmatch(r"20\d{6}", compact):
            dates.append(f"{compact[:4]}年{compact[4:6]}月{compact[6:8]}日")
        if re.fullmatch(r"[0-9A-Z]{15,20}", compact) and re.search(r"[A-Z]", compact):
            tax_ids.append(compact)
        if re.search(r"(?:公司|税务局|合作社|商店|中心|厂|部)$", compact):
            companies.append(compact)
        for value in money_candidates(compact):
            money_values.append(value)
        if looks_like_person_name(compact) and not any(word in compact for word in ("客车", "不征税")):
            drawer_candidates.append(compact)

    # Electronic PDF text often has a stable value order after "开票人":
    # invoice number, date, buyer tax ID, buyer name, seller tax ID, item rows,
    # money/tax/total, drawer, then sometimes extra dates/company names in notes.
    for index, item in enumerate(tail):
        compact = normalize_text(item)
        if re.fullmatch(r"[0-9A-Z]{15,20}", compact) and re.search(r"[A-Z]", compact):
            next_company = ""
            for later in tail[index + 1 : index + 4]:
                later_compact = normalize_text(later)
                if looks_like_company_name(later_compact):
                    next_company = later_compact
                    break
            if compact in tax_ids:
                tax_index = tax_ids.index(compact)
                if tax_index == 0:
                    result["购买方税号"] = result["购买方税号"] or compact
                    if next_company:
                        result["购买方名称"] = result["购买方名称"] or next_company
                elif tax_index == 1:
                    result["销售方税号"] = result["销售方税号"] or compact
                    if next_company:
                        result["销售方名称"] = result["销售方名称"] or next_company

    if not result["发票号码"] and invoice_numbers:
        result["发票号码"] = invoice_numbers[0]
    if not result["开票日期"] and dates:
        result["开票日期"] = dates[0]
    if not result["购买方税号"] and tax_ids:
        result["购买方税号"] = tax_ids[0]
    if not result["销售方税号"] and len(tax_ids) > 1:
        result["销售方税号"] = tax_ids[1]
    if not result["购买方名称"] and companies:
        result["购买方名称"] = companies[0]
    if not result["销售方名称"] and len(companies) > 1:
        result["销售方名称"] = companies[-1]

    if money_values:
        numeric_values = sorted({round(float(value), 2) for value in money_values})
        if "不征税" in raw_text or re.search(r"\n\*\n", raw_text):
            # Toll-road non-tax invoices may contain amount == total and tax as
            # "*".  In this case the largest money value is the amount/total,
            # while tax should not be inferred from another unrelated value.
            total_value = max(numeric_values)
            result["价税合计"] = result["价税合计"] or f"{total_value:.2f}"
            result["金额"] = result["金额"] or f"{total_value:.2f}"
            result["税额"] = result["税额"] or "0.00"
            numeric_values = []
        for total in reversed(numeric_values):
            smaller = [value for value in numeric_values if value < total]
            for amount in reversed(smaller):
                tax = round(total - amount, 2)
                if tax in smaller and tax <= amount:
                    result["价税合计"] = result["价税合计"] or f"{total:.2f}"
                    result["金额"] = result["金额"] or f"{amount:.2f}"
                    result["税额"] = result["税额"] or f"{tax:.2f}"
                    break
            if result["价税合计"] and result["金额"] and result["税额"]:
                break
        if not result["价税合计"]:
            result["价税合计"] = f"{max(numeric_values):.2f}"

    drawer_from_tail = drawer_from_pdf_tail(tail)
    if drawer_from_tail:
        result["开票人"] = drawer_from_tail
    elif not result["开票人"] and drawer_candidates:
        # The first short Chinese name after the "开票人" label is normally the
        # drawer.  Later short strings may come from notes or extra fields.
        result["开票人"] = drawer_candidates[0]


def parse_invoice_text(text: str) -> Dict[str, str]:
    """Parse OCR/PDF text and return the fixed structured invoice field dict."""
    compact_text = normalize_text(text)
    result = {}
    for field, patterns in FIELD_PATTERNS.items():
        result[field] = first_match(compact_text, patterns)
    result["发票类型"] = normalized_invoice_type(result["发票类型"], compact_text)

    apply_pdf_tail_values(result, text)

    if not result["发票代码"]:
        result["发票代码"] = fallback_invoice_code(compact_text)
    if not result["发票号码"]:
        result["发票号码"] = fallback_invoice_number(compact_text)
    if not result["开票日期"]:
        result["开票日期"] = fallback_date(compact_text)
    if not result["价税合计"]:
        result["价税合计"] = fallback_amount(compact_text)

    buyer_tax_from_region = buyer_tax_number_from_lines(text)
    if buyer_tax_from_region:
        result["购买方税号"] = buyer_tax_from_region

    tax_numbers = tax_number_candidates_from_lines(text)
    checksum = checksum_value(compact_text)
    tax_numbers = [item for item in tax_numbers if item != checksum]
    if result["销售方税号"] == checksum:
        result["销售方税号"] = ""
    seller_tax_from_region = seller_tax_number_from_lines(text, result["购买方税号"])
    if seller_tax_from_region:
        result["销售方税号"] = seller_tax_from_region
    if not result["销售方税号"] and tax_numbers:
        for item in tax_numbers:
            if item != result["购买方税号"]:
                result["销售方税号"] = item
                break
    if not result["销售方税号"]:
        result["销售方税号"] = fallback_seller_tax_number(compact_text)
    if result["销售方税号"] == checksum:
        result["销售方税号"] = ""
    if result["购买方税号"] and result["购买方税号"] == result["销售方税号"]:
        # Ordinary invoices often omit buyer tax ID.  In that case OCR may only
        # see one taxpayer ID in the seller block; do not duplicate it into the
        # buyer field.
        if "普通发票" in result["发票类型"] or not buyer_tax_from_region:
            result["购买方税号"] = ""
        else:
            result["销售方税号"] = ""

    companies = company_candidates(text)
    if not looks_like_company_name(result["购买方名称"]):
        result["购买方名称"] = buyer_name_from_lines(text)
    if not result["购买方名称"] and companies:
        result["购买方名称"] = companies[0]
    seller_name_from_region = seller_name_from_lines(text, result["购买方名称"])
    if seller_name_from_region:
        result["销售方名称"] = seller_name_from_region
    if is_stamp_or_noise(result["销售方名称"]) or len(result["销售方名称"]) < 2:
        result["销售方名称"] = ""
    if not result["销售方名称"]:
        result["销售方名称"] = seller_name_from_region
    if not result["销售方名称"] and len(companies) > 1:
        preferred = [
            company
            for company in companies
            if company != result["购买方名称"] and any(keyword in company for keyword in ("税务局", "代开", "有限公司"))
        ]
        result["销售方名称"] = preferred[-1] if preferred else companies[1]

    agent_tax = agent_seller_tax_number(compact_text)
    agent_name = agent_seller_name_from_lines(text)
    if agent_tax:
        result["销售方税号"] = agent_tax
    if agent_name:
        result["销售方名称"] = agent_name

    inferred_amount, inferred_tax = infer_amount_and_tax(compact_text, result["价税合计"])
    if not result["金额"]:
        result["金额"] = inferred_amount
    if not result["税额"]:
        result["税额"] = inferred_tax
    if inferred_amount and inferred_tax:
        try:
            current_amount = float(result["金额"]) if result["金额"] else 0
            current_tax = float(result["税额"]) if result["税额"] else 0
            if current_tax >= current_amount or abs(current_tax - current_amount) < 0.01:
                result["金额"] = inferred_amount
                result["税额"] = inferred_tax
        except ValueError:
            result["金额"] = inferred_amount
            result["税额"] = inferred_tax
    if not result["金额"] and result["价税合计"] and "普通发票" in result["发票类型"] and not result["税额"]:
        result["金额"] = result["价税合计"]

    drawer_near_label = drawer_from_lines_near_label(text)
    if drawer_near_label:
        result["开票人"] = drawer_near_label
    elif result["开票人"] and not looks_like_person_name(result["开票人"]):
        result["开票人"] = ""

    return result


def completion_score(fields: Dict[str, str]) -> float:
    """Calculate how many target fields have non-empty values."""
    if not fields:
        return 0.0
    filled = sum(1 for value in fields.values() if value)
    return round(filled / len(fields), 3)
