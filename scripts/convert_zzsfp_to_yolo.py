"""Convert zzsfp KIE annotations to the project's YOLO invoice-field format.

The zzsfp dataset is stored as JSONL-like files:
    image_name<TAB>[{"transcription": ..., "label": "question|answer", ...}]

Question nodes are linked to answer nodes.  This converter maps the reliable
linked fields into the current 13-class invoice field detector.  Some classes
are intentionally left absent when zzsfp does not annotate them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Iterable


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

QUESTION_TO_CLASS = {
    "No": "invoice_no",
    "开票日期": "invoice_date",
    "开票日期：": "invoice_date",
    "开叠日期：": "invoice_date",
    "开日期": "invoice_date",
    "名称": "buyer_name",
    "名称：": "buyer_name",
    "纳税人识别号": "buyer_tax_id",
    "纳税人识别号：": "buyer_tax_id",
    "金额": "amount",
    "金颈": "amount",
    "金镇": "amount",
    "全领": "amount",
    "税额": "tax",
    "税颜": "tax",
    "税领": "tax",
    "税颈": "tax",
    "价税合计": "total",
}


def jpeg_size(path: Path) -> tuple[int, int]:
    """Return JPEG width, height without external imaging dependencies."""
    with path.open("rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG file: {path}")
        while True:
            marker_start = fh.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = fh.read(1)
            while marker == b"\xff":
                marker = fh.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = fh.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                data = fh.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            fh.seek(length - 2, 1)
    raise ValueError(f"cannot read JPEG size: {path}")


def box_from_points(points: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for x, y in points:
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)


def yolo_line(class_name: str, points: list[list[float]], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box_from_points(points)
    x1 = max(0.0, min(width - 1.0, x1))
    x2 = max(0.0, min(width - 1.0, x2))
    y1 = max(0.0, min(height - 1.0, y1))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_ID[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def parse_annotation_line(line: str) -> tuple[str, list[dict]]:
    image_name, payload = line.split("\t", 1)
    return image_name, json.loads(payload)


def linked_pairs(items: list[dict]) -> list[tuple[dict, dict]]:
    by_id = {int(item["id"]): item for item in items}
    pairs = []
    for item in items:
        if item.get("label") != "question":
            continue
        question_id = int(item["id"])
        for pair in item.get("linking") or []:
            if len(pair) != 2:
                continue
            left, right = int(pair[0]), int(pair[1])
            if left == question_id and right in by_id and by_id[right].get("label") == "answer":
                pairs.append((item, by_id[right]))
    return pairs


def convert_items(items: list[dict], width: int, height: int) -> tuple[list[str], dict[str, str]]:
    selected: dict[str, dict] = {}
    texts: dict[str, str] = {}

    for question, answer in linked_pairs(items):
        q_text = str(question.get("transcription", "")).strip()
        class_name = QUESTION_TO_CLASS.get(q_text)
        if not class_name:
            continue

        # Keep the first occurrence for repeated labels.  zzsfp uses the buyer
        # block for "名称/纳税人识别号"; seller fields are not explicitly linked.
        if class_name not in selected:
            selected[class_name] = answer
            texts[class_name] = str(answer.get("transcription", "")).strip()

    lines = []
    for class_name, answer in selected.items():
        line = yolo_line(class_name, answer["points"], width, height)
        if line:
            lines.append(line)
    return lines, texts


def ensure_dirs(output: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output / rel).mkdir(parents=True, exist_ok=True)


def write_class_files(output: Path) -> None:
    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n"
    (output / "yolo_invoice_fields.yaml").write_text(yaml, encoding="utf-8")


def convert_split(source: Path, output: Path, split: str, report: list[dict]) -> None:
    ann_path = source / f"{split}.json"
    for raw in ann_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        image_name, items = parse_annotation_line(raw)
        image_path = source / "imgs" / image_name
        width, height = jpeg_size(image_path)
        lines, texts = convert_items(items, width, height)

        shutil.copy2(image_path, output / "images" / split / image_name)
        label_path = output / "labels" / split / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        report.append(
            {
                "image": image_name,
                "split": split,
                "width": width,
                "height": height,
                "label_count": len(lines),
                "classes": sorted(texts.keys(), key=lambda name: CLASS_ID[name]),
                "texts": texts,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/zzsfp")
    parser.add_argument("--output", default="datasets/zzsfp_yolo")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_dirs(output)
    write_class_files(output)

    report: list[dict] = []
    convert_split(source, output, "train", report)
    convert_split(source, output, "val", report)

    summary = {
        "source": str(source),
        "output": str(output),
        "images": len(report),
        "train": sum(1 for item in report if item["split"] == "train"),
        "val": sum(1 for item in report if item["split"] == "val"),
        "class_counts": {
            class_name: sum(1 for item in report if class_name in item["classes"])
            for class_name in CLASS_NAMES
        },
        "items": report,
    }
    (output / "conversion_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))
    print(f"YOLO yaml: {output / 'yolo_invoice_fields.yaml'}")
    print(f"Report: {output / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
