"""Create semi-automatic YOLO labels using an existing invoice detector.

This is intended for review-first labeling: the script copies images into a
YOLO dataset, writes predicted boxes as labels, and saves preview images with
class names and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")
    )


def ensure_dirs(output: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val", "previews"):
        (output / rel).mkdir(parents=True, exist_ok=True)


def write_dataset_files(output: Path) -> None:
    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n"
    (output / "yolo_invoice_fields.yaml").write_text(yaml, encoding="utf-8")


def split_for(index: int, count: int, val_ratio: float) -> str:
    if count <= 1 or val_ratio <= 0:
        return "train"
    val_start = int(count * (1 - val_ratio))
    return "val" if index >= val_start else "train"


def draw_preview(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls = int(det["class_id"])
        conf = float(det["confidence"])
        color = (61, 118, 157)
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(str(output_path), image)


def yolo_line(det: dict, width: int, height: int, include_conf: bool) -> str:
    x1, y1, x2, y2 = det["xyxy"]
    x1 = max(0.0, min(width - 1.0, float(x1)))
    y1 = max(0.0, min(height - 1.0, float(y1)))
    x2 = max(0.0, min(width - 1.0, float(x2)))
    y2 = max(0.0, min(height - 1.0, float(y2)))
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    base = f"{int(det['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return f"{base} {float(det['confidence']):.6f}" if include_conf else base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test_samples/public_for_auto_label")
    parser.add_argument("--weights", default="backend/models/invoice_yolo.pt")
    parser.add_argument("--output", default="datasets/invoice_yolo_semiauto_local")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--include-conf", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    files = image_files(source)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no images found: {source}")

    ensure_dirs(output)
    write_dataset_files(output)
    model = YOLO(str(weights))

    report = []
    for index, image_path in enumerate(files):
        split = split_for(index, len(files), args.val_ratio)
        image = cv2.imread(str(image_path))
        if image is None:
            report.append({"image": str(image_path), "error": "cv2.imread failed"})
            continue
        height, width = image.shape[:2]
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls < 0 or cls >= len(CLASS_NAMES):
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf.item())
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            detections.append(
                {
                    "class_id": cls,
                    "class_name": CLASS_NAMES[cls],
                    "confidence": conf,
                    "xyxy": xyxy,
                }
            )

        target_image = output / "images" / split / image_path.name
        target_label = output / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        lines = [yolo_line(det, width, height, args.include_conf) for det in detections]
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        draw_preview(image_path, detections, output / "previews" / f"{image_path.stem}.jpg")
        item = {
            "image": str(image_path),
            "split": split,
            "detections": len(detections),
            "classes": sorted({det["class_name"] for det in detections}),
            "preview": str(output / "previews" / f"{image_path.stem}.jpg"),
        }
        report.append(item)
        print(f"{image_path.name}: {len(detections)} boxes -> {split}")

    summary = {
        "source": str(source),
        "weights": str(weights),
        "output": str(output),
        "images": len(files),
        "total_boxes": sum(item.get("detections", 0) for item in report),
        "items": report,
    }
    (output / "semiauto_label_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))
    print(f"Review previews: {output / 'previews'}")
    print(f"YOLO yaml: {output / 'yolo_invoice_fields.yaml'}")


if __name__ == "__main__":
    main()
