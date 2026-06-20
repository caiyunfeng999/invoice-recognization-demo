"""Generate photographed/scanned-style YOLO invoice training samples.

The script keeps YOLO boxes aligned by applying the same homography to box
corners, then writing a merged dataset that can be used directly by YOLO.

Example:
    cd ~/invoice_ocr_web
    python scripts/augment_invoice_yolo.py \
        --source datasets/invoice_yolo \
        --output datasets/invoice_yolo_aug200 \
        --count 200 \
        --include-originals
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_real_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._") and path.name != ".DS_Store"


def image_files(root: Path, split: str) -> list[Path]:
    return sorted(path for path in (root / "images" / split).rglob("*") if is_real_image_path(path))


def ensure_dirs(root: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / rel).mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not label_path.exists():
        return labels
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        if bw > 0 and bh > 0:
            labels.append((cls, cx, cy, bw, bh))
    return labels


def yolo_to_xyxy(label: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, np.ndarray]:
    cls, cx, cy, bw, bh = label
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return cls, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def xyxy_to_yolo(cls: int, points: np.ndarray, width: int, height: int) -> str | None:
    x1 = float(np.clip(points[:, 0].min(), 0, width - 1))
    y1 = float(np.clip(points[:, 1].min(), 0, height - 1))
    x2 = float(np.clip(points[:, 0].max(), 0, width - 1))
    y2 = float(np.clip(points[:, 1].max(), 0, height - 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def random_homography(width: int, height: int, rng: random.Random) -> np.ndarray:
    max_dx = width * rng.uniform(0.015, 0.055)
    max_dy = height * rng.uniform(0.015, 0.055)
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = np.array(
        [
            [rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [width - 1 - rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [width - 1 - rng.uniform(0, max_dx), height - 1 - rng.uniform(0, max_dy)],
            [rng.uniform(0, max_dx), height - 1 - rng.uniform(0, max_dy)],
        ],
        dtype=np.float32,
    )
    center = (width / 2, height / 2)
    angle = rng.uniform(-2.5, 2.5)
    scale = rng.uniform(0.965, 1.035)
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    affine_3 = np.vstack([affine, [0, 0, 1]]).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(src, dst)
    return affine_3 @ perspective


def apply_scan_effects(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    alpha = rng.uniform(0.82, 1.22)
    beta = rng.uniform(-18, 22)
    out = out * alpha + beta

    if rng.random() < 0.55:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.55, 1.15)
        hsv[:, :, 2] *= rng.uniform(0.88, 1.08)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < 0.55:
        sigma = rng.uniform(2.0, 8.0)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out += noise

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.45:
        k = rng.choice([3, 3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.2, 0.9))

    if rng.random() < 0.35:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if rng.random() < 0.75:
        quality = rng.randint(62, 92)
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return out


def augment_one(image_path: Path, label_path: Path, output_image: Path, output_label: Path, rng: random.Random) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    height, width = image.shape[:2]
    labels = read_yolo_labels(label_path)
    matrix = random_homography(width, height, rng)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped = apply_scan_effects(warped, rng)

    lines = []
    for label in labels:
        cls, corners = yolo_to_xyxy(label, width, height)
        transformed = cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
        line = xyxy_to_yolo(cls, transformed, width, height)
        if line:
            lines.append(line)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), warped)
    output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"source": str(image_path), "image": str(output_image), "labels": len(lines)}


def copy_originals(source: Path, output: Path) -> None:
    for split in ("train", "val"):
        for image_path in image_files(source, split):
            rel = image_path.relative_to(source / "images" / split)
            target_image = output / "images" / split / rel
            target_label = output / "labels" / split / f"{image_path.stem}.txt"
            source_label = source / "labels" / split / f"{image_path.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            if source_label.exists():
                shutil.copy2(source_label, target_label)
            else:
                target_label.write_text("", encoding="utf-8")


def write_yaml(output: Path, class_names: list[str]) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n"
    (output / "yolo_invoice_fields.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/invoice_yolo")
    parser.add_argument("--output", default="datasets/invoice_yolo_aug200")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--include-originals", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    ensure_dirs(output)

    classes_path = source / "classes.txt"
    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shutil.copy2(classes_path, output / "classes.txt")
    write_yaml(output, class_names)

    if args.include_originals:
        copy_originals(source, output)

    train_images = image_files(source, "train")
    if not train_images:
        raise FileNotFoundError(f"no train images under {source}")

    reports = []
    for index in range(args.count):
        image_path = train_images[index % len(train_images)]
        label_path = source / "labels" / "train" / f"{image_path.stem}.txt"
        suffix = f"aug_{index + 1:04d}_{image_path.stem}.jpg"
        report = augment_one(
            image_path,
            label_path,
            output / "images" / "train" / suffix,
            output / "labels" / "train" / f"{Path(suffix).stem}.txt",
            rng,
        )
        reports.append(report)
        if (index + 1) % 25 == 0 or index + 1 == args.count:
            print(f"generated {index + 1}/{args.count}")

    report = {
        "source": str(source),
        "output": str(output),
        "generated": len(reports),
        "include_originals": args.include_originals,
        "samples": reports[:20],
    }
    (output / "augmentation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
