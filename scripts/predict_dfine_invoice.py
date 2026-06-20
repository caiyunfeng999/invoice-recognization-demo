#!/usr/bin/env python3
"""Run D-FINE-L invoice detection and write JSON for the Flask backend.

This script is an adapter around the official D-FINE repository.  It keeps the
web project independent from D-FINE's source tree while still giving the Flask
backend a stable JSON contract.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "_build_dfine_l_server_20260618" / "invoice_dfine_l_combined334_server" / "configs" / "dfine" / "dfine_hgnetv2_l_invoice.yml"
DEFAULT_CLASSES = PROJECT_DIR / "_build_dfine_l_server_20260618" / "invoice_dfine_l_combined334_server" / "datasets" / "invoice_yolo" / "classes.txt"


def find_dfine_repo() -> Path:
    candidates = [
        os.environ.get("DFINE_REPO"),
        str(PROJECT_DIR / "D-FINE"),
        str(PROJECT_DIR.parent / "D-FINE"),
        "/root/D-FINE",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if (path / "src").exists() and (path / "train.py").exists():
            return path
    raise RuntimeError(
        "未找到 D-FINE 仓库。请先设置 DFINE_REPO，例如："
        "export DFINE_REPO=/path/to/D-FINE"
    )


def load_classes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing classes file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_name(label: int, classes: list[str]) -> str:
    if 0 <= label < len(classes):
        return classes[label]
    if 1 <= label <= len(classes):
        return classes[label - 1]
    return str(label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict invoice fields with D-FINE-L and write JSON.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    dfine_repo = find_dfine_repo()
    sys.path.insert(0, str(dfine_repo))
    sys.path.insert(0, str(dfine_repo / "src"))

    from src.core import YAMLConfig  # type: ignore

    if not args.image.exists():
        raise FileNotFoundError(f"missing image: {args.image}")
    if not args.weights.exists():
        raise FileNotFoundError(f"missing weights: {args.weights}")
    if not args.config.exists():
        raise FileNotFoundError(f"missing config: {args.config}")

    classes = load_classes(args.classes)
    device = torch.device(args.device)
    cfg = YAMLConfig(str(args.config), resume=str(args.weights))

    checkpoint = torch.load(args.weights, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module") or checkpoint.get("model") or checkpoint

    model = cfg.model
    model.load_state_dict(state)
    model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    original_width, original_height = image.size
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[original_width, original_height]], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(tensor)
        labels, boxes, scores = postprocessor(outputs, orig_size)

    detections = []
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        score_value = float(score.detach().cpu())
        if score_value < args.conf:
            continue
        class_id = int(label.detach().cpu())
        detections.append(
            {
                "class_id": class_id,
                "label": label_name(class_id, classes),
                "score": round(score_value, 6),
                "box": [round(float(v), 2) for v in box.detach().cpu().tolist()],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"image": str(args.image), "detections": detections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

