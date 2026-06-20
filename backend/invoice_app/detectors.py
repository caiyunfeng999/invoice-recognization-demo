"""Unified invoice field detector backends.

The UI exposes YOLOv8n, Faster R-CNN and D-FINE-L as selectable field
detectors.  YOLO and Faster R-CNN can run directly from this Flask backend when
their Python dependencies and weights are present.  D-FINE-L depends on the
external D-FINE repository runtime, so this module reports a clear unavailable
state unless that integration is installed separately.
"""

import os
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .yolo_detector import Detection, clip_box, detect_invoice_fields, normalize_label


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

MODEL_PATHS = {
    "yolo": {
        "env": "YOLO_INVOICE_MODEL",
        "default": PROJECT_DIR / "model_results_export" / "yolo" / "best_yolo_mAP708.pt",
        "fallback": BACKEND_DIR / "models" / "invoice_yolo.pt",
    },
    "faster_rcnn": {
        "env": "FASTER_RCNN_INVOICE_MODEL",
        "default": PROJECT_DIR / "model_results_export" / "faster_rcnn" / "faster_rcnn_run" / "best.pt",
    },
    "dfine_l": {
        "env": "DFINE_L_INVOICE_MODEL",
        "default": PROJECT_DIR / "model_results_export" / "dfine_l" / "best_dfine_l_mAP711.pth",
    },
}

DETECTORS = {
    "yolo": {
        "key": "yolo",
        "label": "YOLOv8n",
        "description": "轻量快速，适合前端交互和半自动标注预检测。",
        "metric": "mAP50-95 0.708 / mAP50 0.948",
    },
    "faster_rcnn": {
        "key": "faster_rcnn",
        "label": "Faster R-CNN",
        "description": "经典两阶段检测基线，召回较高但模型较大。",
        "metric": "mAP50-95 0.645 / mAP50 0.909",
    },
    "dfine_l": {
        "key": "dfine_l",
        "label": "D-FINE-L",
        "description": "当前验证集 mAP50-95 最高，适合高精度离线对比。",
        "metric": "mAP50-95 0.711 / mAP50 0.947",
    },
}

_FASTER_RCNN_MODEL = None
_FASTER_RCNN_MODEL_PATH = ""
_FASTER_RCNN_CLASS_NAMES: List[str] = []
_FASTER_RCNN_DEVICE = "cpu"


def detector_model_path(key: str) -> str:
    """Resolve the configured model path for one detector."""
    config = MODEL_PATHS[key]
    env_value = os.environ.get(config["env"])
    if env_value:
        return env_value
    default = Path(config["default"])
    if default.exists():
        return str(default)
    fallback = config.get("fallback")
    if fallback and Path(fallback).exists():
        return str(fallback)
    return str(default)


def detector_status(key: str) -> Dict[str, Any]:
    """Return availability information for one detector."""
    model_path = detector_model_path(key)
    model_exists = Path(model_path).exists()
    status = {
        **DETECTORS[key],
        "model_path": model_path,
        "model_exists": model_exists,
        "available": model_exists,
    }

    if key == "yolo":
        try:
            import ultralytics  # noqa: F401

            status["runtime_installed"] = True
        except ImportError:
            status["runtime_installed"] = False
            status["available"] = False
        return status

    if key == "faster_rcnn":
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            status["runtime_installed"] = True
        except ImportError:
            status["runtime_installed"] = False
            status["available"] = False
        return status

    command = os.environ.get("DFINE_PREDICT_COMMAND", "").strip()
    status["runtime_installed"] = bool(command)
    status["available"] = model_exists and bool(command)
    status["command_configured"] = bool(command)
    if not command:
        status["note"] = "D-FINE-L 权重已配置；Web 后端需要设置 DFINE_PREDICT_COMMAND 后才能启用推理。"
    return status


def detectors_status() -> Dict[str, Any]:
    """Return all detector options for the frontend."""
    return {
        "default": "yolo",
        "detectors": [detector_status(key) for key in ("yolo", "faster_rcnn", "dfine_l")],
    }


def _build_faster_rcnn_model(num_invoice_classes: int, min_size: int, max_size: int, score_thresh: float, nms_thresh: float):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_thresh,
        box_nms_thresh=nms_thresh,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_invoice_classes + 1)
    return model


def _get_faster_rcnn_model(score_thresh: float, nms_thresh: float):
    """Load the Faster R-CNN model lazily."""
    global _FASTER_RCNN_MODEL, _FASTER_RCNN_MODEL_PATH, _FASTER_RCNN_CLASS_NAMES, _FASTER_RCNN_DEVICE

    model_path = detector_model_path("faster_rcnn")
    if _FASTER_RCNN_MODEL is not None and _FASTER_RCNN_MODEL_PATH == model_path:
        return _FASTER_RCNN_MODEL, _FASTER_RCNN_CLASS_NAMES, _FASTER_RCNN_DEVICE

    if not Path(model_path).exists():
        raise RuntimeError(f"未找到 Faster R-CNN 发票字段模型：{model_path}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 torch/torchvision，无法运行 Faster R-CNN") from exc

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Faster R-CNN 权重中缺少 class_names，无法映射字段类别")

    model_args = checkpoint.get("args", {})
    min_size = int(model_args.get("min_size", 1280))
    max_size = int(model_args.get("max_size", 1664))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_faster_rcnn_model(len(class_names), min_size, max_size, score_thresh, nms_thresh)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    _FASTER_RCNN_MODEL = model
    _FASTER_RCNN_MODEL_PATH = model_path
    _FASTER_RCNN_CLASS_NAMES = list(class_names)
    _FASTER_RCNN_DEVICE = device
    return model, _FASTER_RCNN_CLASS_NAMES, _FASTER_RCNN_DEVICE


def _bgr_to_tensor(image: np.ndarray):
    import torch
    from torchvision.transforms import functional as F

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return F.to_tensor(rgb).to(torch.device(_FASTER_RCNN_DEVICE))


def detect_with_faster_rcnn(image: np.ndarray, confidence: float = 0.25, nms_thresh: float = 0.45) -> List[Detection]:
    """Run Faster R-CNN on one invoice image."""
    import torch

    height, width = image.shape[:2]
    model, class_names, _device = _get_faster_rcnn_model(float(confidence), float(nms_thresh))
    tensor = _bgr_to_tensor(image)
    detections: List[Detection] = []

    with torch.no_grad():
        output = model([tensor])[0]

    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        score_value = float(score.detach().cpu())
        if score_value < float(confidence):
            continue
        cls_id = int(label.detach().cpu()) - 1
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        label_name = class_names[cls_id]
        detections.append(
            Detection(
                label=label_name,
                field=normalize_label(label_name),
                confidence=round(score_value * 100, 2),
                box=clip_box(tuple(float(x) for x in box.detach().cpu().tolist()), width, height),
            )
        )

    detections.sort(key=lambda det: (det.box[1], det.box[0]))
    return detections


def _parse_external_detection(item: Dict[str, Any], width: int, height: int) -> Detection:
    """Normalize one detection returned by an external detector command."""
    label = str(item.get("label") or item.get("class") or item.get("name") or item.get("category") or "")
    if not label and "class_id" in item:
        label = str(item["class_id"])
    score = item.get("confidence", item.get("score", item.get("conf", 0.0)))
    box = item.get("box") or item.get("bbox") or item.get("xyxy")
    if not box or len(box) < 4:
        raise ValueError("external detector item missing box/bbox/xyxy")
    if item.get("bbox_format") == "xywh":
        x, y, w, h = [float(v) for v in box[:4]]
        box = [x, y, x + w, y + h]
    score_value = float(score)
    if score_value <= 1:
        score_value *= 100
    return Detection(
        label=label,
        field=normalize_label(label),
        confidence=round(score_value, 2),
        box=clip_box(tuple(float(v) for v in box[:4]), width, height),
    )


def detect_with_dfine_l(image: np.ndarray, confidence: float = 0.25) -> List[Detection]:
    """Run D-FINE-L through a user-configured external prediction command.

    Configure ``DFINE_PREDICT_COMMAND`` with placeholders:

    - ``{image}``: temporary input image path
    - ``{weights}``: D-FINE-L weight path
    - ``{output}``: JSON output path
    - ``{conf}``: confidence threshold

    The command should write JSON as either ``{"detections": [...]}`` or a list
    of detection objects.  Each detection accepts ``label``/``class``/``name``,
    ``score``/``confidence`` and ``box``/``bbox``/``xyxy``.
    """
    command_template = os.environ.get("DFINE_PREDICT_COMMAND", "").strip()
    if not command_template:
        raise RuntimeError("D-FINE-L 未配置 DFINE_PREDICT_COMMAND，无法在 Web 后端直接推理")

    model_path = detector_model_path("dfine_l")
    if not Path(model_path).exists():
        raise RuntimeError(f"未找到 D-FINE-L 发票字段模型：{model_path}")

    height, width = image.shape[:2]
    with tempfile.TemporaryDirectory(prefix="invoice_dfine_") as tmpdir:
        image_path = Path(tmpdir) / "input.png"
        output_path = Path(tmpdir) / "predictions.json"
        cv2.imwrite(str(image_path), image)
        command = command_template.format(
            image=shlex.quote(str(image_path)),
            weights=shlex.quote(str(model_path)),
            output=shlex.quote(str(output_path)),
            conf=str(float(confidence)),
        )
        completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True, timeout=180)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"D-FINE-L 推理失败：{message[:600]}")
        if not output_path.exists():
            raise RuntimeError("D-FINE-L 推理命令未生成 JSON 输出文件")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    raw_items = payload.get("detections", payload) if isinstance(payload, dict) else payload
    detections = []
    for item in raw_items:
        try:
            det = _parse_external_detection(item, width, height)
        except (TypeError, ValueError):
            continue
        if det.confidence >= float(confidence) * 100:
            detections.append(det)
    detections.sort(key=lambda det: (det.box[1], det.box[0]))
    return detections


def detect_fields(image: np.ndarray, detector: str = "yolo", confidence: float = 0.35) -> List[Detection]:
    """Run the selected detector and return normalized field detections."""
    key = detector if detector in DETECTORS else "yolo"
    if key == "yolo":
        return detect_invoice_fields(image, confidence=confidence)
    if key == "faster_rcnn":
        return detect_with_faster_rcnn(image, confidence=confidence)
    if key == "dfine_l":
        return detect_with_dfine_l(image, confidence=confidence)
    raise RuntimeError(f"unknown detector: {detector}")
