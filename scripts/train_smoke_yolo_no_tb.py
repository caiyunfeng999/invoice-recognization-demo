from pathlib import Path

from ultralytics import YOLO
import ultralytics.utils.callbacks as callbacks
import ultralytics.utils.callbacks.base as cb_base


def _disable_integration_callbacks(instance):
    return None


cb_base.add_integration_callbacks = _disable_integration_callbacks
callbacks.add_integration_callbacks = _disable_integration_callbacks

ROOT = Path("/Users/caiyunfeng/invoice_ocr_web")

model = YOLO(str(ROOT / "backend/models/yolov8n.pt"))
model.train(
    data=str(ROOT / "datasets/yolo_smoke_train_12cls/data.yaml"),
    epochs=8,
    imgsz=640,
    batch=2,
    workers=0,
    amp=False,
    cache=False,
    project=str(ROOT / "runs/smoke"),
    name="yolo_smoke_12cls",
    exist_ok=True,
    plots=False,
    val=True,
)
