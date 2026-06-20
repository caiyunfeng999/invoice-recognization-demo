"""Train invoice field detector with stable settings.

This script is intended for the AutoDL server environment.  It avoids the two
issues seen during CLI training:

- AMP self-check failure in the current Ultralytics/OpenCV environment.
- DataLoader image decoding failure with worker processes.

Run:
    cd /root/invoice_ocr_web/backend
    YOLO_CONFIG_DIR=./.ultralytics MPLCONFIGDIR=/tmp/matplotlib-cache python train_clean.py
"""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("./models/yolov8n.pt")
    model.train(
        data="./yolo_invoice_fields.yaml",
        epochs=120,
        imgsz=960,
        batch=16,
        workers=0,
        project="./runs/detect",
        name="invoice_fields_v8_clean",
        exist_ok=True,
        amp=False,
    )


if __name__ == "__main__":
    main()
