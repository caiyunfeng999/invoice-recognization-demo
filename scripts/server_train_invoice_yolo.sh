#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/invoice_ocr_web}"
ENV_NAME="${ENV_NAME:-invoice_train}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "== Python environment =="
which python
python -V

echo "== Installing/checking YOLO dependencies =="
python -m pip install --no-cache-dir \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  ultralytics \
  pandas \
  matplotlib \
  pyyaml \
  tqdm \
  pillow

echo "== Verifying torch/cv2/ultralytics =="
python - <<'PY'
import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("ultralytics ok")
PY

echo "== Verifying dataset =="
image_count=$(find datasets/invoice_yolo/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
label_count=$(find datasets/invoice_yolo/labels/train datasets/invoice_yolo/labels/val -type f -name "*.txt" | wc -l | tr -d ' ')
bad_label_files=$(find datasets/invoice_yolo/labels -name "label.txt" -o -name "classes.txt" | wc -l | tr -d ' ')

echo "images: $image_count"
echo "labels: $label_count"
echo "bad label files: $bad_label_files"

if [ "$image_count" != "100" ] || [ "$label_count" != "100" ] || [ "$bad_label_files" != "0" ]; then
  echo "Dataset check failed. Expected images=100, labels=100, bad label files=0." >&2
  exit 1
fi

echo "== Checking image decoding =="
python - <<'PY'
import cv2
import numpy as np
from pathlib import Path

bad = []
for p in Path("datasets/invoice_yolo/images").rglob("*"):
    if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    arr = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        bad.append(str(p))

print("bad images:", len(bad))
for item in bad[:20]:
    print(item)
raise SystemExit(1 if bad else 0)
PY

echo "== Starting training =="
cd "$PROJECT_DIR/backend"
YOLO_CONFIG_DIR=./.ultralytics MPLCONFIGDIR=/tmp/matplotlib-cache python train_clean.py
