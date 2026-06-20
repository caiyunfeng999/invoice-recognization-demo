#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

cd "$BACKEND_DIR"

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-./.ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

MODEL="${MODEL:-yolov10n.pt}"
EPOCHS="${EPOCHS:-120}"
IMGSZ="${IMGSZ:-1440}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"

yolo detect train \
  data=./yolo_invoice_fields.yaml \
  model="$MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  workers="$WORKERS" \
  project=./runs/detect \
  name=invoice_fields_yolov10 \
  exist_ok=True

