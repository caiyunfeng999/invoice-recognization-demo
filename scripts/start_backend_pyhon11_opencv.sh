#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/pyhon11-opencv/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/backend/.matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/backend/.ultralytics}"
export YOLO_INVOICE_MODEL="${YOLO_INVOICE_MODEL:-$PROJECT_DIR/model_results_export/yolo/best_yolo_mAP708.pt}"
export FASTER_RCNN_INVOICE_MODEL="${FASTER_RCNN_INVOICE_MODEL:-$PROJECT_DIR/model_results_export/faster_rcnn/faster_rcnn_run/best.pt}"
export DFINE_L_INVOICE_MODEL="${DFINE_L_INVOICE_MODEL:-$PROJECT_DIR/model_results_export/dfine_l/best_dfine_l_mAP711.pth}"
export DFINE_PREDICT_COMMAND="${DFINE_PREDICT_COMMAND:-$PYTHON_BIN $PROJECT_DIR/scripts/predict_dfine_invoice.py --image {image} --weights {weights} --output {output} --conf {conf}}"

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

cd "$PROJECT_DIR/backend"
exec "$PYTHON_BIN" main.py
