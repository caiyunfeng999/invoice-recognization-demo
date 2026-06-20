#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PUBLIC_DIR="${PUBLIC_DIR:-test_samples/github_public_next200}"
AUG_DIR="${AUG_DIR:-datasets/invoice_yolo_aug200}"

python scripts/download_invoice_samples.py \
  --limit 200 \
  --output "$PUBLIC_DIR" \
  --dataset datasets/invoice_yolo \
  --prefer vat

find "$PUBLIC_DIR" -name '._*' -type f -delete
find "$PUBLIC_DIR" -name '.DS_Store' -type f -delete

python scripts/augment_invoice_yolo.py \
  --source datasets/invoice_yolo \
  --output "$AUG_DIR" \
  --count 200 \
  --include-originals

echo "Public images for review: $PUBLIC_DIR"
echo "Augmented YOLO dataset: $AUG_DIR"
echo "YOLO yaml: $AUG_DIR/yolo_invoice_fields.yaml"
