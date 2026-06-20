#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path("/root/autodl-tmp/invoice_yolo_server")
DEFAULT_MODEL = DEFAULT_ROOT / "best_invoice_combined_334_manual_mAP707.pt"
DEFAULT_DATA = DEFAULT_ROOT / "datasets/invoice_combined_334_manual_12cls_trainval_20260616/yolo_invoice_fields_12cls.yaml"


CONFIGS = [
    # name, lr0, lrf, weight_decay, translate, scale, box, cls, dfl
    ("sweep_a_lr3e4_base", 0.0003, 0.01, 0.0003, 0.010, 0.050, 7.5, 0.50, 1.5),
    ("sweep_b_lr2e4_lowaug", 0.0002, 0.01, 0.0003, 0.005, 0.030, 7.5, 0.50, 1.5),
    ("sweep_c_lr5e4_base", 0.0005, 0.01, 0.0003, 0.010, 0.050, 7.5, 0.50, 1.5),
    ("sweep_d_box_high", 0.0003, 0.01, 0.0003, 0.010, 0.050, 9.0, 0.50, 1.7),
    ("sweep_e_cls_low", 0.0003, 0.01, 0.0003, 0.010, 0.050, 8.0, 0.35, 1.5),
    ("sweep_f_wd_high", 0.0003, 0.01, 0.0006, 0.010, 0.050, 7.5, 0.50, 1.5),
    ("sweep_g_aug_mid", 0.0003, 0.01, 0.0003, 0.015, 0.080, 7.5, 0.50, 1.5),
    ("sweep_h_lr1e4_precise", 0.0001, 0.01, 0.0003, 0.005, 0.020, 8.5, 0.45, 1.7),
]


QUICK_CONFIG_NAMES = {
    "sweep_b_lr2e4_lowaug",
    "sweep_d_box_high",
    "sweep_h_lr1e4_precise",
}


def run_cmd(cmd, cwd):
    print("\n" + "=" * 100, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def train(args):
    root = Path(args.root).resolve()
    model = Path(args.model).resolve() if args.model else DEFAULT_MODEL
    data = Path(args.data).resolve() if args.data else DEFAULT_DATA

    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    if not data.exists():
        raise FileNotFoundError(f"data yaml not found: {data}")

    configs = CONFIGS
    if args.quick:
        configs = [c for c in CONFIGS if c[0] in QUICK_CONFIG_NAMES]

    for name, lr0, lrf, wd, translate, scale, box, cls, dfl in configs:
        cmd = [
            "yolo",
            "detect",
            "train",
            f"model={model}",
            f"data={data}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"workers={args.workers}",
            "amp=False",
            "mosaic=0",
            "fliplr=0",
            "flipud=0",
            f"translate={translate}",
            f"scale={scale}",
            f"lr0={lr0}",
            f"lrf={lrf}",
            f"weight_decay={wd}",
            f"box={box}",
            f"cls={cls}",
            f"dfl={dfl}",
            f"patience={args.patience}",
            "cos_lr=True",
            "project=./runs/detect",
            f"name={name}",
            "exist_ok=True",
        ]
        run_cmd(cmd, cwd=root)


def find_metric_key(row):
    for key in row:
        normalized = key.strip().lower()
        if "map50-95" in normalized or "map50_95" in normalized:
            return key
    return None


def summarize(args):
    root = Path(args.root).resolve()
    runs_dir = root / "runs/detect"
    rows = []

    for run_dir in sorted(runs_dir.glob("sweep_*")):
        result_csv = run_dir / "results.csv"
        if not result_csv.exists():
            continue

        best = None
        best_epoch = None
        with result_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = find_metric_key(row)
                if not key:
                    continue
                try:
                    value = float(row[key])
                except Exception:
                    continue
                epoch_text = row.get("epoch") or row.get("                  epoch") or ""
                if best is None or value > best:
                    best = value
                    best_epoch = epoch_text.strip()

        if best is not None:
            rows.append((best, best_epoch, run_dir.name))

    print("\nBest runs by mAP50-95:")
    for value, epoch, name in sorted(rows, reverse=True):
        epoch_part = f" epoch={epoch}" if epoch else ""
        print(f"{value:.4f}  {name}{epoch_part}")

    if rows:
        best_value, _, best_name = sorted(rows, reverse=True)[0]
        print(f"\nTop run: {best_name} ({best_value:.4f})")
        print(f"Best weight: {runs_dir / best_name / 'weights/best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Run a constrained YOLO hyperparameter sweep for invoice detection.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="invoice_yolo_server root directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="base .pt weight path")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO data yaml path")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1664)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="run only 3 highest-priority configs")
    parser.add_argument("--summary-only", action="store_true", help="only summarize existing sweep results")
    args = parser.parse_args()

    if args.summary_only:
        summarize(args)
        return

    train(args)
    summarize(args)


if __name__ == "__main__":
    main()
