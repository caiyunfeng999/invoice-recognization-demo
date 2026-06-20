"""Download public invoice samples and copy unused images into test_samples.

The script uses only Python standard library so it can run on a clean server.
It downloads the public InvoiceDatasets GitHub archive, prefers VAT invoice
images, skips images already present in the YOLO dataset, and copies the next
batch into a review folder.

Usage:
    cd ~/invoice_ocr_web
    python scripts/download_invoice_samples.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://github.com/FuxiJia/InvoiceDatasets/archive/refs/heads/master.zip"


def existing_image_names(dataset: Path) -> set[str]:
    names: set[str] = set()
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        if not image_dir.exists():
            continue
        for path in image_dir.iterdir():
            if (
                path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                and not path.name.startswith("._")
            ):
                names.add(path.name)
    return names


def download(url: str, archive: Path, retries: int = 5) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "invoice-ocr-web"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if archive.exists():
                archive.unlink()
            time.sleep(min(10, 2 * attempt))
    raise RuntimeError(f"download failed after {retries} retries: {last_error}")


def archive_images(archive: Path, prefer: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as zf:
        infos = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and "/dataset/images/" in info.filename
            and Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and not Path(info.filename).name.startswith("._")
        ]
    prefer_key = f"/{prefer}_"
    return sorted(infos, key=lambda info: (0 if prefer_key in f"/{Path(info.filename).name}" else 1, Path(info.filename).name))


def copy_samples(archive: Path, output: Path, dataset: Path, limit: int, prefer: str) -> list[dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    existing = existing_image_names(dataset)
    copied: list[dict[str, str]] = []

    with zipfile.ZipFile(archive) as zf:
        for info in archive_images(archive, prefer):
            name = Path(info.filename).name
            if name in existing:
                continue
            target = output / name
            if target.exists():
                continue
            target.write_bytes(zf.read(info))
            copied.append({"name": name, "source": info.filename, "target": str(target)})
            if len(copied) >= limit:
                break
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", default="/private/tmp/invoice_ocr_web_download/InvoiceDatasets.zip")
    parser.add_argument("--output", default=str(ROOT / "test_samples" / "github_public_next50"))
    parser.add_argument("--dataset", default=str(ROOT / "datasets" / "invoice_yolo"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--prefer", default="vat", choices=["vat", "taxi"])
    parser.add_argument("--reuse-archive", action="store_true")
    args = parser.parse_args()

    archive = Path(args.archive).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()

    if not args.reuse_archive or not archive.exists():
        download(args.url, archive)

    copied = copy_samples(archive, output, dataset, args.limit, args.prefer)
    report = {
        "archive": str(archive),
        "output": str(output),
        "dataset": str(dataset),
        "count": len(copied),
        "samples": copied,
    }
    report_path = output / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"copied {len(copied)} images to {output}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
