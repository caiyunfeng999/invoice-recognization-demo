"""PDF helper module.

Electronic invoices are often delivered as PDF files.  This module first tries
to extract embedded text.  When a PDF page is a scanned image, it can also be
rendered to an OpenCV image for the same OCR pipeline used by uploaded images.
"""

from typing import Tuple

import cv2
import numpy as np


def open_pdf(raw: bytes):
    """Open a PDF from raw uploaded bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 pymupdf，无法处理 PDF 发票") from exc
    return fitz.open(stream=raw, filetype="pdf")


def extract_pdf_text(raw: bytes, max_pages: int = 3, page_number: int = 1) -> Tuple[str, int]:
    """Extract text from the requested PDF page.

    A positive ``page_number`` means only that page is read.  ``page_number=0``
    can be used internally to read several pages, which is useful for extension.
    """
    doc = open_pdf(raw)
    if len(doc) == 0:
        raise ValueError("PDF 文件没有页面")
    if page_number > 0:
        index = min(max(page_number - 1, 0), len(doc) - 1)
        return doc[index].get_text("text"), len(doc)
    pages = min(len(doc), max_pages)
    text_parts = [doc[index].get_text("text") for index in range(pages)]
    return "\n".join(part for part in text_parts if part.strip()), len(doc)


def pdf_first_page_to_image(raw: bytes, zoom: float = 2.0, page_number: int = 1) -> Tuple[np.ndarray, int]:
    """Render one PDF page to an OpenCV BGR image for OCR or preprocessing."""
    doc = open_pdf(raw)
    if len(doc) == 0:
        raise ValueError("PDF 文件没有页面")

    page_index = min(max(page_number - 1, 0), len(doc) - 1)
    matrix = doc[page_index].get_pixmap(matrix=None)
    if zoom != 1:
        import fitz

        matrix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

    image = np.frombuffer(matrix.samples, dtype=np.uint8).reshape(matrix.height, matrix.width, matrix.n)
    if matrix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, len(doc)
