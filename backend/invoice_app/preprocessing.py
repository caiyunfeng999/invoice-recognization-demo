from dataclasses import dataclass
from typing import Callable, Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class ProcessingResult:
    image: np.ndarray
    description: str


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


def gray(image: np.ndarray) -> ProcessingResult:
    return ProcessingResult(to_gray(image), "灰度化：将彩色发票图像转换为单通道灰度图")


def denoise(image: np.ndarray, kernel_size: int = 3) -> ProcessingResult:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return ProcessingResult(
        cv2.medianBlur(image, kernel_size),
        f"中值滤波去噪：核大小 {kernel_size}，适合去除椒盐噪声",
    )


def gaussian_denoise(image: np.ndarray, kernel_size: int = 5) -> ProcessingResult:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return ProcessingResult(
        cv2.GaussianBlur(image, (kernel_size, kernel_size), 0),
        f"高斯滤波去噪：核大小 {kernel_size}，用于平滑光照和弱噪声",
    )


def otsu(image: np.ndarray) -> ProcessingResult:
    gray_image = to_gray(image)
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ProcessingResult(binary, "Otsu 自适应二值化：自动寻找前景/背景分割阈值")


def adaptive_binary(image: np.ndarray, block_size: int = 35, c_value: int = 11) -> ProcessingResult:
    gray_image = to_gray(image)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(c_value),
    )
    return ProcessingResult(
        binary,
        f"自适应阈值二值化：blockSize={block_size}, C={int(c_value)}，适合光照不均",
    )


def enhance(image: np.ndarray, clip_limit: float = 2.0) -> ProcessingResult:
    gray_image = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return ProcessingResult(
        clahe.apply(gray_image),
        f"CLAHE 对比度增强：clipLimit={float(clip_limit):.1f}，提升文字边缘对比度",
    )


def edges(image: np.ndarray, low: int = 60, high: int = 180) -> ProcessingResult:
    gray_image = to_gray(image)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_map = cv2.Canny(blurred, int(low), int(high))
    return ProcessingResult(edge_map, f"Canny 边缘检测：low={int(low)}, high={int(high)}")


def sobel_edges(image: np.ndarray) -> ProcessingResult:
    gray_image = to_gray(image)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    return ProcessingResult(magnitude, "Sobel 边缘检测：计算水平和垂直梯度幅值")


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def perspective_correct(image: np.ndarray) -> ProcessingResult:
    ratio = image.shape[0] / 700.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 700))
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_map = cv2.Canny(blurred, 60, 180)
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    document = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            document = approx.reshape(4, 2) * ratio
            break

    if document is None:
        return ProcessingResult(image, "透视矫正：未检测到可靠矩形文档轮廓，返回原图")

    rect = order_points(document.astype("float32"))
    tl, tr, br, bl = rect
    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    corrected = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return ProcessingResult(corrected, "透视矫正：通过矩形轮廓检测和透视变换拉正文档")


def deskew(image: np.ndarray) -> ProcessingResult:
    gray_image = to_gray(image)
    binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return ProcessingResult(image, "旋转校正：前景点过少，返回原图")
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return ProcessingResult(rotated, f"旋转校正：估计倾斜角 {angle:.2f} 度并旋转图像")


def text_regions(image: np.ndarray) -> ProcessingResult:
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    gray_image = to_gray(image)
    binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 1000 and w > 30 and h > 8:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            count += 1
    return ProcessingResult(output, f"文本区域定位：形态学膨胀后筛选出 {count} 个候选文本区域")


PROCESSORS: Dict[str, Callable[..., ProcessingResult]] = {
    "gray": gray,
    "denoise": denoise,
    "gaussian": gaussian_denoise,
    "otsu": otsu,
    "adaptive": adaptive_binary,
    "enhance": enhance,
    "edges": edges,
    "sobel": sobel_edges,
    "correct": perspective_correct,
    "deskew": deskew,
    "regions": text_regions,
}


PROCESSOR_LABELS = {
    "gray": "灰度化",
    "denoise": "中值去噪",
    "gaussian": "高斯滤波",
    "otsu": "Otsu 二值化",
    "adaptive": "自适应二值化",
    "enhance": "对比度增强",
    "edges": "Canny 边缘",
    "sobel": "Sobel 边缘",
    "correct": "透视矫正",
    "deskew": "旋转校正",
    "regions": "文本区域定位",
}
