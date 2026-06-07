"""Shared contour detection for preprocessing scripts."""

from __future__ import annotations

from pathlib import Path

import cv2


def detect_contours(
    bgr,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1.4)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def load_bgr(image_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return bgr
