"""激光点检测 — 帧差法与 LAB 二值化法两种策略。"""

from ._colors import cvt_mvlab2cv

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


class LaserSpot(NamedTuple):
    """帧差法检测到的激光点结果。"""

    x: float
    y: float
    peak_value: float
    confidence: float
    noise_level: float
    search_mode: str


def detect_laser_mask(off_frame, on_frame, snr_threshold=8.0):
    """帧差法激光点检测 — 需要激光开/关两张帧。"""
    if off_frame is None or on_frame is None:
        return None
    if off_frame.shape != on_frame.shape:
        return None

    delta = on_frame.astype(np.int16) - off_frame.astype(np.int16)
    score = np.maximum(delta, 0).max(axis=2).astype(np.float32)
    score = cv2.GaussianBlur(score, (3, 3), 0)

    _, peak_value, _, peak_location = cv2.minMaxLoc(score)
    peak_x, peak_y = peak_location

    noise_sample = score[::4, ::4]
    background = float(np.median(noise_sample))
    noise = float(np.median(np.abs(noise_sample - background))) + 1.0

    if peak_value < background + snr_threshold * noise:
        return None

    radius = 3
    x1 = max(0, peak_x - radius)
    x2 = min(score.shape[1], peak_x + radius + 1)
    y1 = max(0, peak_y - radius)
    y2 = min(score.shape[0], peak_y + radius + 1)

    patch = score[y1:y2, x1:x2]
    weights = np.maximum(patch - background, 0)

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return None

    yy, xx = np.mgrid[y1:y2, x1:x2]
    center_x = float((xx * weights).sum() / weight_sum)
    center_y = float((yy * weights).sum() / weight_sum)

    return center_x, center_y, peak_value


def detect_laser_by_mask(lab_frame, gray_frame, min_area):
    """通过红色掩码检测红色激光点"""
    # 0 到10
    ls, us = cvt_mvlab2cv([67, 100, 4, 22, -45, -14])

    # 生成掩码
    mask = cv2.inRange(lab_frame, ls, us)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    red_laser = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    edge = cv2.Canny(red_laser, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_laser = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_laser) < min_area:
            return None
        return cv2.boundingRect(best_laser)


def detect_laser_binary(
    frame: MatLike,
    auto_thresh: Optional["AutoThresholder"] = None,
    lab_lower: Optional[np.ndarray] = None,
    lab_upper: Optional[np.ndarray] = None,
    morph_open: bool = True,
    morph_kernel_size: int = 3,
    block_size: int = 15,
    c_val: int = 6,
) -> Optional[Tuple[float, float, float, MatLike]]:
    """使用 LAB 色彩空间二值化检测激光点（单帧，无需 off/on 切换）。"""
    h, w = frame.shape[:2]

    if lab_lower is not None and lab_upper is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, lab_lower, lab_upper)
    elif auto_thresh is not None:
        if auto_thresh._lower is None:
            try:
                auto_thresh.learn_from_peaks(frame, top_n=30, half_size=4)
            except RuntimeError:
                return None
        lower, upper = auto_thresh.thresholds
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, lower, upper)
    else:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        A = lab[:, :, 1].astype(np.float32)
        redness = np.maximum(A - 128.0, 0.0)
        score = L * redness
        if score.max() <= 0:
            return None
        score_u8 = (score / score.max() * 255.0).astype(np.uint8)
        _bs = block_size if block_size % 2 == 1 else block_size + 1
        mask = cv2.adaptiveThreshold(
            score_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            _bs,
            c_val,
        )

    if morph_open:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 2:
        return None

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    if np.any(mask > 0):
        l_peak = float(lab[:, :, 0][mask > 0].max())
    else:
        l_peak = 0.0
    confidence = min(l_peak / 200.0, 1.0)

    return cx, cy, confidence, mask
