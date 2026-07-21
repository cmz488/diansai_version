"""激光点检测 — 帧差法与 LAB 二值化法两种策略。"""

from ._colors import cvt_mvlab2cv

from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


class LaserSpot(NamedTuple):
    """激光点检测结果。"""

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

    if off_frame.ndim not in (2, 3):
        raise ValueError("off_frame/on_frame 必须是灰度或 BGR 图像")
    delta = on_frame.astype(np.int16) - off_frame.astype(np.int16)
    positive = np.maximum(delta, 0)
    score = (
        positive.max(axis=2) if positive.ndim == 3 else positive
    ).astype(np.float32)
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame 必须是 H×W×3 的 BGR 图像")
    if (lab_lower is None) != (lab_upper is None):
        raise ValueError("lab_lower 和 lab_upper 必须同时提供")
    if morph_kernel_size < 0 or (
        morph_kernel_size > 1 and morph_kernel_size % 2 == 0
    ):
        raise ValueError("morph_kernel_size 必须为 0、1 或大于 1 的奇数")
    if block_size < 3:
        raise ValueError("block_size 必须 >= 3")

    if lab_lower is not None and lab_upper is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, lab_lower, lab_upper)
    elif auto_thresh is not None:
        if auto_thresh._lower is None:
            # 全图最亮点可能是灯光或白纸，禁止在检测过程中隐式学习。
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

    if morph_open and morph_kernel_size > 1:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 2 or area > 1000:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 3.0:
            continue
        component = labels == label
        peak = float(lab[:, :, 0][component].max())
        fill = float(area) / max(float(width * height), 1.0)
        candidates.append((peak + 32.0 * fill, label, peak))

    if not candidates:
        return None

    _, selected_label, l_peak = max(candidates)
    selected = labels == selected_label
    ys, xs = np.nonzero(selected)
    weights = np.maximum(lab[:, :, 0][selected].astype(np.float32), 1.0)
    weight_sum = float(weights.sum())
    cx = float(np.dot(xs, weights) / weight_sum)
    cy = float(np.dot(ys, weights) / weight_sum)
    confidence = min(l_peak / 200.0, 1.0)

    return cx, cy, confidence, mask
