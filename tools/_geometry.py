"""几何工具 — 四点排序与透视校正。"""

from typing import Optional, Tuple

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """四点排序：左上 → 右上 → 右下 → 左下。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect.astype(np.int32)


def perspective_correct_and_validate(
    image: np.ndarray,
    pts: np.ndarray,
    real_aspect_ratio: float,
    target_width: Optional[int] = None,
    tolerance: float = 0.1,
) -> "tuple[np.ndarray, bool, float]":
    """对四点围成的四边形区域做透视校正，并验证其像素宽高比。"""
    if pts.shape != (4, 2):
        raise ValueError(f"pts 形状必须为 (4, 2)，实际为 {pts.shape}")
    if real_aspect_ratio <= 0:
        raise ValueError(f"real_aspect_ratio 必须 > 0，实际为 {real_aspect_ratio}")
    if target_width is not None and target_width <= 0:
        raise ValueError(f"target_width 必须 > 0，实际为 {target_width}")

    pts_src = order_points(pts).astype(np.float32)

    w_top = float(np.linalg.norm(pts_src[1] - pts_src[0]))
    w_bot = float(np.linalg.norm(pts_src[2] - pts_src[3]))
    w_src = (w_top + w_bot) / 2.0

    h_left = float(np.linalg.norm(pts_src[3] - pts_src[0]))
    h_right = float(np.linalg.norm(pts_src[2] - pts_src[1]))
    h_src = (h_left + h_right) / 2.0

    if h_src < 1.0:
        actual_ratio = float("inf")
        is_valid = False
    else:
        actual_ratio = w_src / h_src
        ratio_error = abs(actual_ratio - real_aspect_ratio) / real_aspect_ratio
        is_valid = ratio_error <= tolerance

    if target_width is None:
        target_width = max(int(w_src), 1)

    target_height = max(int(target_width / real_aspect_ratio), 1)

    pts_dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))

    return warped, is_valid, actual_ratio
