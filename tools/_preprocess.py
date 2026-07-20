"""预处理 — LAB 阈值 + 高斯模糊 + Canny + 闭运算，输出矩形检测用边缘图。"""

from typing import Sequence, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


def preprocess(
    frame: MatLike,
    kernel: MatLike,
    rect_lab_thresholds: Tuple[Sequence[int], Sequence[int]],
    canny_thresholds: Tuple[int, int] = (50, 150),
) -> Tuple[MatLike, MatLike]:
    """矩形检测预处理管线：LAB色彩阈值 → 高斯模糊 → Canny → 闭运算。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame 必须是 HxWx3 BGR ndarray")
    if kernel.ndim != 2:
        raise ValueError("kernel 必须是二维 ndarray")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = kernel.shape

    rect_binary = cv2.inRange(lab, rect_lab_thresholds[0], rect_lab_thresholds[1])
    rect_blurred = cv2.GaussianBlur(rect_binary, (width, height), 0)
    rect_edges = cv2.Canny(rect_blurred, canny_thresholds[0], canny_thresholds[1])
    rect_edges = cv2.morphologyEx(rect_edges, cv2.MORPH_CLOSE, kernel)

    return rect_edges, gray
